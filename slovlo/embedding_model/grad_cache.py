from contextlib import nullcontext
from typing import List, Tuple
from itertools import repeat

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from slovlo.embedding_model.rand_context import RandContext
from slovlo.embedding_model.embed import embed, TokenizedInputs


class GradCache:
    def __init__(self, model: DDP, sub_batch_size: int):
        self.model = model
        self.sub_batch_size = sub_batch_size

        self.temperature = 0.01
        self.scaler = torch.cuda.amp.GradScaler()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __call__(self, inputs: TokenizedInputs):
        return self.cache_step(inputs)

    def _get_tensors_from_input(self, input: TokenizedInputs) -> List[torch.Tensor]:
        return list(input.values())

    def _gather_split_embedding_tensors(
        self, local_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gathered = [torch.empty_like(local_embeddings) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_embeddings)
        gathered[self.rank] = local_embeddings

        n_pairs = local_embeddings.shape[0] // 2
        embeddings_split = [
            (embeddings[:n_pairs, :], embeddings[n_pairs:, :])
            for embeddings in gathered
        ]

        query_embeddings, document_embeddings = (
            torch.cat([qe for qe, _ in embeddings_split], dim=0),
            torch.cat([de for _, de in embeddings_split], dim=0),
        )

        return query_embeddings, document_embeddings

    def get_embeddings(self, inputs: TokenizedInputs) -> torch.Tensor:
        return embed(self.model, inputs)

    def step(self, optimizer: Optimizer, lr_scheduler: LambdaLR):
        self.scaler.step(optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()

        skip_lr_sched = scale > self.scaler.get_scale()
        if not skip_lr_sched:
            lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

    def compute_loss(self, local_embeddings: torch.Tensor) -> torch.Tensor:
        (
            global_query_embeddings,
            global_document_embeddings,
        ) = self._gather_split_embedding_tensors(local_embeddings)

        similarity_matrix = (
            torch.matmul(global_query_embeddings, global_document_embeddings.t())
            / self.temperature
        )

        labels = torch.arange(
            similarity_matrix.shape[0], device=similarity_matrix.device
        )
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def batch_inputs(
        self, inputs: TokenizedInputs, batch_size: int
    ) -> List[TokenizedInputs]:
        keys = list(inputs.keys())
        split_tensors = [inputs[k].split(batch_size, dim=0) for k in keys]
        return [
            dict(zip(key, tensor))
            for key, tensor in zip(repeat(keys), zip(*split_tensors))
        ]

    def forward_no_grad(
        self, batches: List[TokenizedInputs]
    ) -> Tuple[torch.Tensor, List[RandContext]]:
        embeddings_batches, rnd_states = [], []

        with torch.no_grad():
            for input_batch in batches:
                rnd_states.append(
                    RandContext(self._get_tensors_from_input(input_batch))
                )

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = self.get_embeddings(input_batch)

                embeddings_batches.append(embeddings)

        return torch.vstack(embeddings_batches), rnd_states

    def build_cache(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings_detached = embeddings.detach().requires_grad_()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = self.compute_loss(embeddings_detached)

        self.scaler.scale(loss).backward()

        return embeddings_detached.grad, loss.detach()

    def forward_backward(
        self,
        input_batches: List[TokenizedInputs],
        grad_cache_batches: List[torch.Tensor],
        rnd_states: List[RandContext],
    ):
        # no_sync disables gradient synchronizations across DDP processes.
        # Within this context, gradients will be accumulated on module variables, which will later be
        # synchronized in the first forward-backward pass exiting the context.
        sync_contexts = [self.model.no_sync for _ in range(len(input_batches) - 1)] + [
            nullcontext
        ]

        for input_batch, grad_cache_batch, rnd_state, sync_context in zip(
            input_batches, grad_cache_batches, rnd_states, sync_contexts
        ):
            with sync_context():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with rnd_state:
                        embeddings = self.get_embeddings(input_batch)

                surrogate = torch.dot(embeddings.flatten(), grad_cache_batch.flatten())
                self.scaler.scale(surrogate).backward()

    def cache_step(self, inputs: TokenizedInputs):
        input_batches = self.batch_inputs(inputs, self.sub_batch_size)

        embeddings, rnd_states = self.forward_no_grad(input_batches)

        grad_cache, loss = self.build_cache(embeddings)

        self.forward_backward(
            input_batches,
            grad_cache.split(self.sub_batch_size),
            rnd_states,
        )

        return loss
