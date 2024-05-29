from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from slovlo.embedding_model.tokenize_samples import tokenize, add_prefix

TokenizedInputs = Dict[str, torch.Tensor]


def mean_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed(model: PreTrainedModel, inputs: TokenizedInputs) -> torch.Tensor:
    model_output = model(**inputs)
    embeddings = mean_pool(model_output.last_hidden_state, inputs["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def batch_mean_pool_normalized_embeddings(
    samples: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_seq_length: int,
) -> torch.Tensor:
    embedded_batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        inputs = tokenize(tokenizer, batch, max_seq_length).to(model.device)
        embedded_batches.append(embed(model, inputs))
    return torch.vstack(embedded_batches)


def get_mean_pool_normalized_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: List[str],
    prefix: str,
    max_seq_length: int,
    embed_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prefixed_samples = add_prefix(samples, prefix)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return batch_mean_pool_normalized_embeddings(
                prefixed_samples, model, tokenizer, embed_batch_size, max_seq_length
            )
