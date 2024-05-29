from typing import List, Tuple, Callable

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from slovlo.embedding_model.embed import get_mean_pool_normalized_embeddings
from slovlo.embedding_model.dataset import PairsDataset


def reciprocal_ranks(ranked_indices: torch.Tensor) -> torch.Tensor:
    arange = torch.arange(ranked_indices.size(0)).unsqueeze(1).to(ranked_indices.device)
    _, non_zero_indices = torch.nonzero(ranked_indices == arange, as_tuple=True)
    ranks = non_zero_indices + 1
    return 1.0 / ranks.float()


def compute_mrr(
    query_embeddings: torch.Tensor, document_embeddings: torch.Tensor
) -> float:
    similarity_scores = F.cosine_similarity(
        query_embeddings.unsqueeze(1), document_embeddings.unsqueeze(0), dim=2
    )
    _, ranked_indices = torch.sort(similarity_scores, descending=True, dim=1)
    return reciprocal_ranks(ranked_indices).mean().item()


def evaluate_mrr(
    dataset: PairsDataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    get_embeddings: Callable[
        [PreTrainedModel, PreTrainedTokenizer, List[str], List[str], int, int],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    mrr_batch_size: int,
    embed_batch_size: int,
    max_seq_length: int,
    query_prefix: str,
    document_prefix: str,
) -> float:
    mrr_loader = DataLoader(
        dataset,
        batch_size=mrr_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    mrrs = []
    for query_samples, document_samples in tqdm(mrr_loader, desc="MRR"):
        query_embeddings = get_embeddings(
            model,
            tokenizer,
            query_samples,
            query_prefix,
            max_seq_length,
            embed_batch_size,
        )
        document_embeddings = get_embeddings(
            model,
            tokenizer,
            document_samples,
            document_prefix,
            max_seq_length,
            embed_batch_size,
        )
        mrrs.append(compute_mrr(query_embeddings, document_embeddings))

    return np.mean(mrrs)
