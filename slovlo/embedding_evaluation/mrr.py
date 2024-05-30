from typing import List, Any, Tuple

import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
from strictfire import StrictFire

from slovlo.jsonl import read_jsonl
from slovlo.embedding_model.embed import mean_pool_normalize_embeddings
from slovlo.embedding_model.tokenize_samples import E5_QUERY_PREFIX, E5_DOCUMENT_PREFIX


def mrr_at_k(
    query_embeddings: np.ndarray, document_embeddings: np.ndarray, k=10
) -> float:
    _, embedding_dim = query_embeddings.shape
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(document_embeddings)

    _, indices = index.search(query_embeddings, k)

    reciprocal_ranks = []
    for i, nearest_neighbor_indices in enumerate(indices):
        # The relevant document for query i is document i.
        try:
            rank = np.where(nearest_neighbor_indices == i)[0][0] + 1
            reciprocal_rank = 1 / rank
        except IndexError:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

    mrr = np.mean(reciprocal_ranks)
    return mrr


def get_bge_m3_embeddings(
    dataset: List[Any], model_path: str, embed_batch_size: int, max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_path, use_fp16=True)
    query_emb = model.encode(
        [sample["query"] for sample in dataset],
        batch_size=embed_batch_size,
        max_length=max_length,
    )["dense_vecs"]

    document_emb = model.encode(
        [sample["document"] for sample in dataset],
        batch_size=embed_batch_size,
        max_length=max_length,
    )["dense_vecs"]

    return query_emb, document_emb


def get_e5_embeddings(
    dataset: List[Any],
    model_path: str,
    embed_batch_size: int,
    max_length: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model = AutoModel.from_pretrained(model_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    query_emb = (
        mean_pool_normalize_embeddings(
            model,
            tokenizer,
            (sample["query"] for sample in dataset),
            E5_QUERY_PREFIX,
            max_length,
            embed_batch_size,
        )
        .cpu()
        .numpy()
    )

    document_emb = (
        mean_pool_normalize_embeddings(
            model,
            tokenizer,
            (sample["document"] for sample in dataset),
            E5_DOCUMENT_PREFIX,
            max_length,
            embed_batch_size,
        )
        .cpu()
        .numpy()
    )

    return query_emb, document_emb


def main(
    dataset_path: str,
    model_path: str,
    embed_batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
):
    dataset = read_jsonl(dataset_path)

    if "bge-m3" in model_path:
        query_emb, document_emb = get_bge_m3_embeddings(
            dataset,
            model_path,
            embed_batch_size=embed_batch_size,
            max_length=max_length,
        )
    elif "e5" in model_path or "slovlo" in model_path:
        query_emb, document_emb = get_e5_embeddings(
            dataset,
            model_path,
            embed_batch_size=embed_batch_size,
            max_length=max_length,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model path.")

    for k in [1, 5, 10]:
        mrr = mrr_at_k(query_emb, document_emb, k)
        print(f"MRR@{k}: {mrr*100:.1f}")


if __name__ == "__main__":
    StrictFire(main)
