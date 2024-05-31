from typing import List, Dict, Any

import numpy as np
from elasticsearch import Elasticsearch, helpers
from strictfire import StrictFire

from slovlo.jsonl import read_jsonl


def create_index_and_index_documents(
    es: Elasticsearch,
    index_name: str,
    documents: List[Dict[str, Any]],
):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "text", "analyzer": "serbian"},
                }
            }
        },
    )

    actions = [
        {"_index": index_name, "_id": doc["id"], "_source": doc} for doc in documents
    ]
    helpers.bulk(es, actions)


def mrr_at_k(
    es: Elasticsearch,
    index_name: str,
    queries: List[Dict[str, Any]],
    k=10,
):
    reciprocal_ranks = []

    for query in queries:
        response = es.search(
            index=index_name,
            body={"query": {"match": {"text": query["text"]}}, "size": k},
        )

        retrieved_ids = [hit["_source"]["id"] for hit in response["hits"]["hits"]]

        try:
            rank = retrieved_ids.index(query["id"]) + 1
            reciprocal_rank = 1 / rank
        except ValueError:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

    mrr = np.mean(reciprocal_ranks)
    return mrr


def main(
    dataset_path: str,
    es_url: str = "http://localhost:9200",
    index_name: str = "documents",
):
    dataset = read_jsonl(dataset_path)
    queries = [
        {"id": idx, "text": sample["query"]} for idx, sample in enumerate(dataset)
    ]
    documents = [
        {"id": idx, "text": sample["document"]} for idx, sample in enumerate(dataset)
    ]

    es = Elasticsearch(es_url)
    create_index_and_index_documents(es, index_name, documents)

    for k in [1, 5, 10]:
        mrr = mrr_at_k(es, index_name, queries, k)
        print(f"MRR@{k}: {mrr*100:.1f}")


if __name__ == "__main__":
    StrictFire(main)
