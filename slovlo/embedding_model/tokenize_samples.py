from typing import List

MAX_SEQ_LENGTH = 512
E5_QUERY_PREFIX = "query: "
E5_DOCUMENT_PREFIX = "document: "


def add_prefix(samples: List[str], prefix: str) -> List[str]:
    return [f"{prefix}{sample}" for sample in samples]


def tokenize(tokenizer, samples: List[str], max_seq_length: int):
    return tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
