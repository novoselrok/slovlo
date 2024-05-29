from typing import List

MAX_SEQ_LENGTH = 512
QUERY_PREFIX = "poizvedba: "
PASSAGE_PREFIX = "dokument: "


def add_prefix(samples: List[str], is_query=False) -> List[str]:
    return [
        f"{QUERY_PREFIX}{sample}" if is_query else f"{PASSAGE_PREFIX}{sample}"
        for sample in samples
    ]


def tokenize(tokenizer, samples: List[str], max_seq_length: int):
    return tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
