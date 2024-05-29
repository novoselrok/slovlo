import json
import gzip


def read_jsonl(file: str):
    if file.endswith(".gz"):
        with gzip.open(file, "rt") as f:
            return [json.loads(line) for line in f]
    else:
        with open(file, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
