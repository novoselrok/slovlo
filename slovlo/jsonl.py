import json
import gzip
from typing import List, Any


def read_jsonl(file: str):
    if file.endswith(".gz"):
        with gzip.open(file, "rt") as f:
            return [json.loads(line) for line in f]
    else:
        with open(file, encoding="utf-8") as f:
            return [json.loads(line) for line in f]


def write_jsonl(output_path: str, data: List[Any], compress=False):
    if compress:
        with gzip.open(output_path, "wt") as f:
            for element in data:
                f.write(json.dumps(element))
                f.write("\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for element in data:
                f.write(json.dumps(element))
                f.write("\n")
