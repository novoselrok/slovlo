import os
import random
from collections import defaultdict
from typing import Optional

from strictfire import StrictFire

from slovlo.jsonl import read_jsonl, write_jsonl


def train_test_split(
    dataset_path: str,
    split_ratio: float,
    output_dir: str,
    max_train_size: Optional[int] = None,
    max_test_size: Optional[int] = None,
    seed: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)

    dataset = read_jsonl(dataset_path)

    print(f"Dataset size: {len(dataset)}")

    train_size = len(dataset) * split_ratio
    if max_train_size is not None:
        train_size = min(train_size, max_train_size)

    test_size = len(dataset) - train_size
    if max_test_size is not None:
        test_size = min(test_size, max_test_size)

    # Split the dataset according to sample ids to prevent leakage.
    num_samples_per_id = defaultdict(int)
    for sample in dataset:
        num_samples_per_id[sample["id"]] += 1

    id_count = list(num_samples_per_id.items())
    random.Random(seed).shuffle(id_count)

    train_set_ids, test_set_ids = set(), set()
    train_sum, test_sum = 0, 0
    for id_, count in id_count:
        if train_sum < train_size:
            train_set_ids.add(id_)
            train_sum += count
        elif test_sum < test_size:
            test_set_ids.add(id_)
            test_sum += count

    train_split = [sample for sample in dataset if sample["id"] in train_set_ids]
    test_split = [sample for sample in dataset if sample["id"] in test_set_ids]

    print(f"Train split size: {len(train_split)}")
    print(f"Test split size: {len(test_split)}")

    write_jsonl(os.path.join(output_dir, "train.jsonl"), train_split)
    write_jsonl(os.path.join(output_dir, "test.jsonl"), test_split)


if __name__ == "__main__":
    StrictFire(train_test_split)
