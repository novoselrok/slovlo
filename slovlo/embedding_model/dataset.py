from typing import Optional

from torch.utils.data import Dataset

from slovlo.jsonl import read_jsonl


class PairsDataset(Dataset):
    def __init__(self, path: str):
        self.data = [(row["query"], row["document"]) for row in read_jsonl(path)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
