from typing import Dict, Any
from dataclasses import dataclass

from slovlo.embedding_model.dataset import PairsDataset


@dataclass
class TrainingArgs:
    base_model_path: str
    output_dir: str

    train_dataset: PairsDataset
    test_dataset: PairsDataset

    train_batch_size: int
    train_sub_batch_size: int
    eval_batch_size: int

    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_schedule: str
    optimizer: str

    log_steps: int

    mrr_batch_size: int = 1000

    def get_hyperparameters(self):
        return {
            "train_batch_size": self.train_batch_size,
            "train_sub_batch_size": self.train_sub_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_schedule": self.lr_schedule,
            "optimizer": self.optimizer,
        }
