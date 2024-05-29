import os
import subprocess
import json

import optuna
from strictfire import StrictFire


def opt_function(
    base_model: str,
    train_dataset_path: str,
    test_dataset_path: str,
    log_path: str,
    num_proc: int,
):
    output_metrics_path = "/tmp/metrics.json"
    output_model_path = "/tmp/model"
    global_batch_size, n_proc = 32768, num_proc
    proc_batch_size = global_batch_size // num_proc

    def _opt_function(**kwargs):
        learning_rate = kwargs["learning_rate"]
        warmup_ratio = kwargs["warmup_ratio"]

        subprocess.call(
            [
                "torchrun",
                "--nnodes=1",
                f"--nproc_per_node={n_proc}",
                "train.py",
                f"--base_model={base_model}",
                f"--train_dataset_path={train_dataset_path}",
                f"--test_dataset_path={test_dataset_path}",
                f"--output_model_path={output_model_path}",
                f"--log_path={log_path}",
                f"--log_steps=9999999",
                f"--output_metrics_path={output_metrics_path}",
                f"--learning_rate={learning_rate}",
                f"--optimizer=adamw",
                f"--train_batch_size={proc_batch_size}",
                f"--train_sub_batch-size=64",
                f"--eval_batch_size=32",
                f"--warmup_ratio={warmup_ratio}",
                f"--weight_decay=0.01",
                f"--lr_schedule=linear",
                f"--epochs=1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with open(output_metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        os.remove(output_metrics_path)

        return 0.0

    return _opt_function


def main(train_dataset_path: str, test_dataset_path: str, log_path: str, num_proc: int):
    pass


if __name__ == "__main__":
    StrictFire(main)
