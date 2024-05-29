import logging
import random
import json
import datetime
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_scheduler,
)
from tqdm import tqdm
from strictfire import StrictFire

from slovlo.embedding_model.dataset import PairsDataset
from slovlo.embedding_model.grad_cache import GradCache
from slovlo.embedding_model.training_args import TrainingArgs
from slovlo.embedding_model.tokenize_samples import tokenize, add_prefix, MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)


def is_main_process():
    return dist.get_rank() == 0


def log_on_main_process(log: str):
    if is_main_process():
        logger.info(log)


def evaluate_on_main_process(
    args: TrainingArgs,
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
):
    if not is_main_process():
        return None

    return {}


def save_model_on_main_process(
    output_dir: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
):
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


def get_optimizer_grouped_parameters(ddp_model: DDP, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p
                for n, p in ddp_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in ddp_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


def load_model(model_path: str) -> PreTrainedModel:
    model = AutoModel.from_pretrained(model_path)
    model.pooler.dense.weight.requires_grad = False
    model.pooler.dense.bias.requires_grad = False
    return model


def train(args: TrainingArgs):
    rank = dist.get_rank()
    device = torch.device(rank)

    sampler = DistributedSampler(args.train_dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        args.train_dataset,
        batch_size=args.train_batch_size,
        drop_last=True,
        pin_memory=True,
        sampler=sampler,
        num_workers=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    ddp_model = DDP(
        load_model(args.base_model_path).to(device),
        device_ids=[rank],
        find_unused_parameters=False,
    )
    grad_cache = GradCache(ddp_model, args.train_sub_batch_size)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        ddp_model, args.weight_decay
    )
    if args.optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9
        )

    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        args.lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=max(1, int(args.warmup_ratio * num_training_steps)),
        num_training_steps=num_training_steps,
    )

    log_on_main_process(
        f"Log every {args.log_steps} steps, {num_training_steps} total steps"
    )

    progress_bar = tqdm(range(num_training_steps)) if is_main_process() else None

    for epoch in range(args.num_epochs):
        log_on_main_process(f"Epoch {epoch + 1}")

        sampler.set_epoch(epoch)

        step = 0
        for query_samples, document_samples in train_loader:
            ddp_model.train()

            prefixed_query_samples, prefixed_document_samples = (
                add_prefix(query_samples, is_query=True),
                add_prefix(document_samples),
            )

            inputs = tokenize(
                tokenizer,
                prefixed_query_samples + prefixed_document_samples,
                MAX_SEQ_LENGTH,
            ).to(device)

            loss = grad_cache(inputs)

            if math.isnan(loss.item()):
                log_on_main_process("Loss is NaN. Stopping...")
                return

            grad_cache.step(optimizer, lr_scheduler)

            if step != 0 and step % args.log_steps == 0:
                log_on_main_process(f"Loss ({step}): {loss.item()}")

                ddp_model.eval()
                evaluate_on_main_process(args, device, ddp_model.module, tokenizer)

            step += 1
            if progress_bar:
                progress_bar.update(1)

        log_on_main_process("Saving encoder...")
        save_model_on_main_process(args.output_dir, ddp_model.module, tokenizer)

        dist.barrier()

    ddp_model.eval()
    return evaluate_on_main_process(args, device, ddp_model.module, tokenizer)


def main(
    base_model: str,
    train_dataset_path: str,
    test_dataset_path: str,
    output_model_path: str,
    epochs: int,
    train_batch_size: int,
    train_sub_batch_size: int,
    eval_batch_size: int,
    warmup_ratio: float,
    learning_rate: float,
    weight_decay: float,
    lr_schedule: str,
    optimizer: str,
    log_path: str,
    log_steps: int,
    output_metrics_path: Optional[str] = None,
):
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600 * 2))

    file_handler = logging.FileHandler(filename=log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s]: %(message)s", datefmt="%d/%m/%Y %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    train_dataset, test_dataset = (
        PairsDataset(train_dataset_path),
        PairsDataset(test_dataset_path),
    )

    training_args = TrainingArgs(
        base_model_path=base_model,
        output_dir=output_model_path,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=epochs,
        train_batch_size=train_batch_size,
        train_sub_batch_size=train_sub_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_schedule=lr_schedule,
        weight_decay=weight_decay,
        optimizer=optimizer,
        log_steps=log_steps,
    )

    log_on_main_process("Started training")
    log_on_main_process(f"Hyperparameters: {training_args.get_hyperparameters()}")

    metrics = train(training_args)
    if is_main_process() and output_metrics_path is not None:
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    StrictFire(main)
