import numpy as np
import polars as pol
import matplotlib.pyplot as plt
import torch
import torch.utils.data as td
import argparse
import wandb
import json
import torchmetrics
import os
import sys

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from dataset_builder import SupportTicketDataset
from utils import collate_fn_pooled_tokens
from trainer import Trainer
from models import BertBasedLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="the dataset path for the training set.",
    )

    parser.add_argument(
        "--val_data_path",
        type=str,
        required=False,
        default=None,
        help="the dataset path for the validation set.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=False,
        default=64,
        help="the training data batch size.",
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        required=False,
        default=128,
        help="the validation data batch size.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default="../models",
        help="the folder where the models will be stored.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="model.ckpt",
        help="the name of the model to be stored as.",
    )

    parser.add_argument(
        "--n_epochs",
        "-N",
        type=int,
        required=False,
        default=100,
        help="the number of epochs to run.",
    )

    parser.add_argument(
        "--use_wandb",
        "-W",
        type=int,
        required=False,
        default=False,
        help="whether to use wandb for data logging or not.",
    )

    parser.add_argument(
        "--hf_lm",
        type=str,
        required=False,
        default="bert-base-uncased",
        help="the language model to use.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="number of parallel workers",
    )

    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-4,
        help="the learning rate of the model",
    )

    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        required=False,
        default=1,
        help="number of steps to accumulate the gradients",
    )

    parser.add_argument(
        "--wandb_name",
        type=str,
        required=False,
        default="model",
        help="name of the log for wandb",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        required=False,
        default=512,
        help="the context length of the language model.",
    )

    parser.add_argument(
        "--n_devices",
        type=int,
        required=False,
        default=1,
        help="number of devices to use for training",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        required=False,
        default="auto",
        help="which accelerator to use for training the model.",
    )

    parser.add_argument(
        "--precision_strategy",
        type=str,
        required=False,
        default="16",
        help="what type of precision to use for training.",
    )

    parser.add_argument(
        "--gpu_strategy",
        type=str,
        required=False,
        default="auto",
        help="type of strategy to apply for multi-gpu training.",
    )

    args = parser.parse_args()

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    model_dir = args.model_dir
    model_name = args.model_name
    n_epochs = args.n_epochs
    use_wandb = int(args.use_wandb)
    hf_lm = args.hf_lm
    num_workers = args.num_workers
    lr = args.lr
    grad_accum_steps = args.grad_accum_steps
    wandb_name = args.wandb_name
    context_length = args.context_length
    n_devices = args.n_devices
    accelerator = args.accelerator
    precision_strategy = args.precision_strategy
    gpu_strategy = args.gpu_strategy

    assert use_wandb in [0, 1], "Expected use_wandb to be either 0/1"

    logger = None
    if use_wandb:
        wandb.init(project="AIforITOpsChaos", name=wandb_name)
        logger = wandb

    df_train = pol.read_csv(train_data_path)
    df_val = None

    if val_data_path is not None:
        df_val = pol.read_csv(val_data_path)

    print(f"Shape of the training data: {df_train.shape}")
    print(f"Shape of the validation data: {df_val.shape}")

    tokenizer = AutoTokenizer.from_pretrained(hf_lm, verbose=0)

    train_ds = SupportTicketDataset(
        data=df_train,
        tokenizer=tokenizer,
        is_bert_based=True,
        context_length=context_length,
    )
    train_dl = td.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        sampler=td.RandomSampler(train_ds),
        num_workers=num_workers,
        collate_fn=collate_fn_pooled_tokens,
    )

    val_dl = None

    if df_val is not None:
        val_ds = SupportTicketDataset(
            data=df_val,
            tokenizer=tokenizer,
            is_bert_based=True,
            context_length=context_length,
        )

        val_dl = td.DataLoader(
            val_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_pooled_tokens,
        )

    with open("../data/voda_idea_data_splits/label_map.json", "r") as fp:
        label_map = json.load(fp)

    n_classes = len(list(label_map.keys()))

    model = BertBasedLM(n_classes=n_classes)

    eval_metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=n_classes),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=n_classes),
        "recall": torchmetrics.Recall(task="multiclass", num_classes=n_classes),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=n_classes),
    }

    torch.cuda.empty_cache()
    trainer = Trainer(
        accelerator=accelerator,
        devices=n_devices,
        strategy=gpu_strategy,
        precision=precision_strategy,
        optimizer_name="adam",
        optimizer_kwargs={"betas": (0.9, 0.999), "lr": lr},
        scheduler_name="cosineAnnWarmRestarts",
        scheduler_kwargs={"T_0": 5, "eta_min": 1e-8},
        model_dir=model_dir,
        model_name=model_name,
        n_epochs=n_epochs,
        grad_accum_steps=grad_accum_steps,
        logger=logger,
        eval_metrics=eval_metrics,
    )

    trainer.fit(model=model, train_dataloader=train_dl, val_dataloader=val_dl)
