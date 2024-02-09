import numpy as np
import polars as pol
import matplotlib.pyplot as plt
import torch
import torch.utils.data as td
import argparse
import wandb
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

    parser.add_argument("--data_path", type=str, required=True, help="the dataset path")
    parser.add_argument(
        "--test_size",
        type=float,
        required=False,
        default=0.2,
        help="the amount of data for the test set",
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

    args = parser.parse_args()

    data_path = args.data_path
    test_size = args.test_size
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

    assert use_wandb in [0, 1], "Expected use_wandb to be either 0/1"

    logger = None
    if use_wandb:
        wandb.init(project="AIforITOpsChaos", name=wandb_name)
        logger = wandb

    df = pol.read_csv(data_path)

    label_map = {
        k: idx for idx, k in enumerate(df["Assigned_Group_fixed"].unique().to_list())
    }

    df = df.with_columns(
        (pol.col("Assigned_Group_fixed").map_dict(label_map)).alias("label")
    )

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=32,
        shuffle=True,
        stratify=df.select("label"),
    )

    df_val, df_test = train_test_split(df_test, test_size=0.5, shuffle=False)

    print(f"Shape of the training data: {df_train.shape}")
    print(f"Shape of the validation data: {df_val.shape}")
    print(f"Shape of the test data: {df_test.shape}")

    tokenizer = AutoTokenizer.from_pretrained(hf_lm, verbose=0)

    train_ds = SupportTicketDataset(
        data=df_train,
        tokenizer=tokenizer,
        is_bert_based=True,
        context_length=context_length,
    )

    val_ds = SupportTicketDataset(
        data=df_val,
        tokenizer=tokenizer,
        is_bert_based=True,
        context_length=context_length,
    )

    test_ds = SupportTicketDataset(
        data=df_test,
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

    val_dl = td.DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_pooled_tokens,
    )

    test_dl = td.DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_pooled_tokens,
    )

    model = BertBasedLM(n_classes=df["Assigned_Group_fixed"].n_unique())

    torch.cuda.empty_cache()
    trainer = Trainer(
        accelerator="gpu",
        precision="16",
        optimizer_name="adam",
        optimizer_kwargs={"betas": (0.9, 0.999), "lr": lr},
        model_dir=model_dir,
        model_name=model_name,
        n_epochs=n_epochs,
        grad_accum_steps=grad_accum_steps,
        logger=logger,
    )

    trainer.fit(model=model, train_dataloader=train_dl, val_dataloader=val_dl)
