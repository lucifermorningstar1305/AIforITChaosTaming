from typing import Tuple, List, Dict, Any, Callable

import numpy as np
import polars as pol
import torch
import torch.utils.data as td
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
import argparse
import os
import json

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from collections import defaultdict
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from src.dataset_builder import SupportTicketDataset
from src.ligthning_trainer import LigthningModel
from src.models import BertBasedLM
from src.utils import collate_fn_pooled_tokens


def calc_single_batch(
    model: pl.LightningModule,
    batch: Tuple,
    device: str = "cuda:0",
    pooling_strategy: str = "mean",
) -> torch.Tensor:
    """Function to process the batch to obtain input_ids and attention_mask"""

    input_ids = batch[0]
    attention_mask = batch[1]

    assert isinstance(
        input_ids, list
    ), f"Expected input_ids of type list. Found {type(input_ids)}"

    assert isinstance(
        attention_mask, list
    ), f"Expected attention_mask to be of type list. Found {type(attention_mask)}"

    size = list(set(x.size(-1) for x in input_ids))[0]
    n_chunks = [len(x) for x in input_ids]

    input_ids_combined = torch.empty((0, size)).to("cuda:0")
    attention_mask_combined = torch.empty((0, size)).to("cuda:0")

    for x, a in zip(input_ids, attention_mask):
        input_ids_combined = torch.cat((input_ids_combined, x.to("cuda:0")))
        attention_mask_combined = torch.cat((attention_mask_combined, a.to("cuda:0")))

    preds = model(input_ids_combined.long(), attention_mask_combined.int())
    preds_splitted = preds.split(n_chunks)

    if pooling_strategy == "mean":
        pooled_preds = torch.stack([torch.mean(pred, dim=0) for pred in preds_splitted])
    elif pooling_strategy == "sum":
        pooled_preds = torch.stack([torch.sum(pred, dim=0) for pred in preds_splitted])

    else:
        raise ValueError("Unknown pooling strategy!")

    return pooled_preds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="the path of the test dataset.",
    )

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="the path of the model checkpoint.",
    )

    parser.add_argument(
        "--hf_lm",
        type=str,
        required=False,
        default="bert-base-uncased",
        help="the name of the hf lm to use",
    )

    parser.add_argument(
        "--is_bert_based",
        type=int,
        required=False,
        default=1,
        help="whether the model is bert based or not.",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        required=False,
        default=512,
        help="the context length of the model.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="the batch size of the test dataset",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=1,
        help="the number of parallel workers for data processing.",
    )

    args = parser.parse_args()

    test_data_path = args.data_path
    model_checkpoint = args.model_checkpoint
    hf_lm = args.hf_lm
    is_bert_based = args.is_bert_based
    context_length = args.context_length
    batch_size = args.batch_size
    num_workers = args.num_workers

    assert is_bert_based in [
        0,
        1,
    ], f"Expected is_bert_based to be binary. Found {is_bert_based}"

    test_data = pol.read_csv(test_data_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_lm)

    test_ds = SupportTicketDataset(
        data=test_data,
        tokenizer=tokenizer,
        is_bert_based=is_bert_based,
        context_length=context_length,
    )

    test_dl = td.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_pooled_tokens,
    )

    with open("./data/voda_idea_data_splits/label_map.json", "r") as fp:
        label_map = json.load(fp)

    n_classes = len(list(label_map.keys()))

    model = BertBasedLM(n_classes=n_classes)
    lit_model = LigthningModel(
        model=model, n_classes=n_classes, optimizer_name=None, optimizer_kwargs=None
    )

    lit_model.load_from_checkpoint(
        model_checkpoint,
        model=model,
        n_classes=n_classes,
        optimizer_name=None,
        optimizer_kwargs=None,
    )

    progress = Progress(
        TextColumn("[progress.descriptio]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("◦"),
        TimeElapsedColumn(),
        TextColumn("◦"),
        TimeRemainingColumn(),
        TextColumn("◦"),
        TextColumn("[#d4b483]{task.fields[test_status]}"),
    )

    metrics = MetricCollection(
        MulticlassAccuracy(num_classes=n_classes),
        MulticlassPrecision(num_classes=n_classes),
        MulticlassRecall(num_classes=n_classes),
        MulticlassF1Score(num_classes=n_classes),
    ).to("cuda:0")

    test_losses = list()
    test_metrics = defaultdict(lambda: list())
    model.eval()
    with torch.no_grad():
        with progress as pbar:
            task = pbar.add_task(
                description="Testing", total=len(test_dl), test_status=""
            )

            for batch in test_dl:
                preds = calc_single_batch(model, batch)
                labels = torch.stack(batch[-1]).to("cuda:0")

                loss = F.cross_entropy(preds, labels)
                eval_res = metrics(preds, labels)

                test_losses.append(loss.item())

                for metric, val in eval_res.items():
                    test_metrics[metric].append(val.item())

                pbar.update(
                    task,
                    advance=1,
                    test_status=f"loss: {loss.item():.2f} acc: {eval_res['MulticlassAccuracy'].item():.2f}",
                )

    print(f"--------------- Test Statistics -------------------------")
    print(f"Loss: {np.mean(test_losses)}")
    for metric, val in test_metrics.items():
        print(f"{metric}: {np.mean(val)}")
