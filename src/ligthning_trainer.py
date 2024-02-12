from typing import Any, Callable, List, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LigthningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str,
        optimizer_kwargs: Dict,
        scheduler_name: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        pred_pooling_strategy: str = "mean",
        eval_metrics: Optional[Dict] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self.pred_pooling_strategy = pred_pooling_strategy
        self.eval_metrics = None

        if eval_metrics is not None:
            self.eval_metrics = {}
            for metric, fn in eval_metrics.items():
                self.eval_metrics[metric] = fn.to("cuda")

        self.optimizer_map = {
            "adam": torch.optim.Adam,
            "adamW": torch.optim.AdamW,
            "adagrad": torch.optim.Adagrad,
            "adamax": torch.optim.Adamax,
            "rmsprop": torch.optim.RMSprop,
            "sgd": torch.optim.SGD,
        }

        self.scheduler_map = {
            "stepLR": torch.optim.lr_scheduler.StepLR,
            "reduceLR": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "cosineAnn": torch.optim.lr_scheduler.CosineAnnealingLR,
            "cosineAnnWarm": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "expLR": torch.optim.lr_scheduler.ExponentialLR,
            "cyclicLR": torch.optim.lr_scheduler.CyclicLR,
        }

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask)

    def _calc_single_batch(self, batch: Tuple[List[torch.Tensor]]) -> torch.Tensor:

        input_ids = batch[0]
        attention_mask = batch[1]

        assert isinstance(
            input_ids, list
        ), f"Expected input_ids to be list. Found type {type(input_ids)}"

        assert isinstance(
            attention_mask, list
        ), f"Expected input_ids to be list. Found type {type(attention_mask)}"

        size = list(set(x.size(-1) for x in input_ids))[0]
        n_chunks = [len(x) for x in input_ids]

        input_ids_combined, attention_mask_combined = torch.empty((0, size)).to(
            self.device
        ), torch.empty((0, size)).to(self.device)

        for x, a in zip(input_ids, attention_mask):
            input_ids_combined = torch.cat((input_ids_combined, x))
            attention_mask_combined = torch.cat((attention_mask_combined, a))

        preds = self.model(input_ids_combined.long(), attention_mask_combined.int())
        preds_split = preds.split(n_chunks)

        if self.pred_pooling_strategy == "mean":
            pooled_preds = torch.stack([torch.mean(x, dim=0) for x in preds_split])

        elif self.pred_pooling_strategy == "max":
            pooled_preds = torch.stack([torch.max(x, dim=0) for x in preds_split])

        else:
            raise ValueError("Unknown pooling strategy!!")

        return pooled_preds

    def _common_steps(self, batch: Tuple[List[torch.Tensor]]) -> Dict:

        preds = self._calc_single_batch(batch)
        labels = torch.stack(batch[-1])

        loss = F.cross_entropy(preds, labels)

        return {
            "preds": preds,
            "labels": labels,
            "loss": loss,
            "batch_size": preds.size(0),
        }

    def training_step(
        self, batch: Tuple[List[torch.Tensor]], batch_idx: torch.Tensor
    ) -> torch.Tensor | Dict[str, Any]:

        train_res = self._common_steps(batch)
        self.log(
            "train_loss",
            train_res["loss"],
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=True,
            rank_zero_only=True,
            logger=True,
            batch_size=train_res["batch_size"],
        )

        return {"loss": train_res["loss"]}

    def validation_step(
        self, batch: Tuple[List[torch.Tensor]], batch_idx: torch.Tensor
    ) -> torch.Tensor | Dict[str, Any] | None:

        val_res = self._common_steps(batch)
        val_loss = val_res["loss"]

        self.log(
            "val_loss",
            val_loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            rank_zero_only=True,
            prog_bar=True,
            batch_size=val_res["batch_size"],
        )

        val_metrics = dict()

        for metric, fn in self.eval_metrics.items():
            val_metrics[f"val_{metric}"] = fn(val_res["preds"], val_res["labels"])
            self.log(
                f"val_{metric}",
                val_metrics[f"val_{metric}"],
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=val_res["batch_size"],
            )

        return {"val_loss": val_loss, **val_metrics}

    def configure_optimizers(self) -> Any:

        optimizer = self.optimizer_map[self.optimizer_name](
            self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_name is not None:
            scheduler = self.scheduler_map[self.scheduler_name](
                optimizer, **self.scheduler_kwargs
            )

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
