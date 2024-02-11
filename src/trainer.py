from typing import Any, List, Tuple, Dict, Optional, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import os
import gc


from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    Column,
)

torch.set_float32_matmul_precision("medium")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"


class Trainer(object):
    def __init__(
        self,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Union[List[int], int, str] = "auto",
        precision: str = "32-true",
        num_nodes: int = 1,
        logger: Optional[Any] = None,
        model_dir: str = "../models",
        model_name: str = "model.ckpt",
        grad_accum_steps: int = 1,
        optimizer_name: str = "adam",
        scheduler_name: Optional[str] = None,
        scheduler_kwargs: Optional[dict] = None,
        optimizer_kwargs: Optional[dict] = None,
        n_epochs: int = 100,
        pooling_strategy: str = "mean",
        load_best_model: bool = True,
        eval_metrics: Optional[Dict] = None,
    ):

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            num_nodes=num_nodes,
        )

        self.device = self.fabric.device
        self.model_dir = model_dir
        self.model_name = model_name
        self.grad_accum_steps = grad_accum_steps
        self.logger = logger
        self.n_epochs = n_epochs
        self.pooling_strategy = pooling_strategy

        optimizer_dict = {
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }

        scheduler_dict = {
            "stepLR": torch.optim.lr_scheduler.StepLR,
            "expLR": torch.optim.lr_scheduler.ExponentialLR,
            "cosineAnnLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "cosineAnnWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        }

        if optimizer_name not in optimizer_dict:
            raise Exception(
                f"Supported optimizers are : {list(optimizer_dict.keys())}. Found {optimizer_name}"
            )

        if scheduler_name is not None and scheduler_name not in scheduler_dict:
            raise Exception(
                f"Supported schedulers are : {list(scheduler_dict.keys())}. Found {scheduler_name}"
            )

        self.optimizer = optimizer_dict[optimizer_name]
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler = (
            scheduler_dict[scheduler_name] if scheduler_name is not None else None
        )

        self.scheduler_kwargs = scheduler_kwargs

        self.progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, table_column=Column(ratio=2)),
            MofNCompleteColumn(),
            TextColumn("◦"),
            TimeElapsedColumn(),
            TextColumn("◦"),
            TimeRemainingColumn(),
            TextColumn("◦"),
            TextColumn("[#d4b483]{task.fields[train_step_status]}"),
            TextColumn("[#c1666b]{task.fields[train_epoch_status]}"),
            TextColumn("[#4357ad]{task.fields[val_epoch_status]}"),
            expand=False,
        )

        self.best_val_loss = float("inf")
        self.load_best_model = load_best_model

        self.eval_metrics = eval_metrics

        if self.eval_metrics is not None:
            for k, v in self.eval_metrics.items():
                self.eval_metrics[k] = v.to(self.device)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(os.path.join(self.model_dir, "best_models")):
            os.mkdir(os.path.join(self.model_dir, "best_models"))

        self.fabric.launch()

    def fit(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """Function to fit the model to the data"""

        ckpt_path = os.path.join(self.model_dir, self.model_name)

        if self.optimizer_kwargs is not None:
            optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)
        else:
            optimizer = self.optimizer(model.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        else:
            scheduler = None

        ###################################################################
        ################# LOAD CHECKPOINTS (IF PRESENT) ###################
        ###################################################################

        if self.load_best_model:
            best_ckpt_path = os.path.join(
                self.model_dir, "best_models", self.model_name
            )
            if os.path.exists(best_ckpt_path):
                full_ckpt = self.fabric.load(best_ckpt_path)

                model.load_state_dict(full_ckpt["model"])
                optimizer.load_state_dict(full_ckpt["optimizer"])

        else:
            if os.path.exists(ckpt_path):
                full_ckpt = self.fabric.load(ckpt_path)

                model.load_state_dict(full_ckpt["model"])
                optimizer.load_state_dict(full_ckpt["optimizer"])

        ####################################################################
        ################### LIGHTNING FABRIC SETUP #########################
        ####################################################################

        model, optimizer = self.fabric.setup(model, optimizer)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(
                val_dataloader, move_to_device=False, use_distributed_sampler=False
            )

        ####################################################################
        ################### SANITY CHECKING ################################
        ####################################################################

        if val_dataloader is not None:
            with self.progress_bar as sanity_pbar:
                task = sanity_pbar.add_task(
                    description="Sanity Checking",
                    total=2,
                    train_step_status="",
                    train_epoch_status="",
                    val_epoch_status="",
                )

                self._sanity_check(model=model, dataloader=val_dataloader, task=task)

                self.progress_bar.remove_task(task)

        ###########################################################################
        ###################### TRAINING / VALIDATAION #############################
        ############################################################################

        for epoch in range(self.n_epochs):

            with self.progress_bar as pbar:
                task = pbar.add_task(
                    description=f"Epoch-{epoch}",
                    total=len(train_dataloader),
                    train_step_status="",
                    train_epoch_status="",
                    val_epoch_status="",
                )

                train_loss = self.training_step(
                    model=model,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    task=task,
                )

                train_loss_gathered = self.fabric.all_gather(train_loss)

                if self.fabric.global_rank == 0:

                    self.progress_bar.update(
                        task,
                        train_epoch_status=f"train_loss: {train_loss_gathered.float().mean():.3f}",
                    )

                    if self.logger is not None:
                        self.logger.log(
                            {"train_loss": train_loss_gathered.float().mean()}
                        )
                        self.logger.log({"epoch": epoch})

                    state = {"model": model, "optimizer": optimizer, "iteration": epoch}
                    self.fabric.save(ckpt_path, state)

                if val_dataloader is not None and self.fabric.global_rank == 0:
                    task2 = pbar.add_task(
                        description="Validation",
                        total=len(val_dataloader),
                        train_step_status="",
                        train_epoch_status="",
                        val_epoch_status="",
                    )

                    val_metrics = self.validation_step(
                        model=model, val_dataloader=val_dataloader, task=task2
                    )

                    for metric, val in val_metrics.items():
                        self.logger.log({metric: val})

                    val_loss = val_metrics["val_loss"].item()
                    val_acc = val_metrics["val_acc"].item()

                    pbar.remove_task(task2)
                    pbar.update(
                        task,
                        val_epoch_status=f"val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}",
                    )

                    if val_loss < self.best_val_loss:
                        print(
                            f"Obtained a best validation loss : {val_loss} which is less than {self.best_val_loss}"
                        )
                        self.best_val_loss = val_loss
                        best_state = {
                            "model": model,
                            "optimizer": optimizer,
                            "iteration": epoch,
                        }
                        self.fabric.save(
                            os.path.join(
                                self.model_dir,
                                "best_models",
                                self.model_name,
                            ),
                            best_state,
                        )

                        print("Saved the best model checkpoint!")

                if scheduler is not None:
                    scheduler.step()

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        task: Progress,
    ) -> float:
        """Function to perform one complete training cycle."""

        losses = list()
        model.train()

        for idx, batch in enumerate(train_dataloader):
            is_accumulating = idx % self.grad_accum_steps != 0

            with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                preds = self._calc_single_batch(model=model, batch=batch)
                label = torch.stack(batch[-1])

                loss = F.cross_entropy(preds, label)

                self.fabric.backward(loss)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())

            loss_gathered = self.fabric.all_gather(loss)

            if self.fabric.global_rank == 0:
                self.progress_bar.update(
                    task,
                    advance=1,
                    train_step_status=f"train_step_loss: {loss_gathered.float().mean().item():.3f}",
                )

                if self.logger is not None:
                    self.logger.log(
                        {"train_step_loss": loss_gathered.float().mean().item()}
                    )

        return np.mean(losses)

    def validation_step(
        self,
        model: nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        task: Progress,
        sanity_check: bool = False,
    ) -> Dict:

        model.eval()

        val_losses = list()
        val_acc = list()
        val_prec = list()
        val_f1 = list()
        val_rec = list()
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                if sanity_check and idx >= 2:
                    break
                preds = self._calc_single_batch(model=model, batch=batch)
                label = torch.stack(batch[-1])

                loss = F.cross_entropy(preds, label)
                val_losses.append(loss.item())

                for metric, fn in self.eval_metrics.items():
                    if metric == "accuracy":
                        val_acc.append(fn(preds, label).item())
                    if metric == "recall":
                        val_rec.append(fn(preds, label).item())
                    if metric == "precision":
                        val_prec.append(fn(preds, label).item())
                    if metric == "f1":
                        val_f1.append(fn(preds, label).item())

                self.progress_bar.update(task, advance=1)

        return {
            "val_loss": np.mean(val_losses),
            "val_acc": np.mean(val_acc),
            "val_rec": np.mean(val_rec),
            "val_prec": np.mean(val_prec),
            "val_f1": np.mean(val_f1),
        }

    def _calc_single_batch(
        self, model: nn.Module, batch: Tuple[torch.Tensor], val_mode: bool = False
    ) -> torch.Tensor:
        """Function to evaluate single batch"""

        input_ids = batch[0]
        attention_mask = batch[1]

        device = "cuda:0" if val_mode else self.device

        assert isinstance(input_ids, list), "Expected input_ids to be a list"
        assert isinstance(attention_mask, list), "Expected attention_mask to be a list"

        n_chunks = [len(x) for x in input_ids]
        size = list(set(x.size(-1) for x in input_ids))[0]

        input_ids_combined, attention_mask_combined = torch.empty(0, size).to(
            device
        ), torch.empty(0, size).to(device)

        for x, a in zip(input_ids, attention_mask):
            input_ids_combined = torch.cat((input_ids_combined, x))
            attention_mask_combined = torch.cat((attention_mask_combined, a))

        preds = model(input_ids_combined.long(), attention_mask_combined.int())

        preds_split = preds.split(n_chunks)

        if self.pooling_strategy == "mean":
            pooled_preds = torch.stack([torch.mean(x, dim=0) for x in preds_split])

        elif self.pooling_strategy == "max":
            pooled_preds = torch.stack([torch.max(x, dim=0) for x in preds_split])

        else:
            raise ValueError("Unknown pooling strategy!!")

        del n_chunks
        del input_ids_combined
        del attention_mask_combined

        return pooled_preds

    def _sanity_check(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        task: Progress,
    ):
        self.validation_step(model, dataloader, task, sanity_check=True)
