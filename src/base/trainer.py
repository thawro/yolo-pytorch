import gc
import logging
import random
from abc import abstractmethod

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.logger.loggers import Loggers, Status
from src.logger.pylogger import log, log_breaking_point, log_msg, remove_last_logged_line
from src.utils.training import get_device_and_id, is_main_process
from src.utils.types import _accelerator, _stage
from src.utils.utils import colorstr

from .callbacks import BaseCallback, Callbacks
from .datamodule import DataModule
from .meters import Meters
from .module import BaseModule
from .storage import MetricsStorage, SystemMonitoringStorage
from .validator import BaseValidator

# TODO:
# fix compile + DDP

# Gradients visualization (with plotly), ideas:
# https://gist.github.com/Flova/8bed128b41a74142a661883af9e51490
# https://github.com/wandb/wandb/blob/main/wandb/wandb_torch.py#L121
# https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/histogram.py

# test classification pipeline
# gifs saving from training evaluation samples (must ensure that sample_idxs are same)


class Trainer:
    module: BaseModule
    datamodule: DataModule
    validator: BaseValidator | None

    def __init__(
        self,
        logger: Loggers,
        accelerator: _accelerator,
        callbacks: list[BaseCallback],
        max_epochs: int = 100,
        limit_batches: int = -1,
        use_DDP: bool = False,
        sync_batchnorm: bool = True,
        use_compile: bool = False,
        run_sanity_check: bool = False,
        half_precision: bool = False,
    ):
        stages = ["train", "val", "eval_val"]
        self.use_DDP = use_DDP
        self.sync_batchnorm = sync_batchnorm
        self.use_compile = use_compile
        self.run_sanity_check = run_sanity_check
        self.meters = {stage: Meters(use_DDP=use_DDP) for stage in stages}
        self.accelerator = accelerator
        self.device, self.device_id = get_device_and_id(self.accelerator, self.use_DDP)
        self.callbacks = Callbacks(callbacks, device_id=self.device_id)
        self.logger = logger
        self.max_epochs = max_epochs
        self._limit_batches = limit_batches
        self.current_step = -1
        self.current_epoch = 0
        self.epochs_metrics = MetricsStorage(name="Epochs")  # every step metrics
        self.validation_metrics = MetricsStorage(name="LogStep")  # validation metrics
        self.system_monitoring = SystemMonitoringStorage()
        self.half_precision = half_precision
        self.val_results = []

    @property
    def map_location(self) -> dict[str, str]:
        return {"cuda:0": self.device}

    def warmup_lr(self, batch_idx: int, dataloader: DataLoader):
        for name in self.module.optimizers:
            _lr_scheduler = self.module._lr_schedulers[name]
            _optimizer = self.module._optimizers[name]
            warmup_epochs = _lr_scheduler["warmup_epochs"]
            warmup_start_bias_lr = _lr_scheduler["warmup_start_bias_lr"]
            warmup_start_momentum = _lr_scheduler["warmup_start_momentum"]
            warmup_end_lr = _optimizer["params"]["lr"]
            warmup_end_momentum = _optimizer["params"].get("momentum", None)
            num_batches = len(dataloader)
            num_warmup_iters = (
                max(round(warmup_epochs * num_batches), 100) if warmup_epochs > 0 else -1
            )
            num_iter = batch_idx + num_batches * self.current_epoch
            if num_iter <= num_warmup_iters:
                xi = [0, num_warmup_iters]  # x interp
                optimizer = self.module.optimizers["optim"]
                for j, x in enumerate(optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    warmup_start_lr = warmup_start_bias_lr if j == 0 else 0.0
                    x["lr"] = np.interp(num_iter, xi, [warmup_start_lr, warmup_end_lr]).item()
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            num_iter, xi, [warmup_start_momentum, warmup_end_momentum]
                        ).item()

    def get_limit_batches(self, dataloader: DataLoader) -> int:
        if self._limit_batches <= 0:
            return len(dataloader)
        return int(self._limit_batches)

    def _update_metrics(self, stages: list[_stage]):
        if "eval_val" in stages:
            storage = self.validation_metrics
        else:
            storage = self.epochs_metrics
        # Optimizers metrics (LR and momentum)
        if "train" in stages:
            # for i, param_group in enumerate(optimizer.param_groups)
            optims_pg_metrics = [
                {"epoch": self.current_epoch, "step": self.current_step} for _ in range(3)
            ]
            for name, optimizer in self.module.optimizers.items():
                for i, param_group in enumerate(optimizer.param_groups):
                    optims_pg_metrics[i][f"{name}_LR"] = param_group["lr"]
                    if "momentum" in param_group:
                        optims_pg_metrics[i][f"{name}_momentum"] = param_group["momentum"]

            for i, metrics in enumerate(optims_pg_metrics):
                self.logger.log_metrics(metrics, self.current_epoch)
                storage.append(metrics, self.current_step, self.current_epoch, split=f"pg_{i}")

            # metrics = {
            #     f"{name}_LR": optimizer.param_groups[0]["lr"]
            #     for name, optimizer in self.module.optimizers.items()
            # }
            # metrics["epoch"] = self.current_epoch
            # metrics["step"] = self.current_step
            # self.logger.log_metrics(metrics, self.current_epoch)
            # storage.append(metrics, self.current_step, self.current_epoch, split="train")

        for stage in stages:
            metrics = self.meters[stage].to_dict()
            storage.append(metrics, self.current_step, self.current_epoch, stage)
            stage_metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
            self.logger.log_metrics(stage_metrics, self.current_epoch)

    def evaluate(self, dataloader: DataLoader, stage: _stage, sanity: bool = False):
        """Evaluate on validation set"""
        self.module.model.net.eval()

        meters = self.meters[stage]
        meters.reset()

        if sanity:
            limit_batches = 10
            random_idx = 0
        else:
            limit_batches = self.get_limit_batches(dataloader)
            random_idx = random.randint(0, limit_batches - 1)

        tqdm_iter = tqdm(dataloader, leave=True, desc=stage, ncols=100, total=limit_batches)

        for idx, batch in enumerate(tqdm_iter):
            self.logger.file_log.info(str(tqdm_iter))
            if limit_batches == 0:
                break
            with torch.no_grad():
                val_metrics, val_results = self.module._validation_step(batch, idx, stage=stage)
            meters.update(val_metrics, self.datamodule.batch_size)

            if self.validator is not None and not sanity:
                self.validator.update_results(val_results)

            if idx == random_idx and not sanity:
                self.val_results.extend(val_results)
            self.callbacks.on_step_end(self)
            limit_batches -= 1
            remove_last_logged_line(self.logger.file_log)
        self.logger.file_log.info(str(tqdm_iter))
        if self.validator is not None and is_main_process():
            validator_metrics = self.validator.evaluate()
            meters.update(validator_metrics, 1, use_DDP=False)

        meters.all_reduce()

    def sanity_check(self, dataloader: DataLoader):
        """Run sanity check"""
        self.evaluate(dataloader, stage="sanity", sanity=True)

    def single_epoch(self, train_dataloader: DataLoader):
        self.module.model.net.train()
        meters = self.meters["train"]
        meters.reset()
        limit_batches = self.get_limit_batches(train_dataloader)

        tqdm_iter = tqdm(train_dataloader, leave=True, desc="Train", ncols=100, total=limit_batches)
        for idx, batch in enumerate(tqdm_iter):
            self.warmup_lr(idx, train_dataloader)
            self.logger.file_log.info(str(tqdm_iter))
            if limit_batches == 0:
                break
            train_metrics = self.module._training_step(batch, idx)
            meters.update(train_metrics, self.datamodule.batch_size)
            self.callbacks.on_step_end(self)
            self.current_step += 1
            self.module.set_attributes(current_step=self.current_step)
            limit_batches -= 1
            remove_last_logged_line(self.logger.file_log)
        self.logger.file_log.info(str(tqdm_iter))
        meters.all_reduce()

    def _wait_for_all_workers(self):
        if self.use_DDP:
            dist.barrier()

    def on_epoch_start(self):
        pass

    def fit(
        self,
        module: BaseModule,
        datamodule: DataModule,
        validator: BaseValidator | None = None,
        pretrained_ckpt_path: str | None = None,
        ckpt_path: str | None = None,
    ):
        log_msg(colorstr("Setting up Trainer"))
        datamodule.setup_dataloaders()
        train_dataloader = datamodule.train_dataloader
        val_dataloader = datamodule.val_dataloader

        log_msg(
            "Dataloaders utilisation (per GPU):\n"
            f"\tTrain: {self.get_limit_batches(train_dataloader)}/{len(train_dataloader)} batches\n"
            f"\tVal  : {self.get_limit_batches(val_dataloader)}/{len(val_dataloader)}  batches"
        )

        module.pass_attributes(
            device_id=self.device_id,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            datamodule=datamodule,
            limit_batches=int(self._limit_batches),
            half_precision=self.half_precision,
        )
        # Correct order (?):
        # Compile -> CUDA -> weights init -> pretrained weights load -> previous run ckpt load -> DDP

        if self.use_compile and not self.use_DDP:
            module.model.compile()
        if self.accelerator == "gpu":
            module.model.to_CUDA(self.device_id)
            module.loss_fn.cuda(self.device_id)

        module.model.init_weights()
        if pretrained_ckpt_path is None:
            log_msg(
                "-> Skipping pretrained weights loading (pretrained_ckpt_path is None)",
                logging.WARN,
            )
        else:
            log_msg(f"Loading pretrained checkpoint (from '{pretrained_ckpt_path}')")
            pretrained_ckpt = torch.load(pretrained_ckpt_path, map_location=self.map_location)
            # NOTE: its trainer ckpt, so we need to extract model state from it
            if "module" in pretrained_ckpt:
                pretrained_ckpt = pretrained_ckpt["module"]["model"]
            module.model.init_pretrained_weights(pretrained_ckpt)

        if datamodule.batch_size < 1:
            e = ValueError("Batch size must be greater than 0")
            log_msg(str(e), logging.ERROR)
            raise e

        if self.half_precision:
            log_msg("-> Using half precision (autocast to float16)")

        self.module = module
        self.datamodule = datamodule
        self.validator = self.get_validator()
        self.module.set_optimizers()
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        self.callbacks.on_fit_start(self)

        if self.use_DDP:
            if self.use_compile:
                self.module.model.compile()
            self.module.model.to_DDP(self.device_id, self.sync_batchnorm)

        self.module.set_attributes(current_step=self.current_step)

        if self.run_sanity_check:
            self.sanity_check(val_dataloader)

        # self._wait_for_all_workers()
        if ckpt_path is None:
            msg = "<<<  Training started  >>>"
        else:
            msg = f"<<<  The training is resumed at: epoch={self.current_epoch}, step={self.current_step}  >>>"
        log_breaking_point(
            msg, n_top=2, n_bottom=2, top_char="*", bottom_char="*", num_chars=100, worker=0
        )
        start_epoch = int(self.current_epoch)
        try:
            for epoch in range(start_epoch, self.max_epochs):
                log_breaking_point(
                    f"<<<  Epoch {epoch} started  >>>", n_top=1, top_char=" ", worker=0
                )
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)  # train_dl uses shuffle=True
                self.current_epoch = epoch
                self.on_epoch_start()
                self.module.on_epoch_start()
                self.callbacks.on_epoch_start(self)
                self.module.set_attributes(current_epoch=epoch)
                self.single_epoch(train_dataloader)
                self.evaluate(val_dataloader, stage="val")
                self._update_metrics(stages=["train", "val"])
                # self._wait_for_all_workers()
                self.module.on_epoch_end()
                self.callbacks.on_epoch_end(self)
                # self._wait_for_all_workers()
                self.val_results.clear()
                log_breaking_point(
                    f"<<<  Epoch {epoch} finished  >>>", n_bottom=1, worker=0, bottom_char="="
                )
        except KeyboardInterrupt as e:
            log_msg(str(e) + "KeyboardInterrupt", logging.ERROR)
            self.callbacks.on_failure(self, Status.KILLED)
            self.logger.finalize(Status.KILLED)
            gc.collect()
            torch.cuda.empty_cache()
            raise e
        self.logger.finalize(status=Status.FINISHED)
        gc.collect()
        torch.cuda.empty_cache()

    def stop(self, status: Status = Status.STOPPED):
        ckpt_dir = self.logger.loggers[0].ckpt_dir
        ckpt_path = str(ckpt_dir / "last.pt")
        self.save_checkpoint(ckpt_path)
        self.logger.finalize(status)

    def load_checkpoint(self, ckpt_path: str):
        log_msg(f"->\tLoading checkpoint from '{ckpt_path}'")
        ckpt_state = torch.load(ckpt_path, map_location=self.map_location)
        self.epochs_metrics.load_state_dict(ckpt_state["metrics"]["steps"])
        self.validation_metrics.load_state_dict(ckpt_state["metrics"]["validation"])
        self.module.load_state_dict(ckpt_state["module"])
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        step, epoch = ckpt_state["step"], ckpt_state["epoch"]
        self.current_step = step
        self.current_epoch = epoch + 1
        log_msg(f"\tLoaded current_step ({step}) and current_epoch ({epoch}) state")

    def save_checkpoint(self, ckpt_path: str):
        if self.device_id != 0:  # save only for cuda:0 (DDP)
            return
        log_msg(
            f"->Saving checkpoint (epoch={self.current_epoch}, step={self.current_step}) to '{ckpt_path}'"
        )
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()
        callbacks_state = self.callbacks.state_dict()
        metrics_state = {
            "steps": self.epochs_metrics.state_dict(),
            "validation": self.validation_metrics.state_dict(),
        }
        logger_state = self.logger.state_dict()

        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "metrics": metrics_state,
            "callbacks": callbacks_state,
            "logger": logger_state,
            "epoch": self.current_epoch,
            "step": self.current_step,
        }
        torch.save(ckpt_state, ckpt_path)

    @abstractmethod
    def get_validator(self) -> BaseValidator:
        raise NotImplementedError()
