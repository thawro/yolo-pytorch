from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Literal

from PIL import Image

from src.logger.pylogger import log_msg

if TYPE_CHECKING:
    from src.base.storage import MetricsStorage

    from .trainer import Trainer

import glob
import time
from abc import abstractmethod
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.logger.loggers import Status
from src.logger.monitoring.system import SystemMetricsMonitor
from src.logger.pylogger import log
from src.utils.files import save_txt_to_file, save_yaml
from src.utils.torch_utils import model_info
from src.utils.training import is_main_process
from src.utils.utils import colorstr

from .results import BaseResult, plot_results
from .visualization import plot_metrics_matplotlib, plot_metrics_plotly, plot_system_monitoring

_log_mode = Literal["step", "epoch", "validation"]


def get_metrics_storage(trainer: Trainer, mode: _log_mode) -> MetricsStorage:
    if mode == "validation":
        return trainer.validation_metrics.aggregate_over_key(key="step")
    elif mode == "epoch":
        return trainer.epochs_metrics.aggregate_over_key(key="epoch")
    else:
        raise ValueError("Wrong logging mode")


class BaseCallback:
    @property
    def name_prefix(self) -> str:
        return colorstr(f"{self.__class__.__name__}: ")

    @abstractmethod
    def on_fit_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_start(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_validation_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_failure(self, trainer: Trainer, status: Status):
        pass

    @abstractmethod
    def on_step_end(self, trainer: Trainer):
        pass

    def state_dict(self) -> dict:
        return {}

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass


class Callbacks:
    def __init__(self, callbacks: list[BaseCallback], device_id: int):
        # make sure that only main process is callbacking
        if not is_main_process():
            callbacks = []
        self.callbacks = callbacks
        self.device_id = device_id

    def on_step_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_step_end(trainer)

    def on_fit_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_fit_start(trainer)

    def on_epoch_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_epoch_start(trainer)

    def on_epoch_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_validation_start(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_start(trainer)

    def on_validation_end(self, trainer: Trainer):
        for callback in self.callbacks:
            callback.on_validation_end(trainer)

    def on_failure(self, trainer: Trainer, status: Status):
        log.warn("Failure mode detected. Running callbacks `on_failure` methods")
        for callback in self.callbacks:
            callback.on_failure(trainer, status)

    def state_dict(self) -> dict[str, dict]:
        state_dict = {}
        for callback in self.callbacks:
            name = callback.__class__.__name__
            state_dict[name] = callback.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, dict]):
        for callback in self.callbacks:
            name = callback.__class__.__name__
            callback.load_state_dict(state_dict.get(name, {}))


class ArtifactsLoggerCallback(BaseCallback):
    """Call logger artifacts logging methods"""

    def on_fit_start(self, trainer: Trainer) -> None:
        data_examples_dir = str(trainer.logger.loggers[0].data_examples_dir)
        trainer.logger.log_artifacts(data_examples_dir, "data_examples")

    def on_step_end(self, trainer: Trainer) -> None:
        """Log logs files"""
        trainer.logger.log_logs()

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Log artifacts directories and/or files"""
        eval_examples_dir = str(trainer.logger.loggers[0].eval_examples_dir)
        trainer.logger.log_artifacts(eval_examples_dir, "eval_examples")

        log_dir = str(trainer.logger.loggers[0].log_path)
        metrics_filepaths = glob.glob(f"{log_dir}/epoch_metrics.*")
        for filepath in metrics_filepaths:
            trainer.logger.log_artifact(filepath, "epoch_metrics")
        log_msg(f"{self.name_prefix}Artifacts logged to remote (eval_examples and epoch_metrics).")

    def on_failure(self, trainer: Trainer, status: Status):
        log_msg("Finalizing loggers.", logging.WARN)
        trainer.logger.log_logs()
        trainer.logger.finalize(status=status)


class SaveModelCheckpoint(BaseCallback):
    def __init__(
        self,
        name: str = "best",
        stage: str | None = None,
        metric: str | None = None,
        mode: Literal["min", "max"] = "min",
        last: bool = False,
    ):
        self.name = name
        self.stage = stage
        self.metric = metric
        self.save_last = last
        self.mode = mode

        self.callback_name = f"{metric if metric else ''} {self.name} ({mode})"

        self.best = torch.inf if mode == "min" else -torch.inf
        if mode == "min":
            self.is_better = lambda x, y: x < y
        else:
            self.is_better = lambda x, y: x > y

    def save_model(self, trainer: Trainer):
        ckpt_dir = trainer.logger.loggers[0].ckpt_dir
        msg = f"{self.name_prefix}"
        if self.metric is not None and self.stage is not None:
            meters = trainer.meters[self.stage]
            if len(meters) == 0:
                e = ValueError(f"{self.metric} not yet logged to meters")
                log_msg(str(e), logging.ERROR)
                raise e
            last = meters[self.metric].avg
            msg += f"{self.stage}/{self.metric}, Current={last:.3e}, Best={self.best:.3e}"

            if self.is_better(last, self.best):
                self.best = last
                msg += f" (Found new best {self.stage}/{self.metric}: {self.best:.3e})"
                trainer.save_checkpoint(str(ckpt_dir / f"{self.name}.pt"))
        if self.save_last:
            trainer.save_checkpoint(str(ckpt_dir / "last.pt"))
        log_msg(msg)

    def on_epoch_end(self, trainer: Trainer):
        self.save_model(trainer)

    @property
    def state_metric_name(self) -> str:
        return f"best_{self.stage}_{self.metric}"

    def state_dict(self) -> dict:
        return {self.state_metric_name: self.best}

    def load_state_dict(self, state_dict: dict):
        self.best = state_dict.get(self.state_metric_name, self.best)
        log_msg(
            f'\tLoaded "{self.__class__.__name__}" state ({self.state_metric_name} = {self.best})'
        )


class ResultsPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(self, name: str, ncols: int = -1):
        self.name = name
        self.ncols = ncols

    def plot(self, trainer: Trainer, prefix: str) -> None:
        dirpath = trainer.logger.loggers[0].eval_examples_dir
        dirpath.mkdir(exist_ok=True, parents=True)
        if self.name != "":
            prefix = f"{prefix}_"
        filepath = str(dirpath / f"{prefix}{self.name}.jpg")
        if len(trainer.val_results) > 0:
            plot_results(
                trainer.val_results, plot_name=self.name, filepath=filepath, ncols=self.ncols
            )
            log_msg(
                f"{self.name_prefix}{self.name.capitalize()} results visualization saved at '{filepath}'"
            )
        else:
            log.warn("No results to visualize")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"validation_{trainer.current_step}")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, prefix=f"epoch_{trainer.current_epoch}")


class MetricsPlotterCallback(BaseCallback):
    """Plot per epoch metrics"""

    def plot(self, trainer: Trainer, mode: _log_mode) -> None:
        log_path = trainer.logger.loggers[0].log_path
        storage = get_metrics_storage(trainer, mode)
        if len(storage.metrics) == 0:
            log.warn(f"No metrics to plot logged yet (mode={mode})")
            return
        step_name = "epoch" if mode == "epoch" else "step"
        mpl_filepath = f"{log_path}/{mode}_metrics.jpg"
        plot_metrics_matplotlib(storage, step_name, filepath=mpl_filepath)

        plotly_filepath = f"{log_path}/{mode}_metrics.html"
        plot_metrics_plotly(storage, step_name, filepath=plotly_filepath)
        log_msg(f"{self.name_prefix}Metrics plots saved at '{mpl_filepath}'")

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.plot(trainer, mode="epoch")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.plot(trainer, mode="validation")


class SystemMetricsMonitoringCallback(BaseCallback):
    """Plot per epoch metrics"""

    def __init__(
        self,
        plot_every_n_sec: int = 30,
        sampling_interval: int = 10,
        samples_before_logging: int = 1,
        verbose: bool = False,
    ) -> None:
        self.prev_time = time.time()
        self.verbose = verbose
        self.plot_every_n_sec = plot_every_n_sec
        self.sampling_interval = sampling_interval
        self.samples_before_logging = samples_before_logging

    def plot(self, trainer: Trainer) -> None:
        log_path = trainer.logger.loggers[0].log_path
        filepath = f"{log_path}/logs/system_metrics.html"

        storage = trainer.system_monitoring
        if len(storage.metrics) > 0:
            plot_system_monitoring(storage, filepath)
            if self.verbose:
                log_msg(f"{self.name_prefix}System metrics saved at {filepath}")

    def on_fit_start(self, trainer: Trainer):
        system_monitor = SystemMetricsMonitor(
            sampling_interval=self.sampling_interval,
            samples_before_logging=self.samples_before_logging,
            metrics_callback=trainer.system_monitoring.update,
        )
        system_monitor.start()

    def on_step_end(self, trainer: Trainer) -> None:
        current_time = time.time()
        duration_sec = current_time - self.prev_time
        if duration_sec > self.plot_every_n_sec:
            self.plot(trainer)
            self.prev_time = current_time


class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    def save(self, trainer: Trainer, mode: _log_mode) -> None:
        storage = get_metrics_storage(trainer, mode)
        log_path = trainer.logger.loggers[0].log_path
        filepath = filepath = f"{log_path}/{mode}_metrics.yaml"

        if len(storage.metrics) == 0:
            log.warn(f"No metrics to save logged yet (mode={mode})")
            return
        metrics = storage.to_dict()
        save_yaml(metrics, filepath)

    def on_epoch_end(self, trainer: Trainer) -> None:
        self.save(trainer, mode="epoch")

    def on_validation_end(self, trainer: Trainer) -> None:
        self.save(trainer, mode="validation")


class MetricsLogger(BaseCallback):
    """Log per epoch metrics to terminal"""

    def on_epoch_end(self, trainer: Trainer) -> None:
        msgs = ""
        for stage, metrics in trainer.epochs_metrics.inverse_nest().items():
            last_epoch_metrics = {name: values[-1]["value"] for name, values in metrics.items()}
            stage_msg = []
            for name, value in last_epoch_metrics.items():
                # optim_LR_names = [f"{name}_LR" for name in trainer.module.optimizers.keys()]
                restricted_names = ["epoch", "step"]
                if name not in restricted_names:
                    if value <= 1e-3 or value >= 1e3:
                        value_str = f"{value:.3e}"
                    else:
                        value_str = f"{value:.3f}"
                    stage_msg.append(f"{name}: {value_str}")
            stage_msg = stage.rjust(8) + ":\t" + "   ".join(stage_msg) + "\n"
            msgs += stage_msg
        prefix = colorstr(f"Epoch {trainer.current_epoch}: ")
        log_msg(f"{prefix}\n{msgs}")


class ModelSummary(BaseCallback):
    def __init__(self, depth: int = 6):
        self.depth = depth

    def on_fit_start(self, trainer: Trainer):
        log_msg(f"{colorstr('Optimizers: ')}\n{trainer.module.optimizers}")
        log_msg(f"{colorstr('LR Schedulers: ')}\n{trainer.module.lr_schedulers}")

        net = trainer.module.model.net
        if isinstance(net, DDP):
            net = net.module
        model_summary, *_ = model_info(net)
        # model_summary = model.summary(self.depth)
        log_msg(f"{self.name_prefix}\n{model_summary}")
        model_dir = trainer.logger.loggers[0].model_dir
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        filepath = f"{model_dir}/model_summary.txt"
        save_txt_to_file(model_summary, filepath)


class DatasetExamplesCallback(BaseCallback):
    def __init__(
        self,
        splits: list[Literal["train", "val", "test"]] = ["train"],
        n: int = 10,
        random_idxs: bool = False,
    ) -> None:
        self.splits = splits
        self.n = n
        self.random_idxs = random_idxs

    def on_fit_start(self, trainer: Trainer):
        log_msg(f"{self.name_prefix}Saving datasets samples examples")
        dirpath = trainer.logger.loggers[0].data_examples_dir
        for split in self.splits:
            if split == "train":
                dataset = trainer.datamodule.train_dataloader.dataset
            elif split == "val":
                dataset = trainer.datamodule.val_dataloader.dataset
            dirpath.mkdir(exist_ok=True, parents=True)
            filepath = str(dirpath / f"{split}.jpg")

            if self.random_idxs:
                idxs = [random.randint(0, len(dataset)) for _ in range(self.n)]
            else:
                idxs = list(range(self.n))
            grid = dataset.plot_examples(idxs, nrows=1)
            Image.fromarray(grid).save(filepath)
            log_msg(f"\t{split} dataset examples saved at '{filepath}'")


class EarlyStoppingCallback(BaseCallback):
    def __init__(
        self,
        metric: str,
        stage: str,
        mode: Literal["min", "max"] = "min",
        patience: int = 30,
    ):
        assert mode in {"min", "max"}
        self.metric = metric
        self.stage = stage
        self.mode = mode
        self.patience = patience
        self.best_epoch = 0
        self.best = torch.inf if mode == "min" else -torch.inf
        if mode == "min":
            self.is_better = lambda x, y: x < y
        else:
            self.is_better = lambda x, y: x > y

    def check(self, trainer: Trainer):
        meters = trainer.meters[self.stage]
        if len(meters) == 0:
            e = ValueError(f"{self.metric} not yet logged to meters")
            log.exception(e)
            raise e
        last = meters[self.metric].avg

        if self.is_better(last, self.best):
            self.best = last
            self.best_epoch = trainer.current_epoch

        epoch_diff = trainer.current_epoch - self.best_epoch
        should_stop = epoch_diff >= self.patience
        if should_stop:
            log_msg(
                f"{self.name_prefix}Stopping training early. No improvement for > {self.patience} epochs."
            )
            trainer.stop(Status.STOPPED)

    def on_epoch_end(self, trainer: Trainer):
        self.check(trainer)

    @property
    def state_metric_name(self) -> str:
        return f"best_{self.stage}_{self.metric}"

    def state_dict(self) -> dict:
        return {self.state_metric_name: self.best}

    def load_state_dict(self, state_dict: dict):
        self.best = state_dict.get(self.state_metric_name, self.best)
        log_msg(
            f'\tLoaded "{self.__class__.__name__}" state ({self.state_metric_name} = {self.best})'
        )
