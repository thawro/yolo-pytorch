import collections.abc
import json
import logging
import sys
from abc import abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Type

from dacite import from_dict
from torch import nn

from src.logger.loggers import Loggers, MLFlowLogger, Status, TerminalLogger
from src.logger.monitoring import NvidiaSmiMonitor
from src.logger.pylogger import get_file_pylogger, log, log_msg
from src.utils import NOW, RESULTS_PATH
from src.utils.files import load_yaml
from src.utils.training import get_device_and_id, is_main_process

from .callbacks import (
    ArtifactsLoggerCallback,
    BaseCallback,
    DatasetExamplesCallback,
    MetricsLogger,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
    SystemMetricsMonitoringCallback,
)
from .datamodule import DataModule
from .model import BaseInferenceModel
from .module import BaseModule
from .trainer import Trainer


@dataclass
class AbstractConfig:
    def to_dict(self) -> dict:
        dct = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            if hasattr(field_value, "to_dict"):
                dct[field_name] = field_value.to_dict()
            else:
                if isinstance(field_value, dict):  # Config class nested in dict (e.g. optimizers)
                    init_dict = {}
                    for k, v in field_value.items():
                        if hasattr(v, "to_dict"):
                            init_dict[k] = v.to_dict()
                        else:
                            init_dict[k] = v
                    dct[field_name] = init_dict
                else:
                    dct[field_name] = field_value
        return dct

    @classmethod
    def from_dict(cls, cfg_dict: dict):
        return from_dict(data_class=cls, data=cfg_dict)

    @classmethod
    def from_yaml_to_dict(
        cls, filepath: str | Path, verbose: bool = True
    ) -> dict | tuple[dict, str]:
        cfg_dict = load_yaml(filepath)
        cfg_dict, msg = update_config(cfg_dict, cls)
        if verbose:
            return cfg_dict, msg
        return cfg_dict


@dataclass
class TransformConfig(AbstractConfig):
    size: int | tuple[int, int] | list[int]


@dataclass
class DatasetConfig(AbstractConfig):
    root: str
    split: str


@dataclass
class DataloaderConfig(AbstractConfig):
    batch_size: int
    pin_memory: bool
    num_workers: int
    train_ds: DatasetConfig
    val_ds: DatasetConfig


@dataclass
class NetConfig(AbstractConfig):
    params: dict
    stride: int


@dataclass
class TrainerConfig(AbstractConfig):
    accelerator: str
    max_epochs: int
    limit_batches: int
    use_DDP: bool
    sync_batchnorm: bool
    use_compile: bool
    half_precision: bool = False


@dataclass
class SetupConfig(AbstractConfig):
    seed: int
    experiment_name: str
    is_train: bool
    ckpt_path: str | None
    pretrained_ckpt_path: str | None
    deterministic: bool
    architecture: str
    dataset: str
    run_name: str | None = None

    def _auto_run_name(self) -> str:
        if self.ckpt_path is None:
            # new run
            return f"{self.timestamp}__{self.dataset}_{self.architecture}"
        # resumed run ckpt_path is like:
        # "<proj_root>/results/<exp_name>/<run_name>/<timestamp>/checkpoints/<ckpt_name>.pt"
        # so run_name is -4 idx after split
        run_name = self.ckpt_path.split("/")[-4]
        return run_name

    def __post_init__(self):
        self.timestamp = NOW
        if self.run_name is None and self.is_train:
            self.run_name = self._auto_run_name()


@dataclass
class CUDNNConfig(AbstractConfig):
    benchmark: bool
    deterministic: bool
    enabled: bool


@dataclass
class OptimizerConfig(AbstractConfig):
    name: str
    params: dict


@dataclass
class LRSchedulerConfig(AbstractConfig):
    name: str
    interval: Literal["epoch", "step"]
    warmup_start_momentum: float
    warmup_start_bias_lr: float
    warmup_epochs: int
    params: dict


@dataclass
class ModuleConfig(AbstractConfig):
    multiscale: bool
    optimizers: dict[str, OptimizerConfig]
    lr_schedulers: dict[str, LRSchedulerConfig]


@dataclass
class InferenceConfig(AbstractConfig):
    input_size: int
    ckpt_path: str


@dataclass
class BaseConfig(AbstractConfig):
    setup: SetupConfig
    cudnn: CUDNNConfig
    dataloader: DataloaderConfig
    transform: TransformConfig
    net: NetConfig
    module: ModuleConfig
    trainer: TrainerConfig
    inference: InferenceConfig

    def __post_init__(self):
        if self.is_debug:
            self.setup.experiment_name = "debug"
            if is_main_process():
                log_msg(
                    "-> Running in debug mode (`limit_batches` is > 0). Updating config: `setup.experiment.name = 'debug'`",
                    logging.CRITICAL,
                )
        self.device, self.device_id = get_device_and_id(
            self.trainer.accelerator, self.trainer.use_DDP
        )
        if not self.setup.is_train:  # no logging for eval/test/inference
            return
        self._initialize_logger()

    def _initialize_logger(self):
        file_log_dirpath = f"{self.log_path}/logs"
        Path(file_log_dirpath).mkdir(exist_ok=True, parents=True)
        file_log_filepath = f"{file_log_dirpath}/{self.device}_log.log"
        file_log = get_file_pylogger(file_log_filepath, "log_file")
        # insert handler to enable file logs in command line aswell
        log.handlers.insert(0, file_log.handlers[0])
        log_msg(f"-> Saving {self.device} logs to {file_log_filepath}")
        for handler in log.handlers:
            formatter = handler.formatter
            if formatter is not None and hasattr(formatter, "_set_device"):
                handler.formatter._set_device(self.device, self.device_id)
        self.logger = self.create_logger(file_log)
        if is_main_process():
            nvidia_log_filepath = f"{file_log_dirpath}/nvidia-smi.log"
            nvidia_monitor = NvidiaSmiMonitor(nvidia_log_filepath, sampling_interval=5)
            nvidia_monitor.start()
            self.logger.start_run()

    @property
    def log_path(self) -> str:
        ckpt_path = self.setup.ckpt_path
        if ckpt_path is None or self.is_debug:
            _log_path = str(RESULTS_PATH / self.setup.experiment_name / self.setup.run_name / NOW)
        else:
            loaded_run_path = Path(ckpt_path).parent.parent.parent
            _log_path = str(loaded_run_path / NOW)
        Path(_log_path).mkdir(exist_ok=True, parents=True)
        return _log_path

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    def get_optimizers_params(self) -> dict[str, dict]:
        log_msg("-> Parsing Optimizers config")
        return {
            name: optimizer_cfg.to_dict() for name, optimizer_cfg in self.module.optimizers.items()
        }

    def get_lr_schedulers_params(self) -> dict[str, dict]:
        log_msg("-> Parsing LR Schedulers config")
        return {
            name: scheduler_cfg.to_dict()
            for name, scheduler_cfg in self.module.lr_schedulers.items()
        }

    @property
    @abstractmethod
    def architectures(self) -> dict[str, Type[nn.Module]]:
        raise NotImplementedError()

    def create_net(self) -> nn.Module:
        arch = self.setup.architecture
        params_repr = "\n".join([f"\t{k}: {v}" for k, v in self.net.params.items()])
        arch_repr = f"{arch}\n{params_repr}"
        log_msg(f"-> Initializing {arch}\n{arch_repr}")

        assert (
            arch in self.architectures
        ), f"model.architecture must be one of {list(self.architectures.keys())}"

        ArchitectureClass = self.architectures[arch]
        net = ArchitectureClass(**self.net.params)
        return net

    @abstractmethod
    def create_datamodule(self) -> DataModule:
        raise NotImplementedError()

    @abstractmethod
    def create_module(self) -> BaseModule:
        raise NotImplementedError()

    def create_callbacks(self) -> list[BaseCallback]:
        log_msg("-> Initializing Callbacks")

        callbacks = [
            MetricsPlotterCallback(),
            MetricsSaverCallback(),
            MetricsLogger(),
            DatasetExamplesCallback(splits=["train", "val"], n=20, random_idxs=True),
            SaveModelCheckpoint(name="best", metric="loss", last=True, mode="min", stage="val"),
            SystemMetricsMonitoringCallback(),
            ModelSummary(depth=5),
            ArtifactsLoggerCallback(),  # make sure it is last
        ]
        for callback in callbacks:
            log_msg(f"\t[v] {callback.__class__.__name__}")

        return callbacks

    def create_logger(self, file_log: logging.Logger) -> Loggers:
        log_msg("-> Initializing Logger")

        loggers = [
            MLFlowLogger(
                self.log_path,
                config=self.to_dict(),
                experiment_name=self.setup.experiment_name,
                run_name=self.setup.run_name,
                resume=True,
            )
        ]
        # loggers = [TerminalLogger(self.log_path, self.to_dict())]
        logger = Loggers(loggers, self.device_id, file_log)
        return logger

    @property
    def TrainerClass(self) -> Type[Trainer]:
        return Trainer

    def create_trainer(self) -> Trainer:
        log_msg("-> Initializing Trainer")
        try:
            callbacks = self.create_callbacks()
            trainer = self.TrainerClass(
                logger=self.logger,
                callbacks=callbacks,
                **self.trainer.to_dict(),
            )
            return trainer
        except Exception as e:
            log_msg(str(e), logging.ERROR)
            self.logger.finalize(Status.FAILED)
            raise e

    @abstractmethod
    def create_inference_model(self, device: str = "cuda:0", **kwargs) -> BaseInferenceModel:
        raise NotImplementedError()


def parse_cli_value(value: str) -> int | float | str | None:
    if value in ["None", "none", "null"]:
        return None
    elif "." in value:
        try:
            return float(value)
        except ValueError:
            return value
    else:
        if value in ["true", "True"]:
            return True
        elif value in ["false", "False"]:
            return False
        try:
            return int(value)
        except ValueError:
            return value


def update_dict(dct: dict, update_dct: dict, msg: str = "") -> tuple[dict, str]:
    for k, v in update_dct.items():
        if isinstance(v, collections.abc.Mapping):
            dct[k], msg = update_dict(dct.get(k, {}), v, msg)
        else:
            new_value = parse_cli_value(v)
            msg += f"\t{k} value updated to {new_value}\n"
            dct[k] = new_value
    return dct, msg


def parse_args_for_config(ConfigClass: Type[BaseConfig] = BaseConfig) -> dict:
    def parse_dict(update_dct: dict) -> dict:
        output_dict = {}
        for key, value in update_dct.items():
            parts = key.split(".")
            current_dict = output_dict
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[parts[-1]] = value
        return output_dict

    args = sys.argv
    valid_args = [arg[2:] for arg in args if arg[:2] == "--" and "=" in arg]
    name_values = [arg.split("=") for arg in valid_args]
    name2value = {name: value for name, value in name_values}
    name2value = parse_dict(name2value)
    cfg_fields = fields(ConfigClass)
    cfg_fields_names = [field.name for field in cfg_fields]
    name2value = {name: value for name, value in name2value.items() if name in cfg_fields_names}
    return name2value


def update_config(cfg_dict: dict, ConfigClass: Type[BaseConfig] = BaseConfig) -> tuple[dict, str]:
    update_dct = parse_args_for_config(ConfigClass)

    msg = ""
    if len(update_dct) > 0:
        msg = f"-> Updating config dict using CLI args:\n{json.dumps(update_dct, indent=2)}\n"
        cfg_dict, msg = update_dict(cfg_dict, update_dct, msg)
    return cfg_dict, msg
