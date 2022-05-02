import os
from pathlib import Path
from .wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import DummyLogger
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from .tools import cfg_to_flatten_config
import types


def instantiate_logger(cfg: DictConfig):
    conf = OmegaConf.to_container(cfg.logger, resolve=True)
    name = conf.pop("logger_name")

    if name == "wandb":
        save_dir = conf["save_dir"]
        project_save_dir = to_absolute_path(save_dir)
        Path(project_save_dir).mkdir(exist_ok=True)
        conf["save_dir"] = project_save_dir
        conf["config"] = cfg_to_flatten_config(cfg)
        logger = WandbLogger(**conf)
        # begin / end already defined
    else:
        def begin(self, *args, **kwargs):
            return

        def end(self, *args, **kwargs):
            return

        if name == "tensorboard":
            logger = TensorBoardLogger(**conf)
            logger.begin = begin
            logger.end = end
        elif name in ["none", None]:
            logger = DummyLogger()
            logger.begin = begin
            logger.end = end
        else:
            raise NotImplementedError("This logger is not recognized.")

    logger.lname = name
    return logger
