import logging
from omegaconf import DictConfig
from typing import Optional


def is_slurm(hydra_conf: DictConfig):
    launcher = hydra_conf.launcher._target_
    if launcher == 'hydra._internal.core_plugins.basic_launcher.BasicLauncher':
        return False
    elif launcher == 'hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher':
        return True
    else:
        raise NotImplementedError(f"This launcher {launcher} is not recognized.")


def prepare_if_slurm(cfg: DictConfig, logger, hydra_conf: Optional[DictConfig] = None):
    import sys

    if hydra_conf is None:
        from hydra.core.hydra_config import HydraConfig
        hydra_conf = HydraConfig.get()

    # Check if it is launched with slurm
    if not is_slurm(hydra_conf):
        # Default value in trainer
        # to get the progress bar
        cfg.trainer.enable_progress_bar = True
    else:
        # Disable tqdm bars
        # in data
        cfg.data.progress_bar = False
        # in trainer
        cfg.trainer.enable_progress_bar = False

        # Redirect stdout and stderr
        # to logging files
        # stdout => logging.INFO => logs.out
        # stderr => logging.ERROR => logs.err
        from temos.tools.logging import StreamToLogger
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
