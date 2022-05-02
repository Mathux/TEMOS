import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import temos.launch.prepare  # noqa

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train")
def _train(cfg: DictConfig):
    cfg.trainer.enable_progress_bar = True
    return train(cfg)


def train(cfg: DictConfig) -> None:
    working_dir = cfg.path.working_dir
    logger.info("Training script. The outputs will be stored in:")
    logger.info(f"{working_dir}")

    # Delayed imports to get faster parsing
    logger.info("Loading libraries")
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from temos.logger import instantiate_logger
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    logger.info("Loading model")
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")

    logger.info("Loading callbacks")
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose"
    }
    callbacks = [
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.latest_ckpt),
        instantiate(cfg.callback.last_ckpt)
    ]
    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=None,
        callbacks=callbacks,
    )
    logger.info("Trainer initialized")

    logger.info("Fitting the model..")
    trainer.fit(model, datamodule=data_module)
    logger.info("Fitting done")

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"Training done. The outputs of this experiment are stored in:\n{working_dir}")


if __name__ == '__main__':
    _train()
