import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import temos.launch.prepare  # noqa

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="interact")
def _interact(cfg: DictConfig):
    return interact(cfg)


def cfg_mean_nsamples_resolution(cfg):
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


def load_checkpoint(model, last_ckpt_path, *, eval_mode):
    # Load the last checkpoint
    # model = model.load_from_checkpoint(last_ckpt_path)
    # this will overide values
    # for example relative to rots2joints
    # So only load state dict is preferable
    import torch
    model.load_state_dict(torch.load(last_ckpt_path)["state_dict"])
    logger.info("Model weights restored.")

    if eval_mode:
        model.eval()
        logger.info("Model in eval mode.")


def interact(newcfg: DictConfig) -> None:
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path

    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    oneinteract = cfg_mean_nsamples_resolution(cfg)

    text = cfg.text
    logger.info(f"Interaction script. The result will be saved there: {cfg.saving}")
    logger.info(f"The sentence is: {text}")

    filename = (text
                .lower()
                .strip()
                .replace(" ", "_")
                .replace(".", "") + "_len_" + str(cfg.length)
                )

    os.makedirs(cfg.saving, exist_ok=True)
    path = Path(cfg.saving)

    import pytorch_lightning as pl
    import numpy as np
    import torch
    from hydra.utils import instantiate
    pl.seed_everything(cfg.seed)

    logger.info("Loading model")
    if cfg.jointstype == "vertices":
        assert cfg.gender in ["male", "female", "neutral"]
        logger.info(f"The topology will be {cfg.gender}.")
        cfg.model.transforms.rots2joints.gender = cfg.gender

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)

    logger.info(f"Model '{cfg.model.modelname}' loaded")

    load_checkpoint(model, last_ckpt_path, eval_mode=True)

    if "amass" in cfg.data.dataname and "xyz" not in cfg.data.dataname:
        model.transforms.rots2joints.jointstype = cfg.jointstype

    model.sample_mean = cfg.mean
    model.fact = cfg.fact

    if not model.hparams.vae and cfg.number_of_samples > 1:
        raise TypeError("Cannot get more than 1 sample if it is not a VAE.")

    from temos.data.tools.collate import collate_text_and_length

    from temos.data.sampling import upsample
    from rich.progress import Progress
    from rich.progress import track

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    import torch
    with torch.no_grad():
        if True:
        # with Progress(transient=True) as progress:
            # task = progress.add_task("Sampling", total=len(dataset.keyids))
            # progress.update(task, description=f"Sampling {keyid}..")
            for index in range(cfg.number_of_samples):
                # batch_size = 1 for reproductability
                element = {"text": text, "length": cfg.length}
                batch = collate_text_and_length([element])

                # fix the seed
                pl.seed_everything(50 + index)

                if cfg.jointstype == "vertices":
                    vertices = model(batch)[0]
                    motion = vertices.numpy()
                    # no upsampling here to keep memory
                    # vertices = upinteract(vertices, cfg.data.framerate, 100)
                else:
                    joints = model(batch)[0]
                    motion = joints.numpy()
                    # upscaling to compare with other methods
                    motion = upsample(motion, cfg.data.framerate, 100)

                if cfg.number_of_samples > 1:
                    npypath = path / f"{filename}_{index}.npy"
                else:
                    npypath = path / f"{filename}.npy"
                np.save(npypath, motion)
                # progress.update(task, advance=1)

    logger.info("All the sampling are done")
    logger.info(f"All the sampling are done. You can find them here:\n{path}")


if __name__ == '__main__':
    _interact()
