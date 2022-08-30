import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import temos.launch.prepare  # noqa

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="stats")
def _stats(cfg: DictConfig):
    return stats(cfg)


def stats(cfg: DictConfig):
    logger.info(f"Computing stats for this data: {cfg.data.dataname}")

    logger.info("Loading data module")
    data_module = hydra.utils.instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    dataset = data_module.train_dataset

    import torch
    datastructs = [x["datastruct"] for x in dataset]

    def compute_stats_feats(datastructs, key):
        feats = torch.cat([datastruct[key] for datastruct in datastructs])
        mean = feats.mean(0)
        std = feats.std(0)
        return mean, std

    for transformation in ["joints2jfeats", "rots2rfeats"]:
        if transformation not in cfg.transforms:
            continue

        savepath = Path(cfg.transforms[transformation].path)
        savepath.mkdir(parents=True, exist_ok=True)

        feat = transformation[transformation.find("2")+1:]

        logger.info(f"Computing {feat} stats.")
        logger.info(f"It will be saved there: {savepath}")

        mean, std = compute_stats_feats(datastructs, feat)

        torch.save(mean, savepath / f"{feat}_mean.pt")
        torch.save(std, savepath / f"{feat}_std.pt")

    logger.info("All stats saved.")


if __name__ == '__main__':
    _stats()
