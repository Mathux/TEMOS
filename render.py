import os
import sys

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError("Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

import temos.launch.blender
import temos.launch.prepare  # noqa
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="render")
def _render_cli(cfg: DictConfig):
    return render_cli(cfg)


def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):
    if not onesample:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [template_path.replace("INDEX", str(index)) for i in range(number_of_samples)]
    else:
        paths = [str(path / "KEYID.npy")]

    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths



def render_cli(cfg: DictConfig) -> None:
    if cfg.npy is None:
        if cfg.folder is None or cfg.split is None:
            raise ValueError("You should either use npy=XXX.npy, or folder=XXX and split=XXX")

        from temos.data.utils import get_split_keyids
        from pathlib import Path
        from evaluate import get_samples_folder
        from sample import cfg_mean_nsamples_resolution, get_path
        keyids = get_split_keyids(path=Path(cfg.path.datasets)/ "kit-splits", split=cfg.split)

        onesample = cfg_mean_nsamples_resolution(cfg)
        model_samples, amass, jointstype = get_samples_folder(cfg.folder,
                                                              jointstype=cfg.jointstype)
        assert "mmm" in jointstype

        path = get_path(model_samples, cfg.split, onesample, cfg.mean, cfg.fact)
        paths = extend_paths(path, keyids, onesample=onesample, number_of_samples=cfg.number_of_samples)
    else:
        paths = [cfg.npy]

    from temos.render.blender import render
    from temos.render.video import Video
    import numpy as np

    init = True
    for path in paths:
        try:
            data = np.load(path)
        except FileNotFoundError:
            logger.info(f"{path} not found")
            continue

        if cfg.mode == "video":
            frames_folder = path.replace(".npy", "_frames")
        else:
            frames_folder = path.replace(".npy", ".png")

        out = render(data, frames_folder,
                     cycle=cfg.cycle, high_res=cfg.high_res,
                     canonicalize=cfg.canonicalize,
                     exact_frame=cfg.exact_frame,
                     num=cfg.num, mode=cfg.mode,
                     faces_path=cfg.faces_path,
                     downsample=cfg.downsample,
                     always_on_floor=cfg.always_on_floor,
                     white_back=cfg.white_back,
                     init=init,
                     gt=cfg.gt)

        init = False

        if cfg.mode == "video":
            if cfg.downsample:
                video = Video(frames_folder, fps=12.5)
            else:
                video = Video(frames_folder, fps=100.0)

            vid_path = path.replace(".npy", ".mp4")
            video.save(out_path=vid_path)
            logger.info(vid_path)

        else:
            logger.info(f"Frame generated at: {out}")



if __name__ == '__main__':
    _render_cli()
