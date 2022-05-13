import logging
import yaml
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import temos.launch.prepare  # noqa


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="evaluate")
def _evaluate(cfg: DictConfig):
    return evaluate(cfg)


def regroup_metrics(metrics):
    from temos.info.joints import mmm_joints
    pose_names = mmm_joints[1:]
    dico = {key: val.numpy() for key, val in metrics.items()}

    APE_pose = dico.pop("APE_pose")
    APE_joints = dico.pop("APE_joints")

    for name, ape in zip(pose_names, APE_pose):
        dico[f"APE_pose_{name}"] = ape

    for name, ape in zip(mmm_joints, APE_joints):
        dico[f"APE_joints_{name}"] = ape

    AVE_pose = dico.pop("AVE_pose")
    AVE_joints = dico.pop("AVE_joints")

    for name, ave in zip(pose_names, AVE_pose):
        dico[f"AVE_pose_{name}"] = ave

    for name, ape in zip(mmm_joints, AVE_joints):
        dico[f"AVE_joints_{name}"] = ave

    return dico

def sanitize(dico):
    dico = {key: "{:.5f}".format(float(val)) for key, val in dico.items()}
    return dico


def get_samples_folder(path, *, jointstype):
    if jointstype == "vertices":
        raise ValueError("No evaluation for vertices, sample the joints instead.")

    output_dir = Path(hydra.utils.to_absolute_path(path))
    candidates = [x for x in os.listdir(output_dir) if "samples" in x]
    if not candidates:
        raise ValueError("There is no samples for this model.")

    amass = False
    for candidate in candidates:
        amass = amass or ("amass" in candidate)

    if amass:
        samples_path = output_dir / f"amass_samples_{jointstype}"
        if not samples_path.exists():
            jointstype = "mmm"
            samples_path = output_dir / f"amass_samples_mmm"
            if not samples_path.exists():
                raise ValueError("You must specify a correct jointstype.")
            logger.info(f"Samples from {jointstype} not found, take mmm instead.")
    else:
        samples_path = output_dir / "samples"
    return samples_path, amass, jointstype


def get_metric_paths(sample_path: Path, amass: bool, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    metric_str = "amass_metrics" if amass else "metrics"

    if onesample:
        file_path = f"{fact_str}{metric_str}_{split}{extra_str}"
        save_path = sample_path / file_path
        return save_path
    else:
        file_path = f"{fact_str}{metric_str}_{split}_multi"
        avg_path = sample_path / (file_path + "_avg")
        best_path = sample_path / (file_path + "_best")
        return avg_path, best_path


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)


def evaluate(cfg: DictConfig) -> None:
    logger.info(f"Evaluation script.")

    from sample import cfg_mean_nsamples_resolution, get_path
    onesample = cfg_mean_nsamples_resolution(cfg)
    model_samples, amass, jointstype = get_samples_folder(cfg.folder,
                                                          jointstype=cfg.jointstype)
    split = cfg.split

    path = get_path(model_samples, cfg.split, onesample, cfg.mean, cfg.fact)
    file_path = f"amass_metrics_{split}" if amass else f"metrics_{split}"

    save_paths = get_metric_paths(model_samples, amass, cfg.split, onesample, cfg.mean, cfg.fact)
    if onesample:
        save_path = save_paths
        logger.info(f"The outputs will be stored in: {save_path}")
    else:
        avg_path, best_path = save_paths
        logger.info(f"The outputs will be stored in: {avg_path} and {best_path}")

    logger.info("Loading the libraries")
    import numpy as np
    import torch
    import json
    from hydra.utils import instantiate
    from temos.data.kit import load_mmm_keyid, load_amass_keyid
    from temos.data.utils import get_split_keyids
    from temos.model.metrics import ComputeMetrics, ComputeMetricsBest
    logger.info("Libraries loaded")

    datapath = Path(cfg.path.datasets) / "kit"
    if amass:
        from temos.data.tools.smpl import smpl_data_to_matrix_and_trans
        rots2joints = instantiate(cfg.rots2joints, jointstype=jointstype)
        amass_path = Path(cfg.path.datasets) / "AMASS"
        correspondance_path = str(Path(cfg.path.datasets) / "kitml_amass_path.json")
        with open(correspondance_path) as correspondance_path_file:
            kitml_correspondances = json.load(correspondance_path_file)

    # If mmmns, it is smpl scale, so it is already in meters
    force_in_meter = cfg.jointstype != "mmmns"
    if onesample:
        CMetrics = ComputeMetrics(force_in_meter=force_in_meter)
    else:
        CMetrics_best = ComputeMetricsBest(force_in_meter=force_in_meter)
        CMetrics_avg = [ComputeMetrics(force_in_meter=force_in_meter) for index in range(cfg.number_of_samples)]

    logger.info(f"Computing the {split} metrics")

    keyids = get_split_keyids(Path(cfg.path.datasets) / "kit-splits", split)
    # keep infos for computing
    all_infos = []
    for keyid in keyids:
        # Load GT data
        # load mmm
        if not amass:
            # Load reference joints in MMM format
            ref_joints = load_mmm_keyid(keyid, datapath)
            ref_joints = torch.from_numpy(ref_joints).float()
        else:
            ref_smpl_data, success = load_amass_keyid(keyid, amass_path, correspondances=kitml_correspondances)
            if not success:
                logger.info(f"{keyid}.npy is not found (in the ground truth). Ignore it (this happend for AMASS)")
                continue

            ref_smpl_data = {"poses": torch.from_numpy(ref_smpl_data["poses"]).float(),
                             "trans": torch.from_numpy(ref_smpl_data["trans"]).float()}

            ref_smpl_data = smpl_data_to_matrix_and_trans(ref_smpl_data, nohands=True)
            ref_joints = rots2joints(ref_smpl_data)

        # save them to compute best metric
        if not onesample:
            model_joints_all = []
            ref_joints_all = []
            length_all = []
        for index in range(cfg.number_of_samples):
            # Load model joints
            seq_id = "" if onesample else f"_{index}"
            model_joints = np.load(path / f"{keyid}{seq_id}.npy")
            model_joints = torch.from_numpy(model_joints).float()

            # Take the common lengths to facilitate the computation
            length = min(len(model_joints), len(ref_joints))

            if onesample:
                # Compute part of the metrics
                CMetrics.update(model_joints[None], ref_joints[None], [length])
            else:
                CMetrics_avg[index].update(model_joints[None], ref_joints[None], [length])
                # keep them all to compute the best one
                model_joints_all.append(model_joints[None])
                ref_joints_all.append(ref_joints[None])
                length_all.append([length])

        if not onesample:
            CMetrics_best.update(model_joints_all, ref_joints_all, length_all)

    if onesample:
        metrics = sanitize(regroup_metrics(CMetrics.compute()))
        logger.info(f"All done, saving at {save_path}")
        save_metric(save_path, metrics)
        logger.info("Done.")

        for key in ["APE_root", "AVE_root"]:
            logger.info(f"{key}: {metrics[key]}")
    else:
        # best metrics
        best_metrics = sanitize(regroup_metrics(CMetrics_best.compute()))

        avgs = []
        for index in range(cfg.number_of_samples):
            avgs.append(regroup_metrics(CMetrics_avg[index].compute()))

        # avg metrics
        avg_metrics = sanitize({key: np.mean([avg[key] for avg in avgs]) for key in avgs[0].keys()})

        logger.info(f"All done, saving at {best_path} and {avg_path}")
        save_metric(avg_path, avg_metrics)
        save_metric(best_path, best_metrics)
        logger.info("Done.")

        for name, metrics in [("avg", avg_metrics), ("best", best_metrics)]:
            logger.info(f"{name}")
            for key in ["APE_root", "AVE_root"]:
                logger.info(f"  {key}: {metrics[key]}")


if __name__ == '__main__':
    _evaluate()
