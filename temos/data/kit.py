import json
import os
from glob import glob
from typing import Dict, Optional
import logging

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

from temos.tools.easyconvert import matrix_to, axis_angle_to
from temos.transforms import Transform
from temos.data.sampling import subsample
from temos.data.tools.smpl import smpl_data_to_matrix_and_trans

from .base import BASEDataModule
from .utils import get_split_keyids

logger = logging.getLogger(__name__)


class KITDataModule(BASEDataModule):
    def __init__(self, data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)
        self.Dataset = KIT

        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        self.transforms = self._sample_set.transforms


class KIT(Dataset):
    dataname = "KIT Motion-Language"

    def __init__(self, datapath: str,
                 splitpath: str,
                 transforms: Transform,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 correspondance_path: str = None,
                 amass_path: str = None,
                 smplh_path: str = None,
                 sampler=None,
                 framerate: float = 12.5,
                 progress_bar: bool = True,
                 pick_one_text: bool = True,
                 load_amass_data=False,
                 load_with_rot=False,
                 downsample=True,
                 tiny: bool = False, **kwargs):

        self.split = split
        self.load_amass_data = load_amass_data
        self.load_with_rot = load_with_rot
        self.downsample = downsample

        if load_amass_data and not self.load_with_rot:
            self.transforms_xyz = transforms_xyz
            self.transforms_smpl = transforms_smpl
            self.transforms = transforms_xyz
        else:
            self.transforms = transforms

        self.sampler = sampler
        self.pick_one_text = pick_one_text

        super().__init__()
        keyids = get_split_keyids(path=splitpath, split=split)

        features_data = {}
        texts_data = {}
        durations = {}

        if load_amass_data:
            with open(correspondance_path) as correspondance_path_file:
                kitml_correspondances = json.load(correspondance_path_file)

        if progress_bar:
            enumerator = enumerate(tqdm(keyids, f"Loading KIT {split}"))
        else:
            enumerator = enumerate(keyids)

        if tiny:
            maxdata = 2
        else:
            maxdata = np.inf

        datapath = Path(datapath)

        num_bad = 0
        if load_amass_data:
            bad_smpl = 0
            good_smpl = 0

        for i, keyid in enumerator:
            if len(features_data) >= maxdata:
                break

            anndata, success = load_annotation(keyid, datapath)
            if not success:
                logger.error(f"{keyid} has no annotations")
                continue

            # read smpl params
            if load_amass_data:
                smpl_data, success = load_amass_keyid(keyid, amass_path,
                                                      correspondances=kitml_correspondances)

                if not success:
                    bad_smpl += 1
                    continue
                else:
                    good_smpl += 1

                smpl_data, duration = downsample_amass(smpl_data, downsample=self.downsample, framerate=framerate)
                smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
            # read xyz joints in MMM format
            else:
                joints = load_mmm_keyid(keyid, datapath)
                joints, duration = downsample_mmm(joints, downsample=self.downsample, framerate=framerate)

            if split != "test" and not tiny:
                # Accept or not the sample, based on the duration
                if not self.sampler.accept(duration):
                    num_bad += 1
                    continue

            # Load rotation features (rfeats) data from AMASS
            if load_amass_data and load_with_rot:
                features = self.transforms.rots2rfeats(smpl_data)
            # Load xyz features (jfeats) data from AMASS
            elif load_amass_data and not load_with_rot:
                joints = self.transforms_smpl.rots2joints(smpl_data)
                features = self.transforms_xyz.joints2jfeats(joints)
            # Load xyz features (jfeats) data from MMM
            else:
                features = self.transforms.joints2jfeats(joints)

            features_data[keyid] = features
            texts_data[keyid] = anndata
            durations[keyid] = duration

        if load_amass_data and not tiny:
            percentage = 100 * bad_smpl / (bad_smpl + good_smpl)
            logger.info(f"There are {bad_smpl} sequences not found ({percentage:.4}%) in AMASS.")

        if split != "test" and not tiny:
            total = len(features_data)
            percentage = 100 * num_bad / (total+num_bad)
            logger.info(f"There are {num_bad} sequences rejected by the sampler ({percentage:.4}%).")

        self.features_data = features_data
        self.texts_data = texts_data

        self.keyids = list(features_data.keys())
        self._split_index = list(self.keyids)
        self._num_frames_in_sequence = durations
        self.nfeats = len(self[0]["datastruct"].features[0])

    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.features_data[keyid]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        if not self.pick_one_text:
            return sequences
        n = len(sequences)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            # Only the first one in evaluation
            index = 0
        text = sequences[index]
        return text

    def load_keyid(self, keyid):
        num_frames = self._num_frames_in_sequence[keyid]
        frame_ix = self.sampler(num_frames)

        datastruct = self._load_datastruct(keyid, frame_ix)
        text = self._load_text(keyid)
        element = {"datastruct": datastruct, "text": text,
                   "length": len(datastruct), "keyid": keyid}
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"


def load_annotation(keyid, datapath):
    metapath = datapath / (keyid + "_meta.json")
    metadata = json.load(metapath.open())

    if metadata["nb_annotations"] == 0:
        logger.error(f"{keyid} has no annotations")
        return None, False

    annpath = datapath / (keyid + "_annotations.json")
    anndata = json.load(annpath.open())
    assert len(anndata) == metadata["nb_annotations"]
    return anndata, True


def load_mmm_keyid(keyid, datapath):
    xyzpath = datapath / (keyid + "_fke.csv")
    xyzdata = pandas.read_csv(xyzpath, index_col=0)
    joints = np.array(xyzdata).reshape(-1, 21, 3)
    return joints

def downsample_mmm(joints, *, downsample, framerate):
    nframes_total = len(joints)
    last_framerate = 100

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)
    joints = torch.from_numpy(joints[frames]).float()
    return joints, duration

def load_amass_keyid(keyid, amass_path, *, correspondances):
    identifier = correspondances[keyid]["identifier"]
    smpl_keyid_path = correspondances[keyid]["path"]

    if identifier == "kit":
        smpl_datapath = Path(amass_path) / "KIT" / "KIT" / smpl_keyid_path
    elif identifier == "cmu":
        smpl_datapath = Path(amass_path) / "CMU" / "CMU" / smpl_keyid_path

        if not os.path.exists(smpl_datapath):
            # try with EKUT folder instead
            smpl_datapath = Path(amass_path) / "EKUT" / "EKUT" / smpl_keyid_path

            # File not found
            if not os.path.exists(smpl_datapath):
                return None, False
    else:
        raise TypeError(f"{identifier} identifier not recognized.")
    try:
        smpl_data = np.load(smpl_datapath)
    except FileNotFoundError:
        return None, False

    smpl_data = {x: smpl_data[x] for x in smpl_data.files}
    return smpl_data, True

def downsample_amass(smpl_data, *, downsample, framerate):
    nframes_total = len(smpl_data["poses"])
    last_framerate = smpl_data["mocap_framerate"].item()

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)

    # subsample
    smpl_data = {"poses": torch.from_numpy(smpl_data["poses"][frames]).float(),
                 "trans": torch.from_numpy(smpl_data["trans"][frames]).float()}
    return smpl_data, duration
