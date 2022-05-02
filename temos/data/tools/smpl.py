from typing import Optional
import temos.tools.geometry as geometry
import torch
from torch import Tensor
from temos.tools.easyconvert import matrix_to, axis_angle_to
from temos.transforms.smpl import RotTransDatastruct


def canonicalize_smplh(poses: Tensor, trans: Optional[Tensor] = None):
    bs, nframes, njoints = poses.shape[:3]

    global_orient = poses[:, :, 0]

    # first global rotations
    rot2d = geometry.matrix_to_axis_angle(global_orient[:, 0])
    rot2d[:, :2] = 0  # Remove the rotation along the vertical axis
    rot2d = geometry.axis_angle_to_matrix(rot2d)

    # Rotate the global rotation to eliminate Z rotations
    global_orient = torch.einsum("ikj,imkl->imjl", rot2d, global_orient)

    # Construct canonicalized version of x
    xc = torch.cat((global_orient[:, :, None], poses[:, :, 1:]), dim=2)

    if trans is not None:
        vel = trans[:, 1:] - trans[:, :-1]
        # Turn the translation as well
        vel = torch.einsum("ikj,ilk->ilj", rot2d, vel)
        trans = torch.cat((torch.zeros(bs, 1, 3, device=vel.device),
                           torch.cumsum(vel, 1)), 1)
        return xc, trans
    else:
        return xc


def smpl_data_to_matrix_and_trans(data, nohands=True):
    trans = data["trans"]
    nframes = len(trans)

    axis_angle_poses = data["poses"]
    axis_angle_poses = data["poses"].reshape(nframes, -1, 3)

    if nohands:
        axis_angle_poses = axis_angle_poses[:, :22]

    matrix_poses = axis_angle_to("matrix", axis_angle_poses)

    return RotTransDatastruct(rots=matrix_poses, trans=trans)
