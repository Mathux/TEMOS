import bpy
# from temos.data import mmm_kinematic_tree as kinematic_tree, mmm_joints
# need to fix import first
import math
import numpy as np
from .materials import colored_material_diffuse_BSDF as colored_material


kinematic_tree = [[0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10],
                  [0, 11, 12, 13, 14, 15],
                  [0, 16, 17, 18, 19, 20]]

mmm_joints = ["root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
              "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"]

# Get the indexes of particular body part
# Feet
LM, RM = mmm_joints.index("LMrot"), mmm_joints.index("RMrot")
LF, RF = mmm_joints.index("LF"), mmm_joints.index("RF")

# Shoulders
LS, RS = mmm_joints.index("LS"), mmm_joints.index("RS")
# Hips
LH, RH = mmm_joints.index("LH"), mmm_joints.index("RH")


JOINTS_MATS = [colored_material(0.0, 0.0, 0.0),
               colored_material(0.6500, 0.175, 0.0043),
               colored_material(0.4500, 0.0357, 0.0349),
               colored_material(0.018, 0.059, 0.600),
               colored_material(0.032, 0.325, 0.521)]


class Joints:
    def __init__(self, data, *, mode, canonicalize, always_on_floor, **kwargs):
        data = prepare_joints(data, canonicalize=canonicalize, always_on_floor=always_on_floor)

        self.data = data
        self.mode = mode

        self.N = len(data)

        self.N = len(data)
        self.trajectory = data[:, 0, [0, 1]]

        self.mat = JOINTS_MATS

    def get_sequence_mat(self, frac):
        return self.mat

    def get_root(self, index):
        return self.data[index][0]

    def get_mean_root(self):
        return self.data[:, 0].mean(0)

    def load_in_blender(self, index, mat):
        skeleton = self.data[index]
        for lst, mat in zip(kinematic_tree, mat):
            for j1, j2 in zip(lst[:-1], lst[1:]):
                cylinder_between(skeleton[j1], skeleton[j2], 0.040, mat)

        return "Cylinder"

    def __len__(self):
        return self.N


def softmax(x, softness=1.0, dim=None):
    maxi, mini = x.max(dim), x.min(dim)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, softness=1.0, dim=0):
    return -softmax(-x, softness=softness, dim=dim)


def get_forward_direction(poses):
    across = poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] - poses[..., LS, :]
    forward = np.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = forward/np.linalg.norm(forward, axis=-1)
    return forward


def cylinder_between(t1, t2, r, mat):
    x1, y1, z1 = t1
    x2, y2, z2 = t2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(
        radius = r,
        depth = dist,
        location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)
    )

    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat


def matrix_of_angles(cos, sin, inv=False):
    sin = -sin if inv else sin
    return np.stack((np.stack((cos, -sin), axis=-1),
                     np.stack((sin, cos), axis=-1)), axis=-2)


def get_floor(poses, jointstype="mmm"):
    assert jointstype == "mmm"
    ndim = len(poses.shape)

    foot_heights = poses[..., (LM, LF, RM, RF), 1].min(-1)
    floor_height = softmin(foot_heights, softness=0.5, dim=-1)
    return floor_height[tuple((ndim - 2) * [None])].T


def canonicalize_joints(joints):
    poses = joints.copy()

    translation = joints[..., 0, :].copy()

    # Let the root have the Y translation
    translation[..., 1] = 0
    # Trajectory => Translation without gravity axis (Y)
    trajectory = translation[..., [0, 2]]

    # Remove the floor
    poses[..., 1] -= get_floor(poses)

    # Remove the trajectory of the joints
    poses[..., [0, 2]] -= trajectory[..., None, :]

    # Let the first pose be in the center
    trajectory = trajectory - trajectory[..., 0, :]

    # Compute the forward direction of the first frame
    forward = get_forward_direction(poses[..., 0, :, :])

    # Construct the inverse rotation matrix
    sin, cos = forward[..., 0], forward[..., 1]
    rotations_inv = matrix_of_angles(cos, sin, inv=True)

    # Rotate the trajectory
    trajectory_rotated = np.einsum("...j,...jk->...k", trajectory, rotations_inv)

    # Rotate the poses
    poses_rotated = np.einsum("...lj,...jk->...lk", poses[..., [0, 2]], rotations_inv)
    poses_rotated = np.stack((poses_rotated[..., 0], poses[..., 1], poses_rotated[..., 1]), axis=-1)

    # Re-merge the pose and translation
    poses_rotated[..., (0, 2)] += trajectory_rotated[..., None, :]
    return poses_rotated


def prepare_joints(joints, canonicalize=True, always_on_floor=False):
    # All face the same direction for the first frame
    if canonicalize:
        data = canonicalize_joints(joints)
    else:
        data = joints

    # Rescaling, shift axis and swap left/right
    data = data * 0.75 / 480

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # Make left/right correct
    data[..., [1]] = -data[..., [1]]

    # Center the first root to the first frame
    data -= data[[0], [0], :]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data
