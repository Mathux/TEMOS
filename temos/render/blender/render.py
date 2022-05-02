import bpy
import os
import sys
import numpy as np
import math

from .materials import colored_material
from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs
from .camera import Camera


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)



def render(npydata, frames_folder, *, mode, faces_path,
           gt=False,
           exact_frame=None,
           num=8,
           downsample=True,
           canonicalize=True,
           always_on_floor=False,
           cycle=True, high_res=True,
           init=True):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(cycle=cycle, high_res=high_res)

    from .tools import mesh_detect
    is_mesh = mesh_detect(npydata)

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_onfloor"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_onfloor"
        img_path = f"{img_name}{ext}"
    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_onfloor"
        img_path = f"{img_name}_{exact_frame}{ext}"

    # remove X% of begining and end
    # as it is almost always static
    # in this part
    if mode == "sequence":
        perc = 0.2
        npydata = prune_begin_end(npydata, perc)

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata, gt=gt, mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
    else:
        from .joints import Joints
        data = Joints(npydata, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor
    plot_floor(data.data)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    from .sampler import get_frameidx
    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render-1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render-1)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"
        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.append(objname)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)

    # bpy.ops.wm.save_as_mainfile(filepath="/Users/mathis/temos/code/all_vertices.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path
