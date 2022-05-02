import bpy
import os
import sys
import numpy as np
import math

from .materials import colored_material
from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs
from .tools import get_initialized_camera


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    return data[to_remove:-to_remove]


def render_vertices(data, frames_folder, gt=False, exact_frame=None,
                    frames_to_keep=8,
                    canonicalize=True, fast=True, high_res=False, init=True, render_img=True):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(fast=fast, high_res=high_res)

    camera = get_initialized_camera()

    # Put everything in this folder
    if not render_img:
        os.makedirs(frames_folder, exist_ok=True)
    else:
        img_path = frames_folder

    # data = prepare_vertices(data, canonicalize=canonicalize)
    # if render sequence
    # remove X% of begining and end
    # as it is almost always static
    if render_img and exact_frame is None:
        perc = 0.0
        data = prune_begin_end(data, perc)

    # Show the trajectory
    trajectory = data[:, :, [0, 1]].mean(1)
    show_traj(trajectory)

    # Create a floor
    plot_floor(data)

    if gt:
        # green
        mat = colored_material(0.009, 0.214, 0.029)
    else:
        # blue
        mat = colored_material(0.022, 0.129, 0.439)

    # create_sphere("sph", mat)
    last_skel_root = data[0, 0]
    nframes = len(data)

    skeletons = data

    # center the camera to the middle
    if render_img:
        dx, dy, dz = roots.mean(0)
        camera.location.x += dx
        camera.location.y += dy


    if render_img and exact_frame is None:
        camera.data.lens = 85
        tokeep = frames_to_keep
        step = len(data)//tokeep
        roots = data[:, 0]

    elif render_img and exact_frame is not None:
        camera.data.lens = 103

        index_frame = int(exact_frame*len(data))

        # doesnt matter
        step = 1
        img_name, ext = os.path.splitext(img_path)
        img_path = f"{img_name}_{exact_frame}{ext}"

    N = len(skeletons)

    # from skimage.color import rgb2hsv, hsv2rgb

    import matplotlib
    cmap = matplotlib.cm.get_cmap('Blues')
    mats = []
    for n in range(N):
        begining = 0.65
        new_rgbcolor = cmap(begining + (1-begining)*n/(N-1))

        # hsvcolor
        mats.append(colored_material(*new_rgbcolor))

    names = []
    for frame, skeleton in enumerate(skeletons):
        islast = frame == (len(skeletons)-1)
        mat = mats[frame]

        if not render_img:
            # update camera
            current_skel_root = skeleton[0]
            delta_root = current_skel_root - last_skel_root
            camera.location.x += delta_root[0]
            camera.location.y += delta_root[1]
            last_skel_root = current_skel_root
        else:
            if frame % step != 0 and not islast:
                continue
            if exact_frame is not None:
                if index_frame != frame:
                    continue

        print(f"Frame: {frame+1}/{nframes}")
        framestr = str(frame).zfill(4)

        name = f"{str(frame).zfill(4)}"
        names.append(name)

        load_numpy_vertices_into_blender(skeleton, name, mat)

        if not render_img:
            bpy.context.scene.render.filepath = os.path.join(frames_folder, f"frame_{framestr}.png")
            bpy.ops.render.render(use_viewport=True, write_still=True)
            delete_objs(name)
        else:
            if islast or exact_frame is not None:
                bpy.context.scene.render.filepath = img_path
                bpy.ops.render.render(use_viewport=True, write_still=True)
                print("the real output is ")
                print(img_path)


    # bpy.ops.wm.save_as_mainfile(filepath="/Users/mathis/temos/code/all_vertices.blend")
    # exit()

    if render_img:
        delete_objs(names)

    delete_objs(["Plane", "myCurve"])
