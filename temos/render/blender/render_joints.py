import bpy
import os
import sys

from .tools import delete_objs
from .joints import get_forward_direction, cylinder_between, kinematic_tree, prepare_joints
from .floor import show_traj, plot_floor
from .materials import colored_material
from .scene import setup_scene  # noqa


def render_joints(data, frames_folder, gt=False, exact_frame=None, canonicalize=True, fast=True, high_res=False, init=True, render_img=True):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(fast=fast, high_res=high_res)

    camera = get_initialized_camera()

    # Put everything in this folder
    if not render_img:
        os.makedirs(frames_folder, exist_ok=True)
    else:
        img_path = frames_folder

    data = prepare_joints(data, canonicalize=canonicalize)

    # Show the trajectory
    trajectory = data[:, 0, [0, 1]]
    show_traj(trajectory)

    # Create a floor
    plot_floor(data)

    # create_sphere("sph", mat)
    last_skel_root = data[0, 0]
    nframes = len(data)

    skeletons = data

    if render_img and exact_frame is None:
        tokeep = 8
        step = len(data)//tokeep
        roots = data[:, 0]
        dx, dy, dz = roots.mean(0)
        # camera.data.lens = 70
        camera.data.lens = 85
        camera.location.x += dx
        camera.location.y += dy
    elif render_img and exact_frame is not None:
        camera.data.lens = 103

        roots = data[:, 0]
        dx, dy, dz = roots.mean(0)
        camera.location.x += dx
        camera.location.y += dy

        index_frame = int(exact_frame*len(data))

        # doesnt matter
        step = 1
        img_name, ext = os.path.splitext(img_path)
        img_path = f"{img_name}_{exact_frame}{ext}"

    for frame, skeleton in enumerate(skeletons):
        islast = frame == (len(skeletons)-1)

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

        for lst, mat in zip(kinematic_tree, mats):
            for j1, j2 in zip(lst[:-1], lst[1:]):
                cylinder_between(skeleton[j1], skeleton[j2], 0.040, mat)





        if not render_img:
            bpy.context.scene.render.filepath = os.path.join(frames_folder, f"frame_{framestr}.png")
            bpy.ops.render.render(use_viewport=True, write_still=True)
            delete_objs("Cylinder")
        else:
            if islast or exact_frame is not None:
                bpy.context.scene.render.filepath = img_path
                bpy.ops.render.render(use_viewport=True, write_still=True)
                print("the real output is ")
                print(img_path)

    delete_objs(["Plane", "myCurve", "Cylinder"])
