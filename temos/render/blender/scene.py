import bpy
from .materials import plane_mat  # noqa


def setup_cycles(cycle=True):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

    if cycle:
        bpy.context.scene.cycles.use_denoising = True

    bpy.context.scene.render.tile_x = 256
    bpy.context.scene.render.tile_y = 256
    bpy.context.scene.cycles.samples = 64
    # bpy.context.scene.cycles.denoiser = 'OPTIX'


# Setup scene
def setup_scene(cycle=True, high_res=True):
    scene = bpy.data.scenes['Scene']
    if high_res:
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    else:
        scene.render.resolution_x = 1280//2
        scene.render.resolution_y = 1024//2

    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.object.light_add(type='SUN', align='WORLD',
                             location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = 1.5

    # rotate camera
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.transform.resize(value=(10, 10, 10), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.select_all(action='DESELECT')

    setup_cycles(cycle=cycle)
    return scene
