import bpy


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material


def plane_mat():
    materials = bpy.data.materials
    material = materials.new(name="plane")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 1024
    checker.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1)
    checker.inputs["Color2"].default_value = (0.3, 0.3, 0.3, 1)
    links.new(checker.outputs["Color"], diffuse.inputs['Color'])
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    diffuse.inputs["Roughness"].default_value = 0.127451
    return material


def plane_mat_uni():
    materials = bpy.data.materials
    material = materials.new(name="plane_uni")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    diffuse.inputs["Roughness"].default_value = 0.127451
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material
