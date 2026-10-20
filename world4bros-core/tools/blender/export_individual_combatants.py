import os
import sys
import math
import bpy
import bmesh
from mathutils import Vector, Euler, Matrix

# Add blender dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_characters_and_zombie import (
    create_character_palette,
    build_character_survivor,
    build_character_scavenger,
    build_character_zombie,
)

def export_model(build_func, out_glb_path, model_name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    palette = create_character_palette()
    mat_keys = list(palette.keys())
    mat_list = [palette[k] for k in mat_keys]
    p_map = {k: i for i, k in enumerate(mat_keys)}

    mesh = bpy.data.meshes.new(f"{model_name}_Mesh")
    bm = bmesh.new()

    build_func(bm, p_map, origin=(0, 0, 0))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(model_name, mesh)
    bpy.context.collection.objects.link(obj)

    for m in mat_list:
        obj.data.materials.append(m)

    bpy.ops.export_scene.gltf(filepath=out_glb_path, export_format='GLB', use_selection=False)
    print(f"[EXPORT_SUCCESS] {model_name} -> {out_glb_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zombie_glb = os.path.join(script_dir, "zombie.glb")
    bot_jax_glb = os.path.join(script_dir, "bot_jax.glb")
    bot_rook_glb = os.path.join(script_dir, "bot_rook.glb")

    export_model(build_character_zombie, zombie_glb, "Zombie_Walker")
    export_model(build_character_survivor, bot_jax_glb, "Bot_Jax")
    export_model(build_character_scavenger, bot_rook_glb, "Bot_Rook")

if __name__ == "__main__":
    main()
