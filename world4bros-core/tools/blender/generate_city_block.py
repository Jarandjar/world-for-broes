import bpy
import bmesh
import random
import math
import sys
import os
from mathutils import Vector, Matrix

# Ensure blender tool directory is in sys.path
script_dir = r"C:\Users\Jalle\OneDrive\Desktop\BUCKS\game\world4bros-core\tools\blender"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from generate_building import build_procedural_building

def add_box(bm, center, size):
    sx, sy, sz = size
    mat = Matrix.Translation(center) @ Matrix.Diagonal((sx, sy, sz, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)

def add_cylinder(bm, center, radius, depth, segments=8):
    mat = Matrix.Translation(center)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=segments,
        radius1=radius,
        radius2=radius,
        depth=depth,
        matrix=mat,
    )

def generate_tree(bm, pos):
    """Spawns a clean low-poly tree."""
    x, y, z = pos
    # Trunk
    trunk_h = 1.8
    add_cylinder(bm, Vector((x, y, z + trunk_h / 2.0)), radius=0.25, depth=trunk_h, segments=6)
    # Foliage (2 stacked low-poly spheres/cones)
    add_cylinder(bm, Vector((x, y, z + trunk_h + 1.0)), radius=1.3, depth=1.8, segments=7)
    add_cylinder(bm, Vector((x, y, z + trunk_h + 2.2)), radius=0.9, depth=1.4, segments=6)

def generate_street_lamp(bm, pos):
    """Spawns a modern low-poly street lamp."""
    x, y, z = pos
    pole_h = 4.2
    # Vertical pole
    add_cylinder(bm, Vector((x, y, z + pole_h / 2.0)), radius=0.1, depth=pole_h, segments=6)
    # Arm extending toward street
    add_box(bm, Vector((x + 0.6, y, z + pole_h)), (1.2, 0.15, 0.15))
    # Lamp head
    add_box(bm, Vector((x + 1.1, y, z + pole_h - 0.2)), (0.5, 0.35, 0.25))

def generate_city_block(seed=777):
    random.seed(seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    block_w = 46.0  # X size of the block
    block_d = 40.0  # Y size of the block
    road_w = 10.0   # Width of surrounding road
    sidewalk_w = 2.5
    curb_h = 0.2

    # --- 1. ENVIRONMENT / ROADS & SIDEWALKS ---
    env_mesh = bpy.data.meshes.new("CityGrid_Mesh")
    bm_env = bmesh.new()

    # Asphalt Base Plane
    total_w = block_w + road_w * 2.5
    total_d = block_d + road_w * 2.5
    add_box(bm_env, Vector((0, 0, -0.05)), (total_w, total_d, 0.1))

    # Raised Concrete Sidewalk Base
    add_box(bm_env, Vector((0, 0, curb_h / 2.0)), (block_w, block_d, curb_h))

    # Road Dashed Markings (X road & Y road)
    dash_len = 2.5
    dash_gap = 2.5
    dash_t = 0.25
    dash_h = 0.02

    # Bottom & Top Road centerlines
    for y_pos in [-(block_d / 2.0 + road_w / 2.0), (block_d / 2.0 + road_w / 2.0)]:
        x = -total_w / 2.0 + 2.0
        while x < total_w / 2.0 - 2.0:
            add_box(bm_env, Vector((x + dash_len / 2.0, y_pos, dash_h)), (dash_len, dash_t, dash_h))
            x += dash_len + dash_gap

    # Left & Right Road centerlines
    for x_pos in [-(block_w / 2.0 + road_w / 2.0), (block_w / 2.0 + road_w / 2.0)]:
        y = -total_d / 2.0 + 2.0
        while y < total_d / 2.0 - 2.0:
            add_box(bm_env, Vector((x_pos, y + dash_len / 2.0, dash_h)), (dash_t, dash_len, dash_h))
            y += dash_len + dash_gap

    # Trees along the sidewalks
    tree_x_positions = [-16.0, -8.0, 8.0, 16.0]
    for tx in tree_x_positions:
        generate_tree(bm_env, (tx, -(block_d / 2.0 - 1.2), curb_h))
        generate_tree(bm_env, (tx, (block_d / 2.0 - 1.2), curb_h))

    # Street Lamps on corners
    lamp_corners = [
        (-(block_w / 2.0 - 1.0), -(block_d / 2.0 - 1.0)),
        ((block_w / 2.0 - 1.0), -(block_d / 2.0 - 1.0)),
        (-(block_w / 2.0 - 1.0), (block_d / 2.0 - 1.0)),
        ((block_w / 2.0 - 1.0), (block_d / 2.0 - 1.0)),
    ]
    for lx, ly in lamp_corners:
        generate_street_lamp(bm_env, (lx, ly, curb_h))

    bm_env.to_mesh(env_mesh)
    bm_env.free()
    env_obj = bpy.data.objects.new("Environment_Streets", env_mesh)
    bpy.context.collection.objects.link(env_obj)

    # --- 2. GENERATE BUILDINGS ON THE BLOCK ---
    # Define 6 building plots (2 rows of 3 buildings)
    plots = [
        {"x": -13.0, "y": -9.0, "w": 10.0, "d": 12.0, "floors": 9, "setback": True},
        {"x": 0.0,   "y": -9.0, "w": 11.0, "d": 12.0, "floors": 5, "setback": False},
        {"x": 13.0,  "y": -9.0, "w": 10.0, "d": 12.0, "floors": 12, "setback": True},
        {"x": -13.0, "y": 9.0,  "w": 10.0, "d": 12.0, "floors": 6, "setback": False},
        {"x": 0.0,   "y": 9.0,  "w": 11.0, "d": 12.0, "floors": 14, "setback": True},
        {"x": 13.0,  "y": 9.0,  "w": 10.0, "d": 12.0, "floors": 4, "setback": False},
    ]

    from types import SimpleNamespace
    for idx, p in enumerate(plots):
        plot_props = SimpleNamespace(
            seed=seed + idx * 37,
            floors=p["floors"],
            floor_height=2.8,
            width=p["w"],
            depth=p["d"],
            enable_setback=p["setback"],
            setback_inset=1.4,
            generate_roof_clutter=True,
        )

        b_obj = build_procedural_building(plot_props)
        b_obj.location = (p["x"], p["y"], curb_h)
        b_obj.name = f"Building_Plot_{idx + 1}"

    # --- 3. ISOMETRIC CAMERA & SUN LIGHTING ---
    cam_data = bpy.data.cameras.new("CityIsoCamera")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 68.0
    cam_obj = bpy.data.objects.new("CityIsoCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Isometric position & angle
    cam_obj.location = (60.0, -60.0, 55.0)
    cam_obj.rotation_euler = (math.radians(58), math.radians(0), math.radians(45))
    bpy.context.scene.camera = cam_obj

    # Directional Sun Light
    light_data = bpy.data.lights.new("Sun", type='SUN')
    light_data.energy = 3.8
    light_obj = bpy.data.objects.new("SunLight", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(50), math.radians(25), math.radians(-35))

    # --- 4. RENDER & SAVE ---
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'MATCAP'
    scene.display.shading.color_type = 'SINGLE'

    # Save .blend file
    out_blend = os.path.join(script_dir, "city_block_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    # Render image preview
    out_img = os.path.join(script_dir, "city_block_preview.png")
    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    print(f"CITY_BLOCK_SUCCESS: Blend saved to {out_blend}, Preview rendered to {out_img}")

if __name__ == "__main__":
    generate_city_block(seed=42)
