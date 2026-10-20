import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# ==============================================================================
# 1. ARENA MATERIALS (Matching Locked Diorama Palette)
# ==============================================================================
def get_or_create_material(name, base_color, roughness=0.5, metallic=0.0, emission_color=(0, 0, 0, 1), emission_strength=0.0):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = base_color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = emission_color
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = emission_color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat

def create_arena_palette():
    return {
        "ground_dirt": get_or_create_material("Mat_Ground_Dirt", (0.16, 0.17, 0.18, 1.0), roughness=0.85),
        "ground_markings": get_or_create_material("Mat_Ground_Markings", (0.85, 0.72, 0.28, 1.0), roughness=0.6),
        "concrete_barrier": get_or_create_material("Mat_Concrete_Barrier", (0.55, 0.56, 0.58, 1.0), roughness=0.7),
        "wood_crate": get_or_create_material("Mat_Wood_Crate", (0.42, 0.28, 0.16, 1.0), roughness=0.6),
        "sandbag_burlap": get_or_create_material("Mat_Sandbag_Burlap", (0.64, 0.54, 0.38, 1.0), roughness=0.9),
        "metal_dark": get_or_create_material("Mat_Metal_Dark", (0.12, 0.13, 0.15, 1.0), roughness=0.35, metallic=0.85),
        "hazard_red": get_or_create_material("Mat_Hazard_Red", (0.78, 0.18, 0.14, 1.0), roughness=0.4),
        "hazard_yellow": get_or_create_material("Mat_Hazard_Yellow", (0.95, 0.82, 0.20, 1.0), roughness=0.3, emission_color=(1.0, 0.85, 0.2, 1.0), emission_strength=2.0),
        "tree_leaves": get_or_create_material("Mat_Tree_Leaves", (0.12, 0.26, 0.18, 1.0), roughness=0.7),
        "tree_trunk": get_or_create_material("Mat_Tree_Trunk", (0.28, 0.18, 0.12, 1.0), roughness=0.8),
        "floodlight_glow": get_or_create_material("Mat_Floodlight_Glow", (1.0, 0.98, 0.90, 1.0), roughness=0.1, emission_color=(1.0, 0.95, 0.8, 1.0), emission_strength=10.0),
        "spawn_teal": get_or_create_material("Mat_Spawn_Teal", (0.15, 0.55, 0.65, 1.0), roughness=0.3, emission_color=(0.2, 0.6, 0.7, 1.0), emission_strength=3.0),
        # Character materials
        "skin_hero": get_or_create_material("Mat_Skin_Hero", (0.88, 0.68, 0.52, 1.0), roughness=0.6),
        "skin_zombie": get_or_create_material("Mat_Skin_Zombie", (0.42, 0.52, 0.38, 1.0), roughness=0.7),
        "jacket_teal": get_or_create_material("Mat_Jacket_Teal", (0.14, 0.35, 0.42, 1.0), roughness=0.5),
        "vest_ochre": get_or_create_material("Mat_Vest_Ochre", (0.78, 0.48, 0.20, 1.0), roughness=0.6),
        "pants_charcoal": get_or_create_material("Mat_Pants_Charcoal", (0.15, 0.16, 0.18, 1.0), roughness=0.6),
        "clothes_tattered": get_or_create_material("Mat_Clothes_Tattered", (0.22, 0.20, 0.22, 1.0), roughness=0.8),
        "glow_eyes": get_or_create_material("Mat_Glow_Eyes", (0.95, 0.85, 0.3, 1.0), roughness=0.1, emission_color=(1.0, 0.85, 0.2, 1.0), emission_strength=5.0),
        "glow_flashlight": get_or_create_material("Mat_Glow_Flashlight", (1.0, 0.98, 0.8, 1.0), roughness=0.1, emission_color=(1.0, 0.95, 0.7, 1.0), emission_strength=8.0),
    }

# ==============================================================================
# 2. PRIMITIVE UTILITIES
# ==============================================================================
def add_box(bm, center, size, mat_index=0, rot_x=0.0, rot_y=0.0, rot_z=0.0):
    sx, sy, sz = size
    mat = (Matrix.Translation(center) @
           Matrix.Rotation(rot_z, 4, 'Z') @
           Matrix.Rotation(rot_y, 4, 'Y') @
           Matrix.Rotation(rot_x, 4, 'X') @
           Matrix.Diagonal((sx, sy, sz, 1.0)))
    f_before = set(bm.faces)
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    for face in bm.faces:
        if face not in f_before:
            face.material_index = mat_index

def add_cylinder(bm, center, radius, depth, mat_index=0, segments=12, rot_x=0.0, rot_y=0.0, rot_z=0.0):
    mat = (Matrix.Translation(center) @
           Matrix.Rotation(rot_z, 4, 'Z') @
           Matrix.Rotation(rot_y, 4, 'Y') @
           Matrix.Rotation(rot_x, 4, 'X'))
    f_before = set(bm.faces)
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
    for face in bm.faces:
        if face not in f_before:
            face.material_index = mat_index

# ==============================================================================
# 3. ARENA OBSTACLE BUILDERS (1:1 with src/sim/arena.ts)
# ==============================================================================
def place_crate(bm, center, size_xy, height, p_map):
    cx, cy, cz = center
    w, d = size_xy
    add_box(bm, Vector((cx, cy, cz + height / 2.0)), (w, d, height), mat_index=p_map["wood_crate"])
    # Metal corner straps
    add_box(bm, Vector((cx, cy, cz + height / 2.0)), (w + 0.02, 0.14, height + 0.02), mat_index=p_map["metal_dark"])
    add_box(bm, Vector((cx, cy, cz + height / 2.0)), (0.14, d + 0.02, height + 0.02), mat_index=p_map["metal_dark"])

def place_sandbags(bm, start_xy, length, direction, p_map):
    # Stacked burlap sandbags along length (height ~0.9m)
    sx, sy = start_xy
    bags = int(length / 0.7)
    for i in range(bags):
        px = sx + (i + 0.5) * 0.7 if direction == 'h' else sx
        py = sy if direction == 'h' else sy + (i + 0.5) * 0.7
        # Bottom row (2 bags)
        add_box(bm, Vector((px, py, 0.20)), (0.65, 0.40, 0.38), mat_index=p_map["sandbag_burlap"])
        # Top row (1 bag centered)
        add_box(bm, Vector((px, py, 0.55)), (0.65, 0.35, 0.34), mat_index=p_map["sandbag_burlap"])

def place_low_wall(bm, center, size, p_map):
    cx, cy, cz = center
    w, d, h = size
    # Heavy beveled concrete road barrier
    add_box(bm, Vector((cx, cy, cz + h / 2.0)), (w, d, h), mat_index=p_map["concrete_barrier"])
    add_box(bm, Vector((cx, cy, cz + h)), (w + 0.04, d + 0.04, 0.08), mat_index=p_map["concrete_barrier"])

def place_barrel(bm, xy, p_map):
    x, y = xy
    # Flammable hazard barrel
    add_cylinder(bm, Vector((x, y, 0.55)), radius=0.34, depth=1.1, mat_index=p_map["hazard_red"])
    add_cylinder(bm, Vector((x, y, 0.55)), radius=0.35, depth=0.22, mat_index=p_map["hazard_yellow"])
    add_cylinder(bm, Vector((x, y, 1.05)), radius=0.36, depth=0.06, mat_index=p_map["metal_dark"])
    add_cylinder(bm, Vector((x, y, 0.05)), radius=0.36, depth=0.06, mat_index=p_map["metal_dark"])

def place_tree(bm, xy, p_map):
    x, y = xy
    # Stylized low-poly diorama pine tree
    # Trunk
    add_cylinder(bm, Vector((x, y, 1.2)), radius=0.22, depth=2.4, mat_index=p_map["tree_trunk"])
    # 3 conical tiered foliage tiers
    add_cylinder(bm, Vector((x, y, 2.5)), radius=1.6, depth=1.4, mat_index=p_map["tree_leaves"])
    add_cylinder(bm, Vector((x, y, 3.6)), radius=1.2, depth=1.2, mat_index=p_map["tree_leaves"])
    add_cylinder(bm, Vector((x, y, 4.6)), radius=0.7, depth=1.0, mat_index=p_map["tree_leaves"])

def place_floodlight_tower(bm, xy, p_map):
    x, y = xy
    # High-intensity industrial floodlight pole (height 6.5m)
    add_cylinder(bm, Vector((x, y, 3.2)), radius=0.10, depth=6.4, mat_index=p_map["metal_dark"])
    # Crossbar bracket
    add_box(bm, Vector((x, y, 6.4)), (1.4, 0.20, 0.12), mat_index=p_map["metal_dark"])
    # Twin floodlight lamps
    add_box(bm, Vector((x - 0.45, y, 6.5)), (0.35, 0.28, 0.24), mat_index=p_map["metal_dark"])
    add_box(bm, Vector((x + 0.45, y, 6.5)), (0.35, 0.28, 0.24), mat_index=p_map["metal_dark"])
    # Glowing lens faces
    add_box(bm, Vector((x - 0.45, y - 0.12, 6.5)), (0.30, 0.04, 0.20), mat_index=p_map["floodlight_glow"])
    add_box(bm, Vector((x + 0.45, y - 0.12, 6.5)), (0.30, 0.04, 0.20), mat_index=p_map["floodlight_glow"])

# ==============================================================================
# 4. DUEL PIT ARENA GENERATOR (40m x 24m)
# ==============================================================================
def build_duel_pit_arena_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    palette = create_arena_palette()

    mat_keys = list(palette.keys())
    mat_list = [palette[k] for k in mat_keys]
    p_map = {k: i for i, k in enumerate(mat_keys)}

    mesh = bpy.data.meshes.new("DuelPit_Arena_Mesh")
    bm = bmesh.new()

    # Arena dimensions: 40m x 24m (sim coordinates: 0..40, 0..24, mapped with origin at center: -20..+20, -12..+12)
    # Ground Slab
    add_box(bm, Vector((0, 0, -0.15)), (42.0, 26.0, 0.30), mat_index=p_map["ground_dirt"])

    # Center circle markings (combat ring)
    add_cylinder(bm, Vector((0, 0, 0.005)), radius=4.0, depth=0.01, mat_index=p_map["ground_markings"], segments=24)
    add_cylinder(bm, Vector((0, 0, 0.010)), radius=3.8, depth=0.01, mat_index=p_map["ground_dirt"], segments=24)

    # Perimeter Wall / Fence (Around 40m x 24m)
    # North / South walls
    add_box(bm, Vector((0, -12.0, 1.0)), (40.2, 0.35, 2.0), mat_index=p_map["concrete_barrier"])
    add_box(bm, Vector((0,  12.0, 1.0)), (40.2, 0.35, 2.0), mat_index=p_map["concrete_barrier"])
    # East / West walls
    add_box(bm, Vector((-20.0, 0, 1.0)), (0.35, 24.2, 2.0), mat_index=p_map["concrete_barrier"])
    add_box(bm, Vector(( 20.0, 0, 1.0)), (0.35, 24.2, 2.0), mat_index=p_map["concrete_barrier"])

    # 4 Corner Floodlight Towers
    place_floodlight_tower(bm, (-19.0, -11.0), p_map)
    place_floodlight_tower(bm, ( 19.0, -11.0), p_map)
    place_floodlight_tower(bm, (-19.0,  11.0), p_map)
    place_floodlight_tower(bm, ( 19.0,  11.0), p_map)

    # 4 Corner Trees (from sim: (3,3), (37,21), (3,21), (37,3))
    # Mapped: x - 20, y - 12
    place_tree(bm, (-17.0, -9.0), p_map)
    place_tree(bm, ( 17.0,  9.0), p_map)
    place_tree(bm, (-17.0,  9.0), p_map)
    place_tree(bm, ( 17.0, -9.0), p_map)

    # Spawns: (4, 12) -> (-16, 0), (36, 12) -> (+16, 0)
    add_cylinder(bm, Vector((-16.0, 0, 0.01)), radius=1.2, depth=0.02, mat_index=p_map["spawn_teal"])
    add_cylinder(bm, Vector(( 16.0, 0, 0.01)), radius=1.2, depth=0.02, mat_index=p_map["spawn_teal"])

    # Crates from sim:
    # { x: 17, y: 10, w: 2, h: 2 } -> center: (-2, -1)
    place_crate(bm, (-2.0, -1.0, 0), (2.0, 2.0), 1.5, p_map)
    # { x: 21, y: 12, w: 2, h: 2 } -> center: (+2, +1)
    place_crate(bm, ( 2.0,  1.0, 0), (2.0, 2.0), 1.5, p_map)
    # { x: 19, y: 5, w: 2, h: 1 } -> center: (0, -6.5)
    place_crate(bm, ( 0.0, -6.5, 0), (2.0, 1.0), 1.2, p_map)
    # { x: 19, y: 18, w: 2, h: 1 } -> center: (0, +6.5)
    place_crate(bm, ( 0.0,  6.5, 0), (2.0, 1.0), 1.2, p_map)

    # Sandbags from sim (4 banks of 5m horizontal):
    # (11, 6) -> X: -9 to -4, Y: -6
    place_sandbags(bm, (-9.0, -6.0), 5.0, 'h', p_map)
    # (24, 17) -> X: +4 to +9, Y: +5
    place_sandbags(bm, ( 4.0,  5.0), 5.0, 'h', p_map)
    # (11, 17) -> X: -9 to -4, Y: +5
    place_sandbags(bm, (-9.0,  5.0), 5.0, 'h', p_map)
    # (24, 6) -> X: +4 to +9, Y: -6
    place_sandbags(bm, ( 4.0, -6.0), 5.0, 'h', p_map)

    # Low Walls from sim (w: 1, h: 3 in sim means 1m thick, 3m long along Y):
    # (26, 9) -> X: +6.5, Y: -1.5
    place_low_wall(bm, ( 6.5, -1.5, 0), (0.8, 3.0, 1.0), p_map)
    # (13, 12) -> X: -6.5, Y: +1.5
    place_low_wall(bm, (-6.5,  1.5, 0), (0.8, 3.0, 1.0), p_map)

    # Barrels from sim:
    # (9, 12) -> (-11, 0)
    place_barrel(bm, (-11.0, 0.0), p_map)
    # (31, 12) -> (+11, 0)
    place_barrel(bm, ( 11.0, 0.0), p_map)
    # (16, 3) -> (-4, -9)
    place_barrel(bm, (-4.0, -9.0), p_map)
    # (24, 21) -> (+4, +9)
    place_barrel(bm, ( 4.0,  9.0), p_map)

    # Characters in Combat:
    # Jax at Spawn 1 (-16, 0)
    add_box(bm, Vector((-16.0, 0, 0.9)), (0.5, 0.3, 1.8), mat_index=p_map["jacket_teal"])
    add_cylinder(bm, Vector((-15.7, 0, 1.2)), radius=0.06, depth=0.2, mat_index=p_map["glow_flashlight"], rot_y=math.radians(90))
    # Rook behind low wall (6.5, -1.5)
    add_box(bm, Vector((6.5, -1.5, 0.7)), (0.5, 0.3, 1.4), mat_index=p_map["vest_ochre"])
    # 2 Zombies in the center pit advancing!
    add_box(bm, Vector((-1.0, 1.0, 0.8)), (0.45, 0.3, 1.6), mat_index=p_map["skin_zombie"])
    add_box(bm, Vector(( 1.0, -1.0, 0.8)), (0.45, 0.3, 1.6), mat_index=p_map["skin_zombie"])

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("DuelPit_Arena_Object", mesh)
    bpy.context.collection.objects.link(obj)

    for m in mat_list:
        obj.data.materials.append(m)

    # Camera Setup: Classic Isometric Arena Overview
    cam_target = bpy.data.objects.new("ArenaCamTarget", None)
    cam_target.location = (0, 0, 1.5)
    bpy.context.collection.objects.link(cam_target)

    cam_data = bpy.data.cameras.new("ArenaCamera")
    cam_data.type = 'PERSP'
    cam_data.lens = 38.0
    cam_obj = bpy.data.objects.new("ArenaCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # High Isometric Angle framing the entire 40m x 24m arena
    cam_obj.location = (28.0, -32.0, 24.0)
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = cam_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam_obj

    # Night Arena Lighting
    # Moonlight Sun (Deep Cool Blue Ambience)
    moon = bpy.data.lights.new("Moonlight", type='SUN')
    moon.energy = 1.2
    moon.color = (0.35, 0.55, 0.85)
    moon_obj = bpy.data.objects.new("Moonlight", moon)
    moon_obj.rotation_euler = (math.radians(55), math.radians(10), math.radians(-40))
    bpy.context.collection.objects.link(moon_obj)

    # 4 Corner Floodlights (Warm Crossing Beams)
    for fx, fy in [(-19.0, -11.0), (19.0, -11.0), (-19.0, 11.0), (19.0, 11.0)]:
        f_light = bpy.data.lights.new(f"FloodLight_{fx}_{fy}", type='SPOT')
        f_light.energy = 850.0
        f_light.spot_size = math.radians(65)
        f_light.spot_blend = 0.3
        f_light.color = (1.0, 0.94, 0.82)
        f_obj = bpy.data.objects.new(f"Flood_{fx}_{fy}", f_light)
        f_obj.location = (fx, fy, 6.4)
        # Aim toward center
        dx, dy, dz = -fx, -fy, -6.4
        rot_y = math.atan2(dx, dz)
        f_obj.rotation_euler = (math.radians(35) if fy < 0 else math.radians(-35), rot_y * 0.2, 0)
        bpy.context.collection.objects.link(f_obj)

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 768
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.join(script_dir, "duel_pit_arena_preview.png")
    out_blend = os.path.join(script_dir, "duel_pit_arena.blend")
    out_glb = os.path.join(script_dir, "duel_pit_arena.glb")

    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB', use_selection=False)

    print(f"ARENA_RENDER_SUCCESS: Image={out_img}, Blend={out_blend}, GLB={out_glb}")

if __name__ == "__main__":
    build_duel_pit_arena_scene()
