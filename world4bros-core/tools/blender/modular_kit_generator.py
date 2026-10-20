import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# ==============================================================================
# 1. SHARED MATERIAL SYSTEM
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

def create_palette():
    return {
        "siding_teal": get_or_create_material("Mat_Siding_Teal", (0.14, 0.32, 0.40, 1.0), roughness=0.6),
        "siding_ochre": get_or_create_material("Mat_Siding_Ochre", (0.75, 0.48, 0.22, 1.0), roughness=0.6),
        "trim_white": get_or_create_material("Mat_Trim_White", (0.94, 0.95, 0.96, 1.0), roughness=0.35),
        "frame_bronze": get_or_create_material("Mat_Frame_Bronze", (0.10, 0.12, 0.16, 1.0), roughness=0.3),
        "glow_warm": get_or_create_material("Mat_Window_Glow", (1.0, 0.85, 0.45, 1.0), roughness=0.1, emission_color=(1.0, 0.82, 0.40, 1.0), emission_strength=4.5),
        "metal_black": get_or_create_material("Mat_Metal_Black", (0.08, 0.08, 0.10, 1.0), roughness=0.3, metallic=0.9),
        "lamp_bulb": get_or_create_material("Mat_Lamp_Bulb", (1.0, 0.95, 0.7, 1.0), roughness=0.1, emission_color=(1.0, 0.92, 0.6, 1.0), emission_strength=7.0),
        "wood_floor": get_or_create_material("Mat_Wood_Floor", (0.45, 0.28, 0.16, 1.0), roughness=0.4),
        "wall_interior": get_or_create_material("Mat_Wall_Interior", (0.88, 0.86, 0.82, 1.0), roughness=0.7),
        "roof_shingle": get_or_create_material("Mat_Roof_Shingle", (0.18, 0.20, 0.24, 1.0), roughness=0.7),
        "fabric_couch": get_or_create_material("Mat_Fabric_Couch", (0.65, 0.22, 0.20, 1.0), roughness=0.8),
        "fabric_bed": get_or_create_material("Mat_Fabric_Bed", (0.24, 0.42, 0.58, 1.0), roughness=0.7),
    }

# ==============================================================================
# 2. BMESH PRIMITIVE HELPERS
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
# 3. MODULAR PIECE GENERATORS (Standard 2.0m W x 3.0m H snap units)
# ==============================================================================
W_WIDTH = 2.0
W_DEPTH = 0.20
W_HEIGHT = 3.0

def build_modular_wall_solid(bm, m_map):
    # Base wall box
    add_box(bm, Vector((0, 0, W_HEIGHT / 2.0)), (W_WIDTH, W_DEPTH, W_HEIGHT), mat_index=m_map["siding_teal"])
    # Baseboard trim (bottom)
    add_box(bm, Vector((0, 0, 0.10)), (W_WIDTH + 0.02, W_DEPTH + 0.04, 0.20), mat_index=m_map["trim_white"])
    # Cornice trim (top)
    add_box(bm, Vector((0, 0, W_HEIGHT - 0.08)), (W_WIDTH + 0.02, W_DEPTH + 0.04, 0.16), mat_index=m_map["trim_white"])
    # Side corner pilasters
    col_w = 0.08
    add_box(bm, Vector((-W_WIDTH / 2.0 + col_w / 2.0, W_DEPTH / 2.0 + 0.015, W_HEIGHT / 2.0)), (col_w, 0.03, W_HEIGHT), mat_index=m_map["trim_white"])
    add_box(bm, Vector((W_WIDTH / 2.0 - col_w / 2.0, W_DEPTH / 2.0 + 0.015, W_HEIGHT / 2.0)), (col_w, 0.03, W_HEIGHT), mat_index=m_map["trim_white"])
    # Horizontal lapboard siding slats
    slats = 14
    step = W_HEIGHT / slats
    for s in range(1, slats):
        add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.015, s * step)), (W_WIDTH, 0.03, 0.04), mat_index=m_map["siding_teal"])

def build_modular_wall_window(bm, m_map, with_lamp=True, with_planter=True):
    win_w = 1.1
    win_h = 1.5
    win_bottom = 0.85
    win_top = win_bottom + win_h
    win_center_z = (win_bottom + win_top) / 2.0
    side_w = (W_WIDTH - win_w) / 2.0

    # Lower apron
    add_box(bm, Vector((0, 0, win_bottom / 2.0)), (W_WIDTH, W_DEPTH, win_bottom), mat_index=m_map["siding_teal"])
    # Upper lintel
    lintel_h = W_HEIGHT - win_top
    add_box(bm, Vector((0, 0, win_top + lintel_h / 2.0)), (W_WIDTH, W_DEPTH, lintel_h), mat_index=m_map["siding_teal"])
    # Left & Right piers
    add_box(bm, Vector((-(win_w + side_w) / 2.0, 0, win_center_z)), (side_w, W_DEPTH, win_h), mat_index=m_map["siding_teal"])
    add_box(bm, Vector(((win_w + side_w) / 2.0, 0, win_center_z)), (side_w, W_DEPTH, win_h), mat_index=m_map["siding_teal"])

    # Baseboard & Cornice
    add_box(bm, Vector((0, 0, 0.10)), (W_WIDTH + 0.02, W_DEPTH + 0.04, 0.20), mat_index=m_map["trim_white"])
    add_box(bm, Vector((0, 0, W_HEIGHT - 0.08)), (W_WIDTH + 0.02, W_DEPTH + 0.04, 0.16), mat_index=m_map["trim_white"])
    col_w = 0.08
    add_box(bm, Vector((-W_WIDTH / 2.0 + col_w / 2.0, W_DEPTH / 2.0 + 0.015, W_HEIGHT / 2.0)), (col_w, 0.03, W_HEIGHT), mat_index=m_map["trim_white"])
    add_box(bm, Vector((W_WIDTH / 2.0 - col_w / 2.0, W_DEPTH / 2.0 + 0.015, W_HEIGHT / 2.0)), (col_w, 0.03, W_HEIGHT), mat_index=m_map["trim_white"])

    # Slats
    slats = 14
    step = W_HEIGHT / slats
    for s in range(1, slats):
        sz = s * step
        if sz < win_bottom or sz > win_top:
            add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.015, sz)), (W_WIDTH, 0.03, 0.04), mat_index=m_map["siding_teal"])
        else:
            add_box(bm, Vector((-(win_w + side_w) / 2.0, W_DEPTH / 2.0 + 0.015, sz)), (side_w, 0.03, 0.04), mat_index=m_map["siding_teal"])
            add_box(bm, Vector(((win_w + side_w) / 2.0, W_DEPTH / 2.0 + 0.015, sz)), (side_w, 0.03, 0.04), mat_index=m_map["siding_teal"])

    # Moldings
    add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.06, win_bottom)), (win_w + 0.18, 0.12, 0.08), mat_index=m_map["trim_white"])
    add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.06, win_top)), (win_w + 0.22, 0.12, 0.10), mat_index=m_map["trim_white"])

    # Window frame & glowing glass
    frame_t = 0.05
    add_box(bm, Vector((-(win_w - frame_t) / 2.0, W_DEPTH / 2.0 + 0.02, win_center_z)), (frame_t, 0.05, win_h), mat_index=m_map["frame_bronze"])
    add_box(bm, Vector(((win_w - frame_t) / 2.0, W_DEPTH / 2.0 + 0.02, win_center_z)), (frame_t, 0.05, win_h), mat_index=m_map["frame_bronze"])
    add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.02, win_center_z)), (0.04, 0.04, win_h - 0.08), mat_index=m_map["frame_bronze"])
    add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.02, win_center_z + 0.12)), (win_w - 0.08, 0.04, 0.04), mat_index=m_map["frame_bronze"])
    add_box(bm, Vector((0, W_DEPTH / 2.0 - 0.01, win_center_z)), (win_w - 0.06, 0.02, win_h - 0.06), mat_index=m_map["glow_warm"])

    if with_planter:
        fb_w = win_w + 0.1
        add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.15, win_bottom - 0.2)), (fb_w, 0.26, 0.24), mat_index=m_map["trim_white"])
        add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.15, win_bottom - 0.1)), (fb_w - 0.06, 0.20, 0.04), mat_index=m_map["metal_black"])

    if with_lamp:
        lx = win_w / 2.0 + 0.25
        lz = win_top - 0.2
        add_box(bm, Vector((lx, W_DEPTH / 2.0 + 0.02, lz)), (0.12, 0.04, 0.16), mat_index=m_map["metal_black"])
        add_cylinder(bm, Vector((lx, W_DEPTH / 2.0 + 0.14, lz + 0.05)), radius=0.015, depth=0.22, mat_index=m_map["metal_black"], rot_x=math.radians(90))
        add_cylinder(bm, Vector((lx, W_DEPTH / 2.0 + 0.25, lz + 0.02)), radius=0.12, depth=0.06, mat_index=m_map["metal_black"])
        add_cylinder(bm, Vector((lx, W_DEPTH / 2.0 + 0.25, lz - 0.03)), radius=0.04, depth=0.06, mat_index=m_map["lamp_bulb"])

def build_modular_wall_door(bm, m_map):
    door_w = 1.05
    door_h = 2.2
    side_w = (W_WIDTH - door_w) / 2.0
    lintel_h = W_HEIGHT - door_h

    # Upper lintel
    add_box(bm, Vector((0, 0, door_h + lintel_h / 2.0)), (W_WIDTH, W_DEPTH, lintel_h), mat_index=m_map["siding_teal"])
    # Left & Right piers
    add_box(bm, Vector((-(door_w + side_w) / 2.0, 0, door_h / 2.0)), (side_w, W_DEPTH, door_h), mat_index=m_map["siding_teal"])
    add_box(bm, Vector(((door_w + side_w) / 2.0, 0, door_h / 2.0)), (side_w, W_DEPTH, door_h), mat_index=m_map["siding_teal"])

    # Baseboard & Cornice
    add_box(bm, Vector((0, 0, W_HEIGHT - 0.08)), (W_WIDTH + 0.02, W_DEPTH + 0.04, 0.16), mat_index=m_map["trim_white"])
    # Door frame molding
    frame_t = 0.07
    add_box(bm, Vector((-(door_w - frame_t) / 2.0, W_DEPTH / 2.0 + 0.02, door_h / 2.0)), (frame_t, 0.05, door_h), mat_index=m_map["trim_white"])
    add_box(bm, Vector(((door_w - frame_t) / 2.0, W_DEPTH / 2.0 + 0.02, door_h / 2.0)), (frame_t, 0.05, door_h), mat_index=m_map["trim_white"])
    add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.02, door_h)), (door_w + 0.12, 0.08, 0.10), mat_index=m_map["trim_white"])

    # Threshold stone slab at base
    add_box(bm, Vector((0, 0.05, 0.05)), (door_w + 0.1, W_DEPTH + 0.15, 0.08), mat_index=m_map["trim_white"])

    # Slats on upper lintel and piers
    slats = 14
    step = W_HEIGHT / slats
    for s in range(1, slats):
        sz = s * step
        if sz > door_h:
            add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.015, sz)), (W_WIDTH, 0.03, 0.04), mat_index=m_map["siding_teal"])
        else:
            add_box(bm, Vector((-(door_w + side_w) / 2.0, W_DEPTH / 2.0 + 0.015, sz)), (side_w, 0.03, 0.04), mat_index=m_map["siding_teal"])
            add_box(bm, Vector(((door_w + side_w) / 2.0, W_DEPTH / 2.0 + 0.015, sz)), (side_w, 0.03, 0.04), mat_index=m_map["siding_teal"])

    # Small overdoor porch awning
    add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.22, door_h + 0.18)), (door_w + 0.35, 0.40, 0.06), mat_index=m_map["roof_shingle"])

def build_modular_wall_half(bm, m_map):
    # Cutaway wall for camera visibility into interiors
    half_h = 0.85
    add_box(bm, Vector((0, 0, half_h / 2.0)), (W_WIDTH, W_DEPTH, half_h), mat_index=m_map["siding_teal"])
    add_box(bm, Vector((0, 0, 0.08)), (W_WIDTH + 0.02, W_DEPTH + 0.04, 0.16), mat_index=m_map["trim_white"])
    # Coping trim cap along top edge
    add_box(bm, Vector((0, 0, half_h)), (W_WIDTH + 0.02, W_DEPTH + 0.05, 0.08), mat_index=m_map["trim_white"])
    # 3 lapboard slats
    for s in [0.28, 0.48, 0.68]:
        add_box(bm, Vector((0, W_DEPTH / 2.0 + 0.015, s)), (W_WIDTH, 0.03, 0.04), mat_index=m_map["siding_teal"])

def build_modular_floor_slab(bm, m_map):
    # 2.0m x 2.0m floor tile with plank seams
    add_box(bm, Vector((0, 0, -0.06)), (2.0, 2.0, 0.12), mat_index=m_map["wood_floor"])
    # Plank groove lines on top face
    for px in [-0.66, -0.33, 0.0, 0.33, 0.66]:
        add_box(bm, Vector((px, 0, 0.005)), (0.015, 2.0, 0.01), mat_index=m_map["frame_bronze"])

def build_modular_stairs(bm, m_map):
    # 2m long, 0.9m wide staircase rising 3.0m
    steps = 10
    step_l = 2.0 / steps
    step_h = 3.0 / steps
    for i in range(steps):
        # Step riser & tread
        cy = -1.0 + (i + 0.5) * step_l
        cz = (i + 0.5) * step_h
        add_box(bm, Vector((0, cy, cz)), (0.9, step_l, step_h), mat_index=m_map["wood_floor"])
    # Handrail
    add_box(bm, Vector((0.42, 0, 1.8)), (0.05, 2.1, 0.06), mat_index=m_map["trim_white"])

def build_modular_roof_gable(bm, m_map, cutaway_front=True):
    # Pitched gable roof section: width 2.0m along X, span 4.0m along Y (from Y=-2 to Y=+2)
    pitch_angle = math.radians(33)
    slope_len = 2.45
    ridge_h = 1.35

    # Back/North slope (intact)
    add_box(bm, Vector((0, 1.05, ridge_h / 2.0)), (2.05, slope_len, 0.12), mat_index=m_map["roof_shingle"], rot_x=-pitch_angle)
    # Ridge cap running along the peak
    add_box(bm, Vector((0, 0, ridge_h + 0.05)), (2.10, 0.38, 0.12), mat_index=m_map["metal_black"])

    if not cutaway_front:
        # Full front slope
        add_box(bm, Vector((0, -1.05, ridge_h / 2.0)), (2.05, slope_len, 0.12), mat_index=m_map["roof_shingle"], rot_x=pitch_angle)
    else:
        # Subtle architectural cutaway eave lip
        add_box(bm, Vector((0, -0.30, ridge_h - 0.16)), (2.05, 0.65, 0.12), mat_index=m_map["roof_shingle"], rot_x=pitch_angle)

    # White fascia trims on eave ends
    add_box(bm, Vector((0, 2.15, 0.10)), (2.08, 0.06, 0.16), mat_index=m_map["trim_white"])

# ==============================================================================
# 4. LOW-POLY INTERIOR FURNITURE (For walk-in playable rooms)
# ==============================================================================
def add_sofa(bm, center, m_map, rot_z=0.0):
    # Cozy living room couch
    cx, cy, cz = center
    # Seat
    add_box(bm, Vector((cx, cy, cz + 0.25)), (1.4, 0.7, 0.3), mat_index=m_map["fabric_couch"], rot_z=rot_z)
    # Backrest
    add_box(bm, Vector((cx, cy - 0.28, cz + 0.55)), (1.4, 0.2, 0.45), mat_index=m_map["fabric_couch"], rot_z=rot_z)
    # Armrests
    add_box(bm, Vector((cx - 0.65, cy, cz + 0.4)), (0.18, 0.7, 0.35), mat_index=m_map["fabric_couch"], rot_z=rot_z)
    add_box(bm, Vector((cx + 0.65, cy, cz + 0.4)), (0.18, 0.7, 0.35), mat_index=m_map["fabric_couch"], rot_z=rot_z)

def add_bed(bm, center, m_map, rot_z=0.0):
    cx, cy, cz = center
    # Wooden frame
    add_box(bm, Vector((cx, cy, cz + 0.2)), (1.5, 2.1, 0.3), mat_index=m_map["wood_floor"], rot_z=rot_z)
    # Mattress & duvet
    add_box(bm, Vector((cx, cy, cz + 0.4)), (1.4, 2.0, 0.22), mat_index=m_map["fabric_bed"], rot_z=rot_z)
    # Pillows
    add_box(bm, Vector((cx - 0.35, cy - 0.7, cz + 0.55)), (0.5, 0.35, 0.12), mat_index=m_map["trim_white"], rot_z=rot_z)
    add_box(bm, Vector((cx + 0.35, cy - 0.7, cz + 0.55)), (0.5, 0.35, 0.12), mat_index=m_map["trim_white"], rot_z=rot_z)

def add_table_and_lamp(bm, center, m_map):
    cx, cy, cz = center
    # Table top
    add_box(bm, Vector((cx, cy, cz + 0.45)), (0.6, 0.6, 0.06), mat_index=m_map["wood_floor"])
    # Leg
    add_cylinder(bm, Vector((cx, cy, cz + 0.22)), radius=0.04, depth=0.44, mat_index=m_map["wood_floor"])
    # Glowing table lamp
    add_cylinder(bm, Vector((cx, cy, cz + 0.62)), radius=0.10, depth=0.18, mat_index=m_map["glow_warm"])

# ==============================================================================
# 5. BLUEPRINT-TO-BUILDING HYBRID ASSEMBLER
# ==============================================================================
def assemble_walkin_suburban_house():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    palette = create_palette()

    # Create Material Index Map
    mat_list = [
        palette["siding_teal"],    # 0
        palette["siding_ochre"],   # 1
        palette["trim_white"],     # 2
        palette["frame_bronze"],   # 3
        palette["glow_warm"],      # 4
        palette["metal_black"],    # 5
        palette["lamp_bulb"],      # 6
        palette["wood_floor"],     # 7
        palette["wall_interior"],  # 8
        palette["roof_shingle"],   # 9
        palette["fabric_couch"],   # 10
        palette["fabric_bed"]      # 11
    ]
    m_map = {
        "siding_teal": 0,
        "siding_ochre": 1,
        "trim_white": 2,
        "frame_bronze": 3,
        "glow_warm": 4,
        "metal_black": 5,
        "lamp_bulb": 6,
        "wood_floor": 7,
        "wall_interior": 8,
        "roof_shingle": 9,
        "fabric_couch": 10,
        "fabric_bed": 11
    }

    mesh = bpy.data.meshes.new("AssembledHouse_Mesh")
    bm = bmesh.new()

    # Helper to stamp a modular sub-bmesh into the main building bmesh with transform
    def stamp_submesh(builder_func, location, rot_z=0.0, **kwargs):
        temp_bm = bmesh.new()
        builder_func(temp_bm, m_map, **kwargs)
        mat = Matrix.Translation(location) @ Matrix.Rotation(rot_z, 4, 'Z')
        f_before = set(bm.faces)
        bmesh.ops.transform(temp_bm, matrix=mat, verts=temp_bm.verts)
        # Append verts/faces to main bm
        vert_map = {}
        for v in temp_bm.verts:
            vert_map[v] = bm.verts.new(v.co)
        for f in temp_bm.faces:
            new_verts = [vert_map[v] for v in f.verts]
            new_f = bm.faces.new(new_verts)
            new_f.material_index = f.material_index
        temp_bm.free()

    # -------------------------------------------------------------
    # 2-STORY HOUSE SPECIFICATION: 4m x 4m footprint (2x2 grid)
    # Grid tile centers: (-1, -1), (1, -1), (-1, 1), (1, 1)
    # Floor 0: Living Room & Kitchenette
    # Floor 1: Master Bedroom & Interior Balcony
    # Front is facing +Y (or -Y depending on view). We will face front at -Y for isometric camera looking from (-X, -Y, +Z)!
    # -------------------------------------------------------------

    # === GROUND FLOOR (Z = 0) ===
    # 4 floor tiles
    stamp_submesh(build_modular_floor_slab, Vector((-1, -1, 0)))
    stamp_submesh(build_modular_floor_slab, Vector(( 1, -1, 0)))
    stamp_submesh(build_modular_floor_slab, Vector((-1,  1, 0)))
    stamp_submesh(build_modular_floor_slab, Vector(( 1,  1, 0)))

    # Perimeter Walls (Ground Floor):
    # Front-South (Y = -2): Camera cutaway view!
    # Left front: Entrance door
    stamp_submesh(build_modular_wall_door, Vector((-1, -2, 0)), rot_z=math.radians(180))
    # Right front: Half wall (cutaway) so player can see living room interior!
    stamp_submesh(build_modular_wall_half, Vector(( 1, -2, 0)), rot_z=math.radians(180))

    # Back-North (Y = +2): Solid & Window
    stamp_submesh(build_modular_wall_window, Vector((-1, 2, 0)), rot_z=0.0, with_planter=False)
    stamp_submesh(build_modular_wall_solid, Vector(( 1, 2, 0)), rot_z=0.0)

    # Left-West (X = -2): Window & Solid
    stamp_submesh(build_modular_wall_solid, Vector((-2, -1, 0)), rot_z=math.radians(90))
    stamp_submesh(build_modular_wall_window, Vector((-2,  1, 0)), rot_z=math.radians(90))

    # Right-East (X = +2): Solid & Solid
    stamp_submesh(build_modular_wall_solid, Vector(( 2, -1, 0)), rot_z=math.radians(-90))
    stamp_submesh(build_modular_wall_solid, Vector(( 2,  1, 0)), rot_z=math.radians(-90))

    # Ground Floor Interior Furniture & Stairs:
    # Cozy rug under sofa
    add_box(bm, Vector((1.0, -0.6, 0.015)), (1.6, 1.5, 0.02), mat_index=m_map["fabric_bed"])
    # Cozy sofa in Living Room
    add_sofa(bm, Vector((1.0, -0.6, 0.0)), m_map, rot_z=math.radians(90))
    # Coffee table & glowing lamp
    add_table_and_lamp(bm, Vector((0.3, -0.6, 0.0)), m_map)
    # Staircase along East wall rising to floor 1 (Z=0 to Z=3.0)
    stamp_submesh(build_modular_stairs, Vector((1.5, 0.9, 0.0)), rot_z=math.radians(180))

    # === SECOND FLOOR (Z = 3.0) ===
    # Floor slabs (leaving opening for stairs at (1, 1))
    stamp_submesh(build_modular_floor_slab, Vector((-1, -1, 3.0)))
    stamp_submesh(build_modular_floor_slab, Vector(( 1, -1, 3.0)))
    stamp_submesh(build_modular_floor_slab, Vector((-1,  1, 3.0)))

    # Second Floor Perimeter Walls:
    # Front-South (Y = -2): Half wall cutaway for bedroom view!
    stamp_submesh(build_modular_wall_half, Vector((-1, -2, 3.0)), rot_z=math.radians(180))
    stamp_submesh(build_modular_wall_window, Vector(( 1, -2, 3.0)), rot_z=math.radians(180))

    # Back-North (Y = +2): High windows
    stamp_submesh(build_modular_wall_window, Vector((-1, 2, 3.0)), rot_z=0.0)
    stamp_submesh(build_modular_wall_solid, Vector(( 1, 2, 3.0)), rot_z=0.0)

    # Left-West (X = -2): Window wall
    stamp_submesh(build_modular_wall_window, Vector((-2, -1, 3.0)), rot_z=math.radians(90))
    stamp_submesh(build_modular_wall_solid, Vector((-2,  1, 3.0)), rot_z=math.radians(90))

    # Right-East (X = +2): Solid walls
    stamp_submesh(build_modular_wall_solid, Vector(( 2, -1, 3.0)), rot_z=math.radians(-90))
    stamp_submesh(build_modular_wall_solid, Vector(( 2,  1, 3.0)), rot_z=math.radians(-90))

    # Bedroom Interior Furniture:
    # Double bed in bedroom
    add_bed(bm, Vector((-1.0, 0.5, 3.0)), m_map, rot_z=0.0)
    add_table_and_lamp(bm, Vector((-1.8, 0.5, 3.0)), m_map)

    # === ROOF LEVEL (Z = 6.0) ===
    # Angled gable roof caps snapping on top
    stamp_submesh(build_modular_roof_gable, Vector((-1, 0, 6.0)))
    stamp_submesh(build_modular_roof_gable, Vector(( 1, 0, 6.0)))

    # Chimney on the roof
    add_box(bm, Vector((0.8, 0.9, 7.1)), (0.5, 0.5, 1.2), mat_index=m_map["siding_ochre"])
    add_box(bm, Vector((0.8, 0.9, 7.75)), (0.6, 0.6, 0.12), mat_index=m_map["metal_black"])

    # Finalize BMesh and link
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("ModularSuburbanHouse_WalkIn", mesh)
    bpy.context.collection.objects.link(obj)

    # Link materials
    for m in mat_list:
        obj.data.materials.append(m)

    # Studio Floor Ground
    bg_mesh = bpy.data.meshes.new("BackdropGround")
    bm_bg = bmesh.new()
    add_box(bm_bg, Vector((0, 0, -0.06)), (16.0, 16.0, 0.12), mat_index=0)
    # Add a small sidewalk curb
    add_box(bm_bg, Vector((0, -2.8, 0.05)), (8.0, 1.4, 0.10), mat_index=1)
    # Sidewalk streetlamp
    add_cylinder(bm_bg, Vector((-2.6, -2.8, 1.4)), radius=0.04, depth=2.8, mat_index=1)
    add_cylinder(bm_bg, Vector((-2.6, -2.8, 2.85)), radius=0.20, depth=0.08, mat_index=1)
    add_cylinder(bm_bg, Vector((-2.6, -2.8, 2.76)), radius=0.08, depth=0.10, mat_index=1)
    bm_bg.to_mesh(bg_mesh)
    bm_bg.free()
    bg_obj = bpy.data.objects.new("Ground_Sidewalk", bg_mesh)
    bg_obj.data.materials.append(palette["roof_shingle"])
    bg_obj.data.materials.append(palette["trim_white"])
    bpy.context.collection.objects.link(bg_obj)

    # -------------------------------------------------------------
    # CAMERA & LIGHTING (Isometric Diorama View showing inside rooms!)
    # -------------------------------------------------------------
    cam_target = bpy.data.objects.new("HouseCamTarget", None)
    cam_target.location = (0, 0, 2.8)
    bpy.context.collection.objects.link(cam_target)

    cam_data = bpy.data.cameras.new("HouseCamera")
    cam_data.type = 'PERSP'
    cam_data.lens = 42.0
    cam_obj = bpy.data.objects.new("HouseStudioCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Position camera at classic isometric angle looking into cutaway front
    cam_obj.location = (8.5, -9.5, 8.0)
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = cam_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam_obj

    # Key Light (Moonlight / Warm Street Ambience)
    sun = bpy.data.lights.new("SunKey", type='SUN')
    sun.energy = 4.5
    sun.color = (1.0, 0.95, 0.88)
    sun_obj = bpy.data.objects.new("SunKey", sun)
    sun_obj.rotation_euler = (math.radians(50), math.radians(15), math.radians(-45))
    bpy.context.collection.objects.link(sun_obj)

    # Cool Sky Fill
    fill = bpy.data.lights.new("SkyFill", type='SUN')
    fill.energy = 1.2
    fill.color = (0.5, 0.7, 1.0)
    fill_obj = bpy.data.objects.new("SkyFill", fill)
    fill_obj.rotation_euler = (math.radians(-30), math.radians(20), math.radians(120))
    bpy.context.collection.objects.link(fill_obj)

    # Interior Warm Fill Lights (Simulating lamps lighting up inside rooms)
    int_light_0 = bpy.data.lights.new("LivingRoomLight", type='POINT')
    int_light_0.energy = 60.0
    int_light_0.color = (1.0, 0.85, 0.5)
    int_light_0_obj = bpy.data.objects.new("LivingRoomLight", int_light_0)
    int_light_0_obj.location = (0.5, -0.6, 1.8)
    bpy.context.collection.objects.link(int_light_0_obj)

    int_light_1 = bpy.data.lights.new("BedroomLight", type='POINT')
    int_light_1.energy = 50.0
    int_light_1.color = (1.0, 0.88, 0.6)
    int_light_1_obj = bpy.data.objects.new("BedroomLight", int_light_1)
    int_light_1_obj.location = (-0.8, 0.3, 4.8)
    bpy.context.collection.objects.link(int_light_1_obj)

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.join(script_dir, "assembled_modular_house_preview.png")
    out_blend = os.path.join(script_dir, "assembled_modular_house.blend")
    out_glb = os.path.join(script_dir, "assembled_modular_house.glb")

    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB', use_selection=False)

    print(f"HOUSE_ASSEMBLED_SUCCESS: Image={out_img}, Blend={out_blend}, GLB={out_glb}")

if __name__ == "__main__":
    assemble_walkin_suburban_house()
