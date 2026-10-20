"""
Modular Snap Kit & Asset Generator for World 4 Bro's
Generates procedural 3D models for snap-kit building pieces, furniture, roads, and props
strictly calibrated to docs/STYLE_BIBLE.md (§0.3 palette, §0.5 scale sheet: 1 tile = 1m, 1 storey = 2.7m).
"""

import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# ==============================================================================
# 1. COLOR & MATERIAL UTILITIES (docs/STYLE_BIBLE.md §0.3)
# ==============================================================================
def hex_to_linear(hex_str):
    """Converts a hex color code (e.g. '#7a4a3a') to Blender linear RGB."""
    h = hex_str.lstrip('#')
    srgb = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    # sRGB to Linear conversion
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in srgb
    ]
    return (linear[0], linear[1], linear[2], 1.0)

PALETTE = {
    # Ground & Streets
    "grass": hex_to_linear("#5f6b45"),
    "dirt": hex_to_linear("#6e5a43"),
    "sand": hex_to_linear("#a8946b"),
    "asphalt": hex_to_linear("#3a3c3f"),
    "asphalt_worn": hex_to_linear("#4a4c4f"),
    "sidewalk": hex_to_linear("#8b8a84"),
    "kerb": hex_to_linear("#a3a199"),
    "road_line_white": hex_to_linear("#d8d6cc"),
    "road_line_yellow": hex_to_linear("#c9a447"),

    # Building materials
    "brick": hex_to_linear("#7a4a3a"),
    "siding_cream": hex_to_linear("#b3a98f"),
    "siding_teal": hex_to_linear("#6f7f86"),
    "wood_dark": hex_to_linear("#4a3a2c"),
    "wood_plank": hex_to_linear("#6b5440"),
    "concrete": hex_to_linear("#8b8a84"),
    "trim": hex_to_linear("#c9c3b3"),
    "roof_shingle": hex_to_linear("#3d3b3a"),
    "roof_red": hex_to_linear("#6a3f34"),
    "glass": (0.1, 0.18, 0.22, 0.7),

    # Metals, Props & Furniture
    "metal_dark": hex_to_linear("#2a2b2d"),
    "metal_rust": hex_to_linear("#8a4325"),
    "metal_steel": hex_to_linear("#77736a"),
    "appliance_white": hex_to_linear("#c9cfd2"),
    "cloth_denim": hex_to_linear("#4d5a6a"),
    "cloth_cushion": hex_to_linear("#7b7259"),
    "blood": hex_to_linear("#6b1414"),
}

def get_or_create_material(name, color, roughness=0.7, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
    return mat

def get_material_library():
    lib = {}
    for k, col in PALETTE.items():
        rough = 0.35 if "metal" in k else 0.8 if "wood" in k or "brick" in k else 0.6
        metal = 0.85 if "metal" in k or "steel" in k else 0.0
        lib[k] = get_or_create_material(f"Mat_{k}", col, roughness=rough, metallic=metal)
    return lib

# ==============================================================================
# 2. MESH MODELING PRIMITIVES
# ==============================================================================
def add_box(bm, center, size, mat_index=0):
    sx, sy, sz = size
    mat = Matrix.Translation(center) @ Matrix.Diagonal((sx, sy, sz, 1.0))
    f_before = set(bm.faces)
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    for face in bm.faces:
        if face not in f_before:
            face.material_index = mat_index

def add_cylinder(bm, center, radius, depth, mat_index=0, segments=12):
    mat = Matrix.Translation(center)
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
# 3. PROCEDURAL SNAP-KIT PIECES (1m grid, 2.7m height)
# ==============================================================================
def build_wall(bm, mat_idx, trim_idx, cut_type="solid", knee_high=False):
    """
    Standard snap-kit wall: 1m wide along X, 0.2m thick along Y, 2.7m high along Z.
    cut_type: 'solid', 'window', 'door'
    knee_high: If true, drops to 0.7m for interior cut-away viewing!
    """
    height = 0.7 if knee_high else 2.7
    w = 1.0
    th = 0.2

    if cut_type == "solid" or knee_high:
        # Base solid wall
        add_box(bm, Vector((0, 0, height / 2.0)), (w, th, height), mat_idx)
        # Top trim
        add_box(bm, Vector((0, 0, height - 0.05)), (w * 1.02, th * 1.08, 0.1), trim_idx)
    elif cut_type == "door":
        # Door opening: 0.8m wide, 2.1m tall in the center
        post_w = (w - 0.8) / 2.0
        # Left post
        add_box(bm, Vector((-w / 2.0 + post_w / 2.0, 0, 2.1 / 2.0)), (post_w, th, 2.1), mat_idx)
        # Right post
        add_box(bm, Vector((w / 2.0 - post_w / 2.0, 0, 2.1 / 2.0)), (post_w, th, 2.1), mat_idx)
        # Header over door
        header_h = height - 2.1
        add_box(bm, Vector((0, 0, 2.1 + header_h / 2.0)), (w, th, header_h), mat_idx)
        # Wooden Door frame
        add_box(bm, Vector((0, 0, 1.05)), (0.76, 0.06, 2.05), trim_idx)
    elif cut_type == "window":
        # Window opening: sill at 0.9m, 1.0m tall opening, 0.8m wide
        # Sill section below
        add_box(bm, Vector((0, 0, 0.45)), (w, th, 0.9), mat_idx)
        # Top lintel above window
        add_box(bm, Vector((0, 0, 1.9 + (height - 1.9) / 2.0)), (w, th, height - 1.9), mat_idx)
        # Left post
        add_box(bm, Vector((-0.45, 0, 1.4)), (0.1, th, 1.0), mat_idx)
        # Right post
        add_box(bm, Vector((0.45, 0, 1.4)), (0.1, th, 1.0), mat_idx)
        # Window sill ledge
        add_box(bm, Vector((0, 0, 0.9)), (0.86, th * 1.2, 0.06), trim_idx)

def build_floor_tile(bm, mat_idx, subfloor_idx):
    """1m x 1m floor tile (1m diamond in isometric)."""
    add_box(bm, Vector((0, 0, -0.05)), (1.0, 1.0, 0.1), subfloor_idx)
    add_box(bm, Vector((0, 0, 0.01)), (0.98, 0.98, 0.02), mat_idx)

def build_roof_gable(bm, shingle_idx, trim_idx):
    """Modular 1m gable roof slope piece (1m x 1m footprint, 0.6m rise)."""
    # Slanted roof plank
    mat = Matrix.Translation(Vector((0, 0, 0.35))) @ Matrix.Rotation(math.radians(25), 4, 'Y') @ Matrix.Diagonal((1.1, 1.0, 0.08, 1.0))
    f_before = set(bm.faces)
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    for f in bm.faces:
        if f not in f_before:
            f.material_index = shingle_idx
    # Edge trim
    add_box(bm, Vector((0, 0, 0.05)), (1.15, 0.08, 0.08), trim_idx)

# ==============================================================================
# 4. PROCEDURAL FURNITURE & PROPS (Vertical Slice Set)
# ==============================================================================
def build_bed(bm, p):
    # Wood frame
    add_box(bm, Vector((0, 0, 0.2)), (1.2, 2.0, 0.4), p["wood_plank"])
    # Headboard
    add_box(bm, Vector((0, 0.95, 0.55)), (1.25, 0.1, 0.9), p["wood_dark"])
    # Mattress
    add_box(bm, Vector((0, -0.05, 0.45)), (1.1, 1.85, 0.25), p["cloth_cushion"])
    # Pillow
    add_box(bm, Vector((0, 0.65, 0.62)), (0.8, 0.4, 0.12), p["trim"])

def build_wardrobe(bm, p):
    # Tall 2-door wooden wardrobe (1.0m wide, 0.6m deep, 2.0m tall)
    add_box(bm, Vector((0, 0, 1.0)), (1.0, 0.6, 2.0), p["wood_dark"])
    # Doors recessed line
    add_box(bm, Vector((0, -0.301, 1.0)), (0.02, 0.01, 1.8), p["metal_dark"])
    # Metal handles
    add_box(bm, Vector((-0.06, -0.32, 1.0)), (0.04, 0.04, 0.12), p["metal_steel"])
    add_box(bm, Vector((0.06, -0.32, 1.0)), (0.04, 0.04, 0.12), p["metal_steel"])

def build_sofa(bm, p):
    # 3-seater living room couch (2.0m wide, 0.85m deep, 0.85m tall)
    # Base
    add_box(bm, Vector((0, 0, 0.25)), (2.0, 0.85, 0.4), p["cloth_denim"])
    # Backrest
    add_box(bm, Vector((0, 0.32, 0.65)), (2.0, 0.22, 0.55), p["cloth_denim"])
    # Armrests
    add_box(bm, Vector((-0.92, 0, 0.5)), (0.2, 0.85, 0.4), p["cloth_denim"])
    add_box(bm, Vector((0.92, 0, 0.5)), (0.2, 0.85, 0.4), p["cloth_denim"])
    # Cushions
    for i in [-0.55, 0.0, 0.55]:
        add_box(bm, Vector((i, -0.08, 0.48)), (0.52, 0.55, 0.15), p["cloth_cushion"])

def build_fridge(bm, p):
    # Grimy kitchen refrigerator (0.8m wide, 0.7m deep, 1.8m tall)
    add_box(bm, Vector((0, 0, 0.9)), (0.8, 0.7, 1.8), p["appliance_white"])
    # Door separation seam
    add_box(bm, Vector((0, -0.351, 1.15)), (0.8, 0.01, 0.02), p["metal_dark"])
    # Door handle
    add_box(bm, Vector((0.32, -0.38, 1.3)), (0.04, 0.05, 0.25), p["metal_steel"])

def build_dumpster(bm, p):
    # Industrial green/teal dumpster (2.0m wide, 1.0m deep, 1.3m tall)
    add_box(bm, Vector((0, 0, 0.65)), (2.0, 1.0, 1.1), p["siding_teal"])
    # Rust along bottom rim
    add_box(bm, Vector((0, 0, 0.1)), (2.04, 1.04, 0.2), p["metal_rust"])
    # Plastic lids
    add_box(bm, Vector((-0.5, 0, 1.25)), (0.95, 1.05, 0.1), p["metal_dark"])
    add_box(bm, Vector((0.5, 0, 1.25)), (0.95, 1.05, 0.1), p["metal_dark"])

def build_hatchback_car(bm, p):
    # Rusty hatchback vehicle (4.5m long along Y, 1.8m wide along X, 1.5m tall along Z)
    # Main chassis body
    add_box(bm, Vector((0, 0, 0.55)), (1.75, 4.2, 0.65), p["siding_cream"])
    # Cabin / Roof
    add_box(bm, Vector((0, -0.3, 1.15)), (1.55, 2.3, 0.65), p["siding_cream"])
    # Windshield & windows
    add_box(bm, Vector((0, 0.88, 1.12)), (1.5, 0.1, 0.55), p["glass"])
    add_box(bm, Vector((0, -1.48, 1.12)), (1.5, 0.1, 0.5), p["glass"])
    # Bumpers
    add_box(bm, Vector((0, 2.15, 0.4)), (1.78, 0.2, 0.3), p["metal_dark"])
    add_box(bm, Vector((0, -2.15, 0.4)), (1.78, 0.2, 0.3), p["metal_dark"])
    # 4 Wheels
    for wx in [-0.88, 0.88]:
        for wy in [-1.3, 1.3]:
            add_cylinder(bm, Vector((wx, wy, 0.35)), radius=0.35, depth=0.22, mat_index=p["metal_dark"], segments=12)

# ==============================================================================
# 5. ASSET BUILD DISPATCHER
# ==============================================================================
def create_model_by_id(asset_id):
    """
    Creates a new Blender object for the specified asset ID from the catalog.
    Returns the created bpy.types.Object.
    """
    mat_lib = get_material_library()
    mat_keys = list(mat_lib.keys())
    p_map = {k: i for i, k in enumerate(mat_keys)}

    mesh = bpy.data.meshes.new(f"{asset_id}_Mesh")
    bm = bmesh.new()

    # Route based on asset ID
    if asset_id == "wall_siding":
        build_wall(bm, p_map["siding_cream"], p_map["trim"], "solid")
    elif asset_id == "wall_siding_window":
        build_wall(bm, p_map["siding_cream"], p_map["trim"], "window")
    elif asset_id == "wall_siding_door":
        build_wall(bm, p_map["siding_cream"], p_map["wood_plank"], "door")
    elif asset_id == "wall_brick":
        build_wall(bm, p_map["brick"], p_map["trim"], "solid")
    elif asset_id == "wall_brick_window":
        build_wall(bm, p_map["brick"], p_map["trim"], "window")
    elif asset_id == "wall_brick_door":
        build_wall(bm, p_map["brick"], p_map["wood_plank"], "door")
    elif asset_id == "floor_wood":
        build_floor_tile(bm, p_map["wood_plank"], p_map["wood_dark"])
    elif asset_id == "floor_concrete":
        build_floor_tile(bm, p_map["concrete"], p_map["metal_dark"])
    elif asset_id == "roof_gable_slope":
        build_roof_gable(bm, p_map["roof_shingle"], p_map["trim"])
    elif asset_id == "bed":
        build_bed(bm, p_map)
    elif asset_id == "wardrobe":
        build_wardrobe(bm, p_map)
    elif asset_id == "sofa":
        build_sofa(bm, p_map)
    elif asset_id == "fridge":
        build_fridge(bm, p_map)
    elif asset_id == "dumpster":
        build_dumpster(bm, p_map)
    elif "hatchback" in asset_id or asset_id == "vehicle_car":
        build_hatchback_car(bm, p_map)
    else:
        # Generic fallback blockout
        add_box(bm, Vector((0, 0, 0.5)), (1.0, 1.0, 1.0), p_map["concrete"])

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(asset_id, mesh)
    bpy.context.collection.objects.link(obj)
    for k in mat_keys:
        obj.data.materials.append(mat_lib[k])

    return obj
