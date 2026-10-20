import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# ==============================================================================
# 1. SHARED MATERIAL SYSTEM (Rich Palette with SSS & Roughness Contrast)
# ==============================================================================
def get_or_create_material(name, base_color, roughness=0.5, metallic=0.0, emission_color=(0,0,0,1), emission_strength=0.0):
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

def create_character_engine_palette():
    return {
        "skin_fair": get_or_create_material("Mat_Skin_Fair", (0.86, 0.68, 0.54, 1.0), roughness=0.52),
        "skin_tan": get_or_create_material("Mat_Skin_Tan", (0.74, 0.50, 0.34, 1.0), roughness=0.52),
        "skin_dark": get_or_create_material("Mat_Skin_Dark", (0.36, 0.23, 0.15, 1.0), roughness=0.52),
        "eye_dark": get_or_create_material("Mat_Eye_Dark", (0.08, 0.08, 0.10, 1.0), roughness=0.2),
        "underwear_black": get_or_create_material("Mat_Underwear_Black", (0.12, 0.12, 0.14, 1.0), roughness=0.7),
        "underwear_waistband": get_or_create_material("Mat_Underwear_Waistband", (0.88, 0.88, 0.90, 1.0), roughness=0.5),
        "hair_dark": get_or_create_material("Mat_Hair_Dark", (0.10, 0.08, 0.06, 1.0), roughness=0.65),
        "hair_brown": get_or_create_material("Mat_Hair_Brown", (0.32, 0.18, 0.10, 1.0), roughness=0.65),
        "hoodie_teal": get_or_create_material("Mat_Hoodie_Teal", (0.13, 0.34, 0.42, 1.0), roughness=0.6),
        "vest_ochre": get_or_create_material("Mat_Vest_Ochre", (0.78, 0.48, 0.20, 1.0), roughness=0.65),
        "shirt_flannel": get_or_create_material("Mat_Shirt_Flannel", (0.68, 0.18, 0.14, 1.0), roughness=0.75),
        "tank_top_white": get_or_create_material("Mat_Tank_Top_White", (0.92, 0.92, 0.94, 1.0), roughness=0.55),
        "pants_charcoal": get_or_create_material("Mat_Pants_Charcoal", (0.16, 0.17, 0.19, 1.0), roughness=0.65),
        "pants_olive": get_or_create_material("Mat_Pants_Olive", (0.26, 0.32, 0.22, 1.0), roughness=0.65),
        "pants_bluejeans": get_or_create_material("Mat_Pants_Bluejeans", (0.18, 0.30, 0.46, 1.0), roughness=0.7),
        "shoes_white": get_or_create_material("Mat_Shoes_White", (0.92, 0.93, 0.95, 1.0), roughness=0.4),
        "boots_dark": get_or_create_material("Mat_Boots_Dark", (0.10, 0.09, 0.08, 1.0), roughness=0.4),
        "boots_sole": get_or_create_material("Mat_Boots_Sole", (0.05, 0.05, 0.06, 1.0), roughness=0.6),
        "beanie_red": get_or_create_material("Mat_Beanie_Red", (0.72, 0.16, 0.14, 1.0), roughness=0.6),
        "cap_tan": get_or_create_material("Mat_Cap_Tan", (0.66, 0.52, 0.36, 1.0), roughness=0.6),
        "metal_weapon": get_or_create_material("Mat_Metal_Weapon", (0.12, 0.12, 0.14, 1.0), roughness=0.25, metallic=0.9),
        "metal_buckle": get_or_create_material("Mat_Metal_Buckle", (0.75, 0.76, 0.80, 1.0), roughness=0.2, metallic=0.95),
        "wood_stock": get_or_create_material("Mat_Wood_Stock", (0.42, 0.25, 0.14, 1.0), roughness=0.55),
        "glow_flashlight": get_or_create_material("Mat_Glow_Flashlight", (1.0, 0.98, 0.80, 1.0), roughness=0.1, emission_color=(1.0, 0.95, 0.70, 1.0), emission_strength=8.0),
        "pedestal_dark": get_or_create_material("Mat_Pedestal_Dark", (0.07, 0.09, 0.12, 1.0), roughness=0.5),
    }

# ==============================================================================
# 2. ADVANCED TAPERED & FACETED PRIMITIVE BUILDERS
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

def add_tapered_frustum(bm, center, size_top, size_bottom, height, mat_index=0, rot_x=0.0, rot_y=0.0, rot_z=0.0):
    # Builds a tapered anatomical block (e.g. bicep, forearm, thigh, calf, or tapered waist)
    stx, sty = size_top
    sbx, sby = size_bottom
    hz = height / 2.0

    # 8 vertices of the frustum
    verts = [
        Vector((-sbx / 2.0, -sby / 2.0, -hz)), # 0: bottom front left
        Vector(( sbx / 2.0, -sby / 2.0, -hz)), # 1: bottom front right
        Vector(( sbx / 2.0,  sby / 2.0, -hz)), # 2: bottom back right
        Vector((-sbx / 2.0,  sby / 2.0, -hz)), # 3: bottom back left
        Vector((-stx / 2.0, -sty / 2.0,  hz)), # 4: top front left
        Vector(( stx / 2.0, -sty / 2.0,  hz)), # 5: top front right
        Vector(( stx / 2.0,  sty / 2.0,  hz)), # 6: top back right
        Vector((-stx / 2.0,  sty / 2.0,  hz)), # 7: top back left
    ]

    mat = (Matrix.Translation(center) @
           Matrix.Rotation(rot_z, 4, 'Z') @
           Matrix.Rotation(rot_y, 4, 'Y') @
           Matrix.Rotation(rot_x, 4, 'X'))

    bm_verts = [bm.verts.new(mat @ v) for v in verts]
    faces_idx = [
        [0, 1, 2, 3], # Bottom
        [4, 7, 6, 5], # Top
        [0, 4, 5, 1], # Front
        [1, 5, 6, 2], # Right
        [2, 6, 7, 3], # Back
        [3, 7, 4, 0], # Left
    ]
    for fi in faces_idx:
        f = bm.faces.new([bm_verts[i] for i in fi])
        f.material_index = mat_index

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
# 3. CHISELED ANATOMICAL HEAD & HAIR BUILDER
# ==============================================================================
def add_chiseled_head(bm, center, p_map, skin_mat, hair_mat, hair_style="messy"):
    cx, cy, cz = center

    # 1. Upper Skull (Brow, Temples & Forehead)
    add_box(bm, Vector((cx, cy + 0.01, cz + 0.08)), (0.28, 0.28, 0.16), skin_mat)

    # 2. Tapered Jawline & Chin (Slopes inward toward the chin)
    add_tapered_frustum(bm, Vector((cx, cy + 0.02, cz - 0.06)),
                        size_top=(0.28, 0.26), size_bottom=(0.18, 0.20), height=0.14, mat_index=skin_mat)

    # 3. Stylized Angular Nose Bridge
    add_box(bm, Vector((cx, cy + 0.155, cz + 0.02)), (0.045, 0.035, 0.07), skin_mat)

    # 4. Inset Eye Sockets with Expressive Pupils
    add_box(bm, Vector((cx - 0.07, cy + 0.145, cz + 0.06)), (0.035, 0.015, 0.035), p_map["eye_dark"])
    add_box(bm, Vector((cx + 0.07, cy + 0.145, cz + 0.06)), (0.035, 0.015, 0.035), p_map["eye_dark"])

    # 5. Stylized Ears
    add_box(bm, Vector((cx - 0.15, cy - 0.01, cz + 0.02)), (0.025, 0.045, 0.07), skin_mat)
    add_box(bm, Vector((cx + 0.15, cy - 0.01, cz + 0.02)), (0.025, 0.045, 0.07), skin_mat)

    # 6. Volumetric Chunky Hair Tufts
    if hair_style == "messy":
        # Top volume
        add_box(bm, Vector((cx, cy - 0.02, cz + 0.17)), (0.30, 0.30, 0.08), hair_mat)
        # Angled front bangs projecting over brow
        add_box(bm, Vector((cx - 0.06, cy + 0.13, cz + 0.14)), (0.12, 0.06, 0.06), hair_mat, rot_z=math.radians(-12))
        add_box(bm, Vector((cx + 0.05, cy + 0.13, cz + 0.14)), (0.14, 0.06, 0.07), hair_mat, rot_z=math.radians(10))
        # Side tufts
        add_box(bm, Vector((cx - 0.15, cy - 0.02, cz + 0.10)), (0.03, 0.18, 0.12), hair_mat)
        add_box(bm, Vector((cx + 0.15, cy - 0.02, cz + 0.10)), (0.03, 0.18, 0.12), hair_mat)
        # Back nape
        add_box(bm, Vector((cx, cy - 0.14, cz + 0.04)), (0.28, 0.05, 0.16), hair_mat)
    elif hair_style == "buzzcut":
        add_box(bm, Vector((cx, cy - 0.01, cz + 0.165)), (0.29, 0.29, 0.04), hair_mat)
        add_box(bm, Vector((cx, cy - 0.14, cz + 0.05)), (0.27, 0.03, 0.14), hair_mat)

# ==============================================================================
# 4. HIGH-FIDELITY MODULAR CHARACTER ENGINE CORE
# ==============================================================================
def generate_character(bm, p_map, origin=(0, 0, 0), config=None):
    ox, oy, oz = origin
    cfg = config or {}

    body_type = cfg.get("body_type", "fit")
    skin_mat = p_map.get(cfg.get("skin_mat", "skin_fair"))
    hair_mat = p_map.get(cfg.get("hair_mat", "hair_dark"))
    hair_style = cfg.get("hair", "messy")

    # Dynamic Anatomical Scaling Matrix
    if body_type == "muscled":
        chest_w, chest_d = 0.62, 0.38
        waist_w, waist_d = 0.44, 0.30
        hip_w, hip_d     = 0.46, 0.32
        arm_top, arm_bot = 0.18, 0.13
        thigh_top, thigh_bot = 0.24, 0.18
        calf_top, calf_bot   = 0.20, 0.14
        pectoral_depth   = 0.07
        belly_depth      = 0.0
    elif body_type == "fat":
        chest_w, chest_d = 0.60, 0.42
        waist_w, waist_d = 0.64, 0.48
        hip_w, hip_d     = 0.54, 0.40
        arm_top, arm_bot = 0.16, 0.12
        thigh_top, thigh_bot = 0.25, 0.20
        calf_top, calf_bot   = 0.22, 0.16
        pectoral_depth   = 0.0
        belly_depth      = 0.14
    else: # "fit"
        chest_w, chest_d = 0.50, 0.30
        waist_w, waist_d = 0.38, 0.24
        hip_w, hip_d     = 0.40, 0.26
        arm_top, arm_bot = 0.14, 0.10
        thigh_top, thigh_bot = 0.20, 0.15
        calf_top, calf_bot   = 0.17, 0.12
        pectoral_depth   = 0.03
        belly_depth      = 0.0

    leg_spacing = hip_w * 0.40

    # -------------------------------------------------------------
    # 1. ANATOMICAL BASE MESH (Tapered Limbs, Pelvis & Torso)
    # -------------------------------------------------------------
    
    # A. Bare Feet (Beveled heel & toe curve)
    add_box(bm, Vector((ox - leg_spacing, oy + 0.05, oz + 0.06)), (thigh_bot * 0.85, 0.26, 0.12), skin_mat)
    add_box(bm, Vector((ox + leg_spacing, oy + 0.05, oz + 0.06)), (thigh_bot * 0.85, 0.26, 0.12), skin_mat)

    # B. Lower Legs / Calves (Tapered from knee down to ankle)
    add_tapered_frustum(bm, Vector((ox - leg_spacing, oy, oz + 0.26)),
                        size_top=(calf_top, calf_top), size_bottom=(calf_bot, calf_bot), height=0.28, mat_index=skin_mat)
    add_tapered_frustum(bm, Vector((ox + leg_spacing, oy, oz + 0.26)),
                        size_top=(calf_top, calf_top), size_bottom=(calf_bot, calf_bot), height=0.28, mat_index=skin_mat)

    # C. Upper Legs / Thighs (Tapered from hip down to knee)
    add_tapered_frustum(bm, Vector((ox - leg_spacing, oy, oz + 0.56)),
                        size_top=(thigh_top, thigh_top), size_bottom=(thigh_bot, thigh_bot), height=0.32, mat_index=skin_mat)
    add_tapered_frustum(bm, Vector((ox + leg_spacing, oy, oz + 0.56)),
                        size_top=(thigh_top, thigh_top), size_bottom=(thigh_bot, thigh_bot), height=0.32, mat_index=skin_mat)

    # D. Pelvis & Base Boxer Briefs (Underwear)
    add_tapered_frustum(bm, Vector((ox, oy, oz + 0.77)),
                        size_top=(waist_w + 0.02, waist_d + 0.02), size_bottom=(hip_w, hip_d), height=0.18, mat_index=p_map["underwear_black"])
    # White elastic waistband
    add_box(bm, Vector((ox, oy, oz + 0.865)), (waist_w + 0.03, waist_d + 0.03, 0.035), p_map["underwear_waistband"])

    # E. Tapered V-Torso (Abdomen & Chest)
    # Lower Abdomen (Waist)
    add_tapered_frustum(bm, Vector((ox, oy, oz + 0.98)),
                        size_top=(chest_w * 0.88, chest_d * 0.88), size_bottom=(waist_w, waist_d), height=0.22, mat_index=skin_mat)
    # Upper Chest & Ribcage (Flaring to broad shoulders)
    add_tapered_frustum(bm, Vector((ox, oy, oz + 1.20)),
                        size_top=(chest_w, chest_d), size_bottom=(chest_w * 0.88, chest_d * 0.88), height=0.24, mat_index=skin_mat)

    # Anatomical Chest Features
    if pectoral_depth > 0:
        # Pectoral muscle slabs catching top keylight
        add_box(bm, Vector((ox, oy + chest_d / 2.0 + pectoral_depth / 2.0 - 0.01, oz + 1.24)),
                (chest_w * 0.82, pectoral_depth, 0.16), skin_mat)
    if belly_depth > 0:
        # Soft protruding belly volume
        add_box(bm, Vector((ox, oy + waist_d / 2.0 + belly_depth / 2.0 - 0.01, oz + 0.96)),
                (waist_w * 0.86, belly_depth, 0.22), skin_mat)

    # F. Tapered Arms & Deltoid Shoulders
    arm_x = chest_w / 2.0 + arm_top / 2.0 + 0.01
    # Deltoid shoulder caps
    add_box(bm, Vector((ox - arm_x, oy, oz + 1.28)), (arm_top + 0.03, arm_top + 0.03, 0.12), skin_mat)
    add_box(bm, Vector((ox + arm_x, oy, oz + 1.28)), (arm_top + 0.03, arm_top + 0.03, 0.12), skin_mat)
    # Biceps (Upper Arm)
    add_tapered_frustum(bm, Vector((ox - arm_x, oy, oz + 1.12)),
                        size_top=(arm_top, arm_top), size_bottom=(arm_top * 0.85, arm_top * 0.85), height=0.22, mat_index=skin_mat)
    add_tapered_frustum(bm, Vector((ox + arm_x, oy, oz + 1.12)),
                        size_top=(arm_top, arm_top), size_bottom=(arm_top * 0.85, arm_top * 0.85), height=0.22, mat_index=skin_mat)
    # Forearms (Tapering to wrists)
    add_tapered_frustum(bm, Vector((ox - arm_x, oy, oz + 0.90)),
                        size_top=(arm_top * 0.88, arm_top * 0.88), size_bottom=(arm_bot, arm_bot), height=0.24, mat_index=skin_mat)
    add_tapered_frustum(bm, Vector((ox + arm_x, oy, oz + 0.90)),
                        size_top=(arm_top * 0.88, arm_top * 0.88), size_bottom=(arm_bot, arm_bot), height=0.24, mat_index=skin_mat)
    # Hands
    add_box(bm, Vector((ox - arm_x, oy, oz + 0.74)), (arm_bot * 0.9, 0.12, 0.10), skin_mat)
    add_box(bm, Vector((ox + arm_x, oy, oz + 0.74)), (arm_bot * 0.9, 0.12, 0.10), skin_mat)

    # G. Chiseled Head & Features
    add_cylinder(bm, Vector((ox, oy, oz + 1.36)), radius=0.07, depth=0.10, mat_index=skin_mat)
    add_chiseled_head(bm, (ox, oy, oz + 1.54), p_map, skin_mat, hair_mat, hair_style=hair_style)

    # -------------------------------------------------------------
    # 2. MODULAR WEARABLE EQUIPMENT LAYERS
    # -------------------------------------------------------------

    # A. FEET SLOT (Boots with Lugged Soles or Sneakers)
    feet_slot = cfg.get("feet", "none")
    if feet_slot == "boots":
        # Outsole rubber base
        add_box(bm, Vector((ox - leg_spacing, oy + 0.05, oz + 0.03)), (thigh_bot * 0.95, 0.30, 0.06), p_map["boots_sole"])
        add_box(bm, Vector((ox + leg_spacing, oy + 0.05, oz + 0.03)), (thigh_bot * 0.95, 0.30, 0.06), p_map["boots_sole"])
        # Leather boot body
        add_box(bm, Vector((ox - leg_spacing, oy + 0.05, oz + 0.11)), (thigh_bot * 0.90, 0.28, 0.14), p_map["boots_dark"])
        add_box(bm, Vector((ox + leg_spacing, oy + 0.05, oz + 0.11)), (thigh_bot * 0.90, 0.28, 0.14), p_map["boots_dark"])
        # Boot shaft collar (covering ankle)
        add_box(bm, Vector((ox - leg_spacing, oy, oz + 0.20)), (calf_bot + 0.05, calf_bot + 0.05, 0.12), p_map["boots_dark"])
        add_box(bm, Vector((ox + leg_spacing, oy, oz + 0.20)), (calf_bot + 0.05, calf_bot + 0.05, 0.12), p_map["boots_dark"])
    elif feet_slot == "sneakers":
        # White rubber sole
        add_box(bm, Vector((ox - leg_spacing, oy + 0.05, oz + 0.025)), (thigh_bot * 0.92, 0.28, 0.05), p_map["shoes_white"])
        add_box(bm, Vector((ox + leg_spacing, oy + 0.05, oz + 0.025)), (thigh_bot * 0.92, 0.28, 0.05), p_map["shoes_white"])
        # Sneaker upper
        add_box(bm, Vector((ox - leg_spacing, oy + 0.05, oz + 0.09)), (thigh_bot * 0.88, 0.26, 0.10), p_map["shoes_white"])
        add_box(bm, Vector((ox + leg_spacing, oy + 0.05, oz + 0.09)), (thigh_bot * 0.88, 0.26, 0.10), p_map["shoes_white"])

    # B. LEGS SLOT (Jeans, Cargo Pants, Combat Fatigues)
    legs_slot = cfg.get("legs", "none")
    if legs_slot in ["cargo", "jeans", "olive"]:
        pants_mat = p_map["pants_charcoal"] if legs_slot == "cargo" else (p_map["pants_bluejeans"] if legs_slot == "jeans" else p_map["pants_olive"])
        # Pelvis trouser cover
        add_tapered_frustum(bm, Vector((ox, oy, oz + 0.77)),
                            size_top=(waist_w + 0.04, waist_d + 0.04), size_bottom=(hip_w + 0.03, hip_d + 0.03), height=0.20, mat_index=pants_mat)
        # Thigh trouser tubes
        add_tapered_frustum(bm, Vector((ox - leg_spacing, oy, oz + 0.56)),
                            size_top=(thigh_top + 0.04, thigh_top + 0.04), size_bottom=(thigh_bot + 0.04, thigh_bot + 0.04), height=0.32, mat_index=pants_mat)
        add_tapered_frustum(bm, Vector((ox + leg_spacing, oy, oz + 0.56)),
                            size_top=(thigh_top + 0.04, thigh_top + 0.04), size_bottom=(thigh_bot + 0.04, thigh_bot + 0.04), height=0.32, mat_index=pants_mat)
        # Calf trouser tubes
        add_tapered_frustum(bm, Vector((ox - leg_spacing, oy, oz + 0.28)),
                            size_top=(calf_top + 0.04, calf_top + 0.04), size_bottom=(calf_bot + 0.05, calf_bot + 0.05), height=0.28, mat_index=pants_mat)
        add_tapered_frustum(bm, Vector((ox + leg_spacing, oy, oz + 0.28)),
                            size_top=(calf_top + 0.04, calf_top + 0.04), size_bottom=(calf_bot + 0.05, calf_bot + 0.05), height=0.28, mat_index=pants_mat)

        if legs_slot == "cargo":
            # Dimensional Cargo Side Pockets with flap lids
            add_box(bm, Vector((ox - (leg_spacing + thigh_top / 2.0 + 0.02), oy, oz + 0.52)), (0.04, 0.14, 0.16), pants_mat)
            add_box(bm, Vector((ox + (leg_spacing + thigh_top / 2.0 + 0.02), oy, oz + 0.52)), (0.04, 0.14, 0.16), pants_mat)
            # Flap lid
            add_box(bm, Vector((ox - (leg_spacing + thigh_top / 2.0 + 0.025), oy, oz + 0.59)), (0.05, 0.15, 0.03), p_map["pants_charcoal"])
            add_box(bm, Vector((ox + (leg_spacing + thigh_top / 2.0 + 0.025), oy, oz + 0.59)), (0.05, 0.15, 0.03), p_map["pants_charcoal"])

    # C. TORSO SLOT (Hoodie, Ballistic Vest, Flannel Shirt)
    torso_slot = cfg.get("torso", "none")
    if torso_slot == "hoodie":
        # Main Hoodie Body (Layered over chest & waist)
        add_tapered_frustum(bm, Vector((ox, oy, oz + 0.98)),
                            size_top=(chest_w * 0.92, chest_d * 0.92), size_bottom=(waist_w + 0.04, waist_d + 0.04), height=0.24, mat_index=p_map["hoodie_teal"])
        add_tapered_frustum(bm, Vector((ox, oy, oz + 1.20)),
                            size_top=(chest_w + 0.05, chest_d + 0.05), size_bottom=(chest_w * 0.92, chest_d * 0.92), height=0.26, mat_index=p_map["hoodie_teal"])
        # Popped Collar / Hood Ring around neck
        add_box(bm, Vector((ox, oy - 0.03, oz + 1.35)), (chest_w * 0.65, chest_d * 0.65, 0.10), p_map["hoodie_teal"])
        # White center zipper line
        add_box(bm, Vector((ox, oy + (chest_d + 0.05) / 2.0 + 0.005, oz + 1.08)), (0.035, 0.015, 0.44), p_map["shoes_white"])
        # Sleeves over arms
        add_box(bm, Vector((ox - arm_x, oy, oz + 1.10)), (arm_top + 0.04, arm_top + 0.04, 0.36), p_map["hoodie_teal"])
        add_box(bm, Vector((ox + arm_x, oy, oz + 1.10)), (arm_top + 0.04, arm_top + 0.04, 0.36), p_map["hoodie_teal"])
        # Sleeve cuffs
        add_box(bm, Vector((ox - arm_x, oy, oz + 0.90)), (arm_bot + 0.03, arm_bot + 0.03, 0.06), p_map["hoodie_teal"])
        add_box(bm, Vector((ox + arm_x, oy, oz + 0.90)), (arm_bot + 0.03, arm_bot + 0.03, 0.06), p_map["hoodie_teal"])

    elif torso_slot == "vest":
        # Undershirt
        add_box(bm, Vector((ox, oy, oz + 1.10)), (chest_w + 0.02, chest_d + 0.02, 0.48), p_map["pants_charcoal"])
        # Heavy Tactical Ballistic Vest (Thick front & back plates)
        add_box(bm, Vector((ox, oy + (chest_d + 0.02) / 2.0 + 0.035, oz + 1.14)), (chest_w * 0.88, 0.07, 0.40), p_map["vest_ochre"])
        add_box(bm, Vector((ox, oy - (chest_d + 0.02) / 2.0 - 0.035, oz + 1.14)), (chest_w * 0.88, 0.07, 0.40), p_map["vest_ochre"])
        # Shoulder Straps with metal clips
        add_box(bm, Vector((ox - 0.18, oy, oz + 1.34)), (0.09, chest_d + 0.08, 0.05), p_map["vest_ochre"])
        add_box(bm, Vector((ox + 0.18, oy, oz + 1.34)), (0.09, chest_d + 0.08, 0.05), p_map["vest_ochre"])
        add_box(bm, Vector((ox - 0.18, oy + (chest_d + 0.08) / 2.0 + 0.005, oz + 1.30)), (0.08, 0.015, 0.04), p_map["metal_buckle"])
        add_box(bm, Vector((ox + 0.18, oy + (chest_d + 0.08) / 2.0 + 0.005, oz + 1.30)), (0.08, 0.015, 0.04), p_map["metal_buckle"])
        # Front Mag Pouches with Flap Lids
        for px in [-0.14, 0.0, 0.14]:
            add_box(bm, Vector((ox + px, oy + (chest_d + 0.02) / 2.0 + 0.08, oz + 1.04)), (0.09, 0.06, 0.13), p_map["pants_olive"])
            add_box(bm, Vector((ox + px, oy + (chest_d + 0.02) / 2.0 + 0.085, oz + 1.10)), (0.095, 0.065, 0.03), p_map["pants_charcoal"])

    elif torso_slot == "flannel":
        # Rugged Lumberjack Flannel Shirt
        add_tapered_frustum(bm, Vector((ox, oy, oz + 0.98)),
                            size_top=(chest_w * 0.92, chest_d * 0.92), size_bottom=(waist_w + 0.04, waist_d + 0.04), height=0.24, mat_index=p_map["shirt_flannel"])
        add_tapered_frustum(bm, Vector((ox, oy, oz + 1.20)),
                            size_top=(chest_w + 0.04, chest_d + 0.04), size_bottom=(chest_w * 0.92, chest_d * 0.92), height=0.26, mat_index=p_map["shirt_flannel"])
        # Flannel collar
        add_box(bm, Vector((ox, oy - 0.02, oz + 1.34)), (chest_w * 0.60, chest_d * 0.60, 0.06), p_map["shirt_flannel"])
        # Sleeves
        add_box(bm, Vector((ox - arm_x, oy, oz + 1.10)), (arm_top + 0.03, arm_top + 0.03, 0.36), p_map["shirt_flannel"])
        add_box(bm, Vector((ox + arm_x, oy, oz + 1.10)), (arm_top + 0.03, arm_top + 0.03, 0.36), p_map["shirt_flannel"])

    # D. HEAD / HAT SLOT
    hat_slot = cfg.get("hat", "none")
    if hat_slot == "beanie":
        add_cylinder(bm, Vector((ox, oy, oz + 1.66)), radius=0.17, depth=0.14, mat_index=p_map["beanie_red"])
        # Knit roll brim
        add_cylinder(bm, Vector((ox, oy, oz + 1.60)), radius=0.18, depth=0.06, mat_index=p_map["beanie_red"])
        # Pom pom on top
        add_cylinder(bm, Vector((ox, oy, oz + 1.76)), radius=0.04, depth=0.06, mat_index=p_map["beanie_red"])
    elif hat_slot == "cap":
        # Tactical Cap crown
        add_cylinder(bm, Vector((ox, oy - 0.02, oz + 1.65)), radius=0.16, depth=0.12, mat_index=p_map["cap_tan"])
        # Forward projecting curved visor brim (+Y)
        add_box(bm, Vector((ox, oy + 0.17, oz + 1.60)), (0.24, 0.14, 0.025), p_map["cap_tan"], rot_x=math.radians(10))

    # E. BACK SLOT
    back_slot = cfg.get("back", "none")
    if back_slot == "backpack":
        # Hiker pack with nylon straps
        add_box(bm, Vector((ox, oy - (chest_d / 2.0 + 0.11), oz + 1.14)), (chest_w * 0.72, 0.18, 0.44), p_map["vest_ochre"])
        # Bedroll strapped on top
        add_cylinder(bm, Vector((ox, oy - (chest_d / 2.0 + 0.11), oz + 1.40)), radius=0.09, depth=chest_w * 0.85, mat_index=p_map["pants_olive"], rot_y=math.radians(90))
        # Black nylon tie-down straps
        add_cylinder(bm, Vector((ox - 0.12, oy - (chest_d / 2.0 + 0.11), oz + 1.40)), radius=0.095, depth=0.03, mat_index=p_map["underwear_black"], rot_y=math.radians(90))
        add_cylinder(bm, Vector((ox + 0.12, oy - (chest_d / 2.0 + 0.11), oz + 1.40)), radius=0.095, depth=0.03, mat_index=p_map["underwear_black"], rot_y=math.radians(90))
    elif back_slot == "radio":
        # Tactical radio pack with long flexible antenna
        add_box(bm, Vector((ox - 0.18, oy - (chest_d / 2.0 + 0.06), oz + 1.28)), (0.09, 0.09, 0.16), p_map["metal_weapon"])
        add_cylinder(bm, Vector((ox - 0.18, oy - (chest_d / 2.0 + 0.06), oz + 1.48)), radius=0.007, depth=0.24, mat_index=p_map["metal_weapon"])

    # F. WEAPON / HAND SLOT
    weapon_slot = cfg.get("weapon", "none")
    if weapon_slot == "flashlight":
        # Tactical flashlight in right hand casting warm night beam
        add_cylinder(bm, Vector((ox + arm_x, oy + 0.28, oz + 0.86)), radius=0.035, depth=0.22, mat_index=p_map["metal_weapon"], rot_x=math.radians(90))
        add_cylinder(bm, Vector((ox + arm_x, oy + 0.39, oz + 0.86)), radius=0.05, depth=0.03, mat_index=p_map["glow_flashlight"], rot_x=math.radians(90))
    elif weapon_slot == "smg":
        # Nailgun SMG held across chest
        add_box(bm, Vector((ox + 0.06, oy + 0.30, oz + 0.98)), (0.10, 0.42, 0.16), p_map["metal_weapon"])
        add_cylinder(bm, Vector((ox + 0.06, oy + 0.54, oz + 1.02)), radius=0.02, depth=0.18, mat_index=p_map["metal_weapon"], rot_x=math.radians(90))
        add_box(bm, Vector((ox + 0.06, oy + 0.28, oz + 0.84)), (0.06, 0.10, 0.18), p_map["wood_stock"])
    elif weapon_slot == "axe":
        # Survival wood-chopping axe held upright
        add_box(bm, Vector((ox + arm_x, oy + 0.12, oz + 0.88)), (0.045, 0.045, 0.72), p_map["wood_stock"])
        add_box(bm, Vector((ox + arm_x, oy + 0.20, oz + 1.20)), (0.04, 0.20, 0.16), p_map["metal_weapon"])

# ==============================================================================
# 5. SHOWCASE RENDER: 4 HIGH-FIDELITY ARCHETYPES SIDE-BY-SIDE
# ==============================================================================
def render_character_engine_showcase():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    palette = create_character_engine_palette()
    mat_keys = list(palette.keys())
    p_map = {k: i for i, k in enumerate(mat_keys)}
    mat_list = [palette[k] for k in mat_keys]

    mesh = bpy.data.meshes.new("CharacterEngine_Showcase_Mesh")
    bm = bmesh.new()

    # 1. Base Mesh in Underwear (Fit physique, bare chest & underwear)
    cfg_base = {
        "body_type": "fit",
        "skin_mat": "skin_fair",
        "hair_mat": "hair_dark",
        "hair": "messy",
        "torso": "none", # Bare chest!
        "legs": "none",  # Underwear only!
        "feet": "none",  # Barefoot!
        "hat": "none",
        "back": "none",
        "weapon": "none"
    }
    generate_character(bm, p_map, origin=(-2.4, 0, 0), config=cfg_base)

    # 2. Fit Survivor ("Jax")
    cfg_survivor = {
        "body_type": "fit",
        "skin_mat": "skin_fair",
        "hair": "buzzcut",
        "torso": "hoodie",
        "legs": "cargo",
        "feet": "sneakers",
        "hat": "none",
        "back": "backpack",
        "weapon": "flashlight"
    }
    generate_character(bm, p_map, origin=(-0.8, 0, 0), config=cfg_survivor)

    # 3. Muscled Heavy Operator ("Titan")
    cfg_muscled = {
        "body_type": "muscled",
        "skin_mat": "skin_tan",
        "hair": "buzzcut",
        "torso": "vest",
        "legs": "olive",
        "feet": "boots",
        "hat": "cap",
        "back": "radio",
        "weapon": "smg"
    }
    generate_character(bm, p_map, origin=(0.8, 0, 0), config=cfg_muscled)

    # 4. Stout / Fat Scavenger ("Gus")
    cfg_fat = {
        "body_type": "fat",
        "skin_mat": "skin_fair",
        "hair": "messy",
        "torso": "flannel",
        "legs": "jeans",
        "feet": "boots",
        "hat": "beanie",
        "back": "none",
        "weapon": "axe"
    }
    generate_character(bm, p_map, origin=(2.4, 0, 0), config=cfg_fat)

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("CharacterEngine_Showcase_Group", mesh)
    bpy.context.collection.objects.link(obj)
    for m in mat_list:
        obj.data.materials.append(m)

    # Add Vinyl Specular Bevel Modifier
    bev = obj.modifiers.new(name="Bevel", type='BEVEL')
    bev.width = 0.012
    bev.segments = 2
    bev.limit_method = 'ANGLE'
    bev.angle_limit = math.radians(38)

    # Studio Pedestal Stage
    bg_mesh = bpy.data.meshes.new("EnginePedestal_Mesh")
    bm_bg = bmesh.new()
    add_box(bm_bg, Vector((0, 0, -0.06)), (14.0, 7.0, 0.12), p_map["pedestal_dark"])
    add_box(bm_bg, Vector((0, 0, 0.005)), (12.0, 0.04, 0.01), p_map["pedestal_dark"])
    bm_bg.to_mesh(bg_mesh)
    bm_bg.free()
    bg_obj = bpy.data.objects.new("StudioPedestal", bg_mesh)
    bg_obj.data.materials.append(palette["pedestal_dark"])
    bpy.context.collection.objects.link(bg_obj)

    # Camera looking at center of the 4 figures
    cam_target = bpy.data.objects.new("Target", None)
    cam_target.location = (0, 0, 0.95)
    bpy.context.collection.objects.link(cam_target)

    cam_data = bpy.data.cameras.new("EngineCam")
    cam_data.type = 'PERSP'
    cam_data.lens = 36.0
    cam_obj = bpy.data.objects.new("EngineCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = (0.0, 6.6, 1.7)
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = cam_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam_obj

    # Studio Lighting (Front Key + Rim + Soft Fill)
    key = bpy.data.lights.new("Key", type='AREA')
    key.energy = 220.0
    key.size = 4.0
    key.color = (1.0, 0.97, 0.94)
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.location = (3.5, 4.5, 3.5)
    bpy.context.collection.objects.link(key_obj)

    rim = bpy.data.lights.new("Rim", type='AREA')
    rim.energy = 180.0
    rim.size = 3.5
    rim.color = (0.45, 0.75, 1.0)
    rim_obj = bpy.data.objects.new("Rim", rim)
    rim_obj.location = (-3.5, -4.0, 3.2)
    bpy.context.collection.objects.link(rim_obj)

    fill = bpy.data.lights.new("Fill", type='AREA')
    fill.energy = 75.0
    fill.size = 5.0
    fill.color = (0.85, 0.90, 0.95)
    fill_obj = bpy.data.objects.new("Fill", fill)
    fill_obj.location = (-4.0, 4.0, 2.0)
    bpy.context.collection.objects.link(fill_obj)

    # Cycles CPU Setup
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 768
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.join(script_dir, "character_engine_showcase_preview.png")
    out_blend = os.path.join(script_dir, "character_engine_showcase.blend")
    out_glb = os.path.join(script_dir, "character_engine_showcase.glb")

    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB', use_selection=False)

    print(f"ENGINE_UPGRADE_SUCCESS: Image={out_img}, Blend={out_blend}, GLB={out_glb}")

if __name__ == "__main__":
    render_character_engine_showcase()
