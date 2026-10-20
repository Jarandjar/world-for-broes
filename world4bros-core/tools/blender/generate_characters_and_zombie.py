import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# ==============================================================================
# 1. MATERIALS FOR CHARACTERS & ZOMBIE
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

def create_character_palette():
    return {
        "skin_hero": get_or_create_material("Mat_Skin_Hero", (0.88, 0.68, 0.52, 1.0), roughness=0.6),
        "skin_zombie": get_or_create_material("Mat_Skin_Zombie", (0.42, 0.52, 0.38, 1.0), roughness=0.7),
        "jacket_teal": get_or_create_material("Mat_Jacket_Teal", (0.14, 0.35, 0.42, 1.0), roughness=0.5),
        "vest_ochre": get_or_create_material("Mat_Vest_Ochre", (0.78, 0.48, 0.20, 1.0), roughness=0.6),
        "pants_charcoal": get_or_create_material("Mat_Pants_Charcoal", (0.15, 0.16, 0.18, 1.0), roughness=0.6),
        "pants_olive": get_or_create_material("Mat_Pants_Olive", (0.28, 0.32, 0.22, 1.0), roughness=0.6),
        "clothes_tattered": get_or_create_material("Mat_Clothes_Tattered", (0.22, 0.20, 0.22, 1.0), roughness=0.8),
        "shoes_white": get_or_create_material("Mat_Shoes_White", (0.92, 0.92, 0.94, 1.0), roughness=0.4),
        "boots_dark": get_or_create_material("Mat_Boots_Dark", (0.10, 0.09, 0.08, 1.0), roughness=0.5),
        "beanie_red": get_or_create_material("Mat_Beanie_Red", (0.68, 0.18, 0.15, 1.0), roughness=0.6),
        "hair_dark": get_or_create_material("Mat_Hair_Dark", (0.12, 0.10, 0.08, 1.0), roughness=0.7),
        "metal_weapon": get_or_create_material("Mat_Metal_Weapon", (0.12, 0.12, 0.14, 1.0), roughness=0.3, metallic=0.85),
        "wood_stock": get_or_create_material("Mat_Wood_Stock", (0.45, 0.26, 0.14, 1.0), roughness=0.5),
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
# 3. CHARACTER BUILDERS (Standard height ~1.8m)
# ==============================================================================

# Character 1: The Survivor ("Jax")
# - Petrol Teal Hooded Jacket
# - Backpack with Bedroll
# - Dark Cargo Pants, White Sneakers
# - Tactical Flashlight
def build_character_survivor(bm, p_map, origin=(0, 0, 0)):
    ox, oy, oz = origin

    # Feet / Sneakers
    add_box(bm, Vector((ox - 0.16, oy + 0.04, oz + 0.08)), (0.16, 0.32, 0.14), mat_index=p_map["shoes_white"])
    add_box(bm, Vector((ox + 0.16, oy + 0.04, oz + 0.08)), (0.16, 0.32, 0.14), mat_index=p_map["shoes_white"])

    # Legs (Cargo Pants)
    add_box(bm, Vector((ox - 0.16, oy, oz + 0.44)), (0.18, 0.20, 0.60), mat_index=p_map["pants_charcoal"])
    add_box(bm, Vector((ox + 0.16, oy, oz + 0.44)), (0.18, 0.20, 0.60), mat_index=p_map["pants_charcoal"])
    # Knee pouches
    add_box(bm, Vector((ox - 0.16, oy + 0.11, oz + 0.42)), (0.12, 0.04, 0.14), mat_index=p_map["jacket_teal"])
    add_box(bm, Vector((ox + 0.16, oy + 0.11, oz + 0.42)), (0.12, 0.04, 0.14), mat_index=p_map["jacket_teal"])

    # Torso (Hooded Teal Jacket)
    add_box(bm, Vector((ox, oy, oz + 1.05)), (0.50, 0.30, 0.62), mat_index=p_map["jacket_teal"])
    # White zipper line
    add_box(bm, Vector((ox, oy + 0.155, oz + 1.05)), (0.04, 0.015, 0.58), mat_index=p_map["shoes_white"])

    # Backpack & Bedroll (Back)
    add_box(bm, Vector((ox, oy - 0.22, oz + 1.08)), (0.38, 0.16, 0.44), mat_index=p_map["vest_ochre"])
    add_cylinder(bm, Vector((ox, oy - 0.22, oz + 1.34)), radius=0.09, depth=0.42, mat_index=p_map["pants_olive"], rot_y=math.radians(90))

    # Left Arm (Relaxed)
    add_box(bm, Vector((ox - 0.34, oy, oz + 1.00)), (0.14, 0.16, 0.52), mat_index=p_map["jacket_teal"])
    add_box(bm, Vector((ox - 0.34, oy, oz + 0.70)), (0.11, 0.13, 0.14), mat_index=p_map["skin_hero"]) # hand

    # Right Arm (Raised forward holding Tactical Flashlight)
    add_box(bm, Vector((ox + 0.34, oy + 0.14, oz + 1.06)), (0.14, 0.36, 0.16), mat_index=p_map["jacket_teal"], rot_x=math.radians(25))
    add_box(bm, Vector((ox + 0.34, oy + 0.34, oz + 1.15)), (0.11, 0.13, 0.14), mat_index=p_map["skin_hero"]) # hand
    # Flashlight body & glowing lens
    add_cylinder(bm, Vector((ox + 0.34, oy + 0.44, oz + 1.17)), radius=0.035, depth=0.22, mat_index=p_map["metal_weapon"], rot_x=math.radians(90))
    add_cylinder(bm, Vector((ox + 0.34, oy + 0.55, oz + 1.17)), radius=0.05, depth=0.03, mat_index=p_map["glow_flashlight"], rot_x=math.radians(90))

    # Head & Hair
    add_box(bm, Vector((ox, oy, oz + 1.52)), (0.28, 0.28, 0.30), mat_index=p_map["skin_hero"])
    # Styled hair / hood trim
    add_box(bm, Vector((ox, oy - 0.02, oz + 1.66)), (0.30, 0.30, 0.10), mat_index=p_map["hair_dark"])
    add_box(bm, Vector((ox, oy - 0.12, oz + 1.54)), (0.30, 0.10, 0.26), mat_index=p_map["hair_dark"])


# Character 2: The Scavenger / Operator ("Rook")
# - Ochre Tactical Ballistic Vest & Beanie
# - Olive Pants & Heavy Combat Boots
# - Holding Nailgun SMG
def build_character_scavenger(bm, p_map, origin=(0, 0, 0)):
    ox, oy, oz = origin

    # Combat Boots (Dark)
    add_box(bm, Vector((ox - 0.18, oy + 0.04, oz + 0.09)), (0.18, 0.34, 0.18), mat_index=p_map["boots_dark"])
    add_box(bm, Vector((ox + 0.18, oy + 0.04, oz + 0.09)), (0.18, 0.34, 0.18), mat_index=p_map["boots_dark"])

    # Legs (Olive Heavy Duty)
    add_box(bm, Vector((ox - 0.18, oy, oz + 0.45)), (0.20, 0.22, 0.60), mat_index=p_map["pants_olive"])
    add_box(bm, Vector((ox + 0.18, oy, oz + 0.45)), (0.20, 0.22, 0.60), mat_index=p_map["pants_olive"])

    # Torso (Shirt + Heavy Ochre Tactical Vest)
    add_box(bm, Vector((ox, oy, oz + 1.05)), (0.52, 0.32, 0.60), mat_index=p_map["pants_charcoal"]) # under shirt
    add_box(bm, Vector((ox, oy, oz + 1.08)), (0.54, 0.36, 0.48), mat_index=p_map["vest_ochre"]) # vest
    # Ammo Pouches on chest
    add_box(bm, Vector((ox - 0.14, oy + 0.19, oz + 1.02)), (0.10, 0.06, 0.14), mat_index=p_map["pants_olive"])
    add_box(bm, Vector((ox, oy + 0.19, oz + 1.02)), (0.10, 0.06, 0.14), mat_index=p_map["pants_olive"])
    add_box(bm, Vector((ox + 0.14, oy + 0.19, oz + 1.02)), (0.10, 0.06, 0.14), mat_index=p_map["pants_olive"])
    # Shoulder Radio with antenna
    add_box(bm, Vector((ox - 0.22, oy - 0.05, oz + 1.34)), (0.07, 0.08, 0.12), mat_index=p_map["metal_weapon"])
    add_cylinder(bm, Vector((ox - 0.22, oy - 0.05, oz + 1.48)), radius=0.008, depth=0.18, mat_index=p_map["metal_weapon"])

    # Arms holding SMG in ready-low posture
    add_box(bm, Vector((ox - 0.34, oy + 0.10, oz + 1.02)), (0.15, 0.28, 0.16), mat_index=p_map["pants_charcoal"], rot_x=math.radians(30))
    add_box(bm, Vector((ox + 0.34, oy + 0.10, oz + 1.02)), (0.15, 0.28, 0.16), mat_index=p_map["pants_charcoal"], rot_x=math.radians(30))

    # Nailgun SMG weapon
    add_box(bm, Vector((ox + 0.06, oy + 0.30, oz + 1.00)), (0.10, 0.42, 0.16), mat_index=p_map["metal_weapon"])
    add_cylinder(bm, Vector((ox + 0.06, oy + 0.54, oz + 1.04)), radius=0.02, depth=0.18, mat_index=p_map["metal_weapon"], rot_x=math.radians(90))
    add_box(bm, Vector((ox + 0.06, oy + 0.28, oz + 0.86)), (0.06, 0.10, 0.18), mat_index=p_map["wood_stock"]) # magazine/grip

    # Head & Red Beanie
    add_box(bm, Vector((ox, oy, oz + 1.50)), (0.28, 0.28, 0.28), mat_index=p_map["skin_hero"])
    # Red Beanie Cap
    add_box(bm, Vector((ox, oy, oz + 1.64)), (0.30, 0.30, 0.14), mat_index=p_map["beanie_red"])
    add_cylinder(bm, Vector((ox, oy, oz + 1.72)), radius=0.04, depth=0.06, mat_index=p_map["beanie_red"])


# Character 3: The Infected Walker (Zombie)
# - Sickly Greenish Skin Tone
# - Tattered Dark Garments
# - Shambling Hunched Forward Stance
# - Glowing Sickly Yellow/Amber Eyes
def build_character_zombie(bm, p_map, origin=(0, 0, 0)):
    ox, oy, oz = origin

    # Hunched forward offset
    # Bare / Shambling feet
    add_box(bm, Vector((ox - 0.18, oy - 0.05, oz + 0.06)), (0.14, 0.28, 0.12), mat_index=p_map["skin_zombie"])
    add_box(bm, Vector((ox + 0.16, oy + 0.10, oz + 0.06)), (0.14, 0.28, 0.12), mat_index=p_map["skin_zombie"])

    # Ragged Ripped Trousers
    add_box(bm, Vector((ox - 0.18, oy - 0.02, oz + 0.40)), (0.18, 0.20, 0.55), mat_index=p_map["clothes_tattered"])
    add_box(bm, Vector((ox + 0.16, oy + 0.06, oz + 0.40)), (0.18, 0.20, 0.55), mat_index=p_map["clothes_tattered"])
    # Exposed decaying knee
    add_box(bm, Vector((ox - 0.18, oy + 0.08, oz + 0.38)), (0.10, 0.04, 0.10), mat_index=p_map["skin_zombie"])

    # Hunched Torso (Angled forward by 18 degrees)
    add_box(bm, Vector((ox, oy + 0.14, oz + 0.98)), (0.48, 0.28, 0.58), mat_index=p_map["clothes_tattered"], rot_x=math.radians(-18))

    # Outstretched Shambling Claws / Arms
    # Left Arm outstretched
    add_box(bm, Vector((ox - 0.32, oy + 0.34, oz + 1.08)), (0.13, 0.44, 0.13), mat_index=p_map["clothes_tattered"], rot_x=math.radians(-50))
    add_box(bm, Vector((ox - 0.32, oy + 0.58, oz + 1.22)), (0.11, 0.14, 0.10), mat_index=p_map["skin_zombie"]) # claw
    # Right Arm reaching out
    add_box(bm, Vector((ox + 0.32, oy + 0.30, oz + 1.00)), (0.13, 0.42, 0.13), mat_index=p_map["clothes_tattered"], rot_x=math.radians(-40))
    add_box(bm, Vector((ox + 0.32, oy + 0.54, oz + 1.12)), (0.11, 0.14, 0.10), mat_index=p_map["skin_zombie"]) # claw

    # Drooping / Twitching Head
    add_box(bm, Vector((ox, oy + 0.30, oz + 1.40)), (0.26, 0.26, 0.26), mat_index=p_map["skin_zombie"], rot_z=math.radians(12), rot_x=math.radians(-12))
    # Glowing Hollow Amber Eyes (emissive night hook!)
    add_box(bm, Vector((ox - 0.07, oy + 0.44, oz + 1.44)), (0.04, 0.02, 0.04), mat_index=p_map["glow_eyes"])
    add_box(bm, Vector((ox + 0.07, oy + 0.44, oz + 1.44)), (0.04, 0.02, 0.04), mat_index=p_map["glow_eyes"])
    # Disheveled dark hair strands
    add_box(bm, Vector((ox, oy + 0.28, oz + 1.54)), (0.28, 0.28, 0.08), mat_index=p_map["hair_dark"])


# ==============================================================================
# 4. ARENA PROPS CATALOG (Crate, Sandbags, Barrel, Low Wall, Medkit)
# ==============================================================================
def build_prop_crate(bm, p_map, origin=(0, 0, 0)):
    ox, oy, oz = origin
    # 1.2m cube wooden crate with dark metal brackets
    add_box(bm, Vector((ox, oy, oz + 0.6)), (1.2, 1.2, 1.2), mat_index=p_map["wood_stock"])
    # Metal corner brackets & strapping
    add_box(bm, Vector((ox, oy, oz + 0.6)), (1.22, 0.16, 1.22), mat_index=p_map["metal_weapon"])
    add_box(bm, Vector((ox, oy, oz + 0.6)), (0.16, 1.22, 1.22), mat_index=p_map["metal_weapon"])

def build_prop_hazard_barrel(bm, p_map, origin=(0, 0, 0)):
    ox, oy, oz = origin
    # Steel hazard barrel in red with yellow stripe
    add_cylinder(bm, Vector((ox, oy, oz + 0.55)), radius=0.34, depth=1.1, mat_index=p_map["beanie_red"])
    # Middle hazard warning band
    add_cylinder(bm, Vector((ox, oy, oz + 0.55)), radius=0.35, depth=0.22, mat_index=p_map["glow_eyes"])
    # Dark rim rings
    add_cylinder(bm, Vector((ox, oy, oz + 1.05)), radius=0.36, depth=0.06, mat_index=p_map["metal_weapon"])
    add_cylinder(bm, Vector((ox, oy, oz + 0.05)), radius=0.36, depth=0.06, mat_index=p_map["metal_weapon"])

def build_prop_medkit(bm, p_map, origin=(0, 0, 0)):
    ox, oy, oz = origin
    # White trauma case with glowing red/amber cross
    add_box(bm, Vector((ox, oy, oz + 0.18)), (0.48, 0.36, 0.24), mat_index=p_map["shoes_white"])
    # Red cross
    add_box(bm, Vector((ox, oy, oz + 0.305)), (0.20, 0.06, 0.015), mat_index=p_map["beanie_red"])
    add_box(bm, Vector((ox, oy, oz + 0.305)), (0.06, 0.20, 0.015), mat_index=p_map["beanie_red"])
    # Metal latches & handle
    add_box(bm, Vector((ox, oy + 0.20, oz + 0.18)), (0.16, 0.04, 0.06), mat_index=p_map["metal_weapon"])


# ==============================================================================
# 5. LINEUP SHOWCASE SCENE (Renders the 3 figures side-by-side with props!)
# ==============================================================================
def render_character_and_catalog_lineup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    palette = create_character_palette()

    mat_keys = list(palette.keys())
    mat_list = [palette[k] for k in mat_keys]
    p_map = {k: i for i, k in enumerate(mat_keys)}

    mesh = bpy.data.meshes.new("CharacterLineup_Mesh")
    bm = bmesh.new()

    # 1. Place Jax (Survivor) at X = -1.6
    build_character_survivor(bm, p_map, origin=(-1.6, 0, 0))

    # 2. Place Rook (Scavenger) at X = 0.0
    build_character_scavenger(bm, p_map, origin=(0.0, 0, 0))

    # 3. Place Zombie (Infected Walker) at X = +1.6
    build_character_zombie(bm, p_map, origin=(1.6, 0, 0))

    # 4. Place Key Arena Props cleanly to the flanks (no occlusion)
    # Military Crate to the far left
    build_prop_crate(bm, p_map, origin=(-3.2, 0.2, 0))
    # Medkit on top of the crate
    build_prop_medkit(bm, p_map, origin=(-3.2, 0.2, 1.2))
    # Red Hazard Barrel to the far right
    build_prop_hazard_barrel(bm, p_map, origin=(3.2, 0.2, 0))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("Character_Lineup_Group", mesh)
    bpy.context.collection.objects.link(obj)

    for m in mat_list:
        obj.data.materials.append(m)

    # Studio Floor / Pedestal
    bg_mesh = bpy.data.meshes.new("Pedestal_Mesh")
    bm_bg = bmesh.new()
    mat_floor = get_or_create_material("Mat_Pedestal_Floor", (0.08, 0.10, 0.14, 1.0), roughness=0.6)
    add_box(bm_bg, Vector((0, 0, -0.06)), (12.0, 8.0, 0.12), mat_index=0)
    # Floor guide lines
    add_box(bm_bg, Vector((0, 0, 0.005)), (10.0, 0.03, 0.01), mat_index=0)
    bm_bg.to_mesh(bg_mesh)
    bm_bg.free()
    bg_obj = bpy.data.objects.new("Studio_Pedestal", bg_mesh)
    bg_obj.data.materials.append(mat_floor)
    bpy.context.collection.objects.link(bg_obj)

    # Camera looking at front face of lineup (center at 0, 0, 1.0)
    cam_target = bpy.data.objects.new("LineupTarget", None)
    cam_target.location = (0, 0, 0.95)
    bpy.context.collection.objects.link(cam_target)

    cam_data = bpy.data.cameras.new("LineupCamera")
    cam_data.type = 'PERSP'
    cam_data.lens = 34.0
    cam_obj = bpy.data.objects.new("LineupCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Front 3/4 camera position (+Y side looking at front of characters)
    cam_obj.location = (0.0, 6.2, 1.7)
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = cam_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam_obj

    # Studio Lighting Setup (Front Key + Rim)
    # Front-Right Key Light
    key = bpy.data.lights.new("LineupKey", type='AREA')
    key.energy = 180.0
    key.size = 3.5
    key.color = (1.0, 0.97, 0.94)
    key_obj = bpy.data.objects.new("LineupKey", key)
    key_obj.location = (2.5, 4.0, 3.2)
    bpy.context.collection.objects.link(key_obj)

    # Back Rim Light (Separates dark clothes from backdrop)
    rim = bpy.data.lights.new("LineupRim", type='AREA')
    rim.energy = 150.0
    rim.size = 3.0
    rim.color = (0.45, 0.75, 1.0)
    rim_obj = bpy.data.objects.new("LineupRim", rim)
    rim_obj.location = (-2.5, -3.5, 3.0)
    bpy.context.collection.objects.link(rim_obj)

    # Front-Left Soft Fill
    fill = bpy.data.lights.new("LineupFill", type='AREA')
    fill.energy = 60.0
    fill.size = 4.0
    fill.color = (0.8, 0.85, 0.95)
    fill_obj = bpy.data.objects.new("LineupFill", fill)
    fill_obj.location = (-3.0, 3.5, 1.8)
    bpy.context.collection.objects.link(fill_obj)

    # Cycles CPU Setup
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.join(script_dir, "character_catalog_lineup_preview.png")
    out_blend = os.path.join(script_dir, "character_catalog_lineup.blend")
    out_glb = os.path.join(script_dir, "character_catalog_lineup.glb")

    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB', use_selection=False)

    print(f"LINEUP_RENDER_SUCCESS: Image={out_img}, Blend={out_blend}, GLB={out_glb}")

if __name__ == "__main__":
    render_character_and_catalog_lineup()
