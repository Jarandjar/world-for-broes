import bpy
import bmesh
import math
import os
import random
from mathutils import Vector, Matrix

# ==============================================================================
# 1. MATERIALS SYSTEM
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
        "concrete": get_or_create_material("Mat_Concrete", (0.55, 0.56, 0.58, 1.0), roughness=0.7),
        "concrete_inner": get_or_create_material("Mat_Concrete_Inner", (0.35, 0.35, 0.37, 1.0), roughness=0.9),
        "rebar": get_or_create_material("Mat_Rebar", (0.18, 0.15, 0.14, 1.0), roughness=0.5, metallic=0.9),
        "wood": get_or_create_material("Mat_Wood", (0.42, 0.28, 0.16, 1.0), roughness=0.6),
        "sandbag": get_or_create_material("Mat_Sandbag", (0.64, 0.54, 0.38, 1.0), roughness=0.9),
        "metal_dark": get_or_create_material("Mat_Metal_Dark", (0.12, 0.13, 0.15, 1.0), roughness=0.35, metallic=0.85),
        "hazard_red": get_or_create_material("Mat_Hazard_Red", (0.78, 0.18, 0.14, 1.0), roughness=0.4),
        "hazard_yellow": get_or_create_material("Mat_Hazard_Yellow", (0.95, 0.82, 0.20, 1.0), roughness=0.3, emission_color=(1.0, 0.85, 0.2, 1.0), emission_strength=3.0),
        "tree_leaves": get_or_create_material("Mat_Tree_Leaves", (0.12, 0.26, 0.18, 1.0), roughness=0.7),
        "tree_trunk": get_or_create_material("Mat_Tree_Trunk", (0.28, 0.18, 0.12, 1.0), roughness=0.8),
        "white": get_or_create_material("Mat_White", (0.92, 0.92, 0.94, 1.0), roughness=0.4),
        "skin_hero": get_or_create_material("Mat_Skin_Hero", (0.88, 0.68, 0.52, 1.0), roughness=0.6),
        "skin_zombie": get_or_create_material("Mat_Skin_Zombie", (0.42, 0.52, 0.38, 1.0), roughness=0.7),
        "jacket_teal": get_or_create_material("Mat_Jacket_Teal", (0.14, 0.35, 0.42, 1.0), roughness=0.5),
        "vest_ochre": get_or_create_material("Mat_Vest_Ochre", (0.78, 0.48, 0.20, 1.0), roughness=0.6),
        "pants_charcoal": get_or_create_material("Mat_Pants_Charcoal", (0.15, 0.16, 0.18, 1.0), roughness=0.6),
        "pants_olive": get_or_create_material("Mat_Pants_Olive", (0.28, 0.32, 0.22, 1.0), roughness=0.6),
        "clothes_tattered": get_or_create_material("Mat_Clothes_Tattered", (0.22, 0.20, 0.22, 1.0), roughness=0.8),
        "glow_eyes": get_or_create_material("Mat_Glow_Eyes", (0.95, 0.85, 0.3, 1.0), roughness=0.1, emission_color=(1.0, 0.85, 0.2, 1.0), emission_strength=5.0),
        "glow_flashlight": get_or_create_material("Mat_Glow_Flashlight", (1.0, 0.98, 0.8, 1.0), roughness=0.1, emission_color=(1.0, 0.95, 0.7, 1.0), emission_strength=8.0),
        "beanie_red": get_or_create_material("Mat_Beanie_Red", (0.68, 0.18, 0.15, 1.0), roughness=0.6),
        "pedestal": get_or_create_material("Mat_Pedestal", (0.07, 0.09, 0.12, 1.0), roughness=0.5),
    }

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
# 2. INDIVIDUAL ITEM BUILDERS
# ==============================================================================
def build_item_crate(bm, p):
    add_box(bm, Vector((0, 0, 0.6)), (1.2, 1.2, 1.2), p["wood"])
    add_box(bm, Vector((0, 0, 0.6)), (1.22, 0.18, 1.22), p["metal_dark"])
    add_box(bm, Vector((0, 0, 0.6)), (0.18, 1.22, 1.22), p["metal_dark"])

def build_item_sandbags(bm, p):
    for i in [-0.7, 0.0, 0.7]:
        add_box(bm, Vector((i, 0, 0.20)), (0.65, 0.40, 0.38), p["sandbag"])
    for i in [-0.35, 0.35]:
        add_box(bm, Vector((i, 0, 0.55)), (0.65, 0.35, 0.34), p["sandbag"])

def build_item_barrier(bm, p):
    add_box(bm, Vector((0, 0, 0.5)), (1.8, 0.6, 1.0), p["concrete"])
    add_box(bm, Vector((0, 0, 1.04)), (1.84, 0.64, 0.08), p["concrete"])

def build_item_barrel(bm, p):
    add_cylinder(bm, Vector((0, 0, 0.55)), radius=0.34, depth=1.1, mat_index=p["hazard_red"])
    add_cylinder(bm, Vector((0, 0, 0.55)), radius=0.35, depth=0.22, mat_index=p["hazard_yellow"])
    add_cylinder(bm, Vector((0, 0, 1.05)), radius=0.36, depth=0.06, mat_index=p["metal_dark"])
    add_cylinder(bm, Vector((0, 0, 0.05)), radius=0.36, depth=0.06, mat_index=p["metal_dark"])

def build_item_medkit(bm, p):
    add_box(bm, Vector((0, 0, 0.22)), (0.60, 0.44, 0.30), p["white"])
    add_box(bm, Vector((0, 0, 0.375)), (0.24, 0.08, 0.02), p["hazard_red"])
    add_box(bm, Vector((0, 0, 0.375)), (0.08, 0.24, 0.02), p["hazard_red"])
    add_box(bm, Vector((0, 0.24, 0.22)), (0.20, 0.05, 0.08), p["metal_dark"])

def build_item_tree(bm, p):
    add_cylinder(bm, Vector((0, 0, 0.6)), radius=0.18, depth=1.2, mat_index=p["tree_trunk"])
    add_cylinder(bm, Vector((0, 0, 1.4)), radius=1.1, depth=1.0, mat_index=p["tree_leaves"])
    add_cylinder(bm, Vector((0, 0, 2.1)), radius=0.8, depth=0.9, mat_index=p["tree_leaves"])
    add_cylinder(bm, Vector((0, 0, 2.8)), radius=0.5, depth=0.8, mat_index=p["tree_leaves"])

def build_item_fracture(bm, p):
    # Intact left
    add_box(bm, Vector((-0.45, 0, 0.5)), (0.7, 0.6, 1.0), p["concrete"])
    # Rebar rods
    add_box(bm, Vector((-0.05, 0.1, 0.4)), (0.25, 0.03, 0.03), p["rebar"])
    add_box(bm, Vector((-0.05, -0.1, 0.7)), (0.25, 0.03, 0.03), p["rebar"])
    # Shattered chunks
    add_box(bm, Vector((0.35, 0.15, 0.6)), (0.35, 0.3, 0.35), p["concrete_inner"], rot_z=math.radians(25), rot_x=math.radians(20))
    add_box(bm, Vector((0.55, -0.2, 0.3)), (0.3, 0.35, 0.25), p["concrete"], rot_y=math.radians(35))
    add_box(bm, Vector((0.75, 0.1, 0.15)), (0.25, 0.25, 0.2), p["concrete_inner"], rot_z=math.radians(-30))

def build_item_jax(bm, p):
    # Jax (Survivor)
    add_box(bm, Vector((-0.15, 0.04, 0.08)), (0.16, 0.30, 0.14), p["white"])
    add_box(bm, Vector(( 0.15, 0.04, 0.08)), (0.16, 0.30, 0.14), p["white"])
    add_box(bm, Vector((-0.15, 0, 0.44)), (0.18, 0.20, 0.60), p["pants_charcoal"])
    add_box(bm, Vector(( 0.15, 0, 0.44)), (0.18, 0.20, 0.60), p["pants_charcoal"])
    add_box(bm, Vector((0, 0, 1.05)), (0.50, 0.30, 0.62), p["jacket_teal"])
    add_box(bm, Vector((0, 0.155, 1.05)), (0.04, 0.015, 0.58), p["white"])
    add_box(bm, Vector((0, -0.20, 1.08)), (0.36, 0.16, 0.42), p["vest_ochre"])
    # Arms + Flashlight
    add_box(bm, Vector((-0.34, 0, 1.00)), (0.14, 0.16, 0.52), p["jacket_teal"])
    add_box(bm, Vector(( 0.34, 0.14, 1.06)), (0.14, 0.36, 0.16), p["jacket_teal"], rot_x=math.radians(25))
    add_cylinder(bm, Vector((0.34, 0.44, 1.17)), radius=0.035, depth=0.22, mat_index=p["metal_dark"], rot_x=math.radians(90))
    add_cylinder(bm, Vector((0.34, 0.55, 1.17)), radius=0.05, depth=0.03, mat_index=p["glow_flashlight"], rot_x=math.radians(90))
    # Head
    add_box(bm, Vector((0, 0, 1.52)), (0.28, 0.28, 0.30), p["skin_hero"])
    add_box(bm, Vector((0, -0.02, 1.66)), (0.30, 0.30, 0.10), p["pants_charcoal"])

def build_item_rook(bm, p):
    # Rook (Scavenger)
    add_box(bm, Vector((-0.16, 0.04, 0.08)), (0.18, 0.32, 0.16), p["metal_dark"])
    add_box(bm, Vector(( 0.16, 0.04, 0.08)), (0.18, 0.32, 0.16), p["metal_dark"])
    add_box(bm, Vector((-0.16, 0, 0.45)), (0.20, 0.22, 0.60), p["pants_olive"])
    add_box(bm, Vector(( 0.16, 0, 0.45)), (0.20, 0.22, 0.60), p["pants_olive"])
    add_box(bm, Vector((0, 0, 1.05)), (0.52, 0.32, 0.60), p["pants_charcoal"])
    add_box(bm, Vector((0, 0, 1.08)), (0.54, 0.36, 0.48), p["vest_ochre"])
    add_box(bm, Vector((-0.14, 0.19, 1.02)), (0.10, 0.06, 0.14), p["pants_olive"])
    add_box(bm, Vector(( 0.14, 0.19, 1.02)), (0.10, 0.06, 0.14), p["pants_olive"])
    # SMG
    add_box(bm, Vector((0.06, 0.30, 1.00)), (0.10, 0.42, 0.16), p["metal_dark"])
    # Head & Beanie
    add_box(bm, Vector((0, 0, 1.50)), (0.28, 0.28, 0.28), p["skin_hero"])
    add_box(bm, Vector((0, 0, 1.64)), (0.30, 0.30, 0.14), p["beanie_red"])

def build_item_zombie(bm, p):
    # Zombie (Infected Walker)
    add_box(bm, Vector((-0.18, -0.04, 0.06)), (0.14, 0.28, 0.12), p["skin_zombie"])
    add_box(bm, Vector(( 0.16,  0.08, 0.06)), (0.14, 0.28, 0.12), p["skin_zombie"])
    add_box(bm, Vector((-0.18, -0.02, 0.40)), (0.18, 0.20, 0.55), p["clothes_tattered"])
    add_box(bm, Vector(( 0.16,  0.06, 0.40)), (0.18, 0.20, 0.55), p["clothes_tattered"])
    add_box(bm, Vector((0, 0.14, 0.98)), (0.48, 0.28, 0.58), p["clothes_tattered"], rot_x=math.radians(-18))
    # Outstretched claws
    add_box(bm, Vector((-0.32, 0.34, 1.08)), (0.13, 0.44, 0.13), p["clothes_tattered"], rot_x=math.radians(-50))
    add_box(bm, Vector(( 0.32, 0.30, 1.00)), (0.13, 0.42, 0.13), p["clothes_tattered"], rot_x=math.radians(-40))
    # Head & glowing eyes
    add_box(bm, Vector((0, 0.30, 1.40)), (0.26, 0.26, 0.26), p["skin_zombie"], rot_z=math.radians(12), rot_x=math.radians(-12))
    add_box(bm, Vector((-0.07, 0.44, 1.44)), (0.04, 0.02, 0.04), p["glow_eyes"])
    add_box(bm, Vector(( 0.07, 0.44, 1.44)), (0.04, 0.02, 0.04), p["glow_eyes"])

# ==============================================================================
# 3. AUTOMATED THUMBNAIL RENDERER
# ==============================================================================
ITEMS = [
    ("thumb_crate", build_item_crate, (0, 0, 0.6), 2.8, 38.0),
    ("thumb_sandbags", build_item_sandbags, (0, 0, 0.35), 2.8, 38.0),
    ("thumb_barrier", build_item_barrier, (0, 0, 0.5), 3.2, 36.0),
    ("thumb_barrel", build_item_barrel, (0, 0, 0.55), 2.6, 38.0),
    ("thumb_medkit", build_item_medkit, (0, 0, 0.22), 1.8, 42.0),
    ("thumb_tree", build_item_tree, (0, 0, 1.5), 5.5, 36.0),
    ("thumb_fracture", build_item_fracture, (0.1, 0, 0.5), 3.0, 36.0),
    ("thumb_jax", build_item_jax, (0, 0, 0.9), 3.4, 38.0),
    ("thumb_rook", build_item_rook, (0, 0, 0.9), 3.4, 38.0),
    ("thumb_zombie", build_item_zombie, (0, 0, 0.8), 3.4, 38.0),
]

def render_all_thumbnails():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    thumb_dir = os.path.join(script_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    for name, builder, target_co, cam_dist, lens in ITEMS:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        palette = create_palette()
        mat_keys = list(palette.keys())
        p_map = {k: i for i, k in enumerate(mat_keys)}
        mat_list = [palette[k] for k in mat_keys]

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()
        builder(bm, p_map)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        for m in mat_list:
            obj.data.materials.append(m)

        # Sleek dark circular pedestal
        ped_mesh = bpy.data.meshes.new("Ped_Mesh")
        bm_ped = bmesh.new()
        add_cylinder(bm_ped, Vector((0, 0, -0.06)), radius=1.6, depth=0.12, mat_index=p_map["pedestal"], segments=24)
        bm_ped.to_mesh(ped_mesh)
        bm_ped.free()
        ped_obj = bpy.data.objects.new("Pedestal", ped_mesh)
        ped_obj.data.materials.append(palette["pedestal"])
        bpy.context.collection.objects.link(ped_obj)

        # Camera
        cam_target = bpy.data.objects.new("Target", None)
        cam_target.location = target_co
        bpy.context.collection.objects.link(cam_target)

        cam_data = bpy.data.cameras.new("ThumbCam")
        cam_data.type = 'PERSP'
        cam_data.lens = lens
        cam_obj = bpy.data.objects.new("ThumbCam", cam_data)
        bpy.context.collection.objects.link(cam_obj)

        # 3/4 isometric product shot
        cam_angle = math.radians(45)
        cx = cam_dist * math.cos(cam_angle)
        # Characters face +Y, props can be viewed from either side
        cy = cam_dist * math.sin(cam_angle) if "thumb_" in name else -cam_dist * math.sin(cam_angle)
        cz = target_co[2] + cam_dist * 0.40
        cam_obj.location = (cx, cy, cz)

        track = cam_obj.constraints.new(type='TRACK_TO')
        track.target = cam_target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        bpy.context.scene.camera = cam_obj

        # Studio Lighting
        key = bpy.data.lights.new("Key", type='AREA')
        key.energy = 160.0
        key.size = 2.0
        key.color = (1.0, 0.96, 0.92)
        key_obj = bpy.data.objects.new("Key", key)
        key_obj.location = (cx + 1.0, cy + 1.0, cz + 1.5)
        bpy.context.collection.objects.link(key_obj)

        rim = bpy.data.lights.new("Rim", type='AREA')
        rim.energy = 140.0
        rim.size = 2.0
        rim.color = (0.45, 0.75, 1.0)
        rim_obj = bpy.data.objects.new("Rim", rim)
        rim_obj.location = (-cx, -cy, cz + 1.0)
        bpy.context.collection.objects.link(rim_obj)

        # Render 256x256 thumbnail
        scene = bpy.context.scene
        scene.render.resolution_x = 256
        scene.render.resolution_y = 256
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 24
        scene.cycles.use_denoising = True

        out_path = os.path.join(thumb_dir, f"{name}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"RENDERED_THUMB: {out_path}")

    print("ALL_THUMBNAILS_COMPLETE")

if __name__ == "__main__":
    render_all_thumbnails()
