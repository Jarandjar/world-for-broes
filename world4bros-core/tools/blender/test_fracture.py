import bpy
import bmesh
import math
import random
import os
from mathutils import Vector, Matrix

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

def create_fracture_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    mat_concrete = get_or_create_material("Mat_Concrete", (0.58, 0.58, 0.60, 1.0), roughness=0.7)
    mat_concrete_inner = get_or_create_material("Mat_Concrete_Inner", (0.35, 0.35, 0.37, 1.0), roughness=0.9)
    mat_rebar = get_or_create_material("Mat_Rebar_Metal", (0.18, 0.15, 0.14, 1.0), roughness=0.5, metallic=0.9)
    mat_debris_dust = get_or_create_material("Mat_Dust", (0.48, 0.46, 0.44, 1.0), roughness=0.95)

    # We will build a concrete barrier that is half-intact and half-fractured into explosive debris shards!
    mesh = bpy.data.meshes.new("FracturedBarrier_Mesh")
    bm = bmesh.new()

    # Intact Left Half of the barrier (from X = -1.5 to X = -0.2)
    # Size: (1.3, 0.6, 1.0)
    mat_intact = Matrix.Translation(Vector((-0.85, 0, 0.5))) @ Matrix.Diagonal((1.3, 0.6, 1.0, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat_intact)
    for f in bm.faces:
        f.material_index = 0

    # Chamfer top cap of intact half
    mat_cap = Matrix.Translation(Vector((-0.85, 0, 1.04))) @ Matrix.Diagonal((1.34, 0.64, 0.08, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat_cap)
    for f in bm.faces:
        if f.material_index != 0:
            f.material_index = 0

    # Exposed Rebar rods protruding from the fracture fracture plane
    for rz in [0.35, 0.65]:
        for ry in [-0.15, 0.15]:
            mat_rod = Matrix.Translation(Vector((-0.1, ry, rz))) @ Matrix.Rotation(math.radians(random.uniform(-10, 10)), 4, 'Z') @ Matrix.Diagonal((0.35, 0.03, 0.03, 1.0))
            bmesh.ops.create_cube(bm, size=1.0, matrix=mat_rod)
            for f in bm.faces:
                if f.material_index != 0:
                    f.material_index = 2

    # Now generate the SHATTERED VORONOI-STYLE CELL CHUNKS on the right side!
    random.seed(42)
    # Generate ~12 discrete fractured cell shards scattering outward from explosion impact at (0.2, 0, 0.5)
    impact_center = Vector((0.3, 0, 0.4))
    
    for i in range(14):
        # Base shard size
        sx = random.uniform(0.18, 0.38)
        sy = random.uniform(0.16, 0.36)
        sz = random.uniform(0.14, 0.32)
        
        # Original position before blast
        orig_x = random.uniform(0.0, 1.4)
        orig_y = random.uniform(-0.28, 0.28)
        orig_z = random.uniform(0.1, 0.9)
        orig_pos = Vector((orig_x, orig_y, orig_z))

        # Blast displacement vector
        blast_dir = (orig_pos - impact_center).normalized()
        # Add random scatter
        blast_dir += Vector((random.uniform(0.1, 0.4), random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.3)))
        blast_dist = random.uniform(0.15, 0.75) * (orig_x + 0.3)
        
        chunk_pos = orig_pos + blast_dir * blast_dist
        # Gravity drop if falling to ground
        if chunk_pos.z < sz / 2.0:
            chunk_pos.z = sz / 2.0 + 0.01

        # Random tumble rotation
        rot_x = math.radians(random.uniform(-45, 45))
        rot_y = math.radians(random.uniform(-45, 45))
        rot_z = math.radians(random.uniform(-60, 60))

        mat_chunk = (Matrix.Translation(chunk_pos) @
                     Matrix.Rotation(rot_z, 4, 'Z') @
                     Matrix.Rotation(rot_y, 4, 'Y') @
                     Matrix.Rotation(rot_x, 4, 'X') @
                     Matrix.Diagonal((sx, sy, sz, 1.0)))
        
        f_before = set(bm.faces)
        bmesh.ops.create_cube(bm, size=1.0, matrix=mat_chunk)
        for f in bm.faces:
            if f not in f_before:
                # Exterior faces concrete, some jagged inner
                f.material_index = 1 if random.random() > 0.4 else 0

    # Add scattered small gravel / rubble debris pebbles on ground around blast
    for _ in range(25):
        gx = random.uniform(-0.2, 2.2)
        gy = random.uniform(-1.2, 1.2)
        gs = random.uniform(0.04, 0.09)
        mat_pebble = Matrix.Translation(Vector((gx, gy, gs / 2.0))) @ Matrix.Rotation(math.radians(random.uniform(0, 360)), 4, 'Z') @ Matrix.Diagonal((gs, gs * 0.8, gs * 0.6, 1.0))
        f_before = set(bm.faces)
        bmesh.ops.create_cube(bm, size=1.0, matrix=mat_pebble)
        for f in bm.faces:
            if f not in f_before:
                f.material_index = 3

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("FracturedBarrier_Object", mesh)
    bpy.context.collection.objects.link(obj)

    obj.data.materials.append(mat_concrete)
    obj.data.materials.append(mat_concrete_inner)
    obj.data.materials.append(mat_rebar)
    obj.data.materials.append(mat_debris_dust)

    # Pedestal Ground
    bg_mesh = bpy.data.meshes.new("FloorMesh")
    bm_bg = bmesh.new()
    mat_floor = get_or_create_material("Mat_Floor", (0.09, 0.11, 0.15, 1.0), roughness=0.6)
    mat_f = Matrix.Translation(Vector((0, 0, -0.05))) @ Matrix.Diagonal((6.0, 5.0, 0.1, 1.0))
    bmesh.ops.create_cube(bm_bg, size=1.0, matrix=mat_f)
    bm_bg.to_mesh(bg_mesh)
    bm_bg.free()
    bg_obj = bpy.data.objects.new("StudioFloor", bg_mesh)
    bg_obj.data.materials.append(mat_floor)
    bpy.context.collection.objects.link(bg_obj)

    # Camera looking at center of fracture (0.3, 0, 0.5)
    cam_target = bpy.data.objects.new("CamTarget", None)
    cam_target.location = (0.3, 0, 0.5)
    bpy.context.collection.objects.link(cam_target)

    cam_data = bpy.data.cameras.new("FractureCam")
    cam_data.type = 'PERSP'
    cam_data.lens = 42.0
    cam_obj = bpy.data.objects.new("FractureStudioCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = (2.8, -3.2, 2.2)
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = cam_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam_obj

    # Key + Rim Lighting
    key = bpy.data.lights.new("KeyLight", type='AREA')
    key.energy = 140.0
    key.size = 2.5
    key.color = (1.0, 0.95, 0.90)
    key_obj = bpy.data.objects.new("KeyLight", key)
    key_obj.location = (2.5, -2.5, 3.0)
    bpy.context.collection.objects.link(key_obj)

    rim = bpy.data.lights.new("RimLight", type='AREA')
    rim.energy = 120.0
    rim.size = 2.0
    rim.color = (0.45, 0.75, 1.0)
    rim_obj = bpy.data.objects.new("RimLight", rim)
    rim_obj.location = (-2.5, 2.5, 2.5)
    bpy.context.collection.objects.link(rim_obj)

    # Cycles CPU Setup
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 768
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.join(script_dir, "fractured_barrier_preview.png")
    out_blend = os.path.join(script_dir, "fractured_barrier.blend")
    out_glb = os.path.join(script_dir, "fractured_barrier.glb")

    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB', use_selection=False)

    print(f"FRACTURE_RENDER_SUCCESS: Image={out_img}, Blend={out_blend}, GLB={out_glb}")

if __name__ == "__main__":
    create_fracture_scene()
