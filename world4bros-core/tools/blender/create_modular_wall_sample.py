import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix

# ==========================================
# MATERIAL BUILDERS
# ==========================================
def get_or_create_material(name, base_color, roughness=0.5, metallic=0.0, emission_color=(0, 0, 0, 1), emission_strength=0.0):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        # Base Color
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = base_color
        # Roughness
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        # Metallic
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        # Emission
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = emission_color
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = emission_color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat

def add_box_with_mat(bm, center, size, mat_index=0, rot_z=0.0):
    sx, sy, sz = size
    mat = Matrix.Translation(center) @ Matrix.Rotation(rot_z, 4, 'Z') @ Matrix.Diagonal((sx, sy, sz, 1.0))
    f_before = set(bm.faces)
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    for face in bm.faces:
        if face not in f_before:
            face.material_index = mat_index

def add_cylinder_with_mat(bm, center, radius, depth, mat_index=0, segments=12, rot_x=0.0, rot_y=0.0, rot_z=0.0):
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

# ==========================================
# BUILD THE DETAILED MODULAR WALL UNIT
# ==========================================
def build_sample_wall():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 1. Define Materials
    # Mat 0: Rich Petrol Teal Siding
    mat_siding = get_or_create_material("Mat_Siding_Teal", base_color=(0.14, 0.32, 0.40, 1.0), roughness=0.6)
    # Mat 1: Architectural White Trim
    mat_trim = get_or_create_material("Mat_Trim_White", base_color=(0.94, 0.95, 0.96, 1.0), roughness=0.35)
    # Mat 2: Dark Bronze Window Frame
    mat_frame = get_or_create_material("Mat_Frame_Bronze", base_color=(0.10, 0.12, 0.16, 1.0), roughness=0.3)
    # Mat 3: Golden Glowing Interior Glass
    mat_glow = get_or_create_material(
        "Mat_Window_Glow",
        base_color=(1.0, 0.85, 0.45, 1.0),
        roughness=0.1,
        emission_color=(1.0, 0.82, 0.40, 1.0),
        emission_strength=4.5
    )
    # Mat 4: Metal Black Fixture
    mat_metal = get_or_create_material("Mat_Metal_Black", base_color=(0.08, 0.08, 0.10, 1.0), roughness=0.3, metallic=0.9)
    # Mat 5: Warm Lamp Bulb
    mat_bulb = get_or_create_material(
        "Mat_Lamp_Bulb",
        base_color=(1.0, 0.95, 0.7, 1.0),
        roughness=0.1,
        emission_color=(1.0, 0.92, 0.6, 1.0),
        emission_strength=8.0
    )

    mesh = bpy.data.meshes.new("ModularWall_Window_Mesh")
    bm = bmesh.new()

    # Materials slot order:
    # 0: mat_siding
    # 1: mat_trim
    # 2: mat_frame
    # 3: mat_glow
    # 4: mat_metal
    # 5: mat_bulb

    wall_w = 2.0
    wall_d = 0.26
    wall_h = 3.0

    # 2. Main Wall Body (Split with Window Aperture)
    # Window dimensions: 1.1m wide, 1.5m tall, sill at 0.85m
    win_w = 1.1
    win_h = 1.5
    win_bottom = 0.85
    win_top = win_bottom + win_h

    # Lower wall apron (under window)
    add_box_with_mat(bm, Vector((0, 0, win_bottom / 2.0)), (wall_w, wall_d, win_bottom), mat_index=0)
    # Upper wall lintel (above window)
    lintel_h = wall_h - win_top
    add_box_with_mat(bm, Vector((0, 0, win_top + lintel_h / 2.0)), (wall_w, wall_d, lintel_h), mat_index=0)
    # Left wall pier
    side_w = (wall_w - win_w) / 2.0
    add_box_with_mat(bm, Vector((-(win_w + side_w) / 2.0, 0, win_bottom + win_h / 2.0)), (side_w, wall_d, win_h), mat_index=0)
    # Right wall pier
    add_box_with_mat(bm, Vector(((win_w + side_w) / 2.0, 0, win_bottom + win_h / 2.0)), (side_w, wall_d, win_h), mat_index=0)

    # 3. Horizontal Lapboard Siding Details (Exterior relief ribs)
    # Adds physical horizontal beveled lapboard slats on the front (+Y face)
    slat_count = 14
    slat_step = wall_h / slat_count
    for s in range(1, slat_count):
        sz = s * slat_step
        # Skip if cutting through window opening
        if sz > win_bottom and sz < win_top:
            # Add slats only on left and right piers
            add_box_with_mat(bm, Vector((-(win_w + side_w) / 2.0, wall_d / 2.0 + 0.015, sz)), (side_w, 0.03, 0.04), mat_index=0)
            add_box_with_mat(bm, Vector(((win_w + side_w) / 2.0, wall_d / 2.0 + 0.015, sz)), (side_w, 0.03, 0.04), mat_index=0)
        else:
            add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.015, sz)), (wall_w, 0.03, 0.04), mat_index=0)

    # 4. Architectural Trims (White Moldings)
    # A. Baseboard skirting trim (bottom)
    add_box_with_mat(bm, Vector((0, 0, 0.1)), (wall_w + 0.02, wall_d + 0.05, 0.20), mat_index=1)
    # B. Cornice / Frieze trim (top)
    add_box_with_mat(bm, Vector((0, 0, wall_h - 0.08)), (wall_w + 0.02, wall_d + 0.05, 0.16), mat_index=1)
    # C. Left & Right vertical corner columns (snap pilasters)
    col_w = 0.08
    add_box_with_mat(bm, Vector((-wall_w / 2.0 + col_w / 2.0, wall_d / 2.0 + 0.02, wall_h / 2.0)), (col_w, 0.04, wall_h), mat_index=1)
    add_box_with_mat(bm, Vector((wall_w / 2.0 - col_w / 2.0, wall_d / 2.0 + 0.02, wall_h / 2.0)), (col_w, 0.04, wall_h), mat_index=1)

    # 5. Window Frame, Sill, Lintel & Glowing Glass
    win_center_z = win_bottom + win_h / 2.0

    # Overhanging Window Sill (thick white ledge)
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.07, win_bottom)), (win_w + 0.18, 0.14, 0.09), mat_index=1)
    # Top Window Lintel / Header Molding
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.07, win_top)), (win_w + 0.22, 0.14, 0.10), mat_index=1)

    # Outer Bronze Window Perimeter Frame
    frame_t = 0.05
    # Left & Right frame
    add_box_with_mat(bm, Vector((-(win_w - frame_t) / 2.0, wall_d / 2.0 + 0.02, win_center_z)), (frame_t, 0.06, win_h), mat_index=2)
    add_box_with_mat(bm, Vector(((win_w - frame_t) / 2.0, wall_d / 2.0 + 0.02, win_center_z)), (frame_t, 0.06, win_h), mat_index=2)
    # Top & Bottom frame
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.02, win_bottom + frame_t / 2.0)), (win_w, 0.06, frame_t), mat_index=2)
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.02, win_top - frame_t / 2.0)), (win_w, 0.06, frame_t), mat_index=2)

    # Window Dividers (Mullion & Transom cross)
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.02, win_center_z)), (0.04, 0.05, win_h - frame_t * 2), mat_index=2)
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + 0.02, win_center_z + 0.15)), (win_w - frame_t * 2, 0.05, 0.04), mat_index=2)

    # GLOWING INTERIOR GLASS PANE
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 - 0.01, win_center_z)), (win_w - 0.08, 0.02, win_h - 0.08), mat_index=3)

    # 6. Exterior Decorative Micro-Details
    # A. Wall-mounted exterior vintage gooseneck lamp fixture
    lamp_x = win_w / 2.0 + 0.25
    lamp_z = win_top - 0.2
    # Base plate on wall
    add_box_with_mat(bm, Vector((lamp_x, wall_d / 2.0 + 0.02, lamp_z)), (0.12, 0.04, 0.16), mat_index=4)
    # Gooseneck curved arm
    add_cylinder_with_mat(bm, Vector((lamp_x, wall_d / 2.0 + 0.14, lamp_z + 0.05)), radius=0.015, depth=0.22, mat_index=4, rot_x=math.radians(90))
    # Lamp shade / hood
    add_cylinder_with_mat(bm, Vector((lamp_x, wall_d / 2.0 + 0.25, lamp_z + 0.02)), radius=0.12, depth=0.06, mat_index=4)
    # Glowing bulb inside lamp
    add_cylinder_with_mat(bm, Vector((lamp_x, wall_d / 2.0 + 0.25, lamp_z - 0.03)), radius=0.04, depth=0.06, mat_index=5)

    # B. Flowerbox / Planter under window (vaultable parkour ledge!)
    fb_w = win_w + 0.1
    fb_h = 0.24
    fb_d = 0.26
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + fb_d / 2.0 + 0.02, win_bottom - 0.2)), (fb_w, fb_d, fb_h), mat_index=1)
    # Soil inside planter
    add_box_with_mat(bm, Vector((0, wall_d / 2.0 + fb_d / 2.0 + 0.02, win_bottom - 0.1)), (fb_w - 0.06, fb_d - 0.06, 0.04), mat_index=4)

    # Finalize BMesh and link to Object
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("ModularWall_SuburbanWindow", mesh)
    bpy.context.collection.objects.link(obj)

    # Assign materials to object slots
    for m in [mat_siding, mat_trim, mat_frame, mat_glow, mat_metal, mat_bulb]:
        obj.data.materials.append(m)

    # Add subtle Bevel modifier to catch specular highlights on edges
    bev = obj.modifiers.new(name="Bevel", type='BEVEL')
    bev.width = 0.015
    bev.segments = 2
    bev.limit_method = 'ANGLE'
    bev.angle_limit = math.radians(40)

    # 7. Studio Camera Setup (Isometric Hero Product Shot)
    cam_target = bpy.data.objects.new("CamTarget", None)
    cam_target.location = (0, 0, 1.5)
    bpy.context.collection.objects.link(cam_target)

    cam_data = bpy.data.cameras.new("WallCamera")
    cam_data.type = 'PERSP'
    cam_data.lens = 45.0
    cam_obj = bpy.data.objects.new("WallStudioCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Position camera in front-right at 3/4 angle
    cam_obj.location = (3.5, 4.2, 2.7)
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = cam_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam_obj

    # 8. Studio Three-Point Lighting Setup
    # Key Light (Warm Sunlight)
    key_light = bpy.data.lights.new("StudioKey", type='AREA')
    key_light.energy = 180.0
    key_light.size = 2.5
    key_light.color = (1.0, 0.95, 0.90)
    key_obj = bpy.data.objects.new("StudioKey", key_light)
    bpy.context.collection.objects.link(key_obj)
    key_obj.location = (2.5, 3.5, 4.0)
    key_obj.rotation_euler = (math.radians(45), math.radians(20), math.radians(-30))

    # Fill Light (Cool Sky)
    fill_light = bpy.data.lights.new("StudioFill", type='AREA')
    fill_light.energy = 60.0
    fill_light.size = 3.0
    fill_light.color = (0.75, 0.85, 1.0)
    fill_obj = bpy.data.objects.new("StudioFill", fill_light)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = (-2.5, 3.0, 2.5)

    # Rim / Backlight (Pops silhouette against background)
    rim_light = bpy.data.lights.new("StudioRim", type='AREA')
    rim_light.energy = 140.0
    rim_light.size = 2.0
    rim_light.color = (1.0, 0.88, 0.75)
    rim_obj = bpy.data.objects.new("StudioRim", rim_light)
    bpy.context.collection.objects.link(rim_obj)
    rim_obj.location = (-1.0, -3.0, 3.5)

    # 9. Clean Studio Backdrop
    bg_mesh = bpy.data.meshes.new("BackdropMesh")
    bm_bg = bmesh.new()
    mat_bg = get_or_create_material("Mat_Studio_Dark", base_color=(0.06, 0.08, 0.12, 1.0), roughness=0.8)
    # Floor plane
    add_box_with_mat(bm_bg, Vector((0, 0, -0.05)), (8.0, 8.0, 0.1), mat_index=0)
    bm_bg.to_mesh(bg_mesh)
    bm_bg.free()
    bg_obj = bpy.data.objects.new("Studio_Floor", bg_mesh)
    bg_obj.data.materials.append(mat_bg)
    bpy.context.collection.objects.link(bg_obj)

    # 10. Render Setup (Cycles CPU for realistic soft shadows and real emission glow)
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.join(script_dir, "sample_modular_wall_preview.png")
    out_blend = os.path.join(script_dir, "sample_modular_wall.blend")
    out_glb = os.path.join(script_dir, "sample_modular_wall.glb")

    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    # Export .GLB
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB', use_selection=False)

    print(f"WALL_RENDER_SUCCESS: Image={out_img}, Blend={out_blend}, GLB={out_glb}")

if __name__ == "__main__":
    build_sample_wall()
