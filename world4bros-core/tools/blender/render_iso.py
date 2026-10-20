"""
Isometric Renderer for World 4 Bro's
Implements the exact camera and lighting specifications from docs/STYLE_BIBLE.md §0.8:
- Metric units: 1 Blender unit = 1 metre = 1 game tile.
- Camera: Orthographic, Rot X=60 deg (30 deg elevation), Y=0 deg, Z=45 deg (looking from SE).
- Scale: 181 px per metre across render width (authoring at 2x resolution).
- Sun from North-West (45 deg up): Top face = 100%, Left face = 85%, Right face = 62%.
- Color Management: 'Standard' (not Filmic/AgX) for authentic palette reproduction.
- Transparent background (RGBA PNG).
- Turntable renders: 8 directions (45 deg steps) or 16 directions (22.5 deg steps for vehicles).
"""

import bpy
import math
import os
import sys
from mathutils import Vector, Euler

def setup_iso_scene(resolution=256, ortho_scale=2.83):
    """
    Configures the active Blender scene for World 4 Bro's isometric rendering.
    ortho_scale = resolution / 181.0 (approx 2.83 for 512px, 1.41 for 256px)
    """
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    # Color Management: Standard (Style Bible §0.8)
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'

    # Render engine & transparent film
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    # Clean existing lights and cameras
    for obj in list(scene.objects):
        if obj.type in {'CAMERA', 'LIGHT'}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # 1. Orthographic Isometric Camera
    # Looking down at 30 deg elevation from south-east: Rot X=60, Y=0, Z=45
    cam_data = bpy.data.cameras.new("IsoCamera")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = ortho_scale
    cam_obj = bpy.data.objects.new("IsoCamera", cam_data)
    cam_obj.rotation_euler = Euler((math.radians(60), 0, math.radians(45)), 'XYZ')
    
    # Distance back along view vector (X+10, Y-10, Z+12.247)
    dist = 15.0
    rad_z = math.radians(45)
    rad_x = math.radians(30)
    cam_obj.location = Vector((
        dist * math.cos(rad_x) * math.sin(rad_z),
        -dist * math.cos(rad_x) * math.cos(rad_z),
        dist * math.sin(rad_x) + 0.5
    ))
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # 2. Sun Light from North-West (45 deg elevation)
    # Sun light rotation: points from NW (-X, +Y, +Z) towards center (+X, -Y, -Z)
    sun_data = bpy.data.lights.new("Sun_NW", type='SUN')
    sun_data.energy = 3.5
    sun_data.color = (1.0, 0.98, 0.95)
    sun_obj = bpy.data.objects.new("Sun_NW", sun_data)
    # Pitch 45 deg down, heading towards SE (135 deg)
    sun_obj.rotation_euler = Euler((math.radians(45), math.radians(30), math.radians(-45)), 'XYZ')
    scene.collection.objects.link(sun_obj)

    # 3. Soft Ambient Fill Sky Light (faint cool fill)
    fill_data = bpy.data.lights.new("Fill_Sky", type='SUN')
    fill_data.energy = 0.8
    fill_data.color = (0.75, 0.85, 0.95)
    fill_obj = bpy.data.objects.new("Fill_Sky", fill_data)
    fill_obj.rotation_euler = Euler((math.radians(120), 0, math.radians(225)), 'XYZ')
    scene.collection.objects.link(fill_obj)

    return cam_obj

def render_turntable(target_obj, output_dir, file_prefix, directions=8):
    """
    Renders the target object at N discrete rotation angles (e.g. 8 directions, 45 deg steps).
    Rotates the target object around world Z while camera stays stationary.
    """
    os.makedirs(output_dir, exist_ok=True)
    orig_rot_z = target_obj.rotation_euler.z

    rendered_files = []
    step_deg = 360.0 / directions

    for i in range(directions):
        angle_deg = i * step_deg
        target_obj.rotation_euler.z = orig_rot_z + math.radians(angle_deg)
        bpy.context.view_layer.update()

        out_name = f"{file_prefix}_dir{i}.png" if directions > 1 else f"{file_prefix}.png"
        out_path = os.path.join(output_dir, out_name)
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        rendered_files.append(out_path)

    target_obj.rotation_euler.z = orig_rot_z
    return rendered_files

if __name__ == "__main__":
    setup_iso_scene(resolution=256, ortho_scale=2.5)
    print("ISO_SCENE_SETUP_SUCCESSFUL")
