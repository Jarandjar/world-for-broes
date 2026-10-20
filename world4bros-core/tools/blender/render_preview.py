import bpy
import sys
import os
import math

script_dir = r"C:\Users\Jalle\OneDrive\Desktop\BUCKS\game\world4bros-core\tools\blender"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from generate_building import build_procedural_building

class MockProps:
    seed = 42
    floors = 8
    floor_height = 3.0
    width = 12.0
    depth = 10.0
    enable_setback = True
    setback_inset = 1.6
    generate_roof_clutter = True

bpy.ops.wm.read_factory_settings(use_empty=True)

# Build procedural building
obj = build_procedural_building(MockProps)

# Set up isometric camera
cam_data = bpy.data.cameras.new("IsoCam")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = 32.0
cam_obj = bpy.data.objects.new("IsoCamera", cam_data)
bpy.context.collection.objects.link(cam_obj)

# Position camera at isometric angle looking at building center (0, 0, 12)
cam_data.ortho_scale = 46.0
cam_obj.location = (30.0, -30.0, 36.0)

# Track to constraint or direct rotation
# Standard isometric rotation: pitch = 60 deg, yaw = 45 deg
cam_obj.rotation_euler = (math.radians(58), math.radians(0), math.radians(45))
bpy.context.scene.camera = cam_obj

# Set up simple sun lighting for clay/greybox look
light_data = bpy.data.lights.new("Sun", type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new("SunLight", light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (math.radians(50), math.radians(20), math.radians(-35))

# Ground plane
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))

# Render settings (Workbench or fast Cycles/EEVEE)
scene = bpy.context.scene
scene.render.resolution_x = 960
scene.render.resolution_y = 720
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'MATCAP'
scene.display.shading.color_type = 'SINGLE'

render_output = os.path.join(script_dir, "isometric_building_preview.png")
scene.render.filepath = render_output
bpy.ops.render.render(write_still=True)
print(f"RENDER_SUCCESS: {render_output}")
