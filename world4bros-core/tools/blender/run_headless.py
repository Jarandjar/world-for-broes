import bpy
import sys
import os

script_dir = r"C:\Users\Jalle\OneDrive\Desktop\BUCKS\game\world4bros-core\tools\blender"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from generate_building import build_procedural_building

class MockProps:
    seed = 101
    floors = 7
    floor_height = 3.0
    width = 12.0
    depth = 10.0
    enable_setback = True
    setback_inset = 1.5
    generate_roof_clutter = True

# Clear default objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Build procedural building
obj = build_procedural_building(MockProps)

# Save .blend file
output_blend = os.path.join(script_dir, "procedural_building_demo.blend")
bpy.ops.wm.save_as_mainfile(filepath=output_blend)
print(f"SUCCESS: Saved procedural building to {output_blend}")
