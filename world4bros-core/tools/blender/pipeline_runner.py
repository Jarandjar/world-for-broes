"""
Master Pipeline Runner for World 4 Bro's
Bridges:
  1. Game Data (src/data/)
  2. Asset Catalog (tools/blender/jobs/asset_catalog.json)
  3. Blender Procedural Generators & 'Make It Ours' Pass
  4. 2.5D Isometric Sprite Outputs (public/assets/)
  5. 3D GLB Pipeline for Unreal Engine 5.7 CLI (tools/unreal/World4Bros/Content/Import/)

Usage:
  blender --background --python tools/blender/pipeline_runner.py -- --slice
  blender --background --python tools/blender/pipeline_runner.py -- --asset=wall_brick
  blender --background --python tools/blender/pipeline_runner.py -- --category="Furniture"
"""

import bpy
import json
import os
import sys

# Ensure tools/blender is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import render_iso
import snap_kit

# The 45-Asset Vertical Slice (Step I2)
VERTICAL_SLICE_IDS = {
    # Ground & Streets
    "grass", "dirt", "asphalt", "sidewalk", "kerb", "lane_line_white", "crosswalk",
    # Snap-kit walls, floors, roofs
    "wall_siding", "wall_siding_window", "wall_siding_door",
    "wall_brick", "wall_brick_window", "wall_brick_door",
    "floor_wood", "floor_concrete", "roof_gable_slope",
    # Furniture (10 pieces)
    "bed", "wardrobe", "fridge", "stove", "sink", "toilet", "sofa", "dining_table", "chair", "cupboard",
    # Vehicle
    "vehicle_car", "vehicle_hatchback",
    # Characters
    "body_base", "zombie_walker", "zombie",
    # Foliage & Props
    "tree_oak", "tree_pine", "bush", "rock_boulder", "dumpster", "street_light", "fence_chainlink", "mailbox", "trash_can",
    # Decals
    "blood_splat_small", "crack_ground", "leaves",
}

def slugify(text):
    return text.lower().replace(":", "").replace("&", "and").replace(" ", "_").replace("-", "_")

def run_pipeline():
    # Parse custom args after '--'
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]

    filter_slice = "--slice" in args
    filter_asset = None
    filter_category = None
    export_ue = "--unreal-export" in args

    for arg in args:
        if arg.startswith("--asset="):
            filter_asset = arg.split("=")[1]
        elif arg.startswith("--category="):
            filter_category = arg.split("=")[1]

    # Resolve paths
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    catalog_path = os.path.join(script_dir, "jobs", "asset_catalog.json")
    public_assets = os.path.join(repo_root, "public", "assets")
    ue_import_dir = os.path.join(repo_root, "tools", "unreal", "World4Bros", "Content", "Import")

    os.makedirs(public_assets, exist_ok=True)
    if export_ue:
        os.makedirs(ue_import_dir, exist_ok=True)

    if not os.path.exists(catalog_path):
        print(f"ERROR: Catalog not found at {catalog_path}. Run 'npm run art:catalog' first.")
        return

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    assets = catalog_data.get("assets", [])
    print(f"\n=======================================================")
    print(f"WORLD 4 BRO'S ASSET PIPELINE")
    print(f"Loaded {len(assets)} total assets from catalog.")
    print(f"Filter mode: slice={filter_slice}, asset={filter_asset}, category={filter_category}")
    print(f"=======================================================\n")

    # Load existing manifest
    manifest_path = os.path.join(public_assets, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception:
            manifest = {}

    processed_count = 0

    for asset in assets:
        a_id = asset["id"]
        a_cat = asset["category"]
        a_make = asset.get("make", "")
        cat_slug = slugify(a_cat)

        # Apply filtering
        if filter_slice and a_id not in VERTICAL_SLICE_IDS:
            continue
        if filter_asset and a_id != filter_asset:
            continue
        if filter_category and a_cat != filter_category:
            continue

        print(f"[{processed_count + 1}] Processing: {a_cat} -> {a_id}")

        # Reset scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Setup isometric camera & lighting (STYLE_BIBLE §0.8)
        ortho_scale = 3.6 if "vehicle" in a_id or "car" in a_id else 2.5
        render_iso.setup_iso_scene(resolution=256, ortho_scale=ortho_scale)

        # Create 3D geometry
        obj = snap_kit.create_model_by_id(a_id)

        # Determine direction count
        dirs = 16 if "16 directions" in a_make else 8 if "8 directions" in a_make else 1

        # Render sprites
        out_cat_dir = os.path.join(public_assets, cat_slug)
        rendered_paths = render_iso.render_turntable(obj, out_cat_dir, a_id, directions=dirs)

        # Optional GLB export for Unreal Engine testing
        glb_rel = None
        if export_ue:
            glb_path = os.path.join(ue_import_dir, f"{a_id}.glb")
            bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', use_selection=False)
            glb_rel = os.path.relpath(glb_path, repo_root).replace("\\", "/")

        # Record in manifest
        manifest[a_id] = {
            "id": a_id,
            "category": a_cat,
            "category_slug": cat_slug,
            "directions": dirs,
            "files": [os.path.relpath(p, repo_root).replace("\\", "/") for p in rendered_paths],
            "glb": glb_rel,
            "status": "done",
            "anchor": [0.5, 1.0],  # Ground contact point
        }

        processed_count += 1

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"\nPipeline batch complete! Processed {processed_count} assets.")
    print(f"Manifest written to: {manifest_path}\n")

if __name__ == "__main__":
    run_pipeline()
