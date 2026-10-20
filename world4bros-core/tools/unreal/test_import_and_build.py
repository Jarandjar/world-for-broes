import unreal
import os

def import_asset(filename, destination_path):
    task = unreal.AssetImportTask()
    task.filename = filename
    task.destination_path = destination_path
    task.replace_existing = True
    task.automated = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    print(f"[UE-IMPORT] Task executed for: {filename} -> {destination_path}")

def run():
    print("[UE-BUILD] Testing Unreal Asset Import...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blender_dir = os.path.join(os.path.dirname(script_dir), "blender")
    zombie_glb = os.path.join(blender_dir, "zombie.glb")
    arena_glb = os.path.join(blender_dir, "duel_pit_arena.glb")
    
    import_asset(zombie_glb, "/Game/Characters")
    import_asset(arena_glb, "/Game/Arena")
    print("[UE-BUILD] Import finished.")

if __name__ == "__main__":
    run()
