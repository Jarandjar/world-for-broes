import unreal
import os

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blender_dir = os.path.join(os.path.dirname(script_dir), "blender")
    zombie_glb = os.path.abspath(os.path.join(blender_dir, "zombie.glb"))
    
    task = unreal.AssetImportTask()
    task.filename = zombie_glb
    task.destination_path = "/Game/Characters/Zombie"
    task.destination_name = "SM_Zombie"
    task.replace_existing = True
    task.automated = True
    task.save = True
    
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    print("[IMPORT_DONE] Zombie imported!")

if __name__ == "__main__":
    run()
