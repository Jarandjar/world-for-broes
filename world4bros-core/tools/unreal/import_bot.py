import unreal
import os

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blender_dir = os.path.join(os.path.dirname(script_dir), "blender")
    bot_glb = os.path.abspath(os.path.join(blender_dir, "bot_jax.glb"))
    
    task = unreal.AssetImportTask()
    task.filename = bot_glb
    task.destination_path = "/Game/Characters/Bot"
    task.destination_name = "SM_Bot"
    task.replace_existing = True
    task.automated = True
    task.save = True
    
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    print("[IMPORT_DONE] Bot imported!")

if __name__ == "__main__":
    run()
