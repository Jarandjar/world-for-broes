import unreal
import os

def test_interchange():
    print("[TEST] Testing InterchangeManager import...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blender_dir = os.path.join(os.path.dirname(script_dir), "blender")
    zombie_glb = os.path.abspath(os.path.join(blender_dir, "zombie.glb"))
    
    source_data = unreal.InterchangeManager.create_source_data(zombie_glb)
    interchange = unreal.InterchangeManager.get_interchange_manager()
    
    params = unreal.ImportAssetParameters()
    params.is_automated = True
    
    success = interchange.import_asset("/Game/Characters", source_data, params)
    print(f"[TEST] Interchange import success: {success}")

if __name__ == "__main__":
    test_interchange()
