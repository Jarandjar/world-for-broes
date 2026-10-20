import unreal
import os

def import_glb(filename, destination_path, asset_name):
    task = unreal.AssetImportTask()
    task.filename = os.path.abspath(filename)
    task.destination_path = destination_path
    task.destination_name = asset_name
    task.replace_existing = True
    task.automated = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    print(f"[UE-SETUP] Imported: {filename} -> {destination_path}/{asset_name}")

def build_arena_level(num_bots=4, num_zombies=12):
    print("=" * 60)
    print("  WORLD4BROS: BUILDING UNREAL ENGINE 5.7 ARENA LEVEL")
    print(f"  Spawning {num_bots} Bots and {num_zombies} Zombies in 3D Level")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    blender_dir = os.path.join(os.path.dirname(script_dir), "blender")
    
    arena_glb = os.path.join(blender_dir, "duel_pit_arena.glb")
    zombie_glb = os.path.join(blender_dir, "zombie.glb")
    bot_glb = os.path.join(blender_dir, "bot_jax.glb")

    # 1. Import meshes
    import_glb(arena_glb, "/Game/Arena", "SM_DuelPitArena")
    import_glb(zombie_glb, "/Game/Characters", "SM_Zombie")
    import_glb(bot_glb, "/Game/Characters", "SM_Bot")

    # 2. Create or open map
    map_path = "/Game/Maps/Arena_DuelPit"
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_subsystem.new_level(map_path)
    print(f"[UE-SETUP] Created Level: {map_path}")

    # Load static mesh assets
    arena_mesh = unreal.EditorAssetLibrary.load_asset("/Game/Arena/SM_DuelPitArena")
    zombie_mesh = unreal.EditorAssetLibrary.load_asset("/Game/Characters/SM_Zombie")
    bot_mesh = unreal.EditorAssetLibrary.load_asset("/Game/Characters/SM_Bot")

    # 3. Spawn Arena static mesh at (0, 0, 0) - scale 100 for Unreal units (cm)
    if arena_mesh:
        arena_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(arena_mesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        if arena_actor:
            arena_actor.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
            arena_actor.set_actor_label("Arena_DuelPit_Base")
            print("[UE-SETUP] Placed Arena Mesh at origin (Scale 100x)")

    # 4. Spawn Lighting & Environment
    sun_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 2000), unreal.Rotator(-45, 45, 0))
    if sun_actor:
        sun_actor.set_actor_label("Sunlight_Directional")

    skylight_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1500), unreal.Rotator(0, 0, 0))
    if skylight_actor:
        skylight_actor.set_actor_label("SkyLight_Ambient")

    # 5. Spawn 4 Raider Bots in tactical cover positions (West side of arena, around X = -1000 to -400)
    bot_spawns = [
        unreal.Vector(-1200, 0, 50),     # Back spawn pad
        unreal.Vector(-600, -400, 50),   # Behind lower sandbag
        unreal.Vector(-600, 400, 50),    # Behind upper sandbag
        unreal.Vector(-300, 0, 50),      # Center crate cover
    ]
    for i in range(min(num_bots, len(bot_spawns))):
        pos = bot_spawns[i]
        rot = unreal.Rotator(0, 0, 0) # Facing East towards zombies
        if bot_mesh:
            b_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(bot_mesh, pos, rot)
        else:
            b_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, rot)
        if b_actor:
            b_actor.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
            b_actor.set_actor_label(f"Bot_Raider_{i+1}")
            print(f"[UE-SETUP] Spawned Bot {i+1} at {pos}")

    # 6. Spawn 12 Quarantined Zombies (East side of arena, swarming westward towards bots)
    # Positions spaced across X = 400 to 1400, Y = -800 to 800
    zombie_cols = 4
    zombie_rows = 3
    z_count = 0
    for col in range(zombie_cols):
        for row in range(zombie_rows):
            if z_count >= num_zombies:
                break
            zx = 400.0 + col * 300.0
            zy = -600.0 + row * 600.0 + ((col % 2) * 150.0) # staggered horde wave
            zz = 50.0
            pos = unreal.Vector(zx, zy, zz)
            rot = unreal.Rotator(0, 180, 0) # Facing West towards bots!
            
            if zombie_mesh:
                z_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(zombie_mesh, pos, rot)
            else:
                z_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, rot)
                
            if z_actor:
                z_actor.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
                z_actor.set_actor_label(f"Zombie_Horde_{z_count+1}")
                print(f"[UE-SETUP] Spawned Zombie {z_count+1} at {pos}")
            z_count += 1

    # 7. Spawn Isometric Camera overlooking the battleground
    cam_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(0, -2200, 1800), unreal.Rotator(-35, 90, 0))
    if cam_actor:
        cam_actor.set_actor_label("Arena_IsometricCamera")
        print("[UE-SETUP] Placed Isometric Camera Actor")

    # 8. Save Level
    level_subsystem.save_current_level()
    print(f"[UE-SETUP] Successfully saved level to: {map_path}")

    # 9. Update DefaultEngine.ini to load this map by default
    config_dir = os.path.join(script_dir, "World4Bros", "Config")
    engine_ini = os.path.join(config_dir, "DefaultEngine.ini")
    
    ini_content = """[/Script/EngineSettings.GeneralProjectSettings]
ProjectID=B4B749B24D98B6F3E6C18B991F2478A0
ProjectName=World4Bros Combat Arena
Description=Tactical Diorama Combat Arena with Bot PvP and Zombie Swarms

[/Script/EngineSettings.GameMapsSettings]
GameDefaultMap=/Game/Maps/Arena_DuelPit
EditorStartupMap=/Game/Maps/Arena_DuelPit

[/Script/Engine.RendererSettings]
r.DefaultFeature.AntiAliasing=2
r.ShadowQuality=5
r.GlobalIllumination.Method=1
r.Reflections.Method=1

[SystemSettings]
r.SetRes=1280x720w
"""
    with open(engine_ini, "w", encoding="utf-8") as f:
        f.write(ini_content)
    print(f"[UE-SETUP] Updated {engine_ini} to load {map_path} on startup!")

if __name__ == "__main__":
    build_arena_level()
