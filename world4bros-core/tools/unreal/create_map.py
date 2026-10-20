import unreal
import os

def create_arena_map():
    print("=" * 60)
    print("  WORLD4BROS: GENERATING ARENA LEVEL IN UNREAL ENGINE 5.7")
    print("  Populating 3D Scene with Arena, 4 Bots, and 12 Swarming Zombies")
    print("=" * 60)

    # Asset paths
    arena_mesh_path = "/Game/Arena/duel_pit_arena/StaticMeshes/SM_DuelPitArena"
    zombie_mesh_path = "/Game/Characters/Zombie/zombie/StaticMeshes/SM_Zombie"
    bot_mesh_path = "/Game/Characters/Bot/bot_jax/StaticMeshes/SM_Bot"

    arena_mesh = unreal.EditorAssetLibrary.load_asset(arena_mesh_path)
    zombie_mesh = unreal.EditorAssetLibrary.load_asset(zombie_mesh_path)
    bot_mesh = unreal.EditorAssetLibrary.load_asset(bot_mesh_path)

    print(f"[MAP-GEN] Loaded Arena: {arena_mesh}")
    print(f"[MAP-GEN] Loaded Zombie: {zombie_mesh}")
    print(f"[MAP-GEN] Loaded Bot: {bot_mesh}")

    map_path = "/Game/Maps/Arena_DuelPit"
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_subsystem.new_level(map_path)
    print(f"[MAP-GEN] Created New Level: {map_path}")

    # 1. Place Arena
    if arena_mesh:
        arena_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(arena_mesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        if arena_actor:
            arena_actor.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
            arena_actor.set_actor_label("Arena_DuelPit_Environment")
            print("[MAP-GEN] Placed Arena Mesh at (0,0,0) with 100x scale")

    # 2. Place Sunlight & Sky
    sun = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 2500), unreal.Rotator(-45, 45, 0))
    if sun:
        sun.set_actor_label("Sunlight_Directional")

    sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 2000), unreal.Rotator(0, 0, 0))
    if sky:
        sky.set_actor_label("SkyLight_Ambient")

    # 3. Spawn 4 Raider Bots in tactical cover positions (West side)
    bot_spawns = [
        unreal.Vector(-1400, 0, 20),      # Back spawn ring
        unreal.Vector(-700, -450, 20),    # Behind lower sandbag bank
        unreal.Vector(-700, 450, 20),     # Behind upper sandbag bank
        unreal.Vector(-350, 0, 20),       # Behind central military crate
    ]
    for i, pos in enumerate(bot_spawns):
        rot = unreal.Rotator(0, 0, 0) # Facing East towards the zombie horde
        b_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(bot_mesh, pos, rot) if bot_mesh else None
        if b_actor:
            b_actor.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
            b_actor.set_actor_label(f"Bot_Raider_{i+1}")
            print(f"[MAP-GEN] Placed Bot {i+1} at {pos}")

    # 4. Spawn 12 Quarantined Zombies in charging wave formation (East side)
    # Staggered wave marching West (facing 180 degrees)
    zombie_coords = [
        (500, -500), (500, -150), (500, 150), (500, 500),      # Front Wave (4 walkers)
        (850, -650), (850, -250), (850, 250), (850, 650),      # Middle Wave (4 runners)
        (1200, -400), (1200, -100), (1200, 100), (1200, 400),   # Rear Horde (4 brutes/runners)
    ]
    for idx, (zx, zy) in enumerate(zombie_coords):
        pos = unreal.Vector(float(zx), float(zy), 20.0)
        rot = unreal.Rotator(0, 180, 0) # Facing West directly at the bots
        z_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(zombie_mesh, pos, rot) if zombie_mesh else None
        if z_actor:
            z_actor.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
            z_actor.set_actor_label(f"Zombie_Infected_{idx+1}")
            print(f"[MAP-GEN] Placed Zombie {idx+1} at {pos}")

    # 5. Place Camera Actor (Isometric Tactical High-Angle View)
    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(0, -2600, 2200), unreal.Rotator(-36, 90, 0))
    if cam:
        cam.set_actor_label("Tactical_Isometric_Camera")
        print("[MAP-GEN] Placed Tactical Isometric Camera")

    # 6. Save level
    level_subsystem.save_current_level()
    print(f"[MAP-GEN] Successfully saved map to: {map_path}")

    # 7. Update DefaultEngine.ini to load this map by default in standalone -game
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_ini = os.path.join(script_dir, "World4Bros", "Config", "DefaultEngine.ini")
    
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
    print(f"[MAP-GEN] Updated DefaultEngine.ini with GameDefaultMap={map_path}")

if __name__ == "__main__":
    create_arena_map()
