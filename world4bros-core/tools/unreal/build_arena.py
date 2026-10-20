"""
Unreal Engine Arena Builder for World 4 Bro's (Step U3)
Reads tools/unreal/arenas.json and procedurally generates tactical diorama arena maps:
- Converts metric coordinates (1m = 100 uu, game x -> Unreal X, game y -> Unreal -Y).
- Clears actors tagged 'W4BGenerated' for clean rebuilds.
- Places floor, crates, sandbags, low walls, barrels, trees, and perimeter walls.
- Places Player Starts and Health Pickups (pies).
- Sets up NW directional lighting, skylight, and isometric camera.
- Spawns tactical bots in cover and marching zombie waves.
- Saves level to /Game/Maps/L_W4B_<ArenaId>.

Usage:
  UnrealEditor.exe World4Bros.uproject -run=pythonscript -script="tools/unreal/build_arena.py" -- --arena=duel_pit
"""

import json
import os
import sys

try:
    import unreal
    IN_UNREAL = True
except ImportError:
    IN_UNREAL = False

def build_arena_map(arena_id="duel_pit"):
    if not IN_UNREAL:
        print("[ERROR] build_arena.py must be executed within Unreal Engine Python environment.")
        return

    print("=" * 65)
    print(f"  WORLD 4 BRO'S: HEADLESS ARENA BUILDER (Step U3)")
    print(f"  Constructing Arena Map: '{arena_id}' from arenas.json")
    print("=" * 65)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "arenas.json")

    if not os.path.exists(json_path):
        print(f"[ERROR] arenas.json not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arenas = data.get("arenas", {})
    if arena_id not in arenas:
        print(f"[ERROR] Arena '{arena_id}' not found in arenas.json. Available: {list(arenas.keys())}")
        return

    arena_spec = arenas[arena_id]
    w_m = arena_spec.get("width", 40)
    h_m = arena_spec.get("height", 24)
    mid_x = w_m / 2.0
    mid_y = h_m / 2.0

    # Coordinate transformation: (game_x, game_y) -> Unreal Vector
    def to_ue_pos(gx, gy, gz=0.0):
        # game x (East) -> +X, game y (South) -> -Y
        ux = (gx - mid_x) * 100.0
        uy = -(gy - mid_y) * 100.0
        uz = gz * 100.0
        return unreal.Vector(ux, uy, uz)

    # 1. Open or Create Level
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem) if hasattr(unreal, 'LevelEditorSubsystem') else None
    map_asset_path = f"/Game/Maps/L_W4B_{arena_id.title().replace('_', '')}"
    if level_subsystem:
        try:
            if unreal.EditorAssetLibrary.does_asset_exist(map_asset_path):
                level_subsystem.load_level(map_asset_path)
                print(f"[ARENA-BUILD] Loaded Existing Map via LevelEditorSubsystem: {map_asset_path}")
            else:
                level_subsystem.new_level(map_asset_path)
                print(f"[ARENA-BUILD] Created New Map via LevelEditorSubsystem: {map_asset_path}")
        except Exception as e:
            print(f"[ARENA-BUILD] LevelEditorSubsystem note: {e}. Spawning in current active world.")
    else:
        print("[ARENA-BUILD] Spawning in current active world.")

    # Guard: EditorLevelLibrary spawn/query calls dereference GEditor's world and
    # crash (ACCESS_VIOLATION) when none is loaded (e.g. bare commandlet mode).
    world = None
    try:
        ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        world = ues.get_editor_world() if ues else None
    except Exception as e:
        print(f"[ARENA-BUILD] World lookup failed: {e}")
    if world is None:
        print("[ERROR] No valid editor world; refusing to spawn actors (would crash). Aborting.")
        return

    # 2. Clear Existing Generated Actors
    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    cleared = 0
    for act in all_actors:
        if act.actor_has_tag("W4BGenerated") or act.is_a(unreal.PlayerStart):
            unreal.EditorLevelLibrary.destroy_actor(act)
            cleared += 1
    if cleared > 0:
        print(f"[ARENA-BUILD] Cleared {cleared} existing W4BGenerated actors.")

    # 3. Load Available Static Meshes
    bot_mesh = unreal.EditorAssetLibrary.load_asset("/Game/Characters/Bot/bot_jax/StaticMeshes/SM_Bot")
    zombie_mesh = unreal.EditorAssetLibrary.load_asset("/Game/Characters/Zombie/zombie/StaticMeshes/SM_Zombie")
    arena_base_mesh = unreal.EditorAssetLibrary.load_asset("/Game/Arena/duel_pit_arena/StaticMeshes/SM_DuelPitArena")

    for name, mesh in (("SM_Bot", bot_mesh), ("SM_Zombie", zombie_mesh), ("SM_DuelPitArena", arena_base_mesh)):
        if mesh is None:
            print(f"[ARENA-BUILD] WARNING: asset {name} missing; skipping its spawns.")

    def tag_actor(act, label):
        if act:
            act.tags.append("W4BGenerated")
            act.set_actor_label(f"W4B_{label}")

    cube_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube")
    sphere_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Sphere")
    cylinder_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cylinder")

    def set_mesh(act, mesh):
        if act and mesh:
            act.static_mesh_component.set_static_mesh(mesh)

    # 4. Spawn perfectly flat procedural floor (no sculpted base mesh)
    floor_act = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    if floor_act:
        plane_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Plane")
        if plane_mesh:
            floor_act.static_mesh_component.set_static_mesh(plane_mesh)
        # Engine plane is 100uu square; scale to arena size in meters
        floor_act.set_actor_scale3d(unreal.Vector(w_m, h_m, 1.0))
        tag_actor(floor_act, "Floor_Plane")

    # 5. Spawn Directional Light (NW Sun, Style Bible §0.2)
    # Sun from NW at 45 deg elevation: Pitch -45, Yaw 45
    sun = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 3000), unreal.Rotator(roll=0, pitch=-45, yaw=45))
    if sun:
        tag_actor(sun, "Sun_NW")
        sun_comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
        if sun_comp:
            sun_comp.set_intensity(6.0)
            sun_comp.set_light_color(unreal.LinearColor(1.0, 0.98, 0.95, 1.0))

    # Ambient Sky Fill
    sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 2500), unreal.Rotator(0, 0, 0))
    if sky:
        tag_actor(sky, "Sky_Ambient")

    # 6. Spawn Tactical Isometric Camera
    # Preset: strict isometric framing with map-size-aware orthographic width.
    import math
    ISO_PITCH = -35.264
    ISO_YAW = 135.0
    ISO_MARGIN = 0.78
    ortho_width = max(w_m, h_m) * 100.0 * ISO_MARGIN
    ISO_DIST = ortho_width * 3.0  # orthographic distance mostly impacts clipping and editor comfort
    pr, yr = math.radians(ISO_PITCH), math.radians(ISO_YAW)
    cam_pos = unreal.Vector(
        -math.cos(pr) * math.cos(yr) * ISO_DIST,
        -math.cos(pr) * math.sin(yr) * ISO_DIST,
        -math.sin(pr) * ISO_DIST)
    cam_rot = unreal.Rotator(roll=0, pitch=ISO_PITCH, yaw=ISO_YAW)
    cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CameraActor, cam_pos, cam_rot)
    if cam:
        tag_actor(cam, "Isometric_Camera")
        try:
            cam.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)
        except Exception as e:
            print(f"[ARENA-BUILD] Could not auto-activate camera for player: {e}")
        cam_comp = cam.get_component_by_class(unreal.CameraComponent)
        if cam_comp:
            cam_comp.set_editor_property("projection_mode", unreal.CameraProjectionMode.ORTHOGRAPHIC)
            cam_comp.set_editor_property("ortho_width", ortho_width)
            cam_comp.set_editor_property("ortho_far_clip_plane", ISO_DIST * 3.0)
        try:
            ues.set_level_viewport_camera_info(cam_pos, cam_rot)
            unreal.SystemLibrary.execute_console_command(world, "viewmode lit")
        except Exception as e:
            print(f"[ARENA-BUILD] Could not configure viewport camera/view mode: {e}")

    # 7. Spawn Crates (Block Movement & Bullets)
    for i, c in enumerate(arena_spec.get("crates", [])):
        cx, cy, cw, ch = c["x"], c["y"], c["w"], c["h"]
        pos = to_ue_pos(cx + cw / 2.0, cy + ch / 2.0, gz=0.6)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, unreal.Rotator(0, 0, 0))
        if actor:
            set_mesh(actor, cube_mesh)
            actor.set_actor_scale3d(unreal.Vector(cw, ch, 1.2))
            tag_actor(actor, f"Crate_{i+1}")

    # 8. Spawn Sandbags (Low Cover, Block Movement & Bullets)
    for i, s in enumerate(arena_spec.get("sandbags", [])):
        sx, sy, slen, sdir = s["x"], s["y"], s["length"], s.get("direction", "h")
        pos = to_ue_pos(sx + (slen / 2.0 if sdir == "h" else 0.5), sy + (0.5 if sdir == "h" else slen / 2.0), gz=0.45)
        scale = unreal.Vector(slen, 0.8, 0.9) if sdir == "h" else unreal.Vector(0.8, slen, 0.9)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, unreal.Rotator(0, 0, 0))
        if actor:
            set_mesh(actor, cube_mesh)
            actor.set_actor_scale3d(scale)
            tag_actor(actor, f"Sandbag_{i+1}")

    # 9. Spawn Low Walls
    for i, lw in enumerate(arena_spec.get("lowWalls", [])):
        lx, ly, lw_w, lw_h = lw["x"], lw["y"], lw["w"], lw["h"]
        pos = to_ue_pos(lx + lw_w / 2.0, ly + lw_h / 2.0, gz=0.5)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, unreal.Rotator(0, 0, 0))
        if actor:
            set_mesh(actor, cube_mesh)
            actor.set_actor_scale3d(unreal.Vector(lw_w, lw_h, 1.0))
            tag_actor(actor, f"LowWall_{i+1}")

    # 10. Spawn Barrels (Movement Blocker, Bullets Penetrate)
    for i, b in enumerate(arena_spec.get("barrels", [])):
        pos = to_ue_pos(b["x"] + 0.5, b["y"] + 0.5, gz=0.45)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, unreal.Rotator(0, 0, 0))
        if actor:
            set_mesh(actor, cylinder_mesh or cube_mesh)
            actor.set_actor_scale3d(unreal.Vector(0.7, 0.7, 0.9))
            tag_actor(actor, f"Barrel_{i+1}")

    # 11. Spawn Trees (Movement Blocker, Bullets Pass Through)
    for i, t in enumerate(arena_spec.get("trees", [])):
        pos = to_ue_pos(t["x"] + 0.5, t["y"] + 0.5, gz=2.0)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, unreal.Rotator(0, 0, 0))
        if actor:
            set_mesh(actor, cylinder_mesh or cube_mesh)
            actor.set_actor_scale3d(unreal.Vector(1.2, 1.2, 4.0))
            tag_actor(actor, f"Tree_{i+1}")

    # 12. Spawn Health Pickups (Pies, Restores 25 HP)
    for i, p in enumerate(arena_spec.get("pies", [])):
        pos = to_ue_pos(p["x"] + 0.5, p["y"] + 0.5, gz=0.25)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, pos, unreal.Rotator(0, 45, 0))
        if actor:
            set_mesh(actor, sphere_mesh or cube_mesh)
            actor.set_actor_scale3d(unreal.Vector(0.35, 0.35, 0.35))
            tag_actor(actor, f"HealthPickup_Pie_{i+1}")

    # 13. Place a deterministic PlayerStart in a clear lane for PIE testing.
    start_tx, start_ty = 11.0, h_m / 2.0
    pstart_pos = to_ue_pos(start_tx, start_ty, gz=0.5)
    pstart = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.PlayerStart,
        pstart_pos,
        unreal.Rotator(roll=0, pitch=0, yaw=0)
    )
    if pstart:
        tag_actor(pstart, "PlayerStart")

    # 14. Spawn Raider Bots at arena-defined spawn points
    spawn_points = arena_spec.get("spawns", [])
    if bot_mesh:
        for i, sp in enumerate(spawn_points[:4]):
            b_pos = to_ue_pos(sp["x"], sp["y"], gz=0.2)
            b_act = unreal.EditorLevelLibrary.spawn_actor_from_object(bot_mesh, b_pos, unreal.Rotator(roll=0, pitch=0, yaw=0))
            if b_act:
                b_act.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
                tag_actor(b_act, f"Bot_Raider_{i+1}")
                print(f"[ARENA-BUILD] Spawned Bot {i+1} at {b_pos}")

    # 15. Spawn 12 Zombies in a right-side wave that scales with arena size
    zombie_coords = []
    for xf in (0.70, 0.78, 0.86):
        for yf in (0.2, 0.4, 0.6, 0.8):
            zombie_coords.append((w_m * xf, h_m * yf))

    if zombie_mesh:
        for i, (zx, zy) in enumerate(zombie_coords):
            z_pos = to_ue_pos(zx, zy, gz=0.2)
            z_act = unreal.EditorLevelLibrary.spawn_actor_from_object(zombie_mesh, z_pos, unreal.Rotator(roll=0, pitch=0, yaw=180))
            if z_act:
                z_act.set_actor_scale3d(unreal.Vector(100.0, 100.0, 100.0))
                tag_actor(z_act, f"Zombie_Infected_{i+1}")
                print(f"[ARENA-BUILD] Spawned Zombie {i+1} at {z_pos}")

    # 16. Save Level
    if level_subsystem:
        try:
            level_subsystem.save_current_level()
            print(f"\n[ARENA-BUILD] Successfully constructed and saved map: {map_asset_path}")
        except Exception as e:
            print(f"\n[ARENA-BUILD] Note: save_current_level deferred: {e}")

    # 17. Configure Default Engine Settings to Boot this Map
    engine_ini = os.path.join(script_dir, "World4Bros", "Config", "DefaultEngine.ini")
    ini_content = f"""[/Script/EngineSettings.GeneralProjectSettings]
ProjectID=B4B749B24D98B6F3E6C18B991F2478A0
ProjectName=World4Bros Combat Arena
Description=Tactical Diorama Combat Arena with Bot PvP and Zombie Swarms

[/Script/EngineSettings.GameMapsSettings]
GameDefaultMap={map_asset_path}
EditorStartupMap={map_asset_path}

[/Script/Engine.RendererSettings]
r.DefaultFeature.AntiAliasing=2
r.AntiAliasingMethod=2
r.ShadowQuality=4
"""
    with open(engine_ini, "w", encoding="utf-8") as f:
        f.write(ini_content)
    print(f"[ARENA-BUILD] Set default startup map to: {map_asset_path}\n")

    shot_path = os.environ.get("W4B_SHOT")
    if shot_path:
        os.makedirs(os.path.dirname(shot_path), exist_ok=True)
        state = {"frames": 0, "handle": None}
        ctx = ues.get_editor_world()

        def _tick(delta):
            state["frames"] += 1
            if state["frames"] == 60:  # let the piloted ortho view and lighting settle
                unreal.SystemLibrary.execute_console_command(ctx, f'HighResShot 1920x1080 filename="{shot_path}"')
                print(f"[ARENA-BUILD] Screenshot requested: {shot_path}")
            elif state["frames"] == 240:
                print(f"[ARENA-BUILD] Screenshot exists: {os.path.exists(shot_path)}")
                unreal.SystemLibrary.execute_console_command(ctx, "QUIT_EDITOR")
                unreal.unregister_slate_post_tick_callback(state["handle"])

        state["handle"] = unreal.register_slate_post_tick_callback(_tick)

if __name__ == "__main__":
    arena_to_build = os.environ.get("W4B_ARENA_ID", "duel_pit")
    for arg in sys.argv:
        if arg.startswith("--arena="):
            arena_to_build = arg.split("=")[1]
    build_arena_map(arena_to_build)
