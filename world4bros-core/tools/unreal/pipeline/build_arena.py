"""
W4B arena builder: turns tools/unreal/arenas.json (npm run arena:export) into Unreal levels.
Tested headless on UE 5.7.3 and re-tested on 5.8.3 (docs/UE_PIPELINE.md section 6). Engine classes only, so it works in any project.

Headless (no editor window, ~15 s):
  UnrealEditor-Cmd.exe <Project>.uproject -run=pythonscript -script="<this file> --arena all"
      -unattended -nullrhi -nosplash -nosound -stdout
  or: powershell tools/unreal/pipeline/ue.ps1 -Project <Project>.uproject -Script build_arena.py -ScriptArgs "--arena all"
Plain Python (no Unreal): prints the placement plan, for checks and tests.
  python tools/unreal/pipeline/build_arena.py --arena duel_pit --plan

What it builds per arena (all tagged W4BGenerated; only those are deleted on a rebuild, hand-placed art stays):
  floor, perimeter walls, cover blockouts (crates, low walls, sandbags: stop bullets), barrels and trees
  (block movement, bullets pass), PlayerStarts facing the centre, pickup markers (TargetPoint tagged
  W4B.Pickup.Pie), a NavMeshBoundsVolume, NW sun, sky light, atmosphere, light fog, and a preview camera.
Collision uses named profiles (W4BCover, W4BBulletPass) defined in the project's DefaultEngine.ini,
because Python cannot address custom collision channels directly.

Traps this script avoids (all seen in real crashes or wrong results):
  - EditorActorSubsystem.spawn_actor_from_object(<mesh>) and the old EditorLevelLibrary spawns crash headless;
    spawn_actor_from_class + set_static_mesh works.
  - unreal.Rotator takes (roll, pitch, yaw) positionally; always pass keywords.
  - glTF/FBX imports are already in centimetres; never scale actors by 100.
"""
import argparse
import json
import math
import os
import sys

try:
    import unreal  # only inside Unreal
except ImportError:
    unreal = None

TAG = "W4BGenerated"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.environ.get("W4B_ARENAS_JSON", os.path.normpath(os.path.join(SCRIPT_DIR, "..", "arenas.json")))

# Gameplay heights in metres. Bullets and bot sight run in a flat combat plane at COMBAT_PLANE_M, so every
# bullet-stopping kind must be taller than it (the web sim is 2D: all of these stop bullets there).
COMBAT_PLANE_M = 0.9
KINDS = {
    "crate":    {"height": 1.4, "profile": "W4BCover",      "mesh": "cube"},
    "low_wall": {"height": 1.1, "profile": "W4BCover",      "mesh": "cube"},
    "sandbags": {"height": 1.0, "profile": "W4BCover",      "mesh": "cube"},
    "barrel":   {"height": 1.0, "profile": "W4BBulletPass", "mesh": "cylinder"},
    "tree":     {"height": 4.0, "profile": "W4BBulletPass", "mesh": "cylinder"},
}
MESHES = {
    "cube": "/Engine/BasicShapes/Cube.Cube",          # 100 cm, pivot at centre
    "cylinder": "/Engine/BasicShapes/Cylinder.Cylinder",
    "plane": "/Engine/BasicShapes/Plane.Plane",        # 100 x 100 cm at z = 0
}
WALL_HEIGHT_M = 3.0
WALL_THICK_M = 0.5
# Camera and sun are screen-relative. Camera yaw -90 looks game-north (screen up = north); -45 turns the view
# 45 degrees like Project Zomboid. The sun shines from behind-left of the camera so both walls that face the
# camera are lit, left brighter than right (STYLE_BIBLE.md section 0: about 85 % / 62 % of the top); its
# shadows fall away from the camera, slightly right. A sun on the far side (shadows to screen lower-right)
# leaves every camera-facing wall dark: tested 2026-10-20, see docs/UE_PIPELINE.md section 6.
CAMERA = {"yaw_deg": -45.0, "pitch_deg": -55.0}
SUN = {"rel_yaw_deg": 10.0, "elevation_deg": 50.0, "intensity_lux": 6.0}


ACRONYMS = {"ffa", "tdm", "ctf"}


def map_name(arena_id):
    """duel_pit -> L_W4B_DuelPit, ffa_yard -> L_W4B_FFAYard"""
    return "L_W4B_" + "".join(p.upper() if p in ACRONYMS else p.capitalize() for p in arena_id.split("_"))


def to_ue(a, gx, gy):
    """Game metres (x east, y south, origin top-left) -> Unreal cm, origin at the arena centre.
    Unreal is left-handed: seen from above with +X to the right, +Y points down the screen, exactly like
    the web game's x/y. So there is no flip; flipping Y would mirror the arenas."""
    return ((gx - a["width"] / 2.0) * 100.0, (gy - a["height"] / 2.0) * 100.0)


def sun_yaw(camera_yaw):
    """Direction the light travels. Camera forward is screen-up, +90 yaw is screen-right (Unreal is left-handed)."""
    return camera_yaw + SUN["rel_yaw_deg"]


def plan_arena(a):
    """Pure placement plan (no Unreal calls): list of dicts with kind, mesh, location/scale/yaw in Unreal units."""
    plan = []
    w, h = a["width"], a["height"]

    def box(kind, gx, gy, gw, gh, height, profile, mesh="cube", tags=()):
        x, y = to_ue(a, gx + gw / 2.0, gy + gh / 2.0)
        plan.append({"kind": kind, "mesh": mesh, "loc": (x, y, height * 50.0), "scale": (gw, gh, height),
                     "yaw": 0.0, "profile": profile, "tags": list(tags)})

    # Floor (top at z = 0) and roads (visual only).
    box("floor", -1, -1, w + 2, h + 2, 0.2, "BlockAll")
    plan[-1]["loc"] = (plan[-1]["loc"][0], plan[-1]["loc"][1], -10.0)
    for i, r in enumerate(a.get("roads", []) or []):
        x, y = to_ue(a, r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0)
        plan.append({"kind": "road", "mesh": "plane", "loc": (x, y, 0.5), "scale": (r["w"], r["h"], 1.0),
                     "yaw": 0.0, "profile": "NoCollision", "tags": ["W4B.Road"]})

    # Perimeter walls just outside the play area.
    t = WALL_THICK_M
    box("wall", -t, -t, w + 2 * t, t, WALL_HEIGHT_M, "W4BCover")
    box("wall", -t, h, w + 2 * t, t, WALL_HEIGHT_M, "W4BCover")
    box("wall", -t, 0, t, h, WALL_HEIGHT_M, "W4BCover")
    box("wall", w, 0, t, h, WALL_HEIGHT_M, "W4BCover")

    solids = a.get("solids")
    rects = solids["rects"] if solids else (
        [dict(c, kind="crate") for c in a["crates"]] + [dict(c, kind="low_wall") for c in a["lowWalls"]] +
        [{"x": s["x"], "y": s["y"] + 0.25, "w": s["length"], "h": 0.5, "kind": "sandbags"} if s["direction"] == "h"
         else {"x": s["x"] + 0.25, "y": s["y"], "w": 0.5, "h": s["length"], "kind": "sandbags"} for s in a["sandbags"]])
    circles = solids["circles"] if solids else (
        [dict(b, radius=0.45, kind="barrel") for b in a["barrels"]] + [dict(t_, radius=0.5, kind="tree") for t_ in a["trees"]])

    for r in rects:
        k = KINDS[r["kind"]]
        box(r["kind"], r["x"], r["y"], r["w"], r["h"], k["height"], k["profile"], k["mesh"], ["W4B.Cover", "W4B." + r["kind"]])
    for c in circles:
        k = KINDS[c["kind"]]
        x, y = to_ue(a, c["x"], c["y"])
        d = c["radius"] * 2.0
        plan.append({"kind": c["kind"], "mesh": k["mesh"], "loc": (x, y, k["height"] * 50.0), "scale": (d, d, k["height"]),
                     "yaw": 0.0, "profile": k["profile"], "tags": ["W4B." + c["kind"]]})

    for i, s in enumerate(a["spawns"]):
        x, y = to_ue(a, s["x"], s["y"])
        plan.append({"kind": "spawn", "loc": (x, y, 100.0), "yaw": math.degrees(math.atan2(-y, -x)),
                     "tags": ["W4B.Spawn", "W4B.Spawn.%d" % (i + 1)]})
    for i, p in enumerate(a["pies"]):
        x, y = to_ue(a, p["x"], p["y"])
        plan.append({"kind": "pickup", "loc": (x, y, 40.0), "yaw": 0.0, "tags": ["W4B.Pickup.Pie", "W4B.Pickup.%d" % (i + 1)]})

    plan.append({"kind": "nav", "loc": (0.0, 0.0, 100.0), "scale": ((w + 4) / 2.0, (h + 4) / 2.0, 3.0), "yaw": 0.0, "tags": []})
    return plan


def summarize(plan):
    counts = {}
    for p in plan:
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
    return counts


# ---------------------------------------------------------------- Unreal side

def _tag(actor, label, extra):
    actor.tags = [TAG] + list(extra)
    actor.set_actor_label(label)


def apply_plan(arena_id, a, plan, map_root):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    path = "%s/%s" % (map_root, map_name(arena_id))

    if unreal.EditorAssetLibrary.does_asset_exist(path):
        world = unreal.EditorLoadingAndSavingUtils.load_map(path)
        removed = 0
        for actor in unreal.GameplayStatics.get_all_actors_with_tag(world, TAG):
            eas.destroy_actor(actor)
            removed += 1
        unreal.log("W4B_BUILD removed %d old generated actors from %s" % (removed, path))
    else:
        if not les.new_level(path, False):
            raise RuntimeError("new_level failed for " + path)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    meshes = {k: unreal.load_asset(v) for k, v in MESHES.items()}
    n = 0
    for p in plan:
        n += 1
        loc = unreal.Vector(*p["loc"])
        rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=p.get("yaw", 0.0))
        kind = p["kind"]
        if kind == "spawn":
            actor = eas.spawn_actor_from_class(unreal.PlayerStart, loc, rot)
        elif kind == "pickup":
            actor = eas.spawn_actor_from_class(unreal.TargetPoint, loc, rot)
        elif kind == "nav":
            actor = eas.spawn_actor_from_class(unreal.NavMeshBoundsVolume, loc, rot)
            actor.set_actor_scale3d(unreal.Vector(*p["scale"]))  # default brush is 200 cm
        else:
            actor = eas.spawn_actor_from_class(unreal.StaticMeshActor, loc, rot)
            comp = actor.static_mesh_component
            comp.set_static_mesh(meshes[p["mesh"]])
            actor.set_actor_scale3d(unreal.Vector(*p["scale"]))
            comp.set_collision_profile_name(p["profile"])
        _tag(actor, "W4B_%s_%02d" % (kind, n), ["W4B.Kind." + kind] + p.get("tags", []))

    # Sun: placed relative to the game camera (see SUN above).
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 2000),
                                     unreal.Rotator(roll=0.0, pitch=-SUN["elevation_deg"], yaw=sun_yaw(CAMERA["yaw_deg"])))
    sc = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("intensity", SUN["intensity_lux"])
    sc.set_editor_property("atmosphere_sun_light", True)
    _tag(sun, "W4B_Sun", ["W4B.Kind.light"])
    for cls, label in ((unreal.SkyAtmosphere, "W4B_SkyAtmosphere"), (unreal.SkyLight, "W4B_SkyLight"),
                       (unreal.ExponentialHeightFog, "W4B_Fog")):
        actor = eas.spawn_actor_from_class(cls, unreal.Vector(0, 0, 500), unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0))
        if cls == unreal.SkyLight:
            c = actor.get_component_by_class(unreal.SkyLightComponent)
            c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
            c.set_editor_property("real_time_capture", True)
        _tag(actor, label, ["W4B.Kind.light"])

    # Preview camera at the game angle, aimed at the arena centre, for screenshots and the art pass.
    yaw, pitch = math.radians(CAMERA["yaw_deg"]), math.radians(-CAMERA["pitch_deg"])
    dist = max(a["width"], a["height"]) * 100.0 * 0.8
    cam_loc = unreal.Vector(-math.cos(yaw) * math.cos(pitch) * dist, -math.sin(yaw) * math.cos(pitch) * dist,
                            math.sin(pitch) * dist)
    cam = eas.spawn_actor_from_class(unreal.CameraActor, cam_loc,
                                     unreal.Rotator(roll=0.0, pitch=CAMERA["pitch_deg"], yaw=CAMERA["yaw_deg"]))
    _tag(cam, "W4B_PreviewCamera", ["W4B.Kind.camera", "W4B.PreviewCamera"])

    if not unreal.EditorLoadingAndSavingUtils.save_map(world, path):
        raise RuntimeError("save_map failed for " + path)
    return path


def main(argv):
    ap = argparse.ArgumentParser(description="Build W4B arena levels from arenas.json")
    ap.add_argument("--arena", default="all", help="duel_pit, ffa_yard or all")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--map-root", default="/Game/W4B/Maps")
    ap.add_argument("--plan", action="store_true", help="print the plan only (works without Unreal)")
    ap.add_argument("--camera-yaw", type=float, default=CAMERA["yaw_deg"], help="game camera yaw; the sun follows it")
    args = ap.parse_args(argv)
    CAMERA["yaw_deg"] = args.camera_yaw

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    arenas = data["arenas"]
    ids = list(arenas) if args.arena == "all" else [args.arena]
    report = {"json": args.json, "version": data.get("version"), "arenas": {}}
    for arena_id in ids:
        if arena_id not in arenas:
            raise SystemExit("unknown arena %r, have %s" % (arena_id, list(arenas)))
        a = arenas[arena_id]
        plan = plan_arena(a)
        counts = summarize(plan)
        entry = {"map": "%s/%s" % (args.map_root, map_name(arena_id)), "counts": counts}
        if not args.plan:
            if unreal is None:
                raise SystemExit("run inside Unreal, or pass --plan")
            entry["map"] = apply_plan(arena_id, a, plan, args.map_root)
        report["arenas"][arena_id] = entry
        print("W4B_BUILD_OK %s %s %s" % (arena_id, entry["map"], json.dumps(counts, sort_keys=True)))

    if unreal is not None and not args.plan:
        out_dir = os.path.join(unreal.Paths.project_saved_dir(), "W4B")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "build_arena_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
