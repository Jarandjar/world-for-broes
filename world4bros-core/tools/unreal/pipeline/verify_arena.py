"""
Checks a built arena level against arenas.json (run after build_arena.py; tested on UE 5.7.3 headless).
Fails (non-zero exit) if counts, tags or collision profiles do not match the plan.

  UnrealEditor-Cmd.exe <Project>.uproject -run=pythonscript -script="<this file> --arena all" -unattended -nullrhi -nosplash -nosound -stdout
"""
import argparse
import json
import os
import sys

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_arena as ba  # noqa: E402  (same folder)


def check(arena_id, a, map_root):
    path = "%s/%s" % (map_root, ba.map_name(arena_id))
    if not unreal.EditorAssetLibrary.does_asset_exist(path):
        return ["%s: level %s does not exist" % (arena_id, path)], {}
    world = unreal.EditorLoadingAndSavingUtils.load_map(path)
    actors = unreal.GameplayStatics.get_all_actors_with_tag(world, ba.TAG)
    plan = ba.plan_arena(a)
    expected = ba.summarize(plan)
    found = {}
    problems = []
    profiles = {p["kind"]: p.get("profile") for p in plan}
    for actor in actors:
        kinds = [str(t)[len("W4B.Kind."):] for t in actor.tags if str(t).startswith("W4B.Kind.")]
        kind = kinds[0] if kinds else "?"
        found[kind] = found.get(kind, 0) + 1
        if isinstance(actor, unreal.StaticMeshActor) and profiles.get(kind):
            prof = str(actor.static_mesh_component.get_collision_profile_name())
            if prof != profiles[kind]:
                problems.append("%s: %s has profile %s, want %s" % (arena_id, actor.get_actor_label(), prof, profiles[kind]))
    for kind, n in expected.items():
        if found.get(kind, 0) != n:
            problems.append("%s: %d %s actors, plan says %d" % (arena_id, found.get(kind, 0), kind, n))
    spawns = unreal.GameplayStatics.get_all_actors_with_tag(world, "W4B.Spawn")
    if len(spawns) != len(a["spawns"]):
        problems.append("%s: %d spawns, json has %d" % (arena_id, len(spawns), len(a["spawns"])))
    return problems, found


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default="all")
    ap.add_argument("--json", default=ba.DEFAULT_JSON)
    ap.add_argument("--map-root", default="/Game/W4B/Maps")
    args = ap.parse_args(argv)
    with open(args.json, "r", encoding="utf-8") as f:
        arenas = json.load(f)["arenas"]
    ids = list(arenas) if args.arena == "all" else [args.arena]
    all_problems = []
    for arena_id in ids:
        problems, found = check(arena_id, arenas[arena_id], args.map_root)
        all_problems += problems
        print("W4B_VERIFY %s %s %s" % (arena_id, "OK" if not problems else "FAIL", json.dumps(found, sort_keys=True)))
    for p in all_problems:
        unreal.log_error("W4B_VERIFY " + p)
    if all_problems:
        raise RuntimeError("%d arena problems" % len(all_problems))


if __name__ == "__main__":
    main(sys.argv[1:])
