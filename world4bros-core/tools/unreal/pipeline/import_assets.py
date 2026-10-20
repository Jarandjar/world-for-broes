"""
Imports .glb/.gltf/.fbx files into Unreal headless and reports their real size, so wrong-scale exports are caught.
Tested on UE 5.7.3 and re-tested on 5.8.3 with the -run=pythonscript commandlet (docs/UE_PIPELINE.md section 6).

  ue.ps1 -Project <P>.uproject -Script tools/unreal/pipeline/import_assets.py -ScriptArgs "--manifest <file.json>"
  ue.ps1 -Project <P>.uproject -Script tools/unreal/pipeline/import_assets.py -ScriptArgs "--src C:/art/crate.glb --dest /Game/W4B/Props"

Manifest: {"assets": [{"src": "C:/.../crate.glb", "dest": "/Game/W4B/Props", "collision": "box", "expect_m": [1.2, 1.2, 1.2]}]}
  collision: "box" adds one simple box (cheap, good for props), "keep" leaves what the importer made.
  expect_m: optional size in metres; the import fails if the mesh is off by more than 25 % (catches 100x scale bugs).

Rules: glTF is metres and Interchange converts to centimetres, so never scale the actors afterwards.
The Content Browser sync is switched off first; with it on, the commandlet crashes after every import.
"""
import argparse
import json
import os
import sys

import unreal

unreal.SystemLibrary.execute_console_command(None, "Interchange.FeatureFlags.Import.SyncToBrowser 0")


def import_one(src, dest, replace=True):
    task = unreal.AssetImportTask()
    task.filename = src
    task.destination_path = dest
    task.replace_existing = replace
    task.automated = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return list(task.imported_object_paths)


def size_m(mesh):
    b = mesh.get_bounding_box()
    s = b.max - b.min
    return [round(s.x / 100.0, 3), round(s.y / 100.0, 3), round(s.z / 100.0, 3)]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--src")
    ap.add_argument("--dest", default="/Game/W4B/Imported")
    ap.add_argument("--collision", default="keep", choices=["keep", "box"])
    args = ap.parse_args(argv)

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            entries = json.load(f)["assets"]
    elif args.src:
        entries = [{"src": args.src, "dest": args.dest, "collision": args.collision}]
    else:
        raise SystemExit("pass --manifest or --src")

    # StaticMeshEditorSubsystem is not created in the commandlet; the older library does the same job there.
    sms = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem) or unreal.EditorStaticMeshLibrary
    problems = []
    report = []
    for e in entries:
        src = os.path.abspath(e["src"])
        if not os.path.exists(src):
            problems.append("missing file " + src)
            continue
        paths = import_one(src, e.get("dest", args.dest))
        meshes = [unreal.load_asset(p) for p in paths]
        meshes = [m for m in meshes if isinstance(m, unreal.StaticMesh)]
        if not meshes:
            problems.append("no static mesh imported from " + src)
        for m in meshes:
            if e.get("collision", "keep") == "box":
                sms.remove_collisions(m)
                sms.add_simple_collisions(m, unreal.ScriptCollisionShapeType.BOX)
                unreal.EditorAssetLibrary.save_loaded_asset(m)
            got = size_m(m)
            item = {"src": src, "asset": m.get_path_name(), "size_m": got}
            want = e.get("expect_m")
            if want:
                worst = max(abs(g - w) / max(w, 0.01) for g, w in zip(got, want))
                item["size_error"] = round(worst, 3)
                if worst > 0.25:
                    problems.append("%s is %s m, expected %s m" % (m.get_path_name(), got, want))
            elif max(got) > 60 or max(got) < 0.02:
                problems.append("%s is %s m: probably a scale bug (glTF must be metres)" % (m.get_path_name(), got))
            report.append(item)
            print("W4B_IMPORT %s %s" % (item["asset"], json.dumps(item["size_m"])))

    out_dir = os.path.join(unreal.Paths.project_saved_dir(), "W4B")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "import_report.json"), "w", encoding="utf-8") as f:
        json.dump({"assets": report, "problems": problems}, f, indent=2)
    for p in problems:
        unreal.log_error("W4B_IMPORT " + p)
    if problems:
        raise RuntimeError("%d import problems" % len(problems))
    print("W4B_IMPORT_OK %d meshes" % len(report))


if __name__ == "__main__":
    main(sys.argv[1:])
