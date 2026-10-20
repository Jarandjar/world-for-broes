"""
Takes a screenshot of a level from its preview camera (actor tagged W4B.PreviewCamera), then closes the editor.
Gives agents "eyes": they can open the PNG. Needs the full editor with rendering, not the -run=pythonscript commandlet.

  UnrealEditor.exe <Project>.uproject -ExecutePythonScript="<this file> --map /Game/W4B/Maps/L_W4B_DuelPit --out C:/W4B/shots/duel_pit.png"
      -RenderOffScreen -unattended -nosplash -nosound
  (-RenderOffScreen renders without opening a window. The first run compiles shaders and takes minutes.)
-ExecutePythonScript closes the editor as soon as the script returns, so the script sets keep-alive while it
waits for the shot (UEditorPythonScriptingLibrary, see EditorPythonExecuter.cpp) and quits when the PNG exists.
"""
import argparse
import os
import sys

import unreal

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", default="1600x900")
ap.add_argument("--camera-tag", default="W4B.PreviewCamera")
ap.add_argument("--settle-frames", type=int, default=90, help="frames to wait before the shot (lighting, streaming)")
ap.add_argument("--timeout-frames", type=int, default=6000)
args = ap.parse_args(sys.argv[1:])

res_x, res_y = (int(v) for v in args.res.lower().split("x"))
out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
world = unreal.EditorLoadingAndSavingUtils.load_map(args.map)
cams = unreal.GameplayStatics.get_all_actors_with_tag(world, args.camera_tag)
camera = cams[0] if cams else None
if camera is None:
    unreal.log_warning("W4B_SHOT no camera tagged %s, using the editor view" % args.camera_tag)

state = {"frame": 0, "task": None, "handle": None}


def finish(ok):
    unreal.unregister_slate_post_tick_callback(state["handle"])
    print("W4B_SHOT %s %s" % ("OK" if ok else "FAIL", out))
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    unreal.SystemLibrary.quit_editor()


def tick(_dt):
    state["frame"] += 1
    if state["task"] is None and state["frame"] >= args.settle_frames:
        state["task"] = unreal.AutomationLibrary.take_high_res_screenshot(res_x, res_y, out, camera)
    elif state["task"] is not None and state["task"].is_task_done():
        finish(os.path.exists(out))
    elif state["frame"] > args.timeout_frames:
        finish(False)


unreal.EditorPythonScripting.set_keep_python_script_alive(True)
state["handle"] = unreal.register_slate_post_tick_callback(tick)
