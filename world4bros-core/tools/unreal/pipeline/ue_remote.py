"""
Runs Python inside an Unreal Editor that is already open (Unreal's built-in Python remote execution).
Lets an agent change the open editor without restarting it: rebuild a level, trigger Live Coding, take a shot.
Tested on UE 5.7.3 (docs/UE_PIPELINE.md section 6). Needs, in the project's DefaultEngine.ini:
  [/Script/PythonScriptPlugin.PythonScriptPluginSettings]
  bRemoteExecution=True

  python tools/unreal/pipeline/ue_remote.py --file tools/unreal/pipeline/build_arena.py --args "--arena duel_pit"
  python tools/unreal/pipeline/ue_remote.py --exec "unreal.SystemLibrary.execute_console_command(None, 'LiveCoding.Compile')"
  python tools/unreal/pipeline/ue_remote.py --exec "print(unreal.SystemLibrary.get_engine_version())"

Uses the engine's own client (Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py).
Exit code 0 = the command ran without a Python error; output is printed with a W4B_REMOTE prefix.
"""
import argparse
import os
import sys
import time

UE_ROOT = os.environ.get("W4B_UE_ROOT", r"C:\Program Files\Epic Games\UE_5.7")
sys.path.insert(0, os.path.join(UE_ROOT, "Engine", "Plugins", "Experimental", "PythonScriptPlugin", "Content", "Python"))
import remote_execution  # noqa: E402  (engine file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="Python file to run inside the editor")
    ap.add_argument("--args", default="", help="arguments for --file (sys.argv inside the editor)")
    ap.add_argument("--exec", help="Python statement(s) to run inside the editor ('unreal' is imported)")
    ap.add_argument("--wait", type=float, default=15.0, help="seconds to look for a running editor")
    ap.add_argument("--project", default=os.environ.get("W4B_UE_PROJECT_NAME", "W4BArena"),
                    help="only talk to the editor that has this project open (several editors may be running)")
    a = ap.parse_args()
    if not a.file and not a.exec:
        ap.error("pass --file or --exec")

    remote = remote_execution.RemoteExecution()
    remote.start()
    try:
        deadline = time.time() + a.wait
        nodes = []
        while not nodes and time.time() < deadline:
            nodes = [n for n in remote.remote_nodes if n.get("project_name") == a.project]
            time.sleep(0.25)
        if not nodes:
            seen = sorted({n.get("project_name", "?") for n in remote.remote_nodes})
            print("W4B_REMOTE no open editor for project %s (seen: %s). Is it open with bRemoteExecution=True?" % (a.project, seen))
            return 2
        node = nodes[0]
        remote.open_command_connection(node["node_id"])
        if a.file:
            command = (os.path.abspath(a.file).replace("\\", "/") + " " + a.args).strip()
            result = remote.run_command(command, unattended=True, exec_mode=remote_execution.MODE_EXEC_FILE)
        else:
            command = "import unreal\n" + a.exec
            result = remote.run_command(command, unattended=True, exec_mode=remote_execution.MODE_EXEC_FILE)
        for line in result.get("output", []):
            print("W4B_REMOTE " + str(line.get("output", "")).rstrip())
        ok = bool(result.get("success"))
        if not ok:
            print("W4B_REMOTE error: " + str(result.get("result")))
        return 0 if ok else 1
    finally:
        remote.stop()


if __name__ == "__main__":
    sys.exit(main())
