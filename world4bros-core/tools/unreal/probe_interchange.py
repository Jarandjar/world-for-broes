import unreal

mgr = unreal.get_editor_subsystem(unreal.InterchangeManager)
print("INTERCHANGE_MGR:", mgr)
if mgr:
    print("MGR_DIR:", dir(mgr))
