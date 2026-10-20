import bpy
import bmesh
import random
import json
from mathutils import Vector, Matrix

bl_info = {
	"name": "World4Bros Modular Snap Building Tool",
	"author": "Antigravity + Copilot",
	"version": (1, 0, 0),
	"blender": (3, 0, 0),
	"location": "View3D > Sidebar (N) > Building Gen",
	"description": "Generate modular snap-piece buildings and export building.json",
	"category": "Add Mesh",
}


def clamp(v, lo, hi):
	return max(lo, min(hi, v))


def add_box(bm, center, size):
	sx, sy, sz = size
	mat = Matrix.Translation(center) @ Matrix.Diagonal((sx, sy, sz, 1.0))
	bmesh.ops.create_cube(bm, size=1.0, matrix=mat)


def _door_segment(side, x, y, w, h, door_at, door_w):
	half = door_w / 2.0
	if side == 'N':
		return {"kind": "door", "x0": x + door_at - half, "y0": y, "x1": x + door_at + half, "y1": y, "exterior": True}
	if side == 'S':
		return {"kind": "door", "x0": x + door_at - half, "y0": y + h, "x1": x + door_at + half, "y1": y + h, "exterior": True}
	if side == 'W':
		return {"kind": "door", "x0": x, "y0": y + door_at - half, "x1": x, "y1": y + door_at + half, "exterior": True}
	return {"kind": "door", "x0": x + w, "y0": y + door_at - half, "x1": x + w, "y1": y + door_at + half, "exterior": True}


def build_pieces(props):
	x = 0.0
	y = 0.0
	w = props.width
	h = props.depth
	side = props.door_side
	door_at = (w / 2.0) if side in ('N', 'S') else (h / 2.0)
	door_w = props.door_width

	pieces = []

	def add_wall(x0, y0, x1, y1, exterior=False):
		pieces.append({"kind": "wall", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "exterior": exterior})

	def side_with_door(s):
		half = door_w / 2.0
		if s in ('N', 'S'):
			yy = y if s == 'N' else y + h
			if side != s:
				add_wall(x, yy, x + w, yy, True)
			else:
				add_wall(x, yy, x + door_at - half, yy, True)
				add_wall(x + door_at + half, yy, x + w, yy, True)
		else:
			xx = x if s == 'W' else x + w
			if side != s:
				add_wall(xx, y, xx, y + h, True)
			else:
				add_wall(xx, y, xx, y + door_at - half, True)
				add_wall(xx, y + door_at + half, xx, y + h, True)

	side_with_door('N')
	side_with_door('S')
	side_with_door('W')
	side_with_door('E')
	pieces.append(_door_segment(side, x, y, w, h, door_at, door_w))

	# Interior layouts
	if props.layout in ('split', 'apartment', 'backroom'):
		ratio = 0.5 if props.layout in ('split', 'apartment') else 0.62
		ratio = clamp(ratio, 0.35, 0.72)
		half = door_w / 2.0

		if side in ('N', 'S'):
			iy = y + h * (ratio if side == 'N' else (1.0 - ratio))
			g0 = clamp(x + door_at - half, x + 1.0, x + w - 1.0)
			g1 = clamp(x + door_at + half, x + 1.0, x + w - 1.0)
			add_wall(x + props.wall_thickness, iy, g0, iy, False)
			add_wall(g1, iy, x + w - props.wall_thickness, iy, False)
			pieces.append({"kind": "door", "x0": g0, "y0": iy, "x1": g1, "y1": iy, "exterior": False})
		else:
			ix = x + w * (ratio if side == 'W' else (1.0 - ratio))
			g0 = clamp(y + door_at - half, y + 1.0, y + h - 1.0)
			g1 = clamp(y + door_at + half, y + 1.0, y + h - 1.0)
			add_wall(ix, y + props.wall_thickness, ix, g0, False)
			add_wall(ix, g1, ix, y + h - props.wall_thickness, False)
			pieces.append({"kind": "door", "x0": ix, "y0": g0, "x1": ix, "y1": g1, "exterior": False})

	# Floor pieces (one per level)
	for f in range(props.floors):
		pieces.append({
			"kind": "floor",
			"x0": x,
			"y0": y,
			"x1": x + w,
			"y1": y + h,
			"level": f
		})

	return {
		"schema": "w4b.building.v1",
		"seed": int(props.seed),
		"grid_m": float(props.tile_size),
		"layout": props.layout,
		"size": {"width": float(w), "depth": float(h), "floors": int(props.floors), "floorHeight": float(props.floor_height)},
		"door": {"side": side, "doorAt": float(door_at), "doorWidth": float(door_w)},
		"pieces": pieces,
	}


def _wall_box_from_piece(p, wall_t, floor_h, floors):
	x0 = float(p["x0"])
	y0 = float(p["y0"])
	x1 = float(p["x1"])
	y1 = float(p["y1"])

	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	full_h = floor_h * floors
	cz = full_h / 2.0

	if dx >= dy:
		cx = (x0 + x1) / 2.0
		cy = y0
		sx = max(0.05, dx)
		sy = wall_t
	else:
		cx = x0
		cy = (y0 + y1) / 2.0
		sx = wall_t
		sy = max(0.05, dy)
	return Vector((cx, cy, cz)), (sx, sy, full_h)


def build_mesh_from_blueprint(bp, props):
	mesh = bpy.data.meshes.new("ModularBuilding_Mesh")
	bm = bmesh.new()

	# Floors
	for f in range(props.floors):
		z = f * props.floor_height + props.floor_height / 2.0
		add_box(
			bm,
			Vector((props.width / 2.0, props.depth / 2.0, z)),
			(props.width, props.depth, props.floor_height * 0.98),
		)

	# Door markers and walls
	for p in bp["pieces"]:
		kind = p.get("kind")
		if kind == "wall":
			c, s = _wall_box_from_piece(p, props.wall_thickness, props.floor_height, props.floors)
			add_box(bm, c, s)
		elif kind == "door":
			# Build visual door frame marker only (opening remains passable)
			x0, y0, x1, y1 = p["x0"], p["y0"], p["x1"], p["y1"]
			cx = (x0 + x1) / 2.0
			cy = (y0 + y1) / 2.0
			dz = props.floor_height * 0.5
			if abs(x1 - x0) >= abs(y1 - y0):
				add_box(bm, Vector((cx, cy, dz)), (max(0.2, abs(x1 - x0)), props.wall_thickness * 0.3, props.floor_height * 0.95))
			else:
				add_box(bm, Vector((cx, cy, dz)), (props.wall_thickness * 0.3, max(0.2, abs(y1 - y0)), props.floor_height * 0.95))

	bm.to_mesh(mesh)
	bm.free()

	obj = bpy.data.objects.new("ModularBuilding", mesh)
	bpy.context.collection.objects.link(obj)
	bpy.ops.object.select_all(action='DESELECT')
	obj.select_set(True)
	bpy.context.view_layer.objects.active = obj

	obj["building_blueprint_json"] = json.dumps(bp)
	obj["building_schema"] = bp.get("schema", "w4b.building.v1")
	return obj


class BuildingGenProperties(bpy.types.PropertyGroup):
	seed: bpy.props.IntProperty(name="Seed", default=4242, min=0)
	tile_size: bpy.props.FloatProperty(name="Tile Size (m)", default=1.0, min=0.5, max=2.0)
	floors: bpy.props.IntProperty(name="Floors", default=2, min=1, max=12)
	floor_height: bpy.props.FloatProperty(name="Floor Height", default=2.8, min=2.2, max=4.2)
	width: bpy.props.FloatProperty(name="Width (m)", default=10.0, min=4.0, max=40.0)
	depth: bpy.props.FloatProperty(name="Depth (m)", default=8.0, min=4.0, max=40.0)
	wall_thickness: bpy.props.FloatProperty(name="Wall Thickness", default=0.35, min=0.1, max=1.0)
	door_width: bpy.props.FloatProperty(name="Door Width", default=1.8, min=0.8, max=3.0)
	door_side: bpy.props.EnumProperty(
		name="Door Side",
		items=[('N', 'North', ''), ('E', 'East', ''), ('S', 'South', ''), ('W', 'West', '')],
		default='S'
	)
	layout: bpy.props.EnumProperty(
		name="Layout",
		items=[
			('none', 'None', ''),
			('split', 'Split', ''),
			('backroom', 'Backroom', ''),
			('apartment', 'Apartment', ''),
		],
		default='split'
	)


class MESH_OT_generate_modular_building(bpy.types.Operator):
	bl_idname = "mesh.generate_modular_building"
	bl_label = "Generate Modular Building"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		props = context.scene.building_gen_props
		random.seed(props.seed)
		blueprint = build_pieces(props)
		build_mesh_from_blueprint(blueprint, props)
		self.report({'INFO'}, f"Generated modular building with {len(blueprint['pieces'])} pieces")
		return {'FINISHED'}


class MESH_OT_randomize_seed(bpy.types.Operator):
	bl_idname = "mesh.randomize_building_seed"
	bl_label = "Randomize Seed"

	def execute(self, context):
		props = context.scene.building_gen_props
		props.seed = random.randint(1, 999999)
		return {'FINISHED'}


class MESH_OT_export_building_blueprint(bpy.types.Operator):
	bl_idname = "mesh.export_building_blueprint"
	bl_label = "Export Building JSON"
	filename_ext = ".json"

	filepath: bpy.props.StringProperty(subtype='FILE_PATH')

	def invoke(self, context, event):
		self.filepath = bpy.path.abspath("//building.json")
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		obj = context.active_object
		if not obj:
			self.report({'ERROR'}, "No active object")
			return {'CANCELLED'}

		raw = obj.get("building_blueprint_json")
		if not raw:
			self.report({'ERROR'}, "Active object has no modular blueprint data")
			return {'CANCELLED'}

		data = json.loads(raw)
		with open(self.filepath, 'w', encoding='utf-8') as f:
			json.dump(data, f, indent=2)

		self.report({'INFO'}, f"Exported blueprint: {self.filepath}")
		return {'FINISHED'}


class VIEW3D_PT_building_gen_panel(bpy.types.Panel):
	bl_label = "Building Generator"
	bl_idname = "VIEW3D_PT_building_gen_panel"
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_category = 'Building Gen'

	def draw(self, context):
		layout = self.layout
		props = context.scene.building_gen_props

		col = layout.column(align=True)
		row = col.row(align=True)
		row.prop(props, "seed")
		row.operator("mesh.randomize_building_seed", text="🎲")

		col.prop(props, "tile_size")
		col.prop(props, "floors")
		col.prop(props, "floor_height")
		col.separator()
		col.prop(props, "width")
		col.prop(props, "depth")
		col.prop(props, "wall_thickness")
		col.prop(props, "door_width")
		col.prop(props, "door_side")
		col.prop(props, "layout")

		layout.separator()
		layout.operator("mesh.generate_modular_building", icon='MESH_GRID')
		layout.operator("mesh.export_building_blueprint", icon='EXPORT')


classes = (
	BuildingGenProperties,
	MESH_OT_generate_modular_building,
	MESH_OT_randomize_seed,
	MESH_OT_export_building_blueprint,
	VIEW3D_PT_building_gen_panel,
)


def register():
	for cls in classes:
		bpy.utils.register_class(cls)
	bpy.types.Scene.building_gen_props = bpy.props.PointerProperty(type=BuildingGenProperties)


def unregister():
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)
	del bpy.types.Scene.building_gen_props


if __name__ == "__main__":
	register()
