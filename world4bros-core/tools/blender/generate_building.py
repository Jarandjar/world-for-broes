import bpy
import bmesh
import random
from mathutils import Vector, Matrix

bl_info = {
    "name": "Procedural Low-Poly Building Generator",
    "author": "Antigravity",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > Building Gen",
    "description": "Generates basic low-poly greybox isometric buildings",
    "category": "Add Mesh",
}

def add_box(bm, center, size):
    """Utility to spawn a box into a bmesh at center (Vector) with size (Vector/tuple)."""
    sx, sy, sz = size
    mat = Matrix.Translation(center) @ Matrix.Diagonal((sx, sy, sz, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)

def add_cylinder(bm, center, radius, depth, segments=8):
    """Utility to spawn a low-poly cylinder (water tanks, pipes)."""
    mat = Matrix.Translation(center)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=segments,
        radius1=radius,
        radius2=radius,
        depth=depth,
        matrix=mat,
    )

def build_procedural_building(props):
    # Set random seed
    random.seed(props.seed)
    
    # Create new mesh and bmesh
    mesh = bpy.data.meshes.new("ProceduralBuilding_Mesh")
    bm = bmesh.new()

    total_floors = props.floors
    floor_h = props.floor_height
    w = props.width
    d = props.depth

    # Determine setback (step-in) floor
    has_setback = props.enable_setback and (total_floors >= 4)
    setback_floor = random.randint(3, max(3, total_floors - 2)) if has_setback else total_floors + 1
    inset = props.setback_inset if has_setback else 0.0

    current_z = 0.0

    # 1. GENERATE BASE & FLOORS
    for f in range(total_floors):
        # Apply setback dimensions if at or above setback floor
        cur_w = (w - inset * 2.0) if f >= setback_floor else w
        cur_d = (d - inset * 2.0) if f >= setback_floor else d

        floor_center_z = current_z + (floor_h / 2.0)

        # Floor mass (Main Wall Box)
        add_box(bm, Vector((0, 0, floor_center_z)), (cur_w, cur_d, floor_h))

        # Horizontal Ledge / Floor Trim (gives clean architectural silhouette)
        ledge_h = 0.18
        add_box(bm, Vector((0, 0, current_z + floor_h)), (cur_w + 0.15, cur_d + 0.15, ledge_h))

        # 2. WINDOW RECESSES (Pure Geometry indentations)
        # Windows along Front and Back (X axis)
        win_w = 1.0
        win_h = 1.4
        win_spacing = 2.0
        num_win_x = max(1, int(cur_w // win_spacing) - 1)
        x_start = -((num_win_x - 1) * win_spacing) / 2.0

        for i in range(num_win_x):
            wx = x_start + i * win_spacing
            wz = floor_center_z + 0.1

            # Front face frame
            add_box(bm, Vector((wx, cur_d / 2.0, wz)), (win_w, 0.12, win_h))
            # Back face frame
            add_box(bm, Vector((wx, -cur_d / 2.0, wz)), (win_w, 0.12, win_h))

        # Windows along Left and Right (Y axis)
        num_win_y = max(1, int(cur_d // win_spacing) - 1)
        y_start = -((num_win_y - 1) * win_spacing) / 2.0

        for j in range(num_win_y):
            wy = y_start + j * win_spacing
            wz = floor_center_z + 0.1

            # Right face frame
            add_box(bm, Vector((cur_w / 2.0, wy, wz)), (0.12, win_w, win_h))
            # Left face frame
            add_box(bm, Vector((-cur_w / 2.0, wy, wz)), (0.12, win_w, win_h))

        # Ground Floor Door Alcove
        if f == 0:
            door_w = 2.0
            door_h = 2.3
            # Front entrance canopy / threshold
            add_box(bm, Vector((0, cur_d / 2.0 + 0.3, door_h)), (door_w + 0.6, 0.8, 0.15))
            # Entrance frame
            add_box(bm, Vector((0, cur_d / 2.0, door_h / 2.0)), (door_w, 0.2, door_h))

        current_z += floor_h

    # 3. ROOF PARAPET (Raised border wall around the roof)
    top_w = (w - inset * 2.0) if total_floors >= setback_floor else w
    top_d = (d - inset * 2.0) if total_floors >= setback_floor else d
    parapet_h = 0.5
    parapet_t = 0.25

    # North / South parapet
    add_box(bm, Vector((0, (top_d - parapet_t) / 2.0, current_z + parapet_h / 2.0)), (top_w, parapet_t, parapet_h))
    add_box(bm, Vector((0, -(top_d - parapet_t) / 2.0, current_z + parapet_h / 2.0)), (top_w, parapet_t, parapet_h))
    # East / West parapet
    add_box(bm, Vector(((top_w - parapet_t) / 2.0, 0, current_z + parapet_h / 2.0)), (parapet_t, top_d, parapet_h))
    add_box(bm, Vector((-(top_w - parapet_t) / 2.0, 0, current_z + parapet_h / 2.0)), (parapet_t, top_d, parapet_h))

    # 4. ROOFTOP CLUTTER (Simple primitive shapes)
    if props.generate_roof_clutter:
        # A. Elevator Housing Penthouse
        ph_w, ph_d, ph_h = 2.8, 2.8, 2.2
        ph_x = random.choice([-1, 1]) * (top_w * 0.18)
        ph_y = random.choice([-1, 1]) * (top_d * 0.18)
        add_box(bm, Vector((ph_x, ph_y, current_z + ph_h / 2.0)), (ph_w, ph_d, ph_h))

        # Antenna on elevator roof
        add_cylinder(bm, Vector((ph_x, ph_y, current_z + ph_h + 1.2)), radius=0.06, depth=2.4, segments=6)

        # B. HVAC / AC Units (1 to 3 units)
        ac_count = random.randint(1, 3)
        for _ in range(ac_count):
            ac_x = random.uniform(-top_w * 0.35, top_w * 0.35)
            ac_y = random.uniform(-top_d * 0.35, top_d * 0.35)
            # Avoid placing inside the elevator room
            if abs(ac_x - ph_x) > 1.8 or abs(ac_y - ph_y) > 1.8:
                ac_sz = random.uniform(0.9, 1.4)
                add_box(bm, Vector((ac_x, ac_y, current_z + 0.45)), (ac_sz, ac_sz * 0.8, 0.9))

        # C. Water Tower (chance)
        if random.random() > 0.4:
            wt_x = -ph_x * 0.8
            wt_y = -ph_y * 0.8
            # Legs
            leg_h = 1.2
            add_box(bm, Vector((wt_x, wt_y, current_z + leg_h / 2.0)), (1.4, 1.4, leg_h))
            # Tank cylinder
            tank_h = 1.8
            tank_r = 1.1
            add_cylinder(bm, Vector((wt_x, wt_y, current_z + leg_h + tank_h / 2.0)), radius=tank_r, depth=tank_h, segments=10)

    # Finalize BMesh and create Object
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("ProceduralBuilding", mesh)
    bpy.context.collection.objects.link(obj)

    # Select and make active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    return obj

# --- BLENDER OPERATOR & UI PANEL ---
class MESH_OT_generate_building(bpy.types.Operator):
    bl_idname = "mesh.generate_procedural_building"
    bl_label = "Generate Building"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.building_gen_props
        build_procedural_building(props)
        self.report({'INFO'}, f"Generated Building (Seed: {props.seed})")
        return {'FINISHED'}

class BuildingGenProperties(bpy.types.PropertyGroup):
    seed: bpy.props.IntProperty(name="Seed", default=42, min=0)
    floors: bpy.props.IntProperty(name="Floors", default=6, min=2, max=30)
    floor_height: bpy.props.FloatProperty(name="Floor Height", default=3.0, min=1.5, max=5.0)
    width: bpy.props.FloatProperty(name="Width", default=10.0, min=4.0, max=30.0)
    depth: bpy.props.FloatProperty(name="Depth", default=10.0, min=4.0, max=30.0)
    enable_setback: bpy.props.BoolProperty(name="Enable Setback Step", default=True)
    setback_inset: bpy.props.FloatProperty(name="Setback Inset", default=1.5, min=0.5, max=4.0)
    generate_roof_clutter: bpy.props.BoolProperty(name="Roof Clutter", default=True)

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
        col.prop(props, "seed")
        col.prop(props, "floors")
        col.prop(props, "floor_height")
        col.separator()
        col.prop(props, "width")
        col.prop(props, "depth")
        col.separator()
        col.prop(props, "enable_setback")
        if props.enable_setback:
            col.prop(props, "setback_inset")
        col.separator()
        col.prop(props, "generate_roof_clutter")

        layout.separator()
        layout.operator("mesh.generate_procedural_building", text="Generate Building", icon='MESH_CUBE')

classes = (
    BuildingGenProperties,
    MESH_OT_generate_building,
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
