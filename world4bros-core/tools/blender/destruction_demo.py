"""
Destruction demo (HD plan / GENERATION_IDEAS): a coloured small-town house whose front wall is
built from bricks, a truck that drives through it, and a gable roof that collapses once its
support is gone. Rigid-body physics is baked to keyframes, exported as an animated GLB for the
orbit viewer (tools/viewer/), and three isometric frames are rendered with the style-sheet camera.

Run headless (from world4bros-core):
  "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --factory-startup --python-exit-code 1 \
      -P tools/blender/destruction_demo.py -- --out tools/viewer/models --frames docs/screens --wall fracture

--wall fracture (default): the front wall is cut into Voronoi chunks, the same method as Blender's
Cell Fracture add-on (implemented here so it runs headless with nothing to install); chunks are
smaller near the impact point. --wall bricks: the first version, a staggered brick wall.

Sizes follow STYLE_BIBLE.md §0.5 (1 unit = 1 m, storey 2.7 m, door 0.9 x 2.1 m); colours follow §0.3.
"""
import math
import os
import random
import sys

import bpy
import bmesh
from mathutils import Vector

# ---------------------------------------------------------------- settings
FRAMES = 140
FPS = 24
HOUSE_W, HOUSE_D, WALL_H, WALL_T = 8.0, 7.0, 2.7, 0.2
BRICK_W, BRICK_H = 0.5, 0.3
ROOF_PITCH = math.radians(35)

PALETTE = {  # STYLE_BIBLE.md §0.3
    "ground": "#5f6b45", "sidewalk": "#8b8a84", "siding": "#b3a98f", "brick": "#7a4a3a",
    "trim": "#c9c3b3", "roof": "#6a3f34", "floor": "#6b5440", "glass": "#5d7480",
    "truck": "#7a3b34", "truck_dark": "#2a2b2d", "door": "#4d5a6a",
}


def args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"out": "tools/viewer/models", "frames": "docs/screens", "wall": "fracture"}
    for i in range(0, len(argv) - 1, 2):
        out[argv[i].lstrip("-")] = argv[i + 1]
    return out


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def material(name: str, hex_colour: str, rough: float = 0.85) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    h = hex_colour.lstrip("#")
    rgb = [srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    return m


def select_only(obj) -> None:
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def box(name, size, loc, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    return o


def rigid(o, kind="ACTIVE", mass=1.0, sleep=False, shape="BOX", friction=0.7):
    select_only(o)
    bpy.ops.rigidbody.object_add()
    rb = o.rigid_body
    rb.type = kind
    rb.mass = mass
    rb.collision_shape = shape
    rb.friction = friction
    rb.restitution = 0.05
    rb.collision_margin = 0.01
    if sleep:
        rb.use_deactivation = True
        rb.use_start_deactivated = True
        rb.deactivate_linear_velocity = 0.2
        rb.deactivate_angular_velocity = 0.3
    return o


def wall_panels(x0, x1, z0, z1, holes):
    """Split the wall rectangle [x0,x1]×[z0,z1] minus rectangular holes (door, windows) into solid panels."""
    xs = sorted({x0, x1, *[h[0] for h in holes], *[h[1] for h in holes]})
    panels = []
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2
        cuts = sorted((h[2], h[3]) for h in holes if h[0] < mid < h[1])
        z = z0
        for h0, h1 in cuts:
            if h0 > z: panels.append((a, b, z, h0))
            z = max(z, h1)
        if z < z1: panels.append((a, b, z, z1))
    return panels


def voronoi_cells(rect, y, thick, points, shrink=0.006):
    """Cell fracture: one convex chunk per seed point, cut from the panel box by the bisecting planes
    to every other seed (normals in the wall plane, so chunks go through the whole wall thickness).
    Each chunk is shrunk a few mm so neighbours never start overlapping."""
    x0, x1, z0, z1 = rect
    cells = []
    for i, p in enumerate(points):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(x1 - x0, thick, z1 - z0), verts=bm.verts)
        bmesh.ops.translate(bm, vec=((x0 + x1) / 2, y, (z0 + z1) / 2), verts=bm.verts)
        for j, q in enumerate(points):
            if i == j:
                continue
            n = Vector((q[0] - p[0], 0.0, q[1] - p[1]))
            if n.length < 1e-6:
                continue
            n.normalize()
            mid = Vector(((p[0] + q[0]) / 2, y, (p[1] + q[1]) / 2))
            res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=mid, plane_no=n, clear_outer=True)
            cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
            if cut_edges:
                bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
        if len(bm.verts) < 4:
            bm.free()
            continue
        c = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
        for v in bm.verts:
            v.co = c + (v.co - c) * (1 - shrink / max(0.05, (v.co - c).length))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        cells.append(bm)
    return cells


def fractured_wall(mats, y, holes, impact, seed=7):
    """Front wall as Voronoi chunks: dense seeds near the impact point (small debris), sparse elsewhere."""
    rng = random.Random(seed)
    chunks = []
    for k, rect in enumerate(wall_panels(-HOUSE_W / 2, HOUSE_W / 2, 0.0, WALL_H, holes)):
        x0, x1, z0, z1 = rect
        area = (x1 - x0) * (z1 - z0)
        near = math.exp(-((((x0 + x1) / 2) - impact[0]) ** 2) / 6.0)   # 1 at the impact, falls off sideways
        n = max(1, int(area * (2.5 + 9 * near)))
        pts = []
        for _ in range(n):
            # Bias seeds toward the impact so the breach shatters finely and the corners stay in big slabs.
            px = rng.uniform(x0, x1)
            if rng.random() < near * 0.6:
                px = min(x1, max(x0, rng.gauss(impact[0], 0.9)))
            pts.append((px, rng.uniform(z0, z1)))
        for i, bm in enumerate(voronoi_cells(rect, y, WALL_T, pts)):
            me = bpy.data.meshes.new(f"Chunk_{k:02d}_{i:02d}")
            bm.to_mesh(me); bm.free()
            o = bpy.data.objects.new(me.name, me)
            bpy.context.collection.objects.link(o)
            o.data.materials.append(mats["brick"])
            select_only(o)
            bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
            vol = max(0.002, o.dimensions.x * o.dimensions.y * o.dimensions.z * 0.6)
            o = rigid(o, mass=vol * 1800, sleep=True, shape="CONVEX_HULL")
            o.rigid_body.use_start_deactivated = False
            chunks.append(o)
    return chunks


def glue(chunks, statics, threshold=60.0, reach=0.03, between_chunks=True):
    """Breakable 'Fixed' constraints between touching chunks (and from chunks to the ground and side
    walls), so a fractured wall stands until a hit is strong enough to snap the glue. Same idea as
    the 'Connect' step people use after Cell Fracture."""
    def box_of(o):
        ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
        return [min(v[i] for v in ws) for i in range(3)], [max(v[i] for v in ws) for i in range(3)]

    def touch(a, b):
        (a0, a1), (b0, b1) = a, b
        return all(a0[i] - reach <= b1[i] and b0[i] - reach <= a1[i] for i in range(3))

    bpy.context.view_layer.update()
    boxes = {o.name: box_of(o) for o in chunks + statics}
    pairs = [(a, b) for i, a in enumerate(chunks) for b in chunks[i + 1:] if touch(boxes[a.name], boxes[b.name])] if between_chunks else []
    pairs += [(a, s) for a in chunks for s in statics if touch(boxes[a.name], boxes[s.name])]
    for a, b in pairs:
        e = bpy.data.objects.new(f"Glue_{a.name}_{b.name}", None)
        e.location = (a.matrix_world.translation + b.matrix_world.translation) / 2
        bpy.context.collection.objects.link(e)
        select_only(e)
        bpy.ops.rigidbody.constraint_add()
        c = e.rigid_body_constraint
        c.type = "FIXED"
        c.object1, c.object2 = a, b
        c.use_breaking = True
        c.breaking_threshold = threshold
    return len(pairs)


def gable_roof(name, roof_mat, gable_mat, x0, x1, overhang=0.4):
    """One strip of a solid gable roof between x0 and x1: a triangular prism from the wall top to the
    ridge, with eave overhang. Slopes use roof_mat, the triangular ends use gable_mat."""
    half_d = HOUSE_D / 2 + overhang
    base = WALL_H + 0.005
    ridge = base + half_d * math.tan(ROOF_PITCH)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    left = [bm.verts.new((x0, -half_d, base)), bm.verts.new((x0, half_d, base)), bm.verts.new((x0, 0, ridge))]
    right = [bm.verts.new((x1, -half_d, base)), bm.verts.new((x1, half_d, base)), bm.verts.new((x1, 0, ridge))]
    gable_l = bm.faces.new((left[0], left[2], left[1]))
    gable_r = bm.faces.new((right[0], right[1], right[2]))
    bm.faces.new((left[0], left[1], right[1], right[0]))           # underside
    bm.faces.new((left[0], right[0], right[2], left[2]))           # front slope
    bm.faces.new((left[1], left[2], right[2], right[1]))           # back slope
    gable_l.material_index = gable_r.material_index = 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(roof_mat)
    o.data.materials.append(gable_mat)
    select_only(o)
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
    return o


def build(mats, wall_mode="fracture"):
    # Ground and sidewalk
    # Thick ground so fast, heavy pieces (the roof) cannot tunnel through it.
    rigid(box("Ground", (40, 40, 2.0), (0, 0, -1.0), mats["ground"]), "PASSIVE", shape="BOX")
    box("Sidewalk", (16, 2.0, 0.05), (0, -HOUSE_D / 2 - 3.0, 0.025), mats["sidewalk"])
    box("Floor", (HOUSE_W, HOUSE_D, 0.08), (0, 0, 0.04), mats["floor"])

    # Back wall and side walls (static). Side walls stop 4 cm short of the top, so the roof bears on the
    # front and back walls only — knock out the front wall and the roof has nothing to stand on at the front.
    rigid(box("WallBack", (HOUSE_W, WALL_T, WALL_H), (0, HOUSE_D / 2, WALL_H / 2), mats["siding"]), "PASSIVE")
    side_h = WALL_H - 0.04
    for sx in (-1, 1):
        rigid(box(f"WallSide{'L' if sx < 0 else 'R'}", (WALL_T, HOUSE_D - WALL_T, side_h), (sx * HOUSE_W / 2, 0, side_h / 2), mats["siding"]), "PASSIVE")

    # Front wall with a door and two windows left open: Voronoi chunks (Cell Fracture style) or bricks
    y = -HOUSE_D / 2
    door = (-0.45, 0.45, 0.0, 2.1)
    windows = [(-3.0, -1.8, 0.9, 2.1), (1.8, 3.0, 0.9, 2.1)]
    bricks = []
    rows = int(round(WALL_H / BRICK_H)) if wall_mode == "bricks" else 0
    if wall_mode != "bricks":
        bricks = fractured_wall(mats, y, [door] + windows, impact=(0.6, 1.0))
        # Tuned by trial (see GENERATION_IDEAS.md §13): strong ties to the ground and side walls keep the wall
        # standing until the hit; weaker chunk-to-chunk ties let the band over the breach tear and drop, so the
        # roof above it loses support. (Strong chunk ties = the band hangs like a beam and holds the roof;
        # weak side ties = the wall slumps under the roof before the truck arrives.)
        n = glue(bricks, [bpy.data.objects["Ground"], bpy.data.objects["WallSideL"], bpy.data.objects["WallSideR"]],
                 threshold=60.0, between_chunks=False)
        n += glue(bricks, [], threshold=30.0)
        print("GLUE", n, "constraints")
    for r in range(rows):
        z0 = r * BRICK_H
        offset = BRICK_W / 2 if r % 2 else 0.0
        x = -HOUSE_W / 2 - offset
        while x < HOUSE_W / 2 - 1e-6:
            x0, x1 = max(x, -HOUSE_W / 2), min(x + BRICK_W, HOUSE_W / 2)
            cx, cz = (x0 + x1) / 2, z0 + BRICK_H / 2
            holes = [door] + windows
            if not any(h[0] < cx < h[1] and h[2] < cz < h[3] for h in holes) and x1 - x0 > 0.05:
                # Bricks touch vertically (no gap to drop through) and start awake, so bricks left hanging
                # over the hole fall; they sleep once still to keep the rest of the wall stable.
                b = box(f"Brick_{r:02d}_{len(bricks):03d}", (x1 - x0 - 0.01, WALL_T, BRICK_H - 0.001), (cx, y, cz), mats["brick"])
                b = rigid(b, mass=4.0, sleep=True)
                b.rigid_body.use_start_deactivated = False
                bricks.append(b)
            x += BRICK_W
    # Glass panes (break out with the bricks) and the door leaf
    panes = [rigid(box(f"Glass_{i}", (w[1] - w[0], 0.04, w[3] - w[2]), ((w[0] + w[1]) / 2, y, (w[2] + w[3]) / 2), mats["glass"]), mass=3.0, sleep=True)
             for i, w in enumerate(windows)]
    door_leaf = rigid(box("Door", (0.88, 0.06, 2.08), (0, y - 0.02, 1.05), mats["door"]), mass=15.0, sleep=True)

    # Gable roof in 1 m strips across the house (both slopes + gable ends per strip, like roof trusses),
    # each resting on the front and back walls. Strips over a hole in the front wall lose their front
    # support and drop into the house; the others stay up. (Two loose slabs slide off a 35° pitch on their
    # own, and one rigid roof just bridges the hole, so strips are what make a believable partial collapse.)
    strips, half_w, gap = 9, HOUSE_W / 2 + 0.3, 0.004
    edges = [-half_w + i * (2 * half_w) / strips for i in range(strips + 1)]
    roof = [rigid(gable_roof(f"Roof_{i}", mats["roof"], mats["siding"], edges[i] + gap, edges[i + 1] - gap),
                  mass=70.0, shape="CONVEX_HULL", friction=0.9)
            for i in range(strips)]

    # Truck that drives through the front wall (kinematic = animated, pushes everything)
    # Kept under the wall top (2.7 m): a kinematic body wedged against the lintel and roof launches them.
    truck = box("Truck", (2.6, 5.0, 1.5), (0.6, -HOUSE_D / 2 - 9.0, 0.9), mats["truck"])
    cab = box("TruckCab", (2.4, 1.6, 0.7), (0.6, -HOUSE_D / 2 - 7.5, 2.0), mats["truck_dark"])
    cab.parent = truck
    cab.matrix_parent_inverse = truck.matrix_world.inverted()
    rigid(truck, "ACTIVE", mass=3000.0)
    truck.rigid_body.kinematic = True
    scene = bpy.context.scene
    for frame, ty in ((1, -HOUSE_D / 2 - 9.0), (20, -HOUSE_D / 2 - 6.0), (52, -HOUSE_D / 2 + 1.0), (76, -HOUSE_D / 2 + 2.2)):
        scene.frame_set(frame)
        truck.location.y = ty
        truck.keyframe_insert("location", frame=frame)
    return bricks + panes + [door_leaf] + roof, truck


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.render.fps = FPS
    scene.unit_settings.system = "METRIC"
    bpy.ops.rigidbody.world_add()
    world = scene.rigidbody_world
    world.point_cache.frame_start, world.point_cache.frame_end = 1, FRAMES
    world.substeps_per_frame = 20
    world.solver_iterations = 20
    return scene


def setup_camera_and_light(scene):
    # STYLE_BIBLE.md §0.8: orthographic, 30° elevation, 45° turn; sun from the north-west (screen upper-left)
    cam_data = bpy.data.cameras.new("IsoCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 22
    cam = bpy.data.objects.new("IsoCam", cam_data)
    scene.collection.objects.link(cam)
    target = Vector((0, -1.5, 1.2))
    d = 40.0
    cam.location = target + Vector((d * math.cos(math.radians(30)) / math.sqrt(2), -d * math.cos(math.radians(30)) / math.sqrt(2), d * math.sin(math.radians(30))))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.2
    sun_data.angle = math.radians(8)
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = Vector((1.0, 0.0, -1.0)).to_track_quat("-Z", "Y").to_euler()

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.32, 0.34, 0.36, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world

    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 1280, 960
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"


def main():
    a = args()
    os.makedirs(a["out"], exist_ok=True)
    os.makedirs(a["frames"], exist_ok=True)
    scene = setup_scene()
    mats = {k: material(k, v, 0.4 if k == "glass" else 0.85) for k, v in PALETTE.items()}
    dynamic, truck = build(mats, a["wall"])
    setup_camera_and_light(scene)

    # Bake the physics into ordinary keyframes so the GLB carries the animation.
    # (bpy.ops.rigidbody.bake_to_keyframes needs a UI window, so record transforms frame by frame instead.)
    track = {o.name: [] for o in dynamic}
    for f in range(1, FRAMES + 1):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for o in dynamic:
            track[o.name].append(o.evaluated_get(dg).matrix_world.copy())
    scene.rigidbody_world.enabled = False
    for o in dynamic:
        o.rotation_mode = "QUATERNION"
        for f, m in enumerate(track[o.name], start=1):
            loc, rot, _ = m.decompose()
            o.location, o.rotation_quaternion = loc, rot
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("rotation_quaternion", frame=f)
    scene.frame_set(1)

    glb = os.path.join(a["out"], "house_destruction.glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", export_animations=True,
                              export_animation_mode="SCENE", export_frame_range=True, export_force_sampling=True,
                              export_cameras=False, export_lights=False)
    print("WROTE", glb)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a["out"], "house_destruction.blend")))

    for frame, label in ((1, "before"), (52, "impact"), (FRAMES, "after")):
        scene.frame_set(frame)
        scene.render.filepath = os.path.abspath(os.path.join(a["frames"], f"destruction-{label}.png"))
        bpy.ops.render.render(write_still=True)
        print("WROTE", scene.render.filepath)


if __name__ == "__main__":
    main()
