import bpy
import bmesh
import random
import math
import sys
import os
import json
from types import SimpleNamespace
from mathutils import Vector, Matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ==========================================
# PRIMITIVE GEOMETRY UTILITIES
# ==========================================
def add_box(bm, center, size, rot_z=0.0):
    sx, sy, sz = size
    mat = Matrix.Translation(center) @ Matrix.Rotation(rot_z, 4, 'Z') @ Matrix.Diagonal((sx, sy, sz, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)

def add_cylinder(bm, center, radius, depth, segments=8, rot_x=0.0, rot_y=0.0, rot_z=0.0):
    mat = (Matrix.Translation(center) @
           Matrix.Rotation(rot_z, 4, 'Z') @
           Matrix.Rotation(rot_y, 4, 'Y') @
           Matrix.Rotation(rot_x, 4, 'X'))
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

def add_wedge(bm, center, size, angle_deg=30.0):
    """Awning wedge shape."""
    sx, sy, sz = size
    mat = Matrix.Translation(center) @ Matrix.Rotation(math.radians(angle_deg), 4, 'X') @ Matrix.Diagonal((sx, sy, sz, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)

# ==========================================
# VEHICLE BUILDERS
# ==========================================
def build_car(bm, pos, rot_z=0.0):
    """Low-poly sedan car."""
    x, y, z = pos
    body_mat = Matrix.Translation(Vector((x, y, z + 0.5))) @ Matrix.Rotation(rot_z, 4, 'Z')
    
    # Lower chassis
    add_box(bm, Vector((x, y, z + 0.45)), (1.9, 4.2, 0.5), rot_z)
    # Cabin / Greenhouse
    add_box(bm, Vector((x, y - 0.2, z + 0.95)), (1.65, 2.3, 0.55), rot_z)
    
    # 4 Wheels
    wheel_radius = 0.32
    wheel_thick = 0.22
    for wx in [-0.95, 0.95]:
        for wy in [-1.3, 1.3]:
            w_pos = body_mat @ Vector((wx, wy, wheel_radius - 0.5))
            add_cylinder(bm, w_pos, radius=wheel_radius, depth=wheel_thick, segments=8, rot_y=math.radians(90))

def build_bus(bm, pos, rot_z=0.0):
    """Low-poly transit city bus (like the green bus in the reference image)."""
    x, y, z = pos
    body_mat = Matrix.Translation(Vector((x, y, z + 1.4))) @ Matrix.Rotation(rot_z, 4, 'Z')
    
    # Main Bus Body
    add_box(bm, Vector((x, y, z + 1.5)), (2.5, 9.4, 2.3), rot_z)
    # Roof AC / vent strips
    add_box(bm, Vector((x, y - 0.5, z + 2.75)), (1.4, 4.0, 0.25), rot_z)
    
    # Window band recess (running along both sides)
    add_box(bm, Vector((x + 1.26, y, z + 1.7)), (0.05, 8.4, 0.8), rot_z)
    add_box(bm, Vector((x - 1.26, y, z + 1.7)), (0.05, 8.4, 0.8), rot_z)
    # Windshield front/rear
    add_box(bm, Vector((x, y + 4.71, z + 1.6)), (2.2, 0.05, 1.0), rot_z)
    add_box(bm, Vector((x, y - 4.71, z + 1.6)), (2.2, 0.05, 1.0), rot_z)

    # 6 Wheels
    wheel_radius = 0.45
    wheel_thick = 0.3
    for wx in [-1.22, 1.22]:
        for wy in [-3.0, 0.0, 3.0]:
            w_pos = body_mat @ Vector((wx, wy, wheel_radius - 1.4))
            add_cylinder(bm, w_pos, radius=wheel_radius, depth=wheel_thick, segments=8, rot_y=math.radians(90))

# ==========================================
# STREET FURNITURE & MARKINGS
# ==========================================
def build_traffic_light(bm, pos, rot_z=0.0):
    """Modern curved intersection traffic light mast."""
    x, y, z = pos
    pole_h = 6.2
    # Vertical mast
    add_cylinder(bm, Vector((x, y, z + pole_h / 2.0)), radius=0.14, depth=pole_h, segments=8)
    # Horizontal arm extending over road
    arm_len = 5.5
    add_box(bm, Vector((x + arm_len / 2.0, y, z + pole_h)), (arm_len, 0.16, 0.16), rot_z)
    # 2 Signal heads (1 pole-mounted, 1 mast-mounted)
    add_box(bm, Vector((x + 0.4, y, z + 3.2)), (0.35, 0.3, 0.9), rot_z)
    add_box(bm, Vector((x + arm_len - 0.8, y, z + pole_h - 0.5)), (0.35, 0.3, 0.9), rot_z)

def build_fire_escape(bm, wall_x, wall_y, start_z, floors, floor_h, width=2.4, depth=1.0):
    """Exterior zigzagging steel fire escape stairs (classic NYC/European style)."""
    cur_z = start_z
    for f in range(1, floors):
        plat_z = cur_z + (f * floor_h)
        # Platform grate
        add_box(bm, Vector((wall_x, wall_y + depth / 2.0, plat_z)), (width, depth, 0.08))

        # Railings (3 sides)
        rail_h = 0.9
        rail_t = 0.04
        # Front rail
        add_box(bm, Vector((wall_x, wall_y + depth, plat_z + rail_h / 2.0)), (width, rail_t, rail_h))
        # Left & Right rail
        add_box(bm, Vector((wall_x - width / 2.0, wall_y + depth / 2.0, plat_z + rail_h / 2.0)), (rail_t, depth, rail_h))
        add_box(bm, Vector((wall_x + width / 2.0, wall_y + depth / 2.0, plat_z + rail_h / 2.0)), (rail_t, depth, rail_h))

        # Diagonal connecting ladder
        if f > 1:
            lad_x = wall_x + (0.4 if f % 2 == 0 else -0.4)
            lad_z = plat_z - (floor_h / 2.0)
            lad_len = math.sqrt(floor_h**2 + (depth * 0.8)**2)
            angle = math.atan2(floor_h, depth * 0.8)
            add_box(bm, Vector((lad_x, wall_y + depth * 0.5, lad_z)), (0.35, 0.08, lad_len), rot_z=0.0)

def build_test_dummy_character(bm, pos, height=1.8):
    """Simple low-poly humanoid test dummy for doorway/interior scale checks."""
    x, y, z = pos
    leg_h = height * 0.46
    torso_h = height * 0.34
    neck_h = height * 0.06
    head_h = height * 0.22

    # Legs
    add_box(bm, Vector((x - 0.11, y, z + leg_h / 2.0)), (0.18, 0.22, leg_h))
    add_box(bm, Vector((x + 0.11, y, z + leg_h / 2.0)), (0.18, 0.22, leg_h))
    # Torso
    add_box(bm, Vector((x, y, z + leg_h + torso_h / 2.0)), (0.5, 0.28, torso_h))
    # Arms
    add_box(bm, Vector((x - 0.34, y, z + leg_h + torso_h * 0.62)), (0.18, 0.18, torso_h * 0.85))
    add_box(bm, Vector((x + 0.34, y, z + leg_h + torso_h * 0.62)), (0.18, 0.18, torso_h * 0.85))
    # Neck + Head
    add_box(bm, Vector((x, y, z + leg_h + torso_h + neck_h / 2.0)), (0.16, 0.16, neck_h))
    add_box(bm, Vector((x, y, z + leg_h + torso_h + neck_h + head_h / 2.0)), (0.36, 0.34, head_h))
    # Feet base
    add_box(bm, Vector((x, y, z + 0.05)), (0.34, 0.34, 0.1))

# ==========================================
# ARCHITECTURAL ARCHETYPES
# ==========================================

def build_apartment_building(bm, base_pos, w, d, floors, floor_h, has_fire_escape=True):
    """Brick / Tenement apartment with window grids, fire escapes & rooftop water tank."""
    bx, by, bz = base_pos
    total_h = floors * floor_h

    # Main mass
    add_box(bm, Vector((bx, by, bz + total_h / 2.0)), (w, d, total_h))

    # Horizontal architectural floor ledges
    for f in range(1, floors + 1):
        add_box(bm, Vector((bx, by, bz + f * floor_h)), (w + 0.2, d + 0.2, 0.18))

    # Detailed Window Matrix
    win_w, win_h, win_space = 1.1, 1.4, 2.1
    nx = max(1, int(w // win_space) - 1)
    ny = max(1, int(d // win_space) - 1)
    xs = -((nx - 1) * win_space) / 2.0
    ys = -((ny - 1) * win_space) / 2.0

    for f in range(floors):
        fz = bz + (f + 0.5) * floor_h
        for i in range(nx):
            wx = bx + xs + i * win_space
            # Front & Back window trims
            add_box(bm, Vector((wx, by + d / 2.0 + 0.06, fz)), (win_w, 0.12, win_h))
            add_box(bm, Vector((wx, by - d / 2.0 - 0.06, fz)), (win_w, 0.12, win_h))
        for j in range(ny):
            wy = by + ys + j * win_space
            # Left & Right window trims
            add_box(bm, Vector((bx + w / 2.0 + 0.06, wy, fz)), (0.12, win_w, win_h))
            add_box(bm, Vector((bx - w / 2.0 - 0.06, wy, fz)), (0.12, win_w, win_h))

    # Fire Escape on the front facade
    if has_fire_escape and floors >= 3:
        build_fire_escape(bm, bx, by + d / 2.0, bz, floors, floor_h, width=2.6, depth=1.1)

    # Ground floor doorway with steps
    door_w, door_h = 1.8, 2.4
    add_box(bm, Vector((bx, by + d / 2.0 + 0.4, bz + 0.15)), (door_w + 0.8, 0.8, 0.3)) # steps
    add_box(bm, Vector((bx, by + d / 2.0 + 0.2, bz + door_h / 2.0)), (door_w, 0.3, door_h)) # door frame
    add_box(bm, Vector((bx, by + d / 2.0 + 0.5, bz + door_h + 0.1)), (door_w + 0.4, 0.8, 0.15)) # canopy

    # Parapet
    add_box(bm, Vector((bx, by, bz + total_h + 0.3)), (w + 0.1, d + 0.1, 0.6))
    add_box(bm, Vector((bx, by, bz + total_h + 0.35)), (w - 0.6, d - 0.6, 0.7))

    # Rooftop Water Tank
    tank_x = bx + w * 0.25
    tank_y = by + d * 0.25
    leg_h = 1.5
    for lx in [-0.8, 0.8]:
        for ly in [-0.8, 0.8]:
            add_cylinder(bm, Vector((tank_x + lx, tank_y + ly, bz + total_h + leg_h / 2.0)), radius=0.07, depth=leg_h, segments=6)
    add_cylinder(bm, Vector((tank_x, tank_y, bz + total_h + leg_h + 1.2)), radius=1.3, depth=2.0, segments=12)

def build_corner_bodega(bm, base_pos, w, d):
    """Corner Bodega / Storefront with striped fabric awning and rooftop billboard."""
    bx, by, bz = base_pos
    floors = random.choice([2, 3])
    floor_h = 3.2
    total_h = floors * floor_h

    # Building Mass
    add_box(bm, Vector((bx, by, bz + total_h / 2.0)), (w, d, total_h))

    # Large Ground-floor Display Windows
    add_box(bm, Vector((bx, by + d / 2.0 + 0.05, bz + 1.5)), (w - 2.0, 0.15, 2.2))
    add_box(bm, Vector((bx + w / 2.0 + 0.05, by, bz + 1.5)), (0.15, d - 2.0, 2.2))

    # Overhanging Awnings (Front and Corner)
    awning_w = w - 1.0
    awning_d = 1.8
    add_box(bm, Vector((bx, by + d / 2.0 + awning_d / 2.0, bz + 2.7)), (awning_w, awning_d, 0.15), rot_z=0.0)
    # Awning support struts
    for ax in [-awning_w / 2.2, awning_w / 2.2]:
        add_cylinder(bm, Vector((bx + ax, by + d / 2.0 + awning_d * 0.8, bz + 1.35)), radius=0.05, depth=2.7, segments=6)

    # Second floor windows
    for f in range(1, floors):
        fz = bz + (f + 0.5) * floor_h
        for wx in [-w / 3.2, 0, w / 3.2]:
            add_box(bm, Vector((bx + wx, by + d / 2.0 + 0.05, fz)), (1.2, 0.12, 1.5))

    # Rooftop Billboard Frame
    board_w = w * 0.7
    board_h = 2.4
    add_box(bm, Vector((bx, by, bz + total_h + 0.8 + board_h / 2.0)), (board_w, 0.25, board_h))
    # Support poles
    for px in [-board_w / 2.5, board_w / 2.5]:
        add_cylinder(bm, Vector((bx + px, by, bz + total_h + 1.0)), radius=0.08, depth=2.0, segments=6)

def build_pagoda_temple(bm, base_pos, w, d):
    """Traditional Asian temple / pagoda architecture (like in bottom-right of reference image)."""
    bx, by, bz = base_pos
    tiers = 3
    tier_h = 2.6
    cur_z = bz
    cur_w = w
    cur_d = d

    # Raised stone plinth / base
    add_box(bm, Vector((bx, by, cur_z + 0.4)), (cur_w + 1.5, cur_d + 1.5, 0.8))
    cur_z += 0.8

    for t in range(tiers):
        # Wall core
        wall_w = cur_w * 0.75
        wall_d = cur_d * 0.75
        add_box(bm, Vector((bx, by, cur_z + tier_h / 2.0)), (wall_w, wall_d, tier_h))

        # Surrounding balcony / columns
        col_r = 0.12
        for cx in [-wall_w / 1.8, wall_w / 1.8]:
            for cy in [-wall_d / 1.8, wall_d / 1.8]:
                add_cylinder(bm, Vector((bx + cx, by + cy, cur_z + tier_h / 2.0)), radius=col_r, depth=tier_h, segments=6)

        # Balcony railing
        rail_h = 0.7
        add_box(bm, Vector((bx, by + wall_d / 1.7, cur_z + rail_h / 2.0)), (wall_w * 1.3, 0.08, rail_h))
        add_box(bm, Vector((bx, by - wall_d / 1.7, cur_z + rail_h / 2.0)), (wall_w * 1.3, 0.08, rail_h))

        # Flared Eaves / Pagoda Overhanging Roof
        roof_z = cur_z + tier_h
        roof_w = cur_w * 1.35
        roof_d = cur_d * 1.35
        roof_thick = 0.4

        # Flared roof slab extending past walls
        add_box(bm, Vector((bx, by, roof_z + roof_thick / 2.0)), (roof_w, roof_d, roof_thick))
        # Swept hip roof pyramid
        add_cylinder(bm, Vector((bx, by, roof_z + roof_thick + 0.6)), radius=roof_w * 0.42, depth=1.2, segments=4, rot_z=math.radians(45))

        cur_z = roof_z + roof_thick + 0.8
        cur_w *= 0.78
        cur_d *= 0.78

    # Top spire / finial
    add_cylinder(bm, Vector((bx, by, cur_z + 1.2)), radius=0.1, depth=2.4, segments=8)

def build_modern_skyscraper(bm, base_pos, w, d, floors=15, floor_h=2.8):
    """High-rise corporate tower with stepped setbacks & vertical fins."""
    bx, by, bz = base_pos
    cur_z = bz
    cur_w = w
    cur_d = d

    # 3-Tier Step-down
    setbacks = [
        {"floors": int(floors * 0.45), "inset": 0.0},
        {"floors": int(floors * 0.35), "inset": 1.4},
        {"floors": int(floors * 0.20), "inset": 2.6},
    ]

    for tier in setbacks:
        tw = max(4.0, cur_w - tier["inset"] * 2.0)
        td = max(4.0, cur_d - tier["inset"] * 2.0)
        th = tier["floors"] * floor_h

        # Main volume
        add_box(bm, Vector((bx, by, cur_z + th / 2.0)), (tw, td, th))

        # Vertical facade fins (mullions)
        fin_spacing = 1.4
        num_fins = max(2, int(tw // fin_spacing))
        fin_start = -((num_fins - 1) * fin_spacing) / 2.0
        for i in range(num_fins):
            fx = bx + fin_start + i * fin_spacing
            add_box(bm, Vector((fx, by + td / 2.0 + 0.08, cur_z + th / 2.0)), (0.1, 0.16, th))
            add_box(bm, Vector((fx, by - td / 2.0 - 0.08, cur_z + th / 2.0)), (0.1, 0.16, th))

        # Roof ledge
        add_box(bm, Vector((bx, by, cur_z + th + 0.15)), (tw + 0.3, td + 0.3, 0.3))

        cur_z += th

    # Rooftop Helipad & Antenna Array
    add_box(bm, Vector((bx, by, cur_z + 0.2)), (tw * 0.8, td * 0.8, 0.4))
    # Helipad 'H' ring
    add_cylinder(bm, Vector((bx, by, cur_z + 0.45)), radius=tw * 0.32, depth=0.1, segments=16)
    # Tall communications mast
    add_cylinder(bm, Vector((bx + tw * 0.2, by + td * 0.2, cur_z + 2.5)), radius=0.08, depth=5.0, segments=8)

def build_sports_park(bm, base_pos, w, d):
    """Soccer/sports pitch with stadium floodlights (like in reference background)."""
    bx, by, bz = base_pos
    # Green turf pitch
    add_box(bm, Vector((bx, by, bz + 0.05)), (w - 1.0, d - 1.0, 0.1))

    # White line markings
    line_h = 0.11
    # Center circle
    add_cylinder(bm, Vector((bx, by, bz + line_h)), radius=2.5, depth=0.02, segments=16)
    # Center halfway line
    add_box(bm, Vector((bx, by, bz + line_h)), (w - 2.0, 0.15, 0.02))
    # Boundary box
    add_box(bm, Vector((bx, by + (d - 2.0) / 2.0, bz + line_h)), (w - 2.0, 0.15, 0.02))
    add_box(bm, Vector((bx, by - (d - 2.0) / 2.0, bz + line_h)), (w - 2.0, 0.15, 0.02))

    # 2 Goal Posts
    for gy in [-(d - 3.0) / 2.0, (d - 3.0) / 2.0]:
        add_box(bm, Vector((bx, by + gy, bz + 0.8)), (2.4, 0.12, 1.4))

    # 4 Stadium Floodlight Masts
    light_h = 10.0
    for lx in [-w / 2.2, w / 2.2]:
        for ly in [-d / 2.2, d / 2.2]:
            # Tall angled mast
            add_cylinder(bm, Vector((bx + lx, by + ly, bz + light_h / 2.0)), radius=0.18, depth=light_h, segments=6)
            # Floodlight bank
            bank_pos = Vector((bx + lx, by + ly, bz + light_h))
            add_box(bm, bank_pos, (2.2, 0.8, 1.2), rot_z=math.radians(30))

# ==========================================
# MAIN HIGH-FIDELITY CITY GENERATOR
# ==========================================
def generate_high_fidelity_city(config_path, out_dir):
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    layout = cfg.get("layout", "standard")
    height_profile = cfg.get("height_profile", "mixed")
    metrics = cfg.get("metrics", {})
    dummy_character = bool(cfg.get("dummy_character", True))
    random.seed(seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def metric(name, default, lo, hi):
        try:
            value = float(metrics.get(name, default))
        except (TypeError, ValueError):
            value = float(default)
        return max(lo, min(hi, value))

    # Grid settings
    if layout == "compact":
        cols, rows = 2, 2
        default_plot_w, default_plot_d = 13.0, 13.0
    elif layout == "avenue":
        cols, rows = 4, 2
        default_plot_w, default_plot_d = 11.0, 13.0
    else:  # standard
        cols, rows = 3, 2
        default_plot_w, default_plot_d = 11.5, 13.0

    plot_w = metric("plot_width_m", default_plot_w, 8.0, 28.0)
    plot_d = metric("plot_depth_m", default_plot_d, 8.0, 30.0)
    road_w = metric("road_width_m", 11.0, 6.0, 24.0)
    curb_h = metric("curb_height_m", 0.22, 0.05, 0.6)
    gap_x = metric("plot_gap_x_m", 2.4, 0.4, 10.0)
    gap_y = metric("plot_gap_y_m", 3.2, 0.4, 10.0)
    sidewalk_pad = metric("sidewalk_pad_m", 3.2, 0.8, 10.0)
    floor_h = metric("floor_height_m", 2.8, 2.2, 4.4)

    if height_profile == "low_rise":
        apartment_range = (3, 6)
        skyscraper_range = (7, 11)
    elif height_profile == "towers":
        apartment_range = (7, 11)
        skyscraper_range = (15, 22)
    else:
        apartment_range = (5, 8)
        skyscraper_range = (12, 17)

    block_w = cols * plot_w + (cols - 1) * gap_x + sidewalk_pad * 2.0
    block_d = rows * plot_d + (rows - 1) * gap_y + sidewalk_pad * 2.0
    total_w = block_w + road_w * 2.4
    total_d = block_d + road_w * 2.4

    # ------------------------------------------
    # 1. ROADS, SIDEWALKS & CROSSWALKS
    # ------------------------------------------
    env_mesh = bpy.data.meshes.new("Environment_Streets_Mesh")
    bm_env = bmesh.new()

    # Asphalt Ground
    add_box(bm_env, Vector((0, 0, -0.05)), (total_w, total_d, 0.1))
    # Raised Sidewalk Plinth
    add_box(bm_env, Vector((0, 0, curb_h / 2.0)), (block_w, block_d, curb_h))

    # Road Dashed Markings
    dash_len, dash_gap, dash_t, dash_h = 2.4, 2.4, 0.25, 0.02
    for y_pos in [-(block_d / 2.0 + road_w / 2.0), (block_d / 2.0 + road_w / 2.0)]:
        x = -total_w / 2.0 + 3.0
        while x < total_w / 2.0 - 3.0:
            add_box(bm_env, Vector((x + dash_len / 2.0, y_pos, dash_h)), (dash_len, dash_t, dash_h))
            x += dash_len + dash_gap

    for x_pos in [-(block_w / 2.0 + road_w / 2.0), (block_w / 2.0 + road_w / 2.0)]:
        y = -total_d / 2.0 + 3.0
        while y < total_d / 2.0 - 3.0:
            add_box(bm_env, Vector((x_pos, y + dash_len / 2.0, dash_h)), (dash_t, dash_len, dash_h))
            y += dash_len + dash_gap

    # Zebra Crosswalks at 4 intersections
    stripe_w = 0.5
    stripe_len = 3.8
    stripe_gap = 0.35
    for cx in [-block_w / 2.0 - road_w / 2.0, block_w / 2.0 + road_w / 2.0]:
        for cy in [-block_d / 2.0, block_d / 2.0]:
            # 6 parallel white zebra stripes
            for s in range(-3, 3):
                add_box(bm_env, Vector((cx + s * (stripe_w + stripe_gap), cy, dash_h)), (stripe_w, stripe_len, dash_h))

    # Traffic Lights on corners
    traffic_corners = [
        (-(block_w / 2.0 + 0.8), -(block_d / 2.0 + 0.8), 0.0),
        ((block_w / 2.0 + 0.8), (block_d / 2.0 + 0.8), math.radians(180)),
    ]
    for tx, ty, trot in traffic_corners:
        build_traffic_light(bm_env, (tx, ty, curb_h), trot)

    # ------------------------------------------
    # 2. VEHICLES (Bus, Cars, Vans)
    # ------------------------------------------
    # Green city transit bus driving on bottom road
    build_bus(bm_env, (-6.0, -(block_d / 2.0 + road_w / 2.0 + 2.0), 0.0), rot_z=math.radians(90))
    # Cars driving & parked
    build_car(bm_env, (12.0, -(block_d / 2.0 + road_w / 2.0 - 2.0), 0.0), rot_z=math.radians(-90))
    build_car(bm_env, (-(block_w / 2.0 + road_w / 2.0 - 2.0), 8.0, 0.0), rot_z=0.0)
    build_car(bm_env, ((block_w / 2.0 + road_w / 2.0 + 2.0), -4.0, 0.0), rot_z=math.radians(180))

    bm_env.to_mesh(env_mesh)
    bm_env.free()
    env_obj = bpy.data.objects.new("Environment_Streets_Vehicles", env_mesh)
    bpy.context.collection.objects.link(env_obj)

    # ------------------------------------------
    # 3. BUILDINGS & ARCHETYPES
    # ------------------------------------------
    # Curated archetype slots for rich visual diversity
    # Top-right: Pagoda temple
    # Top-center: High-rise corporate tower
    # Top-left: Sports park / stadium pitch
    # Bottom-left: Classic tenement apartment with fire escape
    # Bottom-center: Mixed apartment
    # Bottom-right: Corner bodega / storefront with awning
    archetypes = [
        "apartment", "skyscraper", "bodega",
        "sports_park", "skyscraper", "pagoda"
    ]
    random.shuffle(archetypes)
    # Ensure at least 1 pagoda and 1 bodega are present
    if "pagoda" not in archetypes: archetypes[0] = "pagoda"
    if "bodega" not in archetypes: archetypes[1] = "bodega"

    x_start = -((cols - 1) * (plot_w + gap_x)) / 2.0
    y_start = -((rows - 1) * (plot_d + gap_y)) / 2.0

    b_idx = 0
    tactical_intel = {
        "grid": {"cols": cols, "rows": rows, "block_width": block_w, "block_depth": block_d},
        "metrics": {
            "plot_width_m": plot_w,
            "plot_depth_m": plot_d,
            "road_width_m": road_w,
            "sidewalk_pad_m": sidewalk_pad,
            "plot_gap_x_m": gap_x,
            "plot_gap_y_m": gap_y,
            "curb_height_m": curb_h,
            "floor_height_m": floor_h,
        },
        "buildings": [],
        "cover_points": [],
        "sniper_perches": [],
        "test_dummy": None,
        "spawns": {"player": [-block_w / 2.0 - 4.0, 0, curb_h], "enemies": [[block_w / 2.0 + 4.0, 0, curb_h]]}
    }

    for r in range(rows):
        for c in range(cols):
            b_idx += 1
            bx = x_start + c * (plot_w + gap_x)
            by = y_start + r * (plot_d + gap_y)
            arch = archetypes[(b_idx - 1) % len(archetypes)]

            b_mesh = bpy.data.meshes.new(f"Mesh_{arch}_{b_idx}")
            bm = bmesh.new()

            if arch == "pagoda":
                build_pagoda_temple(bm, (0, 0, 0), plot_w, plot_d)
            elif arch == "bodega":
                build_corner_bodega(bm, (0, 0, 0), plot_w, plot_d)
            elif arch == "skyscraper":
                floors = random.randint(skyscraper_range[0], skyscraper_range[1])
                build_modern_skyscraper(bm, (0, 0, 0), plot_w, plot_d, floors=floors)
                tactical_intel["sniper_perches"].append({"pos": [bx, by, floors * floor_h], "elevation": floors * floor_h, "rating": "S"})
            elif arch == "sports_park":
                build_sports_park(bm, (0, 0, 0), plot_w, plot_d)
            else: # apartment
                floors = random.randint(apartment_range[0], apartment_range[1])
                build_apartment_building(bm, (0, 0, 0), plot_w, plot_d, floors=floors, floor_h=floor_h, has_fire_escape=True)
                tactical_intel["sniper_perches"].append({"pos": [bx, by, floors * floor_h], "elevation": floors * floor_h, "rating": "A"})

            bm.to_mesh(b_mesh)
            bm.free()

            b_obj = bpy.data.objects.new(f"Plot_{b_idx}_{arch.capitalize()}", b_mesh)
            b_obj.location = (bx, by, curb_h)
            bpy.context.collection.objects.link(b_obj)

            # Apply Bevel Modifier to all buildings (rounds sharp computer edges into smooth miniature highlights)
            bev = b_obj.modifiers.new(name="Bevel", type='BEVEL')
            bev.width = 0.05
            bev.segments = 1
            bev.limit_method = 'ANGLE'
            bev.angle_limit = math.radians(35)

            tactical_intel["buildings"].append({
                "id": b_obj.name,
                "archetype": arch,
                "world_pos": [bx, by, curb_h],
                "entrance": [bx, by - plot_d / 2.0, curb_h],
            })

    if dummy_character:
        dummy_mesh = bpy.data.meshes.new("Test_Dummy_Mesh")
        bm_dummy = bmesh.new()
        dummy_pos = (-block_w / 2.0 + 2.4, -0.5, curb_h)
        build_test_dummy_character(bm_dummy, dummy_pos, height=1.8)
        bm_dummy.to_mesh(dummy_mesh)
        bm_dummy.free()
        dummy_obj = bpy.data.objects.new("Test_Dummy_Character", dummy_mesh)
        bpy.context.collection.objects.link(dummy_obj)
        tactical_intel["test_dummy"] = {"id": dummy_obj.name, "pos": [dummy_pos[0], dummy_pos[1], dummy_pos[2]], "height": 1.8}

    # ------------------------------------------
    # 4. ISOMETRIC CAMERA & HIGH-END SHADING
    # ------------------------------------------
    cam_data = bpy.data.cameras.new("CityIsoCamera")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max(block_w, block_d) * 1.55
    cam_obj = bpy.data.objects.new("CityIsoCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_dist = max(block_w, block_d) * 1.35
    cam_obj.location = (cam_dist, -cam_dist, cam_dist * 0.95)
    cam_obj.rotation_euler = (math.radians(58), math.radians(0), math.radians(45))
    bpy.context.scene.camera = cam_obj

    # Dual Sun Lighting for Rich Tactile Clay Shading
    light1_data = bpy.data.lights.new("SunKey", type='SUN')
    light1_data.energy = 4.2
    light1_obj = bpy.data.objects.new("SunKey", light1_data)
    bpy.context.collection.objects.link(light1_obj)
    light1_obj.rotation_euler = (math.radians(54), math.radians(24), math.radians(-32))

    light2_data = bpy.data.lights.new("SunFill", type='SUN')
    light2_data.energy = 1.4
    light2_obj = bpy.data.objects.new("SunFill", light2_data)
    bpy.context.collection.objects.link(light2_obj)
    light2_obj.rotation_euler = (math.radians(-40), math.radians(-30), math.radians(120))

    # High-end Workbench Ambient Occlusion & Cavity Settings
    scene = bpy.context.scene
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 900
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'MATCAP'
    scene.display.shading.color_type = 'SINGLE'
    # Cavity / Ambient contact shadows
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'
    scene.display.shading.cavity_ridge_factor = 1.6
    scene.display.shading.cavity_valley_factor = 2.4
    scene.display.shading.curvature_ridge_factor = 1.6
    scene.display.shading.curvature_valley_factor = 2.2

    # Save .blend
    out_blend = os.path.join(out_dir, "city_foundry.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    # Render Preview
    out_img = os.path.join(out_dir, "city_foundry_preview.png")
    scene.render.filepath = out_img
    bpy.ops.render.render(write_still=True)

    # Export .GLB
    out_glb = os.path.join(out_dir, "city_foundry.glb")
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format='GLB',
        use_selection=False,
    )

    # Save tactical metadata
    out_json = os.path.join(out_dir, "tactical_level_data.json")
    with open(out_json, "w") as f:
        json.dump(tactical_intel, f, indent=2)

    print(f"HIGH_FIDELITY_SUCCESS: Blend={out_blend}, Preview={out_img}, GLB={out_glb}")

if __name__ == "__main__":
    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        cfg_path = args[idx + 1]
        out_dir = args[idx + 2]
    else:
        cfg_path = os.path.join(script_dir, "default_config.json")
        out_dir = script_dir
    generate_high_fidelity_city(cfg_path, out_dir)
