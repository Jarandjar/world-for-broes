# Playable Isometric City & Modular Traversal Architecture

> **Design Specification & Technical Blueprint**  
> **Target Project**: *World 4 Bros* / *CityFoundry*  
> **Core Objective**: A fully playable, enterable, multi-tier parkour-ready low-poly isometric urban environment with standardized 3D tile kits, floor slicing, and cross-street traversal networks.

---

## 1. High-Level Vision & Core Principles

To elevate procedural low-poly cities to a rich, commercial-grade tactical playground (matching the aesthetics of stylized diorama reference art while supporting fluid gameplay):

1. **Never Generate Monolithic Blocks**: All buildings are assembled from standardized modular tiles with hollow interiors, wall thicknesses, and room partitions.
2. **The "Playable Shell" Rule**: Every building must be enterable, featuring interior stairwells, floor slabs with pass-through cutouts, and camera-friendly cutaway foreground walls.
3. **The Traversal Chain**: Streets, streetlamps, vehicle roofs, awnings, windows, fire escapes, and rooftops form an unbroken vertical stepping-stone matrix.
4. **The Standardized Golden Grid**: Every piece aligns with a **2.0m × 2.0m footprint** and a **3.0m vertical floor height**.

```
                           THE URBAN ELEVATION MATRIX
 15.0m ─── Rooftop Helipads & Communication Masts
 12.0m ─── Rooftop Decks & Parapet Low-Cover (Sniper Roosts)
  9.0m ─── Floor 3: Exterior Fire Escape Balconies & Catwalks
  6.0m ─── Floor 2: Window Vault Ledges
  4.2m ─── Streetlight Horizontal Balance Arms & Overhead Wires
  3.0m ─── Transit Bus Roofs & Storefront Awnings
  1.4m ─── Sidewalk Dumpsters, Concrete Barriers & Van Roofs
  0.2m ─── Sidewalk Plinth & Curbs
  0.0m ─── Asphalt Roadway & Crosswalks
```

---

## 2. The 17-Piece Building Tile Inventory

All building pieces are strictly dimensioned to socket cleanly into an integer 3D grid: `Grid[X, Y, Z]`.

### Group A: Walls & Apertures (2.0m W × 3.0m H × 0.25m Thick)

| Tile ID | Name | Visual Details | Gameplay Function |
| :--- | :--- | :--- | :--- |
| **`W-01`** | `Wall_Solid` | Plain wall with baseboard trim and exterior brick/paneling. | Solid line-of-sight and bullet block. Sockets on all 4 borders. |
| **`W-02`** | `Wall_Door` | Pre-cut **1.2m W × 2.2m H** centered doorway. | Walkable transition between rooms or street. Accepts interactive doors. |
| **`W-03`** | `Wall_Window` | Recessed window frame (**1.0m W × 1.4m H**, sill at 0.9m). | Shoot-through, peek-through, and vault-through opening. Exterior anchor for awnings and fire escapes. |
| **`W-04`** | `Wall_Half_Cutaway` | Low **0.8m H** waist wall. | **Camera Visibility**: Placed on camera-facing sides so players have an unobstructed view of room interiors. Acts as half-cover. |
| **`W-05`** | `Wall_Corner_Pillar`| 0.25m × 0.25m vertical post. | Structural corner cap sealing 90° exterior turns. |

### Group B: Floor Slabs & Vertical Shafts (2.0m W × 2.0m D × 0.20m Thick)

| Tile ID | Name | Visual Details | Gameplay Function |
| :--- | :--- | :--- | :--- |
| **`F-01`** | `Floor_Solid` | Walkable concrete/wood tileable slab. | Ground plane for rooms and corridors. |
| **`F-02`** | `Floor_Stair_Hole` | Slab with a **2.0m × 1.0m** rectangular cutout. | Allows player avatars to run upstairs without colliding with the ceiling slab. |
| **`F-03`** | `Floor_Roof_Hatch` | Iron trapdoor hatch opening. | Access point between the highest interior room and the rooftop deck. |

### Group C: Vertical Circulation (Stairs & Ladders)

| Tile ID | Name | Visual Details | Gameplay Function |
| :--- | :--- | :--- | :--- |
| **`V-01`** | `Stair_Straight` | 15 steps with side safety handrails (35° incline, spans 2m horizontal × 3m vertical). | **Contains an invisible smooth ramp collider** on top for frictionless running without step-stutter. |
| **`V-02`** | `Ladder_Wall` | Steel rungs (0.6m W × 3.0m H). | Vertical climbable ladder volume; mounts flush to `W-01`. |
| **`V-03`** | `Fire_Escape_Balcony`| Metal grating platform (2.0m W × 1.2m D) with perimeter railings. | Snaps outside upper-floor windows (`W-03`); exterior intermediate landing. |
| **`V-04`** | `Fire_Escape_Stairs` | Diagonal metal exterior flight linking two `V-03` platforms. | Alternative exterior infiltration/escape route bypassing ground-floor doors. |

### Group D: Rooftops & Perimeter Parapets

| Tile ID | Name | Visual Details | Gameplay Function |
| :--- | :--- | :--- | :--- |
| **`R-01`** | `Roof_Deck` | Weatherproof asphalt/gravel 2m × 2m slab. | Walkable roof surface; mounts AC units and water towers. |
| **`R-02`** | `Roof_Parapet_Edge` | 2.0m W × 0.6m H perimeter edge wall. | Low cover for rooftop snipers. Tagged with mantleable ledge collision. |
| **`R-03`** | `Roof_Parapet_Corner`| 90° corner piece for the roof border. | Corners the roof perimeter cleanly. |

### Group E: Traversal & Parkour Sockets

| Tile ID | Name | Dimensions | Gameplay Function |
| :--- | :--- | :--- | :--- |
| **`P-01`** | `Streetlight_Pole` | 4.2m pole height, 1.8m horizontal arm. | **Pole**: Climbable ladder volume.<br>**Arm**: 0.2m wide balance beam to crouch or leap from. |
| **`P-02`** | `Storefront_Awning`| 2.0m W × 1.4m D wedge canopy. | Snaps over `W-02` Door at Z = 2.8m. Walkable top surface; cushions falls from 2nd-floor windows. |
| **`P-03`** | `Dumpster_Cover` | 2.0m W × 1.2m D × 1.3m H. | Vaultable cover; stepping stone from street to awning. |
| **`P-04`** | `Powerline_Cable` | Cable spline linking building anchors. | High-wire tightrope traversal across streets/alleys. |

---

## 3. Floor-Slicing & Camera Occlusion System

When an isometric camera views a multi-story building at a 45° angle, the roof and front walls naturally obscure the character. The system resolves this using a **Hybrid Dollhouse + Floor Slice Architecture**:

```
               ISOMETRIC CAMERA (Looking South-West)
                     ╲
                      ╲  45° Angle
                       ▼
      ┌─────────────────────────────────┐ ◄── North Wall (FULL HEIGHT 3.0m)
      │                                 │     (Provides solid backdrop & shadows)
      │      [PLAYER / SQUAD INSIDE]    │
      │                                 │
      └───[Half-Wall]─────[Half-Wall]───┘ ◄── South Wall (CUTAWAY TO 0.8m)
                                              (Zero camera occlusion)
```

1. **Backdrop Walls (North & West faces)**:
   * Placed at **full height (3.0m)**.
   * Provide atmospheric framing, display shelving, wall art, and cast realistic shadows without blocking the camera.
2. **Foreground Walls (South & East faces)**:
   * Placed as **waist-level cutaways (0.8m)** via `W-04`.
   * Allow unobstructed line-of-sight into the room, enabling tactical door-breaching and cover sweeps.
3. **Floor-by-Floor Cross-Fading**:
   * Each floor is a discrete collision/render group (`Floor_1`, `Floor_2`, `Roof`).
   * When the player enters Floor 1, `Floor_2` and `Roof` fade to 0% opacity.
   * Climbing to Floor 2 dims Floor 1 and renders Floor 2 completely.

---

## 4. Street, Road & Alleyway Grid Kit

Streets are designed to integrate directly with the 2.0m building footprint grid:

```
                    CROSS-SECTION OF A TYPICAL 16-METER STREET
 ┌─────────────┬─────────────────────────────────────────────┬─────────────┐
 │  SIDEWALK   │                 ROADWAY                     │  SIDEWALK   │
 │   (4.0m)    │                  (8.0m)                     │   (4.0m)    │
 │ (2 Blocks)  │                (4 Blocks)                   │ (2 Blocks)  │
 ├─────────────┼──────────────────────┬──────────────────────┼─────────────┤
 │ Curb: +0.2m │ Lane 1: Westbound    │ Lane 2: Eastbound    │ Curb: +0.2m │
 │ (Pedestrians│ (Cars, Transit Bus,  │ (Parked Cars, Cover, │ (Awnings,   │
 │  & Cover)   │  Traffic Flow)       │  Loading Bays)       │  Lamps)     │
 └─────────────┴──────────────────────┴──────────────────────┴─────────────┘
```

### Street Tile Palette:
* **`RD-01` Road Straight (4m × 4m)**: Asphalt road slab with yellow dashed lane markings.
* **`RD-02` Crosswalk (4m × 4m)**: White zebra pedestrian crossing with solid vehicle stop line.
* **`RD-03` Intersection (8m × 8m)**: 4-way junction with traffic light gantry sockets.
* **`SW-01` Sidewalk Straight (2m × 2m)**: Raised concrete curb ($+0.2\text{m}$) with storm drain sockets.
* **`SW-04` Bus Stop (4m × 2m)**: Sidewalk segment with a 2.8m high glass canopy shelter.
* **`AL-01` Alley Corridor (4m × 4m)**: Narrow rear utility corridor connecting back doors and fire escapes.

---

## 5. Cross-Street Traversal & Parkour Chains

To eliminate open-street sniper death zones, the environment embeds continuous vertical traversal chains across the roadway:

```
                   AERIAL & VEHICULAR PARKOUR CHAINS
                   
 [BUILDING A: 2nd Floor]                                [BUILDING B: 2nd Floor]
 [W-03 Window: 3.9m]                                    [W-03 Window: 3.9m]
         │                                                      │
         ▼                                                      ▲
  [P-02 Awning: 2.8m]                                    [P-02 Awning: 2.8m]
         │                                                      ▲
         ▼                                                      │
  [Streetlight Arm: 4.2m] ═════[High-Wire Cable: 4.5m]══════════╣
         │                                                      │
         ▼ (Jump down)                                          ▲ (Jump up)
  [Bus Stop Roof: 2.8m] ───► [Transit Bus Roof: 3.0m] ──────────┘
         │                          │
         ▼                          ▼
  [Dumpster: 1.3m]           [Car Hood: 0.9m]
         │                          │
  ───────┴──────────────────────────┴───────────────────────── Road (0.0m)
```

### Key Parkour Rules:
1. **Window-to-Awning Drop**: 2nd-floor windows align directly above ground-floor shop awnings ($1.1\text{m}$ vertical drop, zero fall damage).
2. **Awning-to-Streetlamp Leap**: Sidewalk streetlamps are placed $1.6\text{m}$ from awnings, allowing horizontal leaps onto the lamp's horizontal arm.
3. **Vehicular Stepping Stones**: A parked transit bus roof ($3.0\text{m}$) and bus shelter roof ($2.8\text{m}$) allow players to cross entire 8-meter roadways without touching the asphalt.
4. **The "10-Meter Cover" Rule**: No open road lane extends more than 10 meters without a piece of low cover (concrete barrier, vehicle hood, or dumpster) to slide behind.

---

## 6. Procedural Assembly & Snapping Rules Engine

When assembling a building block, the engine executes in 5 sequential stages:

1. **Stage 1: Footprint Allocation**:
   * Assigns grid bounds (e.g. `4 × 3` cells for a Bodega, `5 × 4` cells for a Tenement).
2. **Stage 2: Stair Spine Placement**:
   * Stacks `V-01` Stairs and `F-02` Cutouts vertically first, ensuring unblocked vertical circulation before room division.
3. **Stage 3: Cardinal Wall Sorting**:
   * South & East faces automatically receive `W-04` Half-Walls (cutaways) or `W-02` Doors.
   * North & West faces receive `W-01` Solid Walls or `W-03` Windows.
4. **Stage 4: Traversal Chain Snapping**:
   * Snaps `P-02` Awnings above doors, aligns `W-03` Windows directly above awnings, and plants `P-01` Streetlamps opposite awnings.
5. **Stage 5: Roof Cap & Parapet Sealing**:
   * Seals top perimeter with `R-02` Parapet edges and places rooftop trapdoors (`F-03`) with ladder access.

---

## 7. Engine Implementation Roadmap

| Milestone | Target Deliverable |
| :--- | :--- |
| **Phase 1** | Model the **17 core tiles** in Blender as standalone `.glb` pieces with standardized pivot points and collision colliders. |
| **Phase 2** | Build the **Tile Grid Snapping Engine** that consumes a block layout JSON and instances the tiles onto the 3D grid. |
| **Phase 3** | Implement the **Floor-Slicing Shader / Script** to toggle roof and floor layer visibility on player entry. |
| **Phase 4** | Place **parkour trigger zones** (ladder climb volumes, mantle ledges, balance beam splines) on the interactive props. |
| **Phase 5** | Export complete, game-ready city blocks into *World 4 Bros* / Godot / Unreal Engine.
