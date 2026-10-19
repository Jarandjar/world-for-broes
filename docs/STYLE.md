# World 4 Bro's — Visual Style Bible & Animation Specification
*Document Version: 1.0.0 — Approved Phase Z0D Style Direction: Pop-Up Paper & Rubber-Hose Hybrid*

---

## 1. Style Direction Summary
**Pop-Up Paper & Rubber-Hose Hybrid**:
The game renders as an interactive, mechanical pop-up storybook diorama seen through a tilted macro lens. The world is constructed of thick textured cardboard, creased paper cutouts, and physical hinged flaps. Characters and creatures inhabit this world as chunky, bouncy rubber-hose cutout puppets with elastic squash-and-stretch, expressive facial animations, and springy secondary motion.

The tone is unapologetically warm, comedic, and slapstick. The zombie apocalypse is treated as goofy cartoon chaos: defeated zombies burst into green comic smoke puffs, weapons go `"BONK!"`, and pets are immortal companions that celebrate with heart sparkles.

---

## 2. Shape Language & Geometry
Every silhouette in the game strictly follows a functional, readable shape hierarchy:

| Entity Category | Dominant Shapes | Silhouette Rules | Visual Personality |
| :--- | :--- | :--- | :--- |
| **Friends & Companions** *(Player, Pets, Town NPCs)* | Bulbous circles, soft ovals, rounded pills, pillowy cushions | Zero sharp spikes; radius corners $\ge 12$px equivalent; friendly silhouettes readable from dynamic camera zoom. | Warm, huggable, approachable, bouncy. |
| **Zombies & Horde** *(Shamblers, Joggers, Screamers)* | Asymmetrical lumps, wobbly potato contours, loose dough | Mismatched googly eyes (one large, one small), oversized dropped lower jaws, floppy noodle limbs, hunched rounded backs. | Funny, clumsy, bewildered rather than frightening. |
| **Active Hazards & Real Danger** *(Spike traps, Boss cleaves, Explosives)* | Sharp zig-zags, aggressive triangles, jagged sawtooth bursts | High-contrast diagonals, angular spikes ($\le 45^\circ$ vertices), hot saturated rim highlights. | Instantly telegraphs lethal threat and timing. |
| **World Scenery & Buildings** | Folded cardstock planes, layered flats, creased flaps | Visible paper thickness (2–4mm scale bevel), slight physical warping, angled fold score lines. | Tactile mechanical pop-up book diorama. |

---

## 3. Line Weight & Light-Tinted Outlines
Pure digital black (`#000000`) is strictly forbidden across all line art and 3D silhouette outlines. Pure black flattens depth and feels sterile.

### 3.1 Outline Weight Rules
- **Hero Outlines (Player, Bosses, Active Vehicle)**: $3.5\text{px} - 4.5\text{px}$ stroke width.
- **Environment & Standard Props**: $2.0\text{px} - 2.5\text{px}$ stroke width.
- **Internal Details (Pockets, Flaps, Creases)**: $1.0\text{px} - 1.5\text{px}$ tapered hairline strokes.
- **Variable-Width Modulation**: Outlines dynamically taper at tips and thicken at concave intersections and weight-bearing contacts, emulating hand-dipped ink pens.

### 3.2 Environmental Light Tinting
Outlines are tinted dynamically based on ambient sunlight and local point lights:
- **Direct Sunlight / Warm Daylight**: Deep roasted chestnut (`#2E1810`) to rich sepia (`#3B2219`).
- **Cool Ambient Shadow**: Deep slate charcoal (`#1A222B`) to midnight indigo (`#171B2B`).
- **Dusk & Golden Hour**: Rich warm plum-auburn (`#3A1624`).
- **Night & Hollow Biome**: Dark obsidian violet (`#160F24`).
- **Toxic / Chemical Glow**: Deep murky pine-ink (`#0E2117`).

### 3.3 Hand-Drawn Jitter (Juice Dial Effect)
When the Juice Dial is $> 30\%$, outlines undergo a subtle procedural 2s/3s hand-drawn line jitter (cycling 3 hand-inked line sample offsets at 8–12 Hz) while characters continue moving smoothly at 60 fps. This maintains an organic storybook feel without causing visual fatigue.

---

## 4. Biome Color Palettes & Color-Grading LUTs
Each of the 10 biomes is governed by a curated 5-swatch master palette. All scene textures, props, and lighting must derive from these tokens:

```
[Bro Meadow]      #2D5A27 (Forest Base)   #60993E (Meadow Green)  #F4D06F (Warm Butter)   #8B5A2B (Wood Bark)    #E76F51 (Terracotta)
[Pinewood]        #132A26 (Deep Teal)     #204E4A (Pine Fog)      #4A7C59 (Lichen)        #DDA15E (Amber Pitch)  #9B5DE5 (Night Spore)
[Squelchy Bog]    #333D29 (Olive Sludge)  #656D4A (Moss Sponge)   #8F5D88 (Plum Bloom)    #D4A373 (Cattail Tan)  #52B788 (Biolum Spore)
[Rockbottom]      #2B2D42 (Slate Navy)    #8D99AE (Cliff Stone)   #EDF2F4 (Flint Glint)   #D90429 (Warning Ruby) #FFB703 (Gold Vein)
[Sunburn Flats]   #6B2D0C (Adobe Clay)    #D86C27 (Desert Sand)   #F3A738 (Sun Ochre)     #E9D8A6 (Bleached Bone)#00A896 (Oasis Cyan)
[Brrington]       #1D3557 (Deep Fjord)    #457B9D (Ice Blue)      #A8DADC (Frost Powder)  #F1FAEE (Snow Surface) #E63946 (Wool Beanie)
[Scrapyard Gulch] #3A2E2B (Cast Iron)     #8C4830 (Rusted Hull)   #D9822B (Industrial)    #2EC4B6 (Neon Sign)    #F6BD60 (Zinc Plate)
[The Hollow]      #120D1D (Void Velvet)   #2B1B3D (Nightmare Fog) #5A3D7A (Amethyst Gloom)#70D6FF (Spirit Wisp)  #FF0054 (Sanity Dread)
[Splash Cove]     #0077B6 (Tide Deep)     #00B4D8 (Lagoon Teal)   #90E0EF (Wave Crest)    #F3C98B (Shell Sand)   #FF595E (Buoy Coral)
[Mount Grumble]   #1B1212 (Basalt Obsidian)#3D1C14 (Burnt Crust)  #B83918 (Cinder Rust)   #E65F19 (Liquid Magma) #FFB703 (Sulfur Spark)
```

### 4.1 Global Semantic Color Codes
- **Loot & High Value**: Sparkle Gold (`#FFD700`) + Radiant Yellow (`#FFE66D`).
- **Interactive & Friendly**: Cyan Aqua (`#00E5FF`) + Mint (`#48CAE4`).
- **Danger & Aggression**: Safety Orange-Red (`#FF3D00`) + Crimson (`#D00000`).
- **Sanity & Decay**: Phosphor Purple (`#9D4EDD`) + Toxic Lime (`#70E000`).

### 4.2 Status Effect Colours & Icons
Each status effect has its own colour, icon shape and sound word, so colour is never the only signal (World Bible §115, design page `docs/design/index.html`). Comic visuals only: green drips, flames, starbursts, soot and ash; no gore. Flashes follow the reduced-flashing setting.
- Acid `#b6e64a` (drips and bubbles) · Burn `#F08A32` (flame tongues) · Explosion `#F9C45A` (starburst ring) · Stagger `#EAE0C8` (circling stars)
- Electricity `#FDEB90` (zigzag bolts) · Cold `#8fd8e8` (ice shards) · Spores `#c084fc` (floating dots) · Goo `#f472b6` (sticky trail)
- Wet `#67e8f9` (droplets and ripple) · Oil slick `#9aa1ac` (black pool with sheen) · Blind `#f3e7b4` (splat over the face) · Smoke `#9aa1ac` (grey puff clusters)

### 4.3 Proposed Biome Palettes
The six proposed biomes of World Bible §1 and §119 (Toxic Marsh, Cinder Wastes, Storm Plains, Salt Flats, Crystal Caverns, Ruined Downtown) have 4-swatch palettes in §119 and on `docs/design/biomes.html`. Give them full 5-swatch palettes here when Z27 starts.

---

## 5. Shading Ramp & Shader Specification
All 3D geometry in the expansion runs through a custom **Toon Diorma Ramp Shader**:

### 5.1 Three-Band Stepped Ramp
Surfaces evaluate dot product $N \cdot L$:
1. **Highlight Band ($N \cdot L > 0.65$)**: Base color shifted $+12\%$ brightness, $+5^\circ$ warm hue shift (toward peach/gold).
2. **Midtone Band ($0.15 < N \cdot L \le 0.65$)**: Pure authored texture swatch.
3. **Shadow Band ($N \cdot L \le 0.15$)**: Base color shifted $-25\%$ brightness, $+15^\circ$ cool hue shift (toward indigo/periwinkle slate), cross-hatched with paper grain.

### 5.2 Toon Rim Light
Surfaces evaluate fresnel factor $(1.0 - N \cdot V)^3$:
- Rim color is driven by key directional sun/moon light.
- Sharp step function (cutoff at $0.6$) creates a crisp paper cutout edge highlight that pops characters forward against cluttered forest or city backgrounds.

### 5.3 Blob Ambient Occlusion
- Every character, zombie, pet, and vehicle casts a soft, slightly squashed 2D ellipse shadow (`#15121E` with $0.45$ alpha) onto the ground plane ($y=0.01$).
- Shadow scales dynamically with jumping/spring elevation: $\text{scale} = 1.0 / (1.0 + 0.5 \cdot \Delta y)$, $\text{alpha} = 0.45 / (1.0 + 0.8 \cdot \Delta y)$.

---

## 6. Camera Post-Processing & Diorama Lens
To evoke the miniature scale of a tabletop diorama, the post-processing pipeline layers three key stages:

1. **Tilt-Shift Depth of Field (DoF)**:
   - Screen-space vertical gradient applying a dual-pass Gaussian blur to the top $18\%$ and bottom $14\%$ of the screen.
   - The central horizontal band ($68\%$ height) centered on the player stays tack-sharp.
2. **Warm Campfire & Lantern Bloom**:
   - Selective threshold bloom ($> 0.85$ luminance) on fires, torches, headlights, and gold loot beams with a soft, warm diffusion spread.
3. **Subtle Paper Grain Overlay**:
   - Monochromatic 128x128 fine cardstock fiber texture composited in blend mode `Soft Light` at $6\%$ opacity across the final buffer.
   - Eliminates digital plastic gradient banding and reinforces tactile physical paper feel.

---

## 7. UI Sticker Kit & Onomatopoeia Comic Pops

### 7.1 Physical Sticker Kit Design
- All buttons, stat cards, hotbar slots, and dialog windows are styled as die-cut stickers and scraps of textured masking tape.
- **Physical Traits**:
  - Crisp $1.5$px white adhesive border (`#FFFFFF`).
  - Subtle drop shadow: `box-shadow: 2px 3px 0px rgba(25, 20, 30, 0.35)`.
  - Micro rotation jitter: individual UI elements are slightly rotated between $-1.2^\circ$ and $+1.2^\circ$ to look hand-placed.
  - Hover / Touch feedback: element squashes to scale $(1.06, 0.94)$ and straightens to $0^\circ$ on hover.

### 7.2 Onomatopoeia Comic Text ("Word Pops")
Hits, explosions, honks, and comedic interactions spawn 2D billboarded comic sound words:

| Event | Comic Word | Primary Color | Fill Pattern | Sound Effect |
| :--- | :--- | :--- | :--- | :--- |
| Blunt Melee Hit | `"BONK!"` | Bright Yellow `#FFEA00` | Red outline with diagonal comic speed lines | Frying pan clang / mallet thud |
| Zombie Pop Death | `"POOF!"` / `"POP!"` | Toxic Lime `#70E000` | White outline with smoke puffs | Squeak & green bubble burst |
| Metal Impact / Car Bump | `"CLANG!"` | Chrome Orange `#FF9100` | Heavy black outline with spark bursts | Trash can lid rattle |
| Rubber Chicken Swing | `"SQUEEEAK!"` | Hot Magenta `#FF007F` | Wobbly curved rubber text | Pitch-randomized rubber chicken |
| Goofy Car Horn | `"HONK!"` / `"BEEP!"` | Cyan Aqua `#00F5D4` | Radiating sound-wave rings | Goose honk / air horn blast |
| Door Squeak / Closet Hide | `"CREEEAK..."` | Pale Parchment `#E2D4B7` | Dripping ink text | Creaky cabinet door hinge |

**Spawn Animation Curve**:
- Frame 0: Spawn at point of impact, scale $0.2$, angle $\pm 12^\circ$.
- Frame 3 (50 ms): Elastic punch to scale $1.35$.
- Frame 8 (130 ms): Settle to scale $1.0$.
- Frame 20–30 (330–500 ms): Float upward $+1.2$m, fade opacity $1.0 \to 0.0$.

---

## 8. Animation Style Guide & Four-Tier Locomotion

### 8.1 Locomotion Fundamentals
- **Stride Matching**: Walking ($2.8\text{ m/s}$, stride $1.2\text{m}$) and Running ($5.5\text{ m/s}$, stride $1.8\text{m}$) dynamically scale playback speed $\dot{\phi} = \|\vec{v}\| / \text{stride}$. Zero visible foot sliding at any velocity.
- **Banking into Turns**: Turning rotates the puppet's root bone around the velocity tangent by $\theta = -0.08 \cdot v \cdot \omega$, leaning into cornering with a bouncy rebound spring ($f=6\text{ Hz}, \zeta=0.65$).
- **Skid Stop**: Abruptly releasing movement keys while sprinting triggers a 180 ms skid: the character leans backward $18^\circ$, legs slide forward, dust puff particles spawn at shoe soles, and a rubber shoe squeak plays.

### 8.2 Combat Motion Phases
1. **Anticipation (Wind-up)**: $120\text{ ms} - 180\text{ ms}$. Character pulls weapon back, torso squashes down $15\%$, silhouette coils like a spring.
2. **Smear Frame**: 1–2 frames ($16\text{ ms} - 33\text{ ms}$). Weapon mesh stretches along arc with an arc trail decal.
3. **Hit Frame & Isolated Hit-Stop**:
   - `hit_frame` event fires: applies damage, plays audio, spawns comic word.
   - **Isolated Hit-Stop ($45\text{ ms}$)**: Attacker and victim freeze frame completely. The rest of the game world, camera, and ambient particles continue running at full 60 fps.
4. **Follow-Through & Settle**: $150\text{ ms} - 220\text{ ms}$. Weapon overshoots target, arm rebounds slightly, body returns to idle.

### 8.3 Secondary Springs & Stat Body Language
- **Secondary Spring Hierarchy**:
  - Hats & beanies: Pitch & yaw spring ($f=5\text{ Hz}, \zeta=0.6$).
  - Dog tails & ears: Wagging harmonic oscillation ($f=8\text{ Hz}, \zeta=0.5$).
  - Backpack: Heavy vertical lag spring ($f=4\text{ Hz}, \zeta=0.8$).
- **Visual Stat Feedback**:
  - **Cold / Freezing**: Arms tucked into chest, shivering oscillation ($12\text{ Hz}$, amplitude $1.5^\circ$), breath smoke puffs.
  - **Hunger**: Hand clutches belly every 8 seconds, posture slouches $10^\circ$.
  - **Infection**: Neon green veins pulse on forehead, character periodically stumbles during sprints.
  - **Heavy Inventory**: Spine tilts forward $14^\circ$, footstep dust puffs double in size.

---

## 9. Performance Tiering, Quality Presets & Juice Dial

### 9.1 LOD Performance Tiers
| Tier | Target Entities | Features Enabled | CPU Frame Budget |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Base)** | All 150+ crowd zombies, distant critters $> 18$m | 1D blend tree locomotion, baked pose textures, simplified 2-frame death pops, zero IK. | $\le 0.8\text{ ms}$ |
| **Tier 2 (Rich)** | Player, active vehicle, pet companions, nearby zombies $< 18$m | Full 4-tier layered state machine, Foot IK, Hand IK, facial eye blinks/look-at, secondary spring bones. | $\le 1.2\text{ ms}$ |
| **Tier 3 (Hero)** | Bosses (Highway Hulk, Quackzilla), Helicopter, Cutscene zooms | Hero Verlet chain physics (winch rope), multi-layer smoke ribbons, bespoke smear meshes, hero camera shake. | $\le 0.4\text{ ms}$ |

### 9.2 Graphics & Animation Settings
- **Animation Quality Setting**:
  - **Low**: Tier 1 across all entities; static shadows; DoF disabled; bloom disabled. (Mobile default)
  - **Medium**: Tier 2 on player/pet/vehicle; simple shadow blobs; tilt-shift DoF enabled; bloom enabled.
  - **High**: Full Tier 1/2/3 enabled; dynamic light-tinted outlines; tilt-shift DoF; bloom; paper grain overlay.
- **Juice Dial (0% to 100%)**:
  - Master slider governing visual comedy punch:
    - $0\%$: Crisp clean motion, zero camera shake, no outline jitter, subtle comic words.
    - $50\%$: Standard recommended setting (punchy hit-stop, bouncy springs, medium comic words).
    - $100\%$: Maximum slapstick (exaggerated squash/stretch, hand-drawn outline jitter, giant comic onomatopoeia pops, juicy dust clouds).

### 9.3 Atmosphere, Weather & Graphics Quality (built in the 2D core, Z30)
- **Quality dial** `off / low / high` (`?gfx=`, remembered): off = darkness and light holes only; low = shadows, weather, vignette, grain, window lights; high adds puddles, mist layers, dust motes, splashes, storm flashes, the 3-sample grain jitter (§3.3) and the crisp 288 px/unit sprite tier.
- **Sun shadows**: drop shadows swing west → east over the day and stretch at dawn and dusk; clouds soften them; at night a faint moon shadow. Shadow ink is `#15121E` (§5.3), never black.
- **Night**: darkness is obsidian violet (§3.2), houses get warm porch (`#ffd890`) and window (`#ffcf7a`) pools, fireflies glow lime-yellow after dark.
- **Weather** rolls per half day from the biome climate: rain and storms only where warm and wet, snow only where cold, dust haze in hot dry biomes. Rain darkens the ground and adds puddles; snow whitens it; overcast and storms add a cool gloom grade.
- **Paper grain** (§6.3): 128 px cardstock fibre tile, `soft-light`. **Vignette**: soft violet, stronger at night and in storms. Tunables live in `ATMOS_TUNING` in `src/render/atmosphere.ts`.

## 10. Vehicles (Z30 HD pass)
- Light comes from the top-left: the shine streak sits on the car's left flank, the right flank is shaded and the shadow falls down-right in world space, whatever the car's heading.
- A driven car is a hero: its body outline is thicker (0.036 vs 0.024 units). Glass is tinted from data with a pale sky band and two white streaks; chrome is the `trim` colour with a bright upper edge.
- Paint, trim, glass, decal, dirt and rust colours are data (`VEHICLE_LOOKS`). Wear appears in steps at 75/50/25% hp, never per frame. Lamp glows use additive blending and are the only per-frame light work.