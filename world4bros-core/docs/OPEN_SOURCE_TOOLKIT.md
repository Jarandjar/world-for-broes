# Open-Source Toolkit: Blender, Headless Blender and Game Code

Collected 2026-10-20 so Antigravity, Visual Studio (Copilot) and Claude can build every generator the game needs from open code instead of starting from zero. Licences were read from search results, not from each repo; **open the LICENSE file before copying any code**.

Related: [BUILDING_PIPELINE_PLAN.md](BUILDING_PIPELINE_PLAN.md) (Z33), [HD_GRAPHICS_PLAN.md](HD_GRAPHICS_PLAN.md) §0 (isometric redo), [STYLE_BIBLE.md](STYLE_BIBLE.md) §0.8 (Blender camera).

---

## 1. Licence rules for this project (read first)
- **Tools vs. game**: Blender add-ons are often **GPL**. Running a GPL tool to *make* models and sprites is fine; the output is ours. Copying GPL **code** into the game source (`src/`) would make the game GPL. So: GPL code may live only in `tools/` (run by us, not shipped); game code (`src/`) uses only MIT / Apache / zlib / CC0 code.
- **Assets**: ship only CC0 or assets we made. Check pack-by-pack (Quaternius newer packs may have their own licence; Mixamo: check Adobe's terms for rendered sprites).
- **Map data**: OpenStreetMap is ODbL. Use it for reference only unless we follow ODbL.
- Record every third-party tool or asset in `CREDITS.md` with its licence and link.

---

## 2. Headless Blender (run Blender from the command line)
Blender runs without a window, so agents and `npm` scripts can build and render assets.

| Flag | What it does |
| :--- | :--- |
| `-b`, `--background` | Run without the UI (needed for headless) |
| `-P file.py`, `--python file.py` | Run a Python script |
| `--python-expr "…"` | Run a one-line Python expression |
| `--python-exit-code 1` | Exit with this code if the script raises an error (so CI and npm fail properly) |
| `-Y`, `--disable-autoexec` | Do not run scripts stored inside .blend files (safer for files from elsewhere) |
| `-f N` / `-a` / `-s` / `-e` | Render frame N / the animation / start / end frame |
| `-o path`, `-E engine` | Output path / render engine (`-E help` lists engines) |
| `-- …` | Everything after `--` is passed to our script (`sys.argv`) |

Sources: [Blender manual: Command Line Arguments](https://docs.blender.org/manual/en/2.90/advanced/command_line/arguments.html) (Python options), [latest](https://docs.blender.org/manual/de/latest/advanced/command_line/arguments.html), [4.2 LTS](https://docs.blender.org/manual/de/4.2/advanced/command_line/arguments.html), [man page](https://manpath.be/f38/1/blender).

**Pattern for our tools** (all in `tools/blender/`):
```bash
blender -b -Y --python-exit-code 1 -P tools/blender/batch.py -- --jobs tools/blender/jobs/town.json --out public/assets/buildings
```
- `batch.py` reads `sys.argv[sys.argv.index("--") + 1:]`, builds each job with the generator, calls `render_iso.py`, writes atlases and a report JSON.
- Add npm scripts: `"art:jobs"` (export lots from the game), `"art:buildings"` (the command above), `"art:pack"` (atlas packing).
- Add toolkit fetch scripts: `"toolkit:fetch"` (permissive only), `"toolkit:fetch:tools"` (include GPL tools), `"toolkit:fetch:all"` (include study repos) using `tools/vendor/fetch-toolkit.ps1` and `tools/vendor/toolkit-manifest.json`.
- Set the Blender path once in an environment variable (e.g. `BLENDER_EXE`) so every machine and agent can run it.
- Blender can also be used as a Python module (`bpy` on PyPI) — not verified here; the command line above is the reliable route.

---

## 3. Blender generators by game need

### Buildings
| Tool | Licence | Use for |
| :--- | :--- | :--- |
| [Building Tools](https://github.com/ranjian0/building_tools/wiki) (ranjian0) | MIT (per [BlenderNation](https://www.blendernation.com/2020/05/23/quickly-generate-buildings-with-the-free-building-tools-add-on/), [GameFromScratch](https://gamefromscratch.com/building-tools-blender-add-on-released/)) | Floors, windows, doors, railings, balconies, roofs on a footprint. **Best code to study or fork** for our house generator (MIT, so even code reuse is fine). [Info page](https://blender-addons.org/building-tools-addon/) |
| [Auto-Building](https://github.com/Renart84/Auto-Building/blob/main/docs/source/index.rst) | GitHub, check licence | Geometry Nodes facade scatter, roofs, interiors, UVs (Blender 4.5+) |
| [BuildingNodes](https://github.com/durman/buildingnodes) | GitHub, check licence | Facade "styles" from modelled panels |
| [Buildify](https://www.cgchannel.com/?p=137897) | Free, licence unconfirmed | Modular walls / flat roofs from footprints |
| [Blosm](https://github.com/vvoovv/blosm) | GPL | Import OSM towns as reference (footprints, roof shapes, road widths) |
| Ours: `tools/blender/generate_building.py` | Ours | Greybox mid-rise; needs the fixes in `HD_GRAPHICS_PLAN.md` §0.1c |

### Characters and zombies
| Tool | Licence | Use for |
| :--- | :--- | :--- |
| [MPFB 2](https://www.cgchannel.com/2025/03/check-out-open-source-blender-character-generation-plugin-mpfb-2) (MakeHuman for Blender) | Code GPLv3, bundled assets CC0, no claim on output ([licence](https://static.makehumancommunity.org/about/license.html), [overview](https://digitalproduction.com/2025/03/14/mpfb-2-blenders-new-character-generation-sidekick/)) | Generate survivors and zombie base bodies (age, build, gender), clothes; Blender 4.2+ |
| Rigify (built into Blender) | GPL (tool) | Auto-rig; MPFB 2 supports Rigify meta-rigs and IK helpers |
| Mixamo | Free, Adobe terms | Walk/run/attack/die animations; check terms for rendered sprites |

### Trees and plants
| Tool | Licence | Use for |
| :--- | :--- | :--- |
| [Sapling Tree Gen](https://extensions.blender.org/add-ons/sapling-tree-gen/versions/) | GPL v3+ (tool) | Parametric trees; script it headless for many species and seeds |
| [tree-gen](https://gitee.com/gengumeng/tree-gen) (original: friggog/tree-gen on GitHub) | Code GPL-3.0; models free to use except direct sale as assets | Realistic tree species presets |

### General procedural modelling
| Tool | Licence | Use for |
| :--- | :--- | :--- |
| [Sverchok](https://github.com/ttodosi/sverchok) | GPL-3 | Node-based procedural geometry (architecture-friendly) for fences, rails, street furniture |
| Geometry Nodes (built in) | — | Scatter props, roof clutter, kerbs along curves |

### Sprite rendering and packing
| Tool | Licence | Use for |
| :--- | :--- | :--- |
| Ours: `render_iso.py` (to write) | Ours | Exact camera, light and naming from `STYLE_BIBLE.md` §0.8 |
| [Omnidirectional Isometric Render](https://itch.io/t/4776963/free-blender-addon-omnidirectional-isometric-render) | Free, licence unstated | Compare against our renderer |
| [SpriteMaker](https://superhivemarket.com/products/spritemaker) | GPL, ~$5 | 4/8-direction rendering |
| [free-tex-packer-core](https://github.com/zfkun/free-tex-packer-core) | MIT; tool no longer maintained ([note](https://alternativeto.net/software/free-texture-packer/about)) | Pack renders into PixiJS atlases from a Node script (`npm run art:pack`) |

---

## 4. Game-side code (TypeScript / web)
| Project | Licence | Use for |
| :--- | :--- | :--- |
| PixiJS | MIT | The isometric renderer (approved) |
| [iceoh](https://github-redirect.dependabot.com/sammccord/iceoh) | check | Isometric tile maths and depth values for PixiJS |
| [pixi-tile-utilities](https://unpkg.com/pixi-tile-utilities@4.5.3/README.md) | check | Isometric picking, collision, Tiled map import |
| [gen-city](https://github.com/neki-dev/gen-city) | check | Seeded street grids in TypeScript; ideas for `src/sim/worldgen.ts` |
| [agent-q1/Cities](https://github.com/agent-q1/Cities) | Apache 2.0 | Settlement → roads → lots algorithm (Java; port the idea) |
| [Watabou city generator](https://watabou.itch.io/) | Source public (Haxe) | Organic districts and streets |
| [WaveFunctionCollapse](https://awesome.ecosyste.ms/projects/github.com%2Fmxgmn%2FWaveFunctionCollapse) (mxgmn) | Reported MIT ([1](https://luisllamas.es/en/csharp-wave-function-collapse)); repo says "other" — check | Roof pieces, block fill, interior layouts |

### Reference engines and games (study, don't copy)
| Project | Licence | Why look |
| :--- | :--- | :--- |
| Cataclysm: Dark Days Ahead | CC BY-SA 3.0 for contributions, mixed ([Flathub](https://discourse.flathub.org/t/cataclysm-dda-request/6209), [mirror](https://gitee.com/mirrors/cataclysm)) | Huge open survival game: data-driven items, buildings ("mapgen" JSON), zombies, crafting |
| [FIFE](https://gamedev.net/forums/topic/385950/) | GPL v2 | Isometric Fallout-like engine: tile/instance layering |
| [IGE](https://www.github.com/wumix/ige) | check | HTML5 isometric scenegraph |
| Project Zomboid pipeline ([PZwiki tiles](https://pzwiki.net/wiki/New_Tiles)) | Not open source | Tile definitions + TileZed map editing pattern |

---

## 5. What to build, and who builds it
Ordered list of generators the game needs. Each lives in `tools/` (Blender, Python) or `scripts/` (Node), writes to `public/assets/<kind>/` and is registered in `src/data/*` by id.

| # | Generator | Base it on | Output |
| :- | :--- | :--- | :--- |
| 1 | `render_iso.py` | Our style sheet §0.8 | Sprite frames for any object, N directions |
| 2 | `batch.py` + `art:*` npm scripts | Section 2 | Headless runs, reports, exit codes |
| 3 | House kit / building generator | Building Tools (MIT) + our greybox script | Small-town houses, shops, garages, sheds by archetype; layered renders for cut-away |
| 4 | Lot exporter (`art:jobs`) | `townBuildings()` in `src/sim/worldgen.ts` | Job JSON so Blender builds what the game places |
| 5 | Character / zombie builder | MPFB 2 + Rigify + Mixamo | 8-direction animated sprite sheets with clothing layers |
| 6 | Vehicle renderer | Kenney/Quaternius CC0 cars or our models | 16 directions × damage states |
| 7 | Tree and plant generator | Sapling / tree-gen | Trunk + canopy sprites per species and season |
| 8 | Prop kit | Sverchok / Geometry Nodes | Fences, street lights, dumpsters, signs |
| 9 | Ground and road tiles | CC0 textures (Poly Haven, ambientCG) | Isometric diamond tiles with blob transitions |
| 10 | Atlas packer (`art:pack`) | free-tex-packer-core | PixiJS JSON atlases |

**Workflow for agents** (Antigravity, Copilot, Claude): take one row at a time, write the script in `tools/`, add an npm command, run it headless, put one sample render in `docs/screens/`, and update this table, `BUILDING_PIPELINE_PLAN.md`, `HD_GRAPHICS_PLAN.md` and World Bible §69.

---

## 5b. Unreal Engine from the command line (headless)
Installed: **Unreal Engine 5.8.3** at `C:\Program Files\Epic Games\UE_5.8` (5.7 was used until 2026-10-20). Tested 2026-10-20 on 5.7.3 and again on 5.8.3: headless Python scripts run, the arena pipeline works unchanged; 5.8 adds Epic's experimental Unreal MCP server plugin (`Engine/Plugins/Experimental/ModelContextProtocol`) (about 40 s, exit code 0).

**Run a Python script with no editor window:**
```bash
"C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" MyProject.uproject -run=pythonscript -script="path/or/code.py" -unattended -nullrhi -nosplash -nosound -stdout -FullStdOutLogOutput
```
| Flag | What it does |
| :--- | :--- |
| `UnrealEditor-Cmd.exe` | The command-line build of the editor (no UI) |
| `MyProject.uproject` (first argument) | The project to load; can be left out for engine-only scripts, as in the test |
| `-run=pythonscript -script=…` | Run a Python file or a line of Python ([Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python)); some doc pages show `-ExecutePythonScript="file.py"` instead — check per version |
| `-unattended` | No dialogs or prompts (needed for agents/CI) |
| `-nullrhi` | No GPU rendering — fastest for data work (importing, building levels); **leave it out when rendering images** (some tasks such as lightmap builds fail with it, per [this thread](https://forums.unrealengine.com/t/building-lightmaps-on-a-headless-box/433035)) |
| `-stdout -FullStdOutLogOutput` | Send the log to the console so scripts and agents can read it ([forum tip](https://forums.unrealengine.com/t/interacting-with-ue4-editor-by-commandline-best-practice/437852)) |

The project must have the **Python Editor Script Plugin** enabled (it ships with the engine).

**Render images or video headless:** Movie Render Queue can be started from the command line, by giving a Level Sequence plus a `-MoviePipelineConfig` preset, or a whole queue, or a custom Python executor ([Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-command-line-rendering-with-move-render-queue-in-unreal-engine)). This needs the GPU (no `-nullrhi`).

**Control a running editor:** enable the **Remote Control API** plugin (ships with the engine); the editor then listens on a local HTTP port and scripts or agents can call functions, set properties and run Python in it. Simpler and **tested 2026-10-20**: the Python plugin's own **remote execution** (`bRemoteExecution=True`), used by `npm run ue:remote` (`tools/unreal/pipeline/ue_remote.py`): runs Python or a whole script inside the open editor, picks the editor by project name.

**The tested arena pipeline (2026-10-20)** lives in `tools/unreal/pipeline/` with `npm run ue:*` commands: headless level build + verify, screenshots with `-RenderOffScreen`, size-checked imports, live control, a PC check; plus packaging with `RunUAT BuildCookRun` and fast headless game runs with `-benchmark -fps=30`. Commands, flags and every trap found (import crash, spawn crash, Rotator order, centimetre scale, no Y flip): [UE_PIPELINE.md](UE_PIPELINE.md) §4-§6.

**Agents (Claude, Antigravity, Copilot):** any agent that can run terminal commands can drive Unreal this way — Claude ran the test above. For richer control there is a third-party MCP server, [`unreal-engine-mcp`](https://pypi.org/project/unreal-engine-mcp/) (uses the two built-in plugins above; create projects, import assets, build levels and Blueprints, Play In Editor, package; config snippets for Claude Code, VS Code Copilot and others). Its source and licence were not checked — review before installing.

**How Unreal fits this project:** the main game is TypeScript + PixiJS in the browser. Since 2026-10-20 Unreal is the engine for the **arena demo** (phase Z34, [UE_ARENA_PLAN.md](UE_ARENA_PLAN.md)); for the main game its roles are:
- a higher-quality **sprite renderer** (Lumen lighting, real materials) for the isometric art, rendering the same `.glb` pieces Blender makes, with the camera from `STYLE_BIBLE.md` §0.8;
- a place to **prototype** a building, destruction or vehicle in 3D before making sprites;
- a full engine switch would be a separate, much larger decision (rewrite of renderer, UI and tooling).
- Check Epic's licence terms for how Unreal-made assets may be used in a non-Unreal game before shipping them.

Not verified here: the exact Remote Control port. `RunUAT BuildCookRun` syntax: [UE_PIPELINE.md](UE_PIPELINE.md) §4.6 (run it in step U14).

## 5c. Free and open-source content for Unreal (searched 2026-10-20)
Licences come from news and docs summaries; read each licence before shipping. Key rule: several Epic freebies may only be used **inside Unreal-made products**. Since our game is not built in Unreal, prefer CC0 / open sources, or check that a licence allows use in other engines.

### Characters and animation
| Source | Cost / licence | Notes |
| :--- | :--- | :--- |
| [MetaHuman](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/) (Creator built into UE 5.6+) | Free under the Unreal EULA under $1M/year revenue; since June 2025 usable in other engines too ([details](https://digitalproduction.com/2025/06/05/metahumans-graduate-ready-for-unity-godot-and-the-fab-cash-register/), [summary](https://renderu.com/en/news/post/26778)) | Realistic humans, clothing, Mesh to MetaHuman, MetaHuman Animator. Very detailed — good for portraits or close-up sprites; heavy for small iso sprites. No AI training use. |
| [Lyra Starter Game](https://dev.epicgames.com/documentation/en-us/unreal-engine/lyra-sample-game-in-unreal-engine) (Fab) | Free sample | UE5 mannequins Manny and Quinn, optimised locomotion animations, weapon system ([animation notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/animation-in-lyra-sample-game-in-unreal-engine), [retargeting](https://www.unrealengine.com/tech-blog/adapting-lyra-animation-to-your-ue5-game)). **Reference only** since 2026-10-20: the arena uses the 5.7/5.8 templates instead (UE_PIPELINE.md §1.2); Lyra content must not go into the public repo. |
| Unreal 5.7/5.8 templates: First Person **Arena Shooter**, Top Down **Twin Stick** variants | Free with the engine (template code may be shared publicly) | Bots with perception and StateTree, guns, pickups, team score; top-down camera and mouse/stick aim. **The base of the Unreal arena.** |
| [Game Animation Sample](https://www.unrealengine.com/tech-blog/explore-the-updates-to-the-game-animation-sample-project-in-ue-5-7) (5.7, Fab) | Free sample | Motion-matched locomotion for the mannequins; migrate animations if the template's are not enough. |
| [Quaternius Universal Animation Library 1 + 2](https://quaternius.itch.io/universal-animation-library-2) | **CC0** | 8-way locomotion, deaths, **zombie locomotion**; GLB for Unreal 5+, retarget with the IK Retargeter. |
| [MPFB 2 / MakeHuman](https://www.cgchannel.com/2025/03/check-out-open-source-blender-character-generation-plugin-mpfb-2) (Blender) | GPL tool, CC0 assets | Open alternative; export to Unreal as FBX/glTF (see §3). |
| Mixamo | Free, Adobe terms | Animations for the mannequins ([roundup](https://www.motionforgepictures.com/unreal-engine-free-3rd-person-character-animations/)). |

### Cities, towns, buildings, maps
| Source | Cost / licence | Notes |
| :--- | :--- | :--- |
| [City Sample (PCG, UE 5.8)](https://www.unrealengine.com/learning/city-sample-gets-a-major-update-with-pcg-and-unreal-mcp-workflows) | Free (Fab / Launcher) | A procedural city built only with Unreal's PCG tools: uneven terrain, curved roads, sidewalks, vegetation; **shape-grammar assets for buildings and roads**; designed to work with the Unreal MCP server ([PCG docs](https://dev.epicgames.com/documentation/unreal-engine/city-sample-pcg-for-unreal-engine)). The best example of rule-based buildings and roads to learn from. |
| [City Sample (2022 original)](https://www.cgchannel.com/2022/04/download-epic-games-free-city-sample-assets-for-ue5) | Free; Unreal-based products only | 24 modular building kits + 44 sample buildings, vehicles. |
| Unreal **PCG framework** (built in) | Free with the engine | Procedural scattering, splines, shape grammar — towns, roads, forests, props. |
| [DungeonGenerator (shun126)](https://github.com/shun126/DungeonGenerator) | GPL-3 (also on Fab under the Epic licence) | Editor and runtime tiled generation, mini-map, door/key missions — ideas for bunkers and interiors. |
| [Procedural Dungeon Generator](https://ayansingal.itch.io/procedural-dungeon-generator/purchase) | Pay what you want | Simple dungeon/maze generator. |
| [VisusCity Vol2](https://visusarts.itch.io/visuscity-vol2) | Check price | Third-party PCG city / road / neighbourhood kit for UE 5.5. |

### Floors, materials, textures, terrain
| Source | Cost / licence | Notes |
| :--- | :--- | :--- |
| Poly Haven, ambientCG | CC0 | Floors, walls, ground, asphalt — safe for any engine (see §3, §5). |
| [Quixel Megascans](https://dev.epicgames.com/documentation/en-us/unreal-engine/quixel-bridge-plugin-for-unreal-engine) | **Paid since 2025, also inside Unreal** ("Epic Games will no longer offer the Quixel Megascans library for free starting next year", [80.lv](https://80.lv/articles/megascans-no-longer-free-after-2024/), [CG Channel](https://www.cgchannel.com/2024/10/epic-games-has-made-megascans-free-to-all-but-only-until-the-end-of-2024/)); assets claimed free in 2024 are yours to keep; a few samples stay free (e.g. [Derelict Corridor](https://www.cgchannel.com/2025/05/get-260-free-3d-scanned-assets-for-creating-a-derelict-interior-in-ue5/), Fab Standard licence). Corrected 2026-10-20; earlier text said Unreal use was still free. | Photoscanned surfaces and props. Use CC0 (Poly Haven, ambientCG) first. |
| [ArmorPaint](https://www.blendernation.com/2019/11/01/check-out-promising-open-source-3d-painting-tool-armorpaint/) | Source zlib-licensed (free to compile), paid binaries | Substance Painter-style 3D texture painting. |
| [Material Maker](https://alternativeto.net/software/substance-painter) | Free, open source (Godot-based) | Procedural material/texture authoring. |
| [Gaea](https://gamefromscratch.com/?p=10645) | Free tier (commercial use, check current terms) + paid | Terrain heightmaps; [Gaea2Unreal plugin](https://quadspinner.com/Gaea/Unreal) open-sourced under a modified MIT licence. |

### Items and props
| Source | Cost / licence | Notes |
| :--- | :--- | :--- |
| [Fab free giveaways](https://www.cgchannel.com/2024/11/epic-games-to-share-new-free-assets-on-fab-every-two-weeks) | Free if claimed in the window; keep forever | Twice a month (first and third Tuesday), each free for about two weeks ([schedule change](https://gamefromscratch.com/massive-changes-to-the-unreal-engine-fab-marketplace-giveaways/), [ongoing](https://unrealsource.com/d/free-asset-drops-continue-on-fab/)). Some are Personal-licence only — check each. |
| [Free Epic content](https://dev.epicgames.com/documentation/en-us/unreal-engine/free-epic-games-content-for-unreal-engine) | Free; mostly Unreal-only | Samples and permanently free packs. |
| Kenney, Quaternius, Poly Pizza | CC0 (check each) | Low-poly items and props, any engine (see `BUILDING_PIPELINE_PLAN.md`). |
| [Free UE plugins on itch.io](https://itch.io/tools/free/tag-plugin/tag-unreal-engine), [GitHub ue5](https://github.com/topics/ue5), [UE code plugins](https://github.com/topics/unreal-engine-code-plugin) | Varies | Inventory systems, tools; check licences. |

### Recommended for World 4 Bro's
1. **Shape and layout ideas**: study the **City Sample PCG** shape grammar for buildings and roads; mirror the approach in our snap kit and road graph.
2. **Characters**: the UE5 mannequins from the templates + CC0 animations (Quaternius Universal Animation Library, including zombie locomotion) for survivors and zombies; Mixamo only in private repos; MPFB 2 for CC0 bodies; MetaHuman only where its licence and detail level fit.
3. **Surfaces**: CC0 textures (Poly Haven, ambientCG) first; Megascans are paid since 2025.
4. **Terrain**: Gaea heightmaps if we ever need hand-shaped terrain.
5. **Claim Fab freebies** every two weeks; log each one with its licence in `CREDITS.md`.

**Arena in Unreal (Z34, decided 2026-10-20, revised the same day)**: built on Unreal's **First Person C++ template, Arena Shooter variant**, plus the **Twin Stick** camera/aim code; no own engine and not Lyra (public repo and licences, disk, Git LFS quota, agent-friendly C++; [UE_PIPELINE.md](UE_PIPELINE.md) §1). Step-by-step: [UE_ARENA_PLAN.md](UE_ARENA_PLAN.md). City Sample is Unreal-only and very large: reference only for the demo. Fab, Mixamo and Megascans files never go into the public repo.

## 6. Tools already in this repo
- `tools/viewer/index.html` — orbit viewer for any `.glb` from Blender (three.js from unpkg, MIT): orbit, pan, zoom, play/scrub animations, iso view. Open `/tools/viewer/index.html` on the dev server. Add models to its `MODELS` list.
- `tools/blender/destruction_demo.py` — headless Blender rigid-body destruction test (wall breach, roof collapse), exports animated `.glb` and isometric frames. Includes its own **Cell Fracture-style Voronoi fracture** and breakable glue constraints (no add-on needed); `--wall fracture` or `--wall bricks`. Notes in `GENERATION_IDEAS.md` §13.

- `tools/export-unreal.ts` — `npm run ue:export` (same as `arena:export`) writes `tools/unreal/arenas.json` (layouts with the web sim's exact collision shapes) and `tools/unreal/game_data.json` (guns, bot levels, match rules), and copies both into the Unreal project when `W4B_UE_PROJECT` is set.
- `tools/unreal/pipeline/` — the tested Unreal scripts (`npm run ue:doctor`, `ue:arena`, `ue:shots`, `ue:import`, `ue:remote`, `ue:open`); README in the folder, reference in [UE_PIPELINE.md](UE_PIPELINE.md). The older `tools/unreal/*.py` scripts and `tools/run-unreal.mjs` are superseded.

## 7. Not verified yet
- Exact licences of Auto-Building, BuildingNodes, Buildify, iceoh, pixi-tile-utilities, gen-city, WaveFunctionCollapse, OpenRCT2, Tiled.
- `bpy` as a pip package (Blender as a Python module).
- Open-source Blender road generators and the A.N.T. Landscape add-on (not found in this search).
