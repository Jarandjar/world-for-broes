# Claude AI Prompt & Base Visual Structure Specification
*World 4 Bro's / Lantern Hollow — Visual Shell Architecture*

> **How to Use This Document**:
> 1. Copy the **Master Prompt for Claude AI** in Section 1 below.
> 2. Paste it into Claude AI.
> 3. Claude will generate the modular visual structure (HTML layout + CSS design system).
> 4. Paste Claude's HTML into `visual-shell.html` (or `index.html`) and CSS into `src/ui/visual-shell/visual-shell.css` (or `src/style.css`).
> 5. Run `npm run dev` and open `http://localhost:5175/visual-shell.html` to instantly inspect and test it with live simulated game states!

---

## 1. Master Prompt to Copy-Paste into Claude AI

```markdown
You are an expert game UI/UX director and senior creative frontend architect.
We are building the complete **base visual structure** for **"World 4 Bro's"** (also known as *Lantern Hollow*), a 2.5D pop-up cardboard diorama & rubber-hose slapstick survival game (combining the atmospheric survival tension of *Don't Starve* with the playful, tactile papercraft aesthetic of *Paper Mario* and retro comic books).

### 🎯 Objective
Generate the complete HTML & CSS base visual structure for our game interface. The structure must feel tactile, premium, juicy, and responsive across both:
1. **Desktop Viewport (16:9 / 21:9)**: Immersive, clean layout with keyboard hints, gothic serif typography, tactile paper textures, and minimal obstruction.
2. **Mobile Landscape Viewport (Artboard 1 & 2: 844x390 safe-area)**: Ergonomic touch-first design with left-thumb floating joystick zone, right-thumb contextual action clusters, swipeable hotbar, and top status/clock/condition cluster.

### 🎨 Visual Aesthetic & Style Bible
- **Pop-Up Paper & Rubber-Hose Hybrid**: Chunky cardstock borders, scored fold lines, subtle dropshadows, torn paper edges, comic onomatopoeia badges ("BONK!", "POOF!"), and warm candlelight glows.
- **Palette Tokens**:
  - Ink / Night Void: `#0A0C10`, `#11161D`, `#191F26`
  - Parchment / Cream: `#E9E1CC`, `#FAF6EB`, `#C5BAA0`
  - Survival Gold / Fire Glow: `#F08A32`, `#F9C45A`, `#E65100`
  - Health Crimson: `#D9383A`, `#8A1C1E`
  - Hunger Honey / Meat: `#D9822B`, `#8B4513`
  - Sanity Phosphor Violet: `#9C27B0`, `#673AB7`
  - Infection Acid Green: `#70E000`, `#388E3C`
- **Typography**:
  - Headers & Numbers: `'Cinzel', 'Alegreya SC', serif`
  - Narrative & Labels: `'Crimson Pro', 'MedievalSharp', Georgia, serif`
  - Monospace / Keys: `ui-monospace, monospace`
- **Line Art Rule**: NO pure digital `#000000`. Use deep roasted chestnut (`#2E1810`) or midnight indigo (`#171B2B`) with variable width outlines (3px–4px hero, 1.5px inner creases).

---

### 🧩 Core Component Layers Required

#### 1. Atmospheric Overlays (Pointer-events: none)
- Fullscreen vignette overlay (dusk / night gradient)
- Lethal darkness terror pulse effect
- Sanity distortion backdrop filter
- Red screen-edge damage vignette flash

#### 2. Top-Left Game Status & Mode Banner
- Stylized logo badge: "WORLD 4 BRO'S" with edition tag ("2.5D POP-UP DIORAMA")
- Game mode pill (`classic` vs `zombie outbreak`)
- Quick utility buttons: Mobile QR Connect (`📱`), Dev Spawner (`🛠️`), Animation Lab (`🎬`), Settings (`⚙️`)

#### 3. Top-Right Clock Dial & Day/Phase Tracker
- Circular Don't Starve-style pie dial showing Day/Dusk/Night proportions (conic gradient)
- Rotating brass clock hand
- Day counter badge ("DAY 1") + current phase tag ("DUSK")

#### 4. Top-Right Vitals Triad & Conditions Cluster
- Three circular or pill meters:
  1. **Hunger Meter**: 🍗 icon, honey amber fill, numerical readout (0-100)
  2. **Health Meter**: ❤️ icon, crimson fill with subtle heartbeat animation (0-100)
  3. **Sanity Meter**: 🧠 icon, eerie violet fill with wobble (0-100)
- Condition Pills (under vitals):
  - 🔊 Noise Level ripple gauge (bars 0 to 5)
  - ⚖️ Weight / Encumbrance (kg / max)
  - ☣️ Infection rate %

#### 5. Center Dynamic Contextual Action Banner
- Floats above target objects
- Shows physical button prompt (`[SPACE]`, `[E]`, `[R]`, `[F]`)
- Dynamic action label (e.g., "Chop Pine Tree", "Search Metal Toolbox", "Hop in Car", "Feast on Cooked Morsel")
- Micro progress bar for hold-to-interact actions

#### 6. Bottom Inventory Hotbar
- 8 tactile papercraft slots with subtle stitched border
- Slot numbers `1` to `8` in small corner badges
- Selected slot highlight with bounce effect
- Item icon, stack quantity badge, and mini durability degradation bar
- Quick-eat secondary slot prompt (`[F] Quick Eat`)

#### 7. Mobile Virtual Touch Controls Layer (`#touch-controls-layer`)
- **Left 40% Touchscreen**:
  - Virtual analog thumbstick (base circle + draggable knob)
  - Sprint Ring threshold line (triggers sprint when pushed past 46px)
  - Sneak button toggle (turtle icon 🐢)
- **Right Thumb Cluster**:
  - Big primary action button (🪓 ACTION / CHOP / ATTACK)
  - Utility row: Camera Rotate Left (↺), Eat (🍖), Chat (💬), Camera Rotate Right (↻)
  - Crafting drawer pull-tab button (`📜 CRAFT`)

#### 8. Driving Dashboard Cockpit Layer (`#driving-controls-layer`)
- Visible when the player hops into a vehicle (e.g., Rusty Hatchback)
- **Top Center Dashboard Cluster**:
  - Digital km/h speedometer + analog gauge arc
  - Fuel tank gauge (E to F needle / percentage)
  - Engine durability bar (green/yellow/red)
- **Driving Touch Controls**:
  - Responsive Gas Pedal & Brake/Reverse Pedal
  - Left / Right steering pads
  - Handbrake drift button (🛞 DRIFT)
  - Car Horn button (📢 HONK) & Dismount button (🚪 EXIT)

#### 9. Slide-out Drawers & Bottom Sheets
- **Crafting Sidebar / Accordion**: Categories (🛠️ Tools, 🔥 Light, 🍖 Survival, 🚗 Vehicle Parts), recipe cards with cost badges and "CRAFT" button.
- **Vehicle Part-Repair Sheet** (Bottom sheet modal):
  - Vehicle status header + fuel siphoning controls
  - 5 Part Cards (`Engine Block`, `Battery`, `Spark Plugs`, `Radiator`, `Tires`) with condition badges [GOOD / WORN / CRITICAL] and hold-to-repair radial rings.
  - Crank starter timing minigame needle.

#### 10. Comic Pops & Feedback Layer
- Anchored screen-space comic sound effect bubbles (`"BONK!"`, `"POOF!"`, `"CRANK!"`, `"JACKPOT! 🚨"`).

---

### 📦 Deliverable Format
Please provide:
1. **Clean, Semantic HTML Structure**: Fully nested under a root container `#app-visual-root` or `#hud-container`, using the exact ID names listed above so it bridges directly into our TypeScript game engine.
2. **Pure Vanilla CSS**: Modular, utilizing CSS Custom Properties (`:root`), flexbox, grid, glassmorphism, responsive media queries, safe-area insets (`env(safe-area-inset-*)`), and subtle animations. Do NOT use Tailwind or heavy frameworks.
3. Make it look state-of-the-art and feel immediately like an award-winning indie game diorama!
```

---

## 2. Technical ID & Architecture Contract

When integrating Claude AI's generated HTML and CSS into this repository, use the following IDs so the TypeScript systems automatically bind without errors:

| Component System | Required Element IDs | Bound TypeScript Class |
| :--- | :--- | :--- |
| **Vitals Cluster** | `#fill-health`, `#val-health`<br>`#fill-hunger`, `#val-hunger`<br>`#fill-sanity`, `#val-sanity` | `SurvivalStats.ts` |
| **DST Clock Dial** | `#dst-clock`, `#clock-pie`, `#clock-hand`, `#day-counter`, `#time-phase` | `DayNightSystem.ts` |
| **Inventory Hotbar** | `#inventory-bar` (contains 8 `.inv-slot` elements) | `InventorySystem.ts` |
| **Crafting Drawer** | `#crafting-sidebar`, `.crafting-categories`, `#recipes-list` | `CraftingSystem.ts` |
| **Context Action Banner**| `#action-prompt`, `#action-text`, `.key-hint`, `#target-progress-bar`, `#target-progress-fill` | `main.ts` / `OnFootLayout.ts` |
| **Mobile On-Foot** | `#touch-controls-layer`<br>`#touch-joystick-zone`, `#touch-joystick-base`, `#touch-joystick-knob`<br>`#btn-touch-action`, `#touch-action-title`<br>`#btn-touch-cam-left`, `#btn-touch-cam-right`, `#btn-touch-eat`, `#btn-touch-chat` | `OnFootLayout.ts` |
| **Mobile Driving** | `#driving-controls-layer`<br>`#driving-dash-cluster`, `#dash-car-name`<br>`#dash-speed-val`, `#dash-fuel-val`, `#dash-durability-fill` | `DrivingLayout.ts` |
| **Car Repair Sheet** | `#vehicle-repair-sheet`, `#btn-sheet-close`, `#btn-siphon-fuel`, `#btn-pour-fuel`, `#btn-crank-engine` | `VehicleRepairSheet.ts` |
| **Comic Pops Layer** | `#comic-pops-layer` | `ComicPops.ts` |
| **Status / Modals** | `#status-banner`, `#btn-phone-connect`, `#btn-settings`, `#settings-modal`, `#death-modal` | `main.ts` |

---

## 3. How to Test Claude's Visual Structure in this Project

1. Open `visual-shell.html` in this repo.
2. Paste Claude's HTML into the marked injection section:
   ```html
   <!-- [CLAUDE_VISUAL_STRUCTURE_INJECTION_POINT] -->
   ```
3. Paste Claude's CSS into `src/ui/visual-shell/visual-shell.css`.
4. Run:
   ```powershell
   npm run dev
   ```
5. Navigate to `http://localhost:5175/visual-shell.html`.
6. Use the top simulation toolbar to test:
   - 📱 **Mobile Frame Toggle** (844×390 iPhone preview) vs 🖥️ **Full Desktop**
   - 🚶 **On-Foot Mode** vs 🚗 **Driving Cockpit** vs 🔧 **Repair Sheet**
   - ☀️ **Day** vs 🌅 **Dusk** vs 🌙 **Night** vs 🩸 **Horde Night Siren**
   - 🎚️ **Health / Hunger / Sanity / Speed sliders** to watch gauges update live!
