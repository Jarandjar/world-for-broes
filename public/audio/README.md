# 🎵 Wild Survivors Audio Guide: Suno & Sound Effects

This directory (`public/audio/`) holds studio-quality audio assets (`.mp3`, `.wav`, or `.ogg`).

The game engine automatically loads these files on startup and plays them with **zero latency**. If any file is missing, the engine automatically falls back to procedural Web Audio synthesis.

---

## 🎼 1. Background Music (Using Suno.com)
Suno is a state-of-the-art **AI music generator**. It is ideal for crafting the atmospheric wilderness soundtrack.

### Recommended Prompts for Suno:
1. **Daytime Wilderness Theme (`bgm_day.mp3`)**:
   > *Style of Music:* Don't Starve style whimsical eerie acoustic guitar, dark gothic ragtime, banjo, clarinet, upright bass, playful survival, folk instrumental, loopable
   * Instructions: Go to [Suno.com](https://suno.com), toggle **Custom Mode** -> **Instrumental**, paste the style above, download the generated track, and save it as:
     `public/audio/bgm_day.mp3`

2. **Nighttime Darkness & Dread Theme (`bgm_night.mp3`)**:
   > *Style of Music:* Dark ambient drone, creeping cello, tense horror survival, eerie wind chime whispers, minimal suspenseful bass, gothic mystery, instrumental
   * Save as: `public/audio/bgm_night.mp3`

---

## 🔊 2. Sound Effects (SFX)
For micro game sound effects (0.1s – 1s duration), **ElevenLabs Sound Effects** or free game libraries (Kenney / Freesound) work best:

### ElevenLabs Sound Generation Prompts ([elevenlabs.io](https://elevenlabs.io)):
| File Name | ElevenLabs Prompt | Description |
| :--- | :--- | :--- |
| `chop.mp3` | "Heavy iron axe hitting a pine tree trunk, splintering wood impact" | Tree chop impact |
| `treefall.mp3` | "Large pine tree creaking, snapping, crashing down onto forest floor" | Tree felled collapse |
| `pick.mp3` | "Quick cartoon pop harvest gathering berries foliage rustle" | Picking items/resources |
| `footstep.mp3` | "Soft sneaker footstep on autumn dirt and dry forest grass" | Running footstep |
| `eat.mp3` | "Crisp cartoon crunchy bite chewing berries fruit" | Eating food |
| `craft.mp3` | "Magical triumphant woodworking tools crafting chime arpeggio" | Crafting an item |
| `charlie_warning.mp3` | "Terrifying shadowy whisper in the dark, dissonant horror drone" | Night danger warning |
| `charlie_slash.mp3` | "Violent razor sharp monster claw slashing through air with flesh strike"| Night monster attack |
| `honk.mp3` | "Funny loud goose honk bicycle air horn comic sound" | Emote honk |
| `chicken.mp3` | "High pitched rubber chicken squeak toy comical" | Rubber chicken gadget |
| `pan.mp3` | "Heavy cast iron frying pan metallic resonant dong gong" | Frying pan smack |
| `bubble.mp3` | "Soft soap bubble pop comic interface" | Speech bubble chat pop |

---

## ⚡ Why We Don't Call AI APIs During Live Gameplay
1. **Zero Latency**: An API call takes 2–4 seconds to generate audio over the internet. When a player chops a tree or walks, they need the sound in **10 milliseconds**.
2. **Offline & Cost Efficient**: Pre-loading `.mp3` files in `public/audio/` costs $0 in API bandwidth for players, works offline, and loads instantly on both PC and mobile phones.
