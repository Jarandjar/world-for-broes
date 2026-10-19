# 🐛 World4Bros Dev Log & Bug Tracking Engine

A built-in diagnostic and logging system designed to catch runtime errors, record live combat traces, track simulation performance, and export one-click bug reports.

---

## 🚀 Quick Access

- **Top Menu Bar**: Click **`DevLog [F2]`**
- **Keyboard Hotkeys**: Press **`F2`** or **`~`** (Backquote)
- **Browser Developer Console**: Inspect the live global `window.devLog` instance

---

## 🔍 Features & Tabs

### 1. Real-Time Telemetry Stream
Logs all engine subsystems with clean, color-coded level badges:
- `🔴 BUG` / `❌ ERROR`: Uncaught JavaScript exceptions, promise rejections, entity access anomalies.
- `⚠️ WARN`: Out-of-bounds coords, missing textures, depleted resources.
- `⚔️ COMBAT`: Weapon fire, projectile collisions, damage falloff calculations, bot aiming decisions.
- `🌱 SIM`: Day/night clock ticks, weather shifts, animal behavior, zombie pathfinding.
- `⚡ PERF`: FPS, drawn entity counts, chunk batching calls.

### 2. Global Exception Trapping
The system hooks into:
- `window.addEventListener('error')`: Traps uncaught runtime script errors with stack traces.
- `window.addEventListener('unhandledrejection')`: Traps asynchronous promise errors.

### 3. One-Click Bug Reporting
From the Dev Log header, click:
- **`📋 Copy Bug Report (MD)`**: Generates a GitHub-ready markdown diagnostic snapshot and copies it to your clipboard.
- **`💾 Export JSON`**: Downloads a timestamped JSON file containing complete game state, player pos, active arena, entity counts, and the last 60 log entries.
- **`🧪 Test Bug`**: Simulates an error event to test logging and visual error alerts.

---

## 🛠️ Developer API (`window.devLog`)

In the browser console, you can interact directly with the logging engine:

```javascript
// Log custom message
devLog.log('system', 'Player equipped custom rifle');

// Log a warning
devLog.warn('world', 'Chunk generated outside boundary', { cx: 40, cy: -12 });

// Log a bug with stack trace
devLog.bug('combat', 'Bullet trajectory NaN detected', { bulletId: 104 });

// Generate markdown snapshot to console
console.log(devLog.formatMarkdownReport(devLog.generateBugReport()));

// Clear logs or pause stream
devLog.clear();
devLog.togglePause();
```
