import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFile, spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 4040;
const BLENDER_EXE = "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe";
const SCRIPT_DIR = path.join(__dirname, 'blender');
const ENGINE_SCRIPT = path.join(SCRIPT_DIR, 'city_foundry_engine.py');
const TOOLKIT_SCRIPT = path.join(__dirname, 'vendor', 'fetch-toolkit.ps1');
const TOOLKIT_REPORT = path.join(__dirname, 'vendor', 'fetch-report.json');

// HTML Dashboard UI
function getDashboardHtml() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CityFoundry — 3D Asset Studio & Tactical Level Builder</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090c13;
      --card: rgba(18, 24, 38, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --border-active: rgba(0, 240, 255, 0.4);
      --cyan: #00f0ff;
      --cyan-glow: rgba(0, 240, 255, 0.2);
      --purple: #a855f7;
      --amber: #f59e0b;
      --green: #10b981;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Space Grotesk', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
    }

    /* Top Bar */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 28px;
      border-bottom: 1px solid var(--border);
      background: rgba(11, 15, 25, 0.75);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo {
      width: 34px;
      height: 34px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
      color: #000;
      box-shadow: 0 0 16px var(--cyan-glow);
    }

    .brand-title {
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .brand-badge {
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      padding: 3px 8px;
      border-radius: 12px;
      background: rgba(0, 240, 255, 0.12);
      color: var(--cyan);
      border: 1px solid rgba(0, 240, 255, 0.3);
    }

    /* Mode Toggle */
    .mode-switch {
      display: flex;
      background: rgba(0, 0, 0, 0.4);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border);
      gap: 4px;
    }

    .mode-btn {
      padding: 8px 16px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border-radius: 6px;
      transition: all 0.2s;
    }

    .mode-btn.active {
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    .mode-btn.active.tactical {
      background: rgba(245, 158, 11, 0.18);
      color: var(--amber);
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* Main Workspace */
    .workspace {
      display: grid;
      grid-template-columns: 380px 1fr;
      flex: 1;
      height: calc(100vh - 67px);
    }

    /* Sidebar Controls */
    .sidebar {
      border-right: 1px solid var(--border);
      padding: 24px;
      overflow-y: auto;
      background: rgba(13, 17, 28, 0.5);
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .section-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
      margin-bottom: 10px;
      font-family: 'JetBrains Mono', monospace;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    label {
      font-size: 13px;
      font-weight: 500;
      color: #cbd5e1;
    }

    .seed-input-row {
      display: flex;
      gap: 8px;
    }

    input[type="number"], select {
      flex: 1;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border);
      color: #fff;
      padding: 10px 14px;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      outline: none;
      transition: border 0.2s;
    }

    input[type="number"]:focus, select:focus {
      border-color: var(--cyan);
    }

    .btn-dice {
      padding: 0 14px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
      color: #fff;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      transition: background 0.2s;
    }
    .btn-dice:hover { background: rgba(255, 255, 255, 0.12); }

    .toggle-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .toggle-switch {
      position: relative;
      width: 44px;
      height: 24px;
    }

    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(255, 255, 255, 0.1);
      transition: 0.2s;
      border-radius: 24px;
    }
    .slider:before {
      position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
      background: white;
      transition: 0.2s;
      border-radius: 50%;
    }
    input:checked + .slider { background: var(--cyan); }
    input:checked + .slider:before { transform: translateX(20px); }

    .metric-row {
      margin-top: 8px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .metric-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: #cbd5e1;
      margin-bottom: 4px;
      font-family: 'JetBrains Mono', monospace;
    }

    .metric-value {
      color: var(--cyan);
      font-weight: 600;
    }

    input[type="range"] {
      width: 100%;
      accent-color: #00f0ff;
      cursor: pointer;
    }

    /* Action Buttons */
    .btn-generate {
      background: linear-gradient(135deg, #00f0ff, #0099ff);
      color: #000;
      border: none;
      padding: 14px;
      border-radius: 10px;
      font-weight: 700;
      font-size: 15px;
      font-family: inherit;
      cursor: pointer;
      box-shadow: 0 4px 20px var(--cyan-glow);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-generate:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 26px rgba(0, 240, 255, 0.35);
    }
    .btn-generate:active { transform: translateY(1px); }

    .btn-blender {
      background: rgba(234, 115, 23, 0.15);
      border: 1px solid rgba(234, 115, 23, 0.5);
      color: #ea7317;
      padding: 12px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 14px;
      font-family: inherit;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .btn-blender:hover {
      background: rgba(234, 115, 23, 0.28);
    }

    /* Right Preview Stage */
    .stage {
      display: flex;
      flex-direction: column;
      padding: 24px;
      gap: 16px;
      overflow-y: auto;
    }

    .preview-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      position: relative;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    .preview-header {
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(0, 0, 0, 0.25);
    }

    .preview-title {
      font-size: 14px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .preview-meta {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--cyan);
    }

    .preview-canvas-wrap {
      width: 100%;
      height: 480px;
      background: #141721;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }

    .preview-img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      transition: opacity 0.3s;
    }

    .loading-overlay {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(9, 12, 19, 0.85);
      backdrop-filter: blur(6px);
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      z-index: 10;
    }

    .spinner {
      width: 44px;
      height: 44px;
      border: 3px solid rgba(0, 240, 255, 0.15);
      border-top-color: var(--cyan);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Download Export Toolbar */
    .export-bar {
      display: flex;
      gap: 12px;
      padding: 16px 20px;
      border-top: 1px solid var(--border);
      background: rgba(0, 0, 0, 0.2);
    }

    .btn-export {
      flex: 1;
      padding: 10px 14px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      text-decoration: none;
      color: #fff;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
      transition: all 0.2s;
    }
    .btn-export:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.2);
    }

    /* Tactical Data Drawer */
    .tactical-drawer {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .tactical-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }

    .stat-box {
      background: rgba(0, 0, 0, 0.3);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stat-val {
      font-size: 22px;
      font-weight: 700;
      color: var(--amber);
      font-family: 'JetBrains Mono', monospace;
    }

    .stat-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }

    .code-view {
      background: #000;
      border-radius: 8px;
      padding: 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #38bdf8;
      max-height: 180px;
      overflow-y: auto;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .toolkit-actions {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-top: 8px;
    }

    .btn-toolkit {
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid var(--border);
      color: #dbeafe;
      padding: 10px 12px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      text-align: left;
      transition: border 0.2s, background 0.2s;
    }

    .btn-toolkit:hover {
      border-color: var(--border-active);
      background: rgba(0, 240, 255, 0.08);
    }

    .toolkit-links {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }

    .toolkit-link {
      font-size: 12px;
      color: #7dd3fc;
      text-decoration: none;
      border: 1px solid rgba(125, 211, 252, 0.18);
      border-radius: 8px;
      padding: 8px;
      background: rgba(2, 6, 23, 0.6);
      display: block;
    }

    .toolkit-link:hover {
      border-color: rgba(0, 240, 255, 0.45);
      color: #a5f3fc;
    }

    .toolkit-status {
      margin-top: 10px;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
      color: var(--text-muted);
      min-height: 42px;
      white-space: pre-wrap;
      font-family: 'JetBrains Mono', monospace;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-logo">CF</div>
      <div>
        <div class="brand-title">CityFoundry</div>
      </div>
      <div class="brand-badge">Blender 5.2 CLI</div>
    </div>

    <div class="mode-switch">
      <button class="mode-btn active" id="btnModeStudio" onclick="setMode('studio')">🎨 3D Asset Studio</button>
      <button class="mode-btn" id="btnModeTactical" onclick="setMode('tactical')">🎯 Tactical Level Builder</button>
    </div>
  </header>

  <div class="workspace">
    <!-- Sidebar Controls -->
    <div class="sidebar">
      <div>
        <div class="section-title">Block Layout & Seed</div>
        <div class="form-group" style="margin-bottom: 12px;">
          <label>World Seed</label>
          <div class="seed-input-row">
            <input type="number" id="inpSeed" value="888">
            <button class="btn-dice" onclick="randomizeSeed()" title="Randomize Seed">🎲</button>
          </div>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label>Block Layout</label>
          <select id="selLayout">
            <option value="compact">Compact Block (2 × 2 Buildings)</option>
            <option value="standard" selected>Standard Avenue (3 × 2 Buildings)</option>
            <option value="avenue">Mega District (4 × 2 Buildings)</option>
          </select>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label>District / Biome Theme</label>
          <select id="selDistrict">
            <option value="mixed" selected>🌆 Mixed Urban Center</option>
            <option value="downtown">🏙️ Downtown Metropolis (Towers)</option>
            <option value="suburban">🏡 Suburban Residential (Houses & Garages)</option>
            <option value="small_town">🏪 Small Town Main Street (Shops & Diner)</option>
            <option value="industrial">🏭 Industrial Zone (Warehouses & Silos)</option>
            <option value="survivor_slum">🛡️ Survivor Stronghold (Barricades & Watchtowers)</option>
            <option value="historic_pagoda">⛩️ Historic Old Town (Pagoda Temples)</option>
          </select>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label>Damage & Decay State</label>
          <select id="selDamage">
            <option value="pristine" selected>✨ Pristine (Clean Pre-Apocalypse)</option>
            <option value="abandoned">🌿 Abandoned & Overgrown</option>
            <option value="looted">💥 Looted & Smashed</option>
            <option value="barricaded">🚧 Fortified & Barricaded</option>
            <option value="scorched">🔥 Scorched & Ruined</option>
          </select>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label>Interior Cutaway Mode</label>
          <select id="selInterior">
            <option value="solid" selected>🧱 Solid Exterior Shell</option>
            <option value="hollow_shell">🚪 Hollow Enterable Shell</option>
            <option value="dollhouse">✂️ Dollhouse Cutaway (0.8m Half-Walls)</option>
            <option value="furnished_tactical">🪑 Furnished Rooms (Loot & Cover)</option>
          </select>
        </div>

        <div class="form-group" style="margin-bottom: 12px;">
          <label>Lighting & Atmosphere</label>
          <select id="selLighting">
            <option value="midday_clear" selected>☀️ Studio Clay / Crisp Matcap</option>
            <option value="night_neon">🌙 Midnight Cyberpunk Glow</option>
            <option value="golden_hour">🌅 Golden Hour Sunset</option>
            <option value="toxic_twilight">☣️ Toxic Radioactive Smog</option>
            <option value="overcast_bleak">☁️ Overcast Bleak Grey</option>
          </select>
        </div>

        <div class="form-group">
          <label>Skyline Profile</label>
          <select id="selHeight">
            <option value="low_rise">Low-Rise District (3 - 6 Floors)</option>
            <option value="mixed" selected>Mixed Urban (4 - 14 Floors)</option>
            <option value="towers">Skyscraper Towers (8 - 18 Floors)</option>
          </select>
        </div>
      </div>

      <div>
        <div class="section-title">Architectural Features</div>
        <div class="toggle-row">
          <label>Stepped Setback Steps</label>
          <label class="toggle-switch">
            <input type="checkbox" id="chkSetbacks" checked>
            <span class="slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>Rooftop Clutter (AC, Tanks, Antennas)</label>
          <label class="toggle-switch">
            <input type="checkbox" id="chkClutter" checked>
            <span class="slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>Sidewalk Trees & Foliage</label>
          <label class="toggle-switch">
            <input type="checkbox" id="chkTrees" checked>
            <span class="slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>Corner Street Lamps</label>
          <label class="toggle-switch">
            <input type="checkbox" id="chkLamps" checked>
            <span class="slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>Tactical Cover (Barriers & Dumpsters)</label>
          <label class="toggle-switch">
            <input type="checkbox" id="chkCover" checked>
            <span class="slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>Test Dummy Character</label>
          <label class="toggle-switch">
            <input type="checkbox" id="chkDummy" checked>
            <span class="slider"></span>
          </label>
        </div>
      </div>

      <div>
        <div class="section-title">Metric Controls (Meters)</div>
        <div class="metric-row">
          <div class="metric-label"><span>Plot Width</span><span class="metric-value" id="valPlotW">11.5 m</span></div>
          <input type="range" id="rngPlotW" min="8" max="28" step="0.5" value="11.5">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Plot Depth</span><span class="metric-value" id="valPlotD">13.0 m</span></div>
          <input type="range" id="rngPlotD" min="8" max="30" step="0.5" value="13">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Road Width</span><span class="metric-value" id="valRoadW">11.0 m</span></div>
          <input type="range" id="rngRoadW" min="6" max="24" step="0.5" value="11">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Sidewalk Padding</span><span class="metric-value" id="valSidewalk">3.2 m</span></div>
          <input type="range" id="rngSidewalk" min="0.8" max="10" step="0.1" value="3.2">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Plot Gap X</span><span class="metric-value" id="valGapX">2.4 m</span></div>
          <input type="range" id="rngGapX" min="0.4" max="10" step="0.1" value="2.4">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Plot Gap Y</span><span class="metric-value" id="valGapY">3.2 m</span></div>
          <input type="range" id="rngGapY" min="0.4" max="10" step="0.1" value="3.2">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Curb Height</span><span class="metric-value" id="valCurb">0.22 m</span></div>
          <input type="range" id="rngCurb" min="0.05" max="0.6" step="0.01" value="0.22">
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>Floor Height</span><span class="metric-value" id="valFloorH">2.8 m</span></div>
          <input type="range" id="rngFloorH" min="2.2" max="4.4" step="0.05" value="2.8">
        </div>
      </div>

      <div>
        <div class="section-title">Open-Source Toolkit & Jobs</div>
        <div class="toolkit-actions">
          <button class="btn-toolkit" onclick="runToolkitFetch('permissive')">📦 Fetch Permissive Runtime Set</button>
          <button class="btn-toolkit" onclick="runToolkitFetch('tools')">🛠️ Fetch + GPL Toolchain Set</button>
          <button class="btn-toolkit" onclick="runToolkitFetch('all')">🌐 Fetch All (incl. study repos)</button>
          <button class="btn-toolkit" onclick="loadToolkitReport()">🧾 Refresh Fetch Report</button>
        </div>
        <div class="toolkit-status" id="toolkitStatus">Toolkit ready. No fetch run yet.</div>
        <div class="toolkit-links">
          <a class="toolkit-link" href="https://polyhaven.com/textures" target="_blank" rel="noopener">Poly Haven (CC0)</a>
          <a class="toolkit-link" href="https://ambientcg.com" target="_blank" rel="noopener">ambientCG (CC0)</a>
          <a class="toolkit-link" href="https://kenney.nl/assets" target="_blank" rel="noopener">Kenney Assets</a>
          <a class="toolkit-link" href="https://quaternius.com" target="_blank" rel="noopener">Quaternius Assets</a>
          <a class="toolkit-link" href="https://poly.pizza" target="_blank" rel="noopener">Poly Pizza</a>
          <a class="toolkit-link" href="https://www.openstreetmap.org/export" target="_blank" rel="noopener">OSM Export</a>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 10px; margin-top: auto;">
        <button class="btn-generate" onclick="generateCity()">
          <span>⚡</span> Generate City Block
        </button>
        <button class="btn-blender" onclick="openInBlender()">
          <span>🚀</span> Open in Blender GUI
        </button>
      </div>
    </div>

    <!-- Main Preview Stage -->
    <div class="stage">
      <div class="preview-card">
        <div class="preview-header">
          <div class="preview-title">
            <span id="previewTag">🎨 Asset Studio View</span>
          </div>
          <div class="preview-meta" id="statusMeta">Ready</div>
        </div>

        <div class="preview-canvas-wrap">
          <img id="imgPreview" class="preview-img" src="/files/city_foundry_preview.png" alt="Isometric City Preview">
          <div class="loading-overlay" id="loadingOverlay">
            <div class="spinner"></div>
            <div style="font-weight: 600; font-size: 14px;">Blender CLI Generating City Geometry...</div>
            <div style="font-size: 12px; color: var(--text-muted);">Extruding facades, placing rooftop props & rendering isometric view</div>
          </div>
        </div>

        <div class="export-bar">
          <a class="btn-export" href="/files/city_foundry.glb" download="city_foundry.glb">
            <span>📥</span> Download .GLB (Game Ready)
          </a>
          <a class="btn-export" href="/files/city_foundry.blend" download="city_foundry.blend">
            <span>📦</span> Download .BLEND
          </a>
          <a class="btn-export" id="btnDownloadJson" href="/files/tactical_level_data.json" download="tactical_level_data.json">
            <span>🗺️</span> Export Tactical JSON
          </a>
        </div>
      </div>

      <!-- Tactical Intel Drawer -->
      <div class="tactical-drawer" id="tacticalDrawer">
        <div class="section-title" id="intelTitle">World 4 Bros Tactical Intel & Level Data</div>
        <div class="tactical-grid">
          <div class="stat-box">
            <div class="stat-val" id="statBuildings">6</div>
            <div class="stat-label">Tactical Structures</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="statCover">4</div>
            <div class="stat-label">Cover Positions</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="statSnipers">4</div>
            <div class="stat-label">Sniper Perches</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="statSpawns">4</div>
            <div class="stat-label">Spawn Waypoints</div>
          </div>
        </div>

        <div class="code-view" id="codeIntel">// Loading tactical metadata...</div>
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'studio';

    function setMode(mode) {
      currentMode = mode;
      const btnStudio = document.getElementById('btnModeStudio');
      const btnTactical = document.getElementById('btnModeTactical');
      const previewTag = document.getElementById('previewTag');
      const intelTitle = document.getElementById('intelTitle');

      if (mode === 'tactical') {
        btnTactical.classList.add('active', 'tactical');
        btnStudio.classList.remove('active');
        previewTag.textContent = '🎯 World 4 Bros Tactical Combat Map';
        intelTitle.textContent = '🎯 World 4 Bros Tactical Intel & Level Data';
      } else {
        btnStudio.classList.add('active');
        btnTactical.classList.remove('active', 'tactical');
        previewTag.textContent = '🎨 3D Asset Studio View';
        intelTitle.textContent = '📐 Geometry Breakdown & Asset Nodes';
      }
    }

    function randomizeSeed() {
      document.getElementById('inpSeed').value = Math.floor(Math.random() * 90000) + 1000;
    }

    function bindMetricSlider(id, valId, unit = 'm') {
      const input = document.getElementById(id);
      const output = document.getElementById(valId);
      if (!input || !output) return;
      const render = () => { output.textContent = Number(input.value).toFixed(2).replace(/\.00$/, '') + ' ' + unit; };
      input.addEventListener('input', render);
      render();
    }

    function initMetricSliders() {
      bindMetricSlider('rngPlotW', 'valPlotW');
      bindMetricSlider('rngPlotD', 'valPlotD');
      bindMetricSlider('rngRoadW', 'valRoadW');
      bindMetricSlider('rngSidewalk', 'valSidewalk');
      bindMetricSlider('rngGapX', 'valGapX');
      bindMetricSlider('rngGapY', 'valGapY');
      bindMetricSlider('rngCurb', 'valCurb');
      bindMetricSlider('rngFloorH', 'valFloorH');
    }

    async function loadTacticalData() {
      try {
        const res = await fetch('/files/tactical_level_data.json?t=' + Date.now());
        const data = await res.json();
        document.getElementById('statBuildings').textContent = data.buildings?.length || 0;
        document.getElementById('statCover').textContent = data.cover_points?.length || 0;
        document.getElementById('statSnipers').textContent = data.sniper_perches?.length || 0;
        document.getElementById('statSpawns').textContent = (data.spawns?.enemies?.length || 0) + 1;
        document.getElementById('codeIntel').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        console.error('Failed to load tactical data', err);
      }
    }

    async function generateCity() {
      const overlay = document.getElementById('loadingOverlay');
      const statusMeta = document.getElementById('statusMeta');
      overlay.style.display = 'flex';
      statusMeta.textContent = 'Generating...';

      const payload = {
        seed: parseInt(document.getElementById('inpSeed').value, 10),
        layout: document.getElementById('selLayout').value,
        district: document.getElementById('selDistrict').value,
        damage_state: document.getElementById('selDamage').value,
        interior_mode: document.getElementById('selInterior').value,
        lighting_mood: document.getElementById('selLighting').value,
        height_profile: document.getElementById('selHeight').value,
        enable_setbacks: document.getElementById('chkSetbacks').checked,
        roof_clutter: document.getElementById('chkClutter').checked,
        trees: document.getElementById('chkTrees').checked,
        streetlamps: document.getElementById('chkLamps').checked,
        tactical_cover: document.getElementById('chkCover').checked,
        dummy_character: document.getElementById('chkDummy').checked,
        metrics: {
          plot_width_m: parseFloat(document.getElementById('rngPlotW').value),
          plot_depth_m: parseFloat(document.getElementById('rngPlotD').value),
          road_width_m: parseFloat(document.getElementById('rngRoadW').value),
          sidewalk_pad_m: parseFloat(document.getElementById('rngSidewalk').value),
          plot_gap_x_m: parseFloat(document.getElementById('rngGapX').value),
          plot_gap_y_m: parseFloat(document.getElementById('rngGapY').value),
          curb_height_m: parseFloat(document.getElementById('rngCurb').value),
          floor_height_m: parseFloat(document.getElementById('rngFloorH').value),
        },
      };

      const startTime = performance.now();

      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        if (data.success) {
          const img = document.getElementById('imgPreview');
          img.src = '/files/city_foundry_preview.png?t=' + Date.now();
          statusMeta.textContent = \`Generated in \${elapsed}s (Blender CLI)\`;
          loadTacticalData();
        } else {
          statusMeta.textContent = 'Error during generation';
          alert('Generation failed: ' + data.error);
        }
      } catch (err) {
        statusMeta.textContent = 'Network error';
        alert('Request error: ' + err.message);
      } finally {
        overlay.style.display = 'none';
      }
    }

    async function openInBlender() {
      try {
        const res = await fetch('/api/open-blender', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          alert('🚀 Launching Blender GUI with generated scene!');
        } else {
          alert('Failed to launch Blender: ' + data.error);
        }
      } catch (err) {
        alert('Failed to send open command: ' + err.message);
      }
    }

    async function runToolkitFetch(mode) {
      const statusEl = document.getElementById('toolkitStatus');
      statusEl.textContent = 'Running toolkit fetch: ' + mode + '...';
      try {
        const res = await fetch('/api/toolkit/fetch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode })
        });
        const data = await res.json();
        if (!data.success) {
          statusEl.textContent = 'Fetch failed: ' + (data.error || 'unknown error');
          return;
        }
        statusEl.textContent = 'Fetch complete. Mode: ' + mode + '\\nFetched: ' + (data.results?.length ?? 0) + ' entries';
        await loadToolkitReport();
      } catch (err) {
        statusEl.textContent = 'Fetch request failed: ' + err.message;
      }
    }

    async function loadToolkitReport() {
      const statusEl = document.getElementById('toolkitStatus');
      try {
        const res = await fetch('/api/toolkit/report?t=' + Date.now());
        const data = await res.json();
        if (!data.success) {
          statusEl.textContent = 'No report available yet.';
          return;
        }
        const report = data.report || [];
        const cloned = report.filter((x) => x.action === 'cloned' || x.action === 'updated').length;
        const skipped = report.filter((x) => x.action === 'skipped').length;
        const failed = report.filter((x) => x.action === 'failed').length;
        statusEl.textContent = 'Report loaded\\n' +
          'cloned/updated: ' + cloned + ' | skipped: ' + skipped + ' | failed: ' + failed;
      } catch (err) {
        statusEl.textContent = 'Failed to load report: ' + err.message;
      }
    }

    // Initialize data
    initMetricSliders();
    loadTacticalData();
    loadToolkitReport();
  </script>
</body>
</html>`;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  // Serve Dashboard
  if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(getDashboardHtml());
    return;
  }

  // Serve generated static files
  if (req.method === 'GET' && url.pathname.startsWith('/files/')) {
    const filename = path.basename(url.pathname);
    const filePath = path.join(SCRIPT_DIR, filename);

    if (fs.existsSync(filePath)) {
      const ext = path.extname(filename).toLowerCase();
      const contentTypes = {
        '.png': 'image/png',
        '.glb': 'model/gltf-binary',
        '.blend': 'application/x-blender',
        '.json': 'application/json',
      };
      res.writeHead(200, {
        'Content-Type': contentTypes[ext] || 'application/octet-stream',
        'Cache-Control': 'no-cache',
      });
      fs.createReadStream(filePath).pipe(res);
      return;
    } else {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('File Not Found');
      return;
    }
  }

  // API: Generate City Block via Blender CLI
  if (req.method === 'POST' && url.pathname === '/api/generate') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const config = JSON.parse(body || '{}');
        const configPath = path.join(SCRIPT_DIR, 'current_config.json');
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');

        // Execute Blender CLI headlessly
        const args = [
          '--background',
          '--python',
          ENGINE_SCRIPT,
          '--',
          configPath,
          SCRIPT_DIR
        ];

        console.log(`[CityFoundry] Executing Blender CLI: ${BLENDER_EXE} ${args.join(' ')}`);

        execFile(BLENDER_EXE, args, { timeout: 30000 }, (error, stdout, stderr) => {
          if (error) {
            console.error('[CityFoundry] Blender CLI Error:', error, stderr);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: error.message, stderr }));
            return;
          }

          console.log('[CityFoundry] Blender CLI finished successfully.');
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: true,
            preview: '/files/city_foundry_preview.png',
            glb: '/files/city_foundry.glb',
            blend: '/files/city_foundry.blend',
            tacticalData: '/files/tactical_level_data.json',
          }));
        });
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: err.message }));
      }
    });
    return;
  }

  // API: Launch Blender GUI with the generated scene
  if (req.method === 'POST' && url.pathname === '/api/open-blender') {
    const blendFile = path.join(SCRIPT_DIR, 'city_foundry.blend');
    console.log(`[CityFoundry] Spawning Blender GUI: ${BLENDER_EXE} ${blendFile}`);
    spawn(BLENDER_EXE, [blendFile], { detached: true, stdio: 'ignore' }).unref();

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, message: 'Blender launched' }));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/toolkit/fetch') {
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', () => {
      try {
        const payload = JSON.parse(body || '{}');
        const mode = payload.mode === 'all' ? 'all' : payload.mode === 'tools' ? 'tools' : 'permissive';
        const args = ['-ExecutionPolicy', 'Bypass', '-File', TOOLKIT_SCRIPT];
        if (mode === 'tools' || mode === 'all') args.push('-IncludeToolsOnly');
        if (mode === 'all') args.push('-IncludeStudyOnly');

        execFile('pwsh', args, { cwd: path.dirname(__dirname), timeout: 240000 }, (error, stdout, stderr) => {
          if (error) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: error.message, stdout, stderr }));
            return;
          }

          let parsed = [];
          if (fs.existsSync(TOOLKIT_REPORT)) {
            try { parsed = JSON.parse(fs.readFileSync(TOOLKIT_REPORT, 'utf-8')); }
            catch { parsed = []; }
          }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, mode, results: parsed }));
        });
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: err.message }));
      }
    });
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/toolkit/report') {
    if (!fs.existsSync(TOOLKIT_REPORT)) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: false, error: 'report not found' }));
      return;
    }
    try {
      const report = JSON.parse(fs.readFileSync(TOOLKIT_REPORT, 'utf-8'));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, report }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: false, error: err.message }));
    }
    return;
  }

  // Unreal Engine 5.7 Automation API
  if (req.method === 'GET' && url.pathname === '/api/unreal/status') {
    const pyScript = path.join(__dirname, 'unreal', 'ue_controller.py');
    execFile('python', [pyScript, 'status'], (err, stdout, stderr) => {
      if (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: err.message, stderr }));
        return;
      }
      try {
        const status = JSON.parse(stdout);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, status }));
      } catch {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, raw: stdout }));
      }
    });
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/unreal/launch') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const payload = body ? JSON.parse(body) : {};
        const arena = payload.arena || 'duel_pit';
        const bots = String(payload.bots || 4);
        const level = payload.level || 'sharp';
        const zombies = String(payload.zombies || 12);
        const zkind = payload.zkind || 'runner';
        const isHeadless = Boolean(payload.headless);

        const pyScript = path.join(__dirname, 'unreal', 'ue_controller.py');
        const args = [pyScript, 'play', '--arena', arena, '--bots', bots, '--level', level, '--zombies', zombies, '--zkind', zkind];
        if (isHeadless) args.push('--headless');

        const proc = spawn('python', args, { detached: true, stdio: 'ignore' });
        proc.unref();

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          pid: proc.pid,
          mode: isHeadless ? 'headless' : 'standalone',
          command: `python tools/unreal/ue_controller.py play --arena ${arena} --bots ${bots} --level ${level} --zombies ${zombies} --zkind ${zkind}${isHeadless ? ' --headless' : ''}`
        }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: err.message }));
      }
    });
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   🏙️  CityFoundry — 3D Asset Studio & Tactical Level Builder    ║
║                                                                ║
║   ► Web App URL:  http://localhost:${PORT}                       ║
║   ► Engine:       Blender 5.2.2 LTS (CLI)                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
`);
});
