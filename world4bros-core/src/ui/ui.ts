import type { Game, SimInput, Target } from '../sim/game';
import { HOTBAR, INV_SIZE } from '../sim/game';
import { Comp, type PawnWorkRole } from '../sim/components';
import { isRoad, getRoadType, insideBuilding } from '../sim/worldgen';
import type { Inventory } from '../sim/inventory';
import { clamp } from '../engine/math';
import type { Entity } from '../engine/ecs';
import type { Input } from '../engine/input';
import type { StorageAdapter } from '../engine/save';
import type { Renderer, RenderState } from '../render/renderer';
import { itemIcon } from '../render/icons';
import { SLOTS, SLOT_LABELS, type Slot, type StorytellerKind, type PawnSkill } from '../data/schema';
import { Customizer } from './customizer';
import { Gunsmith } from './gunsmith';
import { h } from './dom';
import { SoundEngine } from '../audio/sound';
import { SwarmSim, defaultSwarmCfg, runSwarmHeadless, compareWeaponsPaired, SIM_WEAPONS, type SwarmSimSummary, type SwarmSimCfg } from '../sim/swarm';
import { devLog } from '../utils/dev-log';
import { GEMINI_ASSET_RECIPES } from '../tools/gemini-prompt-helper';

export const sound = new SoundEngine();

type PanelName = 'inventory' | 'craft' | 'build' | 'garage' | 'overseer' | 'faction' | 'customizer' | 'storage' | 'settings' | 'map' | 'editor' | 'gunsmith' | 'range' | 'swarmlab' | 'arena' | 'devlog';
type EditTool = 'terrain' | 'biome' | 'object' | 'building' | 'erase';
export const SAVE_KEY = 'w4b-core-save-v1';

/** HUD, panels and mouse/keyboard routing. Reads the sim, never mutates rendering state directly. */
export class Ui {
  panel: PanelName | null = null;
  buildId: string | null = null;
  buildDir: 'N' | 'E' | 'S' | 'W' = 'E';
  debug = false;
  mouseMoveEnabled = false;
  mouseMoveTarget: { x: number; y: number } | null = null;
  private invTab: 'backpack' | 'interior' | 'base' = 'backpack';
  private mouseNavBtn: HTMLElement | null = null;
  private storageEntity: Entity | null = null;
  private selected: number | null = null;
  private selectedColonist: Entity | null = null;
  private target: Target = null;
  private clock = 0;
  private hotbarTimer = 0;
  private panelTimer = 0;
  private customizer: Customizer;
  private gunsmith: Gunsmith;
  private stepSoundTimer = 0;
  private minimapTimer = 0;
  private mapCanvas: HTMLCanvasElement | null = null;
  private mapPan = { x: 0, y: 0 };
  private mapZoom = 1.0;
  private mapDragging = false;
  private mapDragStart = { x: 0, y: 0, px: 0, py: 0 };
  private el = {
    stats: document.getElementById('stats') as HTMLElement, clock: document.getElementById('clock') as HTMLElement, prompt: document.getElementById('prompt') as HTMLElement,
    banner: document.getElementById('banner') as HTMLElement, toasts: document.getElementById('toasts') as HTMLElement, hotbar: document.getElementById('hotbar') as HTMLElement,
    panel: document.getElementById('panel') as HTMLElement, debug: document.getElementById('debug') as HTMLElement, menu: document.getElementById('menu') as HTMLElement,
    minimap: document.getElementById('minimap') as HTMLCanvasElement,
  };
  private bars: Record<string, HTMLElement> = {};
  private ghost = { tx: 0, ty: 0 };
  private edit: { tool: EditTool; terrain: string; biome: number; kind: string; radius: number; building: string; door: 'N' | 'E' | 'S' | 'W' } = { tool: 'terrain', terrain: 'grass', biome: 0, kind: 'oak', radius: 2, building: 'cottage', door: 'S' };
  private brush: { x: number; y: number; r: number } | null = null;
  public toastMode: 'all' | 'important' | 'off' = (typeof localStorage !== 'undefined' && (localStorage.getItem('w4b_toast_mode') as 'all' | 'important' | 'off')) || 'all';
  private activeToasts = new Map<string, { el: HTMLElement; count: number; timer: ReturnType<typeof setTimeout> }>();
  private toastBtn?: HTMLElement;

  constructor(private game: Game, private renderer: Renderer, private input: Input, private storage: StorageAdapter) {
    this.customizer = new Customizer(game, renderer);
    this.gunsmith = new Gunsmith(game, () => { this.refreshPanel(); this.renderHotbar(); });
    if (this.el.minimap) {
      this.el.minimap.onclick = () => this.toggle('map');
    }
    if (this.el.clock) {
      this.el.clock.style.cursor = 'pointer';
      this.el.clock.title = 'Click to toggle Daylight (Shortcut: L)';
      this.el.clock.onclick = () => this.toggleDaylight();
    }
    for (const [k, label] of [['hunger', 'Hunger'], ['health', 'Health'], ['sanity', 'Sanity']] as const) {
      const fill = h('div', { class: `fill ${k}` });
      this.bars[k] = fill;
      this.el.stats.append(h('div', { class: 'stat' }, h('span', {}, label), h('div', { class: 'bar' }, fill)));
    }
    const btn = (label: string, key: string, fn: () => void): HTMLElement => h('button', { onclick: fn, title: key }, label, h('kbd', {}, key));
    this.mouseNavBtn = btn('Nav: OFF', 'T', () => this.toggleMouseMove());
    this.toastBtn = btn(`Toasts: ${this.toastMode.toUpperCase()}`, 'F8', () => this.toggleToastMode());
    this.updateToastBtn();
    this.el.menu.append(
      btn('Pack', 'I', () => this.toggle('inventory')),
      this.mouseNavBtn,
      btn('Craft', 'R', () => this.toggle('craft')),
      btn('Build', 'B', () => this.toggle('build')),
      btn('Gunsmith', 'U', () => this.toggle('gunsmith')),
      btn('Range', 'Y', () => this.toggle('range')),
      btn('Arena', 'H', () => this.toggle('arena')),
      btn('SwarmLab', 'J', () => this.toggle('swarmlab')),
      btn('Garage', 'V', () => this.toggle('garage')),
      btn('Overseer', 'O', () => this.toggle('overseer')),
      btn('Faction', 'P', () => this.toggle('faction')),
      btn('Map', 'TAB', () => this.toggle('map')),
      btn('Character', 'C', () => this.toggle('customizer')),
      btn('Mode', 'M', () => this.toggle('settings')),
      this.toastBtn,
      btn('DevLog', 'F2', () => this.toggle('devlog')),
      btn('Save', 'F5', () => this.save())
    );

    let lastToastText = '';
    let lastToastTime = 0;
    let toastRepeatCount = 1;

    game.events.on('toast', (e) => {
      this.toast(e.text);
      const now = performance.now();
      if (e.text === lastToastText && now - lastToastTime < 4000) {
        toastRepeatCount++;
      } else {
        if (toastRepeatCount > 1) {
          devLog.log('ui', `(Previous toast repeated ×${toastRepeatCount})`);
        }
        lastToastText = e.text;
        lastToastTime = now;
        toastRepeatCount = 1;
        devLog.log('ui', `Toast: ${e.text}`);
      }
    });
    game.events.on('storage', (e) => { this.storageEntity = e.entity; this.open('storage'); devLog.log('world', 'Opened storage chest'); });
    game.events.on('vehicleTrunk', (e) => { this.storageEntity = e.entity; this.open('storage'); devLog.log('world', 'Opened vehicle trunk'); });
    game.events.on('overseer', () => { this.open('overseer'); devLog.log('ui', 'Opened overseer panel'); });
    game.events.on('faction', () => { this.open('faction'); devLog.log('ui', 'Opened faction panel'); });
    game.events.on('gunsmith', () => { this.open('gunsmith'); devLog.log('ui', 'Opened gunsmith bench'); });
    game.events.on('death', () => { this.toast('You collapse. The fire is a distant memory.', true); devLog.warn('combat', 'Player died'); });
    game.events.on('zombieKilled', (e) => devLog.combat(`Zombie eliminated (${e.kind}) at (${e.x.toFixed(1)}, ${e.y.toFixed(1)})`));
    game.events.on('zombieAttack', (e) => devLog.combat(`Zombie attack: -${e.damage} HP`));
    game.events.on('hit', (e) => {
      devLog.combat(`Impact hit: ${e.kind} at (${e.x.toFixed(1)}, ${e.y.toFixed(1)})`);
      if (e.kind === 'flesh') sound.hitFlesh();
      else if (e.kind === 'rock' || e.kind === 'boulder') sound.mine();
      else sound.chop();
    });
    game.events.on('harvest', () => sound.harvest());
    game.events.on('built', () => sound.craft());
    game.events.on('zombieAttack', () => sound.hitFlesh());
    game.events.on('petBark', () => sound.bark());
    game.events.on('snapCamera', (e) => { this.renderer.cam.snap(e.x, e.y); });
    this.renderHotbar();
  }

  isImportantToast(text: string): boolean {
    const lower = text.toLowerCase();
    return (
      lower.includes('saved') ||
      lower.includes('collapse') ||
      lower.includes('died') ||
      lower.includes('eliminated') ||
      lower.includes('victory') ||
      lower.includes('outbreak') ||
      lower.includes('peace treaty') ||
      lower.includes('siege') ||
      lower.includes('cannot') ||
      lower.includes('error') ||
      lower.includes('toasts:') ||
      lower.includes('mode')
    );
  }

  toast(text: string, force = false): void {
    if (!force) {
      if (this.toastMode === 'off') return;
      if (this.toastMode === 'important' && !this.isImportantToast(text)) return;
    }

    const key = text.replace(/\s*[×x]\d+$/i, '').trim();
    const existing = this.activeToasts.get(key);

    if (existing && document.body.contains(existing.el)) {
      existing.count++;
      clearTimeout(existing.timer);

      existing.el.replaceChildren(
        key,
        h('span', { class: 'toast-count' }, `×${existing.count}`)
      );

      existing.el.style.animation = 'none';
      void existing.el.offsetWidth;
      existing.el.style.animation = 'fade 2.4s forwards';

      existing.timer = setTimeout(() => {
        existing.el.remove();
        this.activeToasts.delete(key);
      }, 2400);
      return;
    }

    const t = h('div', { class: 'toast', title: 'Click to dismiss · [F8] Toggle Toasts' }, text);
    t.onclick = () => {
      t.remove();
      this.activeToasts.delete(key);
    };

    this.el.toasts.append(t);

    while (this.el.toasts.children.length > 3) {
      const first = this.el.toasts.firstElementChild as HTMLElement;
      if (first) {
        for (const [k, v] of this.activeToasts.entries()) {
          if (v.el === first) {
            clearTimeout(v.timer);
            this.activeToasts.delete(k);
            break;
          }
        }
        first.remove();
      }
    }

    const timer = setTimeout(() => {
      t.remove();
      this.activeToasts.delete(key);
    }, 2600);

    this.activeToasts.set(key, { el: t, count: 1, timer });
  }

  updateToastBtn(): void {
    if (!this.toastBtn) return;
    const col = this.toastMode === 'off' ? '#ef4444' : this.toastMode === 'important' ? 'var(--gold)' : '';
    this.toastBtn.replaceChildren(
      'Toasts: ',
      h('b', { style: col ? `color: ${col};` : '' }, this.toastMode.toUpperCase()),
      h('kbd', {}, 'F8')
    );
    this.toastBtn.style.borderColor = col || 'var(--line)';
  }

  toggleToastMode(): void {
    const modes: ('all' | 'important' | 'off')[] = ['all', 'important', 'off'];
    const nextIdx = (modes.indexOf(this.toastMode) + 1) % modes.length;
    this.toastMode = modes[nextIdx];
    try {
      localStorage.setItem('w4b_toast_mode', this.toastMode);
    } catch {}

    this.updateToastBtn();

    const msg = this.toastMode === 'all'
      ? '💬 Toasts: ALL (Calm & Deduplicated)'
      : this.toastMode === 'important'
      ? '🔕 Toasts: QUIET (Important alerts only)'
      : '🔇 Toasts: MUTED (Completely off)';
    this.toast(msg, true);
  }

  toggleMouseMove(): void {
    this.mouseMoveEnabled = !this.mouseMoveEnabled;
    this.mouseMoveTarget = null;
    if (this.mouseNavBtn) {
      this.mouseNavBtn.replaceChildren(
        'Nav: ',
        h('b', { style: this.mouseMoveEnabled ? 'color: var(--gold);' : '' }, this.mouseMoveEnabled ? 'ON' : 'OFF'),
        h('kbd', {}, 'T')
      );
      this.mouseNavBtn.style.borderColor = this.mouseMoveEnabled ? 'var(--gold)' : 'var(--line)';
    }
    this.toast(this.mouseMoveEnabled ? '🖱️ Move by Mouse: ON (Click/hold anywhere to navigate)' : '⌨️ Move by Mouse: OFF (Standard WASD movement)');
  }
  toggleDaylight(): void {
    const g = this.game;
    g.clock.alwaysDay = !g.clock.alwaysDay;
    if (g.clock.alwaysDay) {
      g.clock.t = 0.25;
      this.toast('☀️ Always Daylight: ON');
    } else {
      this.toast('🌙 Day/Night Cycle: RESUMED');
    }
  }
  save(): void { this.storage.save(SAVE_KEY, JSON.stringify(this.game.toSave())); this.toast('Game saved'); }

  toggle(p: PanelName): void { if (this.panel === p) this.close(); else this.open(p); }
  open(p: PanelName): void {
    this.panel = p;
    if (p !== 'build') this.buildId = null;
    if (p === 'map') {
      this.mapPan.x = this.game.playerT.x;
      this.mapPan.y = this.game.playerT.y;
    }
    if (p === 'gunsmith') {
      const held = this.game.playerChar.look.outfit.mainHand?.item;
      if (held && ['hunting_rifle', 'pump_shotgun', 'nailgun_smg', 'potato_cannon', 'slingshot'].includes(held)) {
        this.gunsmith.selectedWeapon = held;
      }
      this.gunsmith.refresh();
    }
    this.renderPanel();
    this.el.panel.classList.add('open');
  }
  close(): void { this.panel = null; this.buildId = null; this.mapCanvas = null; this.el.panel.classList.remove('open'); this.el.panel.replaceChildren(); }

  // ---------------------------------------------------------------- per frame
  simInput(): SimInput {
    const i = this.input;
    const building = this.buildId !== null || (this.panel === 'editor' && this.game.world.staticActive);
    let move = i.axis();
    const hasKeyboardMove = Math.hypot(move.x, move.y) > 0.05;

    const heldItemId = this.game.playerChar.look.outfit.mainHand?.item;
    const heldItemDef = heldItemId ? this.game.content.items.get(heldItemId) : null;
    const isGun = !!heldItemDef?.gun;

    const w = this.renderer.canvas.clientWidth, h = this.renderer.canvas.clientHeight;
    const mouseWorld = this.renderer.cam.screenToWorld(i.mouse.x, i.mouse.y, w, h);

    // Aim stance: hold Ctrl or Right Mouse Button while holding a firearm (and not inside modal/building)
    const aimHeld = isGun && !building && !this.panel && (i.held('ControlLeft') || i.held('ControlRight') || i.mouse.right);
    const shootPressed = aimHeld && (i.mouse.leftPressed || i.pressed('Space'));

    if (hasKeyboardMove || aimHeld) {
      this.mouseMoveTarget = null;
    } else if (this.mouseMoveEnabled && !this.panel && !building) {
      if (i.mouse.leftPressed || i.mouse.left) {
        this.mouseMoveTarget = { x: mouseWorld.x, y: mouseWorld.y };
      }
      if (this.mouseMoveTarget) {
        const dx = this.mouseMoveTarget.x - this.game.playerT.x;
        const dy = this.mouseMoveTarget.y - this.game.playerT.y;
        const dist = Math.hypot(dx, dy);
        if (dist > 0.35) {
          move = { x: dx / dist, y: dy / dist };
        } else {
          this.mouseMoveTarget = null;
          move = { x: 0, y: 0 };
        }
      }
    }

    const t = this.target;
    const actionTargetOk = t !== null && (
      (t.kind === 'object' && t.ok) ||
      t.kind === 'zombie' ||
      (t.kind === 'animal' && t.key === 'SPACE')
    );
    const actionPressed = !aimHeld && (
      i.pressed('Space') || i.touchAction ||
      (!this.mouseMoveEnabled && i.mouse.leftPressed && !building) ||
      (this.mouseMoveEnabled && i.mouse.leftPressed && !building && actionTargetOk)
    );

    return {
      move,
      sprint: i.held('ShiftLeft') || i.held('ShiftRight'),
      actionPressed,
      interactPressed: i.pressed('KeyE') || i.touchUse,
      aimHeld,
      aimWorld: mouseWorld,
      shootPressed,
    };
  }

  renderState(): RenderState {
    let build: RenderState['build'] = null;
    if (this.buildId) { const def = this.game.content.structures.get(this.buildId); if (def) build = { id: def.id, tx: this.ghost.tx, ty: this.ghost.ty, ok: this.game.checkBuild(def, this.ghost.tx, this.ghost.ty).ok }; }

    let aim: RenderState['aim'] = null;
    const heldItemId = this.game.playerChar.look.outfit.mainHand?.item;
    const heldDef = heldItemId ? this.game.content.items.get(heldItemId) : null;
    if (this.game.aimActive && heldDef?.gun && heldItemId) {
      const g = this.game.getEffectiveGun(heldItemId);
      const w = this.renderer.canvas.clientWidth, h = this.renderer.canvas.clientHeight;
      const mouseWorld = this.renderer.cam.screenToWorld(this.input.mouse.x, this.input.mouse.y, w, h);
      let ammoCount = 0;
      if (heldItemId === 'potato_cannon') {
        ammoCount = this.game.playerInv.inv.count('wild_carrot') + this.game.playerInv.inv.count('canned_beans');
      } else if (g.ammoItem) {
        ammoCount = this.game.playerInv.inv.count(g.ammoItem);
      }

      // Visual info targets and threats calculation
      let aimedTarget: { name: string; hp: number; maxHp: number; dist: number } | null = null;
      const threats: { x: number; y: number; name: string; hp: number; maxHp: number; dist: number }[] = [];

      const px = this.game.playerT.x, py = this.game.playerT.y;

      // Scan all potential threats (zombies within 32m) if threatScanner is active
      if (g.visualInfo?.threatScanner) {
        const scanRange = Math.min(32, g.maxRange + 6);
        for (const e of this.game.ecs.query(Comp.Zombie, Comp.Transform)) {
          const zt = this.game.ecs.get(e, Comp.Transform)!;
          const zc = this.game.ecs.get(e, Comp.Zombie)!;
          const d = Math.hypot(zt.x - px, zt.y - py);
          if (d <= scanRange && zc.hp > 0) {
            threats.push({
              x: zt.x,
              y: zt.y,
              name: zc.def.name,
              hp: zc.hp,
              maxHp: zc.def.maxHp,
              dist: d,
            });
          }
        }
      }

      // Find targeted entity under or closest to mouse cursor (within 2.2 tiles of cursor)
      let bestDist = 2.2;
      for (const e of this.game.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = this.game.ecs.get(e, Comp.Transform)!;
        const zc = this.game.ecs.get(e, Comp.Zombie)!;
        if (zc.hp <= 0) continue;
        const mouseDist = Math.hypot(zt.x - mouseWorld.x, zt.y - mouseWorld.y);
        if (mouseDist < bestDist) {
          bestDist = mouseDist;
          aimedTarget = {
            name: zc.def.name,
            hp: zc.hp,
            maxHp: zc.def.maxHp,
            dist: Math.hypot(zt.x - px, zt.y - py),
          };
        }
      }

      for (const e of this.game.ecs.query(Comp.Animal, Comp.Transform)) {
        const at = this.game.ecs.get(e, Comp.Transform)!;
        const ac = this.game.ecs.get(e, Comp.Animal)!;
        if (ac.hp <= 0) continue;
        const mouseDist = Math.hypot(at.x - mouseWorld.x, at.y - mouseWorld.y);
        if (mouseDist < bestDist) {
          bestDist = mouseDist;
          aimedTarget = {
            name: ac.name || ac.kind,
            hp: ac.hp,
            maxHp: ac.maxHp,
            dist: Math.hypot(at.x - px, at.y - py),
          };
        }
      }

      for (const e of this.game.ecs.query(Comp.TargetDummy, Comp.Transform)) {
        const dt = this.game.ecs.get(e, Comp.Transform)!;
        const dummy = this.game.ecs.get(e, Comp.TargetDummy)!;
        const mouseDist = Math.hypot(dt.x - mouseWorld.x, dt.y - mouseWorld.y);
        if (mouseDist < bestDist) {
          bestDist = mouseDist;
          aimedTarget = {
            name: `Target Dummy (${dummy.distanceLabel})`,
            hp: 100,
            maxHp: 100,
            dist: Math.hypot(dt.x - px, dt.y - py),
          };
        }
      }

      const tx = Math.floor(mouseWorld.x);
      const ty = Math.floor(mouseWorld.y);
      const structEntity = this.game.base.structureAt(tx, ty);
      if (structEntity !== undefined) {
        const s = this.game.ecs.get(structEntity, Comp.Structure);
        if (s) {
          const mouseDist = Math.hypot(tx + 0.5 - mouseWorld.x, ty + 0.5 - mouseWorld.y);
          if (mouseDist < bestDist) {
            bestDist = mouseDist;
            aimedTarget = {
              name: s.def.name,
              hp: s.hp,
              maxHp: s.def.hp ?? s.hp,
              dist: Math.hypot(tx + 0.5 - px, ty + 0.5 - py),
            };
          }
        }
      }

      aim = {
        active: true,
        aimSystem: g.aimSystem,
        origin: { x: this.game.playerT.x, y: this.game.playerT.y - 0.4 },
        target: mouseWorld,
        spread: g.spreadAngle,
        maxRange: g.maxRange,
        gunName: heldDef.name,
        ammoName: g.ammoItem ? this.game.content.items.get(g.ammoItem)?.name : 'Carrots / Beans',
        ammoCount,
        visualInfo: g.visualInfo,
        aimedTarget,
        threats,
      };
    }

    return {
      build,
      target: this.target,
      debug: this.debug,
      showClaims: this.panel === 'build',
      brush: this.panel === 'editor' ? this.brush : null,
      moveMarker: this.mouseMoveEnabled ? this.mouseMoveTarget : null,
      aim,
    };
  }

  frame(dt: number, fps: number): void {
    const i = this.input, g = this.game;
    if (i.pressed('KeyI')) this.toggle('inventory');
    if (i.pressed('KeyM') || i.pressed('Tab') || i.pressed('KeyN')) this.toggle('map');
    if (i.pressed('KeyF')) {
      const on = g.toggleFogOfWar();
      this.drawMapCanvas();
      this.toast(on ? 'Fog of War: ON' : '👁️ Fog of War: OFF (World Fully Revealed)');
    }
    if (i.pressed('KeyZ')) {
      this.renderer.cam.toggleMaxView();
      this.toast(this.renderer.cam.targetZoom <= 12 ? '🌐 Max View Distance Enabled (Zoom 8)' : '🔍 Normal View Distance (Zoom 54)');
    }
    if (this.buildId) {
      if (i.pressed('KeyR') || i.pressed('KeyT')) {
        const next: Record<'N' | 'E' | 'S' | 'W', 'N' | 'E' | 'S' | 'W'> = { E: 'S', S: 'W', W: 'N', N: 'E' };
        this.buildDir = next[this.buildDir];
        this.toast(`Placement Facing: ${this.buildDir}`);
      }
    } else {
      if (i.pressed('KeyR')) this.toggle('craft');
      if (i.pressed('KeyT')) this.toggleMouseMove();
    }
    if (i.pressed('KeyB')) this.toggle('build');
    if (i.pressed('KeyU')) this.toggle('gunsmith');
    if (i.pressed('KeyY')) this.toggle('range');
    if (i.pressed('KeyH')) this.toggle('arena');
    if (i.pressed('KeyJ')) this.toggle('swarmlab');
    if (i.pressed('KeyV')) this.toggle('garage');
    if (i.pressed('KeyO')) this.toggle('overseer');
    if (i.pressed('KeyP')) this.toggle('faction');
    if (i.pressed('KeyC')) this.toggle('customizer');
    if (i.pressed('KeyK')) this.toggle('settings');
    if (i.pressed('KeyG')) this.toggleEditor();
    if (i.pressed('KeyL')) this.toggleDaylight();
    if (i.pressed('F2') || i.pressed('Backquote')) this.toggle('devlog');
    if (i.pressed('F3')) { this.debug = !this.debug; this.el.debug.classList.toggle('on', this.debug); }
    if (i.pressed('F5')) this.save();
    if (i.pressed('F8')) this.toggleToastMode();
    if (i.pressed('Escape')) { if (this.buildId) this.buildId = null; else this.close(); }

    this.minimapTimer += dt;
    if (this.minimapTimer > 0.04) {
      this.minimapTimer = 0;
      this.updateMinimap();
    }

    const drivingVeh = g.drivingVehicle();
    if (drivingVeh) {
      const v = g.ecs.get(drivingVeh, Comp.Vehicle)!;
      sound.updateEngine(v.speed);
      if (i.pressed('KeyH')) sound.honk();
      this.renderer.cam.drivingSpeedOffset = Math.min(24, (v.speed / Math.max(1, v.def.maxSpeed)) * 18);
    } else {
      this.renderer.cam.drivingSpeedOffset = 0;
      sound.stopEngine();
      if (Math.hypot(i.axis().x, i.axis().y) > 0.3) {
        this.stepSoundTimer += dt;
        if (this.stepSoundTimer > 0.36) {
          this.stepSoundTimer = 0;
          sound.footstep();
        }
      }
    }
    for (let n = 1; n <= HOTBAR; n++) if (i.pressed(`Digit${n}`)) { g.holdSlot(n - 1); this.renderHotbar(); }
    if (i.pressed('Equal') || i.pressed('NumpadAdd')) this.renderer.cam.setZoom(this.renderer.cam.targetZoom * 1.15);
    if (i.pressed('Minus') || i.pressed('NumpadSubtract')) this.renderer.cam.setZoom(this.renderer.cam.targetZoom / 1.15);
    if (i.mouse.wheel) this.renderer.cam.setZoom(this.renderer.cam.targetZoom * Math.exp(-i.mouse.wheel * 0.0012));

    // build mode: ghost follows the mouse; click places, right-click removes
    if (this.buildId) {
      const def = g.content.structures.get(this.buildId);
      const w = this.renderer.canvas.clientWidth, hh = this.renderer.canvas.clientHeight, p = this.renderer.cam.screenToWorld(i.mouse.x, i.mouse.y, w, hh);
      if (def) { this.ghost.tx = Math.round(p.x - def.size[0] / 2); this.ghost.ty = Math.round(p.y - def.size[1] / 2); }
      if (i.mouse.leftPressed && def) { g.build(def.id, this.ghost.tx, this.ghost.ty, this.buildDir); this.refreshPanel(); }
      if (i.mouse.rightPressed || i.pressed('KeyX')) {
        const e = g.base.structureAt(Math.floor(p.x), Math.floor(p.y));
        if (e !== undefined && g.demolish(e)) this.toast('Salvaged half the materials');
      }
    }
    // map editor: brush follows the mouse on the frozen static map; left paints/places, right erases objects
    this.brush = null;
    if (this.panel === 'editor' && g.world.staticActive) {
      const w = this.renderer.canvas.clientWidth, hh = this.renderer.canvas.clientHeight, p = this.renderer.cam.screenToWorld(i.mouse.x, i.mouse.y, w, hh);
      const e = this.edit;
      this.brush = { x: p.x, y: p.y, r: e.tool === 'object' ? 0.5 : e.radius };
      if (i.mouse.left) {
        if (e.tool === 'terrain') g.world.paint(p.x, p.y, e.radius, e.terrain as never, null);
        else if (e.tool === 'biome') g.world.paint(p.x, p.y, e.radius, null, e.biome);
        else if (e.tool === 'erase') g.world.eraseObjects(p.x, p.y, e.radius);
      }
      if (e.tool === 'object' && i.mouse.leftPressed) g.world.placeObject(e.kind as never, p.x, p.y);
      if (e.tool === 'building' && i.mouse.leftPressed) {
        const bd = g.content.buildings.find((x) => x.id === e.building);
        if (bd) {
          const bw = Math.round((bd.minW + bd.maxW) / 2), bh = Math.round((bd.minH + bd.maxH) / 2);
          if (!g.world.placeBuilding(bd.id, Math.round(p.x - bw / 2), Math.round(p.y - bh / 2), bw, bh, e.door)) this.toast('Cannot place a building outside the map');
        }
      }
      if (e.tool === 'erase' && i.mouse.leftPressed) g.world.eraseBuildings(p.x, p.y);
      if (i.mouse.rightPressed) { g.world.eraseObjects(p.x, p.y, 0.7); g.world.eraseBuildings(p.x, p.y); }
    }
    this.target = this.buildId ? null : g.findTarget();

    // HUD
    const n = g.playerNeeds;
    this.bars.hunger!.style.width = `${(n.hunger / n.maxHunger) * 100}%`; this.bars.health!.style.width = `${(n.health / n.maxHealth) * 100}%`; this.bars.sanity!.style.width = `${(n.sanity / n.maxSanity) * 100}%`;
    for (const k of ['hunger', 'health', 'sanity'] as const) this.bars[k]!.parentElement?.classList.toggle('low', (n[k] / (k === 'hunger' ? n.maxHunger : k === 'health' ? n.maxHealth : n.maxSanity)) < 0.25);
    this.clock += dt;
    if (drivingVeh) {
      const v = g.ecs.get(drivingVeh, Comp.Vehicle)!;
      const speedKmh = Math.abs(Math.round(v.speed * 8));
      this.el.clock.textContent = `🚗 ${v.def.name}  ·  ${speedKmh} km/h  ·  Fuel ${Math.round(v.fuel)}%  ·  HP ${Math.round(v.hp)}%`;
    } else {
      const phaseStr = g.clock.alwaysDay ? '☀️ DAY' : `${g.clock.phase[0]?.toUpperCase()}${g.clock.phase.slice(1)}`;
      const zone = g.faction.getZoneAt(g.playerT.x, g.playerT.y, g.world.spawn.x, g.world.spawn.y);
      const zoneBadge = zone === 'green' ? '🟢 GREEN' : zone === 'yellow' ? '🟡 YELLOW' : '🔴 RED';
      this.el.clock.textContent = `${zoneBadge} · Day ${g.clock.day} · ${phaseStr} · ${g.world.content.biomes.find((b) => b.id === g.world.biomeAt(g.playerT.x, g.playerT.y))?.name ?? ''}`;
    }
    const t = this.target;
    this.el.prompt.classList.toggle('show', !!t);
    if (t) { this.el.prompt.textContent = t.kind === 'object' && !t.ok ? `Need ${t.need === 'axe' ? 'an axe' : t.need === 'pickaxe' ? 'a pickaxe' : 'a tool'}: ${t.prompt}` : `[${t.key}] ${t.prompt}`; this.el.prompt.classList.toggle('bad', t.kind === 'object' && !t.ok); }
    this.el.banner.classList.toggle('show', n.darkTimer > 1 && g.clock.darkness > 0.55);
    if (this.buildId) {
      const d = g.content.structures.get(this.buildId);
      const c = d ? g.checkBuild(d, this.ghost.tx, this.ghost.ty) : null;
      this.el.prompt.classList.add('show');
      this.el.prompt.classList.toggle('bad', !!c && !c.ok);
      const dirArrow = { N: '⬆️ North', E: '➡️ East', S: '⬇️ South', W: '⬅️ West' }[this.buildDir];
      this.el.prompt.textContent = c?.ok
        ? `[R to Rotate: ${dirArrow}] · Click to place ${d?.name}. Right-click removes.`
        : `${c?.reason ?? 'Cannot build here'} · [R to Rotate: ${dirArrow}]`;
    }

    this.hotbarTimer += dt; if (this.hotbarTimer > 0.2) { this.hotbarTimer = 0; this.renderHotbar(); }
    this.panelTimer += dt;
    if (this.panel === 'customizer') this.customizer.update(dt);
    else if (this.panel === 'gunsmith') { /* Gunsmith panel is reactive */ }
    else if (this.panel && this.panelTimer > 0.4) { this.panelTimer = 0; this.refreshPanel(); }
    if (this.debug) {
      const p = g.playerT;
      this.el.debug.textContent = `fps ${fps.toFixed(0)}  entities ${g.ecs.count()}  chunks ${this.renderer.stats.chunks}  drawn ${this.renderer.stats.drawn}  sprites ${this.renderer.cache.size}\npos ${p.x.toFixed(1)}, ${p.y.toFixed(1)}  seed ${g.config.seed}  mode ${g.config.mode}  biome ${g.world.biomeAt(p.x, p.y)}  time ${g.clock.t.toFixed(3)}`;
    }
  }

  // ---------------------------------------------------------------- hotbar and panels
  private invSlotEl(inv: Inventory, i: number, active = false, isSel = false, onclick?: () => void): HTMLElement {
    const st = inv.slots[i], def = st ? this.game.content.items.get(st.item) : undefined;
    const el = h('button', { class: `slot${active ? ' active' : ''}${isSel ? ' sel' : ''}`, title: def?.name ?? 'Empty', onclick });
    if (def && st) { el.append(itemIcon(this.game.content, def.id, 40)); if (st.qty > 1) el.append(h('b', {}, String(st.qty))); }
    return el;
  }
  private slotEl(i: number, active = false, onclick?: () => void): HTMLElement {
    return this.invSlotEl(this.game.playerInv.inv, i, active, this.selected === i && this.panel === 'inventory' && this.invTab === 'backpack', onclick);
  }
  private renderHotbar(): void {
    const held = this.game.playerInv.held;
    this.el.hotbar.replaceChildren(...Array.from({ length: HOTBAR }, (_, i) => { const s = this.slotEl(i, i === held, () => { this.game.holdSlot(i); this.renderHotbar(); }); s.append(h('i', {}, String(i + 1))); return s; }));
  }
  private refreshPanel(): void {
    if (this.panel && this.panel !== 'customizer' && this.panel !== 'editor') {
      if (this.panel === 'gunsmith') this.gunsmith.refresh();
      else this.renderPanel();
    }
  }

  private renderPanel(): void {
    const p = this.panel;
    if (!p) return;
    const body = p === 'inventory' ? this.inventoryPanel() : p === 'craft' ? this.craftPanel() : p === 'build' ? this.buildPanel() : p === 'gunsmith' ? this.gunsmith.el : p === 'range' ? this.provingGroundsPanel() : p === 'arena' ? this.arenaFightPanel() : p === 'devlog' ? this.devLogPanel() : p === 'swarmlab' ? this.swarmlabPanel() : p === 'garage' ? this.garagePanel() : p === 'overseer' ? this.overseerPanel() : p === 'faction' ? this.factionPanel() : p === 'storage' ? this.storagePanel() : p === 'settings' ? this.settingsPanel() : p === 'map' ? this.mapPanel() : p === 'editor' ? this.editorPanel() : this.customizer.el;
    this.el.panel.replaceChildren(p === 'customizer' ? body : h('div', { class: `panel${p === 'inventory' ? ' inv-wide' : ''}` }, h('button', { class: 'x', onclick: () => this.close(), 'aria-label': 'Close' }, '×'), body));
  }

  private garagePanel(): HTMLElement {
    const g = this.game;
    const pt = g.playerT;
    const vehicles = [...g.content.vehicles.values()];

    const fleetHeader = h('div', { class: 'garage-header' },
      h('h2', { style: 'margin: 0;' }, 'Vehicle Fleet & Garage'),
      h('button', {
        class: 'fleet-park-btn',
        onclick: () => {
          g.spawnFleet();
          this.toast('All 6 vehicles parked in town plaza!');
          this.close();
        }
      }, '🏘️ Park Full Fleet in Town Plaza')
    );

    const cards = vehicles.map((v) => {
      const icon = v.id === 'rusty_hatchback' ? '🚗'
        : v.id === 'clown_car' ? '🎪'
        : v.id === 'ice_cream_truck' ? '🍦'
        : v.id === 'golf_cart' ? '⛳'
        : v.id === 'bathtub_buggy' ? '🛁'
        : v.id === 'tesla_roadster' ? '⚡'
        : '🌭';

      const spawnBtn = h('button', {
        class: 'v-act-btn',
        onclick: () => {
          const ang = pt.angle + Math.PI / 2;
          const sx = pt.x + Math.cos(ang) * 2.6;
          const sy = pt.y + Math.sin(ang) * 2.6;
          const ve = g.spawnVehicle(sx, sy, v.id);
          const vt = g.ecs.get(ve, Comp.Transform)!;
          vt.angle = pt.angle;
          this.toast(`Spawned ${v.name}!`);
          this.close();
        }
      }, '🚀 Spawn Beside You');

      const driveBtn = h('button', {
        class: 'v-act-btn highlight',
        onclick: () => {
          const ve = g.spawnVehicle(pt.x, pt.y, v.id);
          const vc = g.ecs.get(ve, Comp.Vehicle)!;
          const vt = g.ecs.get(ve, Comp.Transform)!;
          vt.angle = pt.angle;
          vc.driver = g.player;
          this.toast(`Driving ${v.name}!`);
          this.close();
        }
      }, '🏁 Hop In & Drive');

      return h('div', { class: 'vehicle-card', style: `border-left: 4px solid ${v.color};` },
        h('div', { class: 'v-top' },
          h('div', { class: 'v-title' },
            h('span', { class: 'v-icon' }, icon),
            h('strong', {}, v.name)
          ),
          h('span', { class: 'v-badge' }, v.kind.replace('_', ' ').toUpperCase())
        ),
        h('p', { class: 'v-quirk' }, v.quirk ?? 'Specialized apocalypse vehicle.'),
        h('div', { class: 'v-stats' },
          h('span', { title: 'Top Speed' }, `⚡ ${v.maxSpeed} m/s`),
          h('span', { title: 'Acceleration' }, `🚀 ${v.accel}`),
          h('span', { title: 'Turn Agility' }, `🔄 ${v.turnSpeed}`),
          h('span', { title: 'Max Armor' }, `🛡️ ${v.maxHp} HP`),
          h('span', { title: 'Fuel Capacity' }, `⛽ ${v.maxFuel}L`),
          h('span', { title: 'Trunk Storage' }, `📦 ${v.trunkSlots} slots`)
        ),
        h('div', { class: 'v-buttons' }, spawnBtn, driveBtn)
      );
    });

    return h('div', { class: 'garage-panel' },
      fleetHeader,
      h('p', { class: 'hint' }, 'Choose any vehicle to spawn beside you or take for an immediate test drive:'),
      h('div', { class: 'garage-grid' }, ...cards)
    );
  }

  private provingGroundsPanel(): HTMLElement {
    const g = this.game;

    const arenaBanner = h('div', {
      style: 'background:linear-gradient(90deg, rgba(30,58,138,0.7) 0%, rgba(88,28,135,0.7) 100%); border:1.5px solid #38bdf8; border-radius:10px; padding:12px 14px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'
    },
      h('div', {},
        h('b', { style: 'color:#f8fafc; font-size:15px; display:block;' }, '🏟️ Aim Arena Live Environments (Z28a)'),
        h('span', { style: 'color:#cbd5e1; font-size:12px;' }, 'Spawn into Duel Pit (40x24m) or FFA Yard (64x40m) with full weapons & solid cover')
      ),
      h('div', { style: 'display:flex; gap:8px;' },
        h('button', {
          style: 'padding:8px 16px; background:#0284c7; color:#f0f9ff; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:13px;',
          onclick: () => {
            g.teleportToArena('duel_pit');
            this.toast('🏟️ Spawned inside Duel Pit (40x24m)! Mirrored cover & bullet stoppers active.');
            this.close();
          }
        }, '⚔️ Spawn in Duel Pit (40x24m)'),
        h('button', {
          style: 'padding:8px 16px; background:#7e22ce; color:#faf5ff; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:13px;',
          onclick: () => {
            g.teleportToArena('ffa_yard');
            this.toast('🏟️ Spawned inside FFA Yard (64x40m)! 8 Spawns active.');
            this.close();
          }
        }, '💥 Spawn in FFA Yard (64x40m)')
      )
    );

    const header = h('div', { style: 'display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid var(--line); padding-bottom:8px;' },
      h('div', {},
        h('h2', { style: 'margin:0; color:#f08a32; font-family:var(--display);' }, "🎯 Bro's Ballistics Proving Grounds"),
        h('p', { class: 'hint', style: 'margin:4px 0 0 0;' }, 'Extreme distance target lanes, CQB barricades, and zombie live-fire pens.')
      ),
      h('button', {
        style: 'padding:8px 16px; background:#b4540f; color:#fff7ea; border:none; border-radius:999px; font-weight:bold; cursor:pointer;',
        onclick: () => {
          g.teleportToShootingRange();
          this.toast('🎯 Arrived at Proving Grounds with full loadout & targets!');
          this.close();
        }
      }, '⚡ Teleport to Firing Line')
    );

    // Target stats overview
    const dummyEntities = [...g.ecs.query(Comp.TargetDummy, Comp.Transform)];
    const targetCards = dummyEntities.map((de) => {
      const dummy = g.ecs.get(de, Comp.TargetDummy)!;
      return h('div', {
        style: 'background:rgba(20,25,32,0.85); border:1px solid #334155; border-radius:8px; padding:10px; display:flex; flex-direction:column; gap:4px;'
      },
        h('div', { style: 'display:flex; justify-content:space-between; align-items:center;' },
          h('b', { style: 'color:#f8fafc; font-size:14px;' }, `🎯 ${dummy.distanceLabel}`),
          h('span', { style: 'font-size:11px; color:#94a3b8;' }, `Hits: ${dummy.totalHits}`)
        ),
        h('div', { style: 'font-size:12px; color:#cbd5e1;' },
          dummy.lastDamageTaken > 0
            ? `Last Hit: -${dummy.lastDamageTaken} DMG @ ${dummy.lastHitDistance}m`
            : 'No impacts recorded yet'
        )
      );
    });

    const lanesSection = h('div', { style: 'margin-bottom:14px;' },
      h('h3', { style: 'margin:0 0 8px 0; font-size:15px; color:#e2e8f0;' }, 'Target Lanes (5m - 100m Precision & CQB)'),
      h('div', { style: 'display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:8px; max-height:220px; overflow-y:auto; padding-right:4px;' },
        ...targetCards
      )
    );

    // Quick Action Bar
    const restatBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(30,35,45,0.9); border:1.5px solid #64748b; color:#e2e8f0; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        g.resetShootingRangeStats();
        this.toast('Target dummy impact logs reset to 0!');
        this.renderPanel();
      }
    }, '🔄 Reset Target Logs');

    const respawnTargetsBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(40,30,20,0.9); border:1.5px solid #f59e0b; color:#fde047; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        g.spawnShootingRangeTargets();
        this.toast('Steel gongs & silhouette targets re-anchored!');
        this.renderPanel();
      }
    }, '🎯 Re-anchor Steel Targets');

    const restockAmmoBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(20,40,30,0.9); border:1.5px solid #22c55e; color:#86efac; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        const inv = g.playerInv.inv;
        inv.add('rifle_ammo', 100);
        inv.add('shotgun_shells', 60);
        inv.add('scrap_nails', 150);
        inv.add('pebbles', 80);
        inv.add('wild_carrot', 25);
        this.toast('📦 Firing Line Ammo Crate Unpacked (+390 Rounds)!');
        this.renderHotbar();
      }
    }, '📦 Restock Ammo Crate');

    const openGunsmithBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(50,25,35,0.9); border:1.5px solid #f43f5e; color:#fecdd3; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        this.open('gunsmith');
      }
    }, '🛠️ Open Gunsmith [U]');

    const respawnZombiesBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(45,15,20,0.95); border:1.5px solid #ef4444; color:#fca5a5; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        g.spawnProvingGroundsZombies();
        this.toast('☣️ New mutants released into Quarantine Containment Pens!');
      }
    }, '☣️ Spawn Quarantine Mutants');

    const teleportDuelPitBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(30,58,138,0.95); border:1.5px solid #38bdf8; color:#bae6fd; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        g.teleportToArena('duel_pit');
        this.toast('🏟️ Entered Duel Pit (40x24m)! Mirrored cover & bullet stoppers active.');
        this.close();
      }
    }, '⚔️ Duel Pit (40x24m)');

    const teleportFfaYardBtn = h('button', {
      style: 'padding:8px 14px; background:rgba(88,28,135,0.95); border:1.5px solid #c084fc; color:#f3e8ff; border-radius:8px; cursor:pointer; font-weight:bold;',
      onclick: () => {
        g.teleportToArena('ffa_yard');
        this.toast('🏟️ Entered Free-for-all Yard (64x40m)! 8 Spawns & cover active.');
        this.close();
      }
    }, '💥 FFA Yard (64x40m)');

    const actionsStrip = h('div', { style: 'display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;' },
      teleportDuelPitBtn, teleportFfaYardBtn, restockAmmoBtn, openGunsmithBtn, respawnTargetsBtn, restatBtn, respawnZombiesBtn
    );

    return h('div', { style: 'padding:6px 0;' },
      arenaBanner,
      header,
      lanesSection,
      h('h3', { style: 'margin:12px 0 6px 0; font-size:15px; color:#e2e8f0;' }, 'Proving Grounds Operations'),
      actionsStrip
    );
  }

  private arenaFightPanel(): HTMLElement {
    const g = this.game;
    let selectedArena: 'duel_pit' | 'ffa_yard' = g.activeArena?.def.id ?? 'duel_pit';
    let selectedBotLevel: 'scarecrow' | 'regular' | 'sharp' | 'menace' = 'regular';
    let selectedBotWeapon = 'nailgun_smg';
    let selectedZombieKind = 'walker';
    let selectedZombieCount = 4;

    const container = h('div', { class: 'arena-fight-panel', style: 'padding:4px 0;' });

    const render = (): void => {
      container.replaceChildren();

      // Top Header
      const header = h('div', {
        style: 'display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:10px; margin-bottom:12px;'
      },
        h('div', {},
          h('h2', { style: 'margin:0; color:#f97316; font-family:var(--display); font-size:22px;' }, '🏟️ Army Base Combat Arena & Fight Bench'),
          h('p', { class: 'hint', style: 'margin:4px 0 0 0; color:#94a3b8;' }, 'Army Base live-fire tactical proving grounds. Pick bot duels, zombie containment swarms, or 3-way skirmishes.')
        ),
        h('div', { style: 'display:flex; gap:6px;' },
          h('button', {
            style: 'padding:7px 12px; background:rgba(30,41,59,0.9); border:1px solid #475569; color:#f1f5f9; border-radius:6px; cursor:pointer; font-weight:bold; font-size:12px;',
            onclick: () => {
              g.clearArenaCombatants();
              this.toast('🧹 Cleared all combatants from the arena.');
              render();
            }
          }, '🧹 Clear Arena'),
          h('button', {
            style: 'padding:7px 12px; background:rgba(34,197,94,0.2); border:1px solid #22c55e; color:#86efac; border-radius:6px; cursor:pointer; font-weight:bold; font-size:12px;',
            onclick: () => {
              g.playerNeeds.health = 100;
              const inv = g.playerInv.inv;
              inv.add('rifle_ammo', 150);
              inv.add('shotgun_shells', 100);
              inv.add('scrap_nails', 250);
              inv.add('wild_carrot', 30);
              this.toast('📦 Fully restocked ammo and healed to 100%!');
              this.renderHotbar();
              render();
            }
          }, '📦 Restock & Heal')
        )
      );

      // Arena Environment Selector
      const arenaSelectRow = h('div', {
        style: 'background:rgba(15,23,42,0.8); border:1.5px solid #1e293b; border-radius:8px; padding:10px 14px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'
      },
        h('div', {},
          h('b', { style: 'color:#e2e8f0; font-size:14px; display:block;' }, '📍 Select Arena Battleground (Army Base)'),
          h('span', { style: 'color:#94a3b8; font-size:12px;' },
            selectedArena === 'duel_pit' ? 'Duel Pit (40x24m): 2 spawns, half-turn symmetry, tactical crate/sandbag cover.' : 'FFA Yard (64x40m): 8 spawns, open crossfires, bunker corridors & perimeter walls.'
          )
        ),
        h('div', { style: 'display:flex; gap:8px;' },
          h('button', {
            style: `padding:8px 14px; border-radius:6px; font-weight:bold; cursor:pointer; font-size:13px; border:${selectedArena === 'duel_pit' ? '2px solid #38bdf8; background:#0284c7; color:#fff;' : '1px solid #334155; background:rgba(30,41,59,0.7); color:#cbd5e1;'}`,
            onclick: () => { selectedArena = 'duel_pit'; render(); }
          }, '⚔️ Duel Pit (40x24m)'),
          h('button', {
            style: `padding:8px 14px; border-radius:6px; font-weight:bold; cursor:pointer; font-size:13px; border:${selectedArena === 'ffa_yard' ? '2px solid #a855f7; background:#7e22ce; color:#fff;' : '1px solid #334155; background:rgba(30,41,59,0.7); color:#cbd5e1;'}`,
            onclick: () => { selectedArena = 'ffa_yard'; render(); }
          }, '💥 FFA Yard (64x40m)')
        )
      );

      // Live Arena Combatants Status Bar
      const liveBots = [...g.ecs.query(Comp.Raider, Comp.Character)];
      const arDef = g.activeArena;
      let liveZombies = 0;
      if (arDef) {
        for (const ze of g.ecs.query(Comp.Zombie, Comp.Transform)) {
          const zt = g.ecs.get(ze, Comp.Transform)!;
          if (zt.x >= arDef.originX - 2 && zt.x <= arDef.originX + arDef.def.width + 2 && zt.y >= arDef.originY - 2 && zt.y <= arDef.originY + arDef.def.height + 2) {
            liveZombies++;
          }
        }
      }

      const statusCard = h('div', {
        style: 'background:rgba(2,6,23,0.7); border:1px solid #1e293b; border-radius:6px; padding:8px 12px; margin-bottom:12px; display:flex; justify-content:space-around; align-items:center; font-size:13px; color:#cbd5e1;'
      },
        h('span', {}, `🏟️ Active: `, h('b', { style: 'color:#38bdf8;' }, g.activeArena ? `${g.activeArena.def.name} (${g.activeArena.def.width}x${g.activeArena.def.height}m)` : 'None (click match to enter)')),
        h('span', {}, `🤖 Active Bots: `, h('b', { style: 'color:#f87171;' }, String(liveBots.length))),
        h('span', {}, `🧟 Arena Zombies: `, h('b', { style: 'color:#4ade80;' }, String(liveZombies))),
        h('span', {}, `❤️ Your Health: `, h('b', { style: 'color:#facc15;' }, `${Math.round(g.playerNeeds.health)} / 100`))
      );

      // BOT GUN FIGHT SECTION
      const botTiers: { id: 'scarecrow' | 'regular' | 'sharp' | 'menace'; title: string; desc: string; hp: number; spread: string; color: string }[] = [
        { id: 'scarecrow', title: '🌾 Scarecrow', desc: 'Slow advance, wide bullet spread, ideal for target practice', hp: 50, spread: '±12°', color: '#a3e635' },
        { id: 'regular', title: '🎖️ Regular', desc: 'Balanced strafe, medium reflexes, standard combatant', hp: 65, spread: '±7°', color: '#38bdf8' },
        { id: 'sharp', title: '⚡ Sharp', desc: 'Agile kiter, rapid target acquisition, tight bullet grouping', hp: 80, spread: '±3.5°', color: '#f59e0b' },
        { id: 'menace', title: '💀 Menace', desc: 'Aggressive rusher, deadly laser reflex, near-zero spread', hp: 100, spread: '±1.5°', color: '#ef4444' }
      ];

      const botCards = botTiers.map((t) => {
        const isSel = selectedBotLevel === t.id;
        return h('div', {
          style: `flex:1; min-width:140px; padding:10px; border-radius:8px; cursor:pointer; border:${isSel ? `2px solid ${t.color}` : '1px solid #334155'}; background:${isSel ? 'rgba(30,41,59,0.9)' : 'rgba(15,23,42,0.6)'}; display:flex; flex-direction:column; gap:4px;`,
          onclick: () => { selectedBotLevel = t.id; render(); }
        },
          h('b', { style: `color:${t.color}; font-size:14px;` }, t.title),
          h('span', { style: 'font-size:11px; color:#94a3b8;' }, t.desc),
          h('div', { style: 'margin-top:auto; font-size:11px; color:#cbd5e1; display:flex; justify-content:space-between;' },
            h('span', {}, `🛡️ ${t.hp} HP`),
            h('span', {}, `🎯 Spread ${t.spread}`)
          )
        );
      });

      const botWeapons = [
        { id: 'nailgun_smg', label: '🔫 Nailgun SMG', note: 'Rapid full-auto spray' },
        { id: 'pump_shotgun', label: '💥 Pump Shotgun', note: 'Heavy multi-pellet blast' },
        { id: 'hunting_rifle', label: '🎯 Hunting Rifle', note: 'High damage sniper' }
      ];

      const weaponBtns = botWeapons.map((w) => {
        const isSel = selectedBotWeapon === w.id;
        return h('button', {
          style: `padding:6px 12px; border-radius:6px; font-weight:bold; cursor:pointer; font-size:12px; border:${isSel ? '1.5px solid #f97316; background:#c2410c; color:#fff;' : '1px solid #334155; background:rgba(30,41,59,0.7); color:#cbd5e1;'}`,
          onclick: () => { selectedBotWeapon = w.id; render(); }
        }, w.label);
      });

      const botSection = h('div', {
        style: 'background:rgba(15,23,42,0.7); border:1px solid #334155; border-radius:8px; padding:12px; margin-bottom:12px;'
      },
        h('div', { style: 'display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;' },
          h('h3', { style: 'margin:0; font-size:15px; color:#f8fafc;' }, '🤖 Bot Gun Fight (Pick Difficulty & Loadout)'),
          h('div', { style: 'display:flex; gap:6px;' }, ...weaponBtns)
        ),
        h('div', { style: 'display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;' }, ...botCards),
        h('div', { style: 'display:flex; gap:8px; justify-content:flex-end;' },
          h('button', {
            style: 'padding:8px 14px; background:rgba(51,65,85,0.8); border:1px solid #64748b; color:#f1f5f9; border-radius:6px; font-weight:bold; cursor:pointer; font-size:12px;',
            onclick: () => {
              g.spawnArenaBot(selectedBotLevel, selectedBotWeapon, Math.floor(Math.random() * 8));
              this.toast(`🤖 Added ${selectedBotLevel.toUpperCase()} Bot to Arena!`);
              render();
            }
          }, '➕ Add Bot to Arena'),
          h('button', {
            style: 'padding:9px 18px; background:linear-gradient(135deg, #ea580c 0%, #c2410c 100%); border:none; color:#fff; border-radius:6px; font-weight:bold; cursor:pointer; font-size:13px; box-shadow:0 2px 6px rgba(194,65,12,0.4);',
            onclick: () => {
              g.teleportToArena(selectedArena, 0);
              g.clearArenaCombatants();
              g.spawnArenaBot(selectedBotLevel, selectedBotWeapon, 1);
              this.toast(`⚔️ 1v1 Bot Duel started against ${selectedBotLevel.toUpperCase()} Bot in ${selectedArena === 'duel_pit' ? 'Duel Pit' : 'FFA Yard'}!`);
              this.close();
            }
          }, `⚔️ Start 1v1 Gun Duel vs ${selectedBotLevel.toUpperCase()}`)
        )
      );

      // ZOMBIE SWARM COMBAT SECTION
      const zombieKinds: { id: string; label: string; desc: string; color: string }[] = [
        { id: 'walker', label: '🧟 Walker', desc: 'Standard slow horde infected', color: '#4ade80' },
        { id: 'runner', label: '🏃 Runner', desc: 'Feral sprinters that chase relentlessly', color: '#facc15' },
        { id: 'brute', label: '🦍 Brute', desc: 'Massive armored heavyweight juggernaut', color: '#f87171' },
        { id: 'screamer', label: '😱 Screamer', desc: 'Sonic shrieker alerting all nearby infected', color: '#c084fc' }
      ];

      const zombieCards = zombieKinds.map((zk) => {
        const isSel = selectedZombieKind === zk.id;
        return h('button', {
          style: `flex:1; min-width:110px; padding:8px 10px; border-radius:6px; text-align:left; cursor:pointer; border:${isSel ? `2px solid ${zk.color}` : '1px solid #334155'}; background:${isSel ? 'rgba(30,41,59,0.95)' : 'rgba(15,23,42,0.6)'};`,
          onclick: () => { selectedZombieKind = zk.id; render(); }
        },
          h('b', { style: `color:${zk.color}; font-size:13px; display:block;` }, zk.label),
          h('span', { style: 'font-size:11px; color:#94a3b8;' }, zk.desc)
        );
      });

      const countOptions = [1, 3, 6, 12];
      const countBtns = countOptions.map((c) => {
        const isSel = selectedZombieCount === c;
        return h('button', {
          style: `padding:5px 12px; border-radius:6px; font-weight:bold; cursor:pointer; font-size:12px; border:${isSel ? '1.5px solid #22c55e; background:#16a34a; color:#fff;' : '1px solid #334155; background:rgba(30,41,59,0.7); color:#cbd5e1;'}`,
          onclick: () => { selectedZombieCount = c; render(); }
        }, `${c}x`);
      });

      const zombieSection = h('div', {
        style: 'background:rgba(15,23,42,0.7); border:1px solid #334155; border-radius:8px; padding:12px; margin-bottom:12px;'
      },
        h('div', { style: 'display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;' },
          h('h3', { style: 'margin:0; font-size:15px; color:#f8fafc;' }, '🧟 Zombie Horde Combat (Quarantine Mutants)'),
          h('div', { style: 'display:flex; gap:6px;' }, ...countBtns)
        ),
        h('div', { style: 'display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;' }, ...zombieCards),
        h('div', { style: 'display:flex; gap:8px; justify-content:flex-end;' },
          h('button', {
            style: 'padding:9px 18px; background:linear-gradient(135deg, #15803d 0%, #166534 100%); border:none; color:#fff; border-radius:6px; font-weight:bold; cursor:pointer; font-size:13px; box-shadow:0 2px 6px rgba(22,101,52,0.4);',
            onclick: () => {
              if (!g.activeArena) g.teleportToArena(selectedArena, 0);
              g.spawnArenaZombies(selectedZombieCount, selectedZombieKind);
              this.toast(`🧟 Released ${selectedZombieCount} ${selectedZombieKind.toUpperCase()} mutants into Arena!`);
              this.close();
            }
          }, `🧟 Release ${selectedZombieCount}x ${selectedZombieKind.toUpperCase()} Horde`)
        )
      );

      // THREE-WAY CHAOS & PROVING GROUND OPERATIONS
      const operationsSection = h('div', {
        style: 'background:rgba(15,23,42,0.7); border:1px solid #334155; border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'
      },
        h('div', {},
          h('b', { style: 'color:#f8fafc; font-size:14px; display:block;' }, '🔥 Multi-Faction War & Army Operations'),
          h('span', { style: 'color:#94a3b8; font-size:12px;' }, 'Pit armed raider bots against quarantined mutants in an all-out 3-way skirmish.')
        ),
        h('div', { style: 'display:flex; gap:8px; flex-wrap:wrap;' },
          h('button', {
            style: 'padding:9px 16px; background:linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%); border:none; color:#fff; border-radius:6px; font-weight:bold; cursor:pointer; font-size:13px; box-shadow:0 2px 6px rgba(185,28,28,0.4);',
            onclick: () => {
              g.teleportToArena(selectedArena, 0);
              g.clearArenaCombatants();
              g.spawnArenaBot('regular', 'nailgun_smg', 1);
              g.spawnArenaBot('sharp', 'pump_shotgun', selectedArena === 'ffa_yard' ? 4 : 1);
              g.spawnArenaZombies(6, 'walker');
              g.spawnArenaZombies(2, 'runner');
              this.toast('🔥 3-Way Battle Royale started! Bots, zombies, and player in open war!');
              this.close();
            }
          }, '🔥 Start 3-Way War (Bots vs Zombies)'),
          h('button', {
            style: 'padding:8px 14px; background:rgba(30,41,59,0.9); border:1px solid #475569; color:#f1f5f9; border-radius:6px; cursor:pointer; font-weight:bold; font-size:12px;',
            onclick: () => {
              this.open('range');
            }
          }, '🎯 Ballistics Firing Line [Y]'),
          h('button', {
            style: 'padding:8px 14px; background:rgba(30,41,59,0.9); border:1px solid #475569; color:#f1f5f9; border-radius:6px; cursor:pointer; font-weight:bold; font-size:12px;',
            onclick: () => {
              this.open('gunsmith');
            }
          }, '🛠️ Open Gunsmith [U]')
        )
      );

      container.append(header, arenaSelectRow, statusCard, botSection, zombieSection, operationsSection);
    };

    render();
    return container;
  }

  private devLogPanel(): HTMLElement {
    const g = this.game;
    let activeTab: 'all' | 'bugs' | 'combat' | 'sim' | 'perf' | 'gemini' = 'all';
    let searchQuery = '';

    const container = h('div', { class: 'devlog-panel', style: 'padding:4px 0;' });

    const render = (): void => {
      container.replaceChildren();

      const bugsCount = devLog.getBugCount();
      const allLogs = devLog.getLogs();

      // Top Header
      const header = h('div', {
        style: 'display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:10px; margin-bottom:10px; flex-wrap:wrap; gap:8px;'
      },
        h('div', {},
          h('div', { style: 'display:flex; align-items:center; gap:8px;' },
            h('h2', { style: 'margin:0; color:#38bdf8; font-family:var(--display); font-size:22px;' }, '🐛 Runtime Dev Log & Bug Engine'),
            bugsCount > 0
              ? h('span', { style: 'background:#ef4444; color:#fff; font-size:11px; font-weight:bold; padding:2px 8px; border-radius:999px;' }, `🔴 ${bugsCount} BUGS`)
              : h('span', { style: 'background:#22c55e; color:#fff; font-size:11px; font-weight:bold; padding:2px 8px; border-radius:999px;' }, '🟢 0 BUGS')
          ),
          h('p', { class: 'hint', style: 'margin:4px 0 0 0; color:#94a3b8;' }, 'Real-time telemetry, uncaught bug trapping, combat traces, and one-click bug export.')
        ),
        h('div', { style: 'display:flex; gap:6px; flex-wrap:wrap;' },
          h('button', {
            style: 'padding:6px 12px; background:linear-gradient(135deg, #0284c7 0%, #0369a1 100%); border:none; color:#fff; border-radius:6px; font-weight:bold; font-size:12px; cursor:pointer;',
            onclick: () => {
              const snapshot = devLog.generateBugReport({
                seed: g.config.seed,
                mode: g.config.mode,
                playerPos: { x: g.playerT.x, y: g.playerT.y },
                playerHealth: g.playerNeeds.health,
                activeArena: g.activeArena?.def.name ?? null,
                entityCounts: {
                  total: g.ecs.count(),
                  zombies: [...g.ecs.query(Comp.Zombie)].length,
                  bots: [...g.ecs.query(Comp.Raider)].length,
                  projectiles: [...g.ecs.query(Comp.Projectile)].length,
                  targetDummies: [...g.ecs.query(Comp.TargetDummy)].length,
                },
                fps: this.renderer.stats.drawn,
                simTime: g.simTime,
              });
              const md = devLog.formatMarkdownReport(snapshot);
              if (typeof navigator !== 'undefined' && navigator.clipboard) {
                navigator.clipboard.writeText(md).then(() => {
                  this.toast('📋 Copied full Markdown Bug Report to clipboard!');
                }).catch(() => {
                  this.toast('Report dumped to browser console!');
                  console.log(md);
                });
              } else {
                console.log(md);
                this.toast('Report dumped to browser console!');
              }
            }
          }, '📋 Copy Bug Report (MD)'),
          h('button', {
            style: 'padding:6px 12px; background:rgba(30,41,59,0.9); border:1px solid #475569; color:#f1f5f9; border-radius:6px; font-weight:bold; font-size:12px; cursor:pointer;',
            onclick: () => {
              const snapshot = devLog.generateBugReport({
                seed: g.config.seed,
                mode: g.config.mode,
                playerPos: { x: g.playerT.x, y: g.playerT.y },
                playerHealth: g.playerNeeds.health,
                activeArena: g.activeArena?.def.name ?? null,
                entityCounts: {
                  total: g.ecs.count(),
                  zombies: [...g.ecs.query(Comp.Zombie)].length,
                  bots: [...g.ecs.query(Comp.Raider)].length,
                  projectiles: [...g.ecs.query(Comp.Projectile)].length,
                  targetDummies: [...g.ecs.query(Comp.TargetDummy)].length,
                },
                fps: this.renderer.stats.drawn,
                simTime: g.simTime,
              });
              const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `world4bros-diagnostic-${Date.now()}.json`;
              a.click();
              URL.revokeObjectURL(url);
              this.toast('💾 Downloaded diagnostic JSON snapshot!');
            }
          }, '💾 Export JSON'),
          h('button', {
            style: 'padding:6px 10px; background:rgba(239,68,68,0.15); border:1px solid #ef4444; color:#fca5a5; border-radius:6px; font-weight:bold; font-size:12px; cursor:pointer;',
            onclick: () => {
              devLog.bug('system', 'Simulated test exception for dev log verification', { simulated: true, code: 404 });
              this.toast('🧪 Logged test bug event!');
              render();
            }
          }, '🧪 Test Bug'),
          h('button', {
            style: `padding:6px 10px; border-radius:6px; font-weight:bold; font-size:12px; cursor:pointer; border:1px solid #475569; background:${devLog.getPaused() ? '#eab308; color:#000;' : 'rgba(30,41,59,0.9); color:#cbd5e1;'}`,
            onclick: () => {
              const p = devLog.togglePause();
              this.toast(p ? '⏸️ Log stream paused' : '▶️ Log stream resumed');
              render();
            }
          }, devLog.getPaused() ? '▶️ Resume' : '⏸️ Pause'),
          h('button', {
            style: 'padding:6px 10px; background:rgba(30,41,59,0.9); border:1px solid #475569; color:#cbd5e1; border-radius:6px; font-weight:bold; font-size:12px; cursor:pointer;',
            onclick: () => {
              devLog.clear();
              this.toast('🧹 Cleared log stream');
              render();
            }
          }, '🧹 Clear')
        )
      );

      // Tab Bar & Search Filter
      const tabs: { id: typeof activeTab; label: string; count?: number }[] = [
        { id: 'all', label: 'All Logs', count: allLogs.length },
        { id: 'bugs', label: '🔴 Bugs & Errors', count: bugsCount },
        { id: 'combat', label: '⚔️ Combat' },
        { id: 'sim', label: '🌱 Sim & World' },
        { id: 'perf', label: '⚡ Performance' },
        { id: 'gemini', label: '✨ Gemini Asset Prompts' }
      ];

      const tabBtns = tabs.map((t) => {
        const isSel = activeTab === t.id;
        return h('button', {
          style: `padding:6px 12px; border-radius:6px; font-weight:bold; font-size:12px; cursor:pointer; border:${isSel ? '1.5px solid #38bdf8; background:#0369a1; color:#fff;' : '1px solid #334155; background:rgba(15,23,42,0.7); color:#cbd5e1;'}`,
          onclick: () => { activeTab = t.id; render(); }
        }, `${t.label}${t.count !== undefined ? ` (${t.count})` : ''}`);
      });

      const searchInput = h('input', {
        type: 'text',
        placeholder: '🔍 Filter logs by text or category...',
        value: searchQuery,
        style: 'flex:1; min-width:200px; padding:6px 10px; border-radius:6px; border:1px solid #334155; background:rgba(2,6,23,0.8); color:#f8fafc; font-size:12px;',
        oninput: (e) => {
          searchQuery = (e.target as HTMLInputElement).value;
          renderLogRows();
        }
      });

      const filterBar = h('div', {
        style: 'display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap;'
      },
        h('div', { style: 'display:flex; gap:6px; flex-wrap:wrap;' }, ...tabBtns),
        searchInput
      );

      const contentBox = h('div', { style: 'min-height:300px;' });

      const renderLogRows = (): void => {
        if (activeTab === 'gemini') {
          contentBox.replaceChildren(this.renderGeminiPromptTab());
          return;
        }

        contentBox.replaceChildren();

        let filtered = devLog.getLogs({ search: searchQuery });
        if (activeTab === 'bugs') {
          filtered = filtered.filter((l) => l.level === 'bug' || l.level === 'error');
        } else if (activeTab === 'combat') {
          filtered = filtered.filter((l) => l.category === 'combat' || l.level === 'combat');
        } else if (activeTab === 'sim') {
          filtered = filtered.filter((l) => l.category === 'world' || l.category === 'zombie' || l.category === 'bot');
        } else if (activeTab === 'perf') {
          filtered = filtered.filter((l) => l.level === 'perf' || l.category === 'system');
        }

        if (filtered.length === 0) {
          contentBox.append(
            h('div', {
              style: 'padding:40px 20px; text-align:center; color:#64748b; font-size:13px; background:rgba(2,6,23,0.6); border:1px solid #1e293b; border-radius:8px;'
            }, 'No log entries match the current filter.')
          );
          return;
        }

        const rowsContainer = h('div', {
          style: 'max-height:360px; overflow-y:auto; background:rgba(2,6,23,0.85); border:1px solid #1e293b; border-radius:8px; font-family:var(--note), monospace; font-size:12px;'
        });

        // Show latest logs first
        const reversed = [...filtered].reverse();
        for (const log of reversed) {
          const levelColor = log.level === 'bug' ? '#ef4444'
            : log.level === 'error' ? '#f87171'
            : log.level === 'warn' ? '#facc15'
            : log.level === 'combat' ? '#f97316'
            : log.level === 'sim' ? '#4ade80'
            : log.level === 'perf' ? '#c084fc'
            : '#38bdf8';

          const row = h('div', {
            style: `padding:6px 10px; border-bottom:1px solid #1e293b; display:flex; flex-direction:column; gap:2px; ${log.level === 'bug' || log.level === 'error' ? 'background:rgba(239,68,68,0.08);' : ''}`
          },
            h('div', { style: 'display:flex; align-items:center; gap:8px;' },
              h('span', { style: 'color:#64748b; font-size:11px;' }, log.timestamp),
              h('span', { style: `color:${levelColor}; font-weight:bold; font-size:11px; padding:1px 6px; border-radius:4px; border:1px solid ${levelColor}44;` }, log.level.toUpperCase()),
              h('span', { style: 'color:#94a3b8; font-size:11px;' }, `[${log.category}]`),
              h('span', { style: 'color:#f8fafc; flex:1; word-break:break-word;' }, log.message)
            )
          );

          if (log.data || log.stack) {
            const details = h('pre', {
              style: 'margin:4px 0 0 0; padding:6px 8px; background:rgba(15,23,42,0.9); border-radius:4px; font-size:11px; color:#cbd5e1; overflow-x:auto; max-height:120px;'
            }, log.stack ? `${log.stack}\nData: ${JSON.stringify(log.data, null, 2)}` : JSON.stringify(log.data, null, 2));
            row.append(details);
          }

          rowsContainer.append(row);
        }

        contentBox.append(rowsContainer);
      };

      renderLogRows();
      container.append(header, filterBar, contentBox);
    };

    render();
    return container;
  }

  private renderGeminiPromptTab(): HTMLElement {
    const wrap = h('div', { style: 'display:flex; flex-direction:column; gap:12px; max-height:380px; overflow-y:auto; padding-right:4px;' });

    const intro = h('div', {
      style: 'background:rgba(88,28,135,0.25); border:1px solid #a855f7; border-radius:8px; padding:10px 14px; color:#e9d5ff; font-size:13px;'
    },
      h('b', { style: 'color:#f3e8ff; font-size:14px; display:block; margin-bottom:2px;' }, '✨ Gemini Asset & Content Generation Recipes'),
      'Copy these tailored prompts directly into Gemini or the image generator to create pixel art wear, items, objects, terrain autotiles, and effects matching World4Bros engine specifications.'
    );

    wrap.append(intro);

    for (const recipe of GEMINI_ASSET_RECIPES) {
      const card = h('div', {
        style: 'background:rgba(15,23,42,0.85); border:1px solid #334155; border-radius:8px; padding:12px; display:flex; flex-direction:column; gap:6px;'
      },
        h('div', { style: 'display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;' },
          h('div', {},
            h('b', { style: 'color:#f8fafc; font-size:14px; display:block;' }, recipe.title),
            h('span', { style: 'font-size:11px; color:#94a3b8;' }, recipe.shortDesc)
          ),
          h('div', { style: 'display:flex; gap:6px;' },
            h('button', {
              style: 'padding:4px 10px; background:#0284c7; color:#fff; border:none; border-radius:4px; font-weight:bold; font-size:11px; cursor:pointer;',
              onclick: () => {
                if (typeof navigator !== 'undefined' && navigator.clipboard) {
                  navigator.clipboard.writeText(recipe.prompt).then(() => this.toast(`Copied prompt for ${recipe.title}!`));
                }
              }
            }, '📋 Copy Prompt'),
            h('button', {
              style: 'padding:4px 10px; background:rgba(51,65,85,0.8); color:#cbd5e1; border:1px solid #64748b; border-radius:4px; font-weight:bold; font-size:11px; cursor:pointer;',
              onclick: () => {
                if (typeof navigator !== 'undefined' && navigator.clipboard) {
                  navigator.clipboard.writeText(recipe.schemaExample).then(() => this.toast(`Copied code schema for ${recipe.title}!`));
                }
              }
            }, '📋 Copy Code Schema')
          )
        ),
        h('pre', {
          style: 'margin:4px 0 0 0; padding:8px 10px; background:rgba(2,6,23,0.9); border:1px solid #1e293b; border-radius:6px; font-family:var(--note), monospace; font-size:11px; color:#93c5fd; white-space:pre-wrap; word-break:break-word; max-height:100px; overflow-y:auto;'
        }, recipe.prompt)
      );

      wrap.append(card);
    }

    return wrap;
  }

  private swarmlabPanel(): HTMLElement {
    const defaultCfg = defaultSwarmCfg();
    let currentCfg: SwarmSimCfg = { ...defaultCfg };
    let lastSummary: SwarmSimSummary | null = null;
    let comparisonResults: Record<string, SwarmSimSummary[]> | null = null;
    let isRunningSim = false;

    const container = h('div', { style: 'padding: 4px 0;' });

    const renderLab = (): void => {
      container.replaceChildren();

      // Header
      const header = h('div', { style: 'display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:12px;' },
        h('div', {},
          h('h2', { style: 'margin:0; color:#e07a5f; font-family:var(--display); font-size:22px;' }, '🧪 Swarm Testing & Telemetry Lab'),
          h('p', { class: 'hint', style: 'margin:4px 0 0 0;' }, 'Deterministic headless combat simulation, noise pressure logging, and paired-seed weapon balance matrix.')
        ),
        h('span', { style: 'background:rgba(224,122,95,0.2); border:1px solid #e07a5f; padding:4px 10px; border-radius:999px; font-weight:bold; font-size:12px; color:#f8fafc;' },
          isRunningSim ? '⚙️ SIMULATING...' : 'READY'
        )
      );

      // Controls Strip
      const modeSelect = h('select', {
        style: 'background:var(--field); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:4px 8px;',
        onchange: (e: Event) => {
          currentCfg.mode = (e.target as HTMLSelectElement).value as 'pve' | 'pvpve';
        }
      },
        h('option', { value: 'pvpve', selected: currentCfg.mode === 'pvpve' }, 'PvPvE (Bros vs Raiders vs Horde)'),
        h('option', { value: 'pve', selected: currentCfg.mode === 'pve' }, 'PvE (Bros vs Horde)')
      );

      const patternSelect = h('select', {
        style: 'background:var(--field); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:4px 8px;',
        onchange: (e: Event) => {
          currentCfg.pattern = (e.target as HTMLSelectElement).value as never;
        }
      },
        h('option', { value: 'waves', selected: currentCfg.pattern === 'waves' }, 'Waves (6 Waves, 8s interval)'),
        h('option', { value: 'sleepers', selected: currentCfg.pattern === 'sleepers' }, 'Sleepers (Wakes on Gunfire Noise)'),
        h('option', { value: 'horde', selected: currentCfg.pattern === 'horde' }, 'North Horde Rush'),
        h('option', { value: 'ring', selected: currentCfg.pattern === 'ring' }, 'Perimeter Ring Encirclement')
      );

      const weaponSelect = h('select', {
        style: 'background:var(--field); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:4px 8px;',
        onchange: (e: Event) => {
          currentCfg.broWeapon = (e.target as HTMLSelectElement).value;
        }
      },
        ...Object.values(SIM_WEAPONS).map((w) =>
          h('option', { value: w.key, selected: currentCfg.broWeapon === w.key }, `${w.name} (${w.dmg} DMG)`)
        )
      );

      const runSingleBtn = h('button', {
        style: 'padding:7px 14px; background:#b4540f; color:#fff7ea; border:none; border-radius:8px; font-weight:bold; cursor:pointer;',
        onclick: async () => {
          isRunningSim = true;
          renderLab();
          // Run 45s battle headless
          const sim = new SwarmSim({ ...currentCfg, timeLimit: 45 }, 7);
          while (!sim.over) {
            sim.step(1 / 30);
          }
          lastSummary = sim.summary();
          isRunningSim = false;
          this.toast(`Simulation finished: ${lastSummary.outcome.toUpperCase()} (${lastSummary.kills} Kills, ${lastSummary.accuracy * 100}% Accuracy)`);
          renderLab();
        }
      }, '⚡ Run Single 45s Run (Seed 7)');

      const runBenchmarkBtn = h('button', {
        style: 'padding:7px 14px; background:rgba(30,55,40,0.9); border:1.5px solid #22c55e; color:#86efac; border-radius:8px; font-weight:bold; cursor:pointer;',
        onclick: async () => {
          isRunningSim = true;
          renderLab();
          const testGuns = ['hunting_rifle', 'pump_shotgun', 'nailgun_smg', 'plinker_pistol', 'potato_cannon', 'slingshot'];
          comparisonResults = await compareWeaponsPaired(testGuns, { ...currentCfg, timeLimit: 45 }, 5, 4);
          isRunningSim = false;
          this.toast('🎯 Multi-Gun Benchmark Matrix Complete (4 Paired Runs Each)!');
          renderLab();
        }
      }, '📊 Run 6-Gun Balance Matrix (4 Runs Each)');

      const cfgBar = h('div', { style: 'display:flex; flex-wrap:wrap; gap:8px; align-items:center; background:rgba(20,25,32,0.85); padding:10px; border-radius:8px; margin-bottom:12px; border:1px solid #334155;' },
        h('span', { style: 'font-weight:bold; font-size:12px; color:#94a3b8;' }, 'Scenario Config:'),
        modeSelect, patternSelect, weaponSelect, runSingleBtn, runBenchmarkBtn
      );

      // Latest Run Summary Card
      let summarySection: HTMLElement;
      if (lastSummary) {
        const s = lastSummary;
        const outcomeCol = s.outcome === 'cleared' ? '#4ade80' : s.outcome === 'wiped' ? '#f87171' : '#facc15';
        summarySection = h('div', { style: 'background:rgba(25,30,40,0.9); border:1px solid #475569; border-radius:8px; padding:12px; margin-bottom:12px;' },
          h('div', { style: 'display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;' },
            h('b', { style: 'color:#f8fafc; font-size:15px;' }, `Simulation Telemetry — Outcome: `),
            h('span', { style: `color:${outcomeCol}; font-weight:bold; font-size:14px; text-transform:uppercase;` }, s.outcome)
          ),
          h('div', { style: 'display:grid; grid-template-columns:repeat(auto-fill, minmax(130px, 1fr)); gap:8px;' },
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Kills'), h('b', {}, String(s.kills))),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Kills / Min'), h('b', {}, `${s.killsPerMin}`)),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Hit Accuracy'), h('b', {}, `${Math.round(s.accuracy * 100)}%`)),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Total Damage'), h('b', {}, `${s.dmgDealt}`)),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Dmg / Shot'), h('b', {}, `${s.dmgPerShot}`)),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Avg Kill Dist'), h('b', {}, s.avgKillDist ? `${s.avgKillDist}m` : '–')),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Avg Time-To-Kill'), h('b', {}, s.avgTTK ? `${s.avgTTK}s` : '–')),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Noise Per Kill'), h('b', {}, s.noisePerKill ? `${s.noisePerKill}` : '–')),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Zombies Woken'), h('b', {}, s.alertsPerKill ? `${s.alertsPerKill}` : '–')),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Bro Damage Taken'), h('b', {}, `${s.broDmgTaken}`)),
            h('div', { class: 'stat-box' }, h('span', { class: 'hint' }, 'Bros Lost'), h('b', {}, `${s.broDeaths}`))
          )
        );
      } else {
        summarySection = h('div', { style: 'background:rgba(20,25,35,0.7); border:1px dashed #334155; border-radius:8px; padding:16px; text-align:center; color:#94a3b8; margin-bottom:12px;' },
          'No run telemetry recorded yet. Click "Run Single 45s Run" or "Run 6-Gun Balance Matrix" above to execute.'
        );
      }

      // Comparison Matrix Table
      let matrixSection: HTMLElement;
      if (comparisonResults) {
        const rows = Object.entries(comparisonResults).map(([gunKey, runs]) => {
          const wDef = SIM_WEAPONS[gunKey];
          const avgKills = runs.reduce((a, b) => a + b.kills, 0) / runs.length;
          const avgAcc = runs.reduce((a, b) => a + b.accuracy, 0) / runs.length;
          const avgDmg = runs.reduce((a, b) => a + b.dmgDealt, 0) / runs.length;
          const avgDist = runs.filter((r) => r.avgKillDist !== null).map((r) => r.avgKillDist!);
          const meanDist = avgDist.length ? avgDist.reduce((a, b) => a + b, 0) / avgDist.length : 0;
          const avgTTK = runs.filter((r) => r.avgTTK !== null).map((r) => r.avgTTK!);
          const meanTTK = avgTTK.length ? avgTTK.reduce((a, b) => a + b, 0) / avgTTK.length : 0;
          const avgNoise = runs.filter((r) => r.noisePerKill !== null).map((r) => r.noisePerKill!);
          const meanNoise = avgNoise.length ? avgNoise.reduce((a, b) => a + b, 0) / avgNoise.length : 0;
          const clears = runs.filter((r) => r.outcome === 'cleared').length;
          const wipes = runs.filter((r) => r.outcome === 'wiped').length;

          return h('tr', { style: 'border-bottom:1px solid #334155;' },
            h('td', { style: 'padding:6px; font-weight:bold; color:#f8fafc;' }, wDef ? wDef.name : gunKey),
            h('td', { style: 'padding:6px; text-align:center;' }, String(runs.length)),
            h('td', { style: 'padding:6px; text-align:center; color:#facc15; font-weight:bold;' }, avgKills.toFixed(1)),
            h('td', { style: 'padding:6px; text-align:center;' }, `${Math.round(avgAcc * 100)}%`),
            h('td', { style: 'padding:6px; text-align:center;' }, Math.round(avgDmg).toString()),
            h('td', { style: 'padding:6px; text-align:center;' }, `${meanDist.toFixed(1)}m`),
            h('td', { style: 'padding:6px; text-align:center;' }, `${meanTTK.toFixed(2)}s`),
            h('td', { style: 'padding:6px; text-align:center;' }, Math.round(meanNoise).toString()),
            h('td', { style: 'padding:6px; text-align:center; color:#4ade80;' }, `${Math.round((clears / runs.length) * 100)}%`),
            h('td', { style: 'padding:6px; text-align:center; color:#f87171;' }, `${Math.round((wipes / runs.length) * 100)}%`)
          );
        });

        matrixSection = h('div', { style: 'background:rgba(20,25,32,0.9); border:1px solid #334155; border-radius:8px; padding:10px; overflow-x:auto;' },
          h('h3', { style: 'margin:0 0 8px 0; font-size:14px; color:#e2e8f0;' }, '📊 Gun Comparison Matrix (Paired Seeds 5..8)'),
          h('table', { style: 'width:100%; border-collapse:collapse; font-size:12px; color:#cbd5e1;' },
            h('thead', {},
              h('tr', { style: 'border-bottom:1.5px solid #475569; color:#94a3b8; font-family:var(--display);' },
                h('th', { style: 'padding:6px; text-align:left;' }, 'Weapon'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Runs'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Avg Kills'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Hit %'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Dmg'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Kill Dist'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'TTK'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Noise/Kill'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Clear %'),
                h('th', { style: 'padding:6px; text-align:center;' }, 'Wipe %')
              )
            ),
            h('tbody', {}, ...rows)
          )
        );
      } else {
        matrixSection = h('div');
      }

      container.append(header, cfgBar, summarySection, matrixSection);
    };

    renderLab();
    return container;
  }

  private overseerPanel(): HTMLElement {
    const g = this.game;
    const colony = g.colony;
    const colonists = g.ecs.query(Comp.Pawn, Comp.Character, Comp.Transform);

    if (this.selectedColonist === null || !colonists.includes(this.selectedColonist)) {
      this.selectedColonist = colonists[0] ?? null;
    }

    const storytellers: { id: StorytellerKind; name: string; icon: string; desc: string }[] = [
      { id: 'caspar_chill', name: 'Caspar Chill', icon: '☀️', desc: 'Peaceful builder — gentle wanderers and generous supply drops.' },
      { id: 'randy_ruckus', name: 'Randy Ruckus', icon: '🎲', desc: 'Total unpredictability — random animal stampedes, drops, and sudden surprises.' },
      { id: 'hilda_hardcore', name: 'Hilda Hardcore', icon: '💀', desc: 'Brutal challenges — frequent screamer waves, short circuits, and harsh breakdowns.' },
    ];

    const storyBtns = storytellers.map((st) => {
      const active = colony.storyteller === st.id;
      return h('button', {
        class: `story-btn${active ? ' active' : ''}`,
        title: st.desc,
        onclick: () => {
          colony.storyteller = st.id;
          this.toast(`Storyteller AI changed to ${st.name}`);
          this.renderPanel();
        }
      }, `${st.icon} ${st.name}`);
    });

    const incidentStrip = h('div', { class: 'incident-strip' },
      h('span', { style: 'font-weight: bold; font-size: 12px; margin-right: 4px; align-self: center;' }, 'Incident Test:'),
      h('button', {
        class: 'inc-btn',
        onclick: () => { colony.triggerIncident('wanderer_joins', g); this.renderPanel(); }
      }, '➕ Wanderer Joins'),
      h('button', {
        class: 'inc-btn',
        onclick: () => { colony.triggerIncident('supply_drop', g); this.renderPanel(); }
      }, '🪂 Supply Drop'),
      h('button', {
        class: 'inc-btn',
        onclick: () => { colony.triggerIncident('stampede', g); this.renderPanel(); }
      }, '🐾 Stampede'),
      h('button', {
        class: 'inc-btn',
        onclick: () => { colony.triggerIncident('short_circuit', g); this.renderPanel(); }
      }, '⚡ Short Circuit'),
      h('button', {
        class: 'inc-btn',
        onclick: () => { colony.triggerIncident('screamer_swarm', g); this.renderPanel(); }
      }, '🚨 Screamer Swarm')
    );

    const roles: { id: PawnWorkRole; label: string; icon: string; skill: PawnSkill }[] = [
      { id: 'firefight', label: 'Fire', icon: '🔥', skill: 'combat' },
      { id: 'patient', label: 'Patient', icon: '🩹', skill: 'medicine' },
      { id: 'doctor', label: 'Doctor', icon: '🩺', skill: 'medicine' },
      { id: 'build', label: 'Build', icon: '🔨', skill: 'construction' },
      { id: 'farm', label: 'Farm', icon: '🌾', skill: 'plants' },
      { id: 'mine', label: 'Mine', icon: '⛏️', skill: 'mining' },
      { id: 'haul', label: 'Haul', icon: '📦', skill: 'hauling' },
      { id: 'craft', label: 'Craft', icon: '⚙️', skill: 'crafting' },
      { id: 'guard', label: 'Guard', icon: '🛡️', skill: 'combat' },
    ];

    const thead = h('thead', {},
      h('tr', {},
        h('th', { style: 'text-align: left; padding-left: 8px;' }, 'Colonist'),
        h('th', {}, 'Mood'),
        ...roles.map((r) => h('th', { title: `${r.label} (${r.skill})` }, `${r.icon} ${r.label}`))
      )
    );

    const rows = colonists.map((ce) => {
      const pawn = g.ecs.get(ce, Comp.Pawn)!;
      const ch = g.ecs.get(ce, Comp.Character)!;
      const isSelected = this.selectedColonist === ce;

      const moodVal = clamp(pawn.mood, 0, 100);
      const moodClass = moodVal > 60 ? 'mood-high' : moodVal > 30 ? 'mood-med' : 'mood-low';
      const moodEl = h('div', { class: 'mood-box' },
        h('div', { class: 'mood-bar-bg', title: `Mood: ${Math.round(moodVal)}%` },
          h('div', { class: `mood-bar-fill ${moodClass}`, style: `width: ${moodVal}%;` })
        ),
        pawn.mentalBreak.kind !== 'none'
          ? h('span', { class: 'break-badge' }, `BREAK: ${pawn.mentalBreak.kind.replace('_', ' ')}`)
          : h('span', { style: 'font-size: 11px; color: var(--muted);' }, `${Math.round(moodVal)}%`)
      );

      const prioCells = roles.map((r) => {
        const cur = pawn.workPriorities[r.id] ?? 3;
        const skill = pawn.skills[r.skill];
        const passionIcon = skill.passion === 'burning' ? '🔥' : skill.passion === 'single' ? '⭐' : '';
        const titleText = `${r.label} (Skill Lv ${skill.level}${passionIcon ? ` ${passionIcon}` : ''}) — Priority ${cur === 0 ? 'Disabled' : cur}. Click to change (1-4).`;

        const prioBtn = h('button', {
          class: `prio-btn prio-${cur}`,
          title: titleText,
          onclick: (ev: Event) => {
            ev.stopPropagation();
            const next = cur === 0 ? 1 : cur === 4 ? 0 : cur + 1;
            pawn.workPriorities[r.id] = next;
            this.renderPanel();
          }
        }, cur === 0 ? '—' : String(cur));

        return h('td', {}, prioBtn);
      });

      const nameCell = h('td', {
        class: `pawn-cell${isSelected ? ' selected' : ''}`,
        onclick: () => {
          this.selectedColonist = ce;
          this.renderPanel();
        }
      }, ch.name);

      return h('tr', {},
        nameCell,
        h('td', {}, moodEl),
        ...prioCells
      );
    });

    const table = h('table', { class: 'work-table' },
      thead,
      h('tbody', {}, ...rows)
    );

    let detailEl: HTMLElement | null = null;
    if (this.selectedColonist !== null && g.ecs.has(this.selectedColonist, Comp.Pawn)) {
      const sp = g.ecs.get(this.selectedColonist, Comp.Pawn)!;
      const sch = g.ecs.get(this.selectedColonist, Comp.Character)!;

      const traitsBadges = sp.traits.map((tr) => h('span', { class: 'trait-tag' }, tr.replace('_', ' ').toUpperCase()));

      const skillItems = (Object.keys(sp.skills) as PawnSkill[]).map((sk) => {
        const s = sp.skills[sk];
        const pass = s.passion === 'burning' ? '🔥' : s.passion === 'single' ? '⭐' : '';
        return h('div', { class: 'skill-item' },
          h('span', { style: 'color: var(--muted); text-transform: capitalize;' }, sk),
          h('strong', {}, `Lv ${s.level} ${pass}`)
        );
      });

      const thoughtRows = sp.thoughts.map((th) => {
        const isPos = th.delta >= 0;
        return h('div', { class: `thought-row ${isPos ? 'thought-pos' : 'thought-neg'}` },
          h('span', {}, th.label),
          h('span', { style: 'font-size: 11px;' }, `${Math.ceil(th.duration)}s`)
        );
      });

      const jobDesc = sp.mentalBreak.kind !== 'none'
        ? `🚨 Slapstick Breakdown: ${sp.mentalBreak.kind.replace('_', ' ')} (${Math.ceil(sp.mentalBreak.timer)}s left)`
        : sp.currentJob
        ? `🛠️ Performing ${sp.currentJob.role.toUpperCase()} task (${sp.currentJob.state})`
        : '🛋️ Taking it easy / idle wandering';

      detailEl = h('div', { class: 'colonist-detail' },
        h('div', { style: 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;' },
          h('h3', { style: 'margin: 0; font-family: Alegreya SC, serif; font-size: 18px;' }, `${sch.name} — Colonist Dossier`),
          h('span', { style: 'font-size: 12px; color: var(--gold);' }, jobDesc)
        ),
        h('div', { class: 'traits-wrap' }, ...traitsBadges),
        h('div', { style: 'font-size: 13px; font-weight: bold; margin-top: 8px;' }, 'Proficiencies & Passions:'),
        h('div', { class: 'skills-grid' }, ...skillItems),
        h('div', { style: 'font-size: 13px; font-weight: bold; margin-top: 8px;' }, `Active Psychological Thoughts (${sp.thoughts.length}):`),
        h('div', { class: 'thoughts-list' }, ...thoughtRows)
      );
    }

    return h('div', { class: 'overseer-panel' },
      h('h2', { style: 'margin: 0;' }, 'Colony Overseer & Psychology'),
      h('p', { class: 'hint', style: 'margin: 4px 0 8px;' }, 'Configure RimWorld-style 1-4 Priority Work Matrix (1=Urgent, 4=Low, —=Disabled). Pawns dynamically obey highest priority roles.'),
      h('div', { style: 'font-weight: bold; font-size: 13px;' }, 'Storyteller AI Director:'),
      h('div', { class: 'story-row' }, ...storyBtns),
      incidentStrip,
      table,
      detailEl ?? h('div', {}, '')
    );
  }

  private factionPanel(): HTMLElement {
    const g = this.game;
    const pt = g.playerT;
    const f = g.faction;
    const zone = f.getZoneAt(pt.x, pt.y, g.world.spawn.x, g.world.spawn.y);

    const zoneClass = zone === 'green' ? 'zone-green' : zone === 'yellow' ? 'zone-yellow' : 'zone-red';
    const zoneName = zone === 'green' ? '🟢 GREEN ZONE: Safe Trading & Garage Haven'
      : zone === 'yellow' ? '🟡 YELLOW ZONE: Contested Highway (Consensual PvP)'
      : '🔴 RED ZONE: Outlaw Wasteland (Full Open PvPvE)';
    const zoneDesc = zone === 'green'
      ? 'Immune to PvP damage and base raids. Ideal for trading, crafting, and vehicle repair.'
      : zone === 'yellow'
      ? 'Hostile creatures and contested mineral nodes. PvP opt-in with alert sirens.'
      : 'Full-loot PvPvE wilderness! Tesla shields lower during active raid windows. Pure resource geodes.';

    let activeTotem: Entity | null = null;
    for (const te of g.ecs.query(Comp.Totem, Comp.Transform)) {
      const tt = g.ecs.get(te, Comp.Transform)!;
      if (Math.hypot(tt.x - pt.x, tt.y - pt.y) <= 35) {
        activeTotem = te;
        break;
      }
    }
    const totem = activeTotem !== null ? g.ecs.get(activeTotem, Comp.Totem) : null;
    const currentHour = Math.floor(g.clock.t * 24);

    const shieldActive = totem ? totem.shieldActive : true;
    const shieldStatusEl = h('div', { class: `shield-box ${shieldActive ? 'shield-active' : 'shield-down'}` },
      h('div', { style: 'display: flex; justify-content: space-between; align-items: center;' },
        h('strong', {}, shieldActive ? '🛡️ Tesla Shield Forcefield: ONLINE' : '⚠️ Tesla Shield Forcefield: DOWN (Raid Window Active)'),
        h('span', { style: 'font-size: 12px; color: var(--gold);' }, `Sim Hour: ${currentHour}:00`)
      ),
      h('p', { class: 'hint', style: 'margin: 4px 0 8px;' },
        shieldActive
          ? 'Base structures within 30m of the Clan Totem are 100% immune to damage and offline raids.'
          : 'Scheduled active defense window in effect! Structures can be breached by C4 charges and vehicle rams!'
      ),
      h('div', { style: 'display: flex; gap: 8px;' },
        h('button', {
          class: 'v-act-btn',
          style: 'flex: 1;',
          onclick: () => {
            f.activeRaidOverride = !f.activeRaidOverride;
            this.toast(f.activeRaidOverride ? '⚠️ Raid Window Override: SHIELDS LOWERED!' : '🛡️ Raid Window Override: SHIELDS RESTORED!');
            this.renderPanel();
          }
        }, f.activeRaidOverride ? 'Restore Tesla Shield' : 'Toggle Raid Window Test'),
        h('button', {
          class: 'v-act-btn',
          style: 'flex: 1;',
          onclick: () => {
            f.signPeaceTreaty(2);
            this.toast('Signed 48-Hour Peace Treaty (+2 Days)!');
            this.renderPanel();
          }
        }, `Sign Peace Treaty (${f.treatyDaysLeft}d left)`)
      )
    );

    const localPollution = Math.round(g.siege.getPollutionAt(pt.x, pt.y));
    const pollutionBox = h('div', { class: 'shield-box' },
      h('div', { style: 'display: flex; justify-content: space-between; align-items: center;' },
        h('strong', {}, `🏭 Industrial Pollution: ${localPollution} ppm`),
        h('span', { style: `font-size: 12px; font-weight: bold; color: ${localPollution > 35 ? '#e74c3c' : localPollution > 15 ? '#f39c12' : '#2ecc71'};` },
          localPollution > 35 ? 'CRITICAL (Siege Imminent)' : localPollution > 15 ? 'ELEVATED' : 'CLEAR'
        )
      ),
      h('p', { class: 'hint', style: 'margin: 4px 0 8px;' }, 'Smelters and mining drills generate smog and acoustic noise that attracts mutant siege waves.'),
      h('div', { style: 'display: flex; gap: 8px;' },
        h('button', {
          class: 'v-act-btn highlight',
          style: 'flex: 1;',
          onclick: () => {
            g.siege.triggerIndustrialSiege(g, pt.x, pt.y);
            this.toast('Industrial Siege Assault Triggered!');
            this.close();
          }
        }, '🚨 Trigger Siege Wave Test'),
        h('button', {
          class: 'v-act-btn',
          style: 'flex: 1;',
          onclick: () => {
            if (f.armC4(pt.x, pt.y, g)) {
              this.close();
            }
          }
        }, '🧨 Arm C4 Breaching Plunger')
      )
    );

    const bountyList = f.bounties.map((b) => {
      const isDone = b.completed;
      return h('div', { class: `bounty-item${isDone ? ' done' : ''}` },
        h('div', {},
          h('strong', {}, b.title),
          h('div', { style: 'font-size: 12px; color: var(--muted);' }, b.desc),
          h('div', { style: 'font-size: 11px; color: var(--gold); margin-top: 2px;' }, `Progress: ${b.currentCount}/${b.targetCount} · Reward: ${b.rewardCredits} Credits${b.rewardItem ? ` + ${b.rewardItem}` : ''}`)
        ),
        isDone
          ? h('span', { style: 'color: #2ecc71; font-weight: bold; font-size: 12px;' }, '✓ CLAIMED')
          : h('button', {
              class: 'inc-btn',
              onclick: () => {
                f.advanceBounty(b.id, 1, g);
                this.renderPanel();
              }
            }, 'Advance (+1)')
      );
    });

    return h('div', { class: 'faction-panel' },
      h('h2', { style: 'margin: 0;' }, 'Faction Warfare & Base Raiding'),
      h('div', { class: `zone-header ${zoneClass}` },
        h('div', { style: 'font-weight: bold; font-size: 15px;' }, zoneName),
        h('div', { style: 'font-size: 13px; margin-top: 3px;' }, zoneDesc)
      ),
      h('div', { style: 'display: flex; justify-content: space-between; margin-bottom: 6px;' },
        h('span', { style: 'font-weight: bold;' }, `Clan: ${f.clanName}`),
        h('span', { style: 'color: var(--gold); font-weight: bold;' }, `💰 Balance: ${f.credits} Credits`)
      ),
      shieldStatusEl,
      pollutionBox,
      h('h3', { style: 'margin: 10px 0 4px; font-family: Alegreya SC, serif;' }, '📻 Radio Black Market Bounties'),
      h('div', { class: 'bounty-grid' }, ...bountyList)
    );
  }

  private inventoryPanel(): HTMLElement {
    const tabs: { id: 'backpack' | 'interior' | 'base'; label: string; icon: string }[] = [
      { id: 'backpack', label: 'Backpack', icon: '🎒' },
      { id: 'interior', label: 'Interior & Worn', icon: '🏠' },
      { id: 'base', label: 'Base Storage', icon: '🏰' },
    ];

    const tabRow = h('div', { class: 'tabs', style: 'margin: 0 0 14px; gap: 8px; display: flex; border-bottom: 1.5px solid var(--line); padding-bottom: 8px;' },
      ...tabs.map((t) => h('button', {
        class: `inv-tab-btn${this.invTab === t.id ? ' active' : ''}`,
        onclick: () => {
          this.invTab = t.id;
          this.selected = null;
          this.renderPanel();
        }
      }, `${t.icon} ${t.label}`))
    );

    const body = this.invTab === 'backpack' ? this.backpackSubPanel()
      : this.invTab === 'interior' ? this.interiorSubPanel()
      : this.baseStorageSubPanel();

    return h('div', { class: 'inv-panel' }, tabRow, body);
  }

  private backpackSubPanel(): HTMLElement {
    const g = this.game, inv = g.playerInv.inv;
    const usedCount = inv.slots.filter((s) => s !== null).length;

    const grid = h('div', { class: 'grid' }, ...Array.from({ length: INV_SIZE }, (_, i) => {
      const s = this.slotEl(i, false, () => {
        if (this.selected === null) {
          this.selected = inv.slots[i] ? i : null;
        } else if (this.selected === i) {
          this.selected = null;
        } else {
          const a = inv.slots[this.selected] ?? null, b = inv.slots[i] ?? null;
          inv.slots[this.selected] = b;
          inv.slots[i] = a;
          this.selected = null;
          g.holdSlot(g.playerInv.held);
          this.renderHotbar();
        }
        this.renderPanel();
      });
      if (i < HOTBAR) s.append(h('i', {}, String(i + 1)));
      return s;
    }));

    const sel = this.selected !== null ? inv.slots[this.selected] : null;
    const def = sel ? g.content.items.get(sel.item) : undefined;
    const actions = h('div', { class: 'actions', style: 'margin-top: 10px; min-height: 38px;' });

    if (sel && def && this.selected !== null) {
      const idx = this.selected;
      actions.append(h('strong', { style: 'color: var(--gold); margin-right: 6px;' }, `${def.name} ×${sel.qty}`));
      if (def.food) {
        actions.append(h('button', {
          onclick: () => { g.eat(idx); this.renderPanel(); }
        }, `🍗 Eat (+${def.food.hunger}H)`));
      }
      if (['cream_pie', 'water_balloon', 'stink_sock_bomb', 'road_flare', 'whoopee_cushion_trap'].includes(def.id)) {
        actions.append(h('button', {
          onclick: () => {
            const ang = g.playerT.angle;
            const tx = g.playerT.x + Math.cos(ang) * 5.0;
            const ty = g.playerT.y + Math.sin(ang) * 5.0;
            g.throwItem(def.id, tx, ty);
            this.renderPanel();
            this.renderHotbar();
          }
        }, '🎯 Throw Ahead (5m)'));
      }
      if (def.equip && def.equip.slot !== 'mainHand') {
        actions.append(h('button', {
          onclick: () => { g.wear(idx); this.selected = null; this.renderPanel(); }
        }, '👕 Wear / Equip'));
      }
      if (def.equip?.slot === 'mainHand') {
        actions.append(h('span', { class: 'hint' }, 'Put in slots 1-8 to hold in hand'));
      }
      if (def.gun) {
        actions.append(h('button', {
          style: 'border-color: #f39c12; color: #f1c40f; font-weight: bold;',
          onclick: () => {
            this.gunsmith.selectedWeapon = def.id;
            this.open('gunsmith');
          }
        }, '🛠️ Mod Gun [U]'));
      }
      actions.append(h('button', {
        onclick: () => { inv.remove(def.id, 1); this.renderPanel(); }
      }, 'Drop One'));
      if (sel.qty > 1) {
        actions.append(h('button', {
          onclick: () => { inv.remove(def.id, sel.qty); this.renderPanel(); }
        }, 'Drop All'));
      }
    } else {
      actions.append(h('span', { class: 'hint' }, 'Select an item to view options, consume, or equip.'));
    }

    const bundleBtn = h('button', {
      style: 'padding: 6px 10px; font-size: 11px; background: rgba(30,40,60,0.85); border: 1.5px solid var(--glow); border-radius: 6px; cursor: pointer; color: var(--cream); font-weight: bold;',
      onclick: () => {
        const bundle = [
          { id: 'pool_noodle_of_justice', qty: 1 },
          { id: 'chainsaw_of_regret', qty: 1 },
          { id: 'slingshot', qty: 1 },
          { id: 'pebbles', qty: 50 },
          { id: 'potato_cannon', qty: 1 },
          { id: 'wild_carrot', qty: 25 },
          { id: 'canned_beans', qty: 10 },
          { id: 'hunting_rifle', qty: 1 },
          { id: 'rifle_ammo', qty: 30 },
          { id: 'pump_shotgun', qty: 1 },
          { id: 'shotgun_shells', qty: 24 },
          { id: 'nailgun_smg', qty: 1 },
          { id: 'scrap_nails', qty: 60 },
          { id: 'boing_boots', qty: 1 },
          { id: 'propeller_beanie', qty: 1 },
          { id: 'pillow_vest', qty: 1 },
          { id: 'cream_pie', qty: 5 },
          { id: 'water_balloon', qty: 5 },
          { id: 'road_flare', qty: 5 },
          { id: 'chili_of_regret', qty: 3 },
          { id: 'rainbow_soup', qty: 3 },
          { id: 'crispy_sock_chips', qty: 3 },
        ];
        for (const it of bundle) inv.add(it.id, it.qty);
        this.toast('🎁 Master Firearms Arsenal & Snack Kit added to Pack!');
        this.renderPanel();
        this.renderHotbar();
      }
    }, '🎁 Weapons & Ammo');

    const modGunBtn = h('button', {
      style: 'padding: 6px 10px; font-size: 11px; background: rgba(50,40,20,0.9); border: 1.5px solid #f39c12; border-radius: 6px; cursor: pointer; color: #f1c40f; font-weight: bold;',
      onclick: () => { this.open('gunsmith'); }
    }, '🛠️ Gunsmith [U]');

    const rangeBtn = h('button', {
      style: 'padding: 6px 10px; font-size: 11px; background: rgba(55,25,30,0.9); border: 1.5px solid #e74c3c; border-radius: 6px; cursor: pointer; color: #ff9999; font-weight: bold;',
      onclick: () => {
        g.teleportToShootingRange();
        this.save();
        this.toast('🎯 Teleported to Shooting Range with Armory & Target Lanes!');
        this.close();
      }
    }, '🎯 Shooting Range');

    const herdBtn = h('button', {
      style: 'padding: 6px 10px; font-size: 11px; background: rgba(20,50,30,0.85); border: 1.5px solid #2ecc71; border-radius: 6px; cursor: pointer; color: var(--cream); font-weight: bold;',
      onclick: () => {
        const pt = g.playerT;
        g.spawnAnimal(pt.x + 2.0, pt.y + 1.0, 'cow', 'Daisy');
        g.spawnAnimal(pt.x + 3.0, pt.y + 1.5, 'cow', 'Bessie');
        g.spawnAnimal(pt.x + 1.8, pt.y - 1.8, 'sheep', 'Fluffy');
        g.spawnAnimal(pt.x + 2.5, pt.y - 2.2, 'sheep', 'Cotton');
        g.spawnAnimal(pt.x - 2.0, pt.y + 2.0, 'pig', 'Barnaby');
        g.spawnAnimal(pt.x - 2.8, pt.y + 2.4, 'pig', 'Truffles');
        g.spawnAnimal(pt.x - 1.5, pt.y - 1.5, 'chicken', 'Clucky');
        g.spawnAnimal(pt.x - 2.2, pt.y - 1.8, 'chicken', 'Henrietta');
        g.spawnAnimal(pt.x + 3.5, pt.y, 'deer', 'Bambi');
        g.spawnAnimal(pt.x - 3.2, pt.y, 'rabbit', 'Thumper');
        g.spawnAnimal(pt.x, pt.y - 2.8, 'goose', 'Honkers');
        inv.add('animal_feed', 10);
        inv.add('wild_carrot', 10);
        inv.add('fresh_milk', 2);
        this.toast('🐾 Farm Livestock & Wildlife spawned beside you!');
        this.renderPanel();
        this.renderHotbar();
      }
    }, '🐾 Spawn Farm Livestock Herd');

    const cheatBar = h('div', { style: 'display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px;' }, bundleBtn, modGunBtn, rangeBtn, herdBtn);

    return h('div', {},
      h('div', { style: 'display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;' },
        h('h2', { style: 'margin: 0;' }, 'Backpack Inventory'),
        h('span', { style: 'font-size: 13px; color: var(--gold); font-weight: bold;' }, `Capacity: ${usedCount}/${INV_SIZE} slots  ·  Weight: ${inv.weight().toFixed(1)} kg`)
      ),
      grid,
      actions,
      cheatBar,
      h('p', { class: 'hint' }, 'Click an item to select, click another slot to swap. Hotbar is slots 1-8.')
    );
  }

  private interiorSubPanel(): HTMLElement {
    const g = this.game, look = g.playerChar.look, pt = g.playerT, n = g.playerNeeds;

    // Character preview doll
    const dollCanvas = h('canvas', { width: 120, height: 140, style: 'background: rgba(14,16,20,0.75); border: 1.5px solid var(--line); border-radius: 8px; margin-bottom: 8px; display: block;' }) as HTMLCanvasElement;
    const dctx = dollCanvas.getContext('2d');
    if (dctx) {
      dctx.save();
      dctx.translate(60, 70);
      dctx.scale(130, 130);
      dctx.rotate(Math.PI / 2);
      this.renderer.chars.draw(dctx as any, look, { phase: 0, walk: 0, swing: -1, time: 0 }, 140);
      dctx.restore();
    }

    // Outfit equipment
    const worn = h('div', { class: 'worn' }, ...SLOTS.map((s: Slot) => {
      const inst = look.outfit[s];
      return h('div', { class: 'wornrow' },
        h('span', { style: 'font-weight: bold; font-size: 13px;' }, SLOT_LABELS[s]),
        inst
          ? h('button', {
              style: 'padding: 4px 10px; font-size: 12px; border-radius: 6px;',
              onclick: () => { g.unwear(s); this.renderPanel(); }
            }, `${g.content.items.get(inst.item)?.name ?? inst.item} ✕`)
          : h('em', {}, '—')
      );
    }));

    // Active Gear Perks
    const perks: string[] = [];
    if (look.outfit.feet?.item === 'boing_boots') perks.push('👟 Boing Boots: +15% Run Speed');
    if (look.outfit.head?.item === 'propeller_beanie') perks.push('🚁 Propeller Beanie: +10% Speed & Float');
    if (look.outfit.torso?.item === 'pillow_vest') perks.push('🛏️ Pillow Vest: +20% Impact Absorption');
    const perksEl = perks.length > 0
      ? h('div', { style: 'margin-top: 8px; font-size: 12px; color: var(--gold); display: flex; flex-direction: column; gap: 2px;' }, ...perks.map((p) => h('span', {}, p)))
      : h('div', { class: 'hint', style: 'margin-top: 6px;' }, 'Equip hats, vests, and boots for stat bonuses.');

    // Building Shelter Detection
    const nearBuildings = g.world.buildingsNear(pt.x, pt.y, 2);
    const curBld = nearBuildings.find((b) => insideBuilding(b, pt.x, pt.y));

    let shelterEl: HTMLElement;
    if (curBld) {
      const bDef = g.content.buildings.find((d) => d.id === curBld.def);
      const name = bDef?.name ?? 'Survivor Building';
      shelterEl = h('div', { class: 'shelter-card sheltered' },
        h('div', { style: 'display: flex; justify-content: space-between; align-items: center;' },
          h('h3', { style: 'margin: 0; font-family: Alegreya SC, serif; font-size: 18px; color: #2ecc71;' }, `🏠 ${name}`),
          h('span', { style: 'font-size: 11px; background: rgba(46,204,113,0.2); color: #2ecc71; padding: 2px 8px; border-radius: 999px; font-weight: bold;' }, 'INTERIOR SHELTER')
        ),
        h('p', { style: 'font-size: 13px; margin: 4px 0 8px;' }, bDef?.desc ?? 'Safe haven protected from the outdoor elements.'),
        h('div', { style: 'display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 12px; color: var(--muted); background: rgba(10,12,16,0.6); padding: 6px 8px; border-radius: 6px;' },
          h('span', {}, `📐 Size: ${curBld.w}m × ${curBld.h}m (${curBld.w * curBld.h}m²)`),
          h('span', {}, `🚪 Facing: ${curBld.door} Entrance`),
          h('span', {}, `🧱 Walls: ${bDef?.walls?.toUpperCase() ?? 'WOOD'}`),
          h('span', {}, `🛖 Roof: ${bDef?.roof?.toUpperCase() ?? 'GABLE'}`)
        ),
        h('div', { style: 'margin-top: 8px; display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: #a3e635;' },
          h('span', {}, '🛡️ 100% Weather Shielding (Rain & Darkness Immune)'),
          h('span', {}, '🛋️ Cosy Interior Comfort (+12 Sanity regen / hr)'),
          h('span', {}, '🔥 Hearth Warmth Protected')
        )
      );
    } else {
      shelterEl = h('div', { class: 'shelter-card exposed' },
        h('div', { style: 'display: flex; justify-content: space-between; align-items: center;' },
          h('h3', { style: 'margin: 0; font-family: Alegreya SC, serif; font-size: 18px; color: #f39c12;' }, '🌲 Wilderness / Outdoors'),
          h('span', { style: 'font-size: 11px; background: rgba(243,156,18,0.2); color: #f39c12; padding: 2px 8px; border-radius: 999px; font-weight: bold;' }, 'EXPOSED TO ELEMENTS')
        ),
        h('p', { style: 'font-size: 13px; margin: 4px 0 8px;' }, 'You are outside without roof overhead. Vulnerable to harsh weather, dark shadows, and wild predators.'),
        h('p', { class: 'hint', style: 'margin: 0;' }, '💡 Step inside any town cottage, shop, cabin, or build walls with doors to secure interior comfort bonuses.')
      );
    }

    // Physical Vitals Readout
    const vitalsEl = h('div', { style: 'background: rgba(10,12,16,0.6); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin-top: 8px;' },
      h('div', { style: 'font-weight: bold; font-size: 13px; margin-bottom: 4px;' }, 'Vital Signs & Status:'),
      h('div', { style: 'display: flex; justify-content: space-between; font-size: 13px;' },
        h('span', { style: 'color: #e74c3c;' }, `❤️ Health: ${Math.round(n.health)}/${n.maxHealth}`),
        h('span', { style: 'color: #e67e22;' }, `🍖 Hunger: ${Math.round(n.hunger)}/${n.maxHunger}`),
        h('span', { style: 'color: #9b59b6;' }, `🧠 Sanity: ${Math.round(n.sanity)}/${n.maxSanity}`)
      )
    );

    return h('div', { class: 'cols' },
      h('div', { style: 'flex: 1; min-width: 200px;' },
        h('h2', { style: 'margin: 0 0 6px;' }, 'Character & Worn'),
        dollCanvas,
        worn,
        perksEl
      ),
      h('div', { style: 'flex: 1.3; min-width: 240px;' },
        h('h2', { style: 'margin: 0 0 6px;' }, 'Shelter & Interior'),
        shelterEl,
        vitalsEl
      )
    );
  }

  private baseStorageSubPanel(): HTMLElement {
    const g = this.game, inv = g.playerInv.inv, base = g.baseStorage;
    const usedPack = inv.slots.filter((s) => s !== null).length;
    const usedBase = base.slots.filter((s) => s !== null).length;

    // Fast transfer action buttons
    const depositAllBtn = h('button', {
      class: 'v-act-btn highlight',
      onclick: () => {
        let moved = 0;
        for (let i = 0; i < inv.slots.length; i++) {
          const st = inv.slots[i];
          if (!st) continue;
          const left = base.add(st.item, st.qty);
          if (left < st.qty) moved++;
          inv.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
        }
        if (moved > 0) {
          sound.craft();
          this.toast('📥 Deposited items into Base Storage!');
        } else {
          this.toast('No items to deposit or base storage is full');
        }
        g.holdSlot(g.playerInv.held);
        this.renderHotbar();
        this.renderPanel();
      }
    }, '📥 Deposit All');

    const quickStackBtn = h('button', {
      class: 'v-act-btn',
      onclick: () => {
        let stacked = 0;
        for (let i = 0; i < inv.slots.length; i++) {
          const st = inv.slots[i];
          if (!st) continue;
          if (base.count(st.item) > 0) {
            const left = base.add(st.item, st.qty);
            if (left < st.qty) stacked++;
            inv.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
          }
        }
        if (stacked > 0) {
          sound.craft();
          this.toast('🔄 Quick stacked matching items into Base Depot!');
        } else {
          this.toast('No matching items in Base Depot to quick stack');
        }
        g.holdSlot(g.playerInv.held);
        this.renderHotbar();
        this.renderPanel();
      }
    }, '🔄 Quick Stack');

    const withdrawAllBtn = h('button', {
      class: 'v-act-btn',
      onclick: () => {
        let moved = 0;
        for (let i = 0; i < base.slots.length; i++) {
          const st = base.slots[i];
          if (!st) continue;
          const left = inv.add(st.item, st.qty);
          if (left < st.qty) moved++;
          base.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
        }
        if (moved > 0) {
          sound.craft();
          this.toast('📤 Withdrew items from Base Depot into Pack!');
        } else {
          this.toast('No items to withdraw or backpack is full');
        }
        g.holdSlot(g.playerInv.held);
        this.renderHotbar();
        this.renderPanel();
      }
    }, '📤 Withdraw All');

    const header = h('div', { style: 'display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;' },
      depositAllBtn,
      quickStackBtn,
      withdrawAllBtn
    );

    // Left Grid: Backpack (8x3)
    const moveFromPack = (i: number): void => {
      const st = inv.slots[i];
      if (!st) return;
      const left = base.add(st.item, st.qty);
      inv.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
      sound.harvest();
      g.holdSlot(g.playerInv.held);
      this.renderHotbar();
      this.renderPanel();
    };

    const packGrid = h('div', { class: 'grid' }, ...Array.from({ length: INV_SIZE }, (_, i) => {
      const s = this.invSlotEl(inv, i, false, false, () => moveFromPack(i));
      if (i < HOTBAR) s.append(h('i', {}, String(i + 1)));
      return s;
    }));

    // Right Grid: Base Depot (6x5)
    const moveFromBase = (i: number): void => {
      const st = base.slots[i];
      if (!st) return;
      const left = inv.add(st.item, st.qty);
      base.slots[i] = left > 0 ? { item: st.item, qty: left } : null;
      sound.harvest();
      g.holdSlot(g.playerInv.held);
      this.renderHotbar();
      this.renderPanel();
    };

    const baseGrid = h('div', { class: 'base-storage-grid' }, ...Array.from({ length: base.slots.length }, (_, i) => {
      return this.invSlotEl(base, i, false, false, () => moveFromBase(i));
    }));

    return h('div', {},
      header,
      h('div', { class: 'cols', style: 'align-items: flex-start;' },
        h('div', { style: 'flex: 1;' },
          h('div', { style: 'display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;' },
            h('h3', { style: 'margin: 0; font-family: Alegreya SC, serif; font-size: 18px;' }, '🎒 Player Backpack'),
            h('span', { style: 'font-size: 12px; color: var(--gold);' }, `${usedPack}/${INV_SIZE} slots`)
          ),
          packGrid,
          h('p', { class: 'hint' }, 'Click item to deposit into Base Depot.')
        ),
        h('div', { style: 'flex: 1;' },
          h('div', { style: 'display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;' },
            h('h3', { style: 'margin: 0; font-family: Alegreya SC, serif; font-size: 18px; color: var(--gold);' }, '🏰 Base Storage Depot'),
            h('span', { style: 'font-size: 12px; color: var(--gold);' }, `${usedBase}/${base.slots.length} slots`)
          ),
          baseGrid,
          h('p', { class: 'hint' }, 'Click item to withdraw into Backpack.')
        )
      )
    );
  }

  private craftPanel(): HTMLElement {
    const g = this.game, inv = g.playerInv.inv;
    return h('div', {}, h('h2', {}, 'Craft'), ...g.content.recipes.map((r) => {
      const can = inv.has(r.inputs);
      return h('div', { class: `recipe${can ? '' : ' no'}` },
        itemIcon(g.content, r.output.item, 40),
        h('div', { class: 'rinfo' }, h('strong', {}, `${r.name}${r.output.qty > 1 ? ` ×${r.output.qty}` : ''}`), h('span', {}, Object.entries(r.inputs).map(([k, v]) => `${v} ${g.content.items.get(k)?.name ?? k} (${inv.count(k)})`).join(', '))),
        h('button', { disabled: !can, onclick: () => { g.craft(r.id); this.renderHotbar(); this.renderPanel(); } }, 'Craft'));
    }));
  }

  private buildPanel(): HTMLElement {
    const g = this.game, inv = g.playerInv.inv;
    return h('div', {}, h('h2', {}, 'Build'), h('p', { class: 'hint' }, 'Pick a piece, then click the ground. Place a Claim Flag first: walls, doors, beds and chests need a claim. Right-click salvages.'),
      ...[...g.content.structures.values()].map((s) => h('button', { class: `struct${this.buildId === s.id ? ' on' : ''}${g.canAfford(s) ? '' : ' no'}`, onclick: () => { this.buildId = this.buildId === s.id ? null : s.id; this.renderPanel(); } },
        h('strong', {}, s.name), h('span', {}, g.config.creative ? 'free in creative mode' : Object.entries(s.cost).map(([k, v]) => `${v} ${g.content.items.get(k)?.name ?? k} (${inv.count(k)})`).join(', ')), h('em', {}, s.requiresClaim ? 'needs claim' : s.claimRadius ? `claims ${s.claimRadius} tiles` : ''))));
  }

  private storagePanel(): HTMLElement {
    const g = this.game, inv = g.playerInv.inv, e = this.storageEntity;
    const structureChest = e !== null ? g.ecs.get(e, Comp.Structure)?.storage : undefined;
    const vehicleTrunk = e !== null ? g.ecs.get(e, Comp.Vehicle)?.trunk : undefined;
    const targetStorage = structureChest ?? vehicleTrunk;
    const label = vehicleTrunk ? 'Car Trunk' : 'Chest';
    if (!targetStorage) return h('div', {}, 'Nothing here.');
    const move = (from: typeof inv, to: typeof inv, i: number): void => { const st = from.slots[i]; if (!st) return; const left = to.add(st.item, st.qty); from.slots[i] = left > 0 ? { item: st.item, qty: left } : null; g.holdSlot(g.playerInv.held); this.renderHotbar(); this.renderPanel(); };
    const grid = (src: typeof inv, dst: typeof inv): HTMLElement => h('div', { class: 'grid' }, ...src.slots.map((st, i) => {
      const def = st ? g.content.items.get(st.item) : undefined; const el = h('button', { class: 'slot', title: def?.name ?? 'Empty', onclick: () => move(src, dst, i) });
      if (def && st) { el.append(itemIcon(g.content, def.id, 40)); if (st.qty > 1) el.append(h('b', {}, String(st.qty))); } return el;
    }));
    return h('div', { class: 'cols' }, h('div', {}, h('h2', {}, label), grid(targetStorage, inv)), h('div', {}, h('h2', {}, 'Pack'), grid(inv, targetStorage), h('p', { class: 'hint' }, 'Click an item to move it across.')));
  }

  /** Flip between the infinite generator and the frozen, editable map. */
  setStaticMode(on: boolean): void {
    const w = this.game.world;
    if (on && !w.staticMap) this.toast('Baking static map…');
    w.setStatic(on);
    this.toast(on ? '🗺️ Static Map: ON (editable, saved with your game)' : '🌍 Generative World: ON (static edits kept)');
  }
  toggleEditor(): void {
    if (this.panel !== 'editor' && !this.game.world.staticActive) this.setStaticMode(true);
    this.toggle('editor');
  }

  private editorPanel(): HTMLElement {
    const g = this.game, w = g.world, e = this.edit;
    const chip = (label: string, active: boolean, fn: () => void, color?: string): HTMLElement =>
      h('button', { style: `padding: 5px 9px; margin: 2px; border-radius: 8px; border: 1.5px solid ${active ? 'var(--glow)' : 'var(--line)'}; background: ${active ? 'rgba(80,60,40,0.95)' : 'rgba(30,22,18,0.9)'}; ${color ? `border-left: 8px solid ${color};` : ''}`, onclick: fn }, label);
    const tools: [EditTool, string][] = [['terrain', '🖌️ Terrain'], ['biome', '🌿 Biome'], ['object', '🌲 Object'], ['building', '🏠 Building'], ['erase', '🧹 Erase']];
    const toolRow = h('div', {}, ...tools.map(([t, label]) => chip(label, e.tool === t, () => { e.tool = t; this.renderPanel(); })));
    const sizeRow = h('div', { style: 'margin: 6px 0;' }, h('span', { class: 'hint' }, `Brush radius: ${e.radius} `),
      ...[1, 2, 4, 8, 14].map((r) => chip(String(r), e.radius === r, () => { e.radius = r; this.renderPanel(); })));
    let picker: HTMLElement;
    if (e.tool === 'terrain') {
      picker = h('div', {}, ...[...g.content.terrain.values()].map((t) => chip(t.id, e.terrain === t.id, () => { e.terrain = t.id; this.renderPanel(); }, t.colors[0])));
    } else if (e.tool === 'biome') {
      picker = h('div', {}, ...g.content.biomes.map((b, idx) => chip(b.name, e.biome === idx, () => { e.biome = idx; this.renderPanel(); }, b.tint)));
    } else if (e.tool === 'object') {
      picker = h('div', {}, ...[...g.content.objects.values()].map((o) => chip(o.name, e.kind === o.kind, () => { e.kind = o.kind; this.renderPanel(); })));
    } else if (e.tool === 'building') {
      picker = h('div', {},
        ...g.content.buildings.map((b) => chip(b.name, e.building === b.id, () => { e.building = b.id; this.renderPanel(); }, b.colors.roof)),
        h('p', { class: 'hint' }, 'Door side:'),
        ...(['N', 'E', 'S', 'W'] as const).map((s) => chip(s, e.door === s, () => { e.door = s; this.renderPanel(); })),
        h('p', { class: 'hint' }, 'Click to place. Roofs fade away when you walk inside. Erase or right-click removes a building.'));
    } else {
      picker = h('p', { class: 'hint' }, 'Hold left mouse to erase objects inside the ring; click removes a building. Right-click erases at the cursor with any tool.');
    }
    const counts = w.staticMap?.counts();
    return h('div', {},
      h('h2', {}, 'Map Editor'),
      h('p', { class: 'hint' }, 'Left-drag paints terrain, biome or erases. With Object, left-click places. Right-click erases objects. Hotkey: G.'),
      toolRow, sizeRow, picker,
      h('p', { class: 'hint', style: 'margin-top: 10px;' }, counts ? `Static map: ${w.staticMap!.radius * 2 * 16}×${w.staticMap!.radius * 2 * 16} tiles · ${counts.objects} objects · ${counts.features} features` : 'Static map not baked yet'),
      h('button', { style: 'padding: 8px 16px; margin: 6px 0; border-radius: 999px; width: 100%; border: 1.5px solid var(--line); background: rgba(30,22,18,0.9);', onclick: () => { this.setStaticMode(false); this.close(); } }, '🌍 Back to Generative World'),
      h('button', { style: 'padding: 8px 16px; margin: 6px 0; border-radius: 999px; width: 100%; border: 1.5px solid #b45309; background: rgba(60,25,15,0.9);', onclick: () => { if (confirm('Discard all map edits and re-freeze the generated world?')) { w.rebakeStatic(); this.toast('Static map re-baked from generator'); this.renderPanel(); } } }, '♻️ Reset Map From Generator'));
  }

  private settingsPanel(): HTMLElement {
    const g = this.game;
    const modeDesc = g.config.mode === 'zombie'
      ? 'Zombie Apocalypse Mode: Roaming undead, sound detection, weapons & vehicle ramming!'
      : 'Classic Survival Mode: Peaceful atmosphere, foraging, building, and exploration.';

    const modeBtn = h('button', {
      style: 'padding: 9px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 2px solid var(--glow); border-radius: 999px; width: 100%; font-weight: bold;',
      onclick: () => {
        g.config.mode = g.config.mode === 'zombie' ? 'classic' : 'zombie';
        if (g.config.mode === 'zombie') {
          g.spawnZombie(g.playerT.x + 12, g.playerT.y + 4, 'walker');
          g.spawnZombie(g.playerT.x - 12, g.playerT.y - 4, 'runner');
          this.toast('Zombie Outbreak Mode Activated!');
        } else {
          for (const ze of g.ecs.query(Comp.Zombie)) g.ecs.destroy(ze);
          this.toast('Classic Survival Mode Activated!');
        }
        this.renderPanel();
      }
    }, `Game Mode: ${g.config.mode.toUpperCase()}`);

    const creativeBtn = h('button', {
      style: 'padding: 8px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 1.5px solid var(--line); border-radius: 999px; width: 100%;',
      onclick: () => {
        g.config.creative = !g.config.creative;
        this.toast(`Creative Free-Build: ${g.config.creative ? 'ON' : 'OFF'}`);
        this.renderPanel();
      }
    }, `Creative Free-Build: ${g.config.creative ? 'ON' : 'OFF'}`);

    const soundBtn = h('button', {
      style: 'padding: 8px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 1.5px solid var(--line); border-radius: 999px; width: 100%;',
      onclick: () => {
        sound.setMuted(!sound.isMuted());
        soundBtn.textContent = sound.isMuted() ? 'Sound FX: MUTED' : 'Sound FX: ENABLED';
      }
    }, sound.isMuted() ? 'Sound FX: MUTED' : 'Sound FX: ENABLED');

    const respawnTownBtn = h('button', {
      style: 'padding: 9px 16px; margin: 8px 0; background: rgba(50,38,28,0.9); border: 2px solid var(--gold); border-radius: 999px; width: 100%; font-weight: bold; color: var(--gold); cursor: pointer;',
      onclick: () => {
        g.teleportToTown();
        this.save();
        this.toast('Spawned at Abandoned Town!');
        this.close();
      }
    }, '🏘️ Respawn at Abandoned Town');

    const respawnRangeBtn = h('button', {
      style: 'padding: 9px 16px; margin: 8px 0; background: rgba(35,45,60,0.9); border: 2px solid #58b0f0; border-radius: 999px; width: 100%; font-weight: bold; color: #58b0f0; cursor: pointer;',
      onclick: () => {
        g.teleportToShootingRange();
        this.save();
        this.toast('🎯 Spawned inside Shooting Range (Armory & Target Lanes)!');
        this.close();
      }
    }, '🎯 Spawn at Shooting Range (Weapons & Ballistics)');

    const daylightBtn = h('button', {
      style: 'padding: 8px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 1.5px solid var(--line); border-radius: 999px; width: 100%; font-weight: bold;',
      onclick: () => {
        this.toggleDaylight();
        this.renderPanel();
      }
    }, g.clock.alwaysDay ? '☀️ Daylight: ALWAYS DAY (Night OFF)' : '🌙 Day/Night: CYCLING (Normal)');

    const staticBtn = h('button', {
      style: `padding: 9px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 2px solid ${g.world.staticActive ? 'var(--gold)' : 'var(--line)'}; border-radius: 999px; width: 100%; font-weight: bold;`,
      onclick: () => { this.setStaticMode(!g.world.staticActive); this.renderPanel(); }
    }, g.world.staticActive ? '🗺️ Static Map: ON (editable world)' : '🌍 Generative World (click for Static Map)');
    const editorBtn = h('button', {
      style: 'padding: 8px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 1.5px solid var(--line); border-radius: 999px; width: 100%;',
      onclick: () => this.toggleEditor()
    }, '🛠️ Open Map Editor [G]');

    const toastToggleBtn = h('button', {
      style: `padding: 9px 16px; margin: 8px 0; background: rgba(30,22,18,0.9); border: 2px solid ${this.toastMode === 'off' ? '#ef4444' : this.toastMode === 'important' ? 'var(--gold)' : 'var(--line)'}; border-radius: 999px; width: 100%; font-weight: bold; cursor: pointer;`,
      onclick: () => {
        this.toggleToastMode();
        this.renderPanel();
      }
    }, `💬 Toasts: ${this.toastMode === 'all' ? 'ALL (Calm & Deduplicated)' : this.toastMode === 'important' ? 'QUIET (Important Only)' : 'OFF (Muted)'} [F8]`);

    return h('div', {},
      h('h2', {}, 'Mode & Settings'),
      h('p', { class: 'hint' }, modeDesc),
      modeBtn,
      daylightBtn,
      toastToggleBtn,
      staticBtn,
      editorBtn,
      respawnTownBtn,
      respawnRangeBtn,
      creativeBtn,
      soundBtn,
      h('p', { class: 'hint', style: 'margin-top: 12px;' }, `World Seed: ${g.config.seed}`),
      h('p', { class: 'hint' }, 'Tip: Click Clock or press [L] anytime for Always Daylight · Press [F8] anytime to toggle Toasts!')
    );
  }

  private updateMinimap(): void {
    const cvs = this.el.minimap;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;
    const w = cvs.width, h = cvs.height, hw = w / 2, hh = h / 2;
    const g = this.game, pt = g.playerT;

    ctx.clearRect(0, 0, w, h);
    ctx.save();

    ctx.beginPath();
    ctx.arc(hw, hh, hw - 2, 0, Math.PI * 2);
    ctx.clip();

    ctx.fillStyle = '#14171a';
    ctx.fillRect(0, 0, w, h);

    const radiusMeters = 80;
    const scale = (hw - 4) / radiusMeters;

    const step = 8;
    for (let dy = -radiusMeters; dy <= radiusMeters; dy += step) {
      for (let dx = -radiusMeters; dx <= radiusMeters; dx += step) {
        if (dx * dx + dy * dy > radiusMeters * radiusMeters) continue;
        const wx = pt.x + dx, wy = pt.y + dy;
        const s = { terrain: g.world.terrainIdAt(wx, wy) };
        let col = '#44563a';
        if (s.terrain === 'deep') col = '#1b2c3a';
        else if (s.terrain === 'shallow') col = '#2c475d';
        else if (s.terrain === 'dirt') col = '#735841';
        else if (s.terrain === 'sand') col = '#a8986c';
        else if (s.terrain === 'snow') col = '#c2cbd3';
        else if (s.terrain === 'rock') col = '#5a5e64';
        ctx.fillStyle = col;
        ctx.fillRect(hw + (dx - step / 2) * scale, hh + (dy - step / 2) * scale, step * scale + 0.5, step * scale + 0.5);
      }
    }

    ctx.strokeStyle = '#8a6a4a';
    ctx.lineWidth = Math.max(2, 2.8 * scale);
    for (let dy = -radiusMeters; dy <= radiusMeters; dy += 4) {
      for (let dx = -radiusMeters; dx <= radiusMeters; dx += 4) {
        const wx = pt.x + dx, wy = pt.y + dy;
        const rType = getRoadType(g.world.seed, wx, wy, g.world.spawn);
        if (rType === 'highway') {
          ctx.fillStyle = '#fbbf24';
          ctx.fillRect(hw + dx * scale - 1.5, hh + dy * scale - 1.5, 3.5, 3.5);
        } else if (rType === 'dirt_road') {
          ctx.fillStyle = '#b45309';
          ctx.fillRect(hw + dx * scale - 1.5, hh + dy * scale - 1.5, 2.5, 2.5);
        } else if (rType === 'small_road') {
          ctx.fillStyle = '#94a3b8';
          ctx.fillRect(hw + dx * scale - 1, hh + dy * scale - 1, 2, 2);
        }
      }
    }

    const pcx = Math.floor(pt.x / 16), pcy = Math.floor(pt.y / 16);
    const cr = Math.ceil(radiusMeters / 16) + 1;
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let cy = pcy - cr; cy <= pcy + cr; cy++) {
      for (let cx = pcx - cr; cx <= pcx + cr; cx++) {
        const c = g.world.chunkAt(cx, cy);
        for (const feat of c.features) {
          const fdx = feat.x - pt.x, fdy = feat.y - pt.y;
          if (Math.hypot(fdx, fdy) < radiusMeters * 0.95) {
            const fx = hw + fdx * scale, fy = hh + fdy * scale;
            const icon = feat.type === 'gas_station' ? '⛽' : feat.type === 'radio_tower' ? '📡' : feat.type === 'town' ? '🏘️' : feat.type === 'military_outpost' ? '🛡️' : '⛺';
            ctx.fillText(icon, fx, fy);
          }
        }
      }
    }

    for (const ve of g.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const vt = g.ecs.get(ve, Comp.Transform)!;
      const vdx = vt.x - pt.x, vdy = vt.y - pt.y;
      if (Math.hypot(vdx, vdy) < radiusMeters) {
        ctx.fillStyle = '#5ac8fa';
        ctx.beginPath();
        ctx.arc(hw + vdx * scale, hh + vdy * scale, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0a0c10';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    for (const pe of g.ecs.query(Comp.Pet, Comp.Transform)) {
      const petT = g.ecs.get(pe, Comp.Transform)!;
      const petDx = petT.x - pt.x, petDy = petT.y - pt.y;
      if (Math.hypot(petDx, petDy) < radiusMeters) {
        ctx.fillStyle = '#f9c45a';
        ctx.beginPath();
        ctx.arc(hw + petDx * scale, hh + petDy * scale, 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0a0c10';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // discrete mineral node geodes
    for (const o of g.world.objectsNear(pt.x, pt.y, radiusMeters * 0.95)) {
      if (o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode') {
        const odx = o.x - pt.x, ody = o.y - pt.y;
        const col = o.kind === 'iron_node' ? '#8fa1b3' : o.kind === 'copper_node' ? '#c97746' : o.kind === 'coal_node' ? '#343a40' : o.kind === 'oil_seam' ? '#252a32' : '#b88ef2';
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(hw + odx * scale, hh + ody * scale, 2.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0a0c10';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    if (g.config.mode === 'zombie') {
      ctx.fillStyle = 'rgba(230,60,40,0.85)';
      for (const ze of g.ecs.query(Comp.Zombie, Comp.Transform)) {
        const zt = g.ecs.get(ze, Comp.Transform)!;
        const zdx = zt.x - pt.x, zdy = zt.y - pt.y;
        if (Math.hypot(zdx, zdy) < 45) {
          ctx.beginPath();
          ctx.arc(hw + zdx * scale, hh + zdy * scale, 2.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    ctx.save();
    ctx.translate(hw, hh);
    ctx.rotate(pt.angle);
    ctx.fillStyle = '#f08a32';
    ctx.beginPath();
    ctx.moveTo(6, 0);
    ctx.lineTo(-4, -4);
    ctx.lineTo(-2, 0);
    ctx.lineTo(-4, 4);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#0a0c10';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    ctx.restore();

    ctx.save();
    ctx.strokeStyle = 'rgba(233,225,204,0.45)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(hw, hh, hw - 2, 0, Math.PI * 2);
    ctx.stroke();

    ctx.font = '700 10px "Special Elite", monospace';
    ctx.fillStyle = '#f9c45a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('N', hw, 9);
    ctx.fillText('S', hw, h - 8);
    ctx.fillText('W', 9, hh);
    ctx.fillText('E', w - 8, hh);
    ctx.restore();
  }

  private mapPanel(): HTMLElement {
    const g = this.game;
    const pt = g.playerT;
    const curBiome = g.world.biomeAt(pt.x, pt.y);
    const biomeName = g.content.biomes.find((b) => b.id === curBiome)?.name ?? 'Wilderness';

    const coordsLabel = h('div', { class: 'map-coords' },
      `📍 Coordinates: X: ${Math.round(pt.x)} m, Y: ${Math.round(pt.y)} m · ${biomeName} · Explored: ${g.explored.size} sectors`
    );

    const canvas = h('canvas', { class: 'map-canvas', width: 680, height: 440 }) as HTMLCanvasElement;
    this.mapCanvas = canvas;

    canvas.onmousedown = (e: MouseEvent) => {
      this.mapDragging = true;
      this.mapDragStart = { x: e.clientX, y: e.clientY, px: this.mapPan.x, py: this.mapPan.y };
    };
    window.addEventListener('mousemove', (e: MouseEvent) => {
      if (!this.mapDragging || !this.mapCanvas) return;
      const dx = e.clientX - this.mapDragStart.x;
      const dy = e.clientY - this.mapDragStart.y;
      const scale = 0.85 * this.mapZoom;
      this.mapPan.x = this.mapDragStart.px - dx / scale;
      this.mapPan.y = this.mapDragStart.py - dy / scale;
      this.drawMapCanvas();
    });
    window.addEventListener('mouseup', () => { this.mapDragging = false; });

    canvas.onwheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.2 : 0.83;
      this.mapZoom = clamp(this.mapZoom * factor, 0.25, 4.0);
      this.drawMapCanvas();
    };

    let touchStart = { x: 0, y: 0, px: 0, py: 0 };
    canvas.ontouchstart = (e: TouchEvent) => {
      if (e.touches[0]) {
        touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY, px: this.mapPan.x, py: this.mapPan.y };
      }
    };
    canvas.ontouchmove = (e: TouchEvent) => {
      if (e.touches[0]) {
        const dx = e.touches[0].clientX - touchStart.x;
        const dy = e.touches[0].clientY - touchStart.y;
        const scale = 0.85 * this.mapZoom;
        this.mapPan.x = touchStart.px - dx / scale;
        this.mapPan.y = touchStart.py - dy / scale;
        this.drawMapCanvas();
      }
    };

    const maxViewBtn = h('button', {
      style: 'padding: 5px 12px; margin-right: 6px; background: rgba(30,50,70,0.9); border: 1.5px solid #38bdf8; color: #38bdf8; font-weight: bold; cursor: pointer;',
      onclick: () => {
        this.renderer.cam.toggleMaxView();
        this.toast(this.renderer.cam.targetZoom <= 12 ? '🌐 Max View Distance Enabled (Zoom 8)' : '🔍 Normal View Distance (Zoom 54)');
      }
    }, '🌐 Max View');

    const townBtn = h('button', {
      style: 'padding: 5px 11px; margin-right: 6px; background: rgba(50,38,28,0.9); border: 1.5px solid var(--gold); color: var(--gold); font-weight: bold; cursor: pointer;',
      onclick: () => {
        g.teleportToTown();
        this.mapPan.x = g.playerT.x;
        this.mapPan.y = g.playerT.y;
        this.drawMapCanvas();
        this.save();
        this.toast('Spawned at Abandoned Town!');
      }
    }, '🏘️ Spawn at Town');

    const centerBtn = h('button', {
      style: 'padding: 5px 12px; margin-right: 6px;',
      onclick: () => {
        this.mapPan.x = g.playerT.x;
        this.mapPan.y = g.playerT.y;
        this.drawMapCanvas();
      }
    }, '📍 Center You');

    const zoomInBtn = h('button', {
      style: 'padding: 5px 10px; margin-right: 4px;',
      onclick: () => { this.mapZoom = Math.min(4.0, this.mapZoom * 1.3); this.drawMapCanvas(); }
    }, '➕');

    const zoomOutBtn = h('button', {
      style: 'padding: 5px 10px; margin-right: 6px;',
      onclick: () => { this.mapZoom = Math.max(0.25, this.mapZoom / 1.3); this.drawMapCanvas(); }
    }, '➖');

    const fogBtn = h('button', {
      style: `padding: 5px 12px; margin-right: 6px; background: rgba(35,22,45,0.9); border: 1.5px solid #a855f7; color: ${g.fogOfWar ? '#e9d5ff' : '#4ade80'}; font-weight: bold; cursor: pointer;`,
      onclick: () => {
        const on = g.toggleFogOfWar();
        this.drawMapCanvas();
        this.toast(on ? 'Fog of War: ON' : '👁️ Fog of War: OFF (World Fully Revealed)');
        this.renderPanel();
      }
    }, g.fogOfWar ? '👁️ Fog: ON' : '👁️ Fog: OFF');

    setTimeout(() => this.drawMapCanvas(), 10);

    return h('div', {},
      h('div', { class: 'map-header' },
        h('h2', { style: 'margin: 0;' }, 'Continental World Map'),
        h('div', {}, fogBtn, maxViewBtn, townBtn, centerBtn, zoomInBtn, zoomOutBtn)
      ),
      coordsLabel,
      canvas,
      h('p', { class: 'hint' }, '⛺ Camp  ·  ⛽ Gas Station  ·  📡 Radio Tower  ·  🏘️ Town  ·  🛡️ Outpost  ·  🚗 Vehicle · Pale terrain = higher elevation'),
      h('p', { class: 'hint', style: 'color: var(--gold);' }, 'Tip: Drag to pan · Scroll wheel to zoom · Press [F] or click [Fog: OFF] to reveal entire world!')
    );
  }

  private drawMapCanvas(): void {
    const cvs = this.mapCanvas;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;
    const w = cvs.width, h = cvs.height, hw = w / 2, hh = h / 2;
    const g = this.game, pt = g.playerT;
    const scale = 0.85 * this.mapZoom;

    ctx.fillStyle = '#101214';
    ctx.fillRect(0, 0, w, h);

    const leftWorld = this.mapPan.x - hw / scale;
    const rightWorld = this.mapPan.x + hw / scale;
    const topWorld = this.mapPan.y - hh / scale;
    const bottomWorld = this.mapPan.y + hh / scale;

    const minCx = Math.floor(leftWorld / 16) - 1;
    const maxCx = Math.floor(rightWorld / 16) + 1;
    const minCy = Math.floor(topWorld / 16) - 1;
    const maxCy = Math.floor(bottomWorld / 16) + 1;

    for (let cy = minCy; cy <= maxCy; cy++) {
      for (let cx = minCx; cx <= maxCx; cx++) {
        const k = `${cx},${cy}`;
        const isExp = !g.fogOfWar || g.explored.has(k);
        const scrX = hw + (cx * 16 - this.mapPan.x) * scale;
        const scrY = hh + (cy * 16 - this.mapPan.y) * scale;
        const chunkPx = 16 * scale;

        if (isExp) {
          const sample = { terrain: g.world.terrainIdAt(cx * 16 + 8, cy * 16 + 8), elevation: g.world.elevationAt(cx * 16 + 8, cy * 16 + 8) };
          let col = '#3b823e'; // Vibrant Emerald Meadow
          if (sample.terrain === 'deep') col = '#0f3b4c'; // Deep Navy Ocean
          else if (sample.terrain === 'shallow') col = '#06b6d4'; // Tropical Cyan Lagoon
          else if (sample.terrain === 'dirt') col = '#78593d'; // Warm Fertile Earth
          else if (sample.terrain === 'sand') col = '#d99b45'; // Radiant Golden Sand
          else if (sample.terrain === 'snow') col = '#e2effa'; // Crisp Glacial Snow
          else if (sample.terrain === 'rock') col = '#5a6577'; // Granite Slate Blue
          else if (sample.terrain === 'needles') col = '#624a35'; // Pine Needle Mulch
          else if (sample.terrain === 'moss') col = '#2d6a4f'; // Alpine Moss
          else if (sample.terrain === 'ash') col = '#34333b'; // Volcanic Ash
          ctx.fillStyle = col;
          ctx.fillRect(scrX, scrY, chunkPx + 0.5, chunkPx + 0.5);
          ctx.fillStyle = sample.elevation > 0.56
            ? `rgba(255, 255, 255, ${Math.min(0.25, (sample.elevation - 0.56) * 0.6)})`
            : `rgba(0, 0, 0, ${Math.min(0.18, (0.44 - sample.elevation) * 0.5)})`;
          ctx.fillRect(scrX, scrY, chunkPx + 0.5, chunkPx + 0.5);

          const rType = getRoadType(g.world.seed, cx * 16 + 8, cy * 16 + 8, g.world.spawn);
          if (rType === 'highway') {
            ctx.fillStyle = '#f59e0b'; // Amber Gold Highway
            ctx.fillRect(scrX, scrY + chunkPx * 0.32, chunkPx, chunkPx * 0.36);
          } else if (rType === 'dirt_road') {
            ctx.fillStyle = '#92400e'; // Warm Earth Dirt Road
            ctx.fillRect(scrX, scrY + chunkPx * 0.36, chunkPx, chunkPx * 0.28);
          } else if (rType === 'small_road') {
            ctx.fillStyle = '#64748b'; // Slate Cobblestone Street
            ctx.fillRect(scrX, scrY + chunkPx * 0.40, chunkPx, chunkPx * 0.20);
          }

          const c = g.world.chunkAt(cx, cy);
          for (const feat of c.features) {
            const fx = hw + (feat.x - this.mapPan.x) * scale;
            const fy = hh + (feat.y - this.mapPan.y) * scale;
            ctx.font = feat.type === 'town' ? '18px sans-serif' : '15px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const icon = feat.type === 'gas_station' ? '⛽' :
              feat.type === 'radio_tower' ? '📡' :
              feat.type === 'town' ? '🏛️' :
              feat.type === 'town_district' ? '🏘️' :
              feat.type === 'military_outpost' ? '🛡️' : '⛺';
            ctx.fillText(icon, fx, fy);
            if (this.mapZoom > 0.75) {
              ctx.font = feat.type === 'town' ? 'bold 11px "Special Elite", monospace' : '10px "Special Elite", monospace';
              ctx.fillStyle = feat.type === 'town' ? '#fde047' : 'rgba(233,225,204,0.9)';
              ctx.fillText(feat.name ?? feat.type, fx, fy + 13);
            }
          }

          if (this.mapZoom > 0.65) {
            for (const o of c.objects) {
              if (o.kind === 'iron_node' || o.kind === 'copper_node' || o.kind === 'coal_node' || o.kind === 'oil_seam' || o.kind === 'crystal_geode') {
                const ox = hw + (o.x - this.mapPan.x) * scale;
                const oy = hh + (o.y - this.mapPan.y) * scale;
                if (ox >= 0 && ox <= w && oy >= 0 && oy <= h) {
                  const icon = o.kind === 'iron_node' ? '⚪' : o.kind === 'copper_node' ? '🟠' : o.kind === 'coal_node' ? '⚫' : o.kind === 'oil_seam' ? '🛢️' : '🟣';
                  ctx.font = '11px sans-serif';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillText(icon, ox, oy);
                }
              }
            }
          }
        } else {
          ctx.fillStyle = '#15171a';
          ctx.fillRect(scrX, scrY, chunkPx + 0.5, chunkPx + 0.5);
          ctx.strokeStyle = 'rgba(255,255,255,0.03)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(scrX, scrY);
          ctx.lineTo(scrX + chunkPx, scrY + chunkPx);
          ctx.stroke();
        }
      }
    }

    // Risk zone topology overlay
    const sx = hw + (g.world.spawn.x - this.mapPan.x) * scale;
    const sy = hh + (g.world.spawn.y - this.mapPan.y) * scale;
    // Green Zone (<65m)
    ctx.beginPath();
    ctx.arc(sx, sy, 65 * scale, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(46, 204, 113, 0.45)';
    ctx.lineWidth = 2;
    ctx.stroke();
    // Yellow Zone (<200m)
    ctx.beginPath();
    ctx.arc(sx, sy, 200 * scale, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(243, 156, 18, 0.45)';
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);

    // Clan territory totems & active shields
    for (const te of g.ecs.query(Comp.Totem, Comp.Transform)) {
      const tt = g.ecs.get(te, Comp.Transform)!;
      const tot = g.ecs.get(te, Comp.Totem)!;
      const tx = hw + (tt.x - this.mapPan.x) * scale;
      const ty = hh + (tt.y - this.mapPan.y) * scale;
      ctx.beginPath();
      ctx.arc(tx, ty, 30 * scale, 0, Math.PI * 2);
      ctx.strokeStyle = tot.shieldActive ? 'rgba(56, 189, 248, 0.65)' : 'rgba(239, 68, 68, 0.65)';
      ctx.lineWidth = 1.8;
      ctx.stroke();
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('🛡️', tx, ty);
    }

    // Livestock & wildlife
    for (const ae of g.ecs.query(Comp.Animal, Comp.Transform)) {
      const at = g.ecs.get(ae, Comp.Transform)!;
      const an = g.ecs.get(ae, Comp.Animal)!;
      const ax = hw + (at.x - this.mapPan.x) * scale;
      const ay = hh + (at.y - this.mapPan.y) * scale;
      if (ax >= 0 && ax <= w && ay >= 0 && ay <= h) {
        ctx.fillStyle = an.tamed ? '#4ade80' : '#84cc16';
        ctx.beginPath();
        ctx.arc(ax, ay, 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#052e16';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    for (const ve of g.ecs.query(Comp.Vehicle, Comp.Transform)) {
      const vt = g.ecs.get(ve, Comp.Transform)!;
      const vx = hw + (vt.x - this.mapPan.x) * scale;
      const vy = hh + (vt.y - this.mapPan.y) * scale;
      ctx.fillStyle = '#5ac8fa';
      ctx.beginPath();
      ctx.arc(vx, vy, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#0a0c10';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    for (const pe of g.ecs.query(Comp.Pet, Comp.Transform)) {
      const petT = g.ecs.get(pe, Comp.Transform)!;
      const px = hw + (petT.x - this.mapPan.x) * scale;
      const py = hh + (petT.y - this.mapPan.y) * scale;
      ctx.fillStyle = '#f9c45a';
      ctx.beginPath();
      ctx.arc(px, py, 4.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#0a0c10';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    const px = hw + (pt.x - this.mapPan.x) * scale;
    const py = hh + (pt.y - this.mapPan.y) * scale;
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(pt.angle);
    ctx.fillStyle = '#f08a32';
    ctx.beginPath();
    ctx.moveTo(9, 0);
    ctx.lineTo(-6, -6);
    ctx.lineTo(-3, 0);
    ctx.lineTo(-6, 6);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();

    const barMeters = 100;
    const barPx = barMeters * scale;
    ctx.fillStyle = 'rgba(233,225,204,0.7)';
    ctx.fillRect(16, h - 22, barPx, 3);
    ctx.font = '10px "Special Elite", monospace';
    ctx.fillText(`${barMeters}m`, 16 + barPx / 2, h - 28);
  }
}
