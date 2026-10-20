import { Game, DEFAULT_CONFIG, type GameConfig, NO_INPUT } from './sim/game';
import { Input } from './engine/input';
import { LocalStorageAdapter } from './engine/save';
import { Renderer } from './render/renderer';
import { Ui, SAVE_KEY } from './ui/ui';
import { buildContent } from './data/content';
import { validateContent } from './data/validate';
import { demoBase } from './dev/demo';

const qs = new URLSearchParams(location.search);
const canvas = document.getElementById('game') as HTMLCanvasElement;
const storage = new LocalStorageAdapter();

const problems = validateContent(buildContent());
if (problems.length) console.warn('Content problems:', problems);

const config: GameConfig = {
  ...DEFAULT_CONFIG,
  seed: parseInt(qs.get('seed') ?? '', 10) || DEFAULT_CONFIG.seed,
  creative: qs.get('creative') !== '0',
  mode: qs.get('mode') === 'zombie' ? 'zombie' : 'classic',
  alwaysDay: qs.get('cycle') === '1' ? false : true,
};
let game: Game;
const saved = qs.has('fresh') || qs.has('still') ? null : storage.load(SAVE_KEY);
try { game = saved ? Game.fromSave(JSON.parse(saved)) : new Game(config); } catch { game = new Game(config); }
if (qs.has('night')) {
  game.clock.alwaysDay = false;
  game.clock.t = 0.8;
} else if (qs.has('cycle')) {
  game.clock.alwaysDay = false;
} else if (qs.has('day')) {
  game.clock.alwaysDay = true;
  game.clock.t = 0.25;
}
if (qs.has('time')) game.clock.t = parseFloat(qs.get('time') ?? '0.22');
if (qs.has('demo')) demoBase(game);
if (qs.has('town')) game.teleportToTown();
if (qs.has('range') || qs.has('shooting')) game.teleportToShootingRange();
if (qs.has('fleet') || qs.has('vehicles')) game.spawnFleet();
if (qs.has('arena')) {
  const arenaId = (qs.get('arena') || 'duel_pit') as 'duel_pit' | 'ffa_yard';
  game.teleportToArena(arenaId, 0);
  if (qs.has('bots') || qs.has('duel')) {
    const count = parseInt(qs.get('bots') || '1', 10);
    const lvl = (qs.get('level') || 'regular') as any;
    const gun = (qs.get('gun') || 'nailgun_smg') as any;
    for (let i = 0; i < count; i++) {
      game.spawnArenaBot(lvl, gun, i + 1);
    }
  }
  if (qs.has('zombies') || qs.has('swarm')) {
    const zCount = parseInt(qs.get('zombies') || '6', 10);
    const zKind = (qs.get('zkind') || 'runner') as any;
    game.spawnArenaZombies(zCount, zKind);
  }
}

const input = new Input(canvas);
const renderer = new Renderer(canvas, game);
if (qs.has('zoom')) { renderer.cam.targetZoom = renderer.cam.zoom = parseFloat(qs.get('zoom') ?? '54'); }
const ui = new Ui(game, renderer, input, storage);
if (qs.has('panel')) ui.open(qs.get('panel') as 'inventory' | 'craft' | 'build' | 'customizer');
if (qs.has('build')) { ui.open('build'); ui.buildId = qs.get('build'); }
(window as unknown as { __game: Game }).__game = game;

// touch controls
const touch = document.getElementById('touch') as HTMLElement;
let touchInitialized = false;
function initTouch(): void {
  if (touchInitialized || !touch) return;
  touchInitialized = true;
  touch.classList.add('on');
  const stick = touch.querySelector('.stick') as HTMLElement, knob = touch.querySelector('.knob') as HTMLElement;
  let id = -1, cx = 0, cy = 0;
  stick.addEventListener('pointerdown', (e) => {
    id = e.pointerId;
    stick.setPointerCapture(id);
    const r = stick.getBoundingClientRect();
    cx = r.left + r.width / 2;
    cy = r.top + r.height / 2;
  });
  stick.addEventListener('pointermove', (e) => {
    if (e.pointerId !== id) return;
    let dx = (e.clientX - cx) / 50, dy = (e.clientY - cy) / 50;
    const m = Math.hypot(dx, dy);
    if (m > 1) { dx /= m; dy /= m; }
    input.stick.x = Math.abs(dx) > 0.15 ? dx : 0;
    input.stick.y = Math.abs(dy) > 0.15 ? dy : 0;
    knob.style.transform = `translate(${dx * 34}px, ${dy * 34}px)`;
  });
  const end = (): void => { id = -1; input.stick.x = input.stick.y = 0; knob.style.transform = ''; };
  stick.addEventListener('pointerup', end);
  stick.addEventListener('pointercancel', end);
  (touch.querySelector('.act') as HTMLElement).addEventListener('pointerdown', (e) => { e.preventDefault(); input.touchAction = true; });
  (touch.querySelector('.use') as HTMLElement).addEventListener('pointerdown', (e) => { e.preventDefault(); input.touchUse = true; });
}

if (matchMedia('(pointer: coarse)').matches || ('ontouchstart' in window) || (navigator.maxTouchPoints > 0)) {
  initTouch();
} else {
  window.addEventListener('touchstart', initTouch, { once: true });
}

// fixed-step simulation, variable-rate rendering
const STEP = 1 / 60;
let acc = 0, last = performance.now(), fps = 60, autosave = 0;
function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  fps += (1 / Math.max(dt, 1e-4) - fps) * 0.05;
  acc += dt;
  const simIn = ui.simInput();
  let first = true;
  while (acc >= STEP) { game.step(STEP, first ? simIn : { ...NO_INPUT, move: simIn.move, sprint: simIn.sprint }); first = false; acc -= STEP; }
  ui.frame(dt, fps);
  renderer.render(dt, ui.renderState());
  input.endFrame();
  autosave += dt; if (autosave > 30) { autosave = 0; storage.save(SAVE_KEY, JSON.stringify(game.toSave())); }
  requestAnimationFrame(frame);
}
window.addEventListener('beforeunload', () => storage.save(SAVE_KEY, JSON.stringify(game.toSave())));

if (qs.has('still')) {
  for (let i = 0; i < 90; i++) game.step(STEP, NO_INPUT);
  renderer.cam.snap(game.playerT.x, game.playerT.y);
  for (let i = 0; i < 6; i++) { ui.frame(0.016, 60); renderer.render(0.016, ui.renderState()); }
  (window as unknown as { __done: boolean }).__done = true;
} else requestAnimationFrame(frame);
