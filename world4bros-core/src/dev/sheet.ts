/** Character lab: renders a grid of characters. Open dev/sheet.html (after `npm run dev`) to inspect looks and animations. */
import { buildContent } from '../data/content';
import { makeRng } from '../engine/rng';
import { defaultLook, randomLook, equip, type Look } from '../sim/look';
import { SpriteCache } from '../render/sprites';
import { CharacterRenderer, type Pose } from '../render/characterRenderer';

const content = buildContent();
const cache = new SpriteCache();
const cr = new CharacterRenderer(cache, content);
const canvas = document.getElementById('c') as HTMLCanvasElement;
const g = canvas.getContext('2d') as CanvasRenderingContext2D;
const qs = new URLSearchParams(location.search);
const scale = parseFloat(qs.get('s') ?? '150');
const cols = parseInt(qs.get('cols') ?? '5', 10), rows = parseInt(qs.get('rows') ?? '3', 10);
canvas.width = cols * scale * 1.4; canvas.height = rows * scale * 1.4;

const rng = makeRng(11);
const looks: Look[] = [defaultLook()];
const hairy = defaultLook(); hairy.appearance.hairColor = '#a8456f'; hairy.appearance.hairStyle = 'long'; equip(hairy, content, 'head', null); equip(hairy, content, 'torso', 'trench_coat'); equip(hairy, content, 'neck', 'scarf'); looks.push(hairy);
const sneaky = defaultLook(); equip(sneaky, content, 'head', 'cone_hat'); equip(sneaky, content, 'mainHand', 'rubber_chicken'); equip(sneaky, content, 'torso', 'hivis_vest'); sneaky.appearance.hairStyle = 'mohawk'; looks.push(sneaky);
while (looks.length < cols * rows) looks.push(randomLook(content, rng));

function frame(t: number): void {
  g.fillStyle = '#4b5249'; g.fillRect(0, 0, canvas.width, canvas.height);
  looks.forEach((look, i) => {
    const cx = (i % cols + 0.5) * scale * 1.4, cy = (Math.floor(i / cols) + 0.5) * scale * 1.4;
    const mode = i % 3;
    const pose: Pose = { phase: t * 5 + i, walk: mode === 1 ? 1 : 0, swing: mode === 2 ? ((t * 0.9 + i * 0.13) % 1.4) / 1.0 > 1 ? -1 : (t * 0.9 + i * 0.13) % 1.4 : -1, time: t };
    g.save(); g.translate(cx, cy); g.scale(scale, scale); g.rotate(qs.has('still') ? (i % 8) * 0.785 : t * 0.5 + i);
    cr.draw(g, look, pose, 192);
    g.restore();
  });
}
if (qs.has('still')) { frame(parseFloat(qs.get('t') ?? '1')); (window as unknown as { __done: boolean }).__done = true; }
else { const loop = (ms: number): void => { frame(ms / 1000); requestAnimationFrame(loop); }; requestAnimationFrame(loop); }
