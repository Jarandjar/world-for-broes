import { Game, DEFAULT_CONFIG } from '../src/sim/game';
const g = new Game({ ...DEFAULT_CONFIG, creative: false });
const trees = g.world.objectsNear(g.world.spawn.x, g.world.spawn.y, 40).filter((o) => o.kind === 'pine' || o.kind === 'oak').sort((a, b) => Math.hypot(a.x - g.world.spawn.x, a.y - g.world.spawn.y) - Math.hypot(b.x - g.world.spawn.x, b.y - g.world.spawn.y));
const tree = trees[0];
const t = g.playerT; t.x = tree.x - 0.9; t.y = tree.y; t.angle = 0;

const slot = g.playerInv.inv.active;
const def = slot ? g.world.content.items.get(slot.item) : undefined;
console.log('Active slot:', slot, 'Item def:', def);

const cand = g.world.objectsNear(t.x, t.y, 2.8);
console.log('Nearby cand count:', cand.length);
for (const o of cand) {
  const oDef = g.world.content.objects.get(o.kind);
  const dx = o.x - t.x, dy = o.y - t.y;
  const d = Math.hypot(dx, dy);
  const cos = (dx * Math.cos(t.angle) + dy * Math.sin(t.angle)) / (d || 1);
  console.log('Cand:', o.kind, 'dist:', d, 'cos:', cos, 'tool:', oDef?.tool, 'hp:', o.hp);
}
