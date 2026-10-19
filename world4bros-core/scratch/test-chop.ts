import { Game, DEFAULT_CONFIG } from '../src/sim/game';
const g = new Game({ ...DEFAULT_CONFIG, creative: false });
const trees = g.world.objectsNear(g.world.spawn.x, g.world.spawn.y, 40).filter((o) => o.kind === 'pine' || o.kind === 'oak').sort((a, b) => Math.hypot(a.x - g.world.spawn.x, a.y - g.world.spawn.y) - Math.hypot(b.x - g.world.spawn.x, b.y - g.world.spawn.y));
console.log('Trees near spawn:', trees.length, trees.map(t => ({ id: t.id, kind: t.kind, x: t.x, y: t.y })));
const tree = trees[0];
const t = g.playerT; t.x = tree.x - 0.9; t.y = tree.y; t.angle = 0;
console.log('Player at:', t.x, t.y, 'Target:', tree.x, tree.y);
console.log('Action target:', g.findActionTarget());
