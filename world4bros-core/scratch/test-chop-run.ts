import { Game, DEFAULT_CONFIG, NO_INPUT } from '../src/sim/game';
const g = new Game({ ...DEFAULT_CONFIG, creative: false });
const tree = g.world.objectsNear(g.world.spawn.x, g.world.spawn.y, 40).filter((o) => o.kind === 'pine' || o.kind === 'oak').sort((a, b) => Math.hypot(a.x - g.world.spawn.x, a.y - g.world.spawn.y) - Math.hypot(b.x - g.world.spawn.x, b.y - g.world.spawn.y))[0];
console.log('Tree chosen:', tree);
const t = g.playerT; t.x = tree.x - 0.9; t.y = tree.y; t.angle = 0;
const before = g.playerInv.inv.count('wood');
console.log('Before wood:', before);

for (let swing = 0; swing < 12; swing++) {
  const target = g.findActionTarget();
  console.log(`Swing ${swing} target:`, target ? target.kind : 'NONE', target?.kind === 'object' ? target.obj.kind : '');
  g.step(1 / 60, { ...NO_INPUT, actionPressed: true });
  for (let i = 0; i < 40; i++) g.step(1 / 60, NO_INPUT);
  console.log('Tree hp:', g.world.getObject(tree.id)?.hp, 'kind:', g.world.getObject(tree.id)?.kind);
}
console.log('After wood:', g.playerInv.inv.count('wood'));
