import { Game, DEFAULT_CONFIG } from '../src/sim/game';
const g = new Game({ ...DEFAULT_CONFIG, creative: false });
console.log('spawn:', g.world.spawn);
const scx = Math.floor(g.world.spawn.x / 16), scy = Math.floor(g.world.spawn.y / 16);
console.log('scx, scy:', scx, scy);
const c = g.world.chunkAt(scx, scy);
console.log('objects in spawn chunk count:', c.objects.length);
for (const o of c.objects) {
  console.log('obj:', o.kind, o.x, o.y);
}
