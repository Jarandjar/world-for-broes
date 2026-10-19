import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Game, DEFAULT_CONFIG } from '../src/sim/game';
import { WALL_T, DOOR_W, insideBuilding, wallRects, getStreetDetails, CHUNK } from '../src/sim/worldgen';
import type { Building } from '../src/sim/worldgen';

function townBuildings(g: Game): Building[] {
  const sp = g.world.spawn;
  return g.world.buildingsNear(sp.x, sp.y, 50);
}
const doorPoint = (b: Building): [number, number] =>
  b.door === 'N' ? [b.x + b.doorAt, b.y + WALL_T / 2] : b.door === 'S' ? [b.x + b.doorAt, b.y + b.h - WALL_T / 2]
    : b.door === 'W' ? [b.x + WALL_T / 2, b.y + b.doorAt] : [b.x + b.w - WALL_T / 2, b.y + b.doorAt];

test('the spawn town is populated with non-overlapping buildings that avoid streets', () => {
  const g = new Game(DEFAULT_CONFIG);
  const bs = townBuildings(g);
  assert.ok(bs.length >= 12, `expected a populated town, got ${bs.length}`);
  for (const b of bs) {
    for (let y = b.y; y < b.y + b.h; y++) for (let x = b.x; x < b.x + b.w; x++) {
      assert.equal(getStreetDetails(g.world.seed, x + 0.5, y + 0.5, g.world.spawn).kind, 'none', `${b.id} sits on a street`);
    }
  }
  for (let i = 0; i < bs.length; i++) for (let j = i + 1; j < bs.length; j++) {
    const a = bs[i]!, b = bs[j]!;
    assert.ok(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y, `${a.id} overlaps ${b.id}`);
  }
  const sp = g.world.spawn;
  assert.ok(!g.world.blocked(sp.x, sp.y, 0.3), 'spawn must stay free');
});

test('walls block movement but the door and interior are open; the roof lifts when inside', () => {
  const g = new Game(DEFAULT_CONFIG);
  const b = townBuildings(g)[0]!;
  const interior = g.world.blocked(b.x + b.w / 2, b.y + b.h / 2, 0.3);
  assert.equal(interior, false, 'interior is walkable');
  const [dx, dy] = doorPoint(b);
  assert.equal(g.world.blocked(dx, dy, 0.3), false, 'door gap is walkable');
  const wall = wallRects(b).find((r) => r[2] - r[0] > 1 && r[3] - r[1] > 1 === false || r[3] - r[1] > 1)!;
  const mx = (wall[0] + wall[2]) / 2, my = (wall[1] + wall[3]) / 2;
  assert.equal(g.world.blocked(mx, my, 0.3), true, 'wall segment blocks');
  assert.equal(insideBuilding(b, b.x + b.w / 2, b.y + b.h / 2), true);
  assert.equal(insideBuilding(b, b.x - 3, b.y - 3), false);
  assert.ok(DOOR_W > 1.2, 'door wide enough for the player');
});

test('buildings are deterministic per seed', () => {
  const a = townBuildings(new Game(DEFAULT_CONFIG)).map((b) => `${b.def}:${b.x},${b.y},${b.w},${b.h},${b.door}`).sort();
  const b = townBuildings(new Game(DEFAULT_CONFIG)).map((x) => `${x.def}:${x.x},${x.y},${x.w},${x.h},${x.door}`).sort();
  assert.deepEqual(a, b);
});

test('building palettes are picked deterministically and vary across a town', async () => {
  const { BUILDINGS, buildingColors } = await import('../src/data/buildings');
  const bs = townBuildings(new Game(DEFAULT_CONFIG));
  const roofs = new Set<string>();
  for (const b of bs) {
    const d = BUILDINGS.find((x) => x.id === b.def)!;
    const c = buildingColors(d, b.x, b.y);
    assert.deepEqual(c, buildingColors(d, b.x, b.y), 'same building, same colours');
    assert.ok(c.roof === d.colors.roof || (d.palettes ?? []).some((p) => p.roof === c.roof));
    roofs.add(c.roof);
  }
  assert.ok(roofs.size >= 4, `expected varied roof colours, got ${roofs.size}`);
});

test('static map keeps, places and erases buildings across save/load', () => {
  const g = new Game(DEFAULT_CONFIG);
  const sp = g.world.spawn;
  g.world.setStatic(true);
  const count = townBuildings(g).length;
  assert.ok(count > 0);
  const placed = g.world.placeBuilding('cottage', Math.round(sp.x) + 60, Math.round(sp.y) + 60, 6, 6, 'E');
  assert.ok(placed);
  assert.equal(townBuildings(g).length, count, 'far away building is outside the town query');
  assert.equal(g.world.blocked(placed!.x + 0.2, placed!.y + 3, 0.2), true, 'placed walls collide');
  const g2 = Game.fromSave(JSON.parse(JSON.stringify(g.toSave())));
  assert.equal(townBuildings(g2).length, count, 'town buildings survive save/load');
  assert.equal(g2.world.blocked(placed!.x + 0.2, placed!.y + 3, 0.2), true, 'edited building survives save/load');
  assert.ok(g.world.eraseBuildings(placed!.x + 3, placed!.y + 3) >= 1);
  assert.equal(g.world.blocked(placed!.x + 0.2, placed!.y + 3, 0.2), false);
  void CHUNK;
});
