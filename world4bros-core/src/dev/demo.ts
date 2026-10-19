import type { Game } from '../sim/game';
/** Builds a small walled camp next to the spawn so the base core can be seen immediately (used by ?demo=1). */
export function demoBase(game: Game): void {
  const px = Math.floor(game.playerT.x), py = Math.floor(game.playerT.y);
  for (const o of game.world.objectsNear(px + 6, py - 1, 9)) game.world.damageObject(o.id, 999);
  const b = (id: string, x: number, y: number): void => { game.build(id, px + x, py + y); };
  b('claim_flag', 2, -2);
  for (let x = 4; x <= 10; x++) for (let y = -5; y <= 0; y++) if (x > 4 && x < 10 && y > -5 && y < 0) b('floor_wood', x, y);
  for (let x = 4; x <= 10; x++) { b('wall_wood', x, -5); if (x !== 7) b('wall_wood', x, 0); }
  b('door_wood', 7, 0);
  for (let y = -4; y <= -1; y++) { b('wall_wood', 4, y); b('wall_wood', 10, y); }
  b('chest', 5, -4); b('workbench', 6, -4); b('bed', 9, -4); b('campfire', 7, -2);
  b('torch_post', 3, 1); b('torch_post', 11, 1); b('campfire', 1, 1);
}
