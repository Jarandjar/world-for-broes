/**
 * Arena bot levels (ARENA.md §5). One row per level, read by the web sim (`Game.spawnArenaBot`) and exported
 * to Unreal by `npm run ue:export` (docs/UE_PIPELINE.md §4.1), so both games use the same numbers.
 * Units: sight in metres, aimSpread in radians (half-angle of the miss cone), times in seconds, speed in m/s.
 */
export type ArenaBotLevel = 'scarecrow' | 'regular' | 'sharp' | 'menace';

export interface ArenaBotDef {
  /** Seconds from first seeing a target to the first shot. */
  reaction: number;
  sight: number;
  aimSpread: number;
  shotInterval: number;
  speed: number;
  hp: number;
}

export const ARENA_BOT_LEVEL_IDS: readonly ArenaBotLevel[] = ['scarecrow', 'regular', 'sharp', 'menace'];

export const ARENA_BOT_LEVELS: Record<ArenaBotLevel, ArenaBotDef> = {
  scarecrow: { reaction: 0.9, sight: 16, aimSpread: 0.21, shotInterval: 1.2, speed: 2.2, hp: 50 },
  regular: { reaction: 0.6, sight: 20, aimSpread: 0.12, shotInterval: 0.8, speed: 2.8, hp: 65 },
  sharp: { reaction: 0.35, sight: 25, aimSpread: 0.06, shotInterval: 0.55, speed: 3.1, hp: 80 },
  menace: { reaction: 0.2, sight: 30, aimSpread: 0.026, shotInterval: 0.35, speed: 3.5, hp: 100 },
};
