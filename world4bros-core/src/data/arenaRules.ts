/**
 * Arena match rules (ARENA.md §4, Proposed starting values). Read by the Unreal arena through
 * `npm run ue:export` (docs/UE_PIPELINE.md §4.1); the web arena's rounds (Z28b) should read the same rows.
 * Times in seconds, distances in metres.
 */
export const ARENA_MATCH_RULES = {
  duel: {
    roundTargets: [3, 5, 7, 10],
    defaultRoundTarget: 5,
    /** Pause after a kill before both fighters reset to their spawns. */
    roundPause: 2,
    /** A round with no kill for this long is a draw and resets. */
    drawAfter: 30,
    /** Match time limit; the leader wins when it runs out. */
    matchTimeLimit: 480,
  },
  ffa: {
    minPlayers: 4,
    maxPlayers: 8,
    killLimits: [10, 15, 25],
    defaultKillLimit: 15,
    timeLimitsMinutes: [4, 6, 8],
    defaultTimeLimitMinutes: 6,
    respawnDelay: 3,
    /** Respawn at the spawn farthest from enemies, never within this distance of one. */
    safeSpawnDistance: 12,
    /** Ends early when the protected player fires. */
    spawnProtection: 2,
  },
  pickups: {
    pieHeal: 25,
    pieRespawn: 20,
  },
  /** Guns offered in the arena loadout picker (ids from src/data/items.ts). */
  loadout: ['hunting_rifle', 'pump_shotgun', 'nailgun_smg', 'potato_cannon', 'slingshot'],
} as const;
