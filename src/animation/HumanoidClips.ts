import { AnimationClip } from './AnimationClip';

/**
 * Shared Humanoid Rig Clips (Section 64) for World 4 Bro's.
 * Defines all 16 mandatory core clips with authored keyframe tracks for:
 * `torso`, `head`, `backpack`, `leftArm`, `rightArm`, `leftLeg`, `rightLeg`, `heldTool`.
 */

export function createHumanoidClips(): Record<string, AnimationClip> {
  const clips: Record<string, AnimationClip> = {};

  // 1. IDLE (Duration: 1.6s, Loop: true) - Gentle breathing and personality fidget
  clips['idle'] = new AnimationClip('idle', 1.6, true)
    .addTrack('torso', [
      { time: 0.0, transform: { posY: 0.0, scaleY: 1.0, scaleX: 1.0 } },
      { time: 0.8, transform: { posY: 0.03, scaleY: 1.04, scaleX: 0.98 } },
      { time: 1.6, transform: { posY: 0.0, scaleY: 1.0, scaleX: 1.0 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { posY: 0.0, rotZ: 0.0 } },
      { time: 0.8, transform: { posY: 0.02, rotZ: 0.02 } },
      { time: 1.6, transform: { posY: 0.0, rotZ: 0.0 } },
    ])
    .addTrack('leftArm', [
      { time: 0.0, transform: { rotZ: 0.05 } },
      { time: 0.8, transform: { rotZ: -0.05 } },
      { time: 1.6, transform: { rotZ: 0.05 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: -0.05 } },
      { time: 0.8, transform: { rotZ: 0.05 } },
      { time: 1.6, transform: { rotZ: -0.05 } },
    ])
    .addTrack('leftLeg', [
      { time: 0.0, transform: { rotZ: 0.0, scaleY: 1.0 } },
      { time: 1.6, transform: { rotZ: 0.0, scaleY: 1.0 } },
    ])
    .addTrack('rightLeg', [
      { time: 0.0, transform: { rotZ: 0.0, scaleY: 1.0 } },
      { time: 1.6, transform: { rotZ: 0.0, scaleY: 1.0 } },
    ]);

  // 2. WALK (Duration: 0.8s, Loop: true) - Step cycle with footstep events
  clips['walk'] = new AnimationClip('walk', 0.8, true)
    .addTrack('torso', [
      { time: 0.0, transform: { posY: 0.0, rotZ: 0.0 } },
      { time: 0.2, transform: { posY: 0.05, rotZ: 0.03 } },
      { time: 0.4, transform: { posY: 0.0, rotZ: 0.0 } },
      { time: 0.6, transform: { posY: 0.05, rotZ: -0.03 } },
      { time: 0.8, transform: { posY: 0.0, rotZ: 0.0 } },
    ])
    .addTrack('leftLeg', [
      { time: 0.0, transform: { rotZ: -0.45 } },
      { time: 0.4, transform: { rotZ: 0.45 } },
      { time: 0.8, transform: { rotZ: -0.45 } },
    ])
    .addTrack('rightLeg', [
      { time: 0.0, transform: { rotZ: 0.45 } },
      { time: 0.4, transform: { rotZ: -0.45 } },
      { time: 0.8, transform: { rotZ: 0.45 } },
    ])
    .addTrack('leftArm', [
      { time: 0.0, transform: { rotZ: 0.40 } },
      { time: 0.4, transform: { rotZ: -0.40 } },
      { time: 0.8, transform: { rotZ: 0.40 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: -0.40 } },
      { time: 0.4, transform: { rotZ: 0.40 } },
      { time: 0.8, transform: { rotZ: -0.40 } },
    ])
    .addEvent(0.2, 'footstep', { foot: 'left' })
    .addEvent(0.6, 'footstep', { foot: 'right' });

  // 3. RUN (Duration: 0.6s, Loop: true) - Exaggerated forward stride
  clips['run'] = new AnimationClip('run', 0.6, true)
    .addTrack('torso', [
      { time: 0.0, transform: { posY: 0.0, rotZ: 0.08 } },
      { time: 0.15, transform: { posY: 0.08, rotZ: 0.12 } },
      { time: 0.3, transform: { posY: 0.0, rotZ: 0.08 } },
      { time: 0.45, transform: { posY: 0.08, rotZ: 0.12 } },
      { time: 0.6, transform: { posY: 0.0, rotZ: 0.08 } },
    ])
    .addTrack('leftLeg', [
      { time: 0.0, transform: { rotZ: -0.75 } },
      { time: 0.3, transform: { rotZ: 0.75 } },
      { time: 0.6, transform: { rotZ: -0.75 } },
    ])
    .addTrack('rightLeg', [
      { time: 0.0, transform: { rotZ: 0.75 } },
      { time: 0.3, transform: { rotZ: -0.75 } },
      { time: 0.6, transform: { rotZ: 0.75 } },
    ])
    .addTrack('leftArm', [
      { time: 0.0, transform: { rotZ: 0.75 } },
      { time: 0.3, transform: { rotZ: -0.75 } },
      { time: 0.6, transform: { rotZ: 0.75 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: -0.75 } },
      { time: 0.3, transform: { rotZ: 0.75 } },
      { time: 0.6, transform: { rotZ: -0.75 } },
    ])
    .addEvent(0.15, 'footstep', { foot: 'left' })
    .addEvent(0.45, 'footstep', { foot: 'right' });

  // 4. SNEAK (Duration: 1.0s, Loop: true) - Low-profile crouch stalk
  clips['sneak'] = new AnimationClip('sneak', 1.0, true)
    .addTrack('torso', [
      { time: 0.0, transform: { posY: -0.15, scaleY: 0.85, rotZ: 0.1 } },
      { time: 0.5, transform: { posY: -0.12, scaleY: 0.88, rotZ: 0.08 } },
      { time: 1.0, transform: { posY: -0.15, scaleY: 0.85, rotZ: 0.1 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { posY: -0.1, rotZ: -0.1 } },
      { time: 0.5, transform: { posY: -0.08, rotZ: -0.05 } },
      { time: 1.0, transform: { posY: -0.1, rotZ: -0.1 } },
    ])
    .addTrack('leftLeg', [
      { time: 0.0, transform: { rotZ: -0.3, scaleY: 0.85 } },
      { time: 0.5, transform: { rotZ: 0.3, scaleY: 0.85 } },
      { time: 1.0, transform: { rotZ: -0.3, scaleY: 0.85 } },
    ])
    .addTrack('rightLeg', [
      { time: 0.0, transform: { rotZ: 0.3, scaleY: 0.85 } },
      { time: 0.5, transform: { rotZ: -0.3, scaleY: 0.85 } },
      { time: 1.0, transform: { rotZ: 0.3, scaleY: 0.85 } },
    ]);

  // 5. SPRINT (Duration: 0.45s, Loop: true) - High-energy pump with dust puffs
  clips['sprint'] = new AnimationClip('sprint', 0.45, true)
    .addTrack('torso', [
      { time: 0.0, transform: { posY: 0.0, rotZ: 0.22, scaleY: 0.95 } },
      { time: 0.22, transform: { posY: 0.12, rotZ: 0.25, scaleY: 1.05 } },
      { time: 0.45, transform: { posY: 0.0, rotZ: 0.22, scaleY: 0.95 } },
    ])
    .addTrack('leftLeg', [
      { time: 0.0, transform: { rotZ: -0.95 } },
      { time: 0.22, transform: { rotZ: 0.95 } },
      { time: 0.45, transform: { rotZ: -0.95 } },
    ])
    .addTrack('rightLeg', [
      { time: 0.0, transform: { rotZ: 0.95 } },
      { time: 0.22, transform: { rotZ: -0.95 } },
      { time: 0.45, transform: { rotZ: 0.95 } },
    ])
    .addEvent(0.1, 'footstep', { foot: 'left', sprint: true })
    .addEvent(0.1, 'dust_puff')
    .addEvent(0.32, 'footstep', { foot: 'right', sprint: true })
    .addEvent(0.32, 'dust_puff');

  // 6. TURN (Duration: 0.18s, Loop: false) - Rapid pivot with squash flip
  clips['turn'] = new AnimationClip('turn', 0.18, false)
    .addTrack('torso', [
      { time: 0.0, transform: { scaleX: 1.0, scaleY: 1.0 } },
      { time: 0.09, transform: { scaleX: 0.15, scaleY: 1.25 } },
      { time: 0.18, transform: { scaleX: 1.0, scaleY: 1.0 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { scaleX: 1.0 } },
      { time: 0.09, transform: { scaleX: 0.2 } },
      { time: 0.18, transform: { scaleX: 1.0 } },
    ]);

  // 7. CHOP (Duration: 0.45s, Loop: false) - 3-phase strike with hit_frame event at 0.22s
  clips['chop'] = new AnimationClip('chop', 0.45, false)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: 0.0, posY: 0.0 } },
      { time: 0.14, transform: { rotZ: -0.35, posY: 0.06 } }, // Windup anticipation
      { time: 0.22, transform: { rotZ: 0.45, posY: -0.08 } }, // Slam impact apex
      { time: 0.34, transform: { rotZ: 0.25, posY: -0.04 } }, // Follow-through
      { time: 0.45, transform: { rotZ: 0.0, posY: 0.0 } },   // Settle
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.14, transform: { rotZ: 2.1 } },  // Raised high over head
      { time: 0.22, transform: { rotZ: -1.2 } }, // Downward smash
      { time: 0.34, transform: { rotZ: -0.8 } },
      { time: 0.45, transform: { rotZ: 0.0 } },
    ])
    .addTrack('heldTool', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.14, transform: { rotZ: 0.6 } },
      { time: 0.22, transform: { rotZ: -0.5 } },
      { time: 0.45, transform: { rotZ: 0.0 } },
    ])
    .addEvent(0.22, 'hit_frame', { tool: 'axe', damageRadius: 1.8 });

  // 8. LIGHT_SWING (Duration: 0.30s, Loop: false) - Fast tool swipe with hit_frame at 0.12s
  clips['light_swing'] = new AnimationClip('light_swing', 0.30, false)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.08, transform: { rotZ: -0.2 } },
      { time: 0.12, transform: { rotZ: 0.3 } },
      { time: 0.30, transform: { rotZ: 0.0 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.08, transform: { rotZ: 1.4 } },
      { time: 0.12, transform: { rotZ: -0.9 } },
      { time: 0.30, transform: { rotZ: 0.0 } },
    ])
    .addEvent(0.12, 'hit_frame', { swing: 'light' });

  // 9. THROW (Duration: 0.50s, Loop: false) - Overhead pitch
  clips['throw'] = new AnimationClip('throw', 0.50, false)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.18, transform: { rotZ: -0.4 } }, // Deep pull back
      { time: 0.26, transform: { rotZ: 0.35 } },  // Pitch forward
      { time: 0.50, transform: { rotZ: 0.0 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.18, transform: { rotZ: 2.4 } },  // Hand back behind head
      { time: 0.26, transform: { rotZ: -1.4 } }, // Release point
      { time: 0.50, transform: { rotZ: 0.0 } },
    ])
    .addEvent(0.26, 'throw_release');

  // 10. HURT (Duration: 0.35s, Loop: false) - Flinch recoil
  clips['hurt'] = new AnimationClip('hurt', 0.35, false)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: 0.0, posY: 0.0, scaleY: 1.0 } },
      { time: 0.08, transform: { rotZ: -0.45, posY: 0.08, scaleY: 1.15 } }, // Sharp snap back
      { time: 0.20, transform: { rotZ: -0.15, posY: -0.04, scaleY: 0.90 } },
      { time: 0.35, transform: { rotZ: 0.0, posY: 0.0, scaleY: 1.0 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.08, transform: { rotZ: -0.6 } },
      { time: 0.35, transform: { rotZ: 0.0 } },
    ])
    .addEvent(0.02, 'hurt_impact');

  // 11. DOWN (Duration: 0.60s, Loop: false) - Knockout collapse
  clips['down'] = new AnimationClip('down', 0.60, false)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: 0.0, posY: 0.0 } },
      { time: 0.25, transform: { rotZ: -0.6, posY: -0.3 } },
      { time: 0.60, transform: { rotZ: -Math.PI / 2, posY: -0.55 } }, // Flat on ground
    ])
    .addTrack('head', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.60, transform: { rotZ: -0.3 } },
    ]);

  // 12. GET_UP (Duration: 0.65s, Loop: false) - Slapstick scramble
  clips['get_up'] = new AnimationClip('get_up', 0.65, false)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: -Math.PI / 2, posY: -0.55 } },
      { time: 0.30, transform: { rotZ: -0.3, posY: -0.2, scaleY: 0.85 } },
      { time: 0.65, transform: { rotZ: 0.0, posY: 0.0, scaleY: 1.0 } },
    ]);

  // 13. EAT (Duration: 0.60s, Loop: false) - Chewing bob
  clips['eat'] = new AnimationClip('eat', 0.60, false)
    .addTrack('torso', [
      { time: 0.0, transform: { scaleY: 1.0 } },
      { time: 0.15, transform: { scaleY: 1.08 } },
      { time: 0.30, transform: { scaleY: 0.95 } },
      { time: 0.45, transform: { scaleY: 1.06 } },
      { time: 0.60, transform: { scaleY: 1.0 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.15, transform: { rotZ: 0.18 } },
      { time: 0.30, transform: { rotZ: -0.1 } },
      { time: 0.45, transform: { rotZ: 0.15 } },
      { time: 0.60, transform: { rotZ: 0.0 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.20, transform: { rotZ: 1.8 } }, // Hand to mouth
      { time: 0.60, transform: { rotZ: 0.0 } },
    ])
    .addEvent(0.30, 'eat_bite');

  // 14. DANCE (Duration: 0.8s, Loop: true) - Slapstick party bounce
  clips['dance'] = new AnimationClip('dance', 0.8, true)
    .addTrack('torso', [
      { time: 0.0, transform: { rotZ: -0.25, posY: 0.08, scaleX: 1.1 } },
      { time: 0.4, transform: { rotZ: 0.25, posY: 0.08, scaleX: 1.1 } },
      { time: 0.8, transform: { rotZ: -0.25, posY: 0.08, scaleX: 1.1 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { rotZ: 0.3 } },
      { time: 0.4, transform: { rotZ: -0.3 } },
      { time: 0.8, transform: { rotZ: 0.3 } },
    ])
    .addTrack('leftArm', [
      { time: 0.0, transform: { rotZ: 2.2 } },
      { time: 0.4, transform: { rotZ: 0.5 } },
      { time: 0.8, transform: { rotZ: 2.2 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.5 } },
      { time: 0.4, transform: { rotZ: 2.2 } },
      { time: 0.8, transform: { rotZ: 0.5 } },
    ]);

  // 15. WAVE (Duration: 0.70s, Loop: false) - Friendly greeting
  clips['wave'] = new AnimationClip('wave', 0.70, false)
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.15, transform: { rotZ: 2.3 } },
      { time: 0.30, transform: { rotZ: 2.6 } },
      { time: 0.45, transform: { rotZ: 2.2 } },
      { time: 0.55, transform: { rotZ: 2.6 } },
      { time: 0.70, transform: { rotZ: 0.0 } },
    ]);

  // 16. HONK (Duration: 0.50s, Loop: false) - Chest expansion air horn thrust
  clips['honk'] = new AnimationClip('honk', 0.50, false)
    .addTrack('torso', [
      { time: 0.0, transform: { scaleY: 1.0, scaleX: 1.0, rotZ: 0.0 } },
      { time: 0.15, transform: { scaleY: 1.35, scaleX: 1.25, rotZ: -0.2 } }, // Big puff out
      { time: 0.28, transform: { scaleY: 0.85, scaleX: 0.90, rotZ: 0.3 } },   // Blast forward
      { time: 0.50, transform: { scaleY: 1.0, scaleX: 1.0, rotZ: 0.0 } },
    ])
    .addTrack('head', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.15, transform: { rotZ: -0.4 } }, // Thrown back
      { time: 0.28, transform: { rotZ: 0.35 } },  // Thrust forward
      { time: 0.50, transform: { rotZ: 0.0 } },
    ])
    .addTrack('rightArm', [
      { time: 0.0, transform: { rotZ: 0.0 } },
      { time: 0.20, transform: { rotZ: 1.2 } },
      { time: 0.50, transform: { rotZ: 0.0 } },
    ])
    .addEvent(0.20, 'honk_blast');

  return clips;
}
