import { createHumanoidClips } from '../animation/HumanoidClips';
import { CrossfadeController, LinearBlend1D } from '../animation/BlendTree';
import { LayeredStateMachine } from '../animation/LayeredStateMachine';
import { ProceduralLayer } from '../animation/ProceduralLayer';
import { DampedSpring, SpringPreset } from '../animation/TweenPreset';
import { AnimationEventManager } from '../animation/AnimationEvent';

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`❌ FAILED: ${msg}`);
    process.exit(1);
  }
}

console.log('🧪 Starting Phase Z0C Animation Foundation Test Suite (Section 68)...\n');

const clips = createHumanoidClips();

// 1. Test Required Clips (16 Core Roster)
console.log('1. Testing Required Clips Roster (16 clips)...');
const requiredClips = [
  'idle',
  'walk',
  'run',
  'sneak',
  'sprint',
  'turn',
  'chop',
  'light_swing',
  'throw',
  'hurt',
  'down',
  'get_up',
  'eat',
  'dance',
  'wave',
  'honk',
];

for (const name of requiredClips) {
  assert(!!clips[name], `Required humanoid clip missing: ${name}`);
  const clip = clips[name];
  assert(clip.duration > 0, `Clip ${name} duration must be > 0, got ${clip.duration}`);
  assert(clip.tracks.size >= 1, `Clip ${name} must have at least 1 track, got ${clip.tracks.size}`);

  // Sample clip at start, mid, end without crashing
  const p0 = clip.sample(0);
  const pMid = clip.sample(clip.duration * 0.5);
  const pEnd = clip.sample(clip.duration);
  assert(p0.size > 0 && pMid.size > 0 && pEnd.size > 0, `Sampling failed on ${name}`);
}
console.log(`   ✅ All 16 required humanoid clips present and valid`);

// 2. Test Event Order in Combat Clips (hit_frame timing)
console.log('2. Testing Animation Event Ordering & Timing...');
const chop = clips['chop'];
assert(chop.events.length > 0, 'Chop clip must have registered events');
const hitEvent = chop.events.find(e => e.name === 'hit_frame');
assert(!!hitEvent, 'Chop clip missing hit_frame event');
if (!hitEvent) throw new Error('Unreachable');
assert(hitEvent.time > 0.10 && hitEvent.time < chop.duration * 0.85,
  `hit_frame time (${hitEvent.time}s) must precede recovery in chop (duration: ${chop.duration}s)`);

const lightSwing = clips['light_swing'];
const swingHit = lightSwing.events.find(e => e.name === 'hit_frame');
assert(!!swingHit, 'light_swing clip missing hit_frame event');
if (!swingHit) throw new Error('Unreachable');
assert(swingHit.time > 0.05 && swingHit.time < lightSwing.duration * 0.80,
  `hit_frame time (${swingHit.time}s) must precede recovery in light_swing`);

// Verify Event Manager Triggering
const eventManager = new AnimationEventManager();
let hitFrameTriggered = false;
let footstepCount = 0;

eventManager.addListener('hit_frame', () => {
  hitFrameTriggered = true;
});
eventManager.addListener('footstep', () => {
  footstepCount++;
});

// Simulate chop playback crossing hit_frame at 0.22s
eventManager.update(chop, 0.0, 0.20);
assert(!hitFrameTriggered, 'hit_frame must not trigger before 0.22s');
eventManager.update(chop, 0.20, 0.25);
assert(hitFrameTriggered, 'hit_frame must trigger when passing 0.22s');

// Simulate walk playback crossing footstep events (0.2s and 0.6s)
const walk = clips['walk'];
eventManager.update(walk, 0.0, 0.3);
assert(footstepCount === 1, `Expected 1 footstep at 0.2s, got ${footstepCount}`);
eventManager.update(walk, 0.3, 0.7);
assert(footstepCount === 2, `Expected 2 footsteps at 0.6s, got ${footstepCount}`);
console.log('   ✅ Animation Event order and dispatch tests passed');

// 3. Test Pose Continuity & Crossfades (No Visual Pops)
console.log('3. Testing Pose Continuity Across State Crossfades...');
const controller = new CrossfadeController();
controller.play(clips['idle'], 0);
controller.update(0.5);

// Crossfade from idle to walk over 0.12s
controller.play(clips['walk'], 0.12);

// Step through crossfade at 60 Hz (16.6ms intervals) and assert max delta threshold
const maxAllowedDelta = 0.25; // max translation/rotation jump per 16ms frame
let prevPose = controller.update(0.016);

for (let step = 0; step < 10; step++) {
  const nextPose = controller.update(0.016);
  for (const [node, curTransform] of nextPose.entries()) {
    const prev = prevPose.get(node);
    if (prev) {
      const dRot = Math.abs(curTransform.rotZ - prev.rotZ);
      const dPos = Math.hypot(curTransform.posX - prev.posX, curTransform.posY - prev.posY);
      assert(dRot < maxAllowedDelta, `Visual pop detected on ${node} rotZ: delta ${dRot} >= ${maxAllowedDelta}`);
      assert(dPos < maxAllowedDelta, `Visual pop detected on ${node} pos: delta ${dPos} >= ${maxAllowedDelta}`);
    }
  }
  prevPose = nextPose;
}
console.log('   ✅ Pose continuity validated (delta < 0.25 across all frames, zero visual pops)');

// 4. Test 1D Linear Blend Tree
console.log('4. Testing 1D Linear Blend Tree (Walk -> Run)...');
const blendTree = new LinearBlend1D(clips['walk'], clips['run']);
const blend0 = blendTree.sample(0.2, 0.0);
const blend50 = blendTree.sample(0.2, 0.5);
const blend100 = blendTree.sample(0.2, 1.0);

const leg0 = blend0.get('leftLeg')?.rotZ ?? 0;
const leg50 = blend50.get('leftLeg')?.rotZ ?? 0;
const leg100 = blend100.get('leftLeg')?.rotZ ?? 0;

// Interpolated midpoint must lie between endpoints
const minLeg = Math.min(leg0, leg100);
const maxLeg = Math.max(leg0, leg100);
assert(leg50 >= minLeg - 0.01 && leg50 <= maxLeg + 0.01,
  `Blend 50% leftLeg (${leg50}) must lie between 0% (${leg0}) and 100% (${leg100})`);
console.log('   ✅ 1D Linear Blend Tree smoothly interpolates intermediate poses');

// 5. Test 4-Tier Layered State Machine (Upper Body Masking)
console.log('5. Testing Layered State Machine & Upper Body Masking...');
const stateMachine = new LayeredStateMachine();
stateMachine.playBase(clips['run'], 0);
stateMachine.playUpper(clips['chop'], 0); // Chop masked onto upper body while running!

const combined = stateMachine.update(0.05);

// Legs must come from Run (Base layer)
const runRef = clips['run'].sample(0.05);
const chopRef = clips['chop'].sample(0.05);

assert(combined.has('leftLeg'), 'Combined pose must have leftLeg');
assert(Math.abs((combined.get('leftLeg')?.rotZ ?? 0) - (runRef.get('leftLeg')?.rotZ ?? 0)) < 0.001,
  'Running legs must be preserved when chopping upper body');

// Arms & tool must come from Chop (Upper Body layer)
assert(combined.has('rightArm'), 'Combined pose must have rightArm');
assert(Math.abs((combined.get('rightArm')?.rotZ ?? 0) - (chopRef.get('rightArm')?.rotZ ?? 0)) < 0.001,
  'Right arm strike must be applied from Upper Body Chop clip');
console.log('   ✅ Upper body masking validated: chopping while running works seamlessly');

// 6. Test Procedural Layer & Stride Matching (Zero Foot Sliding)
console.log('6. Testing Stride Matching & Procedural Physics...');
const procedural = new ProceduralLayer();

// Nominal stride length: 1.8m per 0.8s walk = nominal speed 2.25 m/s
const walkSpeedNormal = procedural.calculateStrideSpeed(2.25, 0.8);
assert(Math.abs(walkSpeedNormal - 1.0) < 0.05, `Speed multiplier at nominal must be ~1.0, got ${walkSpeedNormal}`);

// At double speed (4.5 m/s), playback speed must double to 2.0x (zero foot sliding!)
const walkSpeedDouble = procedural.calculateStrideSpeed(4.5, 0.8);
assert(Math.abs(walkSpeedDouble - 2.0) < 0.05, `Speed multiplier at double speed must be ~2.0, got ${walkSpeedDouble}`);

// Test Second-Order Damped Spring
const spring = new DampedSpring(SpringPreset.snappy, 0);
spring.update(0.05, 1.0); // target 1.0
assert(spring.value > 0, 'Spring must respond to target offset');
assert(spring.velocity > 0, 'Spring velocity must be positive');
console.log('   ✅ Stride matching and harmonic spring dynamics verified');

console.log('\n🎉 ALL PHASE Z0C ANIMATION FOUNDATION TESTS PASSED (100% SUCCESS)!');
