import { AnimationPose, TransformData, createDefaultTransform } from './AnimationPose';
import { TweenPreset, EasingFunction } from './TweenPreset';

export interface Keyframe {
  time: number; // in seconds (from 0 to duration)
  transform: Partial<TransformData>;
  easing?: string; // key of TweenPreset (defaults to 'quadEaseInOut')
}

export interface NodeTrack {
  nodeName: string;
  keyframes: Keyframe[];
}

export interface AnimationEventDef {
  time: number; // in seconds
  name: string; // e.g. 'hit_frame', 'footstep', 'dust_puff'
  payload?: any;
}

export class AnimationClip {
  public name: string;
  public duration: number;
  public loop: boolean;
  public tracks: Map<string, NodeTrack> = new Map();
  public events: AnimationEventDef[] = [];

  constructor(name: string, duration: number, loop: boolean = true) {
    this.name = name;
    this.duration = Math.max(0.01, duration);
    this.loop = loop;
  }

  public addTrack(nodeName: string, keyframes: Keyframe[]): this {
    // Ensure keyframes are sorted by time
    const sorted = [...keyframes].sort((a, b) => a.time - b.time);
    this.tracks.set(nodeName, { nodeName, keyframes: sorted });
    return this;
  }

  public addEvent(time: number, name: string, payload?: any): this {
    this.events.push({ time, name, payload });
    this.events.sort((a, b) => a.time - b.time);
    return this;
  }

  /**
   * Samples this clip at an elapsed time, returning a full AnimationPose map.
   */
  public sample(rawTime: number): AnimationPose {
    let t = rawTime;
    if (this.loop) {
      t = t % this.duration;
      if (t < 0) t += this.duration;
    } else {
      t = Math.max(0, Math.min(this.duration, t));
    }

    const pose: AnimationPose = new Map();

    for (const [nodeName, track] of this.tracks.entries()) {
      pose.set(nodeName, this.sampleTrack(track, t));
    }

    return pose;
  }

  private sampleTrack(track: NodeTrack, t: number): TransformData {
    const kfs = track.keyframes;
    if (kfs.length === 0) return createDefaultTransform();
    if (kfs.length === 1 || t <= kfs[0].time) {
      return this.expandTransform(kfs[0].transform);
    }
    if (t >= kfs[kfs.length - 1].time) {
      return this.expandTransform(kfs[kfs.length - 1].transform);
    }

    // Find bounding keyframes
    let prevIndex = 0;
    for (let i = 0; i < kfs.length - 1; i++) {
      if (t >= kfs[i].time && t <= kfs[i + 1].time) {
        prevIndex = i;
        break;
      }
    }

    const kfA = kfs[prevIndex];
    const kfB = kfs[prevIndex + 1];

    const segDuration = kfB.time - kfA.time;
    let localT = segDuration > 0 ? (t - kfA.time) / segDuration : 0;
    localT = Math.max(0, Math.min(1, localT));

    // Apply easing
    const easingKey = kfB.easing || kfA.easing || 'quadEaseInOut';
    const easeFn: EasingFunction = TweenPreset[easingKey] || TweenPreset.quadEaseInOut;
    const easedT = easeFn(localT);

    const fullA = this.expandTransform(kfA.transform);
    const fullB = this.expandTransform(kfB.transform);

    const inv = 1 - easedT;
    return {
      posX: fullA.posX * inv + fullB.posX * easedT,
      posY: fullA.posY * inv + fullB.posY * easedT,
      posZ: fullA.posZ * inv + fullB.posZ * easedT,
      rotX: fullA.rotX * inv + fullB.rotX * easedT,
      rotY: fullA.rotY * inv + fullB.rotY * easedT,
      rotZ: fullA.rotZ * inv + fullB.rotZ * easedT,
      scaleX: fullA.scaleX * inv + fullB.scaleX * easedT,
      scaleY: fullA.scaleY * inv + fullB.scaleY * easedT,
      scaleZ: fullA.scaleZ * inv + fullB.scaleZ * easedT,
    };
  }

  private expandTransform(partial: Partial<TransformData>): TransformData {
    return {
      posX: partial.posX ?? 0,
      posY: partial.posY ?? 0,
      posZ: partial.posZ ?? 0,
      rotX: partial.rotX ?? 0,
      rotY: partial.rotY ?? 0,
      rotZ: partial.rotZ ?? 0,
      scaleX: partial.scaleX ?? 1,
      scaleY: partial.scaleY ?? 1,
      scaleZ: partial.scaleZ ?? 1,
    };
  }
}
