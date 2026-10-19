import { AnimationClip, AnimationEventDef } from './AnimationClip';

export type AnimationEventListener = (eventName: string, payload?: any) => void;

/**
 * Tracks animation time advancement and fires named events exactly when
 * the timeline crosses event thresholds.
 */
export class AnimationEventManager {
  private listeners: Map<string, AnimationEventListener[]> = new Map();
  private lastTime: number = 0;

  public addListener(eventName: string, listener: AnimationEventListener) {
    if (!this.listeners.has(eventName)) {
      this.listeners.set(eventName, []);
    }
    this.listeners.get(eventName)!.push(listener);
  }

  public removeListener(eventName: string, listener: AnimationEventListener) {
    const list = this.listeners.get(eventName);
    if (!list) return;
    this.listeners.set(eventName, list.filter(l => l !== listener));
  }

  public reset(time: number = 0) {
    this.lastTime = time;
  }

  /**
   * Evaluates clip events between prevTime and currentTime.
   * Handles loops seamlessly.
   */
  public update(clip: AnimationClip, prevTime: number, currTime: number) {
    if (clip.events.length === 0) return;

    const dur = clip.duration;
    let t0 = prevTime;
    let t1 = currTime;

    if (clip.loop) {
      const loop0 = Math.floor(t0 / dur);
      const loop1 = Math.floor(t1 / dur);

      if (loop0 === loop1) {
        // Within same loop cycle
        const rem0 = t0 - loop0 * dur;
        const rem1 = t1 - loop1 * dur;
        this.checkEventsInRange(clip.events, rem0, rem1);
      } else {
        // Wrapped around loop boundary
        const rem0 = t0 - loop0 * dur;
        this.checkEventsInRange(clip.events, rem0, dur);
        const rem1 = t1 - loop1 * dur;
        this.checkEventsInRange(clip.events, 0, rem1);
      }
    } else {
      t0 = Math.max(0, Math.min(dur, t0));
      t1 = Math.max(0, Math.min(dur, t1));
      if (t1 > t0) {
        this.checkEventsInRange(clip.events, t0, t1);
      }
    }
  }

  private checkEventsInRange(events: AnimationEventDef[], start: number, end: number) {
    for (const ev of events) {
      if (ev.time > start && ev.time <= end) {
        this.dispatch(ev.name, ev.payload);
      }
    }
  }

  private dispatch(eventName: string, payload?: any) {
    const list = this.listeners.get(eventName);
    if (list) {
      for (const fn of list) {
        fn(eventName, payload);
      }
    }

    // Also dispatch wildcard listeners
    const wildcard = this.listeners.get('*');
    if (wildcard) {
      for (const fn of wildcard) {
        fn(eventName, payload);
      }
    }
  }
}
