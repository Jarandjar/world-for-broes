/** Tiny typed event bus. Systems talk through events so they stay decoupled (and later, network-friendly). */
export class Events<M extends Record<string, unknown>> {
  private handlers = new Map<keyof M, Set<(p: never) => void>>();
  on<K extends keyof M>(type: K, fn: (payload: M[K]) => void): () => void {
    let set = this.handlers.get(type);
    if (!set) { set = new Set(); this.handlers.set(type, set); }
    set.add(fn as (p: never) => void);
    return () => { set?.delete(fn as (p: never) => void); };
  }
  emit<K extends keyof M>(type: K, payload: M[K]): void {
    this.handlers.get(type)?.forEach((fn) => (fn as (p: M[K]) => void)(payload));
  }
}
