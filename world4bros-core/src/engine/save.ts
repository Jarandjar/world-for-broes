/** Storage is behind an adapter so cloud saves or multiplayer persistence can replace localStorage later. */
export interface StorageAdapter {
  load(key: string): string | null;
  save(key: string, value: string): void;
  remove(key: string): void;
}
export class LocalStorageAdapter implements StorageAdapter {
  load(key: string): string | null { try { return localStorage.getItem(key); } catch { return null; } }
  save(key: string, value: string): void { try { localStorage.setItem(key, value); } catch { /* storage may be unavailable */ } }
  remove(key: string): void { try { localStorage.removeItem(key); } catch { /* ignore */ } }
}
export class MemoryStorage implements StorageAdapter {
  private m = new Map<string, string>();
  load(key: string): string | null { return this.m.get(key) ?? null; }
  save(key: string, value: string): void { this.m.set(key, value); }
  remove(key: string): void { this.m.delete(key); }
}
