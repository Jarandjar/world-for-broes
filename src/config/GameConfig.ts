export type GameMode = 'classic' | 'zombie';
export type LootDensity = 'low' | 'normal' | 'high';
export type ZombieDensity = 'low' | 'normal' | 'high';
export type AnimationQuality = 'low' | 'medium' | 'high';

export interface GameConfig {
  mode: GameMode;
  lootDensity: LootDensity;
  zombieDensity: ZombieDensity;
  hordeIntervalDays: number;
  infectionEnabled: boolean;
  friendlyFire: boolean;
  evacGoalEnabled: boolean;
  reducedMotion: boolean;
  screenShake: boolean;
  soundVolume: number;
  audioEnabled: boolean;
  goreFree: boolean;
  debugOverlay: boolean;

  // Phase Z0D: Style & Animation Direction
  animationQuality: AnimationQuality;
  juiceDial: number; // 0 to 100%
  toonRampEnabled: boolean;
  outlinesEnabled: boolean;
  tiltShiftEnabled: boolean;
  bloomEnabled: boolean;
  paperGrainEnabled: boolean;
  blobAOEnabled: boolean;
  lutColorGradeEnabled: boolean;
  verticalSliceDemo: boolean;
}

const STORAGE_KEY = 'world4bros_game_config';

const DEFAULT_CONFIG: GameConfig = {
  mode: 'classic',
  lootDensity: 'high',
  zombieDensity: 'normal',
  hordeIntervalDays: 7,
  infectionEnabled: true,
  friendlyFire: false,
  evacGoalEnabled: true,
  reducedMotion: false,
  screenShake: true,
  soundVolume: 1.0,
  audioEnabled: true,
  goreFree: true,
  debugOverlay: false,

  // Phase Z0D Defaults (Medium on mobile/balanced, High on desktop capable)
  animationQuality: 'high',
  juiceDial: 60, // 60% punchy slapstick
  toonRampEnabled: true,
  outlinesEnabled: true,
  tiltShiftEnabled: true,
  bloomEnabled: true,
  paperGrainEnabled: true,
  blobAOEnabled: true,
  lutColorGradeEnabled: true,
  verticalSliceDemo: false,
};

class GameConfigManager {
  private config: GameConfig;
  private listeners: ((config: GameConfig) => void)[] = [];

  constructor() {
    this.config = this.initFromEnvironment();
  }

  private initFromEnvironment(): GameConfig {
    const config = { ...DEFAULT_CONFIG };

    // 1. Try local storage first (browser environment only)
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored);
          Object.assign(config, parsed);
        }
      } catch (e) {
        console.warn('Could not read GameConfig from localStorage', e);
      }
    }

    // 2. Override with URL search params (e.g. ?mode=zombie, ?debug=1, ?quality=high, ?juice=80, ?slice=1)
    if (typeof window !== 'undefined' && window.location) {
      const params = new URLSearchParams(window.location.search);
      const urlMode = params.get('mode');
      if (urlMode === 'zombie' || urlMode === 'classic') {
        config.mode = urlMode;
      }
      const urlLoot = params.get('loot');
      if (urlLoot === 'low' || urlLoot === 'normal' || urlLoot === 'high') {
        config.lootDensity = urlLoot;
      }
      const urlZombie = params.get('zombies');
      if (urlZombie === 'low' || urlZombie === 'normal' || urlZombie === 'high') {
        config.zombieDensity = urlZombie;
      }
      if (params.has('debug') || params.get('debug') === '1') {
        config.debugOverlay = true;
      }
      if (params.has('gore')) {
        config.goreFree = params.get('gore') !== '1';
      }
      const urlQuality = params.get('quality');
      if (urlQuality === 'low' || urlQuality === 'medium' || urlQuality === 'high') {
        config.animationQuality = urlQuality;
      }
      const urlJuice = params.get('juice');
      if (urlJuice && !isNaN(parseInt(urlJuice))) {
        config.juiceDial = Math.max(0, Math.min(100, parseInt(urlJuice)));
      }
      if (params.has('slice') || params.get('slice') === '1') {
        config.verticalSliceDemo = true;
      }
    }

    return config;
  }

  public get(): GameConfig {
    return this.config;
  }

  public isZombieMode(): boolean {
    return this.config.mode === 'zombie';
  }

  public isClassicMode(): boolean {
    return this.config.mode === 'classic';
  }

  public update(patch: Partial<GameConfig>): void {
    Object.assign(this.config, patch);
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(this.config));
      } catch (e) {
        console.warn('Failed to persist GameConfig to localStorage', e);
      }
    }
    this.notify();
  }

  public subscribe(listener: (config: GameConfig) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      try {
        listener(this.config);
      } catch (err) {
        console.error('Error in GameConfig listener', err);
      }
    }
  }
}

export const gameConfig = new GameConfigManager();
