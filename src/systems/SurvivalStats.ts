import { sounds } from '../audio/SoundEffects';

export class SurvivalStats {
  public health: number = 100;
  public hunger: number = 100;
  public sanity: number = 100;
  public infectionPercent: number = 0;
  public isDead: boolean = false;

  // Statistics for post-game screen
  public treesFelled: number = 0;
  public itemsCrafted: number = 0;
  public daysSurvived: number = 1;

  private onDeathCallback?: (reason: string) => void;
  private onDamageCallback?: () => void;

  constructor() {}

  public setOnDeath(fn: (reason: string) => void) {
    this.onDeathCallback = fn;
  }

  public setOnDamage(fn: () => void) {
    this.onDamageCallback = fn;
  }

  public update(deltaSeconds: number, isNight: boolean, isNearFire: boolean, isMoving: boolean) {
    if (this.isDead) return;

    // 1. Hunger Drain: Calibrated for 20-minute cycle (Base drain 0.025/sec, +0.015 when moving)
    const hungerDrain = (0.025 + (isMoving ? 0.015 : 0)) * deltaSeconds;
    this.hunger = Math.max(0, this.hunger - hungerDrain);

    // If starving (hunger = 0), take 2.5 damage/sec
    if (this.hunger <= 0) {
      this.takeDamage(2.5 * deltaSeconds, 'Starvation');
    }

    // 2. Sanity Drain / Recovery (calibrated for ~4.5 min night)
    if (isNearFire) {
      // Comfort of the fire restores sanity
      this.sanity = Math.min(100, this.sanity + 0.35 * deltaSeconds);
    } else if (isNight) {
      // Dark dread drains sanity
      this.sanity = Math.max(0, this.sanity - 0.22 * deltaSeconds);
    } else {
      // Gentle passive day drain
      this.sanity = Math.max(0, this.sanity - 0.01 * deltaSeconds);
    }

    // Low sanity distortion visual
    const distortionEl = document.getElementById('sanity-distortion');
    if (distortionEl) {
      if (this.sanity < 40) {
        distortionEl.classList.remove('hidden');
        distortionEl.style.opacity = `${(40 - this.sanity) / 40}`;
      } else {
        distortionEl.classList.add('hidden');
      }
    }

    this.renderHUD();
  }

  public eat(hungerBonus: number, healthBonus: number = 0, sanityBonus: number = 0) {
    if (this.isDead) return;
    this.hunger = Math.min(100, this.hunger + hungerBonus);
    this.health = Math.min(100, this.health + healthBonus);
    this.sanity = Math.min(100, this.sanity + sanityBonus);
    sounds.playEat();
    this.renderHUD();
  }

  public takeDamage(amount: number, reason: string = 'The wilderness') {
    if (this.isDead) return;
    this.health = Math.max(0, this.health - amount);
    sounds.playHurt();
    this.onDamageCallback?.();

    // Damage flash screen effect
    const flashEl = document.getElementById('damage-flash');
    if (flashEl) {
      flashEl.classList.add('flash-active');
      setTimeout(() => flashEl.classList.remove('flash-active'), 120);
    }

    if (this.health <= 0) {
      this.isDead = true;
      if (this.onDeathCallback) {
        this.onDeathCallback(reason);
      }
    }
    this.renderHUD();
  }

  public renderHUD() {
    const hungerFill = document.getElementById('fill-hunger');
    const hungerVal = document.getElementById('val-hunger');
    const healthFill = document.getElementById('fill-health');
    const healthVal = document.getElementById('val-health');
    const sanityFill = document.getElementById('fill-sanity');
    const sanityVal = document.getElementById('val-sanity');

    if (hungerFill && hungerVal) {
      hungerFill.style.height = `${Math.round(this.hunger)}%`;
      hungerVal.innerText = `${Math.round(this.hunger)}`;
    }
    if (healthFill && healthVal) {
      healthFill.style.height = `${Math.round(this.health)}%`;
      healthVal.innerText = `${Math.round(this.health)}`;
    }
    if (sanityFill && sanityVal) {
      sanityFill.style.height = `${Math.round(this.sanity)}%`;
      sanityVal.innerText = `${Math.round(this.sanity)}`;
    }
  }

  public reset() {
    this.health = 100;
    this.hunger = 100;
    this.sanity = 100;
    this.isDead = false;
    this.treesFelled = 0;
    this.itemsCrafted = 0;
    this.daysSurvived = 1;
    this.renderHUD();
  }
}
