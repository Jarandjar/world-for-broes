/**
 * Visual Shell Simulator & Interactive Controller
 * Drives live state testing for the base visual structure in visual-shell.html
 */

export class VisualShellController {
  private currentMode: 'on-foot' | 'driving' | 'repair' = 'on-foot';
  private currentTheme: 'day' | 'dusk' | 'night' | 'horde' = 'day';
  private isMobileFrame: boolean = false;

  // State values
  public health: number = 95;
  public hunger: number = 80;
  public sanity: number = 70;
  public speedKmH: number = 0;
  public fuelPercent: number = 75;

  private clockAngle: number = 45;
  private actionPrompts = [
    { key: 'SPACE', text: 'Chop Pine Tree' },
    { key: 'E', text: 'Search Metal Toolbox' },
    { key: 'F', text: 'Feast on Cooked Morsel' },
    { key: 'R', text: 'Climb into Dumpster' },
    { key: 'SPACE', text: 'Hop into Rusty Hatchback' },
    { key: 'E', text: 'Refuel Tank with 5L Canister' }
  ];
  private actionIndex: number = 0;

  constructor() {
    this.initToolbarBindings();
    this.initHotbarBindings();
    this.initClockTick();
  }

  private initToolbarBindings() {
    // Mode toggles
    const btnOnFoot = document.getElementById('sim-mode-onfoot');
    const btnDriving = document.getElementById('sim-mode-driving');
    const btnRepair = document.getElementById('sim-mode-repair');

    btnOnFoot?.addEventListener('click', () => this.setMode('on-foot'));
    btnDriving?.addEventListener('click', () => this.setMode('driving'));
    btnRepair?.addEventListener('click', () => this.setMode('repair'));

    // Theme toggles
    const btnDay = document.getElementById('sim-theme-day');
    const btnDusk = document.getElementById('sim-theme-dusk');
    const btnNight = document.getElementById('sim-theme-night');
    const btnHorde = document.getElementById('sim-theme-horde');

    btnDay?.addEventListener('click', () => this.setTheme('day'));
    btnDusk?.addEventListener('click', () => this.setTheme('dusk'));
    btnNight?.addEventListener('click', () => this.setTheme('night'));
    btnHorde?.addEventListener('click', () => this.setTheme('horde'));

    // Frame toggle (Desktop vs Phone 844x390)
    const btnFrame = document.getElementById('sim-toggle-frame');
    btnFrame?.addEventListener('click', () => this.toggleMobileFrame());

    // Sliders
    const sHealth = document.getElementById('sim-slider-health') as HTMLInputElement | null;
    sHealth?.addEventListener('input', (e) => {
      this.health = Number((e.target as HTMLInputElement).value);
      this.updateVitals();
    });

    const sHunger = document.getElementById('sim-slider-hunger') as HTMLInputElement | null;
    sHunger?.addEventListener('input', (e) => {
      this.hunger = Number((e.target as HTMLInputElement).value);
      this.updateVitals();
    });

    const sSanity = document.getElementById('sim-slider-sanity') as HTMLInputElement | null;
    sSanity?.addEventListener('input', (e) => {
      this.sanity = Number((e.target as HTMLInputElement).value);
      this.updateVitals();
    });

    const sSpeed = document.getElementById('sim-slider-speed') as HTMLInputElement | null;
    sSpeed?.addEventListener('input', (e) => {
      this.speedKmH = Number((e.target as HTMLInputElement).value);
      this.updateSpeed();
    });

    // Test actions
    document.getElementById('sim-btn-cycle-action')?.addEventListener('click', () => {
      this.cycleActionPrompt();
    });

    document.getElementById('sim-btn-damage')?.addEventListener('click', () => {
      this.triggerDamageFlash();
    });

    document.getElementById('sim-btn-comic-pop')?.addEventListener('click', () => {
      this.triggerRandomComicPop();
    });
  }

  public setMode(mode: 'on-foot' | 'driving' | 'repair') {
    this.currentMode = mode;
    document.querySelectorAll('.sim-btn-mode').forEach(b => b.classList.remove('active'));
    document.getElementById(`sim-mode-${mode === 'on-foot' ? 'onfoot' : mode}`)?.classList.add('active');

    const drivingLayer = document.getElementById('driving-controls-layer');
    const touchLayer = document.getElementById('touch-controls-layer');
    const speedSliderWrap = document.getElementById('sim-wrap-speed');

    if (mode === 'driving') {
      drivingLayer?.classList.remove('hidden');
      touchLayer?.classList.add('hidden');
      if (speedSliderWrap) speedSliderWrap.style.display = 'flex';
      this.triggerComicPop('VROOOM! 🚗', 50, 40);
    } else if (mode === 'on-foot') {
      drivingLayer?.classList.add('hidden');
      touchLayer?.classList.remove('hidden');
      if (speedSliderWrap) speedSliderWrap.style.display = 'none';
    } else if (mode === 'repair') {
      this.triggerComicPop('CLANK! 🔧', 50, 40);
    }
  }

  public setTheme(theme: 'day' | 'dusk' | 'night' | 'horde') {
    this.currentTheme = theme;
    document.querySelectorAll('.sim-btn-theme').forEach(b => b.classList.remove('active'));
    document.getElementById(`sim-theme-${theme}`)?.classList.add('active');

    const vignette = document.getElementById('vignette-overlay');
    const timePhase = document.getElementById('time-phase');
    if (!vignette) return;

    vignette.className = '';
    vignette.id = 'vignette-overlay';

    if (theme === 'day') {
      if (timePhase) timePhase.textContent = 'DAY';
      this.clockAngle = 60;
    } else if (theme === 'dusk') {
      vignette.classList.add('dusk');
      if (timePhase) timePhase.textContent = 'DUSK';
      this.clockAngle = 240;
    } else if (theme === 'night') {
      vignette.classList.add('night');
      if (timePhase) timePhase.textContent = 'NIGHT';
      this.clockAngle = 320;
    } else if (theme === 'horde') {
      vignette.classList.add('horde');
      if (timePhase) timePhase.textContent = 'SIREN';
      this.clockAngle = 270;
      this.triggerComicPop('HORDE NIGHT! 🚨', 50, 30);
    }

    const clockHand = document.getElementById('clock-hand');
    if (clockHand) clockHand.style.transform = `translateX(-50%) rotate(${this.clockAngle}deg)`;
  }

  public toggleMobileFrame() {
    this.isMobileFrame = !this.isMobileFrame;
    const stage = document.getElementById('preview-stage');
    const btn = document.getElementById('sim-toggle-frame');
    if (this.isMobileFrame) {
      stage?.classList.add('mobile-frame-active');
      if (btn) btn.textContent = '📱 Mobile: 844x390 [ON]';
      btn?.classList.add('active');
    } else {
      stage?.classList.remove('mobile-frame-active');
      if (btn) btn.textContent = '🖥️ Full Desktop [ON]';
      btn?.classList.remove('active');
    }
  }

  public updateVitals() {
    const fHealth = document.getElementById('fill-health');
    const vHealth = document.getElementById('val-health');
    if (fHealth) fHealth.style.width = `${this.health}%`;
    if (vHealth) vHealth.textContent = `${Math.round(this.health)}`;

    const fHunger = document.getElementById('fill-hunger');
    const vHunger = document.getElementById('val-hunger');
    if (fHunger) fHunger.style.width = `${this.hunger}%`;
    if (vHunger) vHunger.textContent = `${Math.round(this.hunger)}`;

    const fSanity = document.getElementById('fill-sanity');
    const vSanity = document.getElementById('val-sanity');
    if (fSanity) fSanity.style.width = `${this.sanity}%`;
    if (vSanity) vSanity.textContent = `${Math.round(this.sanity)}`;
  }

  public updateSpeed() {
    const valSpeed = document.getElementById('dash-speed-val');
    if (valSpeed) valSpeed.textContent = `${Math.round(this.speedKmH)}`;
  }

  public cycleActionPrompt() {
    this.actionIndex = (this.actionIndex + 1) % this.actionPrompts.length;
    const prompt = this.actionPrompts[this.actionIndex];

    const keyHint = document.querySelector('#action-prompt .key-hint');
    const actionText = document.getElementById('action-text');

    if (keyHint) keyHint.textContent = prompt.key;
    if (actionText) actionText.textContent = prompt.text;

    // Simulate hold progress animation
    const progressFill = document.getElementById('target-progress-fill');
    if (progressFill) {
      progressFill.style.width = '0%';
      setTimeout(() => { if (progressFill) progressFill.style.width = '100%'; }, 50);
      setTimeout(() => { if (progressFill) progressFill.style.width = '0%'; }, 1200);
    }
  }

  public triggerDamageFlash() {
    const flash = document.getElementById('damage-flash');
    if (!flash) return;
    flash.classList.add('flash-active');
    this.health = Math.max(10, this.health - 15);
    this.updateVitals();
    setTimeout(() => flash.classList.remove('flash-active'), 200);
    this.triggerComicPop('OUCH! 💥', 50, 50);
  }

  public triggerComicPop(text: string, xPercent: number = 50, yPercent: number = 50) {
    const container = document.getElementById('comic-pops-layer');
    if (!container) return;

    const el = document.createElement('div');
    el.className = 'comic-pop';
    el.textContent = text;
    el.style.left = `${xPercent}%`;
    el.style.top = `${yPercent}%`;
    container.appendChild(el);

    setTimeout(() => el.remove(), 700);
  }

  public triggerRandomComicPop() {
    const words = ['BONK! 🔨', 'POOF! 💨', 'SQUEAK! 🐭', 'CRANK! ⚙️', 'JACKPOT! 🚨', 'HONK! 🪿', 'SLAP! 🐟'];
    const word = words[Math.floor(Math.random() * words.length)];
    const x = 30 + Math.random() * 40;
    const y = 30 + Math.random() * 40;
    this.triggerComicPop(word, x, y);
  }

  private initHotbarBindings() {
    const slots = document.querySelectorAll('.inv-slot');
    slots.forEach((slot) => {
      slot.addEventListener('click', () => {
        slots.forEach(s => s.classList.remove('selected'));
        slot.classList.add('selected');
        const icon = slot.querySelector('.slot-icon')?.textContent || '';
        this.triggerComicPop(`EQUIP ${icon}`, 50, 80);
      });
    });
  }

  private initClockTick() {
    setInterval(() => {
      this.clockAngle = (this.clockAngle + 0.5) % 360;
      const clockHand = document.getElementById('clock-hand');
      if (clockHand && this.currentTheme === 'day') {
        clockHand.style.transform = `translateX(-50%) rotate(${this.clockAngle}deg)`;
      }
    }, 200);
  }
}

// Auto-instantiate when loaded in browser
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    (window as any).visualShellController = new VisualShellController();
  });
}
