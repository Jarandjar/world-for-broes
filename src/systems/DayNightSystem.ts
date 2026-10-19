import * as THREE from 'three';
import { sounds } from '../audio/SoundEffects';
import { SurvivalStats } from './SurvivalStats';

export class DayNightSystem {
  // Day cycle: 20 minutes per full day (1200s: ~700s Day, ~233s Dusk, ~267s Night)
  public cycleDuration: number = 1200;
  public timeOfDay: number = 80; // start early morning
  public dayCount: number = 1;
  public alwaysDay: boolean = false;

  public ambientLight: THREE.AmbientLight;
  public sunLight: THREE.DirectionalLight;
  private scene: THREE.Scene;

  private darknessTimer: number = 0;
  private charlieAttackInterval: number = 3.5;
  private warningPlayed: boolean = false;
  public isHordeNight: boolean = false;
  public onHordeNightCallback: (() => void) | null = null;
  private currentPhase: string = 'DAY';
  private announcementEl: HTMLElement | null = null;
  private bannerTimeout?: any;

  constructor(scene: THREE.Scene) {
    this.scene = scene;

    this.ambientLight = new THREE.AmbientLight(0xfff5e0, 1.15);
    scene.add(this.ambientLight);

    this.sunLight = new THREE.DirectionalLight(0xfffaed, 1.25);
    this.sunLight.position.set(25, 45, 30);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 1024;
    this.sunLight.shadow.mapSize.height = 1024;
    scene.add(this.sunLight);

    this.createAnnouncementBanner();
  }

  private createAnnouncementBanner() {
    if (typeof document === 'undefined') return;
    let el = document.getElementById('atmosphere-banner');
    if (!el) {
      el = document.createElement('div');
      el.id = 'atmosphere-banner';
      el.className = 'atmosphere-banner hidden';
      document.getElementById('hud-container')?.appendChild(el);
    }
    this.announcementEl = el;
  }

  public showAtmosphereBanner(text: string, subtext?: string) {
    if (typeof document === 'undefined') return;
    if (!this.announcementEl) this.createAnnouncementBanner();
    if (!this.announcementEl) return;

    if (this.bannerTimeout) clearTimeout(this.bannerTimeout);
    this.announcementEl.innerHTML = `
      <div class="banner-title">${text}</div>
      ${subtext ? `<div class="banner-sub">${subtext}</div>` : ''}
    `;
    this.announcementEl.classList.remove('hidden');
    this.announcementEl.classList.add('banner-visible');

    this.bannerTimeout = setTimeout(() => {
      this.announcementEl?.classList.remove('banner-visible');
      setTimeout(() => this.announcementEl?.classList.add('hidden'), 500);
    }, 4500);
  }

  public toggleAlwaysDay(): boolean {
    this.alwaysDay = !this.alwaysDay;
    if (this.alwaysDay) {
      this.timeOfDay = 80;
      this.currentPhase = 'DAY';
      this.isHordeNight = false;
      this.showAtmosphereBanner('☀️ ALWAYS DAYLIGHT: ON', 'Night time toggled off. Continuous warm sunshine.');
    } else {
      this.showAtmosphereBanner('🌙 DAY/NIGHT CYCLE: RESUMED', 'The clock turns once more. Prepare for dusk and nightfall.');
    }
    return this.alwaysDay;
  }

  public update(deltaSeconds: number, stats: SurvivalStats, isPlayerInLight: boolean) {
    if (this.alwaysDay) {
      this.timeOfDay = 80;
    } else {
      this.timeOfDay += deltaSeconds;
      if (this.timeOfDay >= this.cycleDuration) {
        this.timeOfDay = 0;
        this.dayCount += 1;
        stats.daysSurvived = this.dayCount;
      }
    }

    const progress = this.timeOfDay / this.cycleDuration; // 0.0 to 1.0
    const degrees = progress * 360;

    // Phase identification
    // Day: 0 - 210 deg (58%)
    // Dusk: 210 - 280 deg (19%)
    // Night: 280 - 360 deg (23%)
    let phase = 'DAY';
    if (degrees >= 210 && degrees < 280) {
      phase = 'DUSK';
    } else if (degrees >= 280) {
      phase = 'NIGHT';
    }

    // Phase change announcements
    if (phase !== this.currentPhase) {
      if (phase === 'DUSK') {
        const isHordeNight = this.dayCount % 7 === 0;
        if (isHordeNight) {
          this.isHordeNight = true;
          this.showAtmosphereBanner('🚨 7th-DAY HORDE NIGHT HAS ARRIVED! 🚨', 'The air-raid siren wails! The Highway Hulk and relentless hordes converge!');
          sounds.playSiren();
          if (this.onHordeNightCallback) this.onHordeNightCallback();
        } else {
          this.isHordeNight = false;
          this.showAtmosphereBanner('🌅 Dusk is Falling...', 'The wilderness turns cold. Light a campfire before darkness arrives!');
          sounds.playCharlieWarning();
        }
      } else if (phase === 'NIGHT') {
        if (this.isHordeNight) {
          this.showAtmosphereBanner('🌑 BLOOD MOON NIGHT', 'Defend your base & car from the relentless horde!');
        } else {
          this.showAtmosphereBanner('🌑 Pitch Black Night', 'Beware the creeping shadows! Stay inside the firelight.');
        }
      } else if (phase === 'DAY') {
        this.isHordeNight = false;
        this.showAtmosphereBanner(`☀️ Day ${this.dayCount} Dawns`, 'The morning sun returns. Gather resources and prepare.');
      }
      this.currentPhase = phase;
    }

    // Orbit sun light angle
    const sunAngle = (progress * Math.PI * 2) - Math.PI / 2;
    this.sunLight.position.set(
      Math.cos(sunAngle) * 40,
      Math.max(5, Math.sin(sunAngle) * 45),
      Math.sin(sunAngle) * 30
    );

    // Continuous smoothstep lighting & fog
    const vignetteEl = typeof document !== 'undefined' ? document.getElementById('vignette-overlay') : null;

    if (degrees < 210) {
      // DAY: Golden sunlight
      this.ambientLight.color.setHex(0xfff3db);
      this.ambientLight.intensity = 1.15;
      this.sunLight.color.setHex(0xfffaed);
      this.sunLight.intensity = 1.25;

      if (this.scene.fog instanceof THREE.FogExp2) {
        this.scene.fog.color.setHex(0x1a1612);
        this.scene.fog.density = 0.015;
      }
      this.scene.background = new THREE.Color(0x1a1612);
      if (vignetteEl) vignetteEl.style.opacity = '0.15';
    } else if (degrees >= 210 && degrees < 280) {
      // DUSK: Continuous Hermite smooth transition
      const t = (degrees - 210) / 70; // 0 to 1
      const smooth = t * t * (3 - 2 * t);

      // Ambient shifts from warm gold to amber twilight
      const ambR = 0.95 - smooth * 0.45;
      const ambG = 0.85 - smooth * 0.60;
      const ambB = 0.70 - smooth * 0.50;
      this.ambientLight.color.setRGB(ambR, ambG, ambB);
      this.ambientLight.intensity = 1.15 - smooth * 0.85;

      // Sun shifts to burning sunset bronze and fades
      this.sunLight.color.setRGB(1.0, 0.65 - smooth * 0.45, 0.2 - smooth * 0.15);
      this.sunLight.intensity = Math.max(0, 1.25 * (1 - smooth));

      // Fog & Background shift to twilight indigo
      if (this.scene.fog instanceof THREE.FogExp2) {
        this.scene.fog.color.setRGB(0.1 + (1 - smooth) * 0.05, 0.08 - smooth * 0.04, 0.14 - smooth * 0.05);
        this.scene.fog.density = 0.015 + smooth * 0.008;
      }
      this.scene.background = new THREE.Color(0.1 + (1 - smooth) * 0.05, 0.08 - smooth * 0.04, 0.14 - smooth * 0.05);

      if (vignetteEl) {
        vignetteEl.style.opacity = `${0.2 + smooth * 0.45}`;
      }
    } else {
      // NIGHT: Deep blackness
      this.ambientLight.color.setHex(0x060812);
      this.ambientLight.intensity = 0.06;
      this.sunLight.intensity = 0.0;

      if (this.scene.fog instanceof THREE.FogExp2) {
        this.scene.fog.color.setHex(0x04050a);
        this.scene.fog.density = 0.024;
      }
      this.scene.background = new THREE.Color(0x04050a);

      if (vignetteEl) {
        vignetteEl.style.opacity = isPlayerInLight ? '0.45' : '0.95';
      }
    }

    // Update DST Clock UI
    if (typeof document !== 'undefined') {
      const clockHand = document.getElementById('clock-hand');
      const dayCounter = document.getElementById('day-counter');
      const timePhase = document.getElementById('time-phase');

      if (clockHand) {
        clockHand.style.transform = `rotate(${degrees}deg)`;
      }
      if (dayCounter) {
        dayCounter.innerText = `DAY ${this.dayCount}`;
      }
      if (timePhase) {
        timePhase.innerText = this.alwaysDay ? 'DAY ☀️' : phase;
        timePhase.style.color = (phase === 'DAY' || this.alwaysDay) ? '#ebd197' : phase === 'DUSK' ? '#ff7f50' : '#88a0e0';
      }
    }

    // Dynamic Atmosphere BGM (Suno/Studio ambient tracks)
    if (phase === 'NIGHT' && !this.alwaysDay) {
      sounds.playBGM('bgm_night');
    } else {
      sounds.playBGM('bgm_day');
    }

    // Darkness / Charlie mechanic during Night
    const terrorEl = typeof document !== 'undefined' ? document.getElementById('darkness-terror') : null;
    if (!this.alwaysDay && phase === 'NIGHT' && !isPlayerInLight) {
      this.darknessTimer += deltaSeconds;

      if (terrorEl) {
        terrorEl.classList.remove('hidden');
      }

      if (this.darknessTimer > 1.2 && !this.warningPlayed) {
        sounds.playCharlieWarning();
        this.warningPlayed = true;
      }

      if (this.darknessTimer >= this.charlieAttackInterval) {
        // Charlie strikes!
        sounds.playCharlieSlash();
        stats.takeDamage(25, 'The darkness (Charlie)');
        this.darknessTimer = 1.5; // Quick subsequent strikes
      }
    } else {
      this.darknessTimer = 0;
      this.warningPlayed = false;
      if (terrorEl) {
        terrorEl.classList.add('hidden');
      }
    }
  }

  public isNight(): boolean {
    if (this.alwaysDay) return false;
    const degrees = (this.timeOfDay / this.cycleDuration) * 360;
    return degrees >= 280;
  }
}

