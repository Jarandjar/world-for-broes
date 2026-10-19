import * as THREE from 'three';
import { VehicleEntity } from '../entities/VehicleEntity';
import { InventorySystem } from '../systems/InventorySystem';
import { ParticleManager } from '../entities/WorldObjects';
import { ComicPops } from './ComicPops';
import { sounds } from '../audio/SoundEffects';

/**
 * VehicleRepairSheet - Implements Cards UI Bottom Sheet for Vehicle Inspection & Part Repair (Phase Z7).
 * Features:
 * - Part cards grid with condition %, status, and hold-to-repair radial ring fill tween.
 * - Part-fit bounce snap animations upon repair completion with starburst FX.
 * - Fuel siphoning controls (transfer fuel between vehicle tank and inventory canister).
 * - Crank mini-game for engine ignition timing.
 */
export class VehicleRepairSheet {
  private container: HTMLElement | null = null;
  public isOpen: boolean = false;
  public activeVehicle: VehicleEntity | null = null;
  private inventory: InventorySystem | null = null;
  private particles: ParticleManager | null = null;
  private camera: THREE.Camera | null = null;

  // Hold-to-repair tracking
  private repairTimer: any = null;
  private activeRepairSlot: string | null = null;
  private repairProgress: number = 0; // 0 to 1

  // Crank mini-game state
  public isCranking: boolean = false;
  public crankProgress: number = 0;
  public crankDirection: number = 1;
  private crankAnimFrame: any = null;

  constructor() {
    this.createSheetDOM();
  }

  private createSheetDOM() {
    if (typeof document === 'undefined') return;

    let el = document.getElementById('vehicle-repair-sheet');
    if (!el) {
      el = document.createElement('div');
      el.id = 'vehicle-repair-sheet';
      el.className = 'vehicle-repair-sheet hidden';
      document.body.appendChild(el);
    }
    this.container = el;
  }

  public open(
    vehicle: VehicleEntity,
    inventory: InventorySystem,
    particles?: ParticleManager,
    camera?: THREE.Camera
  ) {
    this.activeVehicle = vehicle;
    this.inventory = inventory;
    this.particles = particles || null;
    this.camera = camera || null;
    this.isOpen = true;

    this.render();
    if (this.container) {
      this.container.classList.remove('hidden');
    }

    sounds.playCarDoorOpen();
  }

  public close() {
    this.isOpen = false;
    this.activeVehicle = null;
    this.cancelRepairHold();
    this.stopCrankMiniGame();

    if (this.container) {
      this.container.classList.add('hidden');
    }
  }

  public render() {
    if (!this.container || !this.activeVehicle) return;

    const v = this.activeVehicle;
    let partsHTML = '';

    for (const [slot, part] of v.parts.entries()) {
      const isDamaged = part.condition < 100;
      const statusClass = part.condition < 30 ? 'critical' : part.condition < 70 ? 'worn' : 'good';
      const statusText = part.condition < 30 ? 'CRITICAL' : part.condition < 70 ? 'WORN' : 'GOOD';

      partsHTML += `
        <div class="part-card ${statusClass}" data-slot="${slot}">
          <div class="part-card-header">
            <span class="part-icon">${part.icon}</span>
            <div class="part-meta">
              <span class="part-title">${part.name}</span>
              <span class="part-status-badge ${statusClass}">${statusText}</span>
            </div>
            <span class="part-condition-pct">${Math.round(part.condition)}%</span>
          </div>

          <div class="part-condition-bar">
            <div class="condition-fill ${statusClass}" style="width: ${part.condition}%"></div>
          </div>

          <div class="part-card-actions">
            ${isDamaged ? `
              <button class="btn-hold-repair" data-slot="${slot}">
                <svg class="radial-ring-svg" width="28" height="28" viewBox="0 0 36 36">
                  <path class="radial-ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                  <path class="radial-ring-fill ring-${slot}" stroke-dasharray="100, 100" stroke-dashoffset="100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <span class="repair-label">HOLD TO REPAIR</span>
              </button>
            ` : `
              <div class="part-pristine-badge">✓ OPERABLE</div>
            `}
          </div>
        </div>
      `;
    }

    this.container.innerHTML = `
      <div class="sheet-scrim"></div>
      <div class="sheet-dialog ui-sticker">
        <!-- Sheet Header -->
        <div class="sheet-header">
          <div class="sheet-title-group">
            <span class="sheet-car-icon">🚗</span>
            <div>
              <h2 class="sheet-title">${v.def.name.toUpperCase()}</h2>
              <p class="sheet-subtitle">ENGINE BAY & COMPONENT INSPECTION</p>
            </div>
          </div>
          <button id="btn-sheet-close" class="btn-sheet-close" title="Close [Esc / R]">✕</button>
        </div>

        <!-- Vehicle Health & Tank Summary -->
        <div class="sheet-vitals-row">
          <div class="vital-item">
            <span class="vital-label">CONDITION</span>
            <div class="vital-bar"><div class="vital-fill health-fill" style="width: ${v.durability}%"></div></div>
            <span class="vital-val">${v.durability}%</span>
          </div>
          <div class="vital-item">
            <span class="vital-label">FUEL TANK</span>
            <div class="vital-bar"><div class="vital-fill fuel-fill" style="width: ${(v.fuel / v.def.fuelCapacity) * 100}%"></div></div>
            <span class="vital-val">${v.fuel.toFixed(1)}L / ${v.def.fuelCapacity}L</span>
          </div>
          <div class="vital-item">
            <span class="vital-label">BATTERY CHARGE</span>
            <div class="vital-bar"><div class="vital-fill battery-fill" style="width: ${v.batteryCharge}%"></div></div>
            <span class="vital-val">${v.batteryCharge}%</span>
          </div>
        </div>

        <!-- Parts Grid Cards -->
        <div class="sheet-parts-grid">
          ${partsHTML}
        </div>

        <!-- Quick Fuel Siphon & Crank Footer Row -->
        <div class="sheet-footer-tools">
          <div class="fuel-transfer-group">
            <button id="btn-siphon-fuel" class="tool-btn siphon-btn">
              <span>⛽</span> SIPHON 5L FUEL
            </button>
            <button id="btn-pour-fuel" class="tool-btn pour-btn">
              <span>🛢️</span> POUR 5L CANISTER
            </button>
          </div>

          <!-- Cold Start Crank Mini-Game -->
          <div class="crank-minigame-container">
            <div class="crank-meter-track">
              <div class="crank-sweet-spot"></div>
              <div id="crank-needle" class="crank-needle"></div>
            </div>
            <button id="btn-crank-engine" class="tool-btn crank-btn">
              <span>⚡</span> CRANK IGNITION
            </button>
          </div>
        </div>
      </div>
    `;

    this.bindEvents();
    this.startCrankMiniGame();
  }

  private bindEvents() {
    if (!this.container) return;

    // Close button
    document.getElementById('btn-sheet-close')?.addEventListener('click', () => {
      this.close();
    });

    // Scrim click close
    this.container.querySelector('.sheet-scrim')?.addEventListener('click', () => {
      this.close();
    });

    // Hold-to-repair buttons
    const repairBtns = this.container.querySelectorAll('.btn-hold-repair');
    repairBtns.forEach(btn => {
      const slot = btn.getAttribute('data-slot');
      if (!slot) return;

      const startHold = (e: Event) => {
        e.preventDefault();
        this.startRepairHold(slot);
      };
      const stopHold = (e: Event) => {
        e.preventDefault();
        this.cancelRepairHold();
      };

      btn.addEventListener('pointerdown', startHold);
      btn.addEventListener('pointerup', stopHold);
      btn.addEventListener('pointercancel', stopHold);
      btn.addEventListener('pointerleave', stopHold);
    });

    // Siphon Fuel Button
    document.getElementById('btn-siphon-fuel')?.addEventListener('click', () => {
      if (!this.activeVehicle) return;
      const siphoned = this.activeVehicle.siphonFuel(5);
      if (siphoned > 0) {
        sounds.playChestOpen();
        if (this.camera && this.activeVehicle) {
          ComicPops.show(`SIPHONED ${siphoned.toFixed(1)}L ⛽`, this.activeVehicle.position, this.camera);
        }
        this.render();
      }
    });

    // Pour Fuel Button
    document.getElementById('btn-pour-fuel')?.addEventListener('click', () => {
      if (!this.activeVehicle) return;
      const added = this.activeVehicle.addFuel(5);
      if (added > 0) {
        sounds.playChestOpen();
        if (this.camera && this.activeVehicle) {
          ComicPops.show(`ADDED ${added.toFixed(1)}L 🛢️`, this.activeVehicle.position, this.camera);
        }
        this.render();
      }
    });

    // Crank Ignition Mini-Game Button
    document.getElementById('btn-crank-engine')?.addEventListener('click', () => {
      this.attemptCrankIgnition();
    });
  }

  /**
   * Hold-to-repair radial ring fill tween
   */
  private startRepairHold(slot: string) {
    this.cancelRepairHold();
    this.activeRepairSlot = slot;
    this.repairProgress = 0;

    const ringPath = this.container?.querySelector(`.ring-${slot}`) as SVGPathElement | null;
    const startTime = Date.now();
    const holdDurationMs = 1500; // 1.5 seconds

    this.repairTimer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      this.repairProgress = Math.min(1.0, elapsed / holdDurationMs);

      // Update SVG radial ring offset (100 -> 0)
      if (ringPath) {
        const offset = 100 - this.repairProgress * 100;
        ringPath.setAttribute('stroke-dashoffset', `${offset}`);
      }

      if (this.repairProgress >= 1.0) {
        this.completeRepair(slot);
      }
    }, 30);
  }

  private cancelRepairHold() {
    if (this.repairTimer) {
      clearInterval(this.repairTimer);
      this.repairTimer = null;
    }
    if (this.activeRepairSlot) {
      const ringPath = this.container?.querySelector(`.ring-${this.activeRepairSlot}`) as SVGPathElement | null;
      if (ringPath) {
        ringPath.setAttribute('stroke-dashoffset', '100');
      }
    }
    this.activeRepairSlot = null;
    this.repairProgress = 0;
  }

  private completeRepair(slot: string) {
    this.cancelRepairHold();
    if (!this.activeVehicle) return;

    // Restore part condition to 100%
    this.activeVehicle.repairPart(slot, 100);

    // Audio & celebratory visual effects
    sounds.playCarDoorShut();
    if (this.particles && this.activeVehicle) {
      this.particles.spawnSpawnSparkles(this.activeVehicle.position, 12);
    }
    if (this.camera && this.activeVehicle) {
      ComicPops.show('REPAIRED! 🔧', this.activeVehicle.position, this.camera, '#00F5D4');
    }

    this.render();
  }

  /**
   * Crank mini-game oscillation loop
   */
  private startCrankMiniGame() {
    this.stopCrankMiniGame();
    this.crankProgress = 0.1;
    this.crankDirection = 1;

    const loop = () => {
      if (!this.isOpen) return;

      this.crankProgress += this.crankDirection * 0.025;
      if (this.crankProgress >= 0.95) {
        this.crankProgress = 0.95;
        this.crankDirection = -1;
      } else if (this.crankProgress <= 0.05) {
        this.crankProgress = 0.05;
        this.crankDirection = 1;
      }

      const needle = document.getElementById('crank-needle');
      if (needle) {
        needle.style.left = `${this.crankProgress * 100}%`;
      }

      this.crankAnimFrame = requestAnimationFrame(loop);
    };

    this.crankAnimFrame = requestAnimationFrame(loop);
  }

  private stopCrankMiniGame() {
    if (this.crankAnimFrame) {
      cancelAnimationFrame(this.crankAnimFrame);
      this.crankAnimFrame = null;
    }
  }

  private attemptCrankIgnition() {
    if (!this.activeVehicle) return;

    // Sweet spot is between 0.40 and 0.60
    const inSweetSpot = this.crankProgress >= 0.40 && this.crankProgress <= 0.60;

    if (inSweetSpot) {
      sounds.playEngineRev();
      if (this.particles && this.activeVehicle) {
        this.particles.spawnDustPuffs(this.activeVehicle.position);
      }
      if (this.camera && this.activeVehicle) {
        ComicPops.show('VROOM!! ⚡', this.activeVehicle.position, this.camera, '#2ECC71');
      }
    } else {
      sounds.playSadTrombone();
      if (this.camera && this.activeVehicle) {
        ComicPops.show('SPUTTER... 💨', this.activeVehicle.position, this.camera, '#E74C3C');
      }
    }
  }
}
