import { VehicleEntity } from '../entities/VehicleEntity';

/**
 * DrivingLayout - Implements Artboard 2 Phone Driving Layout (844x390 landscape).
 * Features:
 * - Responsive mobile driving HUD that activates when seated in a vehicle.
 * - Big responsive Gas Pedal (accelerator) and Brake/Reverse pedal.
 * - Left/Right steering touch pads with progressive input.
 * - Handbrake drift button (triggers lateral slip & tire smoke).
 * - Car Horn button (plays sad trombone / horn & broadcasts noise wave).
 * - Exit Car button (dismount).
 * - Top Dashboard Cluster: Analog/Digital Speedometer (km/h), Fuel gauge, and Durability bar.
 */
export class DrivingLayout {
  private container: HTMLElement | null = null;
  public isVisible: boolean = false;

  // Input states from touch/mouse
  public gasInput: number = 0;
  public brakeInput: number = 0;
  public steerInput: number = 0;
  public handbrakeInput: boolean = false;

  // Callbacks
  private onExitCallback: (() => void) | null = null;
  private onHonkCallback: (() => void) | null = null;

  // DOM Elements
  private speedLabel: HTMLElement | null = null;
  private speedGaugeFill: HTMLElement | null = null;
  private fuelGaugeFill: HTMLElement | null = null;
  private fuelLabel: HTMLElement | null = null;
  private durabilityBarFill: HTMLElement | null = null;
  private vehicleNameLabel: HTMLElement | null = null;

  constructor() {
    this.createLayoutDOM();
    this.bindTouchEvents();
  }

  private createLayoutDOM() {
    if (typeof document === 'undefined') return;

    // Check if layer already exists
    let layer = document.getElementById('driving-controls-layer');
    if (!layer) {
      layer = document.createElement('div');
      layer.id = 'driving-controls-layer';
      layer.className = 'driving-layout-layer hidden';
      document.body.appendChild(layer);
    }
    this.container = layer;

    this.container.innerHTML = `
      <!-- Top Center Dashboard Gauge Cluster -->
      <div id="driving-dash-cluster" class="driving-dash-cluster">
        <div class="dash-vehicle-info">
          <span class="dash-car-icon">🚗</span>
          <span id="dash-car-name" class="dash-car-name">RUSTY HATCHBACK</span>
        </div>

        <div class="dash-meters-row">
          <!-- Speedometer -->
          <div class="dash-meter speed-meter">
            <div class="meter-bar-track">
              <div id="meter-speed-fill" class="meter-bar-fill speed-fill" style="width: 0%"></div>
            </div>
            <div class="meter-numeric">
              <span id="dash-speed-val" class="speed-number">0</span>
              <span class="speed-unit">KM/H</span>
            </div>
          </div>

          <!-- Fuel Gauge -->
          <div class="dash-meter fuel-meter" title="Fuel Tank">
            <span class="meter-icon">⛽</span>
            <div class="meter-bar-track small-track">
              <div id="meter-fuel-fill" class="meter-bar-fill fuel-fill" style="width: 100%"></div>
            </div>
            <span id="dash-fuel-val" class="meter-sub-label">100%</span>
          </div>

          <!-- Durability Bar -->
          <div class="dash-meter health-meter" title="Vehicle Condition">
            <span class="meter-icon">🔧</span>
            <div class="meter-bar-track small-track">
              <div id="meter-durability-fill" class="meter-bar-fill durability-fill" style="width: 100%"></div>
            </div>
            <span id="dash-durability-val" class="meter-sub-label">100%</span>
          </div>
        </div>
      </div>

      <!-- Top Right Exit Button -->
      <button id="btn-drive-exit" class="drive-utility-btn exit-btn" title="Exit Vehicle [E / R]">
        <span class="btn-icon">🚪</span>
        <span class="btn-label">EXIT CAR</span>
      </button>

      <!-- Bottom Left Steering Controls -->
      <div class="driving-steer-zone">
        <button id="btn-steer-left" class="drive-steer-pad pad-left" title="Steer Left [A / ◀]">
          <span class="steer-arrow">◀</span>
          <span class="steer-text">LEFT</span>
        </button>
        <button id="btn-steer-right" class="drive-steer-pad pad-right" title="Steer Right [D / ▶]">
          <span class="steer-text">RIGHT</span>
          <span class="steer-arrow">▶</span>
        </button>
      </div>

      <!-- Bottom Right Pedal & Action Controls -->
      <div class="driving-pedal-zone">
        <!-- Horn / Drift secondary action row -->
        <div class="driving-action-row">
          <button id="btn-drive-honk" class="drive-action-btn horn-btn" title="Honk Horn [H]">
            <span class="action-icon">📢</span>
            <span class="action-label">HORN</span>
          </button>
          <button id="btn-drive-drift" class="drive-action-btn drift-btn" title="Handbrake Drift [Space]">
            <span class="action-icon">⚡</span>
            <span class="action-label">DRIFT</span>
          </button>
        </div>

        <!-- Pedals Row -->
        <div class="driving-pedals-row">
          <button id="btn-pedal-brake" class="drive-pedal pedal-brake" title="Brake / Reverse [S / ▼]">
            <span class="pedal-icon">🛑</span>
            <span class="pedal-label">BRAKE / REV</span>
          </button>
          <button id="btn-pedal-gas" class="drive-pedal pedal-gas" title="Accelerate [W / ▲]">
            <span class="pedal-icon">🟢</span>
            <span class="pedal-label">GAS</span>
          </button>
        </div>
      </div>
    `;

    // Query element references
    this.speedLabel = document.getElementById('dash-speed-val');
    this.speedGaugeFill = document.getElementById('meter-speed-fill');
    this.fuelGaugeFill = document.getElementById('meter-fuel-fill');
    this.fuelLabel = document.getElementById('dash-fuel-val');
    this.durabilityBarFill = document.getElementById('meter-durability-fill');
    this.vehicleNameLabel = document.getElementById('dash-car-name');
  }

  private bindTouchEvents() {
    if (typeof document === 'undefined') return;

    // 1. Steering Left
    const btnLeft = document.getElementById('btn-steer-left');
    if (btnLeft) {
      const startLeft = (e: Event) => { e.preventDefault(); this.steerInput = -1.0; btnLeft.classList.add('active'); };
      const stopLeft = (e: Event) => { e.preventDefault(); if (this.steerInput < 0) this.steerInput = 0; btnLeft.classList.remove('active'); };
      btnLeft.addEventListener('pointerdown', startLeft);
      btnLeft.addEventListener('pointerup', stopLeft);
      btnLeft.addEventListener('pointercancel', stopLeft);
      btnLeft.addEventListener('pointerleave', stopLeft);
    }

    // 2. Steering Right
    const btnRight = document.getElementById('btn-steer-right');
    if (btnRight) {
      const startRight = (e: Event) => { e.preventDefault(); this.steerInput = 1.0; btnRight.classList.add('active'); };
      const stopRight = (e: Event) => { e.preventDefault(); if (this.steerInput > 0) this.steerInput = 0; btnRight.classList.remove('active'); };
      btnRight.addEventListener('pointerdown', startRight);
      btnRight.addEventListener('pointerup', stopRight);
      btnRight.addEventListener('pointercancel', stopRight);
      btnRight.addEventListener('pointerleave', stopRight);
    }

    // 3. Gas Pedal
    const btnGas = document.getElementById('btn-pedal-gas');
    if (btnGas) {
      const startGas = (e: Event) => { e.preventDefault(); this.gasInput = 1.0; btnGas.classList.add('active'); };
      const stopGas = (e: Event) => { e.preventDefault(); this.gasInput = 0; btnGas.classList.remove('active'); };
      btnGas.addEventListener('pointerdown', startGas);
      btnGas.addEventListener('pointerup', stopGas);
      btnGas.addEventListener('pointercancel', stopGas);
      btnGas.addEventListener('pointerleave', stopGas);
    }

    // 4. Brake Pedal
    const btnBrake = document.getElementById('btn-pedal-brake');
    if (btnBrake) {
      const startBrake = (e: Event) => { e.preventDefault(); this.brakeInput = 1.0; btnBrake.classList.add('active'); };
      const stopBrake = (e: Event) => { e.preventDefault(); this.brakeInput = 0; btnBrake.classList.remove('active'); };
      btnBrake.addEventListener('pointerdown', startBrake);
      btnBrake.addEventListener('pointerup', stopBrake);
      btnBrake.addEventListener('pointercancel', stopBrake);
      btnBrake.addEventListener('pointerleave', stopBrake);
    }

    // 5. Drift Button
    const btnDrift = document.getElementById('btn-drive-drift');
    if (btnDrift) {
      const startDrift = (e: Event) => { e.preventDefault(); this.handbrakeInput = true; btnDrift.classList.add('active'); };
      const stopDrift = (e: Event) => { e.preventDefault(); this.handbrakeInput = false; btnDrift.classList.remove('active'); };
      btnDrift.addEventListener('pointerdown', startDrift);
      btnDrift.addEventListener('pointerup', stopDrift);
      btnDrift.addEventListener('pointercancel', stopDrift);
      btnDrift.addEventListener('pointerleave', stopDrift);
    }

    // 6. Horn Button
    const btnHorn = document.getElementById('btn-drive-honk');
    if (btnHorn) {
      btnHorn.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        btnHorn.classList.add('active');
        if (this.onHonkCallback) this.onHonkCallback();
      });
      btnHorn.addEventListener('pointerup', () => btnHorn.classList.remove('active'));
    }

    // 7. Exit Button
    const btnExit = document.getElementById('btn-drive-exit');
    if (btnExit) {
      btnExit.addEventListener('click', (e) => {
        e.preventDefault();
        if (this.onExitCallback) this.onExitCallback();
      });
    }
  }

  public show(vehicle: VehicleEntity) {
    this.isVisible = true;
    if (this.container) {
      this.container.classList.remove('hidden');
    }
    if (this.vehicleNameLabel) {
      this.vehicleNameLabel.innerText = vehicle.def.name.toUpperCase();
    }
    // Hide standard on-foot touch layer while driving
    const onFootLayer = document.getElementById('touch-controls-layer');
    if (onFootLayer) onFootLayer.classList.add('hidden');
  }

  public hide() {
    this.isVisible = false;
    this.gasInput = 0;
    this.brakeInput = 0;
    this.steerInput = 0;
    this.handbrakeInput = false;

    if (this.container) {
      this.container.classList.add('hidden');
    }
    // Restore standard on-foot touch layer
    const onFootLayer = document.getElementById('touch-controls-layer');
    if (onFootLayer) onFootLayer.classList.remove('hidden');
  }

  public update(vehicle: VehicleEntity) {
    if (!this.isVisible) return;

    // 1. Speedometer: m/s -> km/h
    const kmh = Math.round(Math.abs(vehicle.speed) * 3.6);
    if (this.speedLabel) {
      this.speedLabel.innerText = `${kmh}`;
    }
    if (this.speedGaugeFill) {
      const topKmh = (vehicle.def.topSpeed || 14) * 3.6;
      const pct = Math.min(100, Math.round((kmh / topKmh) * 100));
      this.speedGaugeFill.style.width = `${pct}%`;
    }

    // 2. Fuel Gauge
    if (this.fuelGaugeFill && this.fuelLabel) {
      const fuelPct = Math.max(0, Math.min(100, Math.round((vehicle.fuel / vehicle.def.fuelCapacity) * 100)));
      this.fuelGaugeFill.style.width = `${fuelPct}%`;
      this.fuelLabel.innerText = `${fuelPct}%`;
      if (fuelPct < 20) {
        this.fuelGaugeFill.classList.add('low-fuel');
      } else {
        this.fuelGaugeFill.classList.remove('low-fuel');
      }
    }

    // 3. Durability Bar
    if (this.durabilityBarFill) {
      const durPct = Math.max(0, Math.min(100, Math.round(vehicle.durability)));
      this.durabilityBarFill.style.width = `${durPct}%`;
      if (durPct < 30) {
        this.durabilityBarFill.classList.add('danger-durability');
      } else {
        this.durabilityBarFill.classList.remove('danger-durability');
      }
    }
  }

  public onExit(cb: () => void) {
    this.onExitCallback = cb;
  }

  public onHonk(cb: () => void) {
    this.onHonkCallback = cb;
  }
}
