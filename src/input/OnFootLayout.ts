import { actionMap, GameAction } from './ActionMap';
import { InventorySystem } from '../systems/InventorySystem';
import { SurvivalStats } from '../systems/SurvivalStats';
import { ITEM_REGISTRY } from '../registries/ItemDef';

export interface ContextualActionInfo {
  type: 'chop' | 'mine' | 'harvest' | 'loot' | 'attack' | 'cook' | 'action';
  label: string;
  icon: string;
}

/**
 * OnFootLayout - Implements Page 1 Phone Layout from Artboard 1 (844x390 landscape).
 * Features:
 * - Left 40% floating virtual thumbstick with dynamic origin and outer Sprint Ring threshold (52px).
 * - Sneak Turtle toggle button.
 * - Big contextual ACTION button with dynamic icon & title updates.
 * - Quick-use consumable and drag-to-aim throw buttons.
 * - 5-slot swipeable bottom hotbar.
 * - Dynamic condition HUD: Infection %, Encumbrance Weight, and Noise Ripple level (0-5).
 */
export class OnFootLayout {
  private container: HTMLElement | null = null;
  private inventory: InventorySystem;
  private stats: SurvivalStats;

  // Touch tracking for floating thumbstick
  private stickTouchId: number | null = null;
  private stickOrigin = { x: 0, y: 0 };
  private readonly STICK_MAX_RADIUS: number = 54;
  private readonly SPRINT_THRESHOLD_RADIUS: number = 46;

  // DOM elements
  private stickZone: HTMLElement | null = null;
  private stickBase: HTMLElement | null = null;
  private stickKnob: HTMLElement | null = null;
  private sprintRing: HTMLElement | null = null;
  private btnSneak: HTMLElement | null = null;
  private btnAction: HTMLElement | null = null;
  private btnInteract: HTMLElement | null = null;
  private btnQuickUse: HTMLElement | null = null;
  private btnThrow: HTMLElement | null = null;
  private hotbarSlots: HTMLElement[] = [];
  private infectionChip: HTMLElement | null = null;
  private weightChip: HTMLElement | null = null;
  private noiseChip: HTMLElement | null = null;

  // Noise tracking
  public currentNoiseLevel: number = 0; // 0 to 5

  // Throw aiming state
  public isAimingThrow: boolean = false;
  public throwAimAngle: number = 0;

  constructor(inventory: InventorySystem, stats: SurvivalStats) {
    this.inventory = inventory;
    this.stats = stats;

    this.createLayoutDOM();
    this.bindTouchEvents();
  }

  private createLayoutDOM() {
    // Remove existing basic touch controls if present
    const existing = document.getElementById('touch-controls-layer');
    if (existing) existing.remove();

    const layer = document.createElement('div');
    layer.id = 'touch-controls-layer';
    layer.className = 'on-foot-layout-layer';
    layer.innerHTML = `
      <!-- Left 40% Floating Joystick Zone -->
      <div id="onfoot-stick-zone" class="stick-zone-left40">
        <div id="onfoot-stick-base" class="stick-base hidden">
          <div id="onfoot-sprint-ring" class="sprint-outer-ring">
            <span class="sprint-label">SPRINT</span>
          </div>
          <div id="onfoot-stick-knob" class="stick-knob"></div>
        </div>
      </div>

      <!-- Sneak Turtle Toggle Button (Left of stick zone) -->
      <button id="btn-touch-sneak" class="touch-pill-btn sneak-btn" title="Toggle Sneak [C]">
        <span class="btn-icon">🐢</span>
        <span class="btn-label">SNEAK</span>
      </button>

      <!-- Top Condition & Threat HUD Chips -->
      <div id="onfoot-threat-hud" class="threat-hud-bar">
        <!-- Noise Ripple Level (0 to 5) -->
        <div id="chip-noise" class="threat-chip noise-chip" title="Acoustic Noise Level">
          <span class="chip-icon">🔊</span>
          <div class="noise-meter-bars">
            <div class="noise-bar bar-1"></div>
            <div class="noise-bar bar-2"></div>
            <div class="noise-bar bar-3"></div>
            <div class="noise-bar bar-4"></div>
            <div class="noise-bar bar-5"></div>
          </div>
          <span id="label-noise-val" class="chip-val">SILENT</span>
        </div>

        <!-- Encumbrance Weight Load -->
        <div id="chip-weight" class="threat-chip weight-chip" title="Backpack Encumbrance">
          <span class="chip-icon">🎒</span>
          <div class="weight-bar-bg"><div id="fill-weight" class="weight-bar-fill"></div></div>
          <span id="label-weight-val" class="chip-val">12/40 kg</span>
        </div>

        <!-- Infection Buildup Meter -->
        <div id="chip-infection" class="threat-chip infection-chip" title="Zombie Infection Level">
          <span class="chip-icon">☣️</span>
          <div class="infection-bar-bg"><div id="fill-infection" class="infection-bar-fill"></div></div>
          <span id="label-infection-val" class="chip-val">0%</span>
        </div>
      </div>

      <!-- Bottom 5-Slot Swipeable Hotbar -->
      <div id="onfoot-swipe-hotbar" class="swipeable-hotbar">
        <div class="hotbar-slots-track" id="hotbar-track">
          <button class="onfoot-slot active" data-slot="0"><span class="slot-num">1</span><span class="slot-icon">🪓</span><span class="slot-qty">1</span></button>
          <button class="onfoot-slot" data-slot="1"><span class="slot-num">2</span><span class="slot-icon">🪵</span><span class="slot-qty">5</span></button>
          <button class="onfoot-slot" data-slot="2"><span class="slot-num">3</span><span class="slot-icon">🍖</span><span class="slot-qty">3</span></button>
          <button class="onfoot-slot" data-slot="3"><span class="slot-num">4</span><span class="slot-icon">🔦</span><span class="slot-qty">1</span></button>
          <button class="onfoot-slot" data-slot="4"><span class="slot-num">5</span><span class="slot-icon">🪨</span><span class="slot-qty">4</span></button>
        </div>
      </div>

      <!-- Right Action Cluster -->
      <div id="onfoot-action-cluster" class="action-cluster-right">
        <!-- Sub-actions arc -->
        <div class="sub-actions-arc">
          <button id="btn-touch-quickuse" class="touch-circle-btn quickuse-btn" title="Quick Use Food / Meds [F]">
            <span class="btn-icon">🍖</span>
            <span class="btn-mini-label">EAT</span>
          </button>

          <button id="btn-touch-throw" class="touch-circle-btn throw-btn" title="Aim & Throw [T]">
            <span class="btn-icon">💣</span>
            <span class="btn-mini-label">THROW</span>
          </button>

          <button id="btn-touch-interact" class="touch-circle-btn interact-btn" title="Interact / Open Door [E]">
            <span class="btn-icon">🚪</span>
            <span class="btn-mini-label">OPEN</span>
          </button>

          <button id="btn-touch-honk" class="touch-circle-btn honk-btn" title="Honk / Alert [H]">
            <span class="btn-icon">📢</span>
            <span class="btn-mini-label">HONK</span>
          </button>
        </div>

        <!-- Big Primary Contextual ACTION Button -->
        <button id="btn-touch-primary-action" class="big-action-button" title="Primary Action [SPACE]">
          <div class="action-inner-ring">
            <span id="label-action-icon" class="main-action-icon">🪓</span>
            <span id="label-action-title" class="main-action-title">ACTION</span>
          </div>
        </button>
      </div>
    `;

    document.getElementById('game-container')?.appendChild(layer);
    this.container = layer;

    // Cache elements
    this.stickZone = document.getElementById('onfoot-stick-zone');
    this.stickBase = document.getElementById('onfoot-stick-base');
    this.stickKnob = document.getElementById('onfoot-stick-knob');
    this.sprintRing = document.getElementById('onfoot-sprint-ring');
    this.btnSneak = document.getElementById('btn-touch-sneak');
    this.btnAction = document.getElementById('btn-touch-primary-action');
    this.btnInteract = document.getElementById('btn-touch-interact');
    this.btnQuickUse = document.getElementById('btn-touch-quickuse');
    this.btnThrow = document.getElementById('btn-touch-throw');
    this.infectionChip = document.getElementById('chip-infection');
    this.weightChip = document.getElementById('chip-weight');
    this.noiseChip = document.getElementById('chip-noise');
    this.hotbarSlots = Array.from(document.querySelectorAll('.onfoot-slot'));
  }

  private triggerHaptic(ms: number = 15) {
    try {
      if ('vibrate' in navigator) navigator.vibrate(ms);
    } catch {
      // Haptics unavailable
    }
  }

  private bindTouchEvents() {
    // 1. Floating Stick in Left 40% of Screen
    if (this.stickZone) {
      this.stickZone.addEventListener('touchstart', (e: TouchEvent) => {
        e.preventDefault();
        if (this.stickTouchId !== null) return;

        const touch = e.changedTouches[0];
        this.stickTouchId = touch.identifier;
        this.stickOrigin = { x: touch.clientX, y: touch.clientY };

        // Position base under finger
        if (this.stickBase) {
          this.stickBase.classList.remove('hidden');
          this.stickBase.style.left = `${touch.clientX}px`;
          this.stickBase.style.top = `${touch.clientY}px`;
        }
        if (this.stickKnob) {
          this.stickKnob.style.transform = `translate(0px, 0px)`;
        }
      }, { passive: false });

      this.stickZone.addEventListener('touchmove', (e: TouchEvent) => {
        e.preventDefault();
        if (this.stickTouchId === null) return;

        for (let i = 0; i < e.changedTouches.length; i++) {
          const touch = e.changedTouches[i];
          if (touch.identifier === this.stickTouchId) {
            const dx = touch.clientX - this.stickOrigin.x;
            const dy = touch.clientY - this.stickOrigin.y;
            const dist = Math.hypot(dx, dy);

            // Clamp knob movement to max radius
            const clampedDist = Math.min(dist, this.STICK_MAX_RADIUS);
            const angle = Math.atan2(dy, dx);
            const knobX = Math.cos(angle) * clampedDist;
            const knobY = Math.sin(angle) * clampedDist;

            if (this.stickKnob) {
              this.stickKnob.style.transform = `translate(${knobX}px, ${knobY}px)`;
            }

            // Analog movement vector
            actionMap.analogMove.x = knobX / this.STICK_MAX_RADIUS;
            actionMap.analogMove.y = knobY / this.STICK_MAX_RADIUS;

            // Sprint Ring threshold detection
            const isCrossingSprint = dist >= this.SPRINT_THRESHOLD_RADIUS;
            if (isCrossingSprint && !actionMap.isSprintLocked) {
              actionMap.isSprintLocked = true;
              this.sprintRing?.classList.add('sprint-active');
              this.triggerHaptic(25);
            } else if (!isCrossingSprint && actionMap.isSprintLocked) {
              actionMap.isSprintLocked = false;
              this.sprintRing?.classList.remove('sprint-active');
            }
            break;
          }
        }
      }, { passive: false });

      const onStickEnd = (e: TouchEvent) => {
        for (let i = 0; i < e.changedTouches.length; i++) {
          if (e.changedTouches[i].identifier === this.stickTouchId) {
            this.stickTouchId = null;
            actionMap.analogMove.x = 0;
            actionMap.analogMove.y = 0;
            actionMap.isSprintLocked = false;
            this.stickBase?.classList.add('hidden');
            this.sprintRing?.classList.remove('sprint-active');
            break;
          }
        }
      };

      this.stickZone.addEventListener('touchend', onStickEnd, { passive: false });
      this.stickZone.addEventListener('touchcancel', onStickEnd, { passive: false });
    }

    // 2. Sneak Turtle Button
    this.btnSneak?.addEventListener('click', (e) => {
      e.preventDefault();
      actionMap.isSneakToggled = !actionMap.isSneakToggled;
      this.btnSneak?.classList.toggle('active', actionMap.isSneakToggled);
      this.triggerHaptic(18);
    });

    // 3. Primary Action Button (Hold-to-repeat + 120ms buffer)
    this.btnAction?.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      actionMap.handleKeyDown('Space');
      this.triggerHaptic(20);
      this.btnAction?.classList.add('pressed');
    });

    const onActionRelease = (e: PointerEvent) => {
      actionMap.handleKeyUp('Space');
      this.btnAction?.classList.remove('pressed');
    };
    this.btnAction?.addEventListener('pointerup', onActionRelease);
    this.btnAction?.addEventListener('pointercancel', onActionRelease);

    // 4. Interact Button
    this.btnInteract?.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      actionMap.handleKeyDown('KeyR');
      this.triggerHaptic(15);
    });
    this.btnInteract?.addEventListener('pointerup', () => actionMap.handleKeyUp('KeyR'));

    // 5. Quick-Use Button
    this.btnQuickUse?.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      actionMap.handleKeyDown('KeyF');
      this.triggerHaptic(15);
    });
    this.btnQuickUse?.addEventListener('pointerup', () => actionMap.handleKeyUp('KeyF'));

    // 6. Throw Button (Drag to aim)
    this.btnThrow?.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.isAimingThrow = true;
      this.btnThrow?.classList.add('aiming');
      this.triggerHaptic(20);
    });

    window.addEventListener('pointermove', (e: PointerEvent) => {
      if (this.isAimingThrow && this.btnThrow) {
        const rect = this.btnThrow.getBoundingClientRect();
        const cx = rect.left + rect.width * 0.5;
        const cy = rect.top + rect.height * 0.5;
        this.throwAimAngle = Math.atan2(e.clientY - cy, e.clientX - cx);
      }
    });

    const onThrowRelease = () => {
      if (this.isAimingThrow) {
        this.isAimingThrow = false;
        this.btnThrow?.classList.remove('aiming');
        actionMap.bufferAction('THROW', performance.now(), { aimAngle: this.throwAimAngle });
        this.triggerHaptic(25);
      }
    };
    window.addEventListener('pointerup', onThrowRelease);
    window.addEventListener('pointercancel', onThrowRelease);

    // 7. Honk Button
    document.getElementById('btn-touch-honk')?.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      actionMap.handleKeyDown('KeyH');
      this.triggerHaptic(30);
    });
    document.getElementById('btn-touch-honk')?.addEventListener('pointerup', () => actionMap.handleKeyUp('KeyH'));

    // 8. 5-Slot Hotbar Touch Tapping & Swiping
    this.hotbarSlots.forEach((slot, index) => {
      slot.addEventListener('click', (e) => {
        e.preventDefault();
        this.setActiveHotbarSlot(index);
        actionMap.handleKeyDown(`Digit${index + 1}`);
        actionMap.handleKeyUp(`Digit${index + 1}`);
        this.triggerHaptic(15);
      });
    });

    // Hotbar track horizontal touch swipe
    const track = document.getElementById('hotbar-track');
    let swipeStartX = 0;
    track?.addEventListener('touchstart', (e) => {
      swipeStartX = e.touches[0].clientX;
    }, { passive: true });
    track?.addEventListener('touchend', (e) => {
      const deltaX = e.changedTouches[0].clientX - swipeStartX;
      if (Math.abs(deltaX) > 40) {
        // Cycle page or hotbar
        this.triggerHaptic(15);
      }
    }, { passive: true });
  }

  /**
   * Sets active highlighted slot on 5-slot mobile hotbar.
   */
  public setActiveHotbarSlot(index: number) {
    this.hotbarSlots.forEach((s, idx) => {
      s.classList.toggle('active', idx === index);
    });
  }

  /**
   * Updates contextual action icon and title dynamically based on nearest object.
   */
  public setContextualAction(action: ContextualActionInfo | null) {
    const iconEl = document.getElementById('label-action-icon');
    const titleEl = document.getElementById('label-action-title');
    if (action) {
      if (iconEl) iconEl.textContent = action.icon;
      if (titleEl) titleEl.textContent = action.label;
    } else {
      if (iconEl) iconEl.textContent = '⚡';
      if (titleEl) titleEl.textContent = 'ACTION';
    }
  }

  /**
   * Updates dynamic noise meter level (0 silent to 5 siren).
   */
  public setNoiseLevel(level: number) {
    this.currentNoiseLevel = Math.max(0, Math.min(5, Math.floor(level)));
    const labels = ['SILENT', 'CREEP', 'CHOP', 'RUN', 'CRACK', 'HONK!'];
    const labelEl = document.getElementById('label-noise-val');
    if (labelEl) labelEl.textContent = labels[this.currentNoiseLevel];

    const bars = document.querySelectorAll('.noise-meter-bars .noise-bar');
    bars.forEach((bar, idx) => {
      bar.classList.toggle('active', idx < this.currentNoiseLevel);
    });
  }

  /**
   * Updates infection percentage meter (0% to 100%).
   */
  public setInfectionPercent(percent: number) {
    const fill = document.getElementById('fill-infection');
    const label = document.getElementById('label-infection-val');
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    if (fill) fill.style.width = `${clamped}%`;
    if (label) label.textContent = `${clamped}%`;
    this.infectionChip?.classList.toggle('danger', clamped >= 75);
  }

  /**
   * Updates weight / encumbrance load (e.g. 15/40 kg).
   */
  public setWeight(currentKg: number, maxKg: number = 40) {
    const fill = document.getElementById('fill-weight');
    const label = document.getElementById('label-weight-val');
    const ratio = Math.max(0, Math.min(1, currentKg / maxKg));
    if (fill) fill.style.width = `${Math.round(ratio * 100)}%`;
    if (label) label.textContent = `${Math.round(currentKg)}/${maxKg} kg`;
    this.weightChip?.classList.toggle('warning', ratio >= 0.85);
  }

  /**
   * Renders item icons in the 5-slot mobile hotbar from InventorySystem.
   */
  public refreshHotbarIcons(inventory: InventorySystem = this.inventory) {
    const slots = inventory.slots.slice(0, 5);
    slots.forEach((slot, idx) => {
      if (idx < this.hotbarSlots.length) {
        const slotEl = this.hotbarSlots[idx];
        const iconEl = slotEl.querySelector('.slot-icon');
        const qtyEl = slotEl.querySelector('.slot-qty');
        if (slot && slot.itemId) {
          const itemDef = ITEM_REGISTRY[slot.itemId];
          if (iconEl) iconEl.textContent = itemDef?.icon || '📦';
          if (qtyEl) qtyEl.textContent = slot.count > 1 ? `${slot.count}` : '';
        } else {
          if (iconEl) iconEl.textContent = '';
          if (qtyEl) qtyEl.textContent = '';
        }
      }
    });
  }
}
