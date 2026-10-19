/**
 * Mobile Touch & Virtual Joystick Input System
 * Provides tactile on-screen joystick, hold-to-chop action button,
 * camera orbit triggers, and mobile drawer management.
 */

export interface TouchControlsCallbacks {
  onMove: (vector: { x: number; y: number }) => void;
  onActionStart: () => void;
  onActionEnd: () => void;
  onRotateCamera: (direction: -1 | 1) => void;
  onEatFood: () => void;
  onChat: () => void;
  onDance: () => void;
  onHonk: () => void;
  onToggleCrafting: () => void;
}

export class TouchControls {
  private callbacks: TouchControlsCallbacks;
  public isTouchDevice: boolean = false;
  public moveVector = { x: 0, y: 0 };

  // Joystick DOM elements
  private joystickZone: HTMLElement | null = null;
  private joystickBase: HTMLElement | null = null;
  private joystickKnob: HTMLElement | null = null;

  // Touch tracking for joystick
  private joystickTouchId: number | null = null;
  private joystickCenter = { x: 0, y: 0 };
  private maxRadius: number = 48; // Maximum knob displacement in pixels

  // Action button hold-to-repeat
  private actionRepeatInterval: number | null = null;

  // Swipe camera tracking (two-finger or edge swipe)
  private twoFingerStartX: number | null = null;

  constructor(callbacks: TouchControlsCallbacks) {
    this.callbacks = callbacks;
    this.detectTouchCapability();
    this.initElements();
    this.bindEvents();
  }

  private detectTouchCapability() {
    this.isTouchDevice =
      'ontouchstart' in window ||
      navigator.maxTouchPoints > 0 ||
      window.matchMedia('(pointer: coarse)').matches;

    if (this.isTouchDevice) {
      document.body.classList.add('touch-device');
    }
  }

  public setTouchHudVisible(visible: boolean) {
    if (visible) {
      document.body.classList.add('force-touch-hud');
    } else {
      document.body.classList.remove('force-touch-hud');
    }
  }

  private initElements() {
    this.joystickZone = document.getElementById('touch-joystick-zone');
    this.joystickBase = document.getElementById('touch-joystick-base');
    this.joystickKnob = document.getElementById('touch-joystick-knob');
  }

  private triggerHaptic(duration: number = 15) {
    try {
      if ('vibrate' in navigator) {
        navigator.vibrate(duration);
      }
    } catch {
      // Haptics not allowed or supported
    }
  }

  private bindEvents() {
    // 1. Virtual Joystick Touch Zone
    if (this.joystickZone && this.joystickBase && this.joystickKnob) {
      this.joystickZone.addEventListener(
        'touchstart',
        (e: TouchEvent) => {
          e.preventDefault();
          if (this.joystickTouchId !== null) return; // Already tracking a finger

          const touch = e.changedTouches[0];
          this.joystickTouchId = touch.identifier;

          const rect = this.joystickZone!.getBoundingClientRect();
          // Anchor joystick base directly under user's thumb
          const startX = touch.clientX;
          const startY = touch.clientY;

          this.joystickCenter = { x: startX, y: startY };

          this.joystickBase!.style.display = 'block';
          this.joystickBase!.style.left = `${startX - rect.left}px`;
          this.joystickBase!.style.top = `${startY - rect.top}px`;
          this.joystickKnob!.style.transform = `translate(0px, 0px)`;

          this.triggerHaptic(10);
        },
        { passive: false }
      );

      window.addEventListener(
        'touchmove',
        (e: TouchEvent) => {
          if (this.joystickTouchId === null) return;

          for (let i = 0; i < e.changedTouches.length; i++) {
            const touch = e.changedTouches[i];
            if (touch.identifier === this.joystickTouchId) {
              e.preventDefault();
              const dx = touch.clientX - this.joystickCenter.x;
              const dy = touch.clientY - this.joystickCenter.y;
              const distance = Math.hypot(dx, dy);

              if (distance === 0) {
                this.moveVector = { x: 0, y: 0 };
                this.callbacks.onMove(this.moveVector);
                this.joystickKnob!.style.transform = `translate(0px, 0px)`;
                return;
              }

              // Clamp displacement to maxRadius
              const clampedDist = Math.min(distance, this.maxRadius);
              const angle = Math.atan2(dy, dx);
              const knobX = Math.cos(angle) * clampedDist;
              const knobY = Math.sin(angle) * clampedDist;

              this.joystickKnob!.style.transform = `translate(${knobX}px, ${knobY}px)`;

              // Normalized movement vector (-1 to 1) with slight deadzone
              const deadzone = 8;
              if (distance < deadzone) {
                this.moveVector = { x: 0, y: 0 };
              } else {
                const intensity = (clampedDist - deadzone) / (this.maxRadius - deadzone);
                this.moveVector = {
                  x: Math.cos(angle) * intensity,
                  y: Math.sin(angle) * intensity,
                };
              }

              this.callbacks.onMove(this.moveVector);
              break;
            }
          }
        },
        { passive: false }
      );

      const releaseJoystick = (e: TouchEvent) => {
        if (this.joystickTouchId === null) return;
        for (let i = 0; i < e.changedTouches.length; i++) {
          if (e.changedTouches[i].identifier === this.joystickTouchId) {
            this.joystickTouchId = null;
            this.moveVector = { x: 0, y: 0 };
            this.callbacks.onMove(this.moveVector);

            if (this.joystickBase && this.joystickKnob) {
              this.joystickKnob.style.transform = `translate(0px, 0px)`;
              this.joystickBase.style.display = 'none';
            }
            break;
          }
        }
      };

      window.addEventListener('touchend', releaseJoystick);
      window.addEventListener('touchcancel', releaseJoystick);
    }

    // 2. Primary Action Button (Hold-to-Chop & Smack)
    const actionBtn = document.getElementById('btn-touch-action');
    if (actionBtn) {
      const startAction = (e: Event) => {
        e.preventDefault();
        this.triggerHaptic(20);
        this.callbacks.onActionStart();

        if (this.actionRepeatInterval !== null) {
          clearInterval(this.actionRepeatInterval);
        }
        // Continuous chopping every 320ms while held down
        this.actionRepeatInterval = window.setInterval(() => {
          this.triggerHaptic(15);
          this.callbacks.onActionStart();
        }, 320);
      };

      const stopAction = (e: Event) => {
        e.preventDefault();
        if (this.actionRepeatInterval !== null) {
          clearInterval(this.actionRepeatInterval);
          this.actionRepeatInterval = null;
        }
        this.callbacks.onActionEnd();
      };

      actionBtn.addEventListener('touchstart', startAction, { passive: false });
      actionBtn.addEventListener('touchend', stopAction);
      actionBtn.addEventListener('touchcancel', stopAction);
      actionBtn.addEventListener('mousedown', startAction);
      actionBtn.addEventListener('mouseup', stopAction);
      actionBtn.addEventListener('mouseleave', stopAction);
    }

    // 3. Camera Orbit Buttons
    const camLeftBtn = document.getElementById('btn-touch-cam-left');
    const camRightBtn = document.getElementById('btn-touch-cam-right');

    camLeftBtn?.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.triggerHaptic(15);
      this.callbacks.onRotateCamera(1); // Rotate clockwise (+45 deg)
    });
    camLeftBtn?.addEventListener('click', () => {
      this.callbacks.onRotateCamera(1);
    });

    camRightBtn?.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.triggerHaptic(15);
      this.callbacks.onRotateCamera(-1); // Rotate counter-clockwise (-45 deg)
    });
    camRightBtn?.addEventListener('click', () => {
      this.callbacks.onRotateCamera(-1);
    });

    // 4. Quick Eat Button
    const eatBtn = document.getElementById('btn-touch-eat');
    eatBtn?.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.triggerHaptic(15);
      this.callbacks.onEatFood();
    });
    eatBtn?.addEventListener('click', () => {
      this.callbacks.onEatFood();
    });

    // 5. Quick Chat Button
    const chatBtn = document.getElementById('btn-touch-chat');
    chatBtn?.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.triggerHaptic(15);
      this.callbacks.onChat();
    });
    chatBtn?.addEventListener('click', () => {
      this.callbacks.onChat();
    });

    // 6. Mobile Crafting Drawer Toggle
    const craftToggleBtn = document.getElementById('btn-touch-crafting-toggle');
    craftToggleBtn?.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.triggerHaptic(15);
      this.callbacks.onToggleCrafting();
    });
    craftToggleBtn?.addEventListener('click', () => {
      this.callbacks.onToggleCrafting();
    });

    // 7. Two-finger horizontal swipe for smooth camera orbit
    window.addEventListener(
      'touchstart',
      (e: TouchEvent) => {
        if (e.touches.length === 2) {
          this.twoFingerStartX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        }
      },
      { passive: true }
    );

    window.addEventListener(
      'touchmove',
      (e: TouchEvent) => {
        if (e.touches.length === 2 && this.twoFingerStartX !== null) {
          const currentX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
          const diff = currentX - this.twoFingerStartX;
          if (Math.abs(diff) > 50) {
            this.callbacks.onRotateCamera(diff > 0 ? 1 : -1);
            this.triggerHaptic(10);
            this.twoFingerStartX = currentX;
          }
        }
      },
      { passive: true }
    );

    window.addEventListener('touchend', (e: TouchEvent) => {
      if (e.touches.length < 2) {
        this.twoFingerStartX = null;
      }
    });
  }
}
