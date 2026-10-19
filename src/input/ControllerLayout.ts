/**
 * ControllerLayout - Virtual Gamepad & Gamepad HUD Overlay.
 * Implements full controller layout with twin sticks, D-pad, face buttons, and shoulder triggers.
 * Automatically polls physical gamepads via the Web Gamepad API and provides touch targets.
 */
export class ControllerLayout {
  private container: HTMLElement | null = null;
  public isVisible: boolean = false;
  private animFrameId: number | null = null;
  private onActionCallback?: (action: string, value: number) => void;

  constructor() {
    this.createDOM();
    this.startPolling();
  }

  private createDOM() {
    if (typeof document === 'undefined') return;

    let el = document.getElementById('controller-layout-overlay');
    if (!el) {
      el = document.createElement('div');
      el.id = 'controller-layout-overlay';
      el.className = 'controller-layout-overlay hidden';
      document.body.appendChild(el);
    }
    this.container = el;

    el.innerHTML = `
      <div class="controller-chassis">
        <!-- Top Bumpers & Triggers -->
        <div class="controller-shoulder-row">
          <div class="shoulder-group left">
            <div id="btn-ctrl-l2" class="trigger-btn">LT / L2</div>
            <div id="btn-ctrl-l1" class="bumper-btn">LB / L1</div>
          </div>
          <div class="center-buttons">
            <div id="btn-ctrl-select" class="meta-btn">VIEW</div>
            <div id="btn-ctrl-guide" class="guide-gem">BRO</div>
            <div id="btn-ctrl-start" class="meta-btn">MENU</div>
          </div>
          <div class="shoulder-group right">
            <div id="btn-ctrl-r1" class="bumper-btn">RB / R1</div>
            <div id="btn-ctrl-r2" class="trigger-btn">RT / R2</div>
          </div>
        </div>

        <!-- Main Body: Sticks, D-Pad, Face Buttons -->
        <div class="controller-body-row">
          <!-- Left Wing: Left Stick & D-Pad -->
          <div class="controller-wing left-wing">
            <div id="ctrl-left-stick" class="analog-stick-boundary">
              <div id="ctrl-left-thumb" class="stick-thumb"></div>
              <span class="stick-label">L-STICK (MOVE)</span>
            </div>
            
            <div class="dpad-cluster">
              <div id="btn-ctrl-du" class="dpad-btn dpad-up">▲</div>
              <div class="dpad-mid-row">
                <div id="btn-ctrl-dl" class="dpad-btn dpad-left">◀</div>
                <div class="dpad-center"></div>
                <div id="btn-ctrl-dr" class="dpad-btn dpad-right">▶</div>
              </div>
              <div id="btn-ctrl-dd" class="dpad-btn dpad-down">▼</div>
            </div>
          </div>

          <!-- Right Wing: Face Buttons & Right Stick -->
          <div class="controller-wing right-wing">
            <div class="face-buttons-cluster">
              <div id="btn-ctrl-y" class="face-btn btn-y" data-label="Y">Y</div>
              <div class="face-mid-row">
                <div id="btn-ctrl-x" class="face-btn btn-x" data-label="X">X</div>
                <div id="btn-ctrl-b" class="face-btn btn-b" data-label="B">B</div>
              </div>
              <div id="btn-ctrl-a" class="face-btn btn-a" data-label="A">A</div>
            </div>

            <div id="ctrl-right-stick" class="analog-stick-boundary">
              <div id="ctrl-right-thumb" class="stick-thumb"></div>
              <span class="stick-label">R-STICK (AIM)</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  public show() {
    this.isVisible = true;
    if (this.container) {
      this.container.classList.remove('hidden');
    }
  }

  public hide() {
    this.isVisible = false;
    if (this.container) {
      this.container.classList.add('hidden');
    }
  }

  public toggle(): boolean {
    if (this.isVisible) this.hide();
    else this.show();
    return this.isVisible;
  }

  /**
   * Polls connected gamepads and highlights buttons visually.
   */
  private startPolling() {
    if (typeof window === 'undefined') return;

    const poll = () => {
      const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
      const gp = gamepads[0] || gamepads[1];

      if (gp && this.isVisible) {
        this.updateFromGamepad(gp);
      }

      this.animFrameId = requestAnimationFrame(poll);
    };

    this.animFrameId = requestAnimationFrame(poll);
  }

  private updateFromGamepad(gp: Gamepad) {
    const updateBtn = (id: string, pressed: boolean) => {
      const el = document.getElementById(id);
      if (el) {
        if (pressed) el.classList.add('active');
        else el.classList.remove('active');
      }
    };

    // Standard Gamepad Mapping
    // 0: A, 1: B, 2: X, 3: Y
    updateBtn('btn-ctrl-a', gp.buttons[0]?.pressed);
    updateBtn('btn-ctrl-b', gp.buttons[1]?.pressed);
    updateBtn('btn-ctrl-x', gp.buttons[2]?.pressed);
    updateBtn('btn-ctrl-y', gp.buttons[3]?.pressed);

    // Bumpers & Triggers
    updateBtn('btn-ctrl-l1', gp.buttons[4]?.pressed);
    updateBtn('btn-ctrl-r1', gp.buttons[5]?.pressed);
    updateBtn('btn-ctrl-l2', gp.buttons[6]?.pressed || (gp.buttons[6]?.value > 0.1));
    updateBtn('btn-ctrl-r2', gp.buttons[7]?.pressed || (gp.buttons[7]?.value > 0.1));

    // Meta buttons
    updateBtn('btn-ctrl-select', gp.buttons[8]?.pressed);
    updateBtn('btn-ctrl-start', gp.buttons[9]?.pressed);

    // D-Pad
    updateBtn('btn-ctrl-du', gp.buttons[12]?.pressed);
    updateBtn('btn-ctrl-dd', gp.buttons[13]?.pressed);
    updateBtn('btn-ctrl-dl', gp.buttons[14]?.pressed);
    updateBtn('btn-ctrl-dr', gp.buttons[15]?.pressed);

    // Analog Sticks (Deadzone 0.15)
    const deadzone = 0.15;
    const updateStick = (thumbId: string, axisX: number, axisY: number) => {
      const thumb = document.getElementById(thumbId);
      if (thumb) {
        const x = Math.abs(axisX) > deadzone ? axisX : 0;
        const y = Math.abs(axisY) > deadzone ? axisY : 0;
        const maxOffset = 22; // px
        thumb.style.transform = `translate(${x * maxOffset}px, ${y * maxOffset}px)`;
      }
    };

    updateStick('ctrl-left-thumb', gp.axes[0] || 0, gp.axes[1] || 0);
    updateStick('ctrl-right-thumb', gp.axes[2] || 0, gp.axes[3] || 0);
  }

  public dispose() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
    }
  }
}
