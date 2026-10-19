/** Keyboard, mouse and touch input collected into simple state. Game code never reads raw events. */
export class Input {
  private down = new Set<string>();
  private edge = new Set<string>();
  mouse = { x: 0, y: 0, left: false, leftPressed: false, right: false, rightPressed: false, wheel: 0 };
  stick = { x: 0, y: 0 };
  touchAction = false; touchUse = false;

  constructor(private target: HTMLElement) {
    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement | null)?.closest('input, select, textarea')) return;
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab', 'ControlLeft', 'ControlRight'].includes(e.code)) e.preventDefault();
      if (!this.down.has(e.code)) this.edge.add(e.code);
      this.down.add(e.code);
    });
    window.addEventListener('keyup', (e) => this.down.delete(e.code));
    window.addEventListener('blur', () => { this.down.clear(); this.stick.x = this.stick.y = 0; });
    target.addEventListener('pointermove', (e) => { const r = target.getBoundingClientRect(); this.mouse.x = e.clientX - r.left; this.mouse.y = e.clientY - r.top; });
    target.addEventListener('pointerdown', (e) => {
      const r = target.getBoundingClientRect(); this.mouse.x = e.clientX - r.left; this.mouse.y = e.clientY - r.top;
      if (e.button === 2) { this.mouse.right = true; this.mouse.rightPressed = true; } else { this.mouse.left = true; this.mouse.leftPressed = true; }
    });
    window.addEventListener('pointerup', (e) => {
      if (e.button === 2) this.mouse.right = false; else this.mouse.left = false;
    });
    target.addEventListener('contextmenu', (e) => e.preventDefault());
    target.addEventListener('wheel', (e) => { e.preventDefault(); this.mouse.wheel += e.deltaY; }, { passive: false });
  }
  held(code: string): boolean { return this.down.has(code); }
  pressed(code: string): boolean { return this.edge.has(code); }
  axis(): { x: number; y: number } {
    let x = (this.held('KeyD') || this.held('ArrowRight') ? 1 : 0) - (this.held('KeyA') || this.held('ArrowLeft') ? 1 : 0) + this.stick.x;
    let y = (this.held('KeyS') || this.held('ArrowDown') ? 1 : 0) - (this.held('KeyW') || this.held('ArrowUp') ? 1 : 0) + this.stick.y;
    const m = Math.hypot(x, y); if (m > 1) { x /= m; y /= m; }
    return { x, y };
  }
  endFrame(): void { this.edge.clear(); this.mouse.leftPressed = false; this.mouse.rightPressed = false; this.mouse.wheel = 0; this.touchAction = false; this.touchUse = false; }
}
