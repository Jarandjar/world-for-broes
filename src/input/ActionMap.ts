/**
 * Universal Action Map & Input Buffering System for World 4 Bro's (Phase Z0B).
 * Maps physical keyboard `event.code` and touch/gamepad inputs to high-level game actions,
 * completely independent of regional keyboard layouts (QWERTY, AZERTY, Dvorak).
 * Includes a deterministic 120ms action buffer to ensure snappy, responsive combat and movement.
 */

export type GameAction =
  | 'MOVE_FORWARD'
  | 'MOVE_BACK'
  | 'MOVE_LEFT'
  | 'MOVE_RIGHT'
  | 'SPRINT'
  | 'SNEAK'
  | 'ACTION_PRIMARY'    // Space / Tap Action: Attack, Chop, Mine, Harvest
  | 'ACTION_SECONDARY'  // KeyE: Interact, Open Door, Enter Vehicle
  | 'QUICK_USE'         // KeyF: Consume active food / bandage
  | 'THROW'             // KeyT: Aim & throw grenade / peel
  | 'ROTATE_CAM_LEFT'   // KeyQ
  | 'ROTATE_CAM_RIGHT'  // KeyE (alt) / Arrow keys
  | 'HONK'              // KeyH: Air horn / vehicle horn
  | 'EMOTE_DANCE'       // KeyG: Party dance
  | 'EMOTE_HIGHFIVE'    // KeyB
  | 'EMOTE_POINT'       // KeyP
  | 'EMOTE_SIT'         // KeyK
  | 'HOTBAR_1'
  | 'HOTBAR_2'
  | 'HOTBAR_3'
  | 'HOTBAR_4'
  | 'HOTBAR_5';

export interface ActionBinding {
  primary: string;    // Physical KeyboardEvent.code (e.g. 'KeyW', 'Space')
  alternate?: string; // Secondary key (e.g. 'ArrowUp')
}

export interface BufferedAction {
  action: GameAction;
  timestamp: number;
  payload?: any;
}

export class ActionMap {
  private bindings: Map<GameAction, ActionBinding> = new Map();
  private codeToActionMap: Map<string, GameAction[]> = new Map();
  private activeActions: Set<GameAction> = new Set();
  private bufferedActions: BufferedAction[] = [];
  private actionListeners: Map<GameAction, Array<(payload?: any) => void>> = new Map();

  // Analog input values (normalized -1 to 1)
  public analogMove = { x: 0, y: 0 };
  public isSneakToggled: boolean = false;
  public isSprintLocked: boolean = false;

  // Buffer expiry duration in milliseconds
  public readonly BUFFER_WINDOW_MS: number = 120;

  constructor() {
    this.initDefaultBindings();
    this.rebuildCodeLookup();
  }

  private initDefaultBindings() {
    this.bindings.set('MOVE_FORWARD', { primary: 'KeyW', alternate: 'ArrowUp' });
    this.bindings.set('MOVE_BACK', { primary: 'KeyS', alternate: 'ArrowDown' });
    this.bindings.set('MOVE_LEFT', { primary: 'KeyA', alternate: 'ArrowLeft' });
    this.bindings.set('MOVE_RIGHT', { primary: 'KeyD', alternate: 'ArrowRight' });
    this.bindings.set('SPRINT', { primary: 'ShiftLeft', alternate: 'ShiftRight' });
    this.bindings.set('SNEAK', { primary: 'KeyC', alternate: 'ControlLeft' });
    this.bindings.set('ACTION_PRIMARY', { primary: 'Space' });
    this.bindings.set('ACTION_SECONDARY', { primary: 'KeyR', alternate: 'KeyX' });
    this.bindings.set('QUICK_USE', { primary: 'KeyF' });
    this.bindings.set('THROW', { primary: 'KeyT' });
    this.bindings.set('ROTATE_CAM_LEFT', { primary: 'KeyQ' });
    this.bindings.set('ROTATE_CAM_RIGHT', { primary: 'KeyE' });
    this.bindings.set('HONK', { primary: 'KeyH' });
    this.bindings.set('EMOTE_DANCE', { primary: 'KeyG' });
    this.bindings.set('EMOTE_HIGHFIVE', { primary: 'KeyB' });
    this.bindings.set('EMOTE_POINT', { primary: 'KeyP' });
    this.bindings.set('EMOTE_SIT', { primary: 'KeyK' });
    this.bindings.set('HOTBAR_1', { primary: 'Digit1' });
    this.bindings.set('HOTBAR_2', { primary: 'Digit2' });
    this.bindings.set('HOTBAR_3', { primary: 'Digit3' });
    this.bindings.set('HOTBAR_4', { primary: 'Digit4' });
    this.bindings.set('HOTBAR_5', { primary: 'Digit5' });
  }

  private rebuildCodeLookup() {
    this.codeToActionMap.clear();
    for (const [action, binding] of this.bindings.entries()) {
      if (binding.primary) {
        if (!this.codeToActionMap.has(binding.primary)) this.codeToActionMap.set(binding.primary, []);
        this.codeToActionMap.get(binding.primary)!.push(action);
      }
      if (binding.alternate) {
        if (!this.codeToActionMap.has(binding.alternate)) this.codeToActionMap.set(binding.alternate, []);
        this.codeToActionMap.get(binding.alternate)!.push(action);
      }
    }
  }

  /**
   * Attaches global keyboard listeners to the browser window.
   */
  public bindToWindow(win: Window = window) {
    win.addEventListener('keydown', (e: KeyboardEvent) => {
      // Don't intercept when user is typing in chat or dev input
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
        return;
      }

      this.handleKeyDown(e.code);

      // Prevent page scrolling on Space / Arrow keys
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
        e.preventDefault();
      }
    });

    win.addEventListener('keyup', (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
        return;
      }
      this.handleKeyUp(e.code);
    });

    win.addEventListener('blur', () => {
      this.reset();
    });
  }

  /**
   * Process physical key down event by code.
   */
  public handleKeyDown(code: string, timestamp: number = performance.now()) {
    const actions = this.codeToActionMap.get(code);
    if (!actions) return;

    for (const action of actions) {
      if (!this.activeActions.has(action)) {
        this.activeActions.add(action);

        // Toggle actions
        if (action === 'SNEAK') {
          if (code === 'KeyC') {
            this.isSneakToggled = !this.isSneakToggled;
          } else {
            this.isSneakToggled = true;
          }
        }

        // Buffer the action event
        this.bufferAction(action, timestamp);

        // Dispatch immediate listeners
        this.dispatch(action);
      }
    }
  }

  /**
   * Process physical key up event by code.
   */
  public handleKeyUp(code: string) {
    const actions = this.codeToActionMap.get(code);
    if (!actions) return;

    for (const action of actions) {
      this.activeActions.delete(action);
      if (action === 'SNEAK' && code !== 'KeyC') {
        this.isSneakToggled = false;
      }
    }
  }

  /**
   * Places an action into the 120ms buffer.
   */
  public bufferAction(action: GameAction, timestamp: number = performance.now(), payload?: any) {
    this.bufferedActions.push({ action, timestamp, payload });
  }

  /**
   * Checks if an action was buffered within the 120ms window and consumes it if found.
   * Returns true if consumed, false otherwise.
   */
  public consumeBufferedAction(action: GameAction, currentTime: number = performance.now()): boolean {
    this.pruneExpiredBuffer(currentTime);

    for (let i = 0; i < this.bufferedActions.length; i++) {
      if (this.bufferedActions[i].action === action) {
        this.bufferedActions.splice(i, 1);
        return true;
      }
    }
    return false;
  }

  /**
   * Checks whether an action is currently held down or toggled.
   */
  public isActionActive(action: GameAction): boolean {
    if (action === 'SNEAK') {
      return this.isSneakToggled;
    }
    if (action === 'SPRINT') {
      return this.isSprintLocked || this.activeActions.has('SPRINT');
    }
    return this.activeActions.has(action);
  }

  /**
   * Calculates normalized 2D movement vector combining digital keys and analog virtual stick.
   */
  public getMovementVector(): { x: number; y: number } {
    let dx = 0;
    let dy = 0;

    if (this.isActionActive('MOVE_FORWARD')) dy -= 1;
    if (this.isActionActive('MOVE_BACK')) dy += 1;
    if (this.isActionActive('MOVE_LEFT')) dx -= 1;
    if (this.isActionActive('MOVE_RIGHT')) dx += 1;

    // Blend touch analog stick
    dx += this.analogMove.x;
    dy += this.analogMove.y;

    const lenSq = dx * dx + dy * dy;
    if (lenSq > 1) {
      const len = Math.sqrt(lenSq);
      dx /= len;
      dy /= len;
    }

    return { x: dx, y: dy };
  }

  /**
   * Subscribe to named action triggers.
   */
  public on(action: GameAction, listener: (payload?: any) => void) {
    if (!this.actionListeners.has(action)) {
      this.actionListeners.set(action, []);
    }
    this.actionListeners.get(action)!.push(listener);
  }

  /**
   * Dispatches event to registered action listeners.
   */
  private dispatch(action: GameAction, payload?: any) {
    const listeners = this.actionListeners.get(action);
    if (listeners) {
      for (const fn of listeners) {
        fn(payload);
      }
    }
  }

  /**
   * Discards buffered actions older than 120ms.
   */
  public pruneExpiredBuffer(currentTime: number = performance.now()) {
    const cutoff = currentTime - this.BUFFER_WINDOW_MS;
    this.bufferedActions = this.bufferedActions.filter(b => b.timestamp >= cutoff);
  }

  /**
   * Reset all active input states.
   */
  public reset() {
    this.activeActions.clear();
    this.bufferedActions = [];
    this.analogMove.x = 0;
    this.analogMove.y = 0;
    this.isSprintLocked = false;
    this.isSneakToggled = false;
  }
}

export const actionMap = new ActionMap();
