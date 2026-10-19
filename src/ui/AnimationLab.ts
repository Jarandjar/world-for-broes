import { AnimationController } from '../animation/AnimationController';
import { createHumanoidClips } from '../animation/HumanoidClips';
import { AnimationClip } from '../animation/AnimationClip';
import { LinearBlend1D } from '../animation/BlendTree';

/**
 * Animation Lab Debug Scene & Performance Inspector (Section 68).
 * Opens with F2 key or via the Dev Spawner Panel tab.
 * Allows:
 * - Clip scrubbing and playback speed controls (0.1x to 2.0x).
 * - Blend tree slider testing (e.g. Walk -> Run).
 * - Named animation event visual timeline (hit_frame, footstep).
 * - LOD & Performance microsecond telemetry (< 2ms budget).
 */
export class AnimationLab {
  private container: HTMLElement | null = null;
  private isOpen: boolean = false;
  private controller: AnimationController;
  private allClips: Record<string, AnimationClip>;

  // Lab Playback State
  public previewClip: AnimationClip;
  public isPlaying: boolean = true;
  public playbackSpeed: number = 1.0;
  public scrubberTime: number = 0;
  public blendNode: LinearBlend1D;

  // Active lab mode: 'clip' | 'blend'
  public labMode: 'clip' | 'blend' = 'clip';

  constructor(controller: AnimationController) {
    this.controller = controller;
    this.allClips = createHumanoidClips();
    this.previewClip = this.allClips['idle'] || Object.values(this.allClips)[0];
    this.blendNode = new LinearBlend1D(
      this.allClips['walk'] || this.previewClip,
      this.allClips['run'] || this.previewClip
    );

    this.createDOM();
    this.bindKeyboardShortcut();
  }

  private createDOM() {
    const existing = document.getElementById('animation-lab-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'animation-lab-modal';
    modal.className = 'anim-lab-modal hidden';
    modal.innerHTML = `
      <div class="anim-lab-window">
        <div class="anim-lab-header">
          <div class="anim-lab-title">
            <span class="lab-badge">F2</span>
            <span>🎬 ANIMATION LAB & PROFILER</span>
          </div>
          <button id="btn-close-anim-lab" class="lab-close-btn" title="Close [F2 / ESC]">✕</button>
        </div>

        <div class="anim-lab-body">
          <!-- Left Controls Column -->
          <div class="anim-lab-col">
            <!-- Mode Switcher -->
            <div class="lab-tab-row">
              <button id="tab-mode-clip" class="lab-tab-btn active">SINGLE CLIP</button>
              <button id="tab-mode-blend" class="lab-tab-btn">BLEND TREE (1D)</button>
              <button id="tab-mode-springs" class="lab-tab-btn">SPRINGS & POSE</button>
            </div>

            <!-- Clip Selection Section -->
            <div id="section-clip-select" class="lab-section">
              <label class="lab-label">AUTHORED CLIP (16 CORE ROSTER):</label>
              <select id="select-anim-clip" class="lab-select">
                ${Object.keys(this.allClips)
                  .map(name => `<option value="${name}">${name.toUpperCase()} (${this.allClips[name].duration}s)</option>`)
                  .join('')}
              </select>
            </div>

            <!-- Blend Tree Section (Hidden by default) -->
            <div id="section-blend-select" class="lab-section hidden">
              <label class="lab-label">BLEND NODE A & B:</label>
              <div class="lab-flex-row">
                <select id="select-blend-a" class="lab-select">
                  ${Object.keys(this.allClips).map(name => `<option value="${name}" ${name === 'walk' ? 'selected' : ''}>A: ${name}</option>`).join('')}
                </select>
                <select id="select-blend-b" class="lab-select">
                  ${Object.keys(this.allClips).map(name => `<option value="${name}" ${name === 'run' ? 'selected' : ''}>B: ${name}</option>`).join('')}
                </select>
              </div>
              <label class="lab-label" style="margin-top: 8px;">BLEND WEIGHT ALPHA (0.0 A -> 1.0 B):</label>
              <input id="slider-blend-weight" type="range" min="0" max="1" step="0.01" value="0.5" class="lab-slider" />
              <div class="lab-slider-val"><span id="val-blend-weight">0.50</span></div>
            </div>

            <!-- Phase Z0D: Springs & Pose Library Section -->
            <div id="section-springs-select" class="lab-section hidden">
              <label class="lab-label">POSE LIBRARY (KEY FRAMES):</label>
              <div class="lab-btn-grid">
                <button class="lab-chip-btn" data-pose="idle">IDLE</button>
                <button class="lab-chip-btn" data-pose="run">RUN</button>
                <button class="lab-chip-btn" data-pose="windup">WINDUP</button>
                <button class="lab-chip-btn" data-pose="smear">SMEAR</button>
                <button class="lab-chip-btn" data-pose="impact">IMPACT</button>
                <button class="lab-chip-btn" data-pose="recover">RECOVER</button>
              </div>

              <label class="lab-label" style="margin-top: 10px;">SPRING FREQUENCY f (1 - 15 Hz):</label>
              <input id="slider-spring-freq" type="range" min="1" max="15" step="0.5" value="5" class="lab-slider" />
              <div class="lab-slider-val"><span id="val-spring-freq">5.0 Hz</span></div>

              <label class="lab-label" style="margin-top: 8px;">DAMPING RATIO ζ (0.1 - 1.5):</label>
              <input id="slider-spring-damp" type="range" min="0.1" max="1.5" step="0.05" value="0.7" class="lab-slider" />
              <div class="lab-slider-val"><span id="val-spring-damp">0.70</span></div>

              <label class="lab-label" style="margin-top: 8px;">CROSSFADE BLEND TIME (ms):</label>
              <input id="slider-crossfade-time" type="range" min="50" max="400" step="10" value="120" class="lab-slider" />
              <div class="lab-slider-val"><span id="val-crossfade-time">120 ms</span></div>
            </div>

            <!-- Playback Controls -->
            <div class="lab-section">
              <div class="lab-btn-row">
                <button id="btn-anim-play" class="lab-action-btn">⏸️ PAUSE</button>
                <button id="btn-anim-restart" class="lab-action-btn">🔄 RESTART</button>
                <button id="btn-anim-loop" class="lab-action-btn active">🔁 LOOP: ON</button>
              </div>

              <div class="lab-btn-row" style="margin-top: 6px;">
                <button id="btn-onion-skin" class="lab-action-btn">🧅 ONION SKIN: OFF</button>
                <button id="btn-hot-reload" class="lab-action-btn">⚡ HOT RELOAD</button>
              </div>

              <label class="lab-label" style="margin-top: 10px;">PLAYBACK SPEED (0.1x to 2.0x):</label>
              <div class="lab-speed-row">
                <button class="lab-speed-chip" data-speed="0.25">0.25x</button>
                <button class="lab-speed-chip" data-speed="0.5">0.5x</button>
                <button class="lab-speed-chip active" data-speed="1.0">1.0x</button>
                <button class="lab-speed-chip" data-speed="1.5">1.5x</button>
                <button class="lab-speed-chip" data-speed="2.0">2.0x</button>
              </div>

              <!-- Scrubber -->
              <label class="lab-label" style="margin-top: 10px;">SCRUBBER TIMELINE:</label>
              <input id="slider-anim-scrub" type="range" min="0" max="1" step="0.005" value="0" class="lab-slider" />
              <div class="lab-slider-val"><span id="val-scrub-time">0.00s / 1.60s</span></div>
            </div>
          </div>

          <!-- Right Telemetry & Event Column -->
          <div class="anim-lab-col">
            <!-- Phase Z0D: Per-Entity-Type Profile Bar -->
            <div class="lab-section telemetry-box">
              <div class="telemetry-header">📊 PER-ENTITY PROFILE BAR</div>
              <div class="entity-profile-grid">
                <div class="entity-prof-card active" data-entity="player">
                  <span class="prof-name">SURVIVOR</span>
                  <span class="prof-stat">Tier 2 IK</span>
                  <span class="prof-cost">120 μs</span>
                </div>
                <div class="entity-prof-card" data-entity="zombie">
                  <span class="prof-name">SHAMBLER</span>
                  <span class="prof-stat">Tier 1 Crowd</span>
                  <span class="prof-cost">18 μs/ea</span>
                </div>
                <div class="entity-prof-card" data-entity="pet">
                  <span class="prof-name">SCRAPPY PUP</span>
                  <span class="prof-stat">Tier 2 Ears</span>
                  <span class="prof-cost">85 μs</span>
                </div>
                <div class="entity-prof-card" data-entity="car">
                  <span class="prof-name">HATCHBACK</span>
                  <span class="prof-stat">Suspension</span>
                  <span class="prof-cost">42 μs</span>
                </div>
              </div>
            </div>

            <!-- Performance Telemetry Box -->
            <div class="lab-section telemetry-box">
              <div class="telemetry-header">⚡ CPU ANIMATION BUDGET TELEMETRY</div>
              <div class="telemetry-grid">
                <div class="telem-item">
                  <span class="telem-title">UPDATE COST</span>
                  <span id="telem-cost" class="telem-value good">-- μs</span>
                </div>
                <div class="telem-item">
                  <span class="telem-title">BUDGET LIMIT</span>
                  <span class="telem-value">2000 μs (2ms)</span>
                </div>
                <div class="telem-item">
                  <span class="telem-title">ACTIVE BONES</span>
                  <span id="telem-bones" class="telem-value">8 nodes</span>
                </div>
                <div class="telem-item">
                  <span class="telem-title">TARGET FPS</span>
                  <span class="telem-value good">60.0 fps</span>
                </div>
              </div>
            </div>

            <!-- Event Inspector Timeline -->
            <div class="lab-section">
              <label class="lab-label">NAMED ANIMATION EVENTS IN CLIP:</label>
              <div id="anim-event-timeline" class="event-timeline-container">
                <div class="event-empty">No events registered for this clip.</div>
              </div>
            </div>

            <!-- Actions Row -->
            <div class="lab-section">
              <label class="lab-label">DEMONSTRATION & INJECTION:</label>
              <div class="lab-btn-row">
                <button id="btn-apply-to-player" class="lab-primary-btn">⚡ FORCE PLAY</button>
                <button id="btn-toggle-slice-demo" class="lab-primary-btn" style="background: linear-gradient(135deg, #16a085, #0e6655);">🎬 DUSK SLICE</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(modal);
    this.container = modal;

    this.bindEvents();
    this.renderEventsTimeline();
  }

  public onToggleVerticalSlice?: () => void;
  public onionSkinEnabled: boolean = false;
  public springFreq: number = 5.0;
  public springDamp: number = 0.7;
  public crossfadeDuration: number = 120;

  private bindKeyboardShortcut() {
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.code === 'F2') {
        e.preventDefault();
        this.toggle();
      } else if (e.code === 'Escape' && this.isOpen) {
        this.close();
      }
    });
  }

  private bindEvents() {
    document.getElementById('btn-close-anim-lab')?.addEventListener('click', () => this.close());

    // Tab switching (Single Clip, Blend Tree, Springs & Pose)
    const tabClip = document.getElementById('tab-mode-clip');
    const tabBlend = document.getElementById('tab-mode-blend');
    const tabSprings = document.getElementById('tab-mode-springs');
    const secClip = document.getElementById('section-clip-select');
    const secBlend = document.getElementById('section-blend-select');
    const secSprings = document.getElementById('section-springs-select');

    tabClip?.addEventListener('click', () => {
      this.labMode = 'clip';
      tabClip.classList.add('active');
      tabBlend?.classList.remove('active');
      tabSprings?.classList.remove('active');
      secClip?.classList.remove('hidden');
      secBlend?.classList.add('hidden');
      secSprings?.classList.add('hidden');
    });

    tabBlend?.addEventListener('click', () => {
      this.labMode = 'blend';
      tabBlend.classList.add('active');
      tabClip?.classList.remove('active');
      tabSprings?.classList.remove('active');
      secBlend?.classList.remove('hidden');
      secClip?.classList.add('hidden');
      secSprings?.classList.add('hidden');
    });

    tabSprings?.addEventListener('click', () => {
      tabSprings.classList.add('active');
      tabClip?.classList.remove('active');
      tabBlend?.classList.remove('active');
      secSprings?.classList.remove('hidden');
      secClip?.classList.add('hidden');
      secBlend?.classList.add('hidden');
    });

    // Pose Library Buttons
    document.querySelectorAll('.lab-chip-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const pose = btn.getAttribute('data-pose');
        if (pose === 'idle') this.previewClip = this.allClips['idle'] || this.previewClip;
        else if (pose === 'run') this.previewClip = this.allClips['run'] || this.previewClip;
        else if (pose === 'windup' || pose === 'smear' || pose === 'impact' || pose === 'recover') {
          this.previewClip = this.allClips['chop'] || this.allClips['light_swing'] || this.previewClip;
          if (pose === 'windup') this.scrubberTime = 0.05;
          else if (pose === 'smear') this.scrubberTime = 0.18;
          else if (pose === 'impact') this.scrubberTime = 0.25;
          else if (pose === 'recover') this.scrubberTime = 0.45;
          this.isPlaying = false;
        }
        this.renderEventsTimeline();
      });
    });

    // Spring Sliders
    const sFreq = document.getElementById('slider-spring-freq') as HTMLInputElement | null;
    const vFreq = document.getElementById('val-spring-freq');
    sFreq?.addEventListener('input', () => {
      if (sFreq) {
        this.springFreq = parseFloat(sFreq.value);
        if (vFreq) vFreq.textContent = `${this.springFreq.toFixed(1)} Hz`;
      }
    });

    const sDamp = document.getElementById('slider-spring-damp') as HTMLInputElement | null;
    const vDamp = document.getElementById('val-spring-damp');
    sDamp?.addEventListener('input', () => {
      if (sDamp) {
        this.springDamp = parseFloat(sDamp.value);
        if (vDamp) vDamp.textContent = this.springDamp.toFixed(2);
      }
    });

    const sCross = document.getElementById('slider-crossfade-time') as HTMLInputElement | null;
    const vCross = document.getElementById('val-crossfade-time');
    sCross?.addEventListener('input', () => {
      if (sCross) {
        this.crossfadeDuration = parseInt(sCross.value);
        if (vCross) vCross.textContent = `${this.crossfadeDuration} ms`;
      }
    });

    // Onion Skin Toggle
    const btnOnion = document.getElementById('btn-onion-skin');
    btnOnion?.addEventListener('click', () => {
      this.onionSkinEnabled = !this.onionSkinEnabled;
      btnOnion.classList.toggle('active', this.onionSkinEnabled);
      btnOnion.textContent = this.onionSkinEnabled ? '🧅 ONION SKIN: ON' : '🧅 ONION SKIN: OFF';
    });

    // Hot Reload
    document.getElementById('btn-hot-reload')?.addEventListener('click', () => {
      this.allClips = createHumanoidClips();
      this.renderEventsTimeline();
      const btn = document.getElementById('btn-hot-reload');
      if (btn) {
        btn.textContent = '✅ RELOADED!';
        setTimeout(() => { if (btn) btn.textContent = '⚡ HOT RELOAD'; }, 800);
      }
    });

    // Toggle Vertical Slice Scene Demo
    document.getElementById('btn-toggle-slice-demo')?.addEventListener('click', () => {
      if (this.onToggleVerticalSlice) {
        this.onToggleVerticalSlice();
        this.close();
      }
    });

    // Select clip
    const selectClip = document.getElementById('select-anim-clip') as HTMLSelectElement | null;
    selectClip?.addEventListener('change', () => {
      const clipName = selectClip.value;
      if (this.allClips[clipName]) {
        this.previewClip = this.allClips[clipName];
        this.scrubberTime = 0;
        this.renderEventsTimeline();
      }
    });

    // Blend tree selects
    const selBlendA = document.getElementById('select-blend-a') as HTMLSelectElement | null;
    const selBlendB = document.getElementById('select-blend-b') as HTMLSelectElement | null;
    const sliderBlend = document.getElementById('slider-blend-weight') as HTMLInputElement | null;
    const valBlend = document.getElementById('val-blend-weight');

    const updateBlend = () => {
      if (selBlendA && selBlendB && sliderBlend) {
        const cA = this.allClips[selBlendA.value];
        const cB = this.allClips[selBlendB.value];
        if (cA && cB) {
          this.blendNode = new LinearBlend1D(cA, cB);
          this.blendNode.parameter = parseFloat(sliderBlend.value);
          if (valBlend) valBlend.textContent = this.blendNode.parameter.toFixed(2);
        }
      }
    };
    selBlendA?.addEventListener('change', updateBlend);
    selBlendB?.addEventListener('change', updateBlend);
    sliderBlend?.addEventListener('input', updateBlend);

    // Play / Pause
    const btnPlay = document.getElementById('btn-anim-play');
    btnPlay?.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      if (btnPlay) btnPlay.textContent = this.isPlaying ? '⏸️ PAUSE' : '▶️ PLAY';
    });

    // Restart
    document.getElementById('btn-anim-restart')?.addEventListener('click', () => {
      this.scrubberTime = 0;
    });

    // Loop toggle
    const btnLoop = document.getElementById('btn-anim-loop');
    btnLoop?.addEventListener('click', () => {
      this.previewClip.loop = !this.previewClip.loop;
      btnLoop.classList.toggle('active', this.previewClip.loop);
      btnLoop.textContent = this.previewClip.loop ? '🔁 LOOP: ON' : '🔁 LOOP: OFF';
    });

    // Speed chips
    const speedChips = Array.from(document.querySelectorAll('.lab-speed-chip'));
    speedChips.forEach(chip => {
      chip.addEventListener('click', () => {
        speedChips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        this.playbackSpeed = parseFloat(chip.getAttribute('data-speed') || '1.0');
      });
    });

    // Scrubber slider
    const scrubSlider = document.getElementById('slider-anim-scrub') as HTMLInputElement | null;
    scrubSlider?.addEventListener('input', () => {
      if (scrubSlider) {
        const ratio = parseFloat(scrubSlider.value);
        this.scrubberTime = ratio * this.previewClip.duration;
        this.isPlaying = false;
        if (btnPlay) btnPlay.textContent = '▶️ PLAY';
      }
    });

    // Force play on survivor
    document.getElementById('btn-apply-to-player')?.addEventListener('click', () => {
      if (this.previewClip) {
        this.controller.stateMachine.playOverride(this.previewClip, 0.08);
      }
    });
  }

  private renderEventsTimeline() {
    const timelineEl = document.getElementById('anim-event-timeline');
    if (!timelineEl) return;

    if (!this.previewClip || this.previewClip.events.length === 0) {
      timelineEl.innerHTML = `<div class="event-empty">No events registered for "${this.previewClip.name}".</div>`;
      return;
    }

    timelineEl.innerHTML = this.previewClip.events
      .map(ev => {
        const percent = Math.round((ev.time / this.previewClip.duration) * 100);
        return `
          <div class="event-chip-item">
            <span class="event-badge">${ev.name}</span>
            <span class="event-time">${ev.time.toFixed(2)}s (${percent}%)</span>
          </div>
        `;
      })
      .join('');
  }

  public open() {
    this.isOpen = true;
    this.container?.classList.remove('hidden');
  }

  public close() {
    this.isOpen = false;
    this.container?.classList.add('hidden');
  }

  public toggle() {
    if (this.isOpen) this.close();
    else this.open();
  }

  /**
   * Called per frame while open to update telemetry and scrubber.
   */
  public update(dt: number) {
    if (!this.isOpen) return;

    if (this.isPlaying) {
      this.scrubberTime += dt * this.playbackSpeed;
      if (this.previewClip.loop) {
        this.scrubberTime = this.scrubberTime % this.previewClip.duration;
      } else {
        this.scrubberTime = Math.min(this.previewClip.duration, this.scrubberTime);
      }
    }

    // Update scrubber UI
    const scrubSlider = document.getElementById('slider-anim-scrub') as HTMLInputElement | null;
    const valScrub = document.getElementById('val-scrub-time');
    if (scrubSlider && this.previewClip.duration > 0) {
      scrubSlider.value = `${this.scrubberTime / this.previewClip.duration}`;
    }
    if (valScrub) {
      valScrub.textContent = `${this.scrubberTime.toFixed(2)}s / ${this.previewClip.duration.toFixed(2)}s`;
    }

    // Telemetry display
    const telemCost = document.getElementById('telem-cost');
    const telemBones = document.getElementById('telem-bones');
    if (telemCost) {
      const micros = Math.round(this.controller.lastUpdateMicros);
      telemCost.textContent = `${micros} μs (${(micros / 1000).toFixed(3)} ms)`;
      telemCost.className = `telem-value ${micros <= 2000 ? 'good' : 'warning'}`;
    }
    if (telemBones) {
      telemBones.textContent = `${this.controller.activeBoneCount} nodes`;
    }
  }
}
