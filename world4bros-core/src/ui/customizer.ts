import type { Game } from '../sim/game';
import type { Renderer } from '../render/renderer';
import { SLOTS, SLOT_LABELS, type ColorSet, type Slot } from '../data/schema';
import { BEARDS, HAIR_COLORS, SKIN_TONES } from '../data/appearance';
import { defaultLook, equip, randomLook, resolvedColors, sanitizeLook } from '../sim/look';
import { makeRng } from '../engine/rng';
import { STYLE } from '../render/ink';
import { clearIconCache } from '../render/icons';
import { h } from './dom';

type Tab = 'look' | 'outfit' | 'share';

/** Character customizer: appearance, hair color, every clothing and equipment slot, and per-item colors. */
export class Customizer {
  readonly el = h('div', { class: 'panel custom' });
  private preview = h('canvas', { class: 'preview', width: 320, height: 210 });
  private tab: Tab = 'look';
  private mode: 'idle' | 'walk' | 'swing' = 'walk';
  private t = 0;
  private seed = 1;
  constructor(private game: Game, private renderer: Renderer) { this.build(); }
  private get look() { return this.game.playerChar.look; }
  private get content() { return this.game.content; }
  private changed(): void { /* sprites are keyed by color, so nothing else to invalidate */ }

  build(): void {
    const tabs = (['look', 'outfit', 'share'] as Tab[]).map((t) => h('button', { class: `tab${t === this.tab ? ' on' : ''}`, onclick: () => { this.tab = t; this.build(); } }, { look: 'Look', outfit: 'Outfit and gear', share: 'Share' }[t]));
    const modes = h('select', { 'aria-label': 'Preview animation', onchange: (e) => { this.mode = (e.target as HTMLSelectElement).value as typeof this.mode; } },
      ...(['idle', 'walk', 'swing'] as const).map((m) => h('option', { value: m, selected: m === this.mode }, m[0]?.toUpperCase() + m.slice(1))));
    const body = this.tab === 'look' ? this.lookTab() : this.tab === 'outfit' ? this.outfitTab() : this.shareTab();
    this.el.replaceChildren(
      h('h2', {}, 'Character'),
      h('div', { class: 'previewbox' }, this.preview, h('div', { class: 'row' }, h('span', {}, 'Preview'), modes)),
      h('div', { class: 'tabs' }, ...tabs),
      h('div', { class: 'tabbody' }, body),
    );
  }

  private swatches(colors: string[], current: string, onPick: (c: string) => void): HTMLElement {
    return h('div', { class: 'swatches' }, ...colors.map((c) => h('button', { class: `sw${c.toLowerCase() === current.toLowerCase() ? ' on' : ''}`, style: `background:${c}`, 'aria-label': `Color ${c}`, onclick: () => { onPick(c); this.build(); } })));
  }

  private lookTab(): HTMLElement {
    const a = this.look.appearance;
    const sel = (label: string, opts: { id: string; name: string }[], cur: string, on: (v: string) => void): HTMLElement =>
      h('label', { class: 'field' }, h('span', {}, label), h('select', { onchange: (e) => { on((e.target as HTMLSelectElement).value); this.build(); } }, ...opts.map((o) => h('option', { value: o.id, selected: o.id === cur }, o.name))));
    return h('div', {},
      h('div', { class: 'field' }, h('span', {}, 'Skin'), this.swatches(SKIN_TONES, a.skin, (c) => { a.skin = c; })),
      sel('Hair style', this.content.hairStyles, a.hairStyle, (v) => { a.hairStyle = v; }),
      h('div', { class: 'field' }, h('span', {}, 'Hair color'), this.swatches(HAIR_COLORS, a.hairColor, (c) => { a.hairColor = c; }),
        h('label', { class: 'pick' }, 'Any color ', h('input', { type: 'color', value: a.hairColor, oninput: (e) => { a.hairColor = (e.target as HTMLInputElement).value; } }))),
      sel('Facial hair', BEARDS, a.beard, (v) => { a.beard = v; }),
      h('label', { class: 'field' }, h('span', {}, 'Build'), h('input', { type: 'range', min: 0.85, max: 1.15, step: 0.01, value: a.build, oninput: (e) => { a.build = parseFloat((e.target as HTMLInputElement).value); } })),
      h('label', { class: 'field' }, h('span', {}, 'Mobility aid'), h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: a.wheelchair, onchange: (e) => { a.wheelchair = (e.target as HTMLInputElement).checked; } }), ' Wheelchair')),
      h('label', { class: 'field' }, h('span', {}, 'Ink wash (muted colors)'), h('input', { type: 'range', min: 0, max: 0.7, step: 0.02, value: STYLE.desat, oninput: (e) => { STYLE.desat = parseFloat((e.target as HTMLInputElement).value); this.renderer.cache.clear(); clearIconCache(); } })),
      h('div', { class: 'row' },
        h('button', { onclick: () => { this.look.appearance = { ...randomLook(this.content, makeRng(++this.seed * 977)).appearance }; this.build(); } }, 'Random look'),
        h('button', { onclick: () => { Object.assign(this.look, randomLook(this.content, makeRng(++this.seed * 131))); this.build(); } }, 'Random everything'),
        h('button', { onclick: () => { Object.assign(this.look, defaultLook()); this.build(); } }, 'Reset')),
    );
  }

  private optionsFor(slot: Slot): { id: string; name: string }[] {
    const inv = this.game.playerInv.inv;
    return [...this.content.items.values()].filter((i) => i.equip?.slot === slot && (this.game.config.creative || inv.count(i.id) > 0 || this.look.outfit[slot]?.item === i.id)).map((i) => ({ id: i.id, name: i.name }));
  }

  private outfitTab(): HTMLElement {
    const rows = SLOTS.map((slot) => {
      const cur = this.look.outfit[slot], opts = this.optionsFor(slot);
      const colors: ColorSet | null = cur ? resolvedColors(this.content, cur) : null;
      const setColor = (k: keyof ColorSet, v: string): void => { const inst = this.look.outfit[slot]; if (inst) inst.colors = { ...inst.colors, [k]: v }; };
      return h('div', { class: 'slotrow' },
        h('span', { class: 'slotname' }, SLOT_LABELS[slot]),
        h('select', { 'aria-label': SLOT_LABELS[slot], onchange: (e) => { const v = (e.target as HTMLSelectElement).value; equip(this.look, this.content, slot, v || null); this.build(); } },
          h('option', { value: '' }, 'None'), ...opts.map((o) => h('option', { value: o.id, selected: o.id === cur?.item }, o.name))),
        colors && h('span', { class: 'colors' }, ...(['primary', 'secondary', 'accent'] as const).map((k) => h('input', { type: 'color', title: k, 'aria-label': `${SLOT_LABELS[slot]} ${k} color`, value: colors[k], oninput: (e) => setColor(k, (e.target as HTMLInputElement).value) }))));
    });
    return h('div', {}, h('p', { class: 'hint' }, this.game.config.creative ? 'Creative mode: every item is available. Pick colors for each piece.' : 'Only items in your pack are listed.'), ...rows);
  }

  private shareTab(): HTMLElement {
    const ta = h('textarea', { rows: 12, spellcheck: false }); ta.value = JSON.stringify(this.look, null, 1);
    return h('div', {}, h('p', { class: 'hint' }, 'Copy this to share your character, or paste one here and apply it.'), ta,
      h('div', { class: 'row' },
        h('button', { onclick: () => { void navigator.clipboard?.writeText(ta.value); } }, 'Copy'),
        h('button', { onclick: () => { try { Object.assign(this.look, sanitizeLook(this.content, JSON.parse(ta.value))); this.build(); } catch { ta.value = 'That is not valid character data.'; } } }, 'Apply')));
  }

  /** Draw the live preview. Called every frame while the panel is open. */
  update(dt: number): void {
    this.t += dt;
    const g = this.preview.getContext('2d') as CanvasRenderingContext2D, W = this.preview.width, H = this.preview.height;
    g.setTransform(1, 0, 0, 1, 0, 0);
    const bg = g.createRadialGradient(W / 2, H / 2, 20, W / 2, H / 2, W * 0.7); bg.addColorStop(0, '#59605a'); bg.addColorStop(1, '#2b302d'); g.fillStyle = bg; g.fillRect(0, 0, W, H);
    g.translate(W / 2 + 6, H / 2 + 4); g.scale(150, 150); g.rotate(this.t * 0.55);
    const swing = this.mode === 'swing' ? (this.t * 0.9) % 1.5 : -1;
    this.renderer.chars.draw(g, this.look, { phase: this.t * 6, walk: this.mode === 'walk' ? 1 : 0, swing: swing > 1 ? -1 : swing, time: this.t }, 192);
    this.changed();
  }
}
