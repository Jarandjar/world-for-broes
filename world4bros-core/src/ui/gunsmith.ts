import type { Game } from '../sim/game';
import type { AttachmentSlot, GunAttachmentDef } from '../data/schema';
import { h } from './dom';

export class Gunsmith {
  readonly el = h('div', { class: 'panel gunsmith-panel', style: 'padding: 16px; max-width: 640px;' });
  selectedWeapon: string = 'hunting_rifle';

  constructor(private game: Game, private onUpdate?: () => void) {
    this.build();
  }

  refresh(): void {
    this.build();
  }

  build(): void {
    const g = this.game;
    const weapons = [
      { id: 'hunting_rifle', name: '🎯 Precision Hunting Rifle' },
      { id: 'pump_shotgun', name: '💥 Heavy Pump Shotgun' },
      { id: 'nailgun_smg', name: '⚡ Scrap Pneumatic SMG' },
      { id: 'potato_cannon', name: '🥔 Kinetic Potato Cannon' },
      { id: 'slingshot', name: '🪨 Slingshot of Justice' },
    ];

    // 1. Weapon Selector Tabs
    const weaponTabs = h('div', { style: 'display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px;' },
      ...weapons.map((w) => h('button', {
        class: `tab${w.id === this.selectedWeapon ? ' on' : ''}`,
        style: `padding: 7px 12px; font-weight: bold; font-size: 13px; border-radius: 8px; cursor: pointer; border: 1.5px solid ${w.id === this.selectedWeapon ? '#38bdf8' : 'var(--line)'}; background: ${w.id === this.selectedWeapon ? 'rgba(56,189,248,0.2)' : 'rgba(20,24,32,0.85)'}; color: ${w.id === this.selectedWeapon ? '#38bdf8' : 'var(--cream)'};`,
        onclick: () => {
          this.selectedWeapon = w.id;
          this.build();
        }
      }, w.name))
    );

    const eff = g.getEffectiveGun(this.selectedWeapon);
    const mods = g.getWeaponMods(this.selectedWeapon);
    const base = g.content.items.get(this.selectedWeapon)?.gun;

    // 2. Weapon Stat Summary & Active Perks Card
    const damageDiff = eff.baseDamage - (base?.baseDamage ?? 0);
    const rangeDiff = eff.maxRange - (base?.maxRange ?? 0);
    const spreadPct = base ? Math.round((1 - eff.spreadAngle / base.spreadAngle) * 100) : 0;
    const noisePct = base ? Math.round((1 - eff.noiseRadius / base.noiseRadius) * 100) : 0;

    const statsCard = h('div', {
      style: 'background: rgba(14,18,26,0.9); border: 1.5px solid #334155; border-radius: 10px; padding: 12px; margin-bottom: 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px;'
    },
      h('div', {},
        h('strong', { style: 'color: #38bdf8; display: block;' }, '💥 Firepower:'),
        h('span', {}, `${eff.baseDamage} DMG ${damageDiff > 0 ? `(+${damageDiff})` : damageDiff < 0 ? `(${damageDiff})` : ''}`),
        h('br'),
        h('strong', { style: 'color: #38bdf8; display: block; margin-top: 4px;' }, '🎯 Effective Range:'),
        h('span', {}, `${eff.maxRange}m ${rangeDiff > 0 ? `(+${rangeDiff}m)` : ''}`)
      ),
      h('div', {},
        h('strong', { style: 'color: #38bdf8; display: block;' }, '🎯 Spread & Accuracy:'),
        h('span', {}, `${spreadPct > 0 ? `${spreadPct}% Tighter Bloom` : spreadPct < 0 ? `${Math.abs(spreadPct)}% Wider Spread` : 'Standard'}`),
        h('br'),
        h('strong', { style: 'color: #38bdf8; display: block; margin-top: 4px;' }, '🔇 Noise Signature:'),
        h('span', {}, `${noisePct > 0 ? `-${noisePct}% Quieter` : noisePct < 0 ? `+${Math.abs(noisePct)}% Loud Blast` : 'Standard'}`)
      )
    );

    // Active Visual Info Badges
    const vBadges: HTMLElement[] = [];
    if (eff.visualInfo.rangefinder) vBadges.push(h('span', { style: 'background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '🎯 Tactical Rangefinder HUD'));
    if (eff.visualInfo.targetHealthBar) vBadges.push(h('span', { style: 'background: rgba(56,189,248,0.2); color: #38bdf8; border: 1px solid #38bdf8; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '🩺 Live Target HP Gauge'));
    if (eff.visualInfo.threatScanner) vBadges.push(h('span', { style: 'background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid #ef4444; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '🚨 Threat Scanner Brackets'));
    if (eff.visualInfo.trajectoryArc) vBadges.push(h('span', { style: 'background: rgba(249,115,22,0.2); color: #f97316; border: 1px solid #f97316; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '🌐 Trajectory Path Arc'));
    if (eff.visualInfo.nightVisionReticle) vBadges.push(h('span', { style: 'background: rgba(168,85,247,0.2); color: #c084fc; border: 1px solid #c084fc; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '🌙 Night Vision Reticle'));
    if (eff.squeakSound) vBadges.push(h('span', { style: 'background: rgba(234,179,8,0.2); color: #facc15; border: 1px solid #facc15; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '🐥 Squeaky Rubber Duck'));
    if (eff.honkSound) vBadges.push(h('span', { style: 'background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #f87171; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, '📯 Car Horn Siren Muzzle'));
    if (eff.meleeDmg > 15) vBadges.push(h('span', { style: 'background: rgba(244,63,94,0.2); color: #fb7185; border: 1px solid #fb7185; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;' }, `⚔️ Bayonet Bash: ${eff.meleeDmg} DMG`));

    const perksSection = h('div', { style: 'margin-bottom: 14px; display: flex; gap: 6px; flex-wrap: wrap;' }, ...vBadges);

    // 3. Preset Quick-Combos
    const presetRow = h('div', { style: 'display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px;' },
      h('button', {
        style: 'padding: 5px 10px; font-size: 11px; font-weight: bold; background: rgba(56,189,248,0.15); border: 1px solid #38bdf8; color: #38bdf8; border-radius: 6px; cursor: pointer;',
        onclick: () => {
          g.setWeaponAttachment(this.selectedWeapon, 'optic', 'optic_rangefinder');
          g.setWeaponAttachment(this.selectedWeapon, 'muzzle', 'muzzle_copper_pipe');
          g.setWeaponAttachment(this.selectedWeapon, 'underbarrel', 'under_cat_laser');
          g.setWeaponAttachment(this.selectedWeapon, 'magazine', 'mag_rubber_band');
          g.setWeaponAttachment(this.selectedWeapon, 'goofy', 'goofy_windup_clock');
          this.build();
          this.onUpdate?.();
        }
      }, '⚡ Max Visual Info HUD Setup'),
      h('button', {
        style: 'padding: 5px 10px; font-size: 11px; font-weight: bold; background: rgba(234,179,8,0.15); border: 1px solid #facc15; color: #facc15; border-radius: 6px; cursor: pointer;',
        onclick: () => {
          g.setWeaponAttachment(this.selectedWeapon, 'optic', 'optic_googly');
          g.setWeaponAttachment(this.selectedWeapon, 'muzzle', 'muzzle_car_horn');
          g.setWeaponAttachment(this.selectedWeapon, 'underbarrel', 'under_plunger_bayonet');
          g.setWeaponAttachment(this.selectedWeapon, 'magazine', 'mag_twin_soda');
          g.setWeaponAttachment(this.selectedWeapon, 'goofy', 'goofy_rubber_duck');
          this.build();
          this.onUpdate?.();
        }
      }, '🤡 Full Goofy Homemade Combo'),
      h('button', {
        style: 'padding: 5px 10px; font-size: 11px; font-weight: bold; background: rgba(16,185,129,0.15); border: 1px solid #10b981; color: #10b981; border-radius: 6px; cursor: pointer;',
        onclick: () => {
          g.setWeaponAttachment(this.selectedWeapon, 'optic', 'optic_magnifier');
          g.setWeaponAttachment(this.selectedWeapon, 'muzzle', 'muzzle_potato_silencer');
          g.setWeaponAttachment(this.selectedWeapon, 'underbarrel', 'under_kitchen_fork');
          g.setWeaponAttachment(this.selectedWeapon, 'magazine', 'mag_rubber_band');
          g.setWeaponAttachment(this.selectedWeapon, 'goofy', 'goofy_fuzzy_dice');
          this.build();
          this.onUpdate?.();
        }
      }, '🥔 Silent Spud Assassin'),
      h('button', {
        style: 'padding: 5px 10px; font-size: 11px; font-weight: bold; background: rgba(239,68,68,0.15); border: 1px solid #ef4444; color: #ef4444; border-radius: 6px; cursor: pointer;',
        onclick: () => {
          g.setWeaponAttachment(this.selectedWeapon, 'optic', null);
          g.setWeaponAttachment(this.selectedWeapon, 'muzzle', null);
          g.setWeaponAttachment(this.selectedWeapon, 'underbarrel', null);
          g.setWeaponAttachment(this.selectedWeapon, 'magazine', null);
          g.setWeaponAttachment(this.selectedWeapon, 'goofy', null);
          this.build();
          this.onUpdate?.();
        }
      }, '🔄 Reset to Factory Stock')
    );

    // 4. Attachment Slot Selectors
    const slots: { key: AttachmentSlot; label: string; icon: string }[] = [
      { key: 'optic', label: 'Optics & Tactical Sight', icon: '🔭' },
      { key: 'muzzle', label: 'Muzzle & Barrel Attachment', icon: '🥫' },
      { key: 'underbarrel', label: 'Underbarrel & Bayonet', icon: '🔦' },
      { key: 'magazine', label: 'Magazine & Feed System', icon: '🥤' },
      { key: 'goofy', label: 'Goofy Homemade Charm & Mod', icon: '🐥' },
    ];

    const allAttachments = Array.from(g.content.attachments.values());

    const slotCards = slots.map((sl) => {
      const curId = mods[sl.key];
      const curAtt = curId ? g.content.attachments.get(curId) : null;
      const opts = allAttachments.filter((a) => a.slot === sl.key);

      const select = h('select', {
        style: 'padding: 6px 10px; background: rgba(20,24,32,0.9); border: 1.5px solid #475569; color: var(--cream); border-radius: 6px; width: 100%; font-size: 13px;',
        onchange: (e) => {
          const val = (e.target as HTMLSelectElement).value;
          g.setWeaponAttachment(this.selectedWeapon, sl.key, val || null);
          this.build();
          this.onUpdate?.();
        }
      },
        h('option', { value: '', selected: !curId }, 'None (Factory Stock)'),
        ...opts.map((o) => h('option', { value: o.id, selected: o.id === curId }, o.name))
      );

      return h('div', {
        style: 'background: rgba(20,24,34,0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; margin-bottom: 8px;'
      },
        h('div', { style: 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;' },
          h('strong', { style: 'color: var(--gold); font-size: 13px;' }, `${sl.icon} ${sl.label}`),
          curAtt ? h('span', { style: 'font-size: 11px; color: #38bdf8;' }, 'INSTALLED') : h('span', { style: 'font-size: 11px; color: var(--muted);' }, 'EMPTY')
        ),
        select,
        curAtt ? h('p', { class: 'hint', style: 'margin: 6px 0 0; font-size: 12px; color: #cbd5e1;' }, curAtt.description) : null
      );
    });

    this.el.replaceChildren(
      h('div', { style: 'display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;' },
        h('h2', { style: 'margin: 0;' }, '🛠️ Gunsmith & Weapon Modding'),
        h('span', { class: 'hint' }, 'Mod attachments, sights, and strange charms')
      ),
      weaponTabs,
      statsCard,
      perksSection,
      h('strong', { style: 'display: block; margin-bottom: 6px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;' }, 'Quick Loadout Presets:'),
      presetRow,
      h('strong', { style: 'display: block; margin-bottom: 6px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;' }, 'Attachment Sockets:'),
      ...slotCards
    );
  }
}
