import { gameConfig, GameMode, LootDensity, ZombieDensity } from '../config/GameConfig';
import { ITEM_REGISTRY, ItemCategory } from '../registries/ItemDef';
import { VEHICLE_REGISTRY } from '../registries/VehicleDef';
import { ZOMBIE_REGISTRY } from '../registries/ZombieDef';
import { PART_REGISTRY } from '../registries/PartDef';
import { InventorySystem } from '../systems/InventorySystem';
import { DayNightSystem } from '../systems/DayNightSystem';
import { sounds } from '../audio/SoundEffects';

export class DebugSpawner {
  private container: HTMLElement | null = null;
  private inventory: InventorySystem;
  private dayNight: DayNightSystem;
  private isOpen: boolean = false;
  private onTriggerHorde?: () => void;

  constructor(inventory: InventorySystem, dayNight: DayNightSystem, onTriggerHorde?: () => void) {
    this.inventory = inventory;
    this.dayNight = dayNight;
    this.onTriggerHorde = onTriggerHorde;

    this.createDOM();
    this.bindKeyboardShortcut();
    this.updateStatusBannerModeBadge();

    // Listen to game config changes
    gameConfig.subscribe(() => {
      this.updateStatusBannerModeBadge();
      this.refreshUI();
    });

    if (gameConfig.get().debugOverlay) {
      this.open();
    }
  }

  private updateStatusBannerModeBadge() {
    const banner = document.getElementById('status-banner');
    if (!banner) return;

    let badge = document.getElementById('mode-badge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'mode-badge';
      badge.className = 'mode-badge';
      const mainInfo = banner.querySelector('.banner-main');
      if (mainInfo) {
        mainInfo.appendChild(badge);
      } else {
        banner.appendChild(badge);
      }
    }

    const isZombie = gameConfig.isZombieMode();
    badge.innerHTML = isZombie
      ? `<span class="badge-zombie">🧟 ZOMBIE MODE (POIs, Cars, Roads, Horde)</span>`
      : `<span class="badge-classic">🌲 CLASSIC MODE (Pure Survival & Campfires)</span>`;
  }

  private bindKeyboardShortcut() {
    window.addEventListener('keydown', (e) => {
      // Toggle with ~ (Backquote) or F1
      if (e.code === 'Backquote' || e.key === 'F1') {
        const chatInput = document.getElementById('chat-input') as HTMLInputElement;
        if (document.activeElement === chatInput) return;
        e.preventDefault();
        this.toggle();
      }
    });
  }

  public toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  public open() {
    this.isOpen = true;
    if (this.container) {
      this.container.classList.remove('hidden');
    }
  }

  public close() {
    this.isOpen = false;
    if (this.container) {
      this.container.classList.add('hidden');
    }
  }

  private createDOM() {
    const el = document.createElement('div');
    el.id = 'debug-spawner-modal';
    el.className = 'debug-spawner hidden';
    el.innerHTML = `
      <div class="debug-panel-content">
        <div class="debug-header">
          <div class="debug-title">🛠️ WORLD 4 BRO'S — DEV CONSOLE & SPAWNER</div>
          <button id="btn-close-debug" class="debug-close-btn">&times;</button>
        </div>

        <!-- Mode & Environment Row -->
        <div class="debug-section">
          <div class="debug-section-title">🎮 Mode & World Rules</div>
          <div class="debug-controls-grid">
            <div class="debug-control-group">
              <label>Game Mode:</label>
              <div class="btn-group">
                <button id="btn-mode-classic" class="debug-btn ${gameConfig.isClassicMode() ? 'active' : ''}">🌲 Classic</button>
                <button id="btn-mode-zombie" class="debug-btn ${gameConfig.isZombieMode() ? 'active' : ''}">🧟 Zombie</button>
              </div>
            </div>

            <div class="debug-control-group">
              <label>Loot Density:</label>
              <div class="btn-group">
                <button class="debug-btn btn-loot" data-val="low">Low</button>
                <button class="debug-btn btn-loot" data-val="normal">Normal</button>
                <button class="debug-btn btn-loot" data-val="high">High (2x)</button>
              </div>
            </div>

            <div class="debug-control-group">
              <label>Time of Day:</label>
              <div class="btn-group">
                <button id="btn-toggle-always-day" class="debug-btn highlight">☀️ Always Day</button>
                <button id="btn-time-day" class="debug-btn">🌅 Day</button>
                <button id="btn-time-dusk" class="debug-btn">🌆 Dusk</button>
                <button id="btn-time-night" class="debug-btn">🌙 Night</button>
              </div>
            </div>

            <div class="debug-control-group">
              <label>Events & Horde:</label>
              <div class="btn-group">
                <button id="btn-trigger-siren" class="debug-btn danger">🚨 Sound Horde Siren</button>
                <button id="btn-clear-inv" class="debug-btn warn">🗑️ Clear Bag</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Tabbed Spawner Tabs -->
        <div class="debug-tabs">
          <button class="debug-tab active" data-tab="items">📦 Items (${Object.keys(ITEM_REGISTRY).length})</button>
          <button class="debug-tab" data-tab="vehicles">🚗 Cars (${Object.keys(VEHICLE_REGISTRY).length})</button>
          <button class="debug-tab" data-tab="parts">🔧 Car Parts (${Object.keys(PART_REGISTRY).length})</button>
          <button class="debug-tab" data-tab="zombies">🧟 Zombies (${Object.keys(ZOMBIE_REGISTRY).length})</button>
        </div>

        <!-- Items Tab -->
        <div id="tab-content-items" class="debug-tab-body">
          <div class="spawner-filter-bar">
            <select id="item-category-filter" class="debug-select">
              <option value="all">All Categories</option>
              <option value="resource">Resources</option>
              <option value="weapon_melee">Melee Weapons</option>
              <option value="weapon_ranged">Ranged Weapons</option>
              <option value="throwable">Throwables</option>
              <option value="armor">Armor & Wearables</option>
              <option value="food">Food & Cauldron</option>
              <option value="medical">Medical & Survival</option>
              <option value="tool">Tools & Utility</option>
              <option value="movement">Movement Toys</option>
              <option value="quest">Quest & Keys</option>
              <option value="junk">Junk & Relics</option>
              <option value="cosmetic">Cosmetics</option>
            </select>
            <input id="item-search-input" class="debug-input" type="text" placeholder="Search item name or ID..." />
          </div>
          <div id="items-grid" class="debug-grid"></div>
        </div>

        <!-- Vehicles Tab -->
        <div id="tab-content-vehicles" class="debug-tab-body hidden">
          <div id="vehicles-grid" class="debug-cards-grid"></div>
        </div>

        <!-- Car Parts Tab -->
        <div id="tab-content-parts" class="debug-tab-body hidden">
          <div id="parts-grid" class="debug-grid"></div>
        </div>

        <!-- Zombies Tab -->
        <div id="tab-content-zombies" class="debug-tab-body hidden">
          <div id="zombies-grid" class="debug-cards-grid"></div>
        </div>
      </div>
    `;

    document.body.appendChild(el);
    this.container = el;

    this.bindEvents();
    this.renderItemsList('all', '');
    this.renderVehiclesList();
    this.renderPartsList();
    this.renderZombiesList();
  }

  private bindEvents() {
    if (!this.container) return;

    // Close button
    document.getElementById('btn-close-debug')?.addEventListener('click', () => this.close());

    // Mode Toggle
    document.getElementById('btn-mode-classic')?.addEventListener('click', () => {
      gameConfig.update({ mode: 'classic' });
      this.refreshUI();
    });
    document.getElementById('btn-mode-zombie')?.addEventListener('click', () => {
      gameConfig.update({ mode: 'zombie' });
      this.refreshUI();
    });

    // Loot Density
    this.container.querySelectorAll('.btn-loot').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const val = (e.currentTarget as HTMLElement).getAttribute('data-val') as LootDensity;
        gameConfig.update({ lootDensity: val });
        this.refreshUI();
      });
    });

    // Time Controls
    document.getElementById('btn-toggle-always-day')?.addEventListener('click', () => {
      this.dayNight.toggleAlwaysDay();
    });
    document.getElementById('btn-time-day')?.addEventListener('click', () => {
      this.dayNight.alwaysDay = true;
      this.dayNight.timeOfDay = 80; // Peak morning
    });
    document.getElementById('btn-time-dusk')?.addEventListener('click', () => {
      this.dayNight.alwaysDay = false;
      this.dayNight.timeOfDay = 220; // Dusk warning
    });
    document.getElementById('btn-time-night')?.addEventListener('click', () => {
      this.dayNight.alwaysDay = false;
      this.dayNight.timeOfDay = 280; // Night
    });

    // Siren
    document.getElementById('btn-trigger-siren')?.addEventListener('click', () => {
      sounds.playGooseHonk();
      if (this.onTriggerHorde) this.onTriggerHorde();
      const promptEl = document.getElementById('action-prompt');
      if (promptEl) {
        promptEl.classList.remove('hidden');
        const text = document.getElementById('action-text');
        if (text) text.innerText = '🚨 HORDE NIGHT SIREN ECHOES ACROSS BIOMES!';
        setTimeout(() => promptEl.classList.add('hidden'), 4000);
      }
    });

    // Clear Bag
    document.getElementById('btn-clear-inv')?.addEventListener('click', () => {
      this.inventory.slots.forEach((s) => {
        s.itemId = null;
        s.count = 0;
      });
      this.inventory.renderHUD();
    });

    // Tab switching
    this.container.querySelectorAll('.debug-tab').forEach((tab) => {
      tab.addEventListener('click', (e) => {
        const target = (e.currentTarget as HTMLElement).getAttribute('data-tab');
        this.container?.querySelectorAll('.debug-tab').forEach((t) => t.classList.remove('active'));
        (e.currentTarget as HTMLElement).classList.add('active');

        this.container?.querySelectorAll('.debug-tab-body').forEach((b) => b.classList.add('hidden'));
        document.getElementById(`tab-content-${target}`)?.classList.remove('hidden');
      });
    });

    // Item Filters
    const catSelect = document.getElementById('item-category-filter') as HTMLSelectElement;
    const searchInput = document.getElementById('item-search-input') as HTMLInputElement;

    const onFilterChange = () => {
      this.renderItemsList(catSelect.value, searchInput.value.toLowerCase().trim());
    };

    catSelect?.addEventListener('change', onFilterChange);
    searchInput?.addEventListener('input', onFilterChange);
  }

  private refreshUI() {
    if (!this.container) return;
    const cfg = gameConfig.get();

    // Mode Buttons
    const classicBtn = document.getElementById('btn-mode-classic');
    const zombieBtn = document.getElementById('btn-mode-zombie');
    if (classicBtn) classicBtn.className = `debug-btn ${cfg.mode === 'classic' ? 'active' : ''}`;
    if (zombieBtn) zombieBtn.className = `debug-btn ${cfg.mode === 'zombie' ? 'active' : ''}`;

    // Loot Buttons
    this.container.querySelectorAll('.btn-loot').forEach((b) => {
      const val = b.getAttribute('data-val');
      b.className = `debug-btn btn-loot ${cfg.lootDensity === val ? 'active' : ''}`;
    });
  }

  private renderItemsList(category: string, search: string) {
    const grid = document.getElementById('items-grid');
    if (!grid) return;
    grid.innerHTML = '';

    const items = Object.values(ITEM_REGISTRY).filter((item) => {
      const matchesCat = category === 'all' || item.category === category;
      const matchesSearch = !search || item.id.includes(search) || item.name.toLowerCase().includes(search);
      return matchesCat && matchesSearch;
    });

    items.forEach((item) => {
      const card = document.createElement('div');
      card.className = `spawner-card tier-${item.rarity.toLowerCase()}`;
      card.innerHTML = `
        <div class="card-icon">${item.icon}</div>
        <div class="card-info">
          <div class="card-name">${item.name}</div>
          <div class="card-sub"><code>${item.id}</code> • [${item.rarity}]</div>
        </div>
        <button class="spawn-btn" title="Add to bag">+1</button>
        <button class="spawn-stack-btn" title="Add max stack">+${Math.min(item.maxStack, 10)}</button>
      `;

      card.querySelector('.spawn-btn')?.addEventListener('click', () => {
        const added = this.inventory.addItem(item.id, 1);
        if (added) sounds.playPick();
      });

      card.querySelector('.spawn-stack-btn')?.addEventListener('click', () => {
        const added = this.inventory.addItem(item.id, Math.min(item.maxStack, 10));
        if (added) sounds.playPick();
      });

      grid.appendChild(card);
    });
  }

  private renderVehiclesList() {
    const grid = document.getElementById('vehicles-grid');
    if (!grid) return;
    grid.innerHTML = '';

    Object.values(VEHICLE_REGISTRY).forEach((v) => {
      const card = document.createElement('div');
      card.className = 'debug-vehicle-card';
      card.innerHTML = `
        <div class="vehicle-title">🚗 ${v.name} <span class="rarity-badge">[${v.rarity}]</span></div>
        <div class="vehicle-specs">
          <div>💺 Seats: <b>${v.seats}</b></div>
          <div>📦 Trunk: <b>${v.trunkSlots} slots</b></div>
          <div>⚡ Top Speed: <b>${v.topSpeed} m/s</b></div>
          <div>⛽ Tank: <b>${v.fuelCapacity}L</b> (${v.fuelConsumptionRate}L/100m)</div>
          <div>📢 Horn: <b>${v.hornType}</b></div>
        </div>
        <div class="vehicle-quirk">✨ <i>${v.quirk}</i></div>
      `;
      grid.appendChild(card);
    });
  }

  private renderPartsList() {
    const grid = document.getElementById('parts-grid');
    if (!grid) return;
    grid.innerHTML = '';

    Object.values(PART_REGISTRY).forEach((p) => {
      const card = document.createElement('div');
      card.className = `spawner-card ${p.isJankFix ? 'jank' : ''}`;
      card.innerHTML = `
        <div class="card-icon">${p.icon}</div>
        <div class="card-info">
          <div class="card-name">${p.name} ${p.isRequired ? '<span class="required-tag">REQ</span>' : ''}</div>
          <div class="card-sub">Slot: <code>${p.slot}</code></div>
          ${p.quirkDescription ? `<div class="card-quirk">💡 ${p.quirkDescription}</div>` : ''}
        </div>
        <button class="spawn-btn" title="Add part as item">+1</button>
      `;

      card.querySelector('.spawn-btn')?.addEventListener('click', () => {
        // Find matching item or spawn direct
        this.inventory.addItem(p.id, 1);
        sounds.playPick();
      });

      grid.appendChild(card);
    });
  }

  private renderZombiesList() {
    const grid = document.getElementById('zombies-grid');
    if (!grid) return;
    grid.innerHTML = '';

    Object.values(ZOMBIE_REGISTRY).forEach((z) => {
      const card = document.createElement('div');
      card.className = `debug-zombie-card ${z.isBoss ? 'boss-card' : ''}`;
      card.innerHTML = `
        <div class="zombie-title">🧟 ${z.name} ${z.isBoss ? '👑 BOSS' : ''}</div>
        <div class="zombie-specs">
          <div>❤️ HP: <b>${z.maxHealth}</b></div>
          <div>🏃 Speed: <b>${z.walkSpeed} / ${z.chaseSpeed} m/s</b></div>
          <div>💥 Damage: <b>${z.damage}</b></div>
          <div>👂 Hearing: <b>${z.hearingSensitivity}x</b></div>
          <div>👀 Sight: <b>${z.sightRadius}m</b></div>
        </div>
        ${z.specialBehavior ? `<div class="zombie-behavior">⚠️ ${z.specialBehavior}</div>` : ''}
        <div class="zombie-drops">
          🎁 Drops: ${z.drops.map((d) => `<code>${d.itemId}</code> (${Math.round(d.chance * 100)}%)`).join(', ')}
        </div>
      `;
      grid.appendChild(card);
    });
  }
}
