import { InventorySystem, ITEM_REGISTRY } from './InventorySystem';
import { sounds } from '../audio/SoundEffects';

export interface CraftingRecipe {
  id: string;
  name: string;
  category: 'tools' | 'light' | 'survival';
  resultItemId: string;
  resultCount: number;
  cost: { itemId: string; count: number }[];
  requiresCampfire?: boolean;
  description: string;
}

export const RECIPES: CraftingRecipe[] = [
  {
    id: 'craft_axe',
    name: 'Flint Axe',
    category: 'tools',
    resultItemId: 'axe',
    resultCount: 1,
    cost: [
      { itemId: 'twigs', count: 1 },
      { itemId: 'flint', count: 1 },
    ],
    description: 'Chops trees in half the swings.',
  },
  {
    id: 'craft_torch',
    name: 'Torch',
    category: 'tools',
    resultItemId: 'torch',
    resultCount: 1,
    cost: [
      { itemId: 'twigs', count: 2 },
      { itemId: 'grass', count: 2 },
    ],
    description: 'Carry light through the night.',
  },
  {
    id: 'craft_pan',
    name: 'Cast Iron Pan',
    category: 'tools',
    resultItemId: 'pan',
    resultCount: 1,
    cost: [
      { itemId: 'flint', count: 2 },
      { itemId: 'log', count: 1 },
    ],
    description: 'Smacks trees and rocks with a loud metallic DONG!',
  },
  {
    id: 'craft_chicken',
    name: 'Rubber Chicken',
    category: 'tools',
    resultItemId: 'chicken',
    resultCount: 1,
    cost: [
      { itemId: 'grass', count: 2 },
      { itemId: 'twigs', count: 1 },
    ],
    description: 'Squeaks violently on hit. Feathers fly everywhere.',
  },
  {
    id: 'craft_marshmallow',
    name: 'Marshmallow Torch',
    category: 'light',
    resultItemId: 'marshmallow',
    resultCount: 1,
    cost: [
      { itemId: 'berries', count: 1 },
      { itemId: 'twigs', count: 1 },
    ],
    description: 'Flaming sweet snack. Lights up the darkness!',
  },
  {
    id: 'craft_horn',
    name: 'Mega Honker',
    category: 'tools',
    resultItemId: 'horn',
    resultCount: 1,
    cost: [
      { itemId: 'log', count: 1 },
      { itemId: 'flint', count: 1 },
    ],
    description: 'Press SPACE to blast a loud comical HONK!',
  },
  {
    id: 'craft_campfire',
    name: 'Campfire',
    category: 'light',
    resultItemId: 'campfire_kit',
    resultCount: 1,
    cost: [
      { itemId: 'log', count: 3 },
      { itemId: 'grass', count: 2 },
    ],
    description: 'Provides heat, light, and safety from the dark.',
  },
  {
    id: 'cook_berries',
    name: 'Roasted Berries',
    category: 'survival',
    resultItemId: 'cooked_berries',
    resultCount: 1,
    cost: [{ itemId: 'berries', count: 1 }],
    requiresCampfire: true,
    description: 'Delicious hot meal. Requires standing by a fire.',
  },
];

export class CraftingSystem {
  private activeCategory: 'tools' | 'light' | 'survival' = 'tools';
  private inventory: InventorySystem;
  private onCraftCallback?: (recipeId: string, resultItemId: string) => void;
  public isNearCampfire: boolean = false;

  constructor(inventory: InventorySystem) {
    this.inventory = inventory;
    this.setupCategoryButtons();
    this.inventory.onChange(() => this.renderRecipes());
  }

  public setOnCraft(fn: (recipeId: string, resultItemId: string) => void) {
    this.onCraftCallback = fn;
  }

  private setupCategoryButtons() {
    const buttons = document.querySelectorAll('.cat-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeCategory = btn.getAttribute('data-cat') as 'tools' | 'light' | 'survival';
        this.renderRecipes();
      });
    });
  }

  public canCraft(recipe: CraftingRecipe): boolean {
    if (recipe.requiresCampfire && !this.isNearCampfire) {
      return false;
    }
    return recipe.cost.every(c => this.inventory.getItemCount(c.itemId) >= c.count);
  }

  public craft(recipe: CraftingRecipe): boolean {
    if (!this.canCraft(recipe)) return false;

    // Deduct cost
    recipe.cost.forEach(c => this.inventory.removeItem(c.itemId, c.count));

    if (recipe.resultItemId !== 'campfire_kit') {
      this.inventory.addItem(recipe.resultItemId, recipe.resultCount);
    }

    sounds.playCraft();

    if (this.onCraftCallback) {
      this.onCraftCallback(recipe.id, recipe.resultItemId);
    }

    this.renderRecipes();
    return true;
  }

  public renderRecipes() {
    const container = document.getElementById('recipes-list');
    if (!container) return;
    container.innerHTML = '';

    const filtered = RECIPES.filter(r => r.category === this.activeCategory);

    filtered.forEach(recipe => {
      const affordable = this.canCraft(recipe);
      const card = document.createElement('div');
      card.className = `recipe-card ${affordable ? 'can-craft' : ''}`;

      let costHtml = recipe.cost
        .map(c => {
          const have = this.inventory.getItemCount(c.itemId);
          const icon = ITEM_REGISTRY[c.itemId]?.icon || '';
          const isEnough = have >= c.count;
          return `<span class="cost-item ${isEnough ? 'available' : 'missing'}">${icon} ${have}/${c.count}</span>`;
        })
        .join('');

      if (recipe.requiresCampfire && !this.isNearCampfire) {
        costHtml += `<span class="cost-item missing">(🔥 Fire needed)</span>`;
      }

      const resultIcon = ITEM_REGISTRY[recipe.resultItemId]?.icon || (recipe.resultItemId === 'campfire_kit' ? '🏕️' : '✨');

      card.innerHTML = `
        <div class="recipe-head">
          <span class="recipe-title">${resultIcon} ${recipe.name}</span>
          <button class="recipe-btn" ${affordable ? '' : 'disabled'}>CRAFT</button>
        </div>
        <div class="recipe-cost">${costHtml}</div>
      `;

      const btn = card.querySelector('.recipe-btn') as HTMLButtonElement;
      btn.onclick = () => this.craft(recipe);

      container.appendChild(card);
    });
  }
}
