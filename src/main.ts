import * as THREE from 'three';
import { sounds } from './audio/SoundEffects';
import { Player } from './entities/Player';
import { ParticleManager, CampfireEntity, InteractiveEntity } from './entities/WorldObjects';
import { WorldGenerator } from './world/WorldGenerator';
import { InventorySystem } from './systems/InventorySystem';
import { SurvivalStats } from './systems/SurvivalStats';
import { CraftingSystem } from './systems/CraftingSystem';
import { DayNightSystem } from './systems/DayNightSystem';
import { NetworkManager } from './network/NetworkManager';
import { TouchControls } from './input/TouchControls';
import { actionMap } from './input/ActionMap';
import { OnFootLayout, ContextualActionInfo } from './input/OnFootLayout';
import { SaveSystem } from './systems/SaveSystem';
import { damp, dampAngle, dampVector3 } from './utils/MathUtils';
import { gameConfig, AnimationQuality } from './config/GameConfig';
import { DebugSpawner } from './ui/DebugSpawner';
import { AnimationLab } from './ui/AnimationLab';
import { ITEM_REGISTRY } from './registries/ItemDef';
import { ToonDioramaPipeline } from './rendering/ToonDioramaPipeline';
import { AtmosphericNoirPipeline } from './rendering/AtmosphericNoirPipeline';
import { VerticalSliceScene } from './scene/VerticalSliceScene';
import { ComicPops } from './ui/ComicPops';
import { DrivingLayout } from './input/DrivingLayout';
import { ControllerLayout } from './input/ControllerLayout';
import { VehicleEntity } from './entities/VehicleEntity';
import { VehicleRepairSheet } from './ui/VehicleRepairSheet';
import { NoirShowcaseScene } from './scene/NoirShowcaseScene';

class GameApp {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private diorama: ToonDioramaPipeline;
  public noir: AtmosphericNoirPipeline;
  public noirShowcase: NoirShowcaseScene;
  private verticalSlice: VerticalSliceScene;
  public drivingLayout: DrivingLayout;
  public controllerLayout: ControllerLayout;
  public repairSheet: VehicleRepairSheet;

  // Camera Orbit & Damped Follow State
  public targetCameraAngle: number = 0;
  public currentCameraAngle: number = 0;
  public cameraDistance: number = 18;
  public targetCameraDistance: number = 18;
  public cameraHeight: number = 16;
  public cameraFocus: THREE.Vector3 = new THREE.Vector3(0, 0, 0);

  // Fixed Timestep Accumulator
  private accumulator: number = 0;
  private readonly FIXED_TIMESTEP: number = 1 / 60;

  // Game Feel: Hit-stop & Camera Punch
  private hitStopTimer: number = 0;
  private cameraPunchTimer: number = 0;
  private cameraPunchIntensity: number = 0;

  public triggerHitStop(seconds: number = 0.045) {
    this.hitStopTimer = seconds;
  }

  public triggerCameraPunch(intensity: number = 0.15) {
    this.cameraPunchTimer = 0.12;
    this.cameraPunchIntensity = intensity;
  }

  // Game Systems
  private particles: ParticleManager;
  private player: Player;
  private world: WorldGenerator;
  private inventory: InventorySystem;
  private stats: SurvivalStats;
  private crafting: CraftingSystem;
  private dayNight: DayNightSystem;
  private network: NetworkManager;
  private netTimer: number = 0;
  private campfires: CampfireEntity[] = [];
  private saveTimer: number = 0;
  private debugSpawner: DebugSpawner;
  private animationLab: AnimationLab;
  private lastDiscoveredPOIId: string | null = null;

  // Input State
  private keys: Record<string, boolean> = {};
  private activeTarget: InteractiveEntity | null = null;
  private clock: THREE.Clock = new THREE.Clock();
  private isStarted: boolean = false;
  private touchControls: TouchControls;
  private onFootLayout: OnFootLayout;
  private touchJoystickDir: THREE.Vector2 = new THREE.Vector2(0, 0);

  constructor() {
    const container = document.getElementById('game-container')!;

    // 1. Scene & Fog
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e0c0a);
    this.scene.fog = new THREE.FogExp2(0x0e0c0a, 0.018);

    // 2. Camera (Angled Isometric View)
    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
    this.updateCameraPosition();

    // 3. WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    // 4. Core Systems
    this.particles = new ParticleManager(this.scene);
    this.player = new Player(this.scene);
    this.player.setOnFootstep((pos) => {
      this.particles.spawnDustPuffs(pos);
    });
    this.world = new WorldGenerator(this.scene, this.particles);
    this.inventory = new InventorySystem();
    this.stats = new SurvivalStats();
    this.stats.setOnDamage(() => {
      this.player.triggerHurt();
      this.triggerCameraPunch(0.18);
    });
    this.crafting = new CraftingSystem(this.inventory);
    this.dayNight = new DayNightSystem(this.scene);
    this.dayNight.alwaysDay = true;
    this.network = new NetworkManager(this.scene);

    // Sync world time from authoritative server
    this.network.onTimeSync((time, day) => {
      this.dayNight.timeOfDay = time;
      this.dayNight.dayCount = day;
    });

    // Handle Co-op Spawning proximity from server
    this.network.onInit((data) => {
      // Only relocate when friends are already online; solo players resume their saved spot
      if (data.spawnPos && (data.othersOnline ?? 0) > 0) {
        this.player.position.set(data.spawnPos.x, 0, data.spawnPos.z);
        this.player.savePreviousState();
        this.cameraFocus.copy(this.player.position);
        this.updateCameraPosition();
        this.particles.spawnSpawnSparkles(this.player.position);
        this.player.showSpeechBubble("Joined the wild! 👋");
      }
    });

    // Spawn campfires placed by remote survivors
    this.network.onCampfireSpawn((fire) => {
      const campfire = new CampfireEntity(this.scene, new THREE.Vector3(fire.x, 0, fire.z), this.particles);
      campfire.fuelSeconds = fire.fuelSeconds;
      this.campfires.push(campfire);
      this.world.entities.push(campfire);
    });

    // Initial Inventory Setup (restore browser save if one exists)
    const save = SaveSystem.load();
    if (save) {
      SaveSystem.apply(save, this.inventory, this.stats);
      this.player.position.set(save.player.x, 0, save.player.z);
    } else {
      this.inventory.addItem('berries', 3);
    }
    this.player.position.y = this.world.getTerrainHeight(this.player.position.x, this.player.position.z);
    this.player.savePreviousState();
    this.cameraFocus.copy(this.player.position);
    this.inventory.renderHUD();
    this.stats.renderHUD();
    this.crafting.renderRecipes();
    this.debugSpawner = new DebugSpawner(this.inventory, this.dayNight, () => {
      this.honkEmote();
    });
    this.animationLab = new AnimationLab(this.player.anim);

    // Phase Z0D / V1: Atmospheric Noir Visual Engine & Ground-Hugging Volumetric Fog
    this.diorama = new ToonDioramaPipeline(this.scene, this.camera);
    this.noir = new AtmosphericNoirPipeline(this.scene, this.camera);
    this.controllerLayout = new ControllerLayout();
    if (typeof document !== 'undefined') {
      document.body.classList.add('atmospheric-noir');
    }
    this.noirShowcase = new NoirShowcaseScene(this.scene, this.camera);
    if (typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('noir')) {
      this.noirShowcase.activate();
    }

    this.verticalSlice = new VerticalSliceScene(this.scene, this.camera, this.particles);
    this.animationLab.onToggleVerticalSlice = () => {
      this.toggleVerticalSlice();
    };

    if (gameConfig.get().verticalSliceDemo) {
      this.verticalSlice.activate();
    }

    // React to mode toggle & diorama effects seamlessly
    const isZombieInitial = gameConfig.isZombieMode();
    this.world.roadNetwork.setVisible(isZombieInitial);
    this.world.settlements.setVisible(isZombieInitial);
    this.world.buildings.setVisible(isZombieInitial);
    this.world.containers.setVisible(isZombieInitial);
    this.world.zombies.setVisible(isZombieInitial);

    gameConfig.subscribe((cfg) => {
      const isZombie = cfg.mode === 'zombie';
      this.world.roadNetwork.setVisible(isZombie);
      this.world.settlements.setVisible(isZombie);
      this.world.buildings.setVisible(isZombie);
      this.world.containers.setVisible(isZombie);
      this.world.zombies.setVisible(isZombie);
      this.diorama.applyDOMStyles();
    });

    // 5. Mobile & Virtual Touch Controls
    this.touchControls = new TouchControls({
      onMove: (vec) => {
        this.touchJoystickDir.set(vec.x, vec.y);
      },
      onActionStart: () => {
        this.performAction();
      },
      onActionEnd: () => {},
      onRotateCamera: (direction) => {
        this.targetCameraAngle += (direction * Math.PI) / 4;
      },
      onEatFood: () => {
        const food = this.inventory.consumeSelected();
        if (food) {
          this.player.triggerEat();
          this.stats.eat(food.hungerBonus || 0, food.healthBonus || 0, food.sanityBonus || 0);
        }
      },
      onChat: () => {
        const chatContainer = document.getElementById('chat-container');
        const chatInput = document.getElementById('chat-input') as HTMLInputElement;
        if (chatContainer && chatInput) {
          chatContainer.classList.toggle('hidden');
          if (!chatContainer.classList.contains('hidden')) {
            chatInput.focus();
          }
        }
      },
      onDance: () => {
        this.player.triggerDance();
      },
      onHonk: () => {
        sounds.playGooseHonk();
        this.player.showSpeechBubble('📢 HONK HONK!!');
        this.network.sendChatMessage('📢 HONK HONK!!');
      },
      onToggleCrafting: () => {
        const sidebar = document.getElementById('crafting-sidebar');
        sidebar?.classList.toggle('drawer-open');
      },
    });

    // 6. Universal Action Map & Artboard 1 Phone Layout (Phase Z0B)
    actionMap.bindToWindow(window);
    this.onFootLayout = new OnFootLayout(this.inventory, this.stats);
    this.drivingLayout = new DrivingLayout();
    this.drivingLayout.onExit(() => this.exitCurrentVehicle());
    this.drivingLayout.onHonk(() => this.honkCurrentVehicle());
    this.repairSheet = new VehicleRepairSheet();
    this.dayNight.onHordeNightCallback = () => this.handleHordeNightStart();
    this.setupActionMapListeners();

    // Event Bindings
    this.bindEvents();
    this.setupCraftingCallbacks();
    this.setupDeathHandler();
    this.setupSettingsAndObjectives();

    // Expose Live State for Antigravity Watcher
    (window as any).__gameApp = this;
    (window as any).game = this;
    (window as any).getGameState = () => ({
      time: Math.round(this.dayNight.timeOfDay),
      phase: this.dayNight.isNight() ? 'NIGHT' : (this.dayNight.timeOfDay / this.dayNight.cycleDuration * 360 >= 210 ? 'DUSK' : 'DAY'),
      day: this.dayNight.dayCount,
      health: Math.round(this.stats.health),
      hunger: Math.round(this.stats.hunger),
      sanity: Math.round(this.stats.sanity),
      playerPos: {
        x: Math.round(this.player.position.x * 10) / 10,
        z: Math.round(this.player.position.z * 10) / 10,
      },
      selectedTool: this.inventory.getSelectedItemId(),
      inventory: this.inventory.slots
        .filter(s => s.itemId)
        .map(s => `${s.count}x ${s.itemId}`),
      treesFelled: this.stats.treesFelled,
      isDead: this.stats.isDead,
      nearestSettlement: this.world.settlements.getNearestSettlement(this.player.position.x, this.player.position.z)?.settlement.name || null,
      currentBuilding: this.world.buildings.currentBuilding?.lot.buildingType || null,
      isHiding: this.world.buildings.isPlayerHiding(),
      verticalSliceActive: this.verticalSlice.isActive,
    });

    // Start Game Loop
    requestAnimationFrame(this.loop.bind(this));
  }

  private setupCraftingCallbacks() {
    this.crafting.setOnCraft((_recipeId, resultItemId) => {
      this.stats.itemsCrafted += 1;
      if (resultItemId === 'campfire_kit') {
        // Place Campfire at player's position
        const fire = new CampfireEntity(this.scene, this.player.position, this.particles);
        this.campfires.push(fire);
        this.world.entities.push(fire);
        this.network.sendCampfirePlace(this.player.position.x, this.player.position.z);
      }
    });
  }

  private setupDeathHandler() {
    this.stats.setOnDeath(reason => {
      SaveSystem.clear(); // permadeath: a fresh survivor next time
      const modal = document.getElementById('death-modal');
      const reasonEl = document.getElementById('death-reason');
      const daysEl = document.getElementById('stat-days');
      const treesEl = document.getElementById('stat-trees');
      const craftedEl = document.getElementById('stat-crafted');

      if (modal && reasonEl && daysEl && treesEl && craftedEl) {
        reasonEl.innerText = `Slain by ${reason}.`;
        daysEl.innerText = `${this.stats.daysSurvived}`;
        treesEl.innerText = `${this.stats.treesFelled}`;
        craftedEl.innerText = `${this.stats.itemsCrafted}`;
        modal.classList.remove('hidden');
      }
    });

    const restartBtn = document.getElementById('btn-restart');
    if (restartBtn) {
      restartBtn.onclick = () => {
        document.getElementById('death-modal')?.classList.add('hidden');
        this.stats.reset();
        this.player.position.set(0, 0, 0);
        this.player.velocity.set(0, 0, 0);
        this.player.targetVelocity.set(0, 0, 0);
        this.player.savePreviousState();
        this.cameraFocus.set(0, 0, 0);
        this.dayNight.timeOfDay = 80;
        this.dayNight.dayCount = 1;
        this.inventory.addItem('berries', 3);
      };
    }
  }

  private setupActionMapListeners() {
    actionMap.on('HOTBAR_1', () => this.inventory.selectSlot(0));
    actionMap.on('HOTBAR_2', () => this.inventory.selectSlot(1));
    actionMap.on('HOTBAR_3', () => this.inventory.selectSlot(2));
    actionMap.on('HOTBAR_4', () => this.inventory.selectSlot(3));
    actionMap.on('HOTBAR_5', () => this.inventory.selectSlot(4));

    actionMap.on('ROTATE_CAM_LEFT', () => {
      this.targetCameraAngle += Math.PI / 4;
    });
    actionMap.on('ROTATE_CAM_RIGHT', () => {
      this.targetCameraAngle -= Math.PI / 4;
    });

    actionMap.on('QUICK_USE', () => {
      const food = this.inventory.consumeSelected();
      if (food) {
        this.player.triggerEat();
        this.stats.eat(food.hungerBonus || 0, food.healthBonus || 0, food.sanityBonus || 0);
      }
    });

    actionMap.on('THROW', () => {
      const slot = this.inventory.getSelectedItem();
      if (slot && slot.itemId) {
        const itemDef = ITEM_REGISTRY[slot.itemId];
        if (itemDef && (itemDef.category === 'throwable' || itemDef.id === 'banana_peel' || itemDef.id === 'grenade')) {
          sounds.playRubberChicken();
          this.player.showSpeechBubble(`Yeet ${itemDef.name}! 🚀`);
          this.inventory.consumeSelected();
        }
      }
    });

    actionMap.on('HONK', () => {
      this.honkEmote();
    });

    actionMap.on('EMOTE_DANCE', () => {
      this.player.triggerDance();
    });
    actionMap.on('EMOTE_HIGHFIVE', () => {
      this.player.triggerHighFive();
      this.network.sendChatMessage('✋ HIGH FIVE, BRO!');
    });
    actionMap.on('EMOTE_POINT', () => {
      this.player.triggerPoint();
      this.network.sendChatMessage('👉 OVER HERE!');
    });
    actionMap.on('EMOTE_SIT', () => {
      this.player.triggerSit();
    });

    actionMap.on('ACTION_SECONDARY', () => {
      this.handleSecondaryAction();
    });
  }

  private bindEvents() {
    // Persist progress when the tab is hidden/closed (phones rarely fire beforeunload)
    const persist = () => {
      if (this.isStarted) SaveSystem.save(this.player.position, this.inventory, this.stats);
    };
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') persist();
    });
    window.addEventListener('pagehide', persist);

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    window.addEventListener('keydown', e => {
      this.keys[e.key.toLowerCase()] = true;

      // Camera rotation (Q / E) or Interact / Hiding (R / E near spot)
      if (e.key.toLowerCase() === 'q') {
        this.targetCameraAngle += Math.PI / 4; // 45 deg clockwise
      } else if (e.key.toLowerCase() === 'e') {
        const activeCar = this.world.vehicles.getActiveVehicle();
        const nearbyCar = this.world.vehicles.getNearbyVehicle(this.player.position);
        const spot = this.world.buildings.getNearbyHidingSpot(this.player.position);
        const container = this.world.containers.getNearbyContainer(this.player.position);
        if (activeCar || nearbyCar || spot || this.world.buildings.isPlayerHiding() || container) {
          this.handleSecondaryAction();
        } else {
          this.targetCameraAngle -= Math.PI / 4; // 45 deg counter-clockwise
        }
      } else if (e.key.toLowerCase() === 'r') {
        this.handleSecondaryAction();
      }

      // F4: Toggle Virtual Controller Layout
      if (e.key === 'F4') {
        this.controllerLayout.toggle();
      }

      // Hotbar selection 1-8
      const slotNum = parseInt(e.key);
      if (!isNaN(slotNum) && slotNum >= 1 && slotNum <= 8) {
        this.inventory.selectSlot(slotNum - 1);
      }

      // Eat Food (Key F or Right click)
      if (e.key.toLowerCase() === 'f') {
        const food = this.inventory.consumeSelected();
        if (food) {
          this.player.triggerEat();
          this.stats.eat(food.hungerBonus || 0, food.healthBonus || 0, food.sanityBonus || 0);
        }
      }

      // Spacebar interact (or handbrake when driving)
      if (e.code === 'Space') {
        const chatInput = document.getElementById('chat-input') as HTMLInputElement;
        if (document.activeElement !== chatInput) {
          if (!this.world.vehicles.getActiveVehicle()) {
            this.performAction();
          }
        }
      }

      // Enter for Comic Speech Bubble Chat
      if (e.key === 'Enter') {
        const chatContainer = document.getElementById('chat-container');
        const chatInput = document.getElementById('chat-input') as HTMLInputElement;
        if (chatContainer && chatInput) {
          if (chatContainer.classList.contains('hidden')) {
            chatContainer.classList.remove('hidden');
            chatInput.focus();
          } else {
            const text = chatInput.value.trim();
            if (text) {
              this.player.showSpeechBubble(text);
              this.network.sendChatMessage(text);
              chatInput.value = '';
            }
            chatContainer.classList.add('hidden');
            (document.activeElement as HTMLElement)?.blur();
          }
        }
      }

      // Emotes (G = Dance, H = Honk, B = High-Five, P = Point/Ping, K = Sit)
      const chatInput = document.getElementById('chat-input');
      if (document.activeElement !== chatInput) {
        if (e.key.toLowerCase() === 'g') {
          this.player.triggerDance();
        } else if (e.key.toLowerCase() === 'h') {
          if (this.world.vehicles.getActiveVehicle()) {
            this.honkCurrentVehicle();
          } else {
            this.honkEmote();
          }
        } else if (e.key.toLowerCase() === 'b') {
          this.player.triggerHighFive();
          this.network.sendChatMessage('✋ HIGH FIVE, BRO!');
        } else if (e.key.toLowerCase() === 'p') {
          this.player.triggerPoint();
          this.network.sendChatMessage('👉 OVER HERE!');
        } else if (e.key.toLowerCase() === 'k') {
          this.player.triggerSit();
        } else if (e.key.toLowerCase() === 'l') {
          this.dayNight.toggleAlwaysDay();
        }
      }
    });

    window.addEventListener('keyup', e => {
      this.keys[e.key.toLowerCase()] = false;
    });

    // Dynamic Mouse Wheel Zoom (12m to 26m)
    window.addEventListener(
      'wheel',
      e => {
        this.targetCameraDistance = THREE.MathUtils.clamp(
          this.targetCameraDistance + (e.deltaY > 0 ? 1.5 : -1.5),
          12,
          26
        );
      },
      { passive: true }
    );

    // Emote buttons
    document.getElementById('btn-dance')?.addEventListener('click', () => {
      this.player.triggerDance();
    });
    document.getElementById('btn-honk')?.addEventListener('click', () => {
      this.honkEmote();
    });
    document.getElementById('btn-highfive')?.addEventListener('click', () => {
      this.player.triggerHighFive();
      this.network.sendChatMessage('✋ HIGH FIVE, BRO!');
    });
    document.getElementById('btn-point')?.addEventListener('click', () => {
      this.player.triggerPoint();
      this.network.sendChatMessage('👉 OVER HERE!');
    });
    document.getElementById('btn-sit')?.addEventListener('click', () => {
      this.player.triggerSit();
    });

    // Phone Connect & QR Modal Bindings
    const phoneModal = document.getElementById('phone-modal');
    const openPhoneModal = () => phoneModal?.classList.remove('hidden');
    const closePhoneModal = () => phoneModal?.classList.add('hidden');

    document.getElementById('btn-phone-connect')?.addEventListener('click', openPhoneModal);
    document.getElementById('btn-start-phone')?.addEventListener('click', openPhoneModal);
    document.getElementById('btn-close-phone-modal')?.addEventListener('click', closePhoneModal);
    document.getElementById('btn-dismiss-phone-modal')?.addEventListener('click', closePhoneModal);

    // DST Clock click toggle
    const dstClock = document.getElementById('dst-clock');
    if (dstClock) {
      dstClock.style.cursor = 'pointer';
      dstClock.title = 'Click or press [L] to toggle Always Daylight';
      dstClock.addEventListener('click', () => {
        this.dayNight.toggleAlwaysDay();
      });
    }

    // Canvas click to interact
    window.addEventListener('pointerdown', e => {
      if ((e.target as HTMLElement).tagName === 'CANVAS') {
        this.performAction();
      }
    });

    // Start Screen Button & Autostart Query Param
    const startOverlay = document.getElementById('start-overlay');
    const startBtn = document.getElementById('btn-start');
    const startGame = () => {
      try {
        sounds.init();
      } catch (err) {
        console.warn('AudioContext init error:', err);
      }
      if (startOverlay) {
        startOverlay.classList.add('hidden');
        startOverlay.style.display = 'none';
        startOverlay.style.pointerEvents = 'none';
      }
      this.isStarted = true;
    };

    if (startBtn) {
      startBtn.addEventListener('click', startGame);
      startBtn.onclick = startGame;
    }

    // Allow clicking anywhere on the overlay box to enter the wilderness (unless clicking QR connect)
    if (startOverlay) {
      startOverlay.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        if (target.id === 'btn-start-phone' || target.closest('#btn-start-phone')) {
          return;
        }
        startGame();
      });
    }

    // Also allow Space or Enter on start screen
    window.addEventListener('keydown', (e) => {
      if (!this.isStarted && (e.code === 'Space' || e.code === 'Enter')) {
        startGame();
      }
    });

    if (new URLSearchParams(window.location.search).has('autostart')) {
      startGame();
    }

    if (new URLSearchParams(window.location.search).has('noir')) {
      this.targetCameraAngle = Math.PI * 0.12;
      this.currentCameraAngle = Math.PI * 0.12;
      this.targetCameraDistance = 6.8;
      this.cameraDistance = 6.8;
      this.cameraHeight = 3.2;
      this.player.position.set(0, 0, 0);
      this.cameraFocus.set(0, 0.4, 0);
      this.noirShowcase.activate(new THREE.Vector3(0, 0, 0));
    }

    if (new URLSearchParams(window.location.search).has('chat')) {
      setTimeout(() => {
        this.player.showSpeechBubble('WHERE ARE MY PANTS?! 😱');
      }, 200);
    }

    if (new URLSearchParams(window.location.search).has('test')) {
      setTimeout(() => {
        // Walk towards starter pine tree
        this.keys['d'] = true;
        this.keys['w'] = true;
        setTimeout(() => {
          this.keys['d'] = false;
          this.keys['w'] = false;
          // Trigger chop swing
          this.performAction();
        }, 1200);
      }, 400);
    }
  }

  private honkEmote() {
    this.player.triggerHonk();
    sounds.playGooseHonk();
    this.particles.spawnSoundWave(this.player.position);
    this.player.showSpeechBubble('📢 HONK HONK!!');
    this.network.sendChatMessage('📢 HONK HONK!!');
    this.world.zombies.broadcastNoise(this.player.position, 35);
  }

  public enterVehicle(car: VehicleEntity) {
    car.enterVehicle(this.player);
    this.drivingLayout.show(car);
    this.player.showSpeechBubble(`Vroom! Driving ${car.def.name}! 🚗`);
  }

  public exitCurrentVehicle() {
    const car = this.world.vehicles.getActiveVehicle();
    if (car) {
      car.exitVehicle(this.player);
      this.drivingLayout.hide();
      this.player.showSpeechBubble('Stepped out of car! 🚶');
    }
  }

  public honkCurrentVehicle() {
    const car = this.world.vehicles.getActiveVehicle();
    if (car) {
      car.honk(this.particles, this.camera, (pos, radius) => {
        this.world.zombies.broadcastNoise(pos, radius);
      });
    }
  }

  public inspectVehicle(car: VehicleEntity) {
    this.repairSheet.open(car, this.inventory, this.particles, this.camera);
  }

  public handleHordeNightStart() {
    this.world.zombies.triggerHordeNight(this.player.position, 20);
    this.triggerCameraPunch(0.35);
  }

  private handleSecondaryAction() {
    // 0. Close Repair Sheet if open
    if (this.repairSheet.isOpen) {
      this.repairSheet.close();
      return;
    }

    // 0.1 Vehicle Boarding / Exit (Phase Z6)
    const activeCar = this.world.vehicles.getActiveVehicle();
    if (activeCar) {
      this.exitCurrentVehicle();
      return;
    }

    const nearbyCar = this.world.vehicles.getNearbyVehicle(this.player.position);
    if (nearbyCar) {
      const frontBumper = nearbyCar.position.clone().add(new THREE.Vector3(0, 0, 1.5).applyAxisAngle(new THREE.Vector3(0, 1, 0), nearbyCar.group.rotation.y));
      const distToHood = this.player.position.distanceTo(frontBumper);
      if (distToHood < 1.6 && !this.repairSheet.isOpen) {
        this.inspectVehicle(nearbyCar);
        return;
      }
      this.enterVehicle(nearbyCar);
      return;
    }

    // 1. Interactive Hiding Spots (Wardrobe, Under-bed, Dumpster)
    const result = this.world.buildings.interactWithHidingSpot(this.player.position);
    if (result.success) {
      if (result.isHiding) {
        this.player.showSpeechBubble('Shh... hiding! 🤫');
        this.player.group.visible = false;
        this.particles.spawnDustPuffs(this.player.position);
      } else {
        this.player.showSpeechBubble('Coast is clear! 👀');
        this.player.group.visible = true;
        this.particles.spawnDustPuffs(this.player.position);
      }
      return;
    }

    // 2. Interactive Containers (Toolbox, Cash Register, Fridge, Vault Safe, etc.)
    const containerResult = this.world.containers.interactWithNearbyContainer(
      this.player.position,
      this.inventory,
      this.particles,
      this.camera
    );
    if (containerResult.success) {
      const c = containerResult.container;
      this.player.showSpeechBubble(`Opened ${c?.def.name}! 📦`);
      return;
    }
  }

  private performAction() {
    const selectedItem = this.inventory.getSelectedItemId();
    if (selectedItem === 'horn') {
      this.honkEmote();
      return;
    }

    if (this.stats.isDead || this.player.isChopping) return;

    // Action Magnetism: Auto-turn to face nearest valid target within reach
    const target = this.activeTarget || this.findClosestInteractable();
    if (target && target.isAlive()) {
      const dx = target.position.x - this.player.position.x;
      const dz = target.position.z - this.player.position.z;
      const angleToTarget = Math.atan2(dx, dz);
      let relAngle = angleToTarget - this.currentCameraAngle;
      while (relAngle > Math.PI) relAngle -= Math.PI * 2;
      while (relAngle < -Math.PI) relAngle += Math.PI * 2;
      this.player.setHorizontalFlip(relAngle < 0);
      if (Math.abs(relAngle) < 0.85) this.player.currentView = 'FRONT';
      else if (Math.abs(relAngle) > 2.25) this.player.currentView = 'BACK';
      else this.player.currentView = 'SIDE';
    }

    // Anticipation starts immediately; impact logic executes precisely on swing apex frame!
    this.player.triggerChop(() => {
      // 1. Check for nearby zombie targets (Phase Z5)
      const nearbyZombie = this.world.zombies.getNearestZombie(this.player.position, 2.4);
      if (nearbyZombie) {
        const knockback = new THREE.Vector3().subVectors(nearbyZombie.position, this.player.position).setY(0);
        const damage = selectedItem === 'pan' ? 25 : selectedItem === 'chicken' ? 6 : 15;
        this.world.zombies.hitZombiesInRadius(nearbyZombie.position, 1.2, damage, knockback, this.particles, this.camera);
        this.triggerHitStop(0.045); // 45ms isolated hit-stop
        this.triggerCameraPunch(0.20);
        if (selectedItem === 'pan') {
          ComicPops.clang(nearbyZombie.position, this.camera);
          sounds.playFryingPanDong();
        } else if (selectedItem === 'chicken') {
          ComicPops.squeak(nearbyZombie.position, this.camera);
          sounds.playRubberChicken();
        } else {
          ComicPops.bonk(nearbyZombie.position, this.camera);
        }
        return;
      }

      if (target && target.isAlive()) {
        const hitSuccess = target.interact(this.inventory, this.stats, 0.1);
        if (hitSuccess) {
          this.triggerHitStop(0.045); // 45ms hit-stop
          this.triggerCameraPunch(0.18);
          if (selectedItem === 'pan') {
            ComicPops.clang(target.position, this.camera);
            sounds.playFryingPanDong();
          } else if (selectedItem === 'chicken') {
            ComicPops.squeak(target.position, this.camera);
            sounds.playRubberChicken();
          } else {
            ComicPops.bonk(target.position, this.camera);
          }
        }
      }
    });
  }

  private updateMovement(delta: number) {
    if (this.stats.isDead) {
      this.player.isMoving = false;
      this.player.targetVelocity.set(0, 0, 0);
      return;
    }

    const chatInput = document.getElementById('chat-input') as HTMLInputElement | null;
    if (document.activeElement === chatInput) {
      this.player.isMoving = false;
      this.player.targetVelocity.set(0, 0, 0);
      return;
    }

    // 0. Driving Mode (Phase Z6)
    const activeCar = this.world.vehicles.getActiveVehicle();
    if (activeCar) {
      let gas = 0;
      let brake = 0;
      let steer = 0;

      if (this.keys['w'] || this.keys['arrowup'] || this.drivingLayout.gasInput > 0) gas = 1;
      if (this.keys['s'] || this.keys['arrowdown'] || this.drivingLayout.brakeInput > 0) brake = 1;
      if (this.keys['a'] || this.keys['arrowleft']) steer -= 1;
      if (this.keys['d'] || this.keys['arrowright']) steer += 1;
      if (this.drivingLayout.steerInput !== 0) steer = this.drivingLayout.steerInput;
      const handbrake = !!(this.keys[' '] || this.drivingLayout.handbrakeInput);

      activeCar.applyInput(gas, brake, steer, handbrake);
      this.drivingLayout.update(activeCar);

      // Player puppet matches car position
      this.player.isMoving = Math.abs(activeCar.speed) > 0.1;
      this.player.velocity.copy(activeCar.velocity);
      this.player.position.copy(activeCar.position);
      return;
    }

    // Auto-exit hiding spot when movement keys or joystick are pressed
    if (this.world.buildings.isPlayerHiding()) {
      const isTryingToMove = !!(
        this.keys['w'] || this.keys['s'] || this.keys['a'] || this.keys['d'] ||
        this.keys['arrowup'] || this.keys['arrowdown'] || this.keys['arrowleft'] || this.keys['arrowright'] ||
        Math.abs(actionMap.analogMove.x) > 0.15 || Math.abs(actionMap.analogMove.y) > 0.15 ||
        this.touchJoystickDir.lengthSq() > 0.05
      );
      if (isTryingToMove) {
        this.handleSecondaryAction();
      } else {
        this.player.isMoving = false;
        this.player.targetVelocity.set(0, 0, 0);
        return;
      }
    }

    // Sprint mode & Sneak mode from ActionMap
    const isSneaking = actionMap.isActionActive('SNEAK');
    const isSprinting = !isSneaking && (actionMap.isActionActive('SPRINT') || !!(this.keys['shift'] || this.keys['shiftleft'] || this.keys['shiftright']));
    this.player.isSprinting = isSprinting;

    let baseSpeed = 6.5;
    if (isSprinting) baseSpeed = 9.2;
    else if (isSneaking) baseSpeed = 3.6; // Sneak crawl (0.55x)

    // Road speed bonus (+10% on roads in Zombie/Road mode)
    const roadBonus = this.world.roadNetwork.getSpeedBonusAt(this.player.position.x, this.player.position.z);
    this.player.moveSpeed = baseSpeed * roadBonus;

    // Movement vector from ActionMap (combines physical codes WASD/Arrows and analog virtual stick)
    const moveVec = actionMap.getMovementVector();
    const inputDir = new THREE.Vector2(moveVec.x, moveVec.y);

    // Fallback to legacy keys if set (e.g. for URLSearchParams ?test)
    if (inputDir.lengthSq() === 0) {
      if (this.keys['w'] || this.keys['arrowup']) inputDir.y -= 1;
      if (this.keys['s'] || this.keys['arrowdown']) inputDir.y += 1;
      if (this.keys['a'] || this.keys['arrowleft']) inputDir.x -= 1;
      if (this.keys['d'] || this.keys['arrowright']) inputDir.x += 1;
      if (this.touchJoystickDir.lengthSq() > 0) {
        inputDir.add(this.touchJoystickDir);
      }
    }

    if (inputDir.lengthSq() > 0) {
      if (inputDir.lengthSq() > 1) {
        inputDir.normalize();
      }
      this.player.isMoving = true;

      // Camera-Relative Movement: WASD always moves relative to current camera yaw
      const moveAngle = this.currentCameraAngle;
      const vx = inputDir.x * Math.cos(moveAngle) - inputDir.y * Math.sin(moveAngle);
      const vz = inputDir.x * Math.sin(moveAngle) + inputDir.y * Math.cos(moveAngle);

      this.player.targetVelocity.set(vx * this.player.moveSpeed, 0, vz * this.player.moveSpeed);

      // Smooth horizontal facing flip with squash tween
      if (inputDir.x !== 0) {
        this.player.setHorizontalFlip(inputDir.x < 0);
      }
    } else {
      this.player.isMoving = false;
      this.player.targetVelocity.set(0, 0, 0);
    }

    // Frame-rate-independent Acceleration & Deceleration (Friction / Surface Grip)
    const surfaceGrip = this.world.roadNetwork.getGripAt(this.player.position.x, this.player.position.z);
    const accelRate = 18 * surfaceGrip;
    this.player.velocity.x = damp(this.player.velocity.x, this.player.targetVelocity.x, accelRate, delta);
    this.player.velocity.z = damp(this.player.velocity.z, this.player.targetVelocity.z, accelRate, delta);

    // Apply movement velocity
    this.player.position.x += this.player.velocity.x * delta;
    this.player.position.z += this.player.velocity.z * delta;

    // POI Discovery speech notification
    if (gameConfig.isZombieMode()) {
      const nearest = this.world.roadNetwork.getNearestPOI(this.player.position.x, this.player.position.z);
      if (nearest && nearest.distance < 8.0 && this.lastDiscoveredPOIId !== nearest.poi.id) {
        this.lastDiscoveredPOIId = nearest.poi.id;
        this.player.showSpeechBubble(`📍 ${nearest.poi.name}`);
      }
    }

    // Slide Collision: Project velocity along circular obstacle tangent surfaces so characters glide smoothly around obstacles
    const obstacles = this.world.getObstacles();
    const playerRadius = 0.55;
    for (let iter = 0; iter < 2; iter++) {
      for (const obs of obstacles) {
        const dx = this.player.position.x - obs.position.x;
        const dz = this.player.position.z - obs.position.z;
        const distSq = dx * dx + dz * dz;
        const minDist = obs.radius + playerRadius;
        if (distSq < minDist * minDist) {
          const dist = Math.sqrt(distSq);
          if (dist > 0.0001) {
            const nx = dx / dist;
            const nz = dz / dist;
            this.player.position.x = obs.position.x + nx * minDist;
            this.player.position.z = obs.position.z + nz * minDist;

            // Project velocity along surface tangent
            const vDotN = this.player.velocity.x * nx + this.player.velocity.z * nz;
            if (vDotN < 0) {
              this.player.velocity.x -= vDotN * nx;
              this.player.velocity.z -= vDotN * nz;
            }
          }
        }
      }
    }
    this.player.position.y = this.world.getTerrainHeight(this.player.position.x, this.player.position.z);

    // Dust puffs while running
    if (this.player.isMoving && this.player.velocity.lengthSq() > 1.0 && Math.random() < (this.player.isSprinting ? 0.45 : 0.28)) {
      this.particles.spawnDustPuffs(this.player.position);
    }

    // Dynamic noise level update for Artboard 1 Phone Layout HUD
    let noiseLevel = 0;
    if (this.player.isChopping) {
      noiseLevel = 5;
    } else if (this.player.isMoving) {
      if (isSprinting) noiseLevel = 4;
      else if (isSneaking) noiseLevel = 1;
      else noiseLevel = 2;
    }
    this.onFootLayout.setNoiseLevel(noiseLevel);
    this.onFootLayout.setWeight(this.inventory.getTotalWeight());
    this.onFootLayout.setInfectionPercent(this.stats.infectionPercent);
    this.onFootLayout.refreshHotbarIcons();
  }

  public updateCameraPosition(delta: number = 0.016) {
    const activeCar = this.world?.vehicles?.getActiveVehicle();
    if (activeCar) {
      const targetFocus = new THREE.Vector3(
        activeCar.position.x + activeCar.velocity.x * 0.5,
        0,
        activeCar.position.z + activeCar.velocity.z * 0.5
      );
      const dist = this.cameraFocus.distanceTo(targetFocus);
      if (dist > 0.15) {
        dampVector3(this.cameraFocus, targetFocus, 8, delta);
      }
    } else if (this.player) {
      // Look-ahead based on player movement velocity
      const targetFocus = new THREE.Vector3(
        this.player.position.x + this.player.velocity.x * 0.35,
        0,
        this.player.position.z + this.player.velocity.z * 0.35
      );

      // Dead zone: ignore sub-0.15 shifts to prevent tiny jitter
      const dist = this.cameraFocus.distanceTo(targetFocus);
      if (dist > 0.15) {
        dampVector3(this.cameraFocus, targetFocus, 8, delta);
      }
    }

    let punch = 0;
    if (this.cameraPunchTimer > 0) {
      this.cameraPunchTimer -= delta;
      punch = Math.sin(this.cameraPunchTimer * 45) * this.cameraPunchIntensity;
    }

    // Dynamic Zoom: Scroll-wheel distance + sprint/driving pull-back
    const drivePullBack = activeCar ? 3.5 : 0;
    const sprintPullBack = this.player?.isSprinting ? 2.5 : 0;
    const targetDist = this.targetCameraDistance + sprintPullBack + drivePullBack;
    this.cameraDistance = damp(this.cameraDistance, targetDist, 6, delta);

    const dist = this.cameraDistance - punch;
    const x = this.cameraFocus.x + Math.sin(this.currentCameraAngle) * dist;
    const z = this.cameraFocus.z + Math.cos(this.currentCameraAngle) * dist;
    const y = this.cameraHeight - punch * 0.4;

    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.cameraFocus.x, 1.2, this.cameraFocus.z);
  }

  private updateTreeOcclusion() {
    const p1 = this.camera.position;
    const p2 = this.player.position;
    const camToPlayer = new THREE.Vector3().subVectors(p2, p1);
    const lineLen = camToPlayer.length();
    if (lineLen < 0.1) return;
    const lineDir = camToPlayer.clone().normalize();

    for (const ent of this.world.entities) {
      if (typeof ent.setOccluded === 'function') {
        const camToTree = new THREE.Vector3().subVectors(ent.position, p1);
        const proj = camToTree.dot(lineDir);
        if (proj > 2.0 && proj < lineLen - 0.5) {
          const perpDistSq = camToTree.lengthSq() - proj * proj;
          if (perpDistSq < 3.0 * 3.0) {
            ent.setOccluded(true);
          }
        }
      }
    }
  }

  private checkLightAndCampfireProximity(): { inLight: boolean; nearFire: boolean } {
    let nearFire = false;
    let inLight = false;

    // Check all active campfires
    for (const fire of this.campfires) {
      if (fire.fuelSeconds > 0) {
        const dist = this.player.position.distanceTo(fire.position);
        if (dist < fire.interactRadius) {
          nearFire = true;
        }
        if (dist < 11.0) {
          inLight = true;
        }
      }
    }

    // Lit torch in hand also provides light
    if (this.player.hasLitTorch(this.inventory)) {
      inLight = true;
    }

    // Day or Dusk gives natural light
    if (!this.dayNight.isNight()) {
      inLight = true;
    }

    return { inLight, nearFire };
  }

  private findClosestInteractable(): InteractiveEntity | null {
    let closest: InteractiveEntity | null = null;
    let minDist = Infinity;

    for (const ent of this.world.entities) {
      const dist = this.player.position.distanceTo(ent.position);
      if (dist < ent.interactRadius && dist < minDist) {
        minDist = dist;
        closest = ent;
      }
    }
    return closest;
  }

  private updateActionPrompt() {
    const promptEl = document.getElementById('action-prompt');
    const textEl = document.getElementById('action-text');
    const touchActionTitle = document.getElementById('touch-action-title');

    // 0. Driving Mode Prompts (Phase Z6)
    const activeCar = this.world.vehicles.getActiveVehicle();
    if (activeCar) {
      if (promptEl && textEl) {
        textEl.innerText = 'Exit Vehicle [E / R]';
        promptEl.classList.remove('hidden');
      }
      return;
    }

    const nearbyVehicle = this.world.vehicles.getNearbyVehicle(this.player.position);
    if (nearbyVehicle) {
      if (promptEl && textEl) {
        textEl.innerText = `Drive ${nearbyVehicle.def.name} [E / R]`;
        promptEl.classList.remove('hidden');
      }
      if (touchActionTitle) {
        touchActionTitle.innerText = 'DRIVE';
      }
      this.onFootLayout.setContextualAction({
        type: 'action',
        label: 'DRIVE',
        icon: '🚗',
      });
      return;
    }

    this.activeTarget = this.findClosestInteractable();

    if (this.activeTarget) {
      const promptText = this.activeTarget.getPrompt(this.inventory);
      if (promptText && promptEl && textEl) {
        textEl.innerText = promptText;
        promptEl.classList.remove('hidden');
        if (touchActionTitle) {
          touchActionTitle.innerText = promptText.split(' ')[0].toUpperCase();
        }

        // Update Artboard 1 Phone Layout Contextual Action Button
        let actionType: ContextualActionInfo['type'] = 'action';
        let icon = '⚡';
        const lower = promptText.toLowerCase();
        if (lower.includes('chop')) { actionType = 'chop'; icon = '🪓'; }
        else if (lower.includes('mine')) { actionType = 'mine'; icon = '⛏️'; }
        else if (lower.includes('pick') || lower.includes('harvest') || lower.includes('gather')) { actionType = 'harvest'; icon = '🧺'; }
        else if (lower.includes('loot') || lower.includes('open') || lower.includes('search')) { actionType = 'loot'; icon = '📦'; }
        else if (lower.includes('cook')) { actionType = 'cook'; icon = '🍳'; }
        else if (lower.includes('attack') || lower.includes('hit')) { actionType = 'attack'; icon = '⚔️'; }

        this.onFootLayout.setContextualAction({
          type: actionType,
          label: promptText.split(' ')[0].toUpperCase(),
          icon
        });
        return;
      }
    }

    // Check nearby hiding spot
    const nearbyHiding = this.world.buildings.getNearbyHidingSpot(this.player.position);
    if (nearbyHiding) {
      if (promptEl && textEl) {
        textEl.innerText = nearbyHiding.isOccupied ? 'Climb Out [R / E]' : nearbyHiding.promptText.replace('[E]', '[R / E]');
        promptEl.classList.remove('hidden');
      }
      if (touchActionTitle) {
        touchActionTitle.innerText = nearbyHiding.isOccupied ? 'EXIT' : 'HIDE';
      }
      this.onFootLayout.setContextualAction({
        type: 'action',
        label: nearbyHiding.isOccupied ? 'EXIT' : 'HIDE',
        icon: '🙈',
      });
      return;
    }

    promptEl?.classList.add('hidden');
    if (touchActionTitle) {
      touchActionTitle.innerText = 'ACTION';
    }
    this.onFootLayout.setContextualAction(null);
  }

  private tempScreenVec = new THREE.Vector3();
  private objectiveCompleted: boolean = false;

  private setupSettingsAndObjectives() {
    const modal = document.getElementById('settings-modal');
    const openSettings = () => {
      const input = document.getElementById('input-room-code') as HTMLInputElement;
      if (input) input.value = this.network.roomCode;
      modal?.classList.remove('hidden');
    };
    const closeSettings = () => modal?.classList.add('hidden');

    document.getElementById('btn-settings')?.addEventListener('click', openSettings);
    document.getElementById('btn-close-settings')?.addEventListener('click', closeSettings);
    document.getElementById('btn-dismiss-settings')?.addEventListener('click', closeSettings);

    const switchBtn = document.getElementById('btn-switch-room');
    const roomInput = document.getElementById('input-room-code') as HTMLInputElement;
    if (switchBtn && roomInput) {
      switchBtn.onclick = () => {
        const code = roomInput.value.trim();
        if (code) {
          this.network.switchRoom(code);
          closeSettings();
        }
      };
    }

    const copyBtn = document.getElementById('btn-copy-invite');
    if (copyBtn) {
      copyBtn.onclick = () => this.network.copyInviteLink();
    }

    const audioBtn = document.getElementById('btn-toggle-audio');
    if (audioBtn) {
      audioBtn.onclick = () => {
        const isMuted = !sounds.toggleMute();
        audioBtn.innerText = isMuted ? '🔇 Audio: MUTED' : '🔊 Audio: ON';
      };
    }

    // Mode Toggle in Settings
    const modeBtn = document.getElementById('btn-settings-toggle-mode');
    const updateModeBtn = () => {
      if (modeBtn) {
        modeBtn.innerText = gameConfig.isZombieMode() ? '🧟 Mode: Zombie' : '🌲 Mode: Classic';
      }
    };
    updateModeBtn();
    gameConfig.subscribe(() => updateModeBtn());

    if (modeBtn) {
      modeBtn.onclick = () => {
        const nextMode = gameConfig.isZombieMode() ? 'classic' : 'zombie';
        gameConfig.update({ mode: nextMode });
        updateModeBtn();
      };
    }

    // Phase Z0D: Animation Quality Setting (Low -> Medium -> High -> Low)
    const qualityBtn = document.getElementById('btn-toggle-quality');
    const updateQualityBtn = () => {
      if (qualityBtn) {
        const q = gameConfig.get().animationQuality;
        qualityBtn.innerText = `✨ Quality: ${q.toUpperCase()}`;
      }
    };
    updateQualityBtn();
    if (qualityBtn) {
      qualityBtn.onclick = () => {
        const current = gameConfig.get().animationQuality;
        const next: AnimationQuality = current === 'low' ? 'medium' : (current === 'medium' ? 'high' : 'low');
        gameConfig.update({ animationQuality: next });
        updateQualityBtn();
      };
    }

    // Phase Z0D: Juice Dial Slider (0 to 100%)
    const juiceSlider = document.getElementById('slider-juice-dial') as HTMLInputElement | null;
    const juiceVal = document.getElementById('val-juice-dial');
    if (juiceSlider && juiceVal) {
      juiceSlider.value = `${gameConfig.get().juiceDial}`;
      juiceVal.innerText = `${gameConfig.get().juiceDial}%`;
      juiceSlider.oninput = () => {
        const val = parseInt(juiceSlider.value);
        gameConfig.update({ juiceDial: val });
        juiceVal.innerText = `${val}%`;
      };
    }

    // Phase Z0D: Tilt-Shift & Paper Grain Toggles
    const tiltBtn = document.getElementById('btn-toggle-tiltshift');
    const grainBtn = document.getElementById('btn-toggle-papergrain');
    if (tiltBtn) {
      tiltBtn.innerText = `Tilt-Shift: ${gameConfig.get().tiltShiftEnabled ? 'ON' : 'OFF'}`;
      tiltBtn.onclick = () => {
        const next = !gameConfig.get().tiltShiftEnabled;
        gameConfig.update({ tiltShiftEnabled: next });
        tiltBtn.innerText = `Tilt-Shift: ${next ? 'ON' : 'OFF'}`;
      };
    }
    if (grainBtn) {
      grainBtn.innerText = `Paper Grain: ${gameConfig.get().paperGrainEnabled ? 'ON' : 'OFF'}`;
      grainBtn.onclick = () => {
        const next = !gameConfig.get().paperGrainEnabled;
        gameConfig.update({ paperGrainEnabled: next });
        grainBtn.innerText = `Paper Grain: ${next ? 'ON' : 'OFF'}`;
      };
    }

    // Phase Z0D: Launch Dusk Vertical Slice Demo
    const launchSliceBtn = document.getElementById('btn-launch-slice-demo');
    if (launchSliceBtn) {
      launchSliceBtn.onclick = () => {
        closeSettings();
        this.toggleVerticalSlice();
      };
    }

    document.getElementById('btn-settings-open-debug')?.addEventListener('click', () => {
      closeSettings();
      this.debugSpawner.open();
    });

    document.getElementById('btn-dev-spawner')?.addEventListener('click', () => {
      this.debugSpawner.toggle();
    });

    document.getElementById('btn-anim-lab-open')?.addEventListener('click', () => {
      this.animationLab.toggle();
    });
  }

  public toggleVerticalSlice() {
    if (this.verticalSlice.isActive) {
      this.verticalSlice.deactivate();
      this.updateCameraPosition();
    } else {
      this.verticalSlice.activate();
    }
  }

  private updateObjectiveTracker() {
    if (this.objectiveCompleted) return;

    const twigs = this.inventory.getItemCount('twigs');
    const flint = this.inventory.getItemCount('flint');
    const firePlaced = this.campfires.length > 0;

    const countTwigs = document.getElementById('count-twigs');
    const countFlint = document.getElementById('count-flint');
    const objTwigs = document.getElementById('obj-twigs');
    const objFlint = document.getElementById('obj-flint');
    const objFire = document.getElementById('obj-fire');

    if (countTwigs) countTwigs.innerText = `${Math.min(3, twigs)}`;
    if (countFlint) countFlint.innerText = `${Math.min(3, flint)}`;

    if (objTwigs) {
      if (twigs >= 3) objTwigs.classList.add('done');
      else objTwigs.classList.remove('done');
    }
    if (objFlint) {
      if (flint >= 3) objFlint.classList.add('done');
      else objFlint.classList.remove('done');
    }
    if (objFire) {
      if (firePlaced) objFire.classList.add('done');
      else objFire.classList.remove('done');
    }

    if (twigs >= 3 && flint >= 3 && firePlaced) {
      this.objectiveCompleted = true;
      setTimeout(() => {
        document.getElementById('objective-tracker')?.classList.add('collapsed');
      }, 3000);
    }
  }

  private updateTargetProgressBar() {
    const barEl = document.getElementById('target-progress-bar');
    const fillEl = document.getElementById('target-progress-fill');
    if (!barEl || !fillEl) return;

    if (this.activeTarget && typeof this.activeTarget.getHitProgress === 'function') {
      const progress = this.activeTarget.getHitProgress();
      if (progress > 0 && progress < 1.0) {
        this.tempScreenVec.copy(this.activeTarget.position);
        this.tempScreenVec.y += 2.2;
        this.tempScreenVec.project(this.camera);

        if (this.tempScreenVec.z < 1) {
          const x = (this.tempScreenVec.x * 0.5 + 0.5) * window.innerWidth;
          const y = (-this.tempScreenVec.y * 0.5 + 0.5) * window.innerHeight;
          barEl.style.left = `${x}px`;
          barEl.style.top = `${y}px`;
          barEl.classList.remove('hidden');
          fillEl.style.width = `${Math.round(progress * 100)}%`;
          return;
        }
      }
    }
    barEl.classList.add('hidden');
  }

  /**
   * Fixed Timestep Physics & Game Rules Tick (Runs at steady 60 Hz)
   */
  private fixedUpdate(dt: number) {
    // 0. Buffered Action Processing (120ms Action Buffer)
    if (!this.player.isChopping) {
      if (actionMap.consumeBufferedAction('ACTION_PRIMARY') || actionMap.consumeBufferedAction('ACTION_SECONDARY')) {
        this.performAction();
      }
    }

    // 1. Movement & Player Puppet State Update
    this.updateMovement(dt);

    // 2. Campfire and Lighting Checks
    const { inLight, nearFire } = this.checkLightAndCampfireProximity();
    this.crafting.isNearCampfire = nearFire;

    const isColdOrLowSanity = this.stats.sanity < 40 || (!inLight && this.dayNight.isNight());
    this.player.update(dt, this.camera, this.inventory, { nearCampfire: nearFire, isColdOrLowSanity });

    // 3. Day / Night Cycle and Charlie Darkness Attack
    this.dayNight.update(dt, this.stats, inLight);

    // 4. Survival Vitals (Hunger, Health, Sanity)
    this.stats.update(dt, this.dayNight.isNight(), nearFire, this.player.isMoving);

    // 5. Multiplayer Synchronization (20 Hz dispatch)
    this.netTimer += dt;
    if (this.netTimer >= 0.05) {
      this.network.sendPlayerUpdate({
        x: this.player.position.x,
        z: this.player.position.z,
        facingLeft: this.player.facingLeft,
        isMoving: this.player.isMoving,
        isChopping: this.player.isChopping,
        isDancing: this.player.isDancing,
        tool: this.inventory.getSelectedItemId(),
        health: this.stats.health,
        isDead: this.stats.isDead,
      });
      this.netTimer = 0;
    }

    // 6. Browser autosave (every 5s)
    this.saveTimer += dt;
    if (this.saveTimer >= 5) {
      SaveSystem.save(this.player.position, this.inventory, this.stats);
      this.saveTimer = 0;
    }
  }

  private loop() {
    requestAnimationFrame(this.loop.bind(this));

    const rawDelta = Math.min(this.clock.getDelta(), 0.1);

    // 40-50ms Hit-stop on heavy impact
    if (this.hitStopTimer > 0) {
      this.hitStopTimer -= rawDelta;
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // Smooth Camera Angle Orbit Easing (exponential frame-rate-independent)
    this.currentCameraAngle = dampAngle(this.currentCameraAngle, this.targetCameraAngle, 8, rawDelta);

    if (this.isStarted) {
      this.accumulator += rawDelta;

      // Fixed timestep updates for physics & game rules
      while (this.accumulator >= this.FIXED_TIMESTEP) {
        this.player.savePreviousState();
        this.fixedUpdate(this.FIXED_TIMESTEP);
        this.accumulator -= this.FIXED_TIMESTEP;
      }

      // Interpolation alpha for jitter-free rendering between fixed states
      const alpha = this.accumulator / this.FIXED_TIMESTEP;
      this.player.interpolateRender(alpha);

      // Smooth camera follow with look-ahead and dead-zone
      this.updateCameraPosition(rawDelta);
      this.updateTreeOcclusion();

      // Render-rate updates for visual entities, particles, network interpolation
      const { inLight } = this.checkLightAndCampfireProximity();
      this.world.update(
        rawDelta,
        this.camera,
        this.player.renderPosition,
        this.inventory,
        this.dayNight.isNight(),
        inLight,
        (dmg) => {
          this.stats.takeDamage(dmg);
          this.player.triggerHurt();
          this.triggerCameraPunch(0.22);
        }
      );
      this.particles.update(rawDelta);
      this.network.update(rawDelta, this.camera);
      this.updateActionPrompt();
      this.updateTargetProgressBar();
      this.updateObjectiveTracker();
    } else {
      this.updateCameraPosition(rawDelta);
      this.updateTreeOcclusion();
    }

    // Phase V1: Update Atmospheric Noir Volumetric Fog & Post-Processing
    const { inLight } = this.checkLightAndCampfireProximity();
    this.noir.update(
      rawDelta,
      this.dayNight.timeOfDay,
      this.dayNight.cycleDuration,
      this.player.position,
      inLight
    );

    // Legacy diorama & vertical slice
    this.diorama.update(rawDelta);
    if (this.noirShowcase.isActive) {
      this.noirShowcase.update(rawDelta);
    }
    if (this.verticalSlice.isActive) {
      this.verticalSlice.update(rawDelta);
    }
    if (this.player) {
      this.diorama.updateBlobAO('player', this.player.position.x, this.player.position.y, this.player.position.z, 0.7, 0.6);
    }

    // Render 3D Scene
    this.animationLab.update(rawDelta);
    this.renderer.render(this.scene, this.camera);
  }
}

// Initialize immediately if DOM is already ready, or wait for DOMContentLoaded
function bootstrap() {
  try {
    const app = new GameApp();
    (window as any).game = app;
  } catch (err) {
    console.error('Failed to initialize GameApp:', err);
  }
}

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
