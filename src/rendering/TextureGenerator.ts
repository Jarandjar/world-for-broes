import * as THREE from 'three';

/**
 * Procedural Gothic Ink & Charcoal Texture Generator for World 4 Bro's.
 * Renders hand-drawn aesthetic 2D canvas textures and maps them to Three.js materials.
 */
export class TextureGenerator {
  private static cache: Map<string, THREE.CanvasTexture> = new Map();

  // Helper to create high-resolution offscreen canvas
  private static createCanvas(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
    if (typeof document === 'undefined') {
      const dummyCanvas: any = {
        width,
        height,
        getContext: () => dummyCtx,
        addEventListener: () => {},
        removeEventListener: () => {},
        style: {}
      };
      const dummyCtx: any = new Proxy({}, {
        get: (_target, prop) => {
          if (prop === 'canvas') return dummyCanvas;
          return () => dummyCtx;
        },
        set: () => true
      });
      return { canvas: dummyCanvas, ctx: dummyCtx };
    }
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingEnabled = true;
    return { canvas, ctx };
  }

  // Helper to create and configure high-quality textures with mipmapping and anisotropic filtering
  public static createTexture(canvas: HTMLCanvasElement): THREE.CanvasTexture {
    const texture = new THREE.CanvasTexture(canvas);
    texture.generateMipmaps = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.anisotropy = 4;
    return texture;
  }

  // Helper for cross-hatching shade
  private static drawCrossHatch(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, density: number = 6) {
    ctx.save();
    ctx.strokeStyle = 'rgba(20, 15, 10, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = -w; i < w + h; i += density) {
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i - h * 0.5, y + h);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- PLAYER CUTOUT PARTS ---

  public static getPlayerHead(): THREE.CanvasTexture {
    if (this.cache.has('player_head')) return this.cache.get('player_head')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // 1. Spiky Wilson-style wild hair (Back layer)
    ctx.fillStyle = '#181410';
    ctx.strokeStyle = '#050302';
    ctx.lineWidth = 3;
    ctx.beginPath();
    // Wild hair spikes
    ctx.moveTo(30, 70);
    ctx.quadraticCurveTo(10, 30, 25, 15);
    ctx.lineTo(40, 28);
    ctx.quadraticCurveTo(50, 5, 64, 8);
    ctx.lineTo(75, 26);
    ctx.quadraticCurveTo(100, 10, 105, 28);
    ctx.lineTo(95, 45);
    ctx.quadraticCurveTo(120, 55, 100, 75);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // 2. Pale Face Shape
    ctx.fillStyle = '#f0e6d6';
    ctx.beginPath();
    ctx.ellipse(64, 68, 30, 38, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Face cross-hatch shading
    this.drawCrossHatch(ctx, 36, 50, 20, 45, 5);

    // 3. Expressive Large Eyes
    ctx.fillStyle = '#fffdfa';
    ctx.beginPath();
    ctx.ellipse(52, 64, 10, 13, 0.1, 0, Math.PI * 2);
    ctx.ellipse(76, 64, 10, 13, -0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Pupils (Worried glance)
    ctx.fillStyle = '#110d0a';
    ctx.beginPath();
    ctx.arc(54, 65, 3.5, 0, Math.PI * 2);
    ctx.arc(74, 65, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Eye highlights
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(53, 63, 1.2, 0, Math.PI * 2);
    ctx.arc(73, 63, 1.2, 0, Math.PI * 2);
    ctx.fill();

    // Nose & Mouth (Scratchy ink strokes)
    ctx.strokeStyle = '#2d2218';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(63, 72);
    ctx.lineTo(66, 80);
    ctx.lineTo(60, 81);
    ctx.stroke();

    // Wobbly mouth
    ctx.beginPath();
    ctx.moveTo(56, 92);
    ctx.quadraticCurveTo(64, 90, 72, 94);
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('player_head', texture);
    return texture;
  }

  public static getPlayerTorso(): THREE.CanvasTexture {
    if (this.cache.has('player_torso')) return this.cache.get('player_torso')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Victorian waistcoat (Dark charcoal/reddish brown)
    ctx.fillStyle = '#2c221a';
    ctx.strokeStyle = '#0a0705';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(44, 20);
    ctx.lineTo(84, 20);
    ctx.lineTo(88, 85);
    ctx.lineTo(40, 85);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // White Shirt Collar / Cravat
    ctx.fillStyle = '#ded4c3';
    ctx.beginPath();
    ctx.moveTo(54, 20);
    ctx.lineTo(64, 45);
    ctx.lineTo(74, 20);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Red necktie
    ctx.fillStyle = '#8f2923';
    ctx.beginPath();
    ctx.moveTo(62, 35);
    ctx.lineTo(66, 35);
    ctx.lineTo(67, 58);
    ctx.lineTo(64, 63);
    ctx.lineTo(61, 58);
    ctx.closePath();
    ctx.fill();

    // Waistcoat buttons
    ctx.fillStyle = '#d4ad65';
    ctx.beginPath();
    ctx.arc(64, 68, 2.5, 0, Math.PI * 2);
    ctx.arc(64, 76, 2.5, 0, Math.PI * 2);
    ctx.fill();

    this.drawCrossHatch(ctx, 42, 25, 20, 55, 6);

    const texture = this.createTexture(canvas);
    this.cache.set('player_torso', texture);
    return texture;
  }

  public static getPlayerArm(): THREE.CanvasTexture {
    if (this.cache.has('player_arm')) return this.cache.get('player_arm')!;
    const { canvas, ctx } = this.createCanvas(64, 128);

    // Dark sleeve
    ctx.fillStyle = '#221a14';
    ctx.strokeStyle = '#080503';
    ctx.lineWidth = 2.5;

    ctx.beginPath();
    ctx.moveTo(22, 15);
    ctx.lineTo(42, 15);
    ctx.lineTo(40, 75);
    ctx.lineTo(24, 75);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Hand (Pale skin, sketchy fingers)
    ctx.fillStyle = '#f0e6d6';
    ctx.beginPath();
    ctx.arc(32, 88, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('player_arm', texture);
    return texture;
  }

  public static getPlayerLeg(): THREE.CanvasTexture {
    if (this.cache.has('player_leg')) return this.cache.get('player_leg')!;
    const { canvas, ctx } = this.createCanvas(64, 128);

    // Slim black trousers
    ctx.fillStyle = '#171310';
    ctx.strokeStyle = '#050302';
    ctx.lineWidth = 2.5;

    ctx.beginPath();
    ctx.moveTo(22, 10);
    ctx.lineTo(40, 10);
    ctx.lineTo(38, 80);
    ctx.lineTo(24, 80);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Pointy gothic leather boot
    ctx.fillStyle = '#0a0807';
    ctx.beginPath();
    ctx.moveTo(22, 80);
    ctx.lineTo(38, 80);
    ctx.lineTo(44, 105);
    ctx.lineTo(16, 105);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('player_leg', texture);
    return texture;
  }

  // --- ENVIRONMENT BILLBOARD TEXTURES ---

  public static getPineTreeTexture(): THREE.CanvasTexture {
    if (this.cache.has('tree_pine')) return this.cache.get('tree_pine')!;
    const { canvas, ctx } = this.createCanvas(256, 512);

    // Trunk
    ctx.fillStyle = '#3a2717';
    ctx.strokeStyle = '#120b06';
    ctx.lineWidth = 5;

    ctx.beginPath();
    ctx.moveTo(105, 300);
    ctx.lineTo(151, 300);
    ctx.lineTo(165, 500);
    ctx.lineTo(91, 500);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Trunk bark lines
    ctx.strokeStyle = '#1e140c';
    ctx.lineWidth = 3;
    for (let y = 320; y < 480; y += 30) {
      ctx.beginPath();
      ctx.moveTo(110 + (y % 15), y);
      ctx.lineTo(145 - (y % 15), y + 10);
      ctx.stroke();
    }

    // Foliage Layers (Gothic drooping tiers, dark forest green)
    const layers = [
      { y: 340, w: 210, h: 90 },
      { y: 260, w: 180, h: 80 },
      { y: 180, w: 150, h: 75 },
      { y: 100, w: 110, h: 70 },
      { y: 30, w: 60, h: 60 },
    ];

    layers.forEach((layer, idx) => {
      ctx.fillStyle = idx % 2 === 0 ? '#1f3826' : '#294a32';
      ctx.strokeStyle = '#0b160e';
      ctx.lineWidth = 4;

      ctx.beginPath();
      ctx.moveTo(128, layer.y - layer.h);
      // Left jagged needle fringe
      ctx.lineTo(128 - layer.w * 0.35, layer.y - layer.h * 0.4);
      ctx.lineTo(128 - layer.w * 0.2, layer.y - layer.h * 0.3);
      ctx.lineTo(128 - layer.w * 0.5, layer.y);
      ctx.lineTo(128 - layer.w * 0.3, layer.y - 10);
      ctx.lineTo(128 - layer.w * 0.4, layer.y + 15);
      // Center bottom curve
      ctx.lineTo(128, layer.y - 5);
      // Right jagged needle fringe
      ctx.lineTo(128 + layer.w * 0.4, layer.y + 15);
      ctx.lineTo(128 + layer.w * 0.3, layer.y - 10);
      ctx.lineTo(128 + layer.w * 0.5, layer.y);
      ctx.lineTo(128 + layer.w * 0.2, layer.y - layer.h * 0.3);
      ctx.lineTo(128 + layer.w * 0.35, layer.y - layer.h * 0.4);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Cross-hatching shade on left side
      this.drawCrossHatch(ctx, 128 - layer.w * 0.45, layer.y - layer.h * 0.5, layer.w * 0.45, layer.h, 7);
    });

    const texture = this.createTexture(canvas);
    this.cache.set('tree_pine', texture);
    return texture;
  }

  public static getTreeStumpTexture(): THREE.CanvasTexture {
    if (this.cache.has('tree_stump')) return this.cache.get('tree_stump')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Stump base
    ctx.fillStyle = '#3a2717';
    ctx.strokeStyle = '#120b06';
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.ellipse(64, 85, 45, 20, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Top ring
    ctx.fillStyle = '#d4ab77';
    ctx.beginPath();
    ctx.ellipse(64, 75, 40, 16, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Growth rings
    ctx.strokeStyle = '#7c572b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(64, 75, 28, 11, 0, 0, Math.PI * 2);
    ctx.ellipse(64, 75, 16, 6, 0, 0, Math.PI * 2);
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('tree_stump', texture);
    return texture;
  }

  public static getBoulderTexture(): THREE.CanvasTexture {
    if (this.cache.has('boulder')) return this.cache.get('boulder')!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    // Jagged craggy rock
    ctx.fillStyle = '#47433f';
    ctx.strokeStyle = '#171412';
    ctx.lineWidth = 5;

    ctx.beginPath();
    ctx.moveTo(60, 210);
    ctx.lineTo(35, 150);
    ctx.lineTo(70, 70);
    ctx.lineTo(130, 45);
    ctx.lineTo(200, 75);
    ctx.lineTo(225, 140);
    ctx.lineTo(205, 210);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Rock facet planes
    ctx.fillStyle = '#5c5853';
    ctx.beginPath();
    ctx.moveTo(70, 70);
    ctx.lineTo(130, 45);
    ctx.lineTo(150, 130);
    ctx.lineTo(95, 150);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Shiny flint fissure
    ctx.fillStyle = '#1e242b';
    ctx.strokeStyle = '#5a738c';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(135, 80);
    ctx.lineTo(170, 110);
    ctx.lineTo(155, 170);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    this.drawCrossHatch(ctx, 35, 130, 70, 80, 6);

    const texture = this.createTexture(canvas);
    this.cache.set('boulder', texture);
    return texture;
  }

  public static getBerryBushTexture(hasBerries: boolean): THREE.CanvasTexture {
    const key = hasBerries ? 'berry_bush_full' : 'berry_bush_empty';
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    // Bramble twigs
    ctx.strokeStyle = '#2d2218';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(128, 220);
    ctx.quadraticCurveTo(100, 160, 70, 120);
    ctx.moveTo(128, 220);
    ctx.quadraticCurveTo(140, 150, 185, 115);
    ctx.moveTo(128, 220);
    ctx.lineTo(128, 100);
    ctx.stroke();

    // Leafy bush mass (Spiky gothic leaves)
    ctx.fillStyle = '#263b1d';
    ctx.strokeStyle = '#0d170a';
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.arc(95, 140, 45, 0, Math.PI * 2);
    ctx.arc(160, 135, 45, 0, Math.PI * 2);
    ctx.arc(128, 95, 48, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    this.drawCrossHatch(ctx, 60, 90, 80, 90, 7);

    // Bright ripe red berries
    if (hasBerries) {
      const berryPositions = [
        { x: 95, y: 115 },
        { x: 125, y: 80 },
        { x: 155, y: 110 },
        { x: 120, y: 140 },
        { x: 165, y: 155 },
      ];

      berryPositions.forEach(pos => {
        ctx.fillStyle = '#c72424';
        ctx.strokeStyle = '#4a0b0b';
        ctx.lineWidth = 2.5;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 11, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Berry gleam
        ctx.fillStyle = '#ff8282';
        ctx.beginPath();
        ctx.arc(pos.x - 3, pos.y - 3, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  public static getGrassTuftTexture(): THREE.CanvasTexture {
    if (this.cache.has('grass_tuft')) return this.cache.get('grass_tuft')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    ctx.fillStyle = '#8f883b';
    ctx.strokeStyle = '#26240d';
    ctx.lineWidth = 2.5;

    const blades = [
      { x1: 64, y1: 110, cx: 35, cy: 50, x2: 20, y2: 25 },
      { x1: 64, y1: 110, cx: 48, cy: 40, x2: 42, y2: 15 },
      { x1: 64, y1: 110, cx: 64, cy: 30, x2: 66, y2: 8 },
      { x1: 64, y1: 110, cx: 80, cy: 45, x2: 90, y2: 18 },
      { x1: 64, y1: 110, cx: 95, cy: 60, x2: 112, y2: 32 },
    ];

    blades.forEach(b => {
      ctx.beginPath();
      ctx.moveTo(b.x1 - 4, b.y1);
      ctx.quadraticCurveTo(b.cx - 2, b.cy, b.x2, b.y2);
      ctx.quadraticCurveTo(b.cx + 2, b.cy, b.x1 + 4, b.y1);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    const texture = this.createTexture(canvas);
    this.cache.set('grass_tuft', texture);
    return texture;
  }

  public static getCampfireBaseTexture(): THREE.CanvasTexture {
    if (this.cache.has('campfire_base')) return this.cache.get('campfire_base')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Stone ring pit
    ctx.fillStyle = '#1c1510';
    ctx.beginPath();
    ctx.ellipse(64, 64, 45, 30, 0, 0, Math.PI * 2);
    ctx.fill();

    // Charcoal & glowing orange embers in center
    ctx.fillStyle = '#d94b18';
    ctx.beginPath();
    ctx.ellipse(64, 64, 25, 16, 0, 0, Math.PI * 2);
    ctx.fill();

    // Surrounding stones
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const sx = 64 + Math.cos(angle) * 40;
      const sy = 64 + Math.sin(angle) * 25;
      ctx.fillStyle = '#574f46';
      ctx.strokeStyle = '#1a1714';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(sx, sy, 10, 8, angle, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    const texture = this.createTexture(canvas);
    this.cache.set('campfire_base', texture);
    return texture;
  }

  public static getCampfireFlameTexture(): THREE.CanvasTexture {
    if (this.cache.has('campfire_flame')) return this.cache.get('campfire_flame')!;
    const { canvas, ctx } = this.createCanvas(128, 256);

    // Stylized comic/ink fire tongues
    ctx.fillStyle = '#ff7b00';
    ctx.strokeStyle = '#801b00';
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.moveTo(64, 20);
    ctx.quadraticCurveTo(90, 80, 100, 160);
    ctx.quadraticCurveTo(110, 220, 64, 240);
    ctx.quadraticCurveTo(20, 220, 28, 160);
    ctx.quadraticCurveTo(40, 80, 64, 20);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Inner bright yellow flame
    ctx.fillStyle = '#ffee44';
    ctx.beginPath();
    ctx.moveTo(64, 70);
    ctx.quadraticCurveTo(80, 120, 85, 175);
    ctx.quadraticCurveTo(80, 225, 64, 235);
    ctx.quadraticCurveTo(45, 225, 43, 175);
    ctx.quadraticCurveTo(48, 120, 64, 70);
    ctx.closePath();
    ctx.fill();

    const texture = this.createTexture(canvas);
    this.cache.set('campfire_flame', texture);
    return texture;
  }

  // --- TOOLS & WEAPONS ---

  public static getAxeTexture(): THREE.CanvasTexture {
    if (this.cache.has('tool_axe')) return this.cache.get('tool_axe')!;
    const { canvas, ctx } = this.createCanvas(64, 128);

    // Curved wood handle
    ctx.strokeStyle = '#4d3319';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(32, 115);
    ctx.quadraticCurveTo(34, 60, 30, 25);
    ctx.stroke();

    // Sharp flint axehead
    ctx.fillStyle = '#2b2927';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(25, 30);
    ctx.lineTo(55, 15);
    ctx.lineTo(58, 48);
    ctx.lineTo(25, 45);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Twig lashings
    ctx.strokeStyle = '#9e8954';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(24, 34); ctx.lineTo(34, 44);
    ctx.moveTo(34, 34); ctx.lineTo(24, 44);
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('tool_axe', texture);
    return texture;
  }

  public static getTorchTexture(): THREE.CanvasTexture {
    if (this.cache.has('tool_torch')) return this.cache.get('tool_torch')!;
    const { canvas, ctx } = this.createCanvas(64, 128);

    // Stick
    ctx.strokeStyle = '#523c27';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(32, 115);
    ctx.lineTo(32, 45);
    ctx.stroke();

    // Wrapped rag
    ctx.fillStyle = '#30261c';
    ctx.fillRect(25, 45, 14, 25);

    // Fiery tip
    ctx.fillStyle = '#ff9900';
    ctx.beginPath();
    ctx.arc(32, 35, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffff66';
    ctx.beginPath();
    ctx.arc(32, 35, 7, 0, Math.PI * 2);
    const texture = this.createTexture(canvas);
    this.cache.set('tool_torch', texture);
    return texture;
  }

  // --- FUNNY / DUMB TOOLS ---

  public static getFryingPanTexture(): THREE.CanvasTexture {
    if (this.cache.has('tool_pan')) return this.cache.get('tool_pan')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Cast iron handle
    ctx.fillStyle = '#151515';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.fillRect(58, 70, 12, 50);

    // Round heavy pan
    ctx.fillStyle = '#222224';
    ctx.beginPath();
    ctx.arc(64, 45, 36, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Inner rim
    ctx.fillStyle = '#323236';
    ctx.beginPath();
    ctx.arc(64, 45, 28, 0, Math.PI * 2);
    ctx.fill();

    // Metal sheen shine
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(64, 45, 22, -Math.PI * 0.7, -Math.PI * 0.2);
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('tool_pan', texture);
    return texture;
  }

  public static getRubberChickenTexture(): THREE.CanvasTexture {
    if (this.cache.has('tool_chicken')) return this.cache.get('tool_chicken')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Yellow rubber chicken body
    ctx.fillStyle = '#f5c800';
    ctx.strokeStyle = '#8a6e00';
    ctx.lineWidth = 3;

    // Body & long skinny neck
    ctx.beginPath();
    ctx.ellipse(64, 75, 20, 35, 0, 0, Math.PI * 2);
    ctx.rect(56, 25, 16, 40);
    ctx.fill();
    ctx.stroke();

    // Red rubber comb on head
    ctx.fillStyle = '#e62424';
    ctx.beginPath();
    ctx.arc(58, 18, 6, 0, Math.PI * 2);
    ctx.arc(64, 14, 7, 0, Math.PI * 2);
    ctx.arc(70, 18, 6, 0, Math.PI * 2);
    ctx.fill();

    // Screaming open beak
    ctx.fillStyle = '#e66b00';
    ctx.beginPath();
    ctx.moveTo(56, 32);
    ctx.lineTo(44, 38);
    ctx.lineTo(56, 44);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Googly eye
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(62, 28, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(61, 28, 2, 0, Math.PI * 2);
    ctx.fill();

    const texture = this.createTexture(canvas);
    this.cache.set('tool_chicken', texture);
    return texture;
  }

  public static getMarshmallowStickTexture(): THREE.CanvasTexture {
    if (this.cache.has('tool_marshmallow')) return this.cache.get('tool_marshmallow')!;
    const { canvas, ctx } = this.createCanvas(64, 128);

    // Long crooked stick
    ctx.strokeStyle = '#4a3319';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(32, 120);
    ctx.lineTo(34, 40);
    ctx.lineTo(30, 15);
    ctx.stroke();

    // Two puffy marshmallows
    ctx.fillStyle = '#f7f4ed';
    ctx.strokeStyle = '#2b1d0e';
    ctx.lineWidth = 2;

    // Bottom marshmallow (toasty brown patches)
    ctx.beginPath();
    ctx.roundRect(22, 45, 22, 18, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#8f5624';
    ctx.fillRect(26, 50, 10, 6);

    // Top flaming marshmallow
    ctx.fillStyle = '#fffef7';
    ctx.beginPath();
    ctx.roundRect(20, 22, 22, 18, 4);
    ctx.fill();
    ctx.stroke();

    // Little flame on top
    ctx.fillStyle = '#ff7b00';
    ctx.beginPath();
    ctx.moveTo(31, 22);
    ctx.quadraticCurveTo(40, 10, 31, 2);
    ctx.quadraticCurveTo(22, 10, 31, 22);
    ctx.fill();

    const texture = this.createTexture(canvas);
    this.cache.set('tool_marshmallow', texture);
    return texture;
  }

  // --- COMIC SPEECH BUBBLE RENDERER ---

  public static createSpeechBubbleTexture(text: string): THREE.CanvasTexture {
    const { canvas, ctx } = this.createCanvas(320, 128);

    // Comic speech bubble body
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#1a140e';
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.roundRect(10, 10, 300, 80, 20);
    // Pointy speech tail downwards
    ctx.moveTo(140, 90);
    ctx.lineTo(160, 118);
    ctx.lineTo(180, 90);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Text inside
    ctx.fillStyle = '#110d08';
    ctx.font = 'bold 24px "MedievalSharp", "Comic Sans MS", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Truncate if long
    const display = text.length > 20 ? text.slice(0, 18) + '...' : text;
    ctx.fillText(display, 160, 50);

    const texture = this.createTexture(canvas);
    return texture;
  }

  // --- GROUND TERRAIN TEXTURE ---

  public static getGroundTexture(): THREE.CanvasTexture {
    if (this.cache.has('ground_terrain')) return this.cache.get('ground_terrain')!;
    const { canvas, ctx } = this.createCanvas(512, 512);

    // Base parchment/dirt color
    ctx.fillStyle = '#382f23';
    ctx.fillRect(0, 0, 512, 512);

    // Savannah grass & moss patches
    ctx.fillStyle = '#443c2c';
    for (let i = 0; i < 40; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const r = 20 + Math.random() * 45;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hand-drawn sketchy cross-hatch ground details
    ctx.strokeStyle = '#272017';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 60; i++) {
      const gx = Math.random() * 480;
      const gy = Math.random() * 480;
      ctx.beginPath();
      ctx.moveTo(gx, gy);
      ctx.lineTo(gx + 12 + Math.random() * 10, gy + 8 + Math.random() * 6);
      ctx.moveTo(gx + 4, gy - 3);
      ctx.lineTo(gx + 16, gy + 5);
      ctx.stroke();
    }

    const texture = this.createTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(12, 12);
    this.cache.set('ground_terrain', texture);
    return texture;
  }

  // --- AMBIENT CRITTER: GOTHIC WILDERNESS BUNNY ---
  public static getBunnyTexture(): THREE.CanvasTexture {
    if (this.cache.has('ambient_bunny')) return this.cache.get('ambient_bunny')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Cute sketchy papercraft bunny
    ctx.fillStyle = '#eae2cf';
    ctx.strokeStyle = '#221a12';
    ctx.lineWidth = 3.5;

    // Ears
    ctx.beginPath();
    ctx.ellipse(50, 32, 8, 24, -0.15, 0, Math.PI * 2);
    ctx.ellipse(78, 32, 8, 24, 0.15, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Inner pink ears
    ctx.fillStyle = '#d48888';
    ctx.beginPath();
    ctx.ellipse(50, 32, 4, 16, -0.15, 0, Math.PI * 2);
    ctx.ellipse(78, 32, 4, 16, 0.15, 0, Math.PI * 2);
    ctx.fill();

    // Body
    ctx.fillStyle = '#eae2cf';
    ctx.beginPath();
    ctx.ellipse(64, 86, 26, 28, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Head
    ctx.beginPath();
    ctx.ellipse(64, 58, 20, 18, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Eyes
    ctx.fillStyle = '#1c150e';
    ctx.beginPath();
    ctx.arc(56, 56, 3, 0, Math.PI * 2);
    ctx.arc(72, 56, 3, 0, Math.PI * 2);
    ctx.fill();

    // Nose & Whiskers
    ctx.fillStyle = '#b85e5e';
    ctx.beginPath();
    ctx.arc(64, 62, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#221a12';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(58, 62); ctx.lineTo(44, 60);
    ctx.moveTo(58, 64); ctx.lineTo(45, 66);
    ctx.moveTo(70, 62); ctx.lineTo(84, 60);
    ctx.moveTo(70, 64); ctx.lineTo(83, 66);
    ctx.stroke();

    // Fluffy puff tail
    ctx.fillStyle = '#f7f4ea';
    ctx.beginPath();
    ctx.arc(38, 92, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('ambient_bunny', texture);
    return texture;
  }

  // --- AMBIENT FIREFLY TEXTURE ---
  public static getFireflyTexture(): THREE.CanvasTexture {
    if (this.cache.has('ambient_firefly')) return this.cache.get('ambient_firefly')!;
    const { canvas, ctx } = this.createCanvas(64, 64);

    const grad = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
    grad.addColorStop(0, '#ffffaa');
    grad.addColorStop(0.3, 'rgba(215, 255, 80, 0.85)');
    grad.addColorStop(0.7, 'rgba(120, 220, 50, 0.25)');
    grad.addColorStop(1, 'rgba(60, 160, 20, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const texture = this.createTexture(canvas);
    this.cache.set('ambient_firefly', texture);
    return texture;
  }

  // --- DRIFTING LEAF TEXTURE ---
  public static getDriftingLeafTexture(): THREE.CanvasTexture {
    if (this.cache.has('ambient_leaf')) return this.cache.get('ambient_leaf')!;
    const { canvas, ctx } = this.createCanvas(64, 64);

    ctx.fillStyle = '#b86127';
    ctx.strokeStyle = '#381a06';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(32, 10);
    ctx.quadraticCurveTo(50, 24, 46, 46);
    ctx.quadraticCurveTo(32, 56, 32, 54);
    ctx.quadraticCurveTo(18, 46, 14, 24);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Central leaf vein
    ctx.beginPath();
    ctx.moveTo(32, 14);
    ctx.lineTo(32, 52);
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('ambient_leaf', texture);
    return texture;
  }

  // --- LURKING NIGHT EYES TEXTURE ---
  public static getWarningEyesTexture(): THREE.CanvasTexture {
    if (this.cache.has('night_eyes')) return this.cache.get('night_eyes')!;
    const { canvas, ctx } = this.createCanvas(128, 64);

    // Pair of sinister yellow-red slitted predator eyes
    const drawEye = (cx: number, cy: number) => {
      // Glow
      const glow = ctx.createRadialGradient(cx, cy, 2, cx, cy, 20);
      glow.addColorStop(0, '#ff4422');
      glow.addColorStop(0.5, 'rgba(255, 170, 0, 0.6)');
      glow.addColorStop(1, 'rgba(255, 50, 0, 0)');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, 20, 0, Math.PI * 2);
      ctx.fill();

      // Slit eye
      ctx.fillStyle = '#ffee44';
      ctx.beginPath();
      ctx.ellipse(cx, cy, 14, 8, 0.1, 0, Math.PI * 2);
      ctx.fill();

      // Black slit pupil
      ctx.fillStyle = '#110505';
      ctx.beginPath();
      ctx.ellipse(cx, cy, 3, 7, 0, 0, Math.PI * 2);
      ctx.fill();
    };

    drawEye(44, 32);
    drawEye(84, 32);

    const texture = this.createTexture(canvas);
    this.cache.set('night_eyes', texture);
    return texture;
  }

  // --- MULTI-DIRECTIONAL CUTOUTS (8-WAY PERSPECTIVE SUPPORT) ---

  // Head seen from the back (Spiky hair crown, dark coat collar, no face)
  public static getPlayerHeadBack(): THREE.CanvasTexture {
    if (this.cache.has('player_head_back')) return this.cache.get('player_head_back')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Dark Spiky Hair Silhouette from Rear
    ctx.fillStyle = '#181410';
    ctx.strokeStyle = '#050302';
    ctx.lineWidth = 3.5;

    ctx.beginPath();
    ctx.moveTo(30, 80);
    ctx.quadraticCurveTo(10, 30, 25, 15);
    ctx.lineTo(40, 28);
    ctx.quadraticCurveTo(50, 5, 64, 8);
    ctx.lineTo(75, 26);
    ctx.quadraticCurveTo(100, 10, 105, 28);
    ctx.lineTo(95, 45);
    ctx.quadraticCurveTo(120, 55, 100, 80);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Dense Hair Strands & Volume Cross-Hatching
    this.drawCrossHatch(ctx, 30, 25, 68, 55, 5);

    // Pale Neck Base at bottom
    ctx.fillStyle = '#e8dccd';
    ctx.beginPath();
    ctx.ellipse(64, 86, 18, 14, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Dark coat collar framing neck
    ctx.fillStyle = '#221a14';
    ctx.beginPath();
    ctx.moveTo(42, 85);
    ctx.lineTo(86, 85);
    ctx.lineTo(80, 102);
    ctx.lineTo(48, 102);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('player_head_back', texture);
    return texture;
  }

  // Torso seen from the back (Coat back seam, shoulder straps, rolled bedroll)
  public static getPlayerTorsoBack(): THREE.CanvasTexture {
    if (this.cache.has('player_torso_back')) return this.cache.get('player_torso_back')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Victorian Coat Back
    ctx.fillStyle = '#261e17';
    ctx.strokeStyle = '#0a0705';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(44, 20);
    ctx.lineTo(84, 20);
    ctx.lineTo(88, 85);
    ctx.lineTo(40, 85);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Center vertical coat seam
    ctx.strokeStyle = '#120d09';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(64, 20);
    ctx.lineTo(64, 85);
    ctx.stroke();

    // Leather Backpack Straps
    ctx.fillStyle = '#614629';
    ctx.strokeStyle = '#291b0d';
    ctx.lineWidth = 2;

    ctx.fillRect(48, 20, 7, 65);
    ctx.strokeRect(48, 20, 7, 65);
    ctx.fillRect(73, 20, 7, 65);
    ctx.strokeRect(73, 20, 7, 65);

    // Rolled Bedroll strapped across shoulders
    ctx.fillStyle = '#7a674e';
    ctx.strokeStyle = '#1c150e';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.roundRect(36, 12, 56, 16, 6);
    ctx.fill();
    ctx.stroke();

    this.drawCrossHatch(ctx, 42, 35, 44, 45, 6);

    const texture = this.createTexture(canvas);
    this.cache.set('player_torso_back', texture);
    return texture;
  }

  // Head seen from the side profile (Pointy gothic nose, profile eye, wild hair billowing back)
  public static getPlayerHeadSide(): THREE.CanvasTexture {
    if (this.cache.has('player_head_side')) return this.cache.get('player_head_side')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Wild spiky hair trailing backwards to the left
    ctx.fillStyle = '#181410';
    ctx.strokeStyle = '#050302';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(64, 80);
    ctx.quadraticCurveTo(20, 75, 12, 45);
    ctx.lineTo(28, 38);
    ctx.quadraticCurveTo(20, 15, 45, 12);
    ctx.lineTo(55, 24);
    ctx.quadraticCurveTo(70, 8, 82, 22);
    ctx.lineTo(76, 42);
    ctx.quadraticCurveTo(90, 52, 78, 70);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Pale Face Side Profile facing Right
    ctx.fillStyle = '#f0e6d6';
    ctx.beginPath();
    ctx.moveTo(64, 42);
    ctx.quadraticCurveTo(75, 42, 78, 54);
    // Brow ridge and pointy gothic nose
    ctx.lineTo(80, 62);
    ctx.lineTo(92, 69); // Pointy nose tip
    ctx.lineTo(82, 73);
    // Lip and chin
    ctx.lineTo(84, 80);
    ctx.lineTo(78, 88);
    ctx.lineTo(60, 88);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Large Anxious Profile Eye
    ctx.fillStyle = '#fffdfa';
    ctx.beginPath();
    ctx.ellipse(72, 60, 8, 12, 0.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Pupil
    ctx.fillStyle = '#110d0a';
    ctx.beginPath();
    ctx.arc(75, 61, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Highlight
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(74, 59, 1.2, 0, Math.PI * 2);
    ctx.fill();

    this.drawCrossHatch(ctx, 45, 35, 25, 45, 5);

    const texture = this.createTexture(canvas);
    this.cache.set('player_head_side', texture);
    return texture;
  }

  // Torso seen from the side profile
  public static getPlayerTorsoSide(): THREE.CanvasTexture {
    if (this.cache.has('player_torso_side')) return this.cache.get('player_torso_side')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    ctx.fillStyle = '#2c221a';
    ctx.strokeStyle = '#0a0705';
    ctx.lineWidth = 3;

    // Slim side profile silhouette
    ctx.beginPath();
    ctx.moveTo(50, 20);
    ctx.lineTo(76, 20);
    ctx.lineTo(80, 85);
    ctx.lineTo(46, 85);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Armhole seam
    ctx.strokeStyle = '#120d09';
    ctx.beginPath();
    ctx.arc(62, 34, 10, 0, Math.PI);
    ctx.stroke();

    this.drawCrossHatch(ctx, 50, 30, 24, 45, 5);

    const texture = this.createTexture(canvas);
    this.cache.set('player_torso_side', texture);
    return texture;
  }

  // Traveler Backpack Layer
  public static getPlayerBackpack(): THREE.CanvasTexture {
    if (this.cache.has('player_backpack')) return this.cache.get('player_backpack')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Sturdy leather satchel body
    ctx.fillStyle = '#5c3d23';
    ctx.strokeStyle = '#1a1008';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.roundRect(40, 35, 48, 50, 8);
    ctx.fill();
    ctx.stroke();

    // Flap
    ctx.fillStyle = '#472d17';
    ctx.beginPath();
    ctx.roundRect(38, 32, 52, 24, 6);
    ctx.fill();
    ctx.stroke();

    // Brass buckles
    ctx.fillStyle = '#d4aa50';
    ctx.fillRect(48, 50, 8, 8);
    ctx.fillRect(72, 50, 8, 8);

    // Bedroll on top
    ctx.fillStyle = '#7a6850';
    ctx.beginPath();
    ctx.roundRect(36, 18, 56, 16, 6);
    ctx.fill();
    ctx.stroke();

    const texture = this.createTexture(canvas);
    this.cache.set('player_backpack', texture);
    return texture;
  }

  // --- PROCEDURAL 2.5D NORMAL MAP FOR TREE LIGHTING ---
  public static getTreeNormalMap(): THREE.CanvasTexture {
    if (this.cache.has('tree_normal')) return this.cache.get('tree_normal')!;
    const { canvas, ctx } = this.createCanvas(256, 512);

    // Base flat normal [128, 128, 255] (pointing straight forward)
    ctx.fillStyle = 'rgb(128, 128, 255)';
    ctx.fillRect(0, 0, 256, 512);

    // Horizontal spherical bevel across trunk (left is red-shifted, right is cyan-shifted)
    const trunkGrad = ctx.createLinearGradient(96, 0, 160, 0);
    trunkGrad.addColorStop(0, 'rgb(60, 128, 220)');
    trunkGrad.addColorStop(0.5, 'rgb(128, 128, 255)');
    trunkGrad.addColorStop(1, 'rgb(196, 128, 220)');
    ctx.fillStyle = trunkGrad;
    ctx.fillRect(96, 200, 64, 312);

    // Foliage canopy convex bevel tiers
    const drawTierBevel = (cy: number, r: number) => {
      const grad = ctx.createRadialGradient(128, cy, 5, 128, cy, r);
      grad.addColorStop(0, 'rgb(128, 128, 255)');
      grad.addColorStop(0.7, 'rgb(110, 145, 235)');
      grad.addColorStop(1, 'rgb(80, 128, 200)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(128, cy, r, 0, Math.PI * 2);
      ctx.fill();
    };

    drawTierBevel(100, 80);
    drawTierBevel(190, 105);
    drawTierBevel(280, 120);

    const texture = this.createTexture(canvas);
    this.cache.set('tree_normal', texture);
    return texture;
  }

  // --- PROCEDURAL 2.5D NORMAL MAP FOR BOULDER LIGHTING ---
  public static getBoulderNormalMap(): THREE.CanvasTexture {
    if (this.cache.has('boulder_normal')) return this.cache.get('boulder_normal')!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    ctx.fillStyle = 'rgb(128, 128, 255)';
    ctx.fillRect(0, 0, 256, 256);

    // Spherical dome gradient
    const grad = ctx.createRadialGradient(128, 128, 10, 128, 128, 115);
    grad.addColorStop(0, 'rgb(128, 128, 255)');
    grad.addColorStop(0.5, 'rgb(140, 140, 240)');
    grad.addColorStop(0.85, 'rgb(160, 110, 210)');
    grad.addColorStop(1, 'rgb(80, 128, 180)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(128, 138, 105, 80, 0, 0, Math.PI * 2);
    ctx.fill();

    const texture = this.createTexture(canvas);
    this.cache.set('boulder_normal', texture);
    return texture;
  }

  // --- INSTANCED GROUND CLUTTER (PEBBLES, PINE NEEDLES, CLOVERS) ---
  public static getGroundClutterTexture(): THREE.CanvasTexture {
    if (this.cache.has('ground_clutter')) return this.cache.get('ground_clutter')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Tiny pebbles
    ctx.fillStyle = '#5c5447';
    ctx.strokeStyle = '#2b261f';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(40, 50, 8, 5, 0.2, 0, Math.PI * 2);
    ctx.ellipse(85, 75, 6, 4, -0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Fallen dry pine needles
    ctx.strokeStyle = '#7a4e28';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(25, 85); ctx.lineTo(45, 95);
    ctx.moveTo(35, 98); ctx.lineTo(55, 90);
    ctx.moveTo(70, 30); ctx.lineTo(95, 38);
    ctx.stroke();

    // Little wilderness clover sprig
    ctx.fillStyle = '#466935';
    ctx.beginPath();
    ctx.arc(60, 45, 4, 0, Math.PI * 2);
    ctx.arc(66, 45, 4, 0, Math.PI * 2);
    ctx.arc(63, 40, 4, 0, Math.PI * 2);
    ctx.fill();

    const texture = this.createTexture(canvas);
    this.cache.set('ground_clutter', texture);
    return texture;
  }

  // ==========================================
  // --- ROAD & HIGHWAY PROCEDURAL TEXTURES ---
  // ==========================================

  public static getRoadTexture(surface: string = 'asphalt'): THREE.CanvasTexture {
    const key = `road_${surface}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 512);

    if (surface === 'dirt' || surface === 'logging_track') {
      // Warm dirt road with wheel ruts
      ctx.fillStyle = '#453523';
      ctx.fillRect(0, 0, 256, 512);

      // Darker wheel ruts
      ctx.fillStyle = '#302315';
      ctx.fillRect(36, 0, 48, 512);
      ctx.fillRect(172, 0, 48, 512);

      // Pebble chips and dirt grain
      ctx.fillStyle = '#614d35';
      for (let i = 0; i < 70; i++) {
        const x = Math.random() * 256;
        const y = Math.random() * 512;
        ctx.beginPath();
        ctx.ellipse(x, y, 3, 2, Math.random(), 0, Math.PI * 2);
        ctx.fill();
      }

      // Grassy dirt fringe borders
      ctx.strokeStyle = '#27381d';
      ctx.lineWidth = 4;
      ctx.beginPath();
      for (let y = 0; y < 512; y += 12) {
        ctx.moveTo(0, y); ctx.lineTo(10 + Math.sin(y * 0.1) * 6, y);
        ctx.moveTo(256, y); ctx.lineTo(246 - Math.cos(y * 0.1) * 6, y);
      }
      ctx.stroke();

    } else if (surface === 'wood_plank' || surface === 'boardwalk_causeway') {
      // Squelchy Bog boardwalk wooden planks
      ctx.fillStyle = '#403020';
      ctx.fillRect(0, 0, 256, 512);

      const plankHeight = 32;
      for (let y = 0; y < 512; y += plankHeight) {
        // Plank body with subtle wood tone variation
        ctx.fillStyle = (y / plankHeight) % 2 === 0 ? '#54402d' : '#4d3926';
        ctx.fillRect(4, y + 2, 248, plankHeight - 4);

        // Dark gap between planks
        ctx.fillStyle = '#171109';
        ctx.fillRect(0, y + plankHeight - 2, 256, 2);

        // Rusty iron nails
        ctx.fillStyle = '#1c150e';
        ctx.beginPath();
        ctx.arc(18, y + plankHeight * 0.5, 2.5, 0, Math.PI * 2);
        ctx.arc(238, y + plankHeight * 0.5, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Outer guide beam runners
      ctx.fillStyle = '#2e2215';
      ctx.fillRect(0, 0, 6, 512);
      ctx.fillRect(250, 0, 6, 512);

    } else if (surface === 'ice' || surface === 'ice_road') {
      // Brrington Tundra slippery black ice road
      ctx.fillStyle = '#567f96';
      ctx.fillRect(0, 0, 256, 512);

      // Deep cyan ice core
      const iceGrad = ctx.createLinearGradient(0, 0, 256, 0);
      iceGrad.addColorStop(0, '#426880');
      iceGrad.addColorStop(0.5, '#76a8c4');
      iceGrad.addColorStop(1, '#426880');
      ctx.fillStyle = iceGrad;
      ctx.fillRect(8, 0, 240, 512);

      // White skate scratch lines and fracture cracks
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.lineWidth = 1.5;
      for (let i = 0; i < 25; i++) {
        const sx = 30 + Math.random() * 196;
        const sy = Math.random() * 512;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + (Math.random() - 0.5) * 40, sy + 30 + Math.random() * 50);
        ctx.stroke();
      }

      // Frosty edge snow fringes
      ctx.fillStyle = '#d6eef7';
      for (let y = 0; y < 512; y += 16) {
        ctx.beginPath();
        ctx.arc(4, y, 6 + Math.sin(y * 0.2) * 3, 0, Math.PI * 2);
        ctx.arc(252, y, 6 + Math.cos(y * 0.2) * 3, 0, Math.PI * 2);
        ctx.fill();
      }

    } else if (surface === 'cooled_lava' || surface === 'lava_bridge') {
      // Mount Grumbleguts cooled obsidian volcanic spans
      ctx.fillStyle = '#120d0e';
      ctx.fillRect(0, 0, 256, 512);

      // Glowing magma fissure vein down the road
      ctx.strokeStyle = '#ff3c00';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(128, 0);
      for (let y = 30; y < 512; y += 30) {
        ctx.lineTo(128 + Math.sin(y * 0.05) * 25, y);
      }
      ctx.stroke();

      // Hot core inner fissure glow
      ctx.strokeStyle = '#ffcc00';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(128, 0);
      for (let y = 30; y < 512; y += 30) {
        ctx.lineTo(128 + Math.sin(y * 0.05) * 25, y);
      }
      ctx.stroke();

      // Ember speckles
      ctx.fillStyle = '#ff8800';
      for (let i = 0; i < 40; i++) {
        const ex = Math.random() * 256;
        const ey = Math.random() * 512;
        ctx.fillRect(ex, ey, 2, 2);
      }

    } else {
      // Standard Asphalt Highway (Desert Interstate & Country Lane)
      ctx.fillStyle = '#272421';
      ctx.fillRect(0, 0, 256, 512);

      // Subtle asphalt grain and aggregate speckles
      ctx.fillStyle = '#3a3632';
      for (let i = 0; i < 120; i++) {
        const x = Math.random() * 256;
        const y = Math.random() * 512;
        ctx.fillRect(x, y, 2, 2);
      }

      // Yellow Dashed Center Line
      ctx.fillStyle = '#f0b830';
      const dashLength = 48;
      const dashGap = 32;
      for (let y = 10; y < 512; y += dashLength + dashGap) {
        ctx.fillRect(124, y, 8, dashLength);
      }

      // Solid White Shoulder Lines
      ctx.fillStyle = '#dedad4';
      ctx.fillRect(14, 0, 4, 512);
      ctx.fillRect(238, 0, 4, 512);

      // Weathered asphalt edge crumb lines
      ctx.strokeStyle = '#141210';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(0, 0); ctx.lineTo(0, 512);
      ctx.moveTo(256, 0); ctx.lineTo(256, 512);
      ctx.stroke();
    }

    const texture = this.createTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    this.cache.set(key, texture);
    return texture;
  }

  // --- ROAD PROPS: TRAFFIC CONE ---
  public static getTrafficConeTexture(): THREE.CanvasTexture {
    if (this.cache.has('prop_traffic_cone')) return this.cache.get('prop_traffic_cone')!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Black square base
    ctx.fillStyle = '#1c1b1a';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(64, 112, 38, 12, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Fluorescent Orange Cone
    ctx.fillStyle = '#f55716';
    ctx.beginPath();
    ctx.moveTo(64, 15);
    ctx.lineTo(88, 110);
    ctx.lineTo(40, 110);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // White reflective stripes
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(56, 45); ctx.lineTo(72, 45);
    ctx.lineTo(76, 62); ctx.lineTo(52, 62);
    ctx.closePath();
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(49, 75); ctx.lineTo(79, 75);
    ctx.lineTo(84, 92); ctx.lineTo(44, 92);
    ctx.closePath();
    ctx.fill();

    // Cross-hatch shade on left
    this.drawCrossHatch(ctx, 42, 30, 24, 75, 5);

    const texture = this.createTexture(canvas);
    this.cache.set('prop_traffic_cone', texture);
    return texture;
  }

  // --- ROAD PROPS: DIRECTION ROAD SIGN ---
  public static getRoadSignTexture(text: string = 'BRO MEADOW'): THREE.CanvasTexture {
    const key = `prop_road_sign_${text}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    // Weathered wooden post
    ctx.fillStyle = '#422f1d';
    ctx.strokeStyle = '#1c130a';
    ctx.lineWidth = 2;
    ctx.fillRect(60, 50, 8, 75);
    ctx.strokeRect(60, 50, 8, 75);

    // Yellow Highway Diamond or Green Signboard
    ctx.fillStyle = '#e6b52e';
    ctx.beginPath();
    ctx.moveTo(64, 10);
    ctx.lineTo(115, 50);
    ctx.lineTo(64, 90);
    ctx.lineTo(13, 50);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Inner black sign border
    ctx.strokeStyle = '#1f1b14';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(64, 16);
    ctx.lineTo(107, 50);
    ctx.lineTo(64, 84);
    ctx.lineTo(21, 50);
    ctx.closePath();
    ctx.stroke();

    // Arrow icon and comic text
    ctx.fillStyle = '#1c1811';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SPEED', 64, 46);
    ctx.font = '900 16px sans-serif';
    ctx.fillText('65', 64, 66);

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  // --- ROAD PROPS: BILLBOARD AD ---
  public static getBillboardTexture(brandIndex: number = 0): THREE.CanvasTexture {
    const key = `prop_billboard_${brandIndex}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 128);

    const ads = [
      { bg: '#b83b27', text: 'BRO-COLA', sub: 'FIZZ FOR SURVIVORS!', icon: '🥤' },
      { bg: '#3a6639', text: 'MUD PIES', sub: '100% SQUELCHY ORGANIC', icon: '🥧' },
      { bg: '#2b5282', text: 'HONK INSURANCE', sub: 'GOOSE DAMAGE COVERAGE', icon: '📢' },
    ];
    const ad = ads[brandIndex % ads.length];

    // Wooden billboard frame
    ctx.fillStyle = '#3d2b1b';
    ctx.fillRect(0, 0, 256, 128);

    // Poster canvas
    ctx.fillStyle = ad.bg;
    ctx.fillRect(8, 8, 240, 112);

    // Weathered paper borders & rivets
    ctx.strokeStyle = '#140d07';
    ctx.lineWidth = 3;
    ctx.strokeRect(8, 8, 240, 112);

    // Ad Text
    ctx.fillStyle = '#ffffff';
    ctx.font = '900 24px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(ad.text, 128, 52);

    ctx.fillStyle = '#ffd27d';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(ad.sub, 128, 80);

    // Decorative Icon
    ctx.font = '28px sans-serif';
    ctx.fillText(ad.icon, 35, 65);
    ctx.fillText(ad.icon, 220, 65);

    // Distressed scratch lines
    this.drawCrossHatch(ctx, 16, 16, 60, 90, 8);

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  // --- ROAD PROPS: STEEL GUARDRAIL ---
  public static getGuardrailTexture(): THREE.CanvasTexture {
    if (this.cache.has('prop_guardrail')) return this.cache.get('prop_guardrail')!;
    const { canvas, ctx } = this.createCanvas(128, 64);

    // Galvanized corrugated zinc steel
    ctx.fillStyle = '#828d94';
    ctx.fillRect(0, 0, 128, 64);

    // Corrugation horizontal ridges
    ctx.fillStyle = '#5f6970';
    ctx.fillRect(0, 0, 128, 12);
    ctx.fillRect(0, 24, 128, 14);
    ctx.fillRect(0, 52, 128, 12);

    ctx.fillStyle = '#a8b4bd';
    ctx.fillRect(0, 12, 128, 12);
    ctx.fillRect(0, 38, 128, 14);

    // Yellow reflective reflector bolt
    ctx.fillStyle = '#f5c842';
    ctx.beginPath();
    ctx.arc(64, 32, 5, 0, Math.PI * 2);
    ctx.fill();

    const texture = this.createTexture(canvas);
    this.cache.set('prop_guardrail', texture);
    return texture;
  }

  // --- ROAD PROPS: PARKED WRECK ---
  public static getParkedWreckTexture(): THREE.CanvasTexture {
    if (this.cache.has('prop_parked_wreck')) return this.cache.get('prop_parked_wreck')!;
    const { canvas, ctx } = this.createCanvas(256, 160);

    // Rusted sedan chassis side profile
    ctx.fillStyle = '#7a4224'; // Rust orange-brown
    ctx.strokeStyle = '#24140b';
    ctx.lineWidth = 3.5;

    ctx.beginPath();
    // Hood, windshield slope, roof, trunk
    ctx.moveTo(20, 125);
    ctx.lineTo(30, 95);
    ctx.lineTo(80, 90);
    ctx.lineTo(110, 45);
    ctx.lineTo(195, 45);
    ctx.lineTo(225, 90);
    ctx.lineTo(245, 95);
    ctx.lineTo(245, 125);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Shattered window openings
    ctx.fillStyle = '#1c1510';
    ctx.beginPath();
    ctx.moveTo(112, 50); ctx.lineTo(145, 50); ctx.lineTo(145, 85); ctx.lineTo(88, 85); ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(155, 50); ctx.lineTo(190, 50); ctx.lineTo(218, 85); ctx.lineTo(155, 85); ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Missing wheels sitting on cinderblocks
    ctx.fillStyle = '#5c544a';
    ctx.fillRect(55, 120, 28, 18);
    ctx.strokeRect(55, 120, 28, 18);
    ctx.fillRect(180, 120, 28, 18);
    ctx.strokeRect(180, 120, 28, 18);

    // Rust cross-hatch stipples & smoke soot
    this.drawCrossHatch(ctx, 40, 70, 70, 45, 6);
    this.drawCrossHatch(ctx, 160, 60, 60, 55, 6);

    const texture = this.createTexture(canvas);
    this.cache.set('prop_parked_wreck', texture);
    return texture;
  }

  // --- SETTLEMENT PROCEDURAL FOUNDATION & SIGN TEXTURES (PHASE Z2) ---

  public static getSettlementPadTexture(theme: string = 'town_pavement'): THREE.CanvasTexture {
    const key = `settlement_pad_${theme}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    if (theme === 'town_pavement') {
      // Concrete sidewalk slabs & street blocks
      ctx.fillStyle = '#2c2722';
      ctx.fillRect(0, 0, 256, 256);

      // Grid of concrete slab expansion joints
      ctx.strokeStyle = '#181410';
      ctx.lineWidth = 3;
      for (let x = 0; x <= 256; x += 64) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 256); ctx.stroke();
      }
      for (let y = 0; y <= 256; y += 64) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(256, y); ctx.stroke();
      }

      // Decorative manhole cover at center-left
      ctx.fillStyle = '#3a332a';
      ctx.strokeStyle = '#14100c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(64, 64, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(64, 64, 15, 0, Math.PI * 2);
      ctx.stroke();

      // Curb edge border highlight
      ctx.strokeStyle = '#5a4f42';
      ctx.lineWidth = 4;
      ctx.strokeRect(2, 2, 252, 252);
      this.drawCrossHatch(ctx, 30, 80, 50, 40, 5);
      this.drawCrossHatch(ctx, 150, 140, 70, 50, 6);

    } else if (theme === 'strip_parking') {
      // Highway diner & motel parking lot with stall lines & oil slicks
      ctx.fillStyle = '#201b17';
      ctx.fillRect(0, 0, 256, 256);

      // Yellow/White parking stall lines
      ctx.strokeStyle = '#e0aa53';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(32, 20); ctx.lineTo(32, 110);
      ctx.moveTo(96, 20); ctx.lineTo(96, 110);
      ctx.moveTo(160, 20); ctx.lineTo(160, 110);
      ctx.moveTo(224, 20); ctx.lineTo(224, 110);
      ctx.stroke();

      // Oil drips in parking stalls
      ctx.fillStyle = '#0f0c0a';
      ctx.beginPath();
      ctx.ellipse(64, 75, 14, 8, 0.2, 0, Math.PI * 2);
      ctx.ellipse(128, 70, 18, 10, -0.1, 0, Math.PI * 2);
      ctx.ellipse(192, 80, 12, 7, 0.3, 0, Math.PI * 2);
      ctx.fill();

    } else if (theme === 'village_plaza') {
      // Cobblestone market square
      ctx.fillStyle = '#34291f';
      ctx.fillRect(0, 0, 256, 256);

      ctx.strokeStyle = '#221912';
      ctx.lineWidth = 2;
      for (let y = 10; y < 250; y += 24) {
        const rowOffset = (Math.floor(y / 24) % 2) * 16;
        for (let x = 10; x < 250; x += 32) {
          ctx.fillStyle = (x + y) % 5 === 0 ? '#423528' : '#382c20';
          ctx.fillRect(x + rowOffset - 14, y - 10, 28, 20);
          ctx.strokeRect(x + rowOffset - 14, y - 10, 28, 20);
        }
      }

      // Mossy green spots between stones
      ctx.fillStyle = 'rgba(76, 112, 58, 0.35)';
      ctx.beginPath();
      ctx.arc(128, 128, 40, 0, Math.PI * 2);
      ctx.fill();

    } else if (theme === 'farm_dirt') {
      // Packed farmyard mud with tractor ruts
      ctx.fillStyle = '#3b2a1a';
      ctx.fillRect(0, 0, 256, 256);

      // Deep tractor wheel ruts
      ctx.fillStyle = '#26190f';
      ctx.fillRect(40, 0, 32, 256);
      ctx.fillRect(184, 0, 32, 256);

      // Tread chevron ridges
      ctx.strokeStyle = '#180f08';
      ctx.lineWidth = 3;
      for (let y = 0; y < 256; y += 20) {
        ctx.beginPath();
        ctx.moveTo(42, y); ctx.lineTo(70, y + 10);
        ctx.moveTo(186, y); ctx.lineTo(214, y + 10);
        ctx.stroke();
      }

      // Straw & dry grass debris
      ctx.strokeStyle = '#9c8147';
      ctx.lineWidth = 1.5;
      for (let i = 0; i < 30; i++) {
        const sx = 80 + (i * 17) % 90;
        const sy = (i * 29) % 250;
        ctx.beginPath();
        ctx.moveTo(sx, sy); ctx.lineTo(sx + 8, sy - 4);
        ctx.stroke();
      }

    } else if (theme === 'junkyard_scrap') {
      // Rusted diamond plate steel & gravel
      ctx.fillStyle = '#3a2416';
      ctx.fillRect(0, 0, 256, 256);

      // Steel plate panels
      ctx.strokeStyle = '#1d1007';
      ctx.lineWidth = 3;
      ctx.strokeRect(10, 10, 110, 110);
      ctx.strokeRect(135, 10, 110, 110);
      ctx.strokeRect(10, 135, 110, 110);
      ctx.strokeRect(135, 135, 110, 110);

      // Rivets along borders
      ctx.fillStyle = '#5c3a21';
      for (let p = 20; p < 240; p += 30) {
        ctx.beginPath(); ctx.arc(10, p, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(125, p, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(245, p, 3, 0, Math.PI * 2); ctx.fill();
      }
      this.drawCrossHatch(ctx, 20, 20, 80, 80, 8);
      this.drawCrossHatch(ctx, 145, 145, 80, 80, 8);

    } else {
      // outpost_concrete: Reinforced military slab with hazard stripes
      ctx.fillStyle = '#302d29';
      ctx.fillRect(0, 0, 256, 256);

      // Yellow/Black hazard stripe border (top and bottom)
      const stripeW = 20;
      for (let x = -stripeW; x < 256 + stripeW; x += stripeW * 2) {
        ctx.fillStyle = '#d4aa3b';
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x + stripeW, 0); ctx.lineTo(x + stripeW - 16, 24); ctx.lineTo(x - 16, 24); ctx.closePath(); ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x, 232); ctx.lineTo(x + stripeW, 232); ctx.lineTo(x + stripeW - 16, 256); ctx.lineTo(x - 16, 256); ctx.closePath(); ctx.fill();
      }

      ctx.strokeStyle = '#151310';
      ctx.lineWidth = 3;
      ctx.strokeRect(8, 26, 240, 204);
      this.drawCrossHatch(ctx, 40, 50, 80, 60, 6);
    }

    const texture = this.createTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    this.cache.set(key, texture);
    return texture;
  }

  public static getBuildingFoundationTexture(buildingType: string = 'house_small', condition: string = 'worn'): THREE.CanvasTexture {
    const key = `foundation_${buildingType}_${condition}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    // Weathered foundation slab floor
    ctx.fillStyle = condition === 'burnt' ? '#18120e' : (condition === 'barricaded' ? '#261f18' : '#33291f');
    ctx.fillRect(0, 0, 256, 256);

    // Wood floor planks
    ctx.strokeStyle = '#1e1610';
    ctx.lineWidth = 2;
    for (let y = 16; y < 256; y += 24) {
      ctx.beginPath(); ctx.moveTo(8, y); ctx.lineTo(248, y); ctx.stroke();
    }

    // Heavy foundation wall perimeter
    ctx.strokeStyle = condition === 'barricaded' ? '#8a6231' : '#4a3826';
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, 248, 248);

    // Doorway opening indicator (green rug / welcome mat at bottom)
    ctx.fillStyle = '#4a7536';
    ctx.fillRect(96, 236, 64, 18);
    ctx.strokeStyle = '#223816';
    ctx.lineWidth = 2;
    ctx.strokeRect(96, 236, 64, 18);

    // Room partition outlines
    ctx.strokeStyle = '#382918';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(128, 8); ctx.lineTo(128, 160);
    ctx.moveTo(8, 160); ctx.lineTo(248, 160);
    ctx.stroke();

    // Building type label stencil
    ctx.fillStyle = '#d4aa67';
    ctx.font = 'bold 15px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(buildingType.replace(/_/g, ' ').toUpperCase(), 128, 55);

    if (condition === 'burnt') {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
      ctx.beginPath();
      ctx.arc(90, 110, 45, 0, Math.PI * 2);
      ctx.arc(170, 180, 35, 0, Math.PI * 2);
      ctx.fill();
    } else if (condition === 'barricaded') {
      // Barricade tape / cross boards
      ctx.strokeStyle = '#c48e42';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(20, 20); ctx.lineTo(236, 236);
      ctx.moveTo(236, 20); ctx.lineTo(20, 236);
      ctx.stroke();
    }

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  public static getSettlementSignTexture(title: string = 'BRO-VILLE', subtitle: string = 'POP: 4 BROS'): THREE.CanvasTexture {
    const key = `settlement_sign_${title}_${subtitle}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 128);

    // Timber signboard background
    ctx.fillStyle = '#3a2717';
    ctx.fillRect(8, 8, 240, 112);

    // Wood grain lines
    ctx.strokeStyle = '#24170d';
    ctx.lineWidth = 2;
    for (let y = 20; y < 115; y += 18) {
      ctx.beginPath(); ctx.moveTo(12, y); ctx.lineTo(244, y); ctx.stroke();
    }

    // Heavy weathered iron border with corner bolt studs
    ctx.strokeStyle = '#6e5132';
    ctx.lineWidth = 6;
    ctx.strokeRect(10, 10, 236, 108);

    ctx.fillStyle = '#ffd685';
    ctx.beginPath();
    ctx.arc(18, 18, 4, 0, Math.PI * 2);
    ctx.arc(238, 18, 4, 0, Math.PI * 2);
    ctx.arc(18, 110, 4, 0, Math.PI * 2);
    ctx.arc(238, 110, 4, 0, Math.PI * 2);
    ctx.fill();

    // Sign Title
    ctx.fillStyle = '#fff0d1';
    ctx.font = 'bold 22px Cinzel, serif';
    ctx.textAlign = 'center';
    ctx.fillText(title.toUpperCase(), 128, 54);

    // Subtitle / Population / Warning
    ctx.fillStyle = '#d49b49';
    ctx.font = 'bold 12px monospace';
    ctx.fillText(subtitle.toUpperCase(), 128, 86);

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  // --- BUILDING WALLS, POP-UP ROOFS & HIDING SPOTS (PHASE Z3) ---

  public static getBuildingWallTexture(type: string = 'house_small', condition: string = 'worn', hasWindow: boolean = true): THREE.CanvasTexture {
    const key = `wall_${type}_${condition}_${hasWindow}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    // Wall base color by building category
    let wallColor = '#423325';
    let trimColor = '#241b12';
    if (type.includes('barn')) {
      wallColor = '#732626'; // Red barn
      trimColor = '#f0e6d2';
    } else if (type.includes('diner') || type.includes('motel')) {
      wallColor = '#4a3d31';
      trimColor = '#d4aa53';
    } else if (type.includes('supermarket') || type.includes('clinic')) {
      wallColor = '#3a3e42';
      trimColor = '#8fa3b0';
    } else if (type.includes('garage') || type.includes('shed')) {
      wallColor = '#38322c';
      trimColor = '#1d1916';
    }

    ctx.fillStyle = wallColor;
    ctx.fillRect(0, 0, 256, 256);

    // Horizontal clapboard / siding planks
    ctx.strokeStyle = '#18120c';
    ctx.lineWidth = 2;
    for (let y = 16; y < 256; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(256, y); ctx.stroke();
    }

    // Heavy vertical timber studs on edges
    ctx.fillStyle = trimColor;
    ctx.fillRect(0, 0, 16, 256);
    ctx.fillRect(240, 0, 16, 256);
    ctx.strokeStyle = '#0f0b07';
    ctx.lineWidth = 3;
    ctx.strokeRect(0, 0, 16, 256);
    ctx.strokeRect(240, 0, 16, 256);

    if (hasWindow) {
      // Centered Window Frame with cross-panes
      const wx = 80;
      const wy = 50;
      const ww = 96;
      const wh = 100;

      // Outer window trim
      ctx.fillStyle = trimColor;
      ctx.fillRect(wx - 6, wy - 6, ww + 12, wh + 12);
      ctx.strokeRect(wx - 6, wy - 6, ww + 12, wh + 12);

      // Glass panes (night/dark tinted glass with reflection diagonal)
      ctx.fillStyle = '#1c2630';
      ctx.fillRect(wx, wy, ww, wh);
      ctx.strokeRect(wx, wy, ww, wh);

      // Glass highlight streak
      ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
      ctx.beginPath();
      ctx.moveTo(wx + 10, wy); ctx.lineTo(wx + 45, wy); ctx.lineTo(wx + 20, wy + wh); ctx.lineTo(wx, wy + wh); ctx.closePath();
      ctx.fill();

      // Window cross muntins
      ctx.strokeStyle = trimColor;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(wx + ww / 2, wy); ctx.lineTo(wx + ww / 2, wy + wh);
      ctx.moveTo(wx, wy + wh / 2); ctx.lineTo(wx + ww, wy + wh / 2);
      ctx.stroke();

      // Weathered window sill
      ctx.fillStyle = '#1a130d';
      ctx.fillRect(wx - 10, wy + wh + 4, ww + 20, 8);
      ctx.strokeRect(wx - 10, wy + wh + 4, ww + 20, 8);
    }

    // Slapstick wear, cracks, cross-hatching
    this.drawCrossHatch(ctx, 25, 170, 70, 60, 6);
    this.drawCrossHatch(ctx, 160, 165, 75, 65, 6);

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  public static getBuildingRoofTexture(type: string = 'house_small', condition: string = 'worn'): THREE.CanvasTexture {
    const key = `roof_${type}_${condition}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(256, 256);

    // Roof color palette
    let roofColor = '#3a2b1f';
    if (type.includes('barn')) roofColor = '#4a251e';
    else if (type.includes('diner') || type.includes('gas_station')) roofColor = '#242a30';
    else if (type.includes('clinic') || type.includes('town_hall')) roofColor = '#2d332d';

    ctx.fillStyle = roofColor;
    ctx.fillRect(0, 0, 256, 256);

    // Overlapping shingle rows with scalloped/weathered edges
    ctx.strokeStyle = '#140e0a';
    ctx.lineWidth = 2.5;
    for (let y = 14; y < 256; y += 22) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(256, y);
      ctx.stroke();

      // Staggered vertical shingle seams
      const offset = (Math.floor(y / 22) % 2) * 16;
      for (let x = offset; x < 256; x += 32) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + 22);
        ctx.stroke();
      }
    }

    // Heavy cardboard fold crease ridge along top
    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.fillRect(0, 0, 256, 8);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    ctx.fillRect(0, 248, 256, 8);

    // Cardboard cross-hatch stippling
    this.drawCrossHatch(ctx, 30, 40, 80, 70, 7);
    this.drawCrossHatch(ctx, 140, 130, 90, 80, 7);

    const texture = this.createTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    this.cache.set(key, texture);
    return texture;
  }

  public static getHidingSpotTexture(spotType: 'wardrobe' | 'under_bed' | 'dumpster', isPeeking: boolean = false): THREE.CanvasTexture {
    const key = `hiding_${spotType}_${isPeeking}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    if (spotType === 'wardrobe') {
      // Weathered wooden wardrobe cabinet
      ctx.fillStyle = '#422e1d';
      ctx.fillRect(10, 8, 108, 112);
      ctx.strokeStyle = '#1c1209';
      ctx.lineWidth = 3;
      ctx.strokeRect(10, 8, 108, 112);

      // Double door panels
      ctx.strokeRect(16, 14, 44, 98);
      ctx.strokeRect(68, 14, 44, 98);

      // Door crack in center
      ctx.fillStyle = '#0f0a06';
      ctx.fillRect(62, 10, isPeeking ? 12 : 4, 106);

      // Brass door handles
      ctx.fillStyle = '#d4aa53';
      ctx.beginPath(); ctx.arc(56, 64, 4, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(72, 64, 4, 0, Math.PI * 2); ctx.fill();

      if (isPeeking) {
        // Comical wide cartoon eyes peeking out from the door crack!
        ctx.fillStyle = '#fff4d1';
        ctx.beginPath(); ctx.ellipse(64, 52, 5, 8, 0, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.ellipse(72, 52, 5, 8, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#120d08';
        ctx.beginPath(); ctx.arc(65, 52, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(73, 52, 3, 0, Math.PI * 2); ctx.fill();
      }

    } else if (spotType === 'under_bed') {
      // Cozy wooden bed with dust ruffle
      ctx.fillStyle = '#3a2717';
      ctx.fillRect(8, 20, 112, 40); // Mattress / headboard
      ctx.strokeStyle = '#180f08';
      ctx.lineWidth = 3;
      ctx.strokeRect(8, 20, 112, 40);

      // Quilted bedspread
      ctx.fillStyle = '#7a3e3e';
      ctx.fillRect(12, 24, 104, 32);

      // Dark under-bed shadow void
      ctx.fillStyle = '#0a0705';
      ctx.fillRect(14, 60, 100, 52);
      ctx.strokeRect(14, 60, 100, 52);

      // Hanging dust ruffle curtain
      ctx.strokeStyle = '#c4b293';
      ctx.lineWidth = 2.5;
      for (let x = 18; x < 110; x += 14) {
        ctx.beginPath();
        ctx.arc(x, 60, 7, 0, Math.PI);
        ctx.stroke();
      }

      if (isPeeking) {
        // Cartoon eyes peeking from under the bed ruffle!
        ctx.fillStyle = '#fff4d1';
        ctx.beginPath(); ctx.ellipse(56, 80, 6, 9, -0.1, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.ellipse(72, 80, 6, 9, 0.1, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#0f0a06';
        ctx.beginPath(); ctx.arc(57, 80, 3.5, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(73, 80, 3.5, 0, Math.PI * 2); ctx.fill();
      }

    } else {
      // dumpster: Olive-drab industrial dumpster with cartoon graffiti
      ctx.fillStyle = '#384232';
      ctx.fillRect(12, 24, 104, 88);
      ctx.strokeStyle = '#181e15';
      ctx.lineWidth = 3;
      ctx.strokeRect(12, 24, 104, 88);

      // Steel reinforcement ribs
      ctx.beginPath();
      ctx.moveTo(38, 24); ctx.lineTo(38, 112);
      ctx.moveTo(64, 24); ctx.lineTo(64, 112);
      ctx.moveTo(90, 24); ctx.lineTo(90, 112);
      ctx.stroke();

      // Slanted lid (cracked open when peeking)
      ctx.fillStyle = '#222920';
      if (isPeeking) {
        ctx.beginPath();
        ctx.moveTo(8, 22); ctx.lineTo(120, 10); ctx.lineTo(122, 18); ctx.lineTo(8, 30); ctx.closePath();
        ctx.fill(); ctx.stroke();

        // Dark opening slit
        ctx.fillStyle = '#080c07';
        ctx.beginPath();
        ctx.moveTo(12, 30); ctx.lineTo(116, 20); ctx.lineTo(116, 38); ctx.lineTo(12, 38); ctx.closePath();
        ctx.fill();

        // Eyes peering out of dumpster crack!
        ctx.fillStyle = '#fff4d1';
        ctx.beginPath(); ctx.ellipse(56, 30, 5, 6, 0, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.ellipse(68, 28, 5, 6, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#0a0d08';
        ctx.beginPath(); ctx.arc(57, 30, 2.5, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(69, 28, 2.5, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.fillRect(8, 18, 112, 10);
        ctx.strokeRect(8, 18, 112, 10);
      }

      // Yellow cartoon biohazard / bro graffiti
      ctx.fillStyle = '#d4aa3b';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('NO ZOMBS', 28, 70);
    }

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }

  /**
   * Phase Z4: Procedural diorama textures for containers.
   */
  public static getContainerTexture(containerId: string): THREE.CanvasTexture {
    const key = `container_${containerId}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const { canvas, ctx } = this.createCanvas(128, 128);

    ctx.save();
    switch (containerId) {
      case 'toolbox': {
        // Red mechanic toolbox
        ctx.fillStyle = '#A31D1D';
        ctx.fillRect(8, 8, 112, 112);
        ctx.strokeStyle = '#380E0E';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(8, 8, 112, 112);
        // Bevel crease
        ctx.fillStyle = '#C23B3B';
        ctx.fillRect(12, 12, 104, 30);
        // Chrome latches
        ctx.fillStyle = '#E0E0E0';
        ctx.strokeStyle = '#2B2B2B';
        ctx.lineWidth = 2;
        ctx.fillRect(32, 38, 14, 22);
        ctx.strokeRect(32, 38, 14, 22);
        ctx.fillRect(82, 38, 14, 22);
        ctx.strokeRect(82, 38, 14, 22);
        // Black handle
        ctx.fillStyle = '#1A1A1A';
        ctx.beginPath();
        ctx.roundRect(42, 16, 44, 10, 4);
        ctx.fill();
        ctx.stroke();
        break;
      }
      case 'cash_register': {
        // Antique brass/teal cash register
        ctx.fillStyle = '#2C534E';
        ctx.fillRect(8, 8, 112, 112);
        ctx.strokeStyle = '#122623';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(8, 8, 112, 112);
        // Brass pop-up price window
        ctx.fillStyle = '#F4D06F';
        ctx.fillRect(20, 14, 88, 24);
        ctx.strokeStyle = '#634710';
        ctx.lineWidth = 2;
        ctx.strokeRect(20, 14, 88, 24);
        ctx.fillStyle = '#1A1A1A';
        ctx.font = 'bold 13px monospace';
        ctx.fillText('$99.99', 36, 31);
        // Push buttons grid
        ctx.fillStyle = '#E8E8E8';
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 4; c++) {
            ctx.beginPath();
            ctx.arc(28 + c * 24, 52 + r * 14, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }
        // Cash drawer bottom panel
        ctx.fillStyle = '#1B3834';
        ctx.fillRect(14, 92, 100, 24);
        ctx.strokeRect(14, 92, 100, 24);
        ctx.fillStyle = '#F4D06F';
        ctx.beginPath();
        ctx.arc(64, 104, 4, 0, Math.PI * 2);
        ctx.fill();
        break;
      }
      case 'fridge': {
        // Retro 50s rounded cream/mint refrigerator
        ctx.fillStyle = '#A2B9AF';
        ctx.fillRect(10, 8, 108, 112);
        ctx.strokeStyle = '#273832';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(10, 8, 108, 112);
        // Freezer split line
        ctx.strokeStyle = '#273832';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(10, 44);
        ctx.lineTo(118, 44);
        ctx.stroke();
        // Chrome vertical latch handle
        ctx.fillStyle = '#F0F0F0';
        ctx.fillRect(94, 52, 10, 36);
        ctx.strokeRect(94, 52, 10, 36);
        ctx.fillStyle = '#C0C0C0';
        ctx.fillRect(96, 56, 6, 28);
        // Retro emblem badge
        ctx.fillStyle = '#C43D3D';
        ctx.beginPath();
        ctx.arc(64, 26, 7, 0, Math.PI * 2);
        ctx.fill();
        break;
      }
      case 'locker': {
        // Olive-grey steel locker
        ctx.fillStyle = '#49524A';
        ctx.fillRect(12, 6, 104, 116);
        ctx.strokeStyle = '#1D241E';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(12, 6, 104, 116);
        // Ventilation louvers
        ctx.strokeStyle = '#1D241E';
        ctx.lineWidth = 2.5;
        for (let i = 0; i < 5; i++) {
          ctx.beginPath();
          ctx.moveTo(34, 18 + i * 7);
          ctx.lineTo(94, 18 + i * 7);
          ctx.stroke();
        }
        // Padlock hasp
        ctx.fillStyle = '#D4AF37';
        ctx.fillRect(86, 60, 10, 16);
        ctx.strokeRect(86, 60, 10, 16);
        // Lower louvers
        for (let i = 0; i < 4; i++) {
          ctx.beginPath();
          ctx.moveTo(34, 90 + i * 7);
          ctx.lineTo(94, 90 + i * 7);
          ctx.stroke();
        }
        break;
      }
      case 'vault_safe': {
        // Titanium heavy vault safe
        ctx.fillStyle = '#23272D';
        ctx.fillRect(8, 8, 112, 112);
        ctx.strokeStyle = '#0F1216';
        ctx.lineWidth = 4;
        ctx.strokeRect(8, 8, 112, 112);
        // Massive circular spoke dial in center
        ctx.fillStyle = '#8B939C';
        ctx.beginPath();
        ctx.arc(64, 64, 32, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0F1216';
        ctx.stroke();
        // Inner wheel
        ctx.fillStyle = '#D9E0E8';
        ctx.beginPath();
        ctx.arc(64, 64, 16, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Dial spokes
        ctx.strokeStyle = '#0F1216';
        ctx.lineWidth = 3;
        for (let a = 0; a < 4; a++) {
          const angle = (a * Math.PI) / 2;
          ctx.beginPath();
          ctx.moveTo(64 + Math.cos(angle) * 16, 64 + Math.sin(angle) * 16);
          ctx.lineTo(64 + Math.cos(angle) * 36, 64 + Math.sin(angle) * 36);
          ctx.stroke();
        }
        // Corner heavy steel rivets
        ctx.fillStyle = '#A4ADB6';
        const corners = [
          [16, 16],
          [112, 16],
          [16, 112],
          [112, 112],
        ];
        corners.forEach(([rx, ry]) => {
          ctx.beginPath();
          ctx.arc(rx, ry, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        });
        break;
      }
      case 'supply_drop_crate': {
        // Military olive wooden supply crate with hazard stripes
        ctx.fillStyle = '#4E593E';
        ctx.fillRect(8, 8, 112, 112);
        ctx.strokeStyle = '#1D2415';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(8, 8, 112, 112);
        // Yellow/Black diagonal hazard stripes along top & bottom
        ctx.fillStyle = '#E8BC38';
        ctx.fillRect(12, 12, 104, 16);
        ctx.fillRect(12, 100, 104, 16);
        ctx.fillStyle = '#1D2415';
        for (let i = 0; i < 6; i++) {
          ctx.beginPath();
          ctx.moveTo(16 + i * 18, 12);
          ctx.lineTo(26 + i * 18, 28);
          ctx.lineTo(20 + i * 18, 28);
          ctx.lineTo(10 + i * 18, 12);
          ctx.fill();
        }
        // Stenciled white military markings
        ctx.fillStyle = '#E0E8D0';
        ctx.font = 'bold 15px sans-serif';
        ctx.fillText('SUPPLY', 32, 60);
        ctx.font = 'bold 11px monospace';
        ctx.fillText('CRATE-4B', 34, 76);
        break;
      }
      case 'vending_machine': {
        // Illuminated snack/soda dispenser
        ctx.fillStyle = '#192033';
        ctx.fillRect(8, 6, 112, 116);
        ctx.strokeStyle = '#0B0F1A';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(8, 6, 112, 116);
        // Glass display window with vibrant cans
        ctx.fillStyle = '#163B4D';
        ctx.fillRect(16, 14, 76, 72);
        ctx.strokeStyle = '#2DE2E6';
        ctx.lineWidth = 2;
        ctx.strokeRect(16, 14, 76, 72);
        // Rows of colorful cans
        const colors = ['#FF3366', '#33CCFF', '#FFCC00', '#33FF99'];
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 4; c++) {
            ctx.fillStyle = colors[(r + c) % colors.length];
            ctx.fillRect(22 + c * 16, 20 + r * 22, 10, 16);
          }
        }
        // Keypad & Coin insert
        ctx.fillStyle = '#E0E0E0';
        ctx.fillRect(98, 24, 14, 30);
        // Push door at bottom
        ctx.fillStyle = '#0F1420';
        ctx.fillRect(16, 94, 96, 20);
        ctx.strokeRect(16, 94, 96, 20);
        break;
      }
      case 'pinata': {
        // Rainbow party striped paper fringe
        const stripes = ['#FF477E', '#FF85A1', '#FFD166', '#06D6A0', '#118AB2', '#9B5DE5'];
        const stripeH = 112 / stripes.length;
        for (let i = 0; i < stripes.length; i++) {
          ctx.fillStyle = stripes[i];
          ctx.fillRect(8, 8 + i * stripeH, 112, stripeH);
        }
        ctx.strokeStyle = '#380E1C';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(8, 8, 112, 112);
        // Festive zig-zag tassels
        ctx.fillStyle = '#FFFFFF';
        for (let i = 0; i < 8; i++) {
          ctx.beginPath();
          ctx.arc(18 + i * 13, 64, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        break;
      }
      case 'suitcase': {
        // Warm brown leather traveler luggage
        ctx.fillStyle = '#6E482B';
        ctx.fillRect(10, 16, 108, 96);
        ctx.strokeStyle = '#27170B';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(10, 16, 108, 96);
        // Brass corner protectors
        ctx.fillStyle = '#D4AF37';
        const corners = [
          [10, 16],
          [108, 16],
          [10, 102],
          [108, 102],
        ];
        corners.forEach(([cx, cy]) => {
          ctx.fillRect(cx, cy, 10, 10);
        });
        // Twin leather buckle straps
        ctx.fillStyle = '#422513';
        ctx.fillRect(36, 16, 12, 96);
        ctx.fillRect(80, 16, 12, 96);
        ctx.fillStyle = '#D4AF37';
        ctx.fillRect(34, 58, 16, 12);
        ctx.fillRect(78, 58, 16, 12);
        // Handle
        ctx.fillStyle = '#27170B';
        ctx.beginPath();
        ctx.arc(64, 16, 10, Math.PI, 0);
        ctx.fill();
        break;
      }
      case 'medicine_cabinet': {
        // Clean white enamel first-aid wall cabinet
        ctx.fillStyle = '#EAEFEF';
        ctx.fillRect(12, 10, 104, 108);
        ctx.strokeStyle = '#232D2D';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(12, 10, 104, 108);
        // Mirrored door border
        ctx.fillStyle = '#D0DFDE';
        ctx.fillRect(20, 18, 88, 92);
        ctx.strokeRect(20, 18, 88, 92);
        // Bold medical red cross
        ctx.fillStyle = '#D92534';
        ctx.fillRect(56, 38, 16, 52);
        ctx.fillRect(38, 56, 52, 16);
        break;
      }
      default: {
        // Generic wooden box
        ctx.fillStyle = '#8B5A2B';
        ctx.fillRect(8, 8, 112, 112);
        ctx.strokeStyle = '#38200B';
        ctx.lineWidth = 3.5;
        ctx.strokeRect(8, 8, 112, 112);
        ctx.strokeRect(18, 18, 92, 92);
        break;
      }
    }
    ctx.restore();

    const texture = this.createTexture(canvas);
    this.cache.set(key, texture);
    return texture;
  }
}


