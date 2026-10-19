import type { G } from './ink';
import { INK, TAU, ell, cap, poly, lumpy, smooth, stroke, shade, alpha } from './ink';
import type { PetC } from '../sim/components';
import { makeRng, type Rng } from '../engine/rng';
import { SpriteCache } from './sprites';
import { type CreatureLook, type RigPart, coat, eye, plain, drawRig } from './animalRenderer';

/** Colours and ear/tail style for one pet breed. Add a breed here and a key in PET_LOOKS. */
export interface PetBreed { fur: string; belly: string; ears: 'pointy' | 'floppy'; tail: 'bushy' | 'whip'; mask?: string; collar: string }

export const PET_BREEDS: Record<PetC['kind'], PetBreed> = {
  wolf_pup:    { fur: '#968c80', belly: '#e6dfd2', ears: 'pointy', tail: 'bushy', mask: '#6a625a', collar: '#c4473d' },
  hunting_dog: { fur: '#b67f4e', belly: '#ecd2b0', ears: 'floppy', tail: 'whip', collar: '#3d7ac4' },
};

const part = (key: string, box: [number, number, number, number], draw: (g: G, rng: Rng) => void): RigPart => ({ key, box, draw });

/** Builds a pet rig from a breed: fur-textured body with belly blaze and collar, expressive head, wagging tail. */
export function petLook(id: string, b: PetBreed): CreatureLook {
  return {
    anim: 'pup', shadow: [0.22, 0.16],
    legs: { at: [[-0.1, -0.08], [-0.1, 0.08], [0.09, -0.08], [0.09, 0.08]], part: part(`pet:${id}:leg`, [-0.06, -0.05, 0.1, 0.05], (g, r) => { coat(g, r, () => cap(g, -0.02, 0, 0.05, 0, 0.03, 0.026), b.fur, 'soft', { box: 0.06, line: 0.013 }); plain(g, r, () => ell(g, 0.06, 0, 0.022, 0.026), b.belly, 0.011); }) },
    tail: { at: [-0.15, 0], part: part(`pet:${id}:tail`, [-0.24, -0.08, 0.03, 0.08], (g, r) => b.tail === 'bushy'
      ? coat(g, r, () => smooth(g, [[0.01, -0.025], [-0.08, -0.05], [-0.19, -0.03], [-0.21, 0], [-0.19, 0.03], [-0.08, 0.05], [0.01, 0.025]]), b.fur, 'soft', { box: 0.2, line: 0.014 })
      : coat(g, r, () => cap(g, 0, 0, -0.17, 0, 0.022, 0.01), b.fur, 'fur', { box: 0.18, line: 0.012 })) },
    body: part(`pet:${id}:body`, [-0.22, -0.16, 0.22, 0.16], (g, r) => {
      const body = (): void => lumpy(g, 0, 0, 0.18, 0.12, makeRng(41), 10, 0.04);
      coat(g, r, body, b.fur, 'soft', { box: 0.2, line: 0.018 });
      g.save(); g.beginPath(); body(); g.clip();
      if (b.mask) { g.fillStyle = alpha(b.mask, 0.6); g.beginPath(); g.ellipse(-0.04, 0, 0.12, 0.05, 0, 0, TAU); g.fill(); }   // saddle
      g.fillStyle = alpha(b.belly, 0.75); for (const s of [-1, 1]) { g.beginPath(); g.ellipse(0.02, s * 0.11, 0.13, 0.035, 0, 0, TAU); g.fill(); }
      g.restore();
      plain(g, r, () => ell(g, 0.12, 0, 0.03, 0.095), b.collar, 0.012);
      plain(g, r, () => ell(g, 0.15, 0.0, 0.018, 0.018), '#f2c94c', 0.008);                 // name tag
    }),
    head: { at: [0.16, 0], part: part(`pet:${id}:head`, [-0.08, -0.17, 0.2, 0.17], (g, r) => {
      for (const s of [-1, 1]) {
        if (b.ears === 'pointy') { coat(g, r, () => poly(g, [[0.0, s * 0.045], [-0.07, s * 0.13], [0.06, s * 0.085]]), shade(b.fur, -0.12), 'soft', { box: 0.1, line: 0.013 }); g.fillStyle = alpha('#e8b0a8', 0.7); g.beginPath(); poly(g, [[0.005, s * 0.06], [-0.045, s * 0.11], [0.04, s * 0.08]]); g.fill(); }
        else coat(g, r, () => ell(g, 0.0, s * 0.085, 0.06, 0.03, s * 0.6), shade(b.fur, -0.25), 'soft', { box: 0.08, line: 0.013 });
      }
      coat(g, r, () => ell(g, 0.02, 0, 0.088, 0.08), b.fur, 'soft', { box: 0.1, line: 0.018 });
      coat(g, r, () => smooth(g, [[0.05, -0.04], [0.14, -0.025], [0.155, 0], [0.14, 0.025], [0.05, 0.04]]), b.belly, 'none', { box: 0.1, line: 0.014 });
      plain(g, r, () => ell(g, 0.15, 0, 0.018, 0.022), INK, 0.008);
      g.fillStyle = alpha('#ffffff', 0.7); g.beginPath(); g.arc(0.153, -0.008, 0.006, 0, TAU); g.fill();
      for (const s of [-1, 1]) { eye(g, 0.05, s * 0.038, 0.016); stroke(g, [[0.035, s * 0.06], [0.065, s * 0.062]], 0.006, INK, 0.5); }   // eyes + brows
    }) },
  };
}

export const PET_LOOKS: Record<PetC['kind'], CreatureLook> = {
  wolf_pup: petLook('wolf_pup', PET_BREEDS.wolf_pup),
  hunting_dog: petLook('hunting_dog', PET_BREEDS.hunting_dog),
};

/** Renders companion pets on the shared creature rig: fur coats, fast happy wag, sit pose, rim light. */
export class PetRenderer {
  private cache = new SpriteCache();

  draw(g: G, pet: PetC, x: number, y: number, angle: number, dt: number): void {
    pet.animPhase += dt * (pet.state === 'follow' ? 5.0 : pet.state === 'attack' ? 7.0 : 2.0);
    pet.tailWag += dt * (pet.state === 'sit' ? 14.0 : 10.0);   // sitting pets wag hardest

    g.save();
    g.translate(x, y);
    g.rotate(angle);
    drawRig(g, this.cache, PET_LOOKS[pet.kind] ?? PET_LOOKS.hunting_dog, { phase: pet.animPhase, tail: pet.tailWag, move: pet.state === 'sit' ? 0 : pet.state === 'attack' ? 1 : 0.8, graze: false, alarm: pet.state === 'attack', sit: pet.state === 'sit' });
    g.restore();

    // Companion name label overhead
    g.save();
    g.font = '0.22px "Crimson Pro", Georgia, serif';
    g.textAlign = 'center';
    g.fillStyle = 'rgba(249,196,90,0.95)';
    g.strokeStyle = 'rgba(10,12,16,0.85)';
    g.lineWidth = 0.04;
    g.strokeText(pet.name, x, y - 0.32);
    g.fillText(pet.name, x, y - 0.32);
    g.restore();
  }
}
