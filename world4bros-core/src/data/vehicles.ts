import type { VehicleDef, Id } from './schema';

/** Decal painters live in src/render/vehicleRenderer.ts (DECALS registry), keyed by these names. */
export type VehicleDecal = 'none' | 'stripes' | 'flames' | 'number' | 'stars' | 'pinstripe' | 'bolt' | 'ducks';

/** Render-only paint job for a vehicle. Every field is optional; the renderer falls back to DEFAULT_VEHICLE_LOOK. */
export interface VehicleLook {
  /** Chrome trim, bumpers and hubcaps. */
  trim?: string;
  /** Window tint (glass is drawn darker, with sky reflections on top). */
  glass?: string;
  /** Main decal colour and an optional second colour. */
  accent?: string;
  accent2?: string;
  decal?: VehicleDecal;
  /** Racing number for the 'number' decal. */
  number?: number;
  /** Baseline road dirt 0..1, before any damage. */
  grime?: number;
  /** Baseline rust 0..1, before any damage. Dents and rust grow as hp drops. */
  rust?: number;
  /** Dirt and rust tones. */
  dirt?: string;
  rustTone?: string;
}

export const DEFAULT_VEHICLE_LOOK: Required<VehicleLook> = {
  trim: '#c3ccd4', glass: '#25323d', accent: '#efe8d6', accent2: '#15171b', decal: 'none', number: 4, grime: 0.25, rust: 0,
  dirt: '#5b4a36', rustTone: '#9a4a22',
};

/** Paint jobs keyed by vehicle id. Add a vehicle here to give it trim, glass tint and a decal. */
export const VEHICLE_LOOKS: Record<Id, VehicleLook> = {
  rusty_hatchback: { trim: '#b9c1c8', glass: '#26343f', accent: '#e9dcb8', accent2: '#5a2a1c', decal: 'stripes', grime: 0.5, rust: 0.45 },
  clown_car: { trim: '#e7c35a', glass: '#1e293b', accent: '#facc15', accent2: '#38bdf8', decal: 'stars', grime: 0.15 },
  ice_cream_truck: { trim: '#d3dbe2', glass: '#223247', accent: '#f472b6', accent2: '#7dd3fc', decal: 'pinstripe', grime: 0.3 },
  golf_cart: { trim: '#d8dee4', glass: '#bae6fd', accent: '#f5f5f4', accent2: '#15803d', decal: 'number', number: 18, grime: 0.35 },
  bathtub_buggy: { trim: '#d6dde3', glass: '#7dd3fc', accent: '#facc15', accent2: '#f97316', decal: 'ducks', grime: 0.2 },
  hot_dog_van: { trim: '#d8dee4', glass: '#1e293b', accent: '#facc15', accent2: '#dc2626', decal: 'flames', grime: 0.35, rust: 0.15 },
  tesla_roadster: { trim: '#e2e8f0', glass: '#0f172a', accent: '#38bdf8', accent2: '#f8fafc', decal: 'bolt', grime: 0.05 },
};

export function vehicleLook(id: Id): Required<VehicleLook> {
  return { ...DEFAULT_VEHICLE_LOOK, ...(VEHICLE_LOOKS[id] ?? {}) };
}

export const VEHICLES: VehicleDef[] = [
  {
    id: 'rusty_hatchback',
    name: 'Rusty Hatchback',
    kind: 'hatchback',
    size: [2.2, 1.2],
    maxSpeed: 7.8,
    accel: 5.5,
    turnSpeed: 2.2,
    maxHp: 100,
    maxFuel: 100,
    trunkSlots: 12,
    lightRadius: 10,
    color: '#8a4a34',
    quirk: 'Reliable scout car. Roof rack holds luggage and gear.',
  },
  {
    id: 'clown_car',
    name: 'Clown Car of Wonders',
    kind: 'clown_car',
    size: [1.7, 1.1],
    maxSpeed: 9.2,
    accel: 7.8,
    turnSpeed: 3.0,
    maxHp: 80,
    maxFuel: 80,
    trunkSlots: 6,
    lightRadius: 9,
    color: '#d92b3a',
    quirk: 'Ultra zippy & nimble. Tiny turning radius and springy suspension.',
  },
  {
    id: 'ice_cream_truck',
    name: 'Chilly Chimes Ice Cream Truck',
    kind: 'ice_cream_truck',
    size: [2.8, 1.4],
    maxSpeed: 6.4,
    accel: 4.2,
    turnSpeed: 1.6,
    maxHp: 170,
    maxFuel: 140,
    trunkSlots: 24,
    lightRadius: 12,
    color: '#d8e9f0',
    quirk: 'Heavy rolling fortress with massive 24-slot freezer trunk.',
  },
  {
    id: 'golf_cart',
    name: 'Golf Cart Deluxe',
    kind: 'golf_cart',
    size: [1.8, 1.05],
    maxSpeed: 6.0,
    accel: 6.2,
    turnSpeed: 2.6,
    maxHp: 65,
    maxFuel: 75,
    trunkSlots: 8,
    lightRadius: 8,
    color: '#3a5a40',
    quirk: 'Whisper-quiet electric drive. Draws almost zero zombie attention.',
  },
  {
    id: 'bathtub_buggy',
    name: 'Bathtub Buggy',
    kind: 'bathtub_buggy',
    size: [2.1, 1.15],
    maxSpeed: 8.4,
    accel: 7.0,
    turnSpeed: 2.4,
    maxHp: 90,
    maxFuel: 90,
    trunkSlots: 6,
    lightRadius: 9,
    color: '#eef1f4',
    quirk: 'Exposed chrome V8 engine with roaring open zoomie pipes.',
  },
  {
    id: 'hot_dog_van',
    name: 'Frankfurter Cruiser',
    kind: 'hot_dog_van',
    size: [2.7, 1.35],
    maxSpeed: 7.2,
    accel: 5.0,
    turnSpeed: 1.8,
    maxHp: 145,
    maxFuel: 120,
    trunkSlots: 18,
    lightRadius: 11,
    color: '#d88a38',
    quirk: 'Giant rooftop hot dog. Heavy bumper pushes through obstacles.',
  },
  {
    id: 'tesla_roadster',
    name: 'Cyber Roadster',
    kind: 'tesla_roadster',
    size: [2.3, 1.25],
    maxSpeed: 12.5,
    accel: 9.8,
    turnSpeed: 2.8,
    maxHp: 110,
    maxFuel: 120,
    trunkSlots: 8,
    lightRadius: 14,
    color: '#e11d48',
    quirk: 'Ludicrous Mode electric torque. Whisper quiet, emits neon sparks and carries Starman!',
  },
];
