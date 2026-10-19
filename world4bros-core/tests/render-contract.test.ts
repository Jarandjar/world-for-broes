import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildContent } from '../src/data/content';
import { PARTS } from '../src/render/parts';
import { OBJECT_DRAWERS } from '../src/render/objects';
import { STRUCTURE_DRAWERS } from '../src/render/structures';
import { HAIR_STYLES } from '../src/data/appearance';

const content = buildContent();

test('every wearable item has a drawer (slot:item)', () => {
  const missing: string[] = [];
  for (const i of content.items.values()) {
    if (!i.equip || i.equip.slot === 'hands') continue;           // gloves are drawn as part of the arm
    const slot = i.equip.slot;
    const ok = PARTS[`${slot}:${i.id}`] || (slot === 'offHand' && PARTS[`mainHand:${i.id}`]);
    if (!ok) missing.push(`${slot}:${i.id}`);
  }
  assert.deepEqual(missing, []);
});

test('every hair style, world object and structure has a drawer', () => {
  assert.deepEqual(HAIR_STYLES.filter((h) => !PARTS[`hair:${h.id}`]).map((h) => h.id), []);
  assert.deepEqual([...content.objects.keys()].filter((k) => !OBJECT_DRAWERS[k]), []);
  assert.deepEqual([...content.structures.keys()].filter((k) => !STRUCTURE_DRAWERS[k]), []);
});
