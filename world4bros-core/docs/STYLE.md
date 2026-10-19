# Style: Ink Hollow (top-down)

Reference: a hand-inked storybook forest. Heavy ink outlines with a thinner doubled line, dense hatching on the shadow side, grain and scratches, muted washes, gnarled twisted trees. The camera is top-down and pulled back, so characters are small, readable shapes with a lot of detail when zoomed in.

## Rules
- Every shape goes through `inked()`: fill, volume gradient, hatching (two angles), grain, outline. Do not draw flat fills.
- Light comes from the top-left; shadows and hatching fall to the bottom-right. Ground shadows are flat soft ellipses.
- Colors are muted by `STYLE.desat` (default 0.34). It is a user setting (Ink wash slider) from 0 (full color) to 0.7 (nearly monochrome).
- Night is very dark with warm firelight holes. The campfire is the warmest and brightest thing on screen.
- Shapes: round and soft for friends, lumpy for scenery, gnarled and spiky for dead things. Top-down trees show canopy rings or radiating branches, never side views.
- No outlines in pure black: ink is `#15171b`. No white edges or sticker looks.

## Character
Top-down rig, facing +x in local space: shadow, legs and feet (stride), torso, belt, back item (pack and bedroll), scarf, arms with hands and held items, head, hair, headwear, face gear. Parts are separate cached sprites so every slot can change independently. All items accept primary, secondary and accent colors. Hair has 10 styles and any color.

## Status effects (planned, Z26)
Each effect gets its own icon shape and sound word as well as a colour, drawn in comic form (drips, flames, starbursts, soot). Colours and shapes: World Bible §115 and `docs/STYLE.md` §4.2 at the repo root; target look in `../docs/design/index.html`. Put the colours in `src/render/ink.ts` or effect data, not in logic.

## Adding art
Add a drawer to `PARTS` (`src/render/parts.ts`) named `slot:itemId`, then add the item to `src/data/items.ts`. Use the helpers (`ik`, `cap`, `ell`, `rrect`, `lumpy`, `stroke`). Check it in `dev/sheet.html`.
