# Design system: Art Nouveau

One identity for the game (`css/style.css`, `js/viz.js`, lab and quest CSS) and the Map (`site/assets/css/map.css`).
Light, warm, ornamental, legible. Think Mucha posters and Vienna Secession book design, rendered with restraint.

## Tokens (define on `:root`; every colour in the site comes from these)

```css
:root {
  --paper: #f3eadb;        /* page background: warm cream */
  --paper-2: #ebdfc8;      /* panels, cards */
  --paper-3: #e0d2b6;      /* sunken areas, table stripes */
  --ink: #2a211b;          /* body text */
  --ink-2: #5a4a3d;        /* secondary text */
  --line: #b89f78;         /* hairlines, borders */
  --gold: #b8892e;         /* primary accent: titles' ornaments, active states */
  --gold-2: #e2c272;       /* highlights on gold */
  --teal: #246b66;         /* links, primary buttons */
  --teal-2: #3f8f88;
  --sage: #6f8f4e;
  --rose: #b05a4c;
  --plum: #6b4a7a;
  --copper: #9c6b3f;
  --good: #4f7f3f;  --warn: #b8892e;  --bad: #a7403a;   /* semantic */
  /* stage colours (replace CYC.ROLES[].color to match) */
  --c-transform: #246b66;  --c-predict: #6f8f4e;  --c-quantize: #b05a4c;
  --c-entropy: #6b4a7a;    --c-perceptual: #b8892e; --c-prior: #9c6b3f;
}
```

Set `color-scheme: light`. No dark mode for now (a single committed look). Keep contrast ≥ 4.5:1 for text.

## Type

Google Fonts: `Cormorant Garamond` (display, 600/700, used for titles and card names, with generous tracking on
small caps), `Source Serif 4` (body, 400/600, 18–19px on the Map, 16–17px in the game, 65–70ch measure),
`JetBrains Mono` (code, numbers in meters/tables, `font-variant-numeric: tabular-nums`). Fallback stacks on every face.
Eyebrows and labels: small caps via `font-variant: small-caps` in Cormorant, letter-spacing 0.08em. No pixel fonts.

## Ornament (restraint)

- Hairline double rules (`border: 1px solid var(--line); box-shadow: inset 0 0 0 3px var(--paper), inset 0 0 0 4px var(--line)`).
- Corner flourishes: one inline SVG whiplash curve (a single path, stroke var(--gold)) reused at panel corners via
  `background-image` at the four corners; keep it small (24–32px) and only on primary panels.
- Section titles get a centred ornament rule: `— ❦ —` style using a thin SVG or a Unicode fleuron in gold.
- Stage colour appears as a 4px left rule on cards and as the fill of small badges, never as large backgrounds.
- Portraits and posters are framed with a 6px cream mat and a 1px gold line, slight paper shadow.
- Buttons: cream fill, gold hairline, teal text; primary button inverts (teal fill, cream text). Radius 2px.
- Meters/bars: teal fill on paper-3 track with a gold hairline; the entropy-bound marker in rose.
- Canvas visualisations (viz.js, labs, quests): read tokens from CSS variables; backgrounds paper-2; marks in stage
  colours; text in ink. Grid lines `--line` at 40% alpha.

## Assets

Generated locally with `generate-image`. Prompt frame for all: "Art Nouveau illustration in the manner of Alphonse
Mucha, with expressive impressionist oil-painting brushwork; ornamental floral border; palette of teal, old gold,
sage green, dusty rose and plum on warm cream; soft natural light; no text, no letters, no logos."
- Algorithm portraits: 384×384, an allegorical figure or still life for the `persona` field (e.g. DCT: a figure
  holding ripples of overlapping waves rendered as stained glass). Save `img/fighters/<id>.png`.
- Task posters: 576×384, a scene for the application (a records archive, a photographer's studio, a concert hall, a
  telephone exchange, a server hall painted like a cathedral nave). Save `img/heists/<id>.png`.
- Map hero banners: 1152×432, a scene from the article's framing story. Save `docs/map/zoo/img/hero/<slug>.png`.
Downscale from 1024-class renders with `magick -filter Lanczos` (painterly, not pixelated); `image-rendering: auto`.

## Layout

Unchanged in structure: game screens as before, Map with sidebar/TOC. Phone width 400px must still work. Remove CRT
scanlines, stripes, hard pixel shadows, uppercase-only headings and arcade flashes. Motion: fades only; respect
reduced motion.
