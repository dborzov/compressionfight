# Choose Your Codec — notes for coding agents

Arcade-style educational game about data compression codec design. The player picks a **heist** (an application
with requirements), fills six **chairs** (codec pipeline roles) with **fighters** (algorithms), watches a live
pipeline run real toy math on sample data, then gets graded. `README.md` explains the game model.

## Layout: static site, no build step

Plain HTML + CSS + ES2020 JavaScript. No frameworks, no bundler, no npm dependencies, no transpilation.
Every module attaches to `window.CYC` (`CYC.Engine`, `CYC.Rules`, `CYC.Viz`, `CYC.UI`, `CYC.Demos`, `CYC.Quests`,
plus the data tables). Scripts load in this order from `index.html`:

```
data.js, curriculum.js, quests_data.js, engine.js, rules.js, demos.js, quests.js, viz.js, ui.js
```

Module contracts live in `docs/` and are the source of truth when code and docs disagree:

- `docs/ARCHITECTURE.md` — vocabulary (role, fighter, squad, heist, level), `CYC.Engine.run` result and Viz
  descriptor shapes, `CYC.Rules.score`, UI modes and screens, portrait fallback.
- `docs/DEMOS.md` — `CYC.Demos.render(container, fighterId)`: the 34 per-fighter worked-example labs, with the
  exact inputs and expected outputs each one must show.
- `docs/SIDEQUESTS.md` — `CYC.Quests.render(container, questId, {onComplete})`: the eight mini games and their
  scoring formulas.

## Content vs code

| Content (information theory, must stay factually honest) | Code |
|---|---|
| `js/data.js` roles, families, fighters (stats, moves, weaknesses, `used in`, `persona`), heists, real codecs, metaphors, combos, duels | `js/engine.js` toy pipeline |
| `js/curriculum.js` the Fixer's tutor lines, lessons, quizzes, trade-off presets | `js/rules.js` scoring |
| `js/quests_data.js` side-quest text (howto / watch / debrief) | `js/demos.js`, `js/quests.js`, `js/viz.js`, `js/ui.js` |
| `things_to_explain/` research notes feeding the content (leave in place) | `css/style.css`, `index.html` |

Rules of the house:

- Content files carry real claims about real codecs and algorithms. Do not invent numbers, codec facts, or
  history; if something is a simplification, say so in the text.
- Engine math must be real where it is cheap to be real (actual LZ77 matching, actual Huffman tree, actual
  DCT-II, actual k-means, actual Golomb-Rice, actual int4 blocks). Approximations are allowed only when labelled
  in the stage caption (see the list in `docs/ARCHITECTURE.md`). Every number shown to the player must be
  computed from the stated input by the stated formula; no hard-coded "results".
- `CYC.Demos.calc.*` and `CYC.Quests.calc.*` are pure functions so node can test them without a DOM.

## Running and testing

Preview: open `index.html` directly in a browser (file:// works), or serve the folder:

```
python3 -m http.server 8000     # then http://localhost:8000/
```

Tests are plain node scripts (no test runner; they set `global.window = global` and `require` the modules):

```
node test/engine_test.js
node test/demos_test.js
node test/quests_test.js
```

Run all three before committing changes to `js/`. Some may not exist yet while their module is in progress.

## Images

Generated locally with the `generate-image` CLI (FLUX.2 Klein; see the `local-image-generation` skill). Style:
pixel-art 90s arcade fighting-game character select / mission poster.

- `img/fighters/<id>.png` — 256×256 portrait, one per fighter id in `CYC.FIGHTERS`, plus `none.png` for the
  empty chair. The prompt comes from the fighter's `persona` field in `js/data.js`.
- `img/heists/<id>.png` — 384×256 poster, one per heist id in `CYC.HEISTS`.

Generate at the model's native 1024 resolution, then downscale to the target size. Keep raw 1024px outputs out
of the repo (`scratch/` is git-ignored). The UI must keep working with a missing image (coloured tile fallback
with initials).

## Conventions

- Plain ES2020; no modules/imports, no frameworks, no build. Everything on `window.CYC`.
- Element ids and `data-*` hooks referenced across `ui.js`, `demos.js`, `quests.js` and tests are stable API;
  do not rename them casually.
- Phone width (360px) must work: no horizontal page scroll, controls reachable by touch.
- All asset references are relative (`img/...`, `js/...`, `css/...`), never `/img/...`: the site is served
  under a sub-path.
- Seeded randomness (mulberry32) everywhere so runs are reproducible.
- Several agents may edit different files concurrently. Stay inside the files you own; `git add` explicit paths.

## Deployment

GitHub Pages serves the repo root of `main` (there is a `.nojekyll`): https://dborzov.github.io/compressionfight/
Nothing to build; a push to `main` is a deploy. The planned custom path `https://borzov.ca/compressionfight`
goes through a Cloudflare Worker proxy in `deploy/worker/` (see `deploy/README.md`); it is not live until the
route is uncommented and deployed.
