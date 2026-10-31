# Choose Your Codec

An arcade-style educational game about data compression codec engineering. Pick a **heist** (an application
that needs compression), read its stat sheet (the requirements), then build the **squad** (the codec pipeline)
by filling six chairs with **fighters** (algorithms). A live pipeline shows your sample data changing stage by
stage with honest bit counts; running the heist grades each chair against the job's requirements and reveals
the real codec.

Open `index.html` in a browser. No build step, no server required.

## The model behind the game

Every practical codec is a stack of the same roles. The game's chairs, in pipeline order:

| Chair | Role | Question it answers |
|---|---|---|
| The Lens | Transform | Which directions matter? |
| The Oracle | Prediction / modeling | Can I guess the next thing from what I already have? |
| The Butcher | Quantization | How do I turn numbers into a few bits? (the only lossy stage) |
| The Packer | Entropy coding | Some symbols are common; can I give them shorter codes? |
| The Con Artist | Perceptual model | What will the human never notice? |
| The Inside Man | Shared prior / side information | What does the decoder already know before the first byte? |

Heist stats (latency tolerance, encoder/decoder budgets, random access, bandwidth pressure, decoder memory,
lossless, can-ship-side-info) collide with fighter stats (delay, enc/dec cost, random-access friendliness,
squeeze, memory, lossless). The scoring rules in `js/rules.js` name the requirement and the property that
collide; that collision is the lesson.

## Layout

- `js/data.js` roster: roles, families, 34 fighters, 13 heists, real codecs, metaphors, combos, duels
- `js/curriculum.js` the Fixer's tutor lines, lessons per level, checkpoint quizzes, trade-off presets
- `js/quests_data.js`, `js/quests.js` side-quest mini games (Shannon's guessing game, Huffman tree, bit budget, JND, dictionary training, k-means safehouses, block packing, random access race)
- `js/engine.js` toy pipeline with real math (LZ77, Huffman, arithmetic/ANS estimates, Rice, DCT, k-means, int4 blocks)
- `js/rules.js` scoring
- `js/demos.js` per-fighter worked examples
- `js/viz.js`, `js/ui.js`, `css/style.css`, `index.html` front end
- `img/` pixel-art portraits and posters generated locally with FLUX.2 Klein
- `docs/` contracts the modules were built against
- `test/` node tests: `node test/engine_test.js`, `node test/demos_test.js`, `node test/quests_test.js`
