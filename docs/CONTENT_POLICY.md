# Content policy: this site teaches data compression, literally

The character-select / roster presentation is a visual frame and a bit of quirk. Every word of explanatory text,
every name, every label teaches data compression directly, in the register of `docs/map/style.md` (concrete,
first-principles, terminology introduced properly). No heist metaphors in the text. No invented names for things.

## Names (binding)

Algorithm entries (`CYC.FIGHTERS[].name`) use the real algorithm name, short form; the `real` field keeps the long
form. The old codename must not appear anywhere in user-visible text, captions, lessons, quests or articles.

| id | old codename | new `name` |
|---|---|---|
| dct | COSINE | DCT |
| mdct | LAPWING | MDCT |
| dwt | RIPPLE | Wavelet (DWT) |
| prism | PRISM | YCbCr / mid-side |
| pca | AXIS | PCA / Hadamard rotation |
| bwt | SHUFFLER | Burrows-Wheeler (BWT) |
| neural_enc | NEURON | Learned encoder |
| lz77 | ECHO | LZ77 |
| lzw | SCRIBE | LZW |
| delta | GHOST | Delta coding |
| lpc | PROPHET | Linear prediction (LPC) |
| spatial | NEIGHBOR | Spatial prediction |
| motion | TRACKER | Motion compensation |
| cm | MENTALIST | Context mixing |
| neural_prior | ORACLE PRIME | Learned probability model |
| uniform | GRID | Uniform quantizer |
| qmatrix | SCULPTOR | Quantization matrix |
| companding | BENDER | μ-law companding |
| vq | CATALOG | Vector quantization |
| rvq | MATRYOSHKA | Residual VQ |
| pq | SPLITTER | Product quantization |
| blockint | BLOCKY | Block int4 |
| huffman | WOODCUTTER | Huffman coding |
| arith | FENCER | Arithmetic coding |
| ans | STACKER | ANS |
| rice | RULER | Golomb-Rice |
| rle | REPEATER | Run-length coding |
| psychoacoustic | HYPNOTIST | Psychoacoustic model |
| chroma | ILLUSIONIST | Chroma subsampling |
| gan | FORGER | Adversarial loss |
| tables | RULEBOOK | Standard tables |
| dictionary | PHRASEBOOK | Trained dictionary |
| codebook | KEYMASTER | VQ codebook |
| decoder_net | BRAINJAR | Learned decoder |
| unrope | REWIND | Pre-RoPE storage |
| svd_lowrank | PROJECTOR | Low-rank SVD |
| mla | BOTTLENECK | Multi-head latent attention |
| anchor_delta | ANCHORMAN | Anchor delta coding |
| coupled | SIAMESE | Coupled quantization |
| waterline | WATERLINE | Water-filling allocation |
| fp4 | QUARTER | NVFP4 |
| escape | ESCAPE HATCH | Outlier escape |
| evict | REAPER | Token eviction |
| sinkguard | BOUNCER | Sink & window exemption |
| spotlight | SPOTLIGHT | Top-k selection |
| basis | BLUEPRINT | Trained basis |
| tiered | UNDERSTUDY | Tiered cache |
| none | EMPTY CHAIR | None (pass-through) |

Pipeline stages (`CYC.ROLES[].codename` → drop the codename; `title` is the name):

| id | old codename | name | question |
|---|---|---|---|
| transform | THE LENS | Transform | Which directions carry the energy? |
| predict | THE ORACLE | Prediction | What can be guessed from what is already decoded? |
| quantize | THE BUTCHER | Quantization | How many bits per value, and where may error go? (the only lossy stage) |
| entropy | THE PACKER | Entropy coding | How are skewed symbols turned into bits? |
| perceptual | THE CON ARTIST | Perceptual model | What will the consumer not notice? |
| prior | THE INSIDE MAN | Shared prior | What does the decoder already hold before the stream arrives? |

Tasks (formerly heists; `CYC.HEISTS[].name`; the `codec` field is the real-codec label and is shown, not hidden):

| id | old | new name |
|---|---|---|
| archive | THE ARCHIVE JOB | General-purpose file archive |
| pixelart | PIXEL PERFECT | Lossless images |
| studio | STUDIO MASTER | Lossless audio |
| gallery | THE GALLERY | Photographs |
| walkman | WALKMAN | Music distribution |
| primetime | PRIME TIME | Streaming video |
| library | THE LIBRARY | Maximum-ratio text archive |
| phonecall | THE PHONE CALL | Real-time speech |
| logpipe | THE LOG PIPE | Small JSON messages |
| vectorvault | THE VECTOR VAULT | Billion-scale vector search |
| weightsvault | THE WEIGHTS VAULT | LLM weights on a laptop |
| kvcache | THE CONTEXT WINDOW | KV cache resident in HBM |
| webphoto | THE SHOWROOM | Modern web images |
| onebit | THE ONE-BIT VAULT | KV cache at 1–2 bits |
| overflow | THE WAREHOUSE RUN | KV cache offload to host memory |
| coldkv | THE COLD STORAGE | KV cache cold storage |
| kvcourier | THE COURIER | KV cache over a network |
| purge | THE PURGE | KV token eviction |
| bornsmall | THE ARCHITECT'S CUT | KV compression by architecture |

Levels (`CYC.LEVELS[].name`): 1 Lossless basics · 2 Lossy coding · 3 Shared priors · 4 ML-era formats · 5 KV cache.

Requirement stats: drop the metaphor labels (`CYC.METAPHORS[].label`) and show the real term as the label:
Lossless required · Latency tolerance · Encoder compute · Decoder compute · Random access · Bandwidth pressure ·
Decoder memory · Side information allowed. Keep the `hint` text but rewrite it literally.

Other vocabulary (everywhere: UI chrome, data strings, tutor, quests, verdicts, engine/lab captions, articles):

| drop | use |
|---|---|
| heist, job (as metaphor), vault, crew, squad, chair, fighter, pro, the Fixer, tag team, friendly fire, side quest, loot, van, alarm timer, getaway ride, glovebox, inside man, planning time, vault doors, leave no trace, run the heist, the take, the real crew, squad room, squad chemistry | task, pipeline, stage, algorithm, tutor, synergy, conflict, exercise, results, requirements, pipeline builder, run the pipeline, the real codec, pipeline interactions |
| "choose your fighter" | "choose an algorithm" |
| rank letters S/A/B/C/D | keep (a grade), with the headline rewritten plainly |
| signature move (`move` field) | "key idea" (`move` stays as the field name; content is the one-sentence mechanism) |

## Voice

Explanatory text (why, weakness, reqNotes, reveal, tutor lines, lessons, verdicts, captions, quest how-tos) teaches
in the style of `docs/map/style.md`: concrete, numbers first, terminology defined, no hand-waving, no jokes at the
expense of clarity. Short sentences. A tutor line may be warm, but it explains what to do and why in compression
terms ("Pick a predictor for audio: the residual of a good predictor is a narrow spike around zero, which is what
Golomb-Rice is optimal for"). Where a string already is literal and correct, keep it; only metaphors are removed.
Personas (`persona` field) describe the painted allegorical portrait and may stay figurative; they are not shown as
text.

## Links

The game links into the Map wherever a family is mentioned: every algorithm card ("Read the introduction"), every
stage header (the stage's article from `CYC.MAP.roles`), every task briefing (`CYC.MAP.heists` or codec-dissections),
every lesson card (its `demo` fighter's article), every result verdict (the fighter's article), the tutor box (when
a line mentions a family, append a link).
