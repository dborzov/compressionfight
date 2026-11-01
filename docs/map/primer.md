# A Field Map of Data Compression

---

## Part 0 — The One Idea That Organizes Everything

Before the taxonomy, one unifying idea. Almost everything in this document is a corollary of it.

>[!Key] **Compression = modeling + coding.** 
>A compressor is:
> *  a _model_ that predicts the data
> * a _coder_ that turns good predictions into short bit strings.


Shannon's result gives the exchange rate: if your model assigns probability `P` to the symbol that actually occurs, the optimal code spends `-log2(P)` bits on it. Predict well (high `P`) → pay few bits. Predict badly → pay many. A perfect coder attached to a model pays exactly the model's _cross-entropy_ on the data — which is why "how well can you compress X" and "how well can you predict X" are literally the same question. (This is also why a modern LLM plus an arithmetic coder is, without any modification, a state-of-the-art text compressor.)

This splits the entire toolkit into three families of parts:

1. **Coders (entropy coders).** The universal back end. Given symbols plus probability estimates, they emit bits at nearly `-log2(P)` per symbol. Huffman, arithmetic, ANS, Rice. This part is essentially _solved_ — modern coders get within a fraction of a percent of the Shannon bound. All the interesting engineering variation is in speed, adaptivity, and hardware friendliness.
    
2. **Models, transforms, and predictors.** The front end. Everything else in your list — LZ, delta, BWT, MTF, PPM, Markov models, motion compensation, DCT — is machinery for _exposing structure_: rewriting the data so that, by the time it reaches the entropy coder, the symbols are highly predictable / heavily skewed. None of these save bits by themselves; they set up the shot.
    
3. **Quantizers.** The one and only place where information is deliberately _destroyed_. Everything lossy — MP3, JPEG, video, NVFP4 — is "throw away what the consumer won't miss (quantize), then pack what's left losslessly (model + code)."
    

So every practical format is a pipeline with this skeleton:

```
 raw data
    │
    ▼
[ lossy formats only: transform → quantize ]     ← decide what information survives
    │
    ▼
[ model / predict / transform / match ]          ← expose the remaining redundancy
    │
    ▼
[ entropy code ]                                 ← cash predictions in for bits
    │
    ▼
 compressed bits (should look like white noise)
```

A useful sanity check you can apply to any format: the final output should be statistically indistinguishable from random noise. If gzipping a `.zip` file shrank it, the zip left money on the table. It doesn't — which tells you deflate's entropy stage did its job.

Two words worth pinning down, because the whole lossy world hangs on the distinction:

- **Redundancy** — structure that lets you reconstruct data exactly from fewer bits ("the next pixel is probably like the last one"). Exploited by _lossless_ tools.
- **Irrelevance** — information the consumer genuinely cannot perceive or does not need (audio masked by a louder tone, chroma detail your eye can't resolve, mantissa bits a neural net doesn't care about). Exploited by _lossy_ tools, i.e., quantization guided by a model of the consumer.

Lossless compression attacks only redundancy. Lossy compression attacks both — which is why its ratios are 10–100× larger.

---

## Part 1 — The Jobs: A Taxonomy of Compression Problems

"Which tool is right" is meaningless until you can state the job precisely. In practice a compression job is pinned down by five axes. Learn these five and you can characterize any codec, format, or system in one sentence.

### Axis 1 — The fidelity contract

- **Lossless.** Decoder output is bit-identical to input. Non-negotiable for executables, source code, text, databases, medical/legal archives, and any data that will be _processed further_ rather than consumed by a human.
- **Lossy.** Output only needs to be _good enough_ for a defined consumer. This immediately raises two questions that define the whole sub-field:
    - _Who is the consumer?_ Human ear (psychoacoustics), human eye (psychovisual models), a neural network (accuracy on a task), a downstream analytic (error bounds on floats). The consumer model tells you what is irrelevant.
    - _Where on the rate–distortion curve?_ Every lossy codec is a knob trading bitrate against distortion. "Quality 85 JPEG," "128 kbps MP3," "CRF 23 x264" are all positions on this curve.
- **Perceptually / effectively lossless.** Technically lossy, but distortion is below the consumer's detection threshold. Marketing term with real engineering content: it means the quantizer was tuned against a perceptual model, not a math-error metric.

### Axis 2 — The data's redundancy profile

Different data has different _kinds_ of exploitable structure, and each kind has dedicated tools. This is the single most useful checklist when facing new data:

| Redundancy type              | What it looks like                              | Toy example                                               | Tool family that eats it                                                              |
| ---------------------------- | ----------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Statistical skew**         | Some symbols much more frequent than others     | English text: `e` ≫ `z`                                   | Entropy coding                                                                        |
| **Exact repeats**            | The same byte strings recur, near or far        | Logs, HTML, source code, tarballs                         | LZ / dictionary matching; at macro scale, dedup                                       |
| **Context correlation**      | Next symbol predictable from previous k symbols | After `q` comes `u`; after `<htm` comes `l`               | Markov / PPM / context models, context-adaptive coding                                |
| **Local smoothness**         | Neighboring samples are numerically close       | Audio waveforms, photo pixels, sensor timestamps          | Prediction + residual (delta, LPC, filters); decorrelating transforms (DCT, wavelets) |
| **Cross-signal correlation** | Parallel channels move together                 | L/R stereo, RGB channels                                  | Channel transforms (mid/side, RGB→YCbCr)                                              |
| **Cross-item redundancy**    | Whole files/blocks duplicated across a corpus   | 500 VM images sharing an OS; nightly backups              | Deduplication, delta sync, shared dictionaries                                        |
| **Perceptual irrelevance**   | Detail the consumer can't perceive              | Quiet tone next to a loud one; chroma detail              | Quantization guided by a consumer model                                               |
| **Numeric structure**        | Value distributions inside binary numbers       | Float32 tensors: exponents low-entropy, mantissas ≈ noise | Bit/byte-plane splitting, block-scaled quantization                                   |

Note that most real data has _several_ of these at once — which is precisely why real formats are pipelines of several tools (Part 2, composition rules).

### Axis 3 — Access pattern and delivery model

This axis is the most underrated one, and it's where "the MP3 job" and "the Zoom job" diverge despite both being "compress audio."

- **Encode/decode symmetry.**
    - _Encode once, decode many_ (MP3 release, YouTube video, software package): the encoder can burn hours of CPU and use exhaustive search; only decode cost matters at scale. Expect asymmetric designs — expensive optimal parsing, cheap table-driven decoding.
    - _Symmetric / encode-heavy_ (backups, on-the-fly network compression, live streams): encode speed is on the critical path. Expect greedy, single-pass, hardware-light encoders.
- **Latency budget.** How much of the future may the encoder look at before it must emit bits?
    - _Interactive real time_ (VoIP, video calls, game streaming, cloud gaming): total mouth-to-ear budget ~150 ms, of which the codec may get 5–30 ms. This caps frame size and forbids lookahead — a hard structural constraint, not a tuning parameter.
    - _Buffered streaming_ (Netflix, live TV): seconds of buffer; codec latency is nearly free, so you can use bidirectional prediction (B-frames) and multi-pass rate control.
    - _Offline / archival:_ unbounded. Use anything.
- **Random access / seekability.** Can the decoder jump to the middle?
    - Sequential-only streams let context, dictionaries, and adaptive models grow forever (best ratio).
    - Seekable formats must _reset_ context at intervals — independent blocks, keyframes, restart markers. Every reset costs ratio; the reset interval is a ratio ↔ seek-granularity dial. (Same mechanism also enables parallel encode/decode and limits damage from corruption.)
- **Error environment.** File on disk / TCP: assume reliable. UDP / radio: packets vanish, so the format needs independent packets, concealment strategies, and recovery frames. This constraint alone reshapes codecs (see Opus vs MP3, and video-call vs streaming video configs).
- **Update pattern.** Write-once archive, append-only log, mutable file that must be re-synced (the rsync job), or a deduplicated store receiving daily near-duplicates.

### Axis 4 — Compute, memory, and energy envelope

- **Where does the decoder run?** If the answer is "in silicon on a billion phones and TVs" (video), the format must be standardized, decoder-cheap, fixed for a decade, and designed around hardware-friendly primitives. If it's "in software we ship," you can iterate freely (zstd's rapid evolution vs H.264's frozen spec).
- **Memory window.** LZ match window, BWT block size, context-model tables — all trade RAM for ratio. A 64 KB-RAM microcontroller and a 256 GB server are different jobs.
- **Throughput class.** Rough tiers worth memorizing: _storage/network path_ compression wants GB/s per core (LZ4, snappy, zstd-fast); _general file_ compression wants ~100s of MB/s (zstd default, gzip); _archival_ tolerates ~1–10 MB/s (xz, PPM); _research/record-setting_ tolerates KB/s (PAQ, neural).
- **Energy.** Mobile encode (your phone recording 4K) pushes toward hardware encoders even at some ratio cost.

### Axis 5 — Where the prior knowledge lives

Every compressor needs a model of the data. The engineering question is where that model is stored and how it's paid for:

- **Learned from the stream itself (adaptive):** self-contained, great on large inputs, but pays a "cold start" tax — the first kilobytes compress poorly while statistics warm up.
- **Shipped in the header (semi-static):** two-pass encoder measures the data, writes a table (e.g., deflate's dynamic Huffman trees). Costs header bytes; bad for tiny inputs.
- **Shared dictionary:** model trained on a corpus sample once, distributed to both sides out-of-band (zstd `--train` dictionaries, Brotli's built-in web-text dictionary). This is _the_ fix for the small-message problem (compressing thousands of 2 KB JSON documents).
- **Baked into a trained model:** neural codecs (Lyra, Encodec) push this to the extreme — megabytes of network weights on both sides encode "what speech sounds like," so the wire only needs to carry a few kbps of "which speech."

The pattern: the more prior knowledge both ends share, the fewer bits the wire needs. Dedup, rsync, dictionaries, and neural codecs are all points on this one spectrum.

### Stating a job in one sentence

Put the axes together and any job becomes a fill-in-the-blank:

> _"Compress data with redundancy profile ⟨R⟩ under fidelity contract ⟨F⟩, with access pattern ⟨A⟩, inside compute envelope ⟨C⟩, with prior knowledge living at ⟨K⟩."_

Worked contrast — the example from your question:

|Axis|**The MP3 job** (music distribution)|**The VoIP/Zoom job** (Opus, Lyra)|
|---|---|---|
|Fidelity|Lossy, consumer = human ear, transparent at ~192 kbps|Lossy, consumer = intelligibility first, naturalness second|
|Data|Full-band music, arbitrary content|Mostly one voice, band-limited, long silences|
|Symmetry|Encode once / decode billions of times|Fully symmetric, both directions, live|
|Latency|Irrelevant (hundreds of ms fine)|**Hard budget: ~20 ms frames, minimal lookahead**|
|Errors|Reliable file/HTTP|UDP loss: needs concealment, in-band FEC, per-packet independence|
|Compute|Cheap decode on 1990s hardware|Real-time encode+decode on a phone, on battery|
|Knowledge|Generic psychoacoustic model in the spec|Speech-specific priors; neural codecs ship a whole voice model|

Same medium, almost no overlapping constraints — which is why the resulting codecs share almost no design (MP3: big MDCT windows + psychoacoustics + Huffman; Opus: LPC speech model + short MDCT + packet-loss machinery). Once you can build this table for a new problem, tool selection in Part 2 becomes almost mechanical.

### 🗺 Cheatsheet 1 — The Jobs

- **One idea:** compression = modeling + coding; bits paid = `-log2 P(symbol)` under your model.
- **Five axes define any job:**
    - **Fidelity contract**
        - Lossless (bit-exact; data that gets processed further)
        - Lossy (rate–distortion knob; requires a _consumer model_ to define irrelevance)
        - Consumer = ear / eye / neural net / error-bounded analytics
    - **Redundancy profile** (usually several at once)
        - Statistical skew → entropy coding
        - Exact repeats → LZ / dedup
        - Context correlation → Markov / PPM / context-adaptive coding
        - Local smoothness → prediction+residual, decorrelating transforms
        - Cross-channel / cross-item → channel transforms / dedup, delta sync
        - Perceptual irrelevance → quantization + consumer model
        - Numeric structure → bit/byte splitting, block-scaled formats
    - **Access pattern**
        - Symmetry: encode-once-decode-many ⇒ spend on the encoder
        - Latency: interactive (ms, no lookahead) vs buffered vs offline
        - Random access ⇒ context resets (blocks/keyframes) ⇒ ratio ↔ seekability dial
        - Error environment: reliable vs lossy channel ⇒ packet independence, concealment
    - **Compute envelope**
        - Hardware decoder ⇒ standardized, frozen, decade-long formats (video)
        - Throughput tiers: GB/s path ▸ 100s MB/s general ▸ MB/s archival ▸ KB/s record-chasing
        - Memory window ↔ ratio
    - **Where knowledge lives** (a single spectrum)
        - Adaptive from stream → header tables → shared dictionaries → trained neural weights
        - More shared prior ⇒ fewer wire bits; small inputs _need_ shared priors
- **Litmus test:** compressed output should look like random noise; if it re-compresses, the entropy stage underperformed.

---

## Part 2 — The Toolkit: Atomic Algorithms as Roles in a Pipeline

The classification that makes everything click is **by role in the pipeline**, not by family resemblance or historical lineage. Every atomic algorithm you listed plays exactly one of five roles: it _codes_, it _matches_, it _predicts_, it _rearranges_, or it _quantizes_. Below is the toolkit organized that way, with each tool's defining properties: what redundancy it eats, what it emits, and what it costs.

### 2A — Entropy coders: the universal back end

**Role:** convert symbols + probability estimates into bits, at close to `-log2 P` bits per symbol. **Eats:** statistical skew. **Emits:** noise-like bits. **Always the last stage.**

- **Huffman coding.** Builds a prefix code: each symbol gets a whole number of bits. Fast (table lookups, trivially hardware-friendly), but the whole-bit granularity is a real tax when probabilities are extreme.
    - _Toy example:_ `P(A)=0.9, P(B)=0.1`. Entropy is 0.47 bits/symbol, but Huffman can't code A in less than 1 bit — you pay more than **2×** the optimal rate. This single weakness explains a lot of the ecosystem below.
    - _Workaround you know as "symbol grouping":_ code pairs/triples as super-symbols (`AA`, `AB`, …). The whole-bit tax amortizes over the group; cost is exponentially bigger tables. Rarely worth it today — the modern answer is ANS.
- **Arithmetic / range coding.** Codes the entire message as one long fraction; each symbol effectively consumes a _fractional_ number of bits, hitting the Shannon bound. Its killer feature isn't the ratio on static distributions — it's that the probability model can change _per symbol_, which unlocks the whole adaptive/context-modeling world (PPM, CABAC, PAQ, neural coding). Costs: multiplications per symbol (slower than Huffman), inherently serial, and it was patent-encumbered for the '80s–'90s — a non-technical fact that shaped real formats (it's why JPEG's arithmetic mode exists but nobody uses it, and why bzip2 uses Huffman).
- **ANS — Asymmetric Numeral Systems** (tANS/FSE, rANS). The 2010s breakthrough: arithmetic-grade ratios at Huffman-grade speed, by encoding into a single integer state with table-driven (tANS) or multiply-based (rANS) transitions. Quirk: decoding runs in _reverse_ order of encoding, so it fits block-based designs, not symbol-by-symbol adaptive ones. This is the default entropy stage of the modern era: zstd (FSE), JPEG XL, Apple LZFSE, CRAM.
- **Rice/Golomb codes.** Parameter-k prefix codes that are optimal for geometric distributions ("small values common, big values rare") — exactly the shape of prediction residuals. No tables, no state, cheap in hardware. This is why they show up wherever there's a predictor: FLAC, lossless image codecs, seismic/sensor formats.
- **Adaptivity — a property, not an algorithm.** Any coder can be driven three ways:
    - _Static:_ fixed hard-coded tables (deflate's "static Huffman" blocks; fine for a known distribution).
    - _Semi-static:_ two passes — measure, ship a table in the header, then code (deflate's "dynamic" blocks). Costs header bytes; hurts tiny inputs.
    - _Adaptive:_ both sides update the model symbol-by-symbol from decoded history. No header, learns local statistics, best ratio — but serial, slower, and any corruption desynchronizes everything downstream. Adaptive + arithmetic is the ratio-maximalist combo (PPM, CABAC, PAQ).

### 2B — Match models: turning repeats into references

**Role:** find that "this chunk of data already occurred" and emit a short back-reference instead. **Eats:** exact repeats, near and far. **Emits:** a mixed stream of literals + (offset, length) copy commands — which, note, still has skewed statistics.

- **LZ77 (sliding window).** Keep the last W bytes; replace repeats with `(offset back, length)` pairs.
    - _Toy:_ `to be or not to be` → `to be or not` + `(offset=13, len=5)`.
    - Its two big dials map straight onto job axes: **window size** = how far back repeats can be found = memory cost (deflate: 32 KB, a 1990 decision that still constrains gzip; LZMA: up to GBs; zstd `--long`: 128 MB+ for tarball-scale repeats). **Parse effort** = how hard the encoder searches for the _best_ set of matches (greedy → lazy → optimal parsing). Parse effort is exactly what "compression levels" sell: `zstd -1` vs `-19` and `gzip -1` vs `-9` change encode-side search only — decode speed stays nearly constant. That decode-invariance is _the_ property that made the LZ family the backbone of encode-once-decode-many distribution.
- **LZ78 / LZW.** Instead of a sliding window, both sides grow an explicit phrase dictionary and emit dictionary indices. Historically important (Unix `compress`, GIF), patent-plagued (the famous GIF/Unisys mess), and now essentially retired: LZ77's implicit "dictionary = the data itself" proved more flexible.
- **Shared / external dictionaries.** Pre-load the LZ window with a corpus sample so even the first bytes of a tiny input find matches. zstd's trained dictionaries and Brotli's built-in ~120 KB web-text dictionary. This is the standard cure for the small-message problem — thousands of 2 KB JSONs compress terribly alone, wonderfully against a shared dictionary.
- **The same idea at macro scale.** Deduplication (find repeated _chunks_ across a whole storage system via fingerprints) and rsync/delta-sync (find repeats where the "window" is a file on the _other machine_) are LZ matching with the window blown up to terabytes or across a network. Part 3.6 walks through these.

### 2C — Predictors: guess the next value, encode the miss

**Role:** predict each value from what's already been decoded; encode only the _residual_ (prediction error). A good predictor turns smooth, correlated data into residuals that are small, zero-centered, and geometrically distributed — i.e., exactly what Rice codes or an entropy coder with a peaky model want. **Eats:** local smoothness and context correlation.

- **Delta coding.** The trivial predictor: "next = previous."
    - _Toy:_ timestamps `1700000000, 1700000060, 1700000120, …` → deltas `60, 60, 60, …` → an entropy coder or RLE now eats it alive. Delta-of-delta (predict "same spacing") flattens it further — the trick behind time-series databases like Gorilla/Prometheus.
    - Signed residuals get **zigzag-mapped** (`0, -1, 1, -2, 2 → 0, 1, 2, 3, 4`) so "small magnitude" becomes "small unsigned integer."
- **Linear prediction.** Predict the next sample as a weighted sum of the last few. This is **LPC** in FLAC (weights fitted per block to the audio) and the **per-row filters** in PNG (fixed predictor menu: left / up / average / Paeth, chosen per row). Same tool, one fitted, one chosen from a menu.
- **Context models (your "Markov chain something something").** An order-k Markov model keeps, per k-symbol context, a probability distribution over the next symbol, and feeds it to an arithmetic coder. `P(u | "q") ≈ 0.98` → the `u` costs 0.03 bits.
    - **PPM (Prediction by Partial Matching)** is the classic engineering realization: try the longest context seen before (say order 5); if the next symbol is new in that context, emit an _escape_ and fall back to order 4, 3, … This was the compression-ratio king of the 1990s; its costs (RAM for context tables, serial adaptive arithmetic coding) kept it out of mainstream formats.
    - **Context mixing (PAQ family) and neural models** are the modern extreme: run many models, mix their predictions (logistic mixing, then RNNs, then transformers), arithmetic-code against the blend. This lineage owns the benchmark records (Hutter Prize / enwik9) at speeds of KB/s — a pure demonstration that modeling is the whole game.
- **Motion-compensated prediction** — the thing you asked the name of. In video, predict a block of the current frame as a _shifted_ block from a previously decoded **reference frame**; transmit the shift (**motion vector**) plus the residual. Frames predicted this way are **P-frames** (from past references) and **B-frames** (from past _and_ future references); self-contained frames are **I-frames / keyframes**. It's the predictor role again — just 2D, across time, with search. Full walkthrough in Part 3.4.

### 2D — Transforms & reorderings: rearrange so other tools can bite

**Role:** save no bits themselves; reversibly restructure data so its redundancy lands in a form some _other_ stage eats. The connective tissue of every pipeline.

- **RLE (run-length encoding).** `AAAAABBB` → `(A,5)(B,3)`. Only useful when something upstream manufactures runs — RLE is almost always found _downstream of_ MTF, quantization (zero runs), or a palette.
- **Move-to-front (MTF).** Keep symbols in a list; emit each symbol's current position, then move it to the front. Converts _locality_ ("recently seen symbols recur") into _global skew_ ("small numbers dominate") — translating a kind of redundancy entropy coders can't see into the kind they can.
- **Burrows–Wheeler Transform (BWT).** Sort all rotations of a block; output the last column (plus an index). Effect: characters that occur in _similar contexts_ end up adjacent.
    - _Toy:_ `banana$` → BWT → `annb$aa` — the `n`s cluster (both preceded by `a`), the `a`s cluster.
    - On text, the BWT output is long runs of locally-repeating symbols — which is precisely MTF's food, whose output is precisely an entropy coder's food. Hence the canonical stack **BWT → MTF → RLE → entropy coder** (= bzip2). BWT is the elegant trick that gets context-model-class ratios out of cheap non-adaptive back-end stages; costs are blockwise operation (no streaming, seek resets) and sort-heavy encoding.
- **Decorrelating transforms (DCT, MDCT, wavelets).** Rotate a block of correlated samples into frequency-ish coefficients so the energy concentrates in a few of them.
    - _Toy:_ 8 pixels of a smooth ramp `[10,12,14,16,18,20,22,24]` → DCT → one big "average" (DC) coefficient, one modest "tilt" coefficient, six ≈ 0. Sixteen meaningful numbers became ~2.
    - Crucially, this is the _setup move for quantization_: after the transform, "throw away small high-frequency coefficients" maps cleanly onto "discard what the eye/ear misses." Nearly every lossy media codec is transform → quantize at its core (JPEG: DCT; MP3/AAC/Opus-CELT: MDCT; JPEG 2000: wavelets).
- **Channel & layout transforms.** RGB → YCbCr (separate brightness from color so each can be treated per the eye's acuity); stereo → mid/side; array-of-structs → struct-of-arrays; **byte shuffle** on float arrays (group all byte-0s, all byte-1s… — exponent bytes are near-constant and compress hugely, mantissa bytes are noise and stop polluting the rest); column stores (Parquet) as the same idea at table scale. One principle: _group bytes that share statistics, separate ones that don't_, so downstream LZ/entropy stages see clean distributions instead of interleaved mush.
- **Blocking / framing.** Splitting into fixed-size blocks isn't a compression trick at all — it's plumbing that buys **parallelism, random access, bounded memory, and corruption containment**, and it always _costs_ ratio (every block boundary resets context and matches). The block size is a dial: bigger blocks → better ratio, worse seek/parallel granularity. You'll see this dial on everything: bzip2's 100–900 KB blocks, video GOP length, FLAC frames, Parquet row groups, dedup chunk sizes.

### 2E — Quantizers: the only lossy stage

**Role:** map many possible values to fewer representable ones. The _only_ stage that discards information; everything after it is lossless packing. The design question is never "how to quantize" (rounding is easy) but "**what does the consumer not care about?**" — a quantizer is only as good as its consumer model.

- **Scalar quantization.** `round(x / step)`; `step` is the rate–distortion knob. In media codecs the step varies per coefficient: JPEG's quantization tables and MP3's psychoacoustic bit allocation are _consumer models compiled into step sizes_ — coarse where the eye/ear is insensitive (high frequencies, masked tones), fine where it isn't.
- **Vector quantization (VQ).** Quantize whole vectors at once: keep a **codebook** of representative vectors, transmit nearest-codeword indices. A GIF's 256-color palette is VQ on RGB triples. VQ's modern renaissance: **residual VQ** in neural codecs (SoundStream/Encodec/Lyra), where cascaded codebooks quantize a learned latent into the discrete "audio tokens" that also power audio LLMs.
- **Block-scaled number formats (NVFP4 and friends).** Your tensor example, and a perfect specimen: a block of 16–32 values shares one scale factor (stored in higher precision), each value keeps ~4 bits of significand. It's quantization tailored to two facts: (1) the redundancy — neighboring tensor values share magnitude, so the exponent is nearly common across a block; (2) the consumer — a neural net's accuracy tolerates coarse mantissas if the dynamic range survives. Same rate–distortion logic as JPEG, different consumer model (a loss curve instead of an eyeball).

### 2F — Composition rules: why real stacks look the way they do

This is the payoff of the role-based classification. Six rules generate essentially every real-world pipeline:

1. **Each stage eats one redundancy type and must emit output shaped for the next stage's appetite.** Pipelines are chains of matched interfaces, not grab-bags.
2. **The entropy coder is always last** — its output is incompressible by construction, so nothing useful can follow it.
3. **Lossy stages come first** (transform → quantize), because you decide what information survives _before_ you pack it. Every lossy codec is literally "irrelevance remover in front of a lossless compressor."
4. **Never stack two tools that eat the same redundancy.** The second one starves (gzip on a JPEG: ~0%). Corollary: compress _before_ encrypting — ciphertext has no exploitable structure at all.
5. **Predictors and matchers are front-ends that manufacture skew; coders are back-ends that cash it in.** A front-end without a back-end leaves money on the table; a back-end without a front-end only sees byte frequencies.
6. **Every structural convenience (blocks, keyframes, packet independence, context resets) is bought with ratio.** Where you set those dials is dictated by Axis 3 of the job, not by the data.

Two canonical worked examples of the rules in action:

- **Why LZ77 + Huffman (deflate) is the right pairing.** LZ77 eats the _repeat_ redundancy but its output — literals, match lengths, offsets — is itself heavily skewed (short matches and common letters dominate; offsets cluster) — statistical redundancy LZ cannot touch. Huffman eats exactly that. Conversely, Huffman alone is blind to repeats: it sees `abcabcabc` as flat byte frequencies. Each covers the other's blind spot (rules 1 & 5). Deflate even codes the three token types with separate Huffman trees — separating streams with different statistics (the 2D layout principle, in miniature). zstd is the same architecture with each part upgraded: bigger window, smarter parsing, FSE/tANS instead of Huffman for the tight distributions.
- **Why BWT needs MTF and RLE.** BWT's output has _locality_, not global skew — an entropy coder sees near-flat symbol frequencies over the whole block and shrugs. MTF translates locality → small-number skew; RLE crushes the zero-runs MTF produces; _now_ the entropy coder feasts (rule 1, three links deep). bzip2 is exactly this chain.

### 🗺 Cheatsheet 2 — The Toolkit

- **Classify every algorithm by pipeline role:**
    - **Code** (skew → bits, always last)
        - Huffman — fast, whole-bit tax (bad at P>0.5); fix via symbol grouping, or better, ANS
        - Arithmetic/range — optimal, per-symbol model swaps ⇒ enables all adaptive/context coding; slower, serial
        - ANS (tANS/FSE, rANS) — arithmetic ratio at Huffman speed; block-oriented; the modern default
        - Rice/Golomb — free-standing optimal for geometric residuals; the predictor's companion
        - Drive any of them static / semi-static (header table) / adaptive (best ratio, serial, fragile)
    - **Match** (repeats → references)
        - LZ77 — dials: window (memory ↔ reach) & parse effort (= "compression levels", encode-only!)
        - LZ78/LZW — explicit dictionary; historical (compress, GIF)
        - Shared dictionaries — pre-warmed window; the small-message cure
        - Dedup / rsync / delta = LZ with a terabyte- or network-sized window
    - **Predict** (smoothness → small residuals; pair with Rice/zigzag)
        - Delta, delta-of-delta (time series)
        - Linear prediction: LPC (FLAC), PNG row filters
        - Context models: order-k Markov → PPM (escape & fallback) → context mixing / neural (ratio records, KB/s)
        - Motion compensation: reference frames + motion vectors + residual; I/P/B frames
    - **Rearrange** (translate redundancy into an edible form; saves nothing by itself)
        - RLE (needs upstream run-makers) · MTF (locality → skew) · BWT (context → locality; blockwise)
        - DCT/MDCT/wavelets (smoothness → few big coefficients; the setup for quantization)
        - Channel/layout: YCbCr, mid/side, byte-shuffle floats, column stores — group like statistics
        - Blocking/framing: buys parallelism + seek + robustness, always costs ratio
    - **Quantize** (the only lossy stage; quality = quality of the consumer model)
        - Scalar + perceptual step sizes (JPEG tables, psychoacoustics)
        - Vector quantization: palettes → residual VQ → neural-codec tokens
        - Block-scaled formats (NVFP4/MX): shared per-block scale + few-bit values; consumer = model accuracy
- **Composition rules:**
    - Chain stages by matching output shape → next stage's appetite
    - Entropy coder last · lossy first · never two eaters of the same redundancy (⇒ compress before encrypt)
    - Front-ends manufacture skew, back-ends cash it in — you generally want both
    - Structural conveniences (blocks/keyframes/packets) are paid for in ratio; set them per the _job_, not the data
- **Canonical stacks to memorize:**
    - deflate/zstd: `LZ77 → entropy(tokens)`
    - bzip2: `BWT → MTF → RLE → entropy`
    - FLAC/PNG: `predict → residual → entropy`
    - JPEG/MP3/video: `transform → perceptual quantize → [predict/RLE] → entropy`
    - PPM/PAQ/LLM: `context model → adaptive arithmetic`
    - Neural codec: `learned transform → VQ → (entropy)`

---

## Part 3 — The Lay of the Land: Domain Walkthroughs

Now the framework earns its keep. For each domain: the job stated on the five axes, then the timeline of landmark formats, each decomposed into toolkit roles — so you can _see_ the job constraints selecting the tools.

### 3.1 General-purpose lossless (files, packages, network payloads)

**The job:** unknown byte streams (Axis 2: expect repeats + skew + some context correlation, but guarantee nothing) · lossless · usually one-shot sequential · the defining tension is **Axis 4: which throughput tier?** — which is why this domain's signature UX is a _speed dial_ (compression levels) rather than a quality dial.

**Timeline, decomposed:**

- **`compress` (1984) — LZW alone.** Match stage only, no entropy back-end. Fast for its day, mediocre ratio, killed by patents.
- **deflate / zip / gzip / zlib (1990–95) — LZ77(32 KB window) → semi-static Huffman.** The complete two-stage architecture (match + code) in one cheap package. So well-matched to its era's constraints that it became the substrate of everything: ZIP, gzip, PNG, HTTP, PDF, git. Its frozen 32 KB window is also the textbook lesson in how Axis-4 decisions calcify.
- **bzip2 (1996) — RLE → BWT(≤900 KB blocks) → MTF → RLE → Huffman.** The rearrange-heavy path to near-context-model ratios without patented arithmetic coding. Better ratio than gzip, slower both ways, blockwise (parallelizable — see `pbzip2`). Now mostly displaced by xz/zstd.
- **LZMA / 7-Zip / xz (~2000s) — LZ77 with huge window (up to GBs) + optimal parsing → adaptive range coder with context modeling of the token stream.** Turn every ratio dial to max: reach (window), parse effort, and an adaptive-arithmetic back-end that models literals in context. The archival corner: encode slow, decode moderate, ratios that held the practical crown for a decade. Still the default for software distribution (`.tar.xz`).
- **Snappy / LZ4 (~2011) — LZ77 only, byte-aligned, no entropy stage at all.** The opposite corner: rule 5 deliberately violated (no back-end) to buy multi-GB/s decode. Born inside Google/databases for the storage path, where compression must be cheaper than the disk/network I/O it saves. Ratio ~2× vs gzip's ~3× — and nobody in this tier cares.
- **zstd (2016) — LZ77 (levels 1–22, `--long` window to 128 MB+) → FSE(tANS) for match tokens + fast Huffman for literals; trained shared dictionaries; frame-parallel.** The modern synthesis: one codec spanning LZ4-ish speed to xz-ish ratio on a single dial, with the small-message problem solved via `--train` dictionaries. Rapidly became the default in filesystems, databases, package managers, HTTP (`Content-Encoding: zstd`).
- **Brotli (2015) — LZ77 + context modeling + built-in ~120 KB web-text dictionary → Huffman.** zstd's web-specialized sibling: the shared dictionary and context models are pre-tuned for HTML/CSS/JS, buying ratio exactly where the web's Axis-5 profile allows.
- **The ratio-unbounded fringe — PPM, PAQ, cmix, NNCP, LLM-based.** Pure context modeling + adaptive arithmetic coding, compute be damned (KB/s). Owns the Hutter Prize / enwik9 benchmarks; practically used almost nowhere; conceptually important as the existence proof that better _prediction_ is the only remaining frontier.

**The map in one picture:** every entry above is `match front-end (or BWT rearrange, or context model) → entropy back-end`, and the domain's whole history is a walk along the speed↔ratio frontier as CPU, RAM, and patent constraints shifted.

### 3.2 Images

**The job splits on Axis 1** — lossless graphics (screenshots, diagrams, line art: sharp edges, flat regions, exact repeats) vs lossy photographs (smooth gradients + perceptual irrelevance: the eye's low chroma acuity and insensitivity to fine high-frequency texture). Both: encode-once-decode-many, random access per-image, decoder must be cheap and universal.

- **GIF (1987) — palette (≤256 colors = crude VQ) → LZW on the index stream.** A quantizer and a matcher, no entropy stage. Every limitation follows: 256 colors, mediocre ratio, patent drama.
- **PNG (1996) — per-row predictors (None/Sub/Up/Average/Paeth) → deflate.** Textbook predictor + generic back-end. The filters exist because rule 4 says deflate alone starves on photographic data: natural images have _smoothness_, not exact repeats, so a predictor first converts smoothness into near-zero residuals that deflate's Huffman stage can eat. On flat-color graphics the residuals become long zero runs and PNG shines; on photos it's honest but big — the _lossless_ contract simply can't touch perceptual irrelevance.
- **JPEG (1992) — the canonical lossy stack, worth knowing cold:**
    1. RGB → **YCbCr** + **chroma subsampling** (4:2:0 keeps ¼ of the color samples) — channel transform + the first quantization, cashing in the eye's low color acuity;
    2. split into **8×8 blocks** (framing: bounded memory on 1990 hardware, some corruption containment);
    3. **DCT** per block — smoothness → a few big low-frequency coefficients;
    4. **quantize** with a perceptual table — coarse steps for high frequencies (the consumer model, and the sole quality knob);
    5. **zigzag scan + RLE** of the resulting zero runs (rearrange to manufacture runs);
    6. **Huffman** (arithmetic mode existed, patents killed it — an Axis-4-adjacent constraint shaping a format). Every pipeline role appears once, in the canonical order of rules 2–3. Read this list against Cheatsheet 2 and the whole framework snaps into focus.
- **JPEG 2000 (2000) — wavelets + arithmetic coding (EBCOT).** Technically superior (progressive decode, no 8×8 blocking artifacts); lost in the market to JPEG's installed base + patent fears. A lesson that Axis 4's "decoder universality" can outweigh ratio.
- **WebP / HEIC / AVIF (2010s) — a still image = one intra frame of a video codec** (VP8 / HEVC / AV1 respectively). Video's intra toolbox (rich spatial prediction modes, variable block sizes, adaptive arithmetic coding) had lapped JPEG, so images just borrowed it. AVIF ≈ 50% of JPEG's bits at equal quality.
- **JPEG XL (2022) — VarDCT + strong spatial predictors + ANS**, plus a lossless mode and bit-exact JPEG recompression (~20% smaller). The state of the art on pure merit; adoption is, as ever, an Axis-4 story about decoders everywhere.

### 3.3 Audio

**Three genuinely different jobs** share this medium; keep them separate and every codec explains itself.

- **Job A: lossless archival — FLAC (2001).** Blocks → stereo decorrelation (mid/side) → **linear prediction (LPC)** fitted per block → **Rice-coded residuals**. Pure predict-and-code; ~50–60% of WAV. Note the toolkit pairing doing exactly what Cheatsheet 2 promises: fitted predictor → geometric residuals → Rice.
- **Job B: music distribution — MP3 (1993), AAC (1997).** Encode once, decode on everything, latency irrelevant; consumer = the ear, so the design center is the **psychoacoustic model** (frequency masking: a loud tone hides quiet neighbors; you can't hear what's masked, so spend ~0 bits there).
    - Stack: **MDCT** filterbank (long ~25 ms windows for frequency resolution — affordable because latency is free) → **quantization steered per-band by the psychoacoustic model** (the consumer model _is_ the codec) → **Huffman**. AAC is the same architecture with the compromises removed. Transparent quality around 128–192 kbps vs 1411 kbps for CD PCM — a ~10× win, nearly all of it from cashing in _irrelevance_, which is exactly the lossy premium promised in Part 0.
- **Job C: real-time conversation — Opus (2012).** Rebuild the Axis-3 column from the Part 1 table: ~20 ms frames, minimal lookahead, UDP loss, symmetric battery-powered encode. Opus's answers: a **speech model** (SILK: LPC-based, exploiting "this is one human voice") + a low-latency **MDCT mode** (CELT) with seamless switching; per-packet independence; **packet-loss concealment** and **in-band FEC** (redundant coarse copies of previous frames); bitrate re-negotiable every 20 ms. None of these features improves ratio — every one of them buys survival on Axis 3, which is the whole point.
- **Job C, extreme edition: neural codecs — Lyra (2021), SoundStream, Encodec.** Push Axis 5 to the limit: both endpoints ship megabytes of trained weights, so the wire carries ~3 kbps (vs Opus's ~24+ for comparable speech). Stack: **learned encoder** (a transform, same role as MDCT, just learned) → **residual VQ** into discrete tokens (the quantizer) → **learned decoder that _generates_ plausible audio** from the tokens. It's the same three lossy roles as JPEG — transform, quantize, pack — with two stages swapped for neural networks and the "irrelevance model" learned instead of hand-built. Costs, per the framework: compute per sample, and a new failure mode — reconstruction is _plausible_ rather than _faithful_ (the decoder can hallucinate), which redefines the fidelity contract itself. Bonus: those RVQ tokens are exactly what audio-generating LLMs operate on — compression tokens became a _representation_.

### 3.4 Video

**The boss-level job, so state it fully:** raw 1080p30 truecolor is 1920 × 1080 × 3 B × 30 fps ≈ 187 MB/s ≈ **1.5 Gbps**; a good stream ships at ~5 Mbps, so the job is **~300:1** — achievable only because video is drowning in _every_ redundancy type at once: temporal (frames nearly repeat), spatial (smoothness), statistical, and perceptual. Plus brutal Axis 3/4 constraints: seeking, broadcast, packet loss (calls), and decode **in silicon on billions of devices** — which is why video codecs are decade-scale frozen standards with committee names.

**The hybrid codec architecture.** Every mainstream codec since ~1990 — MPEG-2, H.264/AVC, HEVC, VP9, AV1, VVC — is the _same machine_ with better parts. One pass through it, mapped to toolkit roles:

1. **Partition** the frame into blocks (16×16 macroblocks → adaptive trees up to 128×128). _Framing._
2. **Predict each block**, choosing per-block between:
    - **Intra prediction** — extrapolate from already-decoded neighboring pixels in the _same_ frame (_spatial predictor_; a still image codec is this stage alone);
    - **Inter prediction / motion compensation** — copy a block from a **reference frame**, displaced by a transmitted **motion vector** with sub-pixel precision (_temporal predictor_ — the "reference frames + deltas" machinery you asked about). Frame types: **I** (intra-only: seek points, error recovery, expensive), **P** (predicts from past frames), **B** (predicts from past _and_ future — best ratio, needs buffering ⇒ banned in low-latency configs). The I-to-I spacing (**GOP length / keyframe interval**) is the Part-2F ratio↔seekability dial, set per job: ~2 s for streaming, infrequent-plus-gradual-refresh for video calls.
3. **Transform the residual** (integer DCT-like, plus wavelet-ish options in newer codecs) — _decorrelate what prediction missed_.
4. **Quantize** the coefficients — the only lossy step, and the knob **rate control** turns per-block to hit a target bitrate (CBR for broadcast, VBR/CRF for files) while a perceptual model decides _where_ quality is spent.
5. **Entropy-code everything** — modes, motion vectors, coefficients — with **CABAC**-style context-adaptive binary arithmetic coding: dozens of contexts ("is the neighbor block skipped?") sharpen every probability. Adaptive-arithmetic ratio was worth hardening into silicon here because Axis 3 says decode-many-forever.
6. **In-loop filters** (deblocking etc.) clean the reconstruction _inside_ the prediction loop, so future frames predict from the filtered result — a feedback detail unique to video's predictor-of-predictions structure.

**The generational story is then one sentence:** H.264 (2003) → HEVC (2013) → AV1 (2018) → VVC (2020) each buy roughly **30–50% bitrate at equal quality for ~5–10× encoder complexity**, by adding _more and smarter prediction_ (more intra modes, flexible partitions, better motion) and _sharper entropy coding_ — never by changing the skeleton. Moore's law is spent almost entirely on the encoder, because Axis 3 says encode-once-decode-many and Axis 4 keeps decoders cheap.

**Same medium, different job, different config:** a video _call_ takes the same codec and flips the Axis-3 switches — no B-frames, tiny or intra-refresh GOPs, slice-per-packet, reference-frame selection driven by receiver ACKs (predict only from frames the other side confirmed). Compare with Opus-vs-MP3: identical pattern.

### 3.5 Tensors, ML data, and scientific arrays

**The job:** giant float arrays; the consumer is a _training run or inference pass_, so "distortion" = accuracy loss on a task — a consumer model you can actually measure, unlike an eyeball.

- **Lossy, in the compute path: block-scaled quantization (NVFP4, MXFP4, block floating point).** Split tensors into blocks of 16–32 elements; each block shares one higher-precision **scale** (sometimes with a second per-tensor scale layered on top); each element keeps a ~4-bit value. Framework reading: the _redundancy_ is that neighboring weights/activations share magnitude (so the exponent is nearly block-constant — pull it out once); the _irrelevance_ is that networks tolerate coarse mantissas if dynamic range and a few outliers survive (hence the scale hierarchy and outlier tricks). It's JPEG's rate–distortion logic with a loss curve as the eye. Calibration (choosing scales against real activation statistics) plays the role of the psychoacoustic model.
- **Lossless, at rest: checkpoints and datasets.** Raw float32 barely compresses — mantissas are noise (Axis-2 numeric structure). The standard move is rearrange-then-general-purpose: **byte-shuffle** (group sign/exponent bytes, which are highly skewed, away from mantissa bytes) **→ zstd**. Same principle in HDF5/Blosc for scientific arrays, and error-bounded lossy variants (ZFP, SZ) for simulation output where "consumer" = analyses with known tolerance.
- **VQ as representation.** The neural-codec pattern from 3.3 generalizes: learned transform + vector quantization turns any modality into discrete tokens — compression machinery quietly becoming the interface format of modern ML.

### 3.6 Storage systems, deduplication, and sync

The least codec-like domain, and the best test of the framework — because it's _pure Axis-2 cross-item redundancy + Axis-5 shared knowledge_, with hardly any new algorithms.

- **Deduplication (backup systems, VM storage: Data Domain, borg, restic).** The redundancy is whole repeated _chunks_ across terabytes and across days of snapshots — far beyond any LZ window. So rebuild LZ at macro scale: split streams into chunks, fingerprint each (SHA-family hash), store each unique chunk once, and represent files as chunk-reference lists. The one sharp idea is **content-defined chunking**: fixed-size chunks break catastrophically when one byte is inserted (every subsequent boundary shifts — nothing matches anymore), so instead cut chunks where a **rolling hash** of the local bytes hits a pattern; boundaries then stick to _content_, and an insertion perturbs only its own chunk. That's the whole trick. Per-chunk zstd runs downstream (different redundancy, per rule 4's converse: stack eaters of _different_ redundancy).
- **rsync — delta sync across a network.** Job: update a remote copy of a file cheaply. Framework reading: it's LZ77 where the match window is _the old version of the file, sitting on the other machine_ (Axis 5: the receiver's copy is the shared dictionary). Mechanics: receiver sends per-block weak-rolling + strong checksums of its old file; sender slides the rolling checksum over the new file, finds matching blocks, and transmits only literals + block references. The rolling hash plays the same role as in dedup: finding matches at arbitrary alignment, cheaply.
- **git packfiles.** Loose objects are just zlib. Packs add a match layer above deflate: heuristically sort similar objects (same path, similar size — versions of one file land together), **delta-encode each object against a similar one** (chains of deltas), then deflate the result. Repeats-across-versions eaten by delta; leftover skew eaten by zlib. Two eaters, two redundancies, correct order.
- **Columnar analytics formats (Parquet, ORC).** A table stored row-wise interleaves unlike statistics — the 2D layout sin. Store it **column-wise** and each column becomes homogeneous, low-entropy data begging for the light toolkit: **dictionary encoding** (few distinct values → small indices), **RLE + bit-packing** (sorted/repetitive columns), **delta** (timestamps, IDs) — then generic zstd/snappy per page as the back-end. This is the "group like statistics" principle from 2D promoted from bytes to database schema, and it's why the same table is often 5–10× smaller in Parquet than in gzipped CSV _and_ faster to query (decode only the columns you touch — Axis 3 random access, by construction).

### 🗺 Cheatsheet 3 — Domains at a Glance

- **General lossless:** unknown bytes ⇒ match + entropy, UX = speed dial
    - deflate `LZ77/32K → Huffman` (the frozen substrate) · bzip2 `BWT→MTF→RLE→Huff` · xz `bigLZ→adaptive range` (archival corner) · LZ4 `LZ only` (GB/s corner) · **zstd** `LZ→FSE/Huff + dictionaries` (the modern span) · Brotli (web-tuned dict) · PAQ/LLM (ratio-unbounded fringe)
- **Images:** split on fidelity contract
    - Lossless graphics: PNG `row predictors → deflate` (GIF: palette-VQ → LZW, retired)
    - Lossy photos: JPEG `YCbCr+subsample → 8×8 DCT → perceptual quant → zigzag/RLE → Huffman` — the canonical lossy stack
    - Moderns = video intra frames (WebP/HEIC/AVIF) or JPEG XL (VarDCT+ANS); adoption gated by decoder ubiquity
- **Audio:** three jobs, three designs
    - Archive: FLAC `LPC → Rice` (predict-and-code, ~2:1)
    - Distribute: MP3/AAC `MDCT → psychoacoustic quant → Huffman` (~10:1; latency free ⇒ long windows)
    - Converse: Opus `SILK speech-LPC + CELT short-MDCT` + PLC/FEC (~20 ms frames; every feature buys Axis-3 survival)
    - Converse, extreme: Lyra/Encodec `learned encoder → RVQ tokens → generative decoder` (~3 kbps; knowledge lives in weights; fidelity → plausibility)
- **Video:** ~300:1 via every redundancy at once; one skeleton since 1990
    - `partition → predict (intra | motion-comp vs reference frames; I/P/B, GOP dial) → residual DCT → quantize (rate control) → CABAC → in-loop filters`
    - Generations (H.264→HEVC→AV1→VVC): −30–50% bits for ~10× _encoder_ cost — decode stays silicon-cheap
    - Calls = same codec, low-latency switches: no B-frames, intra refresh, ACK-driven references
- **Tensors:** consumer = task accuracy
    - In-path lossy: block-scaled formats (NVFP4/MX): shared per-block scale + 4-bit values; calibration = the "psychoacoustic model"
    - At-rest lossless: byte-shuffle → zstd (exponents compress, mantissas are noise); error-bounded ZFP/SZ for sims
- **Storage/sync:** cross-item redundancy + shared knowledge, few new algorithms
    - Dedup: content-defined chunking (rolling hash) → fingerprints → store once; per-chunk zstd downstream
    - rsync: LZ whose window is the receiver's old copy; rolling checksum finds unaligned matches
    - git packs: delta vs similar objects → zlib
    - Parquet: columnar layout → dictionary/RLE/delta per column → zstd (grouping-like-statistics at schema scale)

---

## Closing — How to Attack a New Compression Problem

The framework, compiled into a checklist:

1. **State the job on the five axes** (fidelity contract & consumer · redundancy profile · access pattern · compute envelope · where knowledge can live). Most of the design is already decided here.
2. **Lossy allowed?** If yes, find the consumer model first — it defines irrelevance, which is where the big ratios are. Pick transform + quantizer to match (perceptual tables, block scales, VQ...). If no, skip to 3.
3. **Inventory the redundancy** and assign one eater per type, in pipeline order: rearrange/split unlike statistics apart → match repeats (size the window to the repeat distance — file? corpus? other machine?) → predict smoothness (encode residuals) → entropy-code the leftovers.
4. **Set the structural dials from Axis 3, not from the data:** block/keyframe/reset intervals per your seek, parallelism, and loss requirements — and accept the ratio you pay for them.
5. **Spend compute where the symmetry says:** encode-once-decode-many ⇒ buy ratio with encoder search and keep the decoder table-driven; symmetric/real-time ⇒ greedy single-pass everything.
6. **Small inputs?** You need shared prior knowledge — train a dictionary or ship a model.
7. **Verify with the noise test:** if the output re-compresses, a redundancy survived — find which type and which stage let it through.
8. **Before building anything: check whether zstd (plus a trained dictionary, plus maybe a byte-shuffle or delta pre-pass) already solves it.** It usually does — and now you know exactly why.
