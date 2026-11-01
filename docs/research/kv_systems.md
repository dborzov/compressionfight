# KV-cache compression: the SYSTEMS context ("the job")

Research notes for the game. Scope: where the KV cache lives, how attention reads it, and what sparse
attention changes. The codecs themselves (KIVI, CacheGen's coder, KVTC, DeltaKV, ShadowKV...) are
another slice; here they appear only as "who lives in this job". The player is assumed to know what K and V are.

Register: `req` stats are 0..5 as in `js/data.js` (`CYC.METAPHORS`): latency = ALARM TIMER, encBudget =
PLANNING TIME, decBudget = GETAWAY RIDE, ra = VAULT DOORS, bandwidth = GETAWAY VAN, decMem = GLOVEBOX,
lossless = LEAVE NO TRACE, priorOK = INSIDE MAN ALLOWED. Chairs: LENS (transform), ORACLE (predict),
BUTCHER (quantize), PACKER (entropy), CON ARTIST (perceptual), INSIDE MAN (prior).

Sources are cited inline as [Sn]; list at the end. "Derived" marks arithmetic I did from config values.

---

## 0. The numbers card (for the tutor)

**Size of one token's KV, dense attention, 16-bit:**
`bytes/token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes`

| Model | layers × kv_heads × head_dim | bytes/token (bf16) | 128K context |
|---|---|---|---|
| Llama-2-7B (MHA, 32 kv heads) | 32 × 32 × 128 | 524,288 = 512 KiB | 64 GiB (derived) |
| Llama-3-8B (GQA, 8 kv heads) | 32 × 8 × 128 | 131,072 = 128 KiB | 16 GiB (derived) |
| Llama-3-70B (GQA, 8 kv heads) | 80 × 8 × 128 | 327,680 = 320 KiB [S10][S11] | 40 GiB (the game's `kvcache` brief says exactly this) |
| DeepSeek-V3 / V3.2 (MLA) | 61 × (512 latent + 64 RoPE) shared by K and V | 61 × 576 × 2 = 70,272 ≈ 68.6 KiB (derived from config [S12]) | 8.4 GiB |
| DeepSeek-V3.2 + indexer key cache | + 61 × 128 × 2 | 85,888 ≈ 83.9 KiB (derived; vLLM states 83.9 GiB at 1M tokens [S7]) | 10.5 GiB |
| DeepSeek-V3.2 MLA in FP8 (FlashMLA layout) | 61 × 656 B | 40,016 ≈ 39 KiB (656 B/entry from ESS [S4]) | 4.9 GiB |
| GLM-5.1 (DSA, 78 sparse layers) | — | "about 100 KB" per token [S2]; 13.09 GB at 128K [S2] | 13 GB |
| DeepSeek-V4 (CSA + HCA, bf16) | — | 9.62 GiB per sequence at 1M context [S7] ≈ 10 KiB/token (derived) | ~1.2 GiB |

Notes. MLA caches ONE latent per token per layer that serves as both K and V, so there is no ×2; that is why 61
layers of a 671B model cost less than 32 layers of an 8B GQA model. The 656-byte FP8 entry decomposes as
512 B of FP8 latent + 16 B of fp32 block scales + 128 B of bf16 RoPE key (64 × 2) = 656 (my decomposition;
the total is from [S4], the layout is inferred, flag as unverified). ESS says the indexer cache is 16.8% of total
cache storage [S4], consistent with a 128-byte FP8 indexer key per layer per token (derived).

**Where it lives and how fast the pipes are** (vendor numbers, [S13]):

| Tier | Capacity | Bandwidth |
|---|---|---|
| HBM, H100 SXM | 80 GB | 3.35 TB/s |
| HBM, H200 | 141 GB | 4.8 TB/s |
| HBM, B200 | 180–192 GB (sources disagree) | ~8 TB/s |
| NVLink 4 (Hopper) / NVLink 5 (Blackwell), per GPU | — | 900 GB/s / 1.8 TB/s bidirectional |
| NVLink-C2C (Grace Hopper CPU↔GPU) | Grace LPDDR ~480 GB per CPU [S2] | ~900 GB/s |
| PCIe Gen5 ×16 (CPU↔GPU on H200 nodes) | host DRAM: hundreds of GB to TBs (HiSparse node: "22 TB", 11 TB pinned [S2]) | 64 GB/s per direction nominal; ESS measured 37 GB/s H2D with a GPU-side gather kernel and **0.79 GB/s** with cudaMemcpyAsync on scattered 656-byte entries [S4] |
| NVMe SSD / network (prefix caches, CacheGen) | TBs | single-digit GB/s; WAN much less |

**Bandwidth is the stat.** Decode is a per-token memory-bound loop: every generated token re-reads the whole
cache (dense) or the selected part of it (sparse). Llama-3-70B at 128K, batch 1: 40 GiB of KV per token at
4.8 TB/s = ~9 ms per token on an H200 before any weights are read (derived). SemiAnalysis derives a decode
arithmetic intensity of ~120.8 FLOP/byte for GLM-5 in MQA mode against an H800 practical peak of ~258, i.e.
bandwidth-bound even when sparse [S9]. HiSparse: GLM-5.1 at top-k 2048 and ~100 KB/token reads roughly
200 MB per generated token, ~7 GB/s per request at 30 ms TPOT [S2]. And capacity: Qwen3-32B with 4-bit
weights runs out of a 24 GB GPU after about 24,000 generated tokens [S3]; GLM-5.1 full KV saturates an
8×H200 node near 60 concurrent 32K requests (~240 GB of KV) [S2].

---

## 1. Taxonomy of KV-cache jobs as heist profiles

The same bytes, four very different vaults. The existing heist `kvcache` is JOB A.

### JOB A — THE CONTEXT WINDOW (shrink the cache in HBM during decode)

brief: "The cache lives in HBM and attention reads every cached token of every layer, every step. Make each
token smaller so more tokens and more users fit, and so the per-token read is cheaper. New tokens arrive one
at a time; the codec runs inside the forward pass."

req: `{ lossless: false, latency: 0, encBudget: 0, decBudget: 0, ra: 5, bandwidth: 4, decMem: 1, priorOK: false }`

reqNotes:
- Online and symmetric: one token (or one block of 32–128 tokens) is encoded the moment it is produced. No
  second pass, no calibration on this data. A small fp16 "residual window" of the newest tokens is the only
  buffering allowed (KIVI).
- Decoder is the attention kernel: dequantize a block in registers, dot with q, done. No tables, no
  variable-length codes, no sequential state. The Packer's chair stays empty.
- Random access is total for dense attention and per-selected-token for sparse attention; either way every
  token must be decodable alone.
- Bandwidth = HBM bandwidth. Fewer bytes per token is fewer bytes per decode step, which is time per token.
- Granularity is the serving system's page (16 tokens in vLLM-style paged attention [S3], 256 native
  positions in vLLM's DeepSeek-V4 recipe [S7]). The codec's block must divide the page.
- Keys have outlier channels, values do not: per-channel scaling for K, per-token for V, or rotate first
  (the ROTATE & CLAMP combo).
- priorOK false by default. A Hadamard rotation is data-independent and free; a data-fitted codebook or
  calibrated clipping (KVQuant, CommVQ) is a legitimate INSIDE MAN but ties the codec to one model.

Lives here: KIVI, KVQuant, QuaRot/SpinQuant-rotated int4 KV, FP8 KV in vLLM/SGLang/TensorRT-LLM, DeepSeek's
token-wise FP8 latent cache at decode [S7], CommVQ, Coupled Quantization.

### JOB B — THE WAREHOUSE RUN (offload to CPU DRAM, fetch sparsely over PCIe)

brief: "Sparse attention only touches top-k = 2048 tokens per step, so the rest of a 128K context can sit in
host DRAM. Each step the GPU asks for ~2048 scattered entries per layer; the misses must cross PCIe before
that layer's attention runs. The channel is a 64 GB/s pipe shared by every request on the GPU."

req: `{ lossless: false, latency: 1, encBudget: 1, decBudget: 1, ra: 5, bandwidth: 5, decMem: 2, priorOK: true }`

reqNotes:
- Random access at entry granularity: the minimum swap unit is a single token's layer-local KV record
  (656 B in ESS [S4], one "KV record" per selected position in HiSparse [S2]). The access pattern is a
  gather of ~2048 arbitrary positions, decided by the indexer microseconds before the read.
- Latency 1: the fetch must complete inside one layer's attention critical path (~60 µs per layer on H200
  for GLM-5.2 [S2]) or be prefetched a few layers ahead. HiSparse budgets 112 µs of IO per layer at batch 16
  over PCIe vs 29 µs over NVLink-C2C [S2].
- Bandwidth 5: PCIe is the scarce resource, not HBM. All-miss traffic for GLM-5.1 would be ~200 MB per
  generated token per request; "a dozen concurrent requests per GPU would saturate" Gen5 ×16 on misses alone
  [S2]. With an LRU hot buffer the miss rate is 13.4% (B = 4096) so the real draw is ~1 GB/s per request.
- Encoder may be slightly heavier than JOB A: entries are written to host once (at prefill, or on eviction
  from the hot buffer) and read many times. A block-wise codec with a per-block scale, or a low-rank key
  projection, is affordable. Still no iterative search per token.
- Decoder: dequantize on the GPU after the gather, before attention. A fused "gather + dequant" kernel is the
  getaway ride; 2048 entries × 78 layers per step.
- decMem 2: the hot buffer. HiSparse keeps B ∈ [2k, 4k] records per request per layer in HBM (~0.4 GB per
  request at 128K vs 13.09 GB dense) plus a page table and LRU metadata [S2].
- The indexer's own keys stay in HBM, uncompressed or lightly compressed (see §3). They are not part of this
  job's payload; they are its address book.
- priorOK true: the serving system owns both ends, so trained codebooks or model-specific projections are
  fine (ShadowKV's low-rank keys, PQCache's PQ codebooks).
- Scattered small transfers are the trap: cudaMemcpyAsync per entry reached 0.79 GB/s; a GPU kernel doing
  address-based loads from pinned memory (UVA) reached 37 GB/s [S4].

Lives here: HiSparse (SGLang) [S1][S2], ESS (Baidu, simulator study: +69.4% throughput at 32K, +123% at
128K) [S4], ShadowKV, SpeCache, PQCache, DeltaKV, InfiniGen, TensorRT-LLM KV-cache compression at the tier
boundary, HiCache/LMCache tiering. HiSparse results: 2.1–4.7× throughput at 32K–200K, negligible gain at 4K [S2].

### JOB C — THE COLD STORAGE (persist or stream a prefix cache: disk, network, another node)

brief: "A long shared prefix (a document, a system prompt, an agent's history) was prefilled once. Store its
KV cache and ship it to whichever GPU needs it next, instead of recomputing. The whole prefix moves at once;
decode happens once on arrival. The channel is a network or an SSD."

req: `{ lossless: false, latency: 4, encBudget: 3, decBudget: 2, ra: 1, bandwidth: 5, decMem: 2, priorOK: true }`

reqNotes:
- Offline-ish encode: the full prefix is available, so look-ahead across tokens, per-layer statistics, delta
  coding against anchor tokens and a real entropy coder are all on the table. CacheGen uses exactly these:
  KV-cache size reduced 3.5–4.3×, total context-loading delay 3.2–3.7× lower than KV-reuse baselines, with
  adaptive compression levels per chunk as bandwidth changes [S5].
- Decode once, on the GPU, into the serving system's native layout (fp16/FP8 pages). Then it is JOB A's
  problem again. Decoder cost per byte can be 100× JOB A's because it runs once per prefix, not once per step.
- Random access 1: chunk-level (per layer, per group of tokens), not per token. Variable-length codes are
  fine inside a chunk.
- Bandwidth 5: a 128K Llama-3-70B prefix is 40 GiB; at 10 Gbit/s that is 32 s, longer than recomputing it.
  The codec exists to make loading faster than prefill.
- The alternative squad is "the empty chair": recompute the prefix (prefill) or store it raw (Mooncake, vLLM
  prefix caching, HiCache tiers). SemiAnalysis' InferenceX trace shows the tiering at work: GPU-memory prefix
  reuse fell from 90.3% to 54.8% as concurrency went 8→16 while host-memory reuse rose 6.0%→40.3%; overall hit
  rate stayed above 95% [S9].

Lives here: CacheGen (SIGCOMM 2024), LMCache, KVTC (transform coding), Mooncake / HiCache raw tiering.

### JOB D — THE PURGE (eviction / selection: quantize unimportant tokens to zero bits)

brief: "Roughly 20% of tokens collect 80% of attention weight [S3]. Drop the other 80% entirely. The codec
is a one-bit decision per token: keep or discard. The catch: who tells you which tokens matter, and does
dropping them actually free memory?"

req: `{ lossless: false, latency: 0, encBudget: 1, decBudget: 0, ra: 5, bandwidth: 3, decMem: 1, priorOK: false }`

reqNotes:
- This is the degenerate BUTCHER: a quantizer with two levels, "all bits" and "zero bits", chosen per token
  by an importance score. Everything about rate-distortion still applies; the distortion is measured in
  AIME/MATH accuracy (SnapKV at a 2,048-token budget: AIME-24 34.6% vs 57.1% dense [S3]).
- The score problem: H2O-style cumulative attention scores need the N×N attention matrix, and FlashAttention
  never writes it to HBM. The reference H2O falls back to eager attention [S3]. Deployable variants score
  once at the end of prefill from a short observation window (SnapKV; only ~25 recent queries are
  informative under RoPE [S3]), use geometry that needs no scores (StreamingLLM's sinks + window;
  TriAttention's pre-RoPE key geometry), or compact only every ~128 decoded tokens [S3].
- The freeing problem: paged KV frees a block only when it is completely empty. Evict 14,400 of 16,000 tokens
  and the 1,600 survivors sit in ~1,000 of the original ~1,000 blocks: "VRAM usage is unchanged" [S3]. You
  must physically compact survivors into fresh blocks (forward packing), which costs copies (3 vs 18 copies
  for hole-filling vs order-preserving in the blog's example).
- Selection WITHOUT eviction (Quest, DSA top-k, InfLLM) is not this job: it saves bandwidth and compute but
  keeps every token, so memory still grows with context [S3]. That is JOB B's premise.
- Lossless is impossible by construction; the game's "lossless" stat is false and the CON ARTIST chair is the
  interesting one: the importance model decides where to be brutal.

Lives here: StreamingLLM, H2O, Scissorhands, TOVA, SnapKV, PyramidKV, Ada-KV, R-KV, TriAttention (the only
one in [S3] that addresses both infra problems; 3,072-token budget matches dense AIME-25 at 2.5× throughput
and 10.7× less KV memory for Qwen3-8B [S3]).

### Variants worth a line

- JOB A′ — per-token FP8 at decode only, bf16 at prefill (DeepSeek-V3.2/V4 practice [S7]): prefill writes
  the cache in bulk and is compute-bound; decode is bandwidth-bound and reads it. Compressing at the
  prefill→decode handoff is free encode time.
- JOB B′ — disaggregated prefill/decode: the prefill node ships the whole cache to the decode node's host
  pool over the network [S2]. JOB C's channel, JOB B's consumer.

---

## 2. Systems constraints that rule fighters in or out (as requirement notes)

1. **The decoder is a GPU kernel, not a program.** Every thread must find its element by arithmetic on an
   index. This kills variable-length codes (WOODCUTTER, FENCER, STACKER, RULER) in JOBS A, B and D. Only
   fixed-length codes survive: block integers (BLOCKY), VQ/PQ indices (CATALOG, SPLITTER), low-rank
   projections (AXIS). The PACKER is allowed only in JOB C, where decoding happens once, in bulk.
2. **FlashAttention never materializes the scores.** Any method that wants "which tokens did attention look
   at" must either rebuild that signal from queries and keys itself (an indexer, a page summary, a trained
   proxy) or drop FlashAttention and lose far more than it saves [S3]. For the game: the CON ARTIST for KV
   caches cannot peek at the Butcher's output; it must predict importance from the data before the kernel runs.
3. **Paged memory frees nothing until a page is empty.** Blocks of ~16 tokens (vLLM default) or 256 native
   positions (DeepSeek-V4 recipe) are the allocation unit. Per-token dropping does not return memory;
   compaction does, at copy cost [S3][S7]. For codecs: the block of a quantizer should tile the page (32 or
   64 tokens, or a layer-local record), and headers belong per page, not per token.
4. **Prefill writes in bulk, decode appends one token.** Prefill is compute-bound and has the whole prompt
   in view (good encoder conditions: statistics, anchors, look-ahead). Decode is bandwidth-bound and sees one
   token per step (no statistics, no look-ahead). A codec may legitimately have two encoders: a smart one
   for the prompt and a dumb one for generated tokens, as long as the decoder is one kernel. DeepSeek uses
   bf16 at prefill and FP8 at decode [S7].
5. **Batch serving shares the channel.** HBM bandwidth and PCIe bandwidth are divided among every request on
   the GPU. Per-request numbers (7 GB/s, 200 MB/token) multiply by concurrency; capacity caps batch size and
   batch size is throughput [S2][S4][S9]. Smaller tokens buy batch, not just context.
6. **The indexer is the new address book.** Under DSA every token keeps a small key (128 dims, FP8; FP4 in
   V4) that is read in full every step [S6][S7][S12]. It must stay in HBM and stay fast; its precision is
   its own, separate quantization problem.
7. **Top-k selection without offload does not shrink memory.** "Top-k selection typically requires the full
   context to be in HBM" [S9]; sparse attention cut compute, which made memory the binding constraint and
   spawned JOB B.
8. **Scattered small transfers over PCIe are a different animal from bulk copies.** 2048 entries of 656 B at
   arbitrary host addresses: a copy engine managed per entry gets 0.79 GB/s, a GPU gather kernel 37 GB/s of
   a 64 GB/s nominal link [S4]. Entry size, alignment and whether the gather can be a single kernel decide
   whether JOB B is possible at all.
9. **Prefix caching and disaggregation want a self-contained cache.** Compressor state for V4's compressed
   layers is managed "like sliding-window KV" so prefix caching and disaggregated prefill still work [S7]. A
   codec whose state depends on everything before it (an adaptive context model, an LZ window) breaks
   sharing of prefixes across requests.

---

## 3. What sparse attention changes

**DSA (DeepSeek-V3.2, GLM-5.x).** A "lightning indexer" with its own small queries and keys scores every
past token: I_{t,s} = Σ_j w_{t,j} · ReLU(q^I_{t,j} · k^I_s) over H^I indexer heads; the main attention then
reads only the top-k = 2048 latent entries [S6]. Config: 64 indexer heads, head dim 128, top-k 2048 [S12]. The
indexer "has a small number of heads and can be implemented in FP8" [S6]; TensorRT-LLM keeps the indexer K
cache and its MQA math in blockwise FP8 [S14]. The indexer is still O(L²) per token, attention becomes O(Lk).
Below 2K tokens attention is dense [S9]. DSA runs on the MQA mode of MLA: one latent per token shared by all
query heads, so a selected entry pays for 128 heads at once [S6].

What this does to the cache:
- Two caches per token per layer now: the **latent** (576 × bf16, or 656 B in FP8) and the **indexer key**
  (128 × FP8 ≈ 128 B; "a few hundred bytes per token" including page-table entries in HiSparse [S2]; 16.8%
  of total cache storage in ESS [S4]). The latent is read for 2048 tokens per step; the indexer key is read
  for ALL tokens per step.
- Consequence for codecs: the indexer key must stay resident and dense-readable (it IS the scan). The latent
  can be compressed harder and even moved off-GPU, because only 2048 of them per layer are touched. HiSparse
  and ESS both keep the indexer cache on GPU and offload only the latent [S2][S4].
- Temporal locality is high: consecutive steps "reselect largely overlapping positions". Keeping only the
  current top-k set and nothing else loses ~30% of the next step's selections; an LRU buffer of 2k gets
  13.4% misses, 4k gets 6.7%, Bélády's optimum at 4k is 8.2%; FIFO 17.2% and random 16.1% at 4k (GLM-5.1,
  100,384-token LongBenchV2 prompt, 78 layers, first 1,000 decode steps) [S2]. ESS: misses per layer per batch
  "16.66 to 605" at a 0.2 GPU-resident ratio, stable above 0.2; recommends a GPU buffer ≥ 6.4K entries [S4].
- **IndexShare** (GLM-5.2): one indexer per group of 4 layers, 21 anchor + 57 shared layers out of 78;
  indexer cache and indexer FLOPs down 75% [S9][S2]. Shared selections also allow exact layer-wise prefetch:
  the anchor layer's misses are known before the shared layers run, hiding "roughly half" of IO [S2].

**HiSparse hierarchy (SGLang).** Host pinned DRAM holds the authoritative full KV; per request and per layer
the GPU holds a B-slot LRU buffer (B ≥ k; useful range 2k–4k), a page table and the selection state. One
fused CUDA kernel per layer resolves top-k against the buffer, picks LRU victims, fetches misses with GPU-side
loads from pinned memory, updates the page table and emits device locations for attention. New tokens are
written through to GPU and backed up to host on a side stream. Exposed IO at low concurrency 7–8 ms per token
synchronous, ~3 ms with prefetch; TPOT 15.9 vs 16.0 ms dense for DeepSeek-V4 at concurrency 16 [S2].

**CSA / HCA (DeepSeek-V4)** [S7][S8]. V4 compresses the sequence itself before caching:
- c4a ("Compressed Sparse Attention"): each cached entry is a learned weighted sum of 8 tokens at stride 4,
  so the cache is ~1/4 the length; DSA top-k = 512 compressed entries ≈ 2048 raw tokens. 30 of 61 layers.
- c128a ("Heavily Compressed Attention"): one entry per 128 tokens (stride 128), ~1/128 the length; top-k
  8192 is effectively dense at 1M context. 31 of 61 layers, no indexer.
- A 128-token sliding window of uncompressed KV keeps local context; K and V are shared (another 2×, with an
  inverse RoPE on the output). FP8 attention cache and FP4 indexer cache in the recommended recipe.
- Result: 9.62 GiB per 1M-token sequence in bf16 vs 83.9 GiB for a V3.2-style stack (8.7×); NVIDIA quotes
  "90% reduction in KV cache memory" and 73% fewer per-token FLOPs vs V3.2 [S8]. Serving cost: five cache
  kinds in three page sizes; a logical block of 256 native positions holds 64 c4a or 2 c128a entries [S7].

For the game: CSA/HCA is a transform + quantizer baked into the architecture (a learned, strided, lossy
"temporal downsample" of the token axis), chosen at training time. It is the model-level chair (§4), and it
shrinks the job for every codec downstream: fewer, denser entries, each one worth more.

---

## 4. The survey's taxonomy mapped to the chairs

Li et al. (TMLR 2025) [S15][S16] split the field into token-level, model-level and system-level. The game's
chairs cover token-level; model-level is "architecture decisions outside the codec"; system-level is the
heist profile itself.

| Survey category | Representative methods [S16] | Game mapping |
|---|---|---|
| Token-level: **quantization** (fixed-precision, mixed-precision, outlier redistribution) | KIVI, KVQuant, GEAR, ZipCache, QJL, QuaRot, SpinQuant, SmoothQuant, QServe | BUTCHER (BLOCKY, CATALOG/SPLITTER) + LENS (AXIS for rotations). The core of JOB A. |
| Token-level: **low-rank decomposition** (SVD, tensor, learned) | Palu, ShadowKV, Eigen Attention, LoRC, DecoQuant, LESS, MatryoshkaKV | LENS (AXIS/PCA; NEURON for learned). Keys are low-rank, values much less so. |
| Token-level: **selection** (static; dynamic with eviction; dynamic without eviction) | SnapKV, H2O, Scissorhands, Keyformer, NACL, Quest, InfLLM, RetrievalAttention, ClusterKV | JOB D (with eviction) and the premise of JOB B (without). CON ARTIST decides importance; BUTCHER quantizes to 0 bits. |
| Token-level: **budget allocation** (layer-wise, head-wise) | PyramidKV, PyramidInfer, Ada-KV, RazorAttention, DuoAttention, HeadKV | CON ARTIST / rate control: where to spend bits (the waterfill quest's logic across layers and heads). |
| Token-level: **merging** (intra-layer, cross-layer) | CaM, D2O, Dynamic Memory Compression, LOOK-M, MiniCache, KVSharer | ORACLE + BUTCHER: predict one token from its neighbour and keep the residual, or VQ along the token axis. Cross-layer merging is the ancestor of xKV. |
| Model-level: **attention grouping and sharing** | GQA, QCQA, Weighted GQA, Cross-Layer Attention, Layer-Condensed KV, MLKV, Value Residual Learning | Outside the codec: "compression baked into the model". GQA = 8× fewer KV heads (Llama-3-70B: 320 KiB vs 2.5 MiB with MHA). Fewer things to store rather than smaller things. |
| Model-level: **architecture alteration** | DeepSeek-V2 (MLA), Infini-attention, YOCO, CEPE, XC-Cache, Block Transformer | MLA is a trained LENS with a trained decoder (the up-projection) inside the model: a learned low-rank code (576 dims) that attention reads directly, never reconstructing K and V. CSA/HCA extends this along the token axis. |
| Model-level: **non-transformer / hybrid** | RWKV, Mamba, RetNet, MixCon, GoldFinch | Constant-size state: the cache is gone, replaced by a fixed recurrent summary (the ultimate "quantize history to a fixed budget"). |
| System-level: **memory management** (architectural, prefix-aware) | vLLM PagedAttention, vTensor, eLLM, ChunkAttention, MemServe | The page granularity constraint of §2; prefix caching = JOB C's reason to exist. |
| System-level: **scheduling** | SGLang, BatchLLM, FastSwitch, LayerKV, CachedAttention, ALISA | Batch serving (§2.5); who gets HBM when. |
| System-level: **hardware-aware** (I/O, heterogeneous, SSD) | FlashAttention, Bifurcated Attention, HCache, NEO, FastDecode, InfiniGen, HeadInfer, InstInfer, FlexGen | The tiers of §0 and JOB B/C: FlashAttention is the kernel constraint, heterogeneous/SSD designs are the offload jobs. |

Not in the list but belonging to the map: StreamingLLM (selection, static), CacheGen (system-level, network
streaming), DSA / HiSparse / ESS (post-survey; selection-without-eviction plus heterogeneous memory).

The lesson for the player: the biggest "compressions" in the table (GQA 8×, MLA ~5× vs GQA, CSA/HCA ~9× vs
MLA) happened in the model-level row, at training time, by people who never thought of themselves as writing
a codec. The codec chairs work on what is left.

---

## 5. Quiz-grade facts and Fixer lines

### Facts (each with its source)

1. KV bytes per token = 2 × layers × kv_heads × head_dim × bytes. Llama-3-70B in bf16: 327,680 B = 320 KiB;
   128K context = 40 GiB [S10][S11].
2. MLA caches one 576-dim latent per token per layer, shared between K and V; DeepSeek-V3's 61 layers cost
   ~68.6 KiB per token in bf16, about a fifth of Llama-3-70B despite 10× the parameters (derived from [S12]).
3. In FlashMLA's FP8 format a latent entry is 656 bytes [S4]; DSA's top-k 2048 gather per layer is therefore
   ~1.3 MB of scattered reads per layer per request (derived).
4. FlashAttention never writes the N×N attention matrix to HBM, so eviction methods that need attention scores
   (H2O) cannot run on it; the reference H2O uses eager attention [S3].
5. vLLM-style paged attention allocates ~16-token blocks and frees a block only when every token in it is
   gone; evicting 14,400 of 16,000 tokens can free nothing [S3].
6. Roughly 20% of tokens get 80% of attention weight; under RoPE only ~25 recent queries are informative for
   a SnapKV-style observation window [S3].
7. DSA: 64 indexer heads of dimension 128, top-k 2048, indexer in FP8; attention is dense below 2K tokens
   [S6][S9][S12].
8. HiSparse LRU miss rates for GLM-5.1 at k = 2048: 30% with top-k-only (B = 2048), 13.4% with LRU at B =
   4096, 6.7% at B = 8192; Bélády's optimum at 4096 is 8.2% [S2].
9. GPU footprint per 128K request under HiSparse: ~0.4 GB (B = 4096) vs 13.09 GB dense, ~30× [S2].
10. PCIe Gen5 ×16 is ~64 GB/s per direction; a dozen requests' all-miss traffic (200 MB/token each at 30 ms)
    would saturate it [S2]. Per-entry cudaMemcpyAsync gets 0.79 GB/s; a GPU gather kernel over pinned memory
    gets 37 GB/s [S4].
11. GLM-5.2's IndexShare lets one indexer serve 4 DSA layers, cutting indexer cache and indexer FLOPs by 75%
    [S9].
12. DeepSeek-V4 caches a weighted sum of 8 tokens (stride 4) in 30 layers and of 128 tokens (stride 128) in 31
    layers; 9.62 GiB per 1M-token sequence in bf16 vs 83.9 GiB for a V3.2-style stack [S7].
13. CacheGen shrinks a stored KV cache 3.5–4.3× and cuts context-loading delay 3.2–3.7× with negligible
    quality impact [S5].
14. H200: 141 GB HBM3e at 4.8 TB/s; reading a 40 GiB cache once takes ~9 ms, the floor on time-per-token for
    dense attention at batch 1 (derived from [S13]).

### The Fixer, introducing the jobs

- "Same loot, four vaults. Shrink it where it sits, run it across the parking lot, mail it to another city,
  or burn what you don't need. Different van each time."
- "In here the getaway ride is a bicycle. One multiply, one add, inside the attention kernel. If your Packer
  needs to read the previous code word first, he stays home."
- "The alarm goes every thirty milliseconds. That's the whole job: how much can you carry out of HBM in one
  trip before the next token wants it?"
- "FlashAttention doesn't leave receipts. Nobody ever writes down which tokens got looked at. If your plan
  starts with 'read the attention scores', you don't have a plan."
- "The vault is rented by the room, sixteen boxes a room. Empty fifteen boxes and you still pay for the room.
  Want the rent back? Move the last box out. That's a copy, and copies cost."
- "Sparse attention was supposed to save memory. It saved compute. Top-k still wants the whole context sitting
  in HBM so it can pick. That's why the warehouse run exists."
- "The indexer is your address book. Two hundred bytes a token, read cover to cover every step. You don't
  compress the address book. You compress what it points to."
- "Two thousand boxes a step, scattered across the warehouse, over a pipe that does sixty-four gigabytes a
  second on a good day and shares it with everyone else on the GPU. Your crew either fetches them in one
  trip or doesn't come back."
- "Last step's picks are mostly this step's picks. Keep a hot shelf twice the size of the pick list and
  you'll miss one in eight. Keep only the pick list and you'll miss one in three."
- "Cold storage is the only job where you get to plan. The whole prefix is on the table, you encode once,
  the Packer's welcome. Just don't make loading it slower than recomputing it."
- "The biggest shrink in this business was never a codec. It was the architects deciding to store eight heads
  instead of sixty-four, then one latent instead of two tensors. You work on what they left you."

---

## 6. Side-quest ideas

### `tiers` — THE WAREHOUSE RUN (memory-tier budget game)

concept: Place the KV cache of N tokens across HBM / host DRAM / SSD under capacity limits; minimize time per
token given what attention reads each step.

Playfield: three shelves (HBM: capacity C_hbm, bandwidth 4.8 TB/s; DRAM: C_dram, PCIe 64 GB/s; SSD: large,
6 GB/s), a context of N = 128K tokens shown as a bar (plus an "indexer keys" strip that is pinned to HBM
and cannot be moved). Controls: a split slider assigning tokens to tiers (newest tokens first in HBM by
default), a "hot buffer size" B slider (1k–8k), a MODE toggle DENSE / SPARSE (k = 2048), a PRECISION toggle
bf16 / FP8 / int4 for the HBM tier and for the DRAM tier separately, a CONCURRENCY slider (1–64 requests
sharing the GPU). Readout per decode step: bytes read from each tier, time = max(Σ bytes_hbm / BW_hbm,
Σ bytes_pcie / BW_pcie, Σ bytes_ssd / BW_ssd) (fully overlapped; a toggle "no overlap" sums them), plus a
"misses" estimate from a canned curve: miss rate = 0.30 at B = k, 0.134 at 2k, 0.067 at 4k, interpolated
in log(B/k) (HiSparse LRU numbers [S2]). Capacity violations turn the shelf red and cap concurrency.
Model presets: Llama-3-70B (320 KiB/token dense, no sparse mode), DeepSeek-V3.2 (656 B latent + 132 B indexer
per layer, 61 layers, sparse available), GLM-5.1 (100 KB/token, 78 layers). Score: 100 × (reference
time-per-token / yours) where the reference is the HiSparse-style placement (B = 2k, FP8, all else in DRAM),
capped at 100; a second readout shows max concurrency at your placement.

Lesson shown by play: dense + HBM-only hits the capacity wall at a handful of requests; dense + offload is
hopeless (40 GiB/step over PCIe); sparse + hot buffer makes DRAM usable, and the hot-buffer size trades HBM
capacity against PCIe traffic; int4 in DRAM barely matters (PCIe traffic is the miss traffic, which is small),
while FP8 in HBM matters a lot (it is read every step).

### `pagepurge` — RENTED BY THE ROOM (paged-memory eviction puzzle)

concept: Dropping tokens frees nothing until a page is empty; compaction frees pages at copy cost.

Playfield: 16 pages × 16 slots (256 tokens) drawn as a grid of rooms with boxes; each token carries an
importance score (seeded: a heavy-hitter pattern with ~20% of tokens holding 80% of total weight, plus
4 "attention sink" tokens at the start that are always important). Controls: click or drag to mark tokens
EVICT; a budget readout "keep ≤ 64 tokens"; a COMPACT button that moves survivors into the lowest pages
(forward packing) and counts copies; an ORDER toggle (hole-filling: fill holes from the end, fewest copies;
order-preserving: shift everything, most copies). Readouts: tokens kept, attention mass kept (Σ importance
of survivors / total), pages freed, copies made. Also a "what the allocator sees" bar: pages freed × 16
tokens vs tokens evicted. Two rounds: round 1 forbids COMPACT (the player discovers that even evicting 192
of 256 tokens frees 0–2 pages); round 2 allows it. Score: 100 × attention mass kept × (pages freed / 12),
minus 2 per copy over the hole-filling minimum; show the Bélády-style greedy (keep the top-64 by score, then
compact) as the reference.

Debrief hook: the blog's example, 14,400 of 16,000 evicted and nothing freed [S3]; "the algorithmic idea is
half the work" [S3]; and the connection to codec design: a quantizer's block should tile the page, and
headers belong to pages, not tokens.

---

## 7. What I could not verify / caveats

- The 656-byte FP8 latent entry is from ESS [S4]; my decomposition (512 FP8 + 16 B scales + 128 B bf16 RoPE)
  is inferred from FlashMLA's documented FP8 layout as I remember it, not from a fetched source.
- Llama-3-70B's config (80 layers, 64 heads, 8 kv heads, head dim 128) could not be fetched directly (gated
  repo, HTTP 401); the 320 KiB/token figure is confirmed by two secondary sources [S10][S11].
- Raschka's DSA page [S0] is high-level only; indexer head count and dimension come from the HF config of
  DeepSeek-V3.2 [S12], and the FP8 claim from the technical report [S6] and TensorRT-LLM [S14]. The report
  itself does not print numeric values for H^I or d^I.
- ESS numbers are from an internal simulator with linear interpolation, not hardware measurements; the paper
  does not name the GPU model or host DRAM size [S4]. The Baidu affiliation is from the reading list; the
  arXiv abstract page lists authors without affiliations.
- The NVIDIA blog [S3] was readable only in its first third (the rest is embedded image data); no section on
  batching or CPU offload was found in the readable part. It does not mention KIVI.
- The SemiAnalysis article [S9] is partly paywalled; no per-token KV size for GLM-5.3 or DRAM market figures
  were visible. The "memory capacity binds" argument and the InferenceX numbers are from the visible part.
- CacheGen's internal design (delta against anchor tokens, layer-wise quantization, arithmetic coding) is from
  memory and belongs to the codec slice; only the headline numbers were verified from the abstract [S5].
- B200 HBM capacity is quoted as 180 GB or 192 GB depending on the source [S13].
- The survey's full subcategory list was taken from the companion GitHub list [S16]; the arXiv abstract [S15]
  only names the three levels.

---

## Sources

- [S0] Raschka, "DeepSeek Sparse Attention" explainer, https://sebastianraschka.com/llms-from-scratch/ch04/15_deepseek_sparse_attention/
- [S1] LMSYS blog, "HiSparse in SGLang", 2026-04-10, https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/
- [S2] HiSparse paper, arXiv 2608.07009, https://arxiv.org/html/2608.07009
- [S3] NVIDIA (Song Han's group), "KV Cache Compression and Its Infra Problems", https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/
- [S4] ESS: Extended Sparse Server for DeepSeek-V3.2, arXiv 2512.10576, https://arxiv.org/html/2512.10576
- [S5] CacheGen, SIGCOMM 2024, arXiv 2310.07240, https://arxiv.org/abs/2310.07240
- [S6] DeepSeek-V3.2 technical report, arXiv 2512.02556, https://arxiv.org/html/2512.02556v1
- [S7] vLLM blog, "DeepSeek-V4 in vLLM", 2026-04-24, https://vllm.ai/blog/2026-04-24-deepseek-v4
- [S8] NVIDIA developer blog, "Build with DeepSeek V4 using NVIDIA Blackwell", https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/
- [S9] SemiAnalysis, "Sparse savings, persistent demand: inside GLM-5.3", https://newsletter.semianalysis.com/p/sparse-savings-persistent-demand-inside-glm53
- [S10] fast.io, "Llama 3.1 context window" (KV formula and 320 KB/token), https://fast.io/resources/llama-3-1-context-window.md
- [S11] VMware, "LLM inference sizing and performance guidance" (0.000305 GiB/token for 70B), https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/
- [S12] Hugging Face configs: deepseek-ai/DeepSeek-V3 (61 layers, kv_lora_rank 512, qk_rope_head_dim 64) and deepseek-ai/DeepSeek-V3.2 (index_n_heads 64, index_head_dim 128, index_topk 2048), https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json
- [S13] GPU spec roundups: https://www.runpod.io/blog/nvidia-h200-gpu-specs-vram-price-and-ai-performance , https://www.runpod.io/blog/nvidia-b200-gpu-specs-vram-price-and-ai-performance , https://inferencex.semianalysis.com/gpu-specs
- [S14] TensorRT-LLM tech blog, "Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs", https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs
- [S15] Li et al., "A Survey on LLM Acceleration based on KV Cache Management", TMLR 2025, arXiv 2412.19442, https://arxiv.org/abs/2412.19442
- [S16] Awesome-KV-Cache-Management (companion list), https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
