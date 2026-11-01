---
title: "The Map"
summary: "A systematic taxonomy of the compression toolkit, organised by the role each algorithm plays in a codec pipeline, with an introduction article for every family."
---

# The Map: a taxonomy of the compression toolkit

Every practical codec is a pipeline of the same stages: a **transform** that rearranges the data, a **predictor** that
guesses what comes next, a **quantizer** that throws away what the consumer will not miss (the only lossy stage), an
**entropy coder** that cashes predictability in for bits, a **perceptual model** that defines "will not miss", and a
**shared prior** the decoder already holds before the first byte arrives. The [primer](primer.md) explains why. This
page lists what can fill each stage, grouped into families and sub-families, and links to an introduction article for
each family. In the game, every algorithm card links back here.

Read in order if you are new: the three **Foundations** articles first, then any stage.

## Foundations

- [Modeling and coding: bits are surprise](zoo/modeling-and-coding.md) — entropy as the floor, −log₂ p as the price of a symbol, cross-entropy as the mismatch tax, and why "how well can you compress X" and "how well can you predict X" are the same question.
- [The anatomy of a job](zoo/anatomy-of-a-job.md) — the five axes that pin down any compression problem (fidelity contract, redundancy profile, access pattern, compute envelope, where the prior lives), and why random access, latency and block boundaries are paid for in ratio.
- [Rate and distortion](zoo/rate-and-distortion.md) — the one irreversible chair, the R(D) curve, Δ²/12, 6 dB per bit, reverse water-filling, and the consumer model that decides what counts as error.

## Entropy coding (symbols + probabilities → bits; always the last stage)

- **Prefix codes** (whole bits per symbol) — [article](zoo/prefix-codes.md)
  - Huffman coding (static, two-pass semi-static with transmitted tables, canonical Huffman)
  - Golomb-Rice codes (one parameter, optimal for geometric residuals: FLAC, JPEG-LS)
  - Run-length encoding and zigzag scans (manufacturing a single symbol out of a run of zeros: JPEG, fax)
  - Fixed-length codes (the deliberate empty stage: random access, SIMD, GPU kernels)
- **Fractional-bit coders** (reach the entropy bound) — [article](zoo/arithmetic-and-ans.md)
  - Arithmetic coding and range coding (one number for the whole message; per-symbol adaptive models)
  - Binary adaptive coders (CABAC in H.264/HEVC, the MQ coder in JPEG 2000)
  - Asymmetric numeral systems: rANS (interleaved, SIMD) and tANS/FSE (table-driven: zstd, JPEG XL)
- **Adaptivity** as a property: static tables, semi-static headers, fully adaptive models (cross-cuts both families; see [side information](zoo/side-information.md))

## Prediction and modeling (expose structure; lower the entropy; produce no bits)

- **Dictionary matchers: the LZ family** — [article](zoo/lz-family.md)
  - LZ77 and the sliding window (DEFLATE, zstd, LZ4, Brotli, LZMA); window size and parse effort as the two dials
  - LZ78 / LZW and explicit phrase dictionaries (GIF, compress)
  - Macro-scale matching: deduplication with content-defined chunking, rsync delta sync, git packfiles
- **Linear and spatial predictors** — [article](zoo/linear-prediction.md)
  - Delta and delta-of-delta (time series, timestamps)
  - Linear predictive coding: fixed polynomial predictors and fitted LPC (FLAC, speech codecs)
  - Spatial filters and intra prediction (PNG Sub/Up/Paeth, H.264/AV1 directional modes)
  - Anchor-based prediction (groups of pictures; CacheGen's anchor tokens)
- **Statistical context models** — [article](zoo/context-models.md)
  - Order-k Markov models, PPM with escapes
  - Context mixing (PAQ, cmix) and the Hutter Prize
  - Contexts inside binary coders (CABAC's neighbour contexts)
  - Learned probability models: hyperpriors in learned image codecs, language models as compressors
- **Temporal prediction: motion compensation** — [article](zoo/motion-compensation.md)
  - Block matching, motion vectors, sub-pixel interpolation
  - I/P/B frames, the GOP length as a ratio-vs-seek dial, in-loop filters
  - Low-latency configurations (no B-frames, intra refresh)

## Transforms and rearrangements (save nothing alone; set up the other stages)

- **Frequency transforms** — [article](zoo/block-transforms.md)
  - Block DCT (JPEG, video integer DCTs), energy compaction and transform coding gain
  - Lapped transforms: the MDCT and time-domain aliasing cancellation (MP3, AAC, Opus CELT)
  - Wavelets and multiresolution (JPEG 2000)
- **Rotations, projections and rank** — [article](zoo/rotations-and-rank.md)
  - Karhunen-Loève / PCA as the optimal decorrelator; OPQ rotations for product quantization
  - Hadamard rotations that preserve dot products (QuaRot, SpinQuant)
  - Low-rank truncation of key matrices (ShadowKV, xKV); undoing RoPE before quantizing
- **Channel and layout transforms** — [article](zoo/channel-and-layout.md)
  - RGB→YCbCr and chroma planes; stereo mid/side; integer-reversible variants (YCoCg-R)
  - Byte shuffling of float arrays; struct-of-arrays; columnar storage (Parquet)
  - Blocking and framing as plumbing (parallelism, seek, corruption containment)
- **Sorting transforms** — [article](zoo/sorting-transforms.md)
  - Burrows-Wheeler transform, move-to-front, the bzip2 stack
  - BWT as a search index (FM-index) in genomics
- **Learned transforms** — [article](zoo/learned-transforms.md)
  - Autoencoders as analysis/synthesis pairs; neural audio codecs (SoundStream, EnCodec, DAC)
  - Learned image compression (Ballé et al.), the VAE in diffusion models
  - Transforms baked into model architecture: GQA, Multi-head Latent Attention, DeepSeek-V4 compressed attention

## Quantization (the only lossy stage)

- **Scalar quantizers** — [article](zoo/scalar-quantization.md)
  - Uniform quantization, dead zones, rate-distortion-optimized rounding (video RDO, GPTQ)
  - Non-uniform and companded quantizers (μ-law, MP3's power law, NF4); Lloyd-Max
  - Frequency-weighted step sizes (JPEG tables) and bit allocation across coordinates (water-filling, KVTC)
  - Outlier handling: dense-and-sparse decomposition, escape codes
- **Vector quantizers** — [article](zoo/vector-quantization.md)
  - k-means / Lloyd's algorithm, codebook design, palettes as VQ
  - Residual VQ (SoundStream, EnCodec; audio tokens), product quantization (Faiss), additive quantization
  - Coupled quantization of correlated channels (joint entropy vs marginals)
- **Block-scaled integer and float formats** — [article](zoo/block-float-formats.md)
  - int8/int4 with per-block scales and minimums (GGUF Q4_0, Q4_K, bitsandbytes), AWQ/GPTQ encoders
  - NVFP4 / MX formats: micro-scaled floats decoded by the tensor core
  - Per-channel vs per-token grouping for keys and values (KIVI); eviction as a two-level quantizer
- **Token selection and eviction** (see [attention as the audience](zoo/attention-as-audience.md))

## Perceptual and consumer models (what will the reader not notice?)

- **Human perceptual models** — [article](zoo/perceptual-models.md)
  - Psychoacoustic masking, pre-echo and short blocks (MP3, AAC, Opus)
  - Contrast sensitivity, chroma subsampling, just-noticeable differences (JPEG, video)
  - Adversarial and feature losses for generative decoders (neural codecs, HiFiC)
- **A transformer as the audience** — [article](zoo/attention-as-audience.md)
  - Attention sinks and recent windows kept at full precision (KIVI, KVQuant)
  - Top-k selection with cheap proxies (DSA indexer, landmarks, PQ codes, 1-bit copies)
  - Eviction (StreamingLLM, H2O, SnapKV) and the paged-memory problem; FlashAttention leaves no scores

## Shared priors and side information (what the decoder already holds)

- **Side information** — [article](zoo/side-information.md)
  - Standard tables in the spec vs self-describing headers (DEFLATE fixed vs dynamic blocks)
  - Trained dictionaries (zstd --train, Brotli's web dictionary, HTTP compression dictionaries)
  - Codebooks (VQ, neural codecs, Faiss indexes), trained transform bases (KVTC, SpinQuant)
  - Learned decoders as priors; cross-entropy mismatch when the prior was trained on other data
  - Tiered copies: a lossless original one hop away (HiSparse, SpeCache, ShadowKV)

## Tasks: the systems that pick the stages

- [The five vaults of the KV cache](zoo/kv-cache-vaults.md) — one tensor, five jobs: resident in HBM, offloaded over PCIe, parked in cold storage, streamed over a network, or evicted; and the model-level shrinks (GQA, MLA) that happened before any codec.
- [Codecs as pipelines: worked dissections](zoo/codec-dissections.md) — DEFLATE, PNG, FLAC, JPEG, MP3, Opus, H.264, zstd, SoundStream, GGUF, Faiss PQ, each decomposed chair by chair, with the job's stat sheet beside it.
