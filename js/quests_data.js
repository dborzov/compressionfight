// Choose Your Codec — side quests (content). Mechanics are implemented in js/quests.js per docs/SIDEQUESTS.md.
// Each quest: the Fixer's `howto` (what to do, step by step), `watch` (what to look at while playing),
// `debrief` (what it taught, in the field's own words), and links to lessons/fighters.

window.CYC = window.CYC || {};

CYC.SIDEQUESTS = [
  {
    id: 'shannon', level: 1, name: 'GUESS THE NEXT LETTER', concept: 'Entropy of English, measured with your own head',
    fighters: ['cm', 'neural_prior'], lesson: 'entropy',
    howto: [
      'Claude Shannon did this in 1951 with his wife Betty. A sentence is hidden. You guess the next letter; if you are wrong, guess again until you hit it. We count how many guesses each letter took.',
      'Type a letter (or space) and press Enter. Letters you have already tried are crossed out. The sentence reveals itself one letter at a time.',
      'Your cost for a letter is log<sub>2</sub>(number of guesses): one guess = 0 bits, two = 1 bit, four = 2 bits. That is a rough stand-in for −log<sub>2</sub> p of the letter under your brain\'s model.',
    ],
    watch: 'Watch the running bits-per-letter number. Early in a word you need many guesses; after "th" you need one. Context is the whole game.',
    debrief: 'Shannon got about 1 bit per letter from human guessers, against 4.7 bits for a fixed 26-letter code and 4.1 for letter frequencies alone. Your brain is a context model: it predicts, and prediction is what lowers the entropy. PAQ and the LLM-compressors mechanize this guessing. The number you just produced is a compression ratio.',
    fail: 'Over 3 bits per letter? Use context: after a consonant cluster a vowel is likely; after "q" comes "u"; the most common letter is e, then t, a, o, i, n.',
  },
  {
    id: 'hufftree', level: 1, name: 'BUILD THE TREE', concept: 'Huffman coding and the Kraft budget',
    fighters: ['huffman'], lesson: 'prefix',
    howto: [
      'Six symbols with their counts. Build a prefix code by merging: click two nodes, they fuse into a parent whose count is the sum. Keep going until one root remains.',
      'Each leaf\'s code is its path from the root (left 0, right 1). The Kraft bar at the bottom shows Σ 2<sup>−length</sup>: a valid prefix code never goes above 1, and an optimal one fills it exactly.',
      'Your score is the average code length against the Huffman optimum and against the entropy H. Merge the two SMALLEST counts each time and you will hit the optimum.',
    ],
    watch: 'Watch which symbols end up deep in the tree. Rare symbols get long codes; that is not a punishment, it is where the savings on common symbols come from.',
    debrief: 'Huffman\'s greedy rule (always merge the two least frequent) is provably optimal among codes that spend a whole number of bits per symbol. The gap to H you see is the price of integer lengths. Arithmetic coding and ANS remove it by not assigning codes to symbols at all.',
    fail: 'Average length far above optimal? You merged a big node early. Always pick the two smallest counts on the table.',
  },
  {
    id: 'waterfill', level: 2, name: 'THE BIT BUDGET', concept: 'Bit allocation across transform bands (reverse water-filling)',
    fighters: ['dct', 'uniform', 'qmatrix'], lesson: 'transform',
    howto: [
      'An image went through the 8×8 DCT. Its 64 coefficients are grouped into 8 frequency bands, low to high, and you can see each band\'s variance (how much energy lives there).',
      'You have a budget of bits per block. Drag each band\'s slider to give it 0 to 8 bits per coefficient. The total must stay under budget. The image on the right is reconstructed live from your allocation; the MSE is below it.',
      'Goal: lowest MSE. There is a mathematically optimal allocation, and the game will show it when you lock in: you are trying to discover the rule by hand.',
    ],
    watch: 'Watch what one bit does to the low-frequency band versus the highest band. Then watch the image: blockiness comes from starving the low bands, blur from starving the high ones.',
    debrief: 'The optimal rule is reverse water-filling: pick a water level θ and give each band R = ½ log<sub>2</sub>(σ²/θ) bits, zero if its variance is below the water. Bands below θ are dropped entirely. JPEG\'s quantization table is this rule in disguise, bent by the eye\'s contrast sensitivity. The energy compaction of the DCT is what makes the high bands fall under water.',
    fail: 'Spread the bits evenly and you are quantizing pixels in disguise. Starve the bands with tiny variance; they are mostly already zero.',
  },
  {
    id: 'jnd', level: 2, name: 'SPOT THE DIFFERENCE', concept: 'Just-noticeable distortion: where lossy becomes free',
    fighters: ['chroma', 'qmatrix', 'psychoacoustic'], lesson: 'perceptual',
    howto: [
      'Two images. One is the original, the other has been DCT-quantized at some step size and put back. Click the one you think is degraded. If you cannot tell, click "SAME TO ME".',
      'The game runs a staircase: when you are right, the next pair is quantized more gently; when you are wrong or give up, more harshly. After about 12 trials it converges on YOUR threshold: the coarsest step you cannot see.',
      'That threshold is a step size Δ, and the game converts it to bits saved against lossless.',
    ],
    watch: 'Watch where the damage shows first: flat regions reveal blocking, sharp edges reveal ringing. Then look at the chroma-only version: you will tolerate much more there.',
    debrief: 'You just measured a just-noticeable difference. A perceptual codec operates right under that line everywhere, and the line is different for luma, chroma, low and high frequencies, loud and quiet audio bands. Every lossy codec\'s "quality" setting is a scalar along this staircase.',
    fail: 'Guessing at random converges to a very coarse threshold; that is fine, the point is the method. Try again with the chroma-only variant.',
  },
  {
    id: 'phrasebook', level: 3, name: 'TRAIN THE PHRASEBOOK', concept: 'Trained dictionaries: choosing side information under a budget',
    fighters: ['dictionary', 'lz77'], lesson: 'sideinfo',
    howto: [
      'Six sample messages from a log stream are shown. A hidden seventh message, from the same stream, will be compressed with LZ77 using whatever you put in the dictionary.',
      'Click fragments (keys, values, punctuation runs) in the samples to add them to the dictionary. Your budget is 96 bytes. The dictionary preloads the LZ window.',
      'Press COMPRESS. You see the test message with matches highlighted and the compressed size with and without your dictionary. Then the game shows what a greedy frequency-based trainer (like zstd\'s COVER) would have picked.',
    ],
    watch: 'Watch which fragments actually match in the test message. A fragment that appears in every sample is gold; one that appears once is wasted budget.',
    debrief: 'A dictionary is the frequent-substring prior of a data stream, shipped once. zstd --train scores segments by how often and how long they match across the samples, under a size budget, which is exactly what you did by hand. The second half of the lesson: run the same dictionary on the "foreign" message and watch the gain vanish. That is cross-entropy mismatch, D<sub>KL</sub>(P‖Q), as a byte count.',
    fail: 'Saved little? You picked content values (ids, timestamps) that change per message. Pick the schema: keys, quotes, colons, enum values that repeat.',
  },
  {
    id: 'safehouses', level: 3, name: 'SAFEHOUSES', concept: 'Vector quantization: placing a codebook',
    fighters: ['vq', 'rvq', 'pq', 'codebook'], lesson: 'vq',
    howto: [
      'The map shows 60 drop-off points. You get K = 4 safehouses. Click the map to place them; drag to move. Every point is assigned to its nearest safehouse, and the total squared distance (distortion) is shown.',
      'Each point will be transmitted as a 2-bit safehouse number instead of two coordinates. The safehouse list is the codebook: it is sent once, the numbers are the stream.',
      'When you are happy, press RUN K-MEANS: Lloyd\'s algorithm will iterate from YOUR placement (move each safehouse to the mean of its points, re-assign, repeat). Then compare with the best of 10 random restarts.',
    ],
    watch: 'Watch k-means get stuck when two safehouses start in one cluster: it will never cross the empty region. That is a local minimum, and it is why codebook training uses restarts and careful initialization.',
    debrief: 'You built a VQ codebook. Bits per point went from 2 × 32 to 2; the error is the spread of each cluster. Residual VQ would now run the same game on the leftover arrows. Product quantization would split the map into x and y and play two 1-D games. In every case the safehouses must be shipped to the decoder: they are the Keymaster\'s keys.',
    fail: 'High distortion? Look for a safehouse serving two clusters while another serves none. Move the idle one into the crowded area.',
  },
  {
    id: 'blockpack', level: 4, name: 'PACK THE BLOCK', concept: 'Block quantization formats: bits per weight vs error',
    fighters: ['blockint', 'pca'], lesson: 'blocks',
    howto: [
      '256 real-looking weights (bell-shaped, a few outliers) are shown as a strip. Choose a format: bits per weight (2–8), block size (8–256), with or without a per-block minimum (zero-point), and with or without a Hadamard rotation before quantizing.',
      'The game quantizes and dequantizes with your format and shows two numbers: bits per weight = (scale bits + min bits + n × b) / n, and the reconstruction error. The target is: error as low as you can get while staying under 4.5 bits per weight.',
      'Lock in, and the game names the real format closest to your choice (Q4_0, Q4_1, Q4_K-style, Q8_0, NF4) and what it would weigh for a 7B model.',
    ],
    watch: 'Watch the outliers. With block size 256 one outlier sets the scale for 255 neighbours and they collapse to zero. With block size 32 it ruins 31. With rotation it ruins nobody, because it no longer exists as a single huge value.',
    debrief: 'Every weight format on Hugging Face is a point in this space. Small blocks cost scale bits but contain outliers; a minimum costs more bits but recentres skewed blocks; a rotation costs a tiny matmul at load time and flattens the distribution. There is no entropy coder anywhere in this: the kernel needs element i at a fixed address.',
    fail: 'Over budget? 8 bits per weight with 8-weight blocks is 10 bpw. Spend bits on weights, not on scales: block size 32 with 4 bits is the usual sweet spot.',
  },
  {
    id: 'raceaccess', level: 4, name: 'RANDOM ACCESS RACE', concept: 'Variable-length codes vs random access; checkpoints as a cost',
    fighters: ['huffman', 'ans', 'motion'], lesson: 'noentropy',
    howto: [
      'An array of 4096 values is stored two ways: fixed 8-bit codes, and Huffman codes (about 5.1 bits each on this data). You will be asked for random elements: "give me element 3,187".',
      'Fixed length jumps straight there. Huffman must decode from the last checkpoint. You choose the checkpoint spacing with the slider (every 1, 16, 64, 256, 1024 elements, or none). Each checkpoint costs a 16-bit offset.',
      'Press FETCH ten times. The game shows total bytes (payload + checkpoints) and total decode steps for both layouts. Find the spacing where Huffman is still smaller and not absurdly slower.',
    ],
    watch: 'Watch the two totals move in opposite directions as you change the spacing. There is no setting where Huffman wins on both. That is the trade.',
    debrief: 'Checkpoints are keyframes. Video GOPs, zstd frames, Parquet row groups, FLAC seek tables and GGUF blocks are all the same decision: how often to pay for a resynchronization point so that random access is affordable. A GPU kernel sets the spacing to one, which is the same as fixed-length codes, which is why GGUF has no entropy coder.',
    fail: 'Chose "none"? One fetch of element 4,000 decodes 4,000 symbols. Add checkpoints until the step count is tolerable and read what it cost in bytes.',
  },
  {
    id: 'siamese', level: 5, name: 'JOINED AT THE HIP', concept: 'Joint vs marginal entropy: why vector quantization beats scalar on correlated channels',
    fighters: ['coupled', 'vq', 'blockint'], lesson: 'jointentropy',
    howto: [
      'Four thousand pairs of channel values (a, b) from one layer of a key cache. They are correlated: b is mostly 0.9 × a plus a little noise. You have 2 bits per pair to spend.',
      'Mode SCALAR: each channel gets its own 1-bit quantizer (two levels per axis, found by k-means on that axis alone). That is a 2×2 grid of reconstruction points. Mode JOINT: one 2-D k-means with 4 centroids on the pairs. Same 2 bits per pair.',
      'Press RUN for each mode and compare the MSE. Then press ENTROPY: the game bins each channel into 16 levels and prints H(a), H(b), H(a,b) from the counts. The gap H(a)+H(b)−H(a,b) is the mutual information you were paying for twice.',
      'Optional: toggle WEIGHTED to weight the k-means by |a| (a stand-in for Fisher information). MSE goes up; weighted error goes down. That is why Coupled Quantization weights its centroids.',
    ],
    watch: 'Watch where the four scalar-grid points land: three of them sit in empty space where no data lives. The four joint centroids sit on the diagonal where the data is.',
    debrief: 'Independent scalar quantizers assume independent channels. Correlated channels have joint entropy below the sum of marginals, and only a quantizer that sees the pair can collect that difference. Coupled Quantization does this on 4-8 channels of real KV caches: 1 bit per channel where scalar 1-bit quantization gives perplexity in the thousands. The Vector Vault taught the same thing about embeddings; here it decides whether a 128k context fits a consumer GPU.',
    fail: 'Mutual information near zero? Check the correlation; the round\'s data is seeded, press NEW ROUND for a different slope.',
  },
  {
    id: 'anchorchain', level: 5, name: 'ANCHOR OR CHAIN?', concept: 'Delta coding with parallel decode: anchors vs chains',
    fighters: ['anchor_delta', 'delta', 'motion'], lesson: 'leavinggpu',
    howto: [
      'A 64-token × 16-channel slice of a KV cache: each channel has an offset, a slow random walk and small noise, like CacheGen measured. You will code it with a fixed quantization step.',
      'Pick a predictor: NONE (code raw values), CHAIN (each token minus the previous token), or ANCHOR (every g-th token coded raw at 8 bits, the others as differences from their anchor). Slide g between 4 and 32.',
      'Three live numbers: entropy of the quantized symbols in bits per element, decode dependency depth (how many sequential steps a GPU thread needs to reach token 63), and worst-case reconstruction error after quantization.',
      'Press LOCK IN when you have found the predictor you would ship over a network to a GPU decoder.',
    ],
    watch: 'Watch the error row for CHAIN: quantization error random-walks down the chain and grows with distance from the start. ANCHOR\'s error is bounded by one step. CHAIN wins about 0.3 bit of entropy; ANCHOR wins parallel decode and bounded drift.',
    debrief: 'CacheGen chose anchors of 10 tokens: 8-bit anchors, deltas coded against them, one GPU thread per token. Video codecs chose both: I-frames are anchors, P-frames chain, B-frames reference two anchors; the GOP length is your slider g. Deltas against the previous token have the lowest entropy; deltas against a fixed anchor have the decoder you can afford.',
    fail: 'Picked CHAIN? Look at the dependency depth: 63 sequential steps on a GPU thread, and a drift you cannot bound. Picked NONE? You left 0.5-1 bit per element on the table.',
  },
  {
    id: 'rankkeys', level: 5, name: 'RANK THE KEYS', concept: 'Low-rank truncation: keys are low-rank, values are not',
    fighters: ['svd_lowrank', 'pca', 'mla'], lesson: 'keysvalues',
    howto: [
      'A synthetic key matrix K (256 tokens × 64 dims) with geometrically decaying singular values plus noise, a value matrix V with flat singular values, and 8 query vectors.',
      'Slide the rank r from 1 to 64. The game keeps the top-r singular directions and shows: storage (L·r + r·d numbers vs L·d), the attention-score error ‖softmax(QKᵀ) − softmax(QK_rᵀ)‖₁, and the overlap between the exact top-16 tokens and the truncated top-16.',
      'Then press SWITCH TO V and move the same slider: storage falls identically, but the reconstruction error does not.',
      'Press LOCK IN at the smallest rank where top-16 overlap is still at least 15 of 16.',
    ],
    watch: 'Watch the singular-value bars: fast decay is energy compaction along the token axis, the same thing the DCT does along a row of pixels. V has no decay, so there is nothing to truncate.',
    debrief: 'ShadowKV keeps rank 160 of 1024 for pre-RoPE keys and offloads the values whole, because this asymmetry is real in transformers. xKV goes further: adjacent layers share the important directions even though they disagree token by token, so one basis can serve four layers. MLA is the trained version of the same cut, decided before pretraining. Rank truncation is the Lens chair\'s oldest move, the Karhunen-Loève transform, applied along the token axis.',
    fail: 'Needed rank 60 of 64 before the overlap held? Check which matrix you were truncating. Values are not low-rank; that is the point.',
  },
  {
    id: 'pagepurge', level: 5, name: 'RENTED BY THE ROOM', concept: 'Paged memory and eviction: why dropping tokens frees nothing',
    fighters: ['evict', 'spotlight'], lesson: 'softmaxaudience',
    howto: [
      'A 16 × 16 grid: 256 tokens in 16 pages of 16. Each token has an importance score (brighter is more important). Your budget is 64 tokens.',
      'Click tokens to evict them (or press AUTO to evict the 192 least important). The readout shows tokens kept and PAGES FREED. A page is freed only when all 16 of its tokens are gone.',
      'Press COMPACT: survivors are copied into fresh pages. Choose HOLE-FILL (move survivors from the sparsest pages into holes of the fullest, fewest copies) or IN-ORDER (keep token order, more copies). The readout counts copies and pages freed.',
      'Press LOCK IN when you have 64 survivors in as few pages as possible with as few copies as you can manage.',
    ],
    watch: 'Watch PAGES FREED after AUTO: often zero, because the important tokens are spread across every page. Then watch what compaction costs in copies.',
    debrief: 'vLLM-style paged attention frees a block only when it is empty. NVIDIA\'s example: evict 14,400 of 16,000 tokens and the 1,600 survivors still occupy about 1,000 blocks; VRAM is unchanged until you physically compact, and compaction is copies. This is why many eviction papers never shipped and why TriAttention compacts into fresh pages. The same accounting governs any allocator with fixed blocks: filesystems, GC heaps, and the GOP structure of a video stream.',
    fail: 'Freed pages but spent hundreds of copies? Try HOLE-FILL: it moves the fewest tokens by emptying the sparsest pages first.',
  },
];
