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
];
