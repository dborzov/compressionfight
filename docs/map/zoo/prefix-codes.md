---
title: "Prefix codes: whole bits"
slug: prefix-codes
chair: entropy
family: "Prefix codes"
summary: "How to hand out bit strings so a stream decodes without separators, why Kraft's inequality makes code lengths a budget, why Huffman spends that budget optimally and still loses when one symbol dominates, and why residual coders need no table at all."
fighters: [huffman, rice, rle]
related: [arithmetic-and-ans, modeling-and-coding, linear-prediction]
reading_time: 15
---

## A term paper instead of a final

Cambridge, Massachusetts, spring of 1951. David Huffman is a graduate student in Robert Fano's information theory course at MIT, and Fano has offered the class a deal: write a term paper on a problem he poses, or sit the final exam. The problem is this. You have a set of symbols and you know how often each occurs. Find the most efficient binary code for them: the one with the smallest average number of bits per symbol, subject to the requirement that a receiver can split the stream back into symbols without any separators. Fano does not mention that he and Shannon have both attacked it and that neither has a proof their methods are optimal. (That part is history. What follows about the night before is a story, and you should imagine it as one.)

Picture Huffman months later, notes spread across a desk, about to give up and study for the exam. He has been trying to build the code the way everyone builds it: from the top, splitting the symbols into two roughly equal-probability groups, then splitting each group, assigning a 0 or 1 at each split. It produces decent codes. It never produces one he can prove is best, because the first split commits to a structure before the rare symbols have been looked at. In the story, the thought that stops him from throwing the notes away is a reversal: stop deciding what the frequent symbols get first. Decide what the *rarest two* get first. They should be the longest codes, and they might as well be siblings, differing only in their last bit. Merge them into one pretend symbol with their combined frequency, and now you have a smaller problem of the same kind. Repeat until one symbol remains. The code builds itself from the bottom up, and the proof falls out of the same observation.

Before we follow him, we need to feel the problem he was handed, because the hard part is not the merging. It is why separators are not needed and what that costs.

## The problem with separators

Six symbols with the frequencies from the textbook example (Cormen, Leiserson, Rivest and Stein use these): A 45, B 13, C 12, D 16, E 9, F 5, per hundred symbols. A fixed code gives each three bits, 300 bits per hundred symbols. We want fewer, so A, nearly half the stream, should get a short word. Suppose A is `0`, B is `1`, C is `01`. Now the receiver sees `01`. Was that A then B, or was it C? Without a gap in the stream there is no way to know, and the whole point of a binary channel is that there are no gaps: only zeros and ones.

The way out is a constraint on the codewords themselves: **no codeword may be the beginning of another.** If A is `0`, nothing else may start with `0`, so every other word starts with `1`. A code with this property is called a **prefix code** (prefix-free would be the more honest name), and it is the entire answer to the separator problem. The receiver reads bits until the bits so far form a codeword; because no codeword continues into a longer one, the moment it matches is the moment it is complete. Decoding is instantaneous: no lookahead, no backtracking, no gaps.

A prefix code is the same object as a binary tree with the symbols at its leaves: go left on 0, right on 1, and the path from the root to a leaf spells the codeword. Leaves cannot be ancestors of other leaves, which is the prefix property restated. So "find the best prefix code" means "find the best tree", and the average code length is the average leaf depth, weighted by frequency.

## Code lengths are a budget

Here is the question that turns this from puzzle into mathematics: *which sets of lengths are even possible?* We want A to have length 1. Can B, C, D also have length 2? Can everything be short?

Think of it with real estate. Every node at depth $l$ in a binary tree "owns" a fraction $2^{-l}$ of the leaves at any deeper level: a codeword of length 1 claims half of all possible bit strings as its descendants, a codeword of length 2 claims a quarter, and so on. Two codewords in a prefix code cannot share descendants. So their claims cannot add up to more than the whole:

$$\sum_i 2^{-l_i} \le 1.$$

This is **Kraft's inequality**, and the converse is also true: any list of lengths satisfying it can be realized as a prefix code. It is the right way to see the problem. Code lengths are not free to choose; they draw on a budget of exactly 1, where a length-$l$ word costs $2^{-l}$ of it. One length-1 word spends half the budget. Three length-2 words would spend three quarters; with the half already gone, only one of them fits. Short words are expensive in the only currency that matters.

Now the connection to the [floor](../modeling-and-coding/). If we were free to pick any real-valued lengths, the choice $l_i = -\log_2 p_i$ would spend the budget exactly ($\sum 2^{\log_2 p_i} = \sum p_i = 1$) and achieve average length $H(p)$, the entropy. Prefix codes can only use integer lengths. So the problem Fano posed is really: *spend a budget of 1 on integer-length claims, to minimize the frequency-weighted average length.* An optimal code spends the budget completely (if $\sum 2^{-l_i} < 1$, some word can be shortened), and it is as close to $-\log_2 p_i$ as integers allow.

## Huffman's merge

Back to the six symbols, following the bottom-up idea. The two rarest are F (5) and E (9). Make them siblings under a node of weight 14. Now the two smallest weights are C (12) and B (13): siblings under 25. Next, 14 and D (16): siblings under 30. Then 25 and 30 under 55, and finally A (45) and 55 under the root at 100.

{{< fig src="img/prefix-codes/clrs_tree.png" alt="The Huffman tree for A45 B13 C12 D16 E9 F5" caption="The CLRS tree built by merging. A gets 1 bit, B C D get 3, E and F get 4. Mean length 2.24 bits against an entropy of 2.22, and the Kraft sum 1/2 + 3/8 + 2/16 is exactly 1: the budget is spent to the last cent." >}}

```python
import heapq, numpy as np
counts = {"A": 45, "B": 13, "C": 12, "D": 16, "E": 9, "F": 5}
heap = [(w, i, [s]) for i, (s, w) in enumerate(counts.items())]
heapq.heapify(heap)
code = {s: "" for s in counts}
while len(heap) > 1:
    w1, _, left = heapq.heappop(heap)      # the two smallest weights...
    w2, _, right = heapq.heappop(heap)
    for s in left:  code[s] = "0" + code[s]   # ...become siblings: one more bit
    for s in right: code[s] = "1" + code[s]
    heapq.heappush(heap, (w1 + w2, len(heap) + 100, left + right))
p = np.array(list(counts.values())) / 100
L = np.array([len(code[s]) for s in counts])
for s in counts:
    print(f"{s}: p={counts[s]/100:.2f}  code={code[s]:5s} len={len(code[s])}")
print(f"mean length {p @ L:.3f}   entropy {-(p*np.log2(p)).sum():.3f}   Kraft sum {sum(2.0**-L):.3f}")
# A: p=0.45  code=0     len=1
# B: p=0.13  code=101   len=3
# C: p=0.12  code=100   len=3
# D: p=0.16  code=111   len=3
# E: p=0.09  code=1101  len=4
# F: p=0.05  code=1100  len=4
# mean length 2.240   entropy 2.220   Kraft sum 1.000
```

Mean length 2.24 bits per symbol: 224 bits per hundred against 300 for the fixed code, and only 0.02 bits above the entropy of 2.22. Why is the merge optimal? Two facts, each short. First, in any optimal tree the two least frequent symbols can be assumed to be siblings at the deepest level (if they were not, swapping them with whatever is deepest cannot increase the average). Second, merging those two into one symbol of combined weight turns an optimal tree for $n$ symbols into an optimal tree for $n-1$, with the average length changed by exactly the combined weight. So the greedy choice is safe at every step, and induction does the rest. Huffman's 1952 paper is four pages long. It is **optimal among all prefix codes**, which is a strong claim, and the claim has a quiet boundary clause: among codes that assign one whole codeword per symbol.

{{< lab id="huffman" >}}

## How close is close?

Because an optimal code spends the Kraft budget exactly, and because integer lengths can always be chosen as $\lceil -\log_2 p_i \rceil$ (which satisfies Kraft), Huffman's average length $L$ is squeezed:

$$H(p) \le L < H(p) + 1.$$

The lower bound is Shannon's floor. The upper bound says you lose less than one bit per symbol to rounding. Gallager tightened it in 1978: $L < H + p_{\max} + 0.086$, where $p_{\max}$ is the largest symbol probability. On the CLRS alphabet that bound is $2.22 + 0.45 + 0.086 = 2.76$, and the real loss is 0.02. When the alphabet is large and no symbol dominates, Huffman is nearly perfect, which is why it ran the world for forty years: DEFLATE, JPEG, MP3, and the literal stream of zstd all use it.

But look at what the bound is telling you. The loss scales with $p_{\max}$. The trouble starts when one symbol takes over.

{{< fig src="img/prefix-codes/huffman_vs_entropy.png" alt="Huffman mean length and entropy as P(A) sweeps from 0.1 to 0.95" caption="Sweep P(A) from 0.1 to 0.95 while the other five symbols share the rest in their CLRS proportions. Below P(A) = 0.5 the two curves touch. Above it they peel apart, because A cannot cost less than one bit no matter how certain it becomes. At P(A) = 0.9 Huffman pays 1.77 times the entropy." >}}

## The one-bit tax

Take the extreme: a two-symbol source with $P(A) = 0.9$, $P(B) = 0.1$. The entropy is $0.469$ bits per symbol. The best prefix code is A `0`, B `1`: one bit per symbol, more than double the floor. Not a bad Huffman tree; the *only* tree. There are no fractional codewords, so any symbol, however probable, costs at least one bit. Call this the one-bit tax. It is the real weakness of every prefix code, and it bites exactly where compression should be easiest: data so predictable that one symbol has $p > 0.5$.

What can be done inside the world of prefix codes? You cannot shrink a codeword below one bit, so change what a symbol *is*. Code pairs: AA has probability 0.81, AB and BA 0.09 each, BB 0.01, and Huffman on this four-letter alphabet gives 1.29 bits per pair, 0.645 per original symbol. Triples get to 0.53. Each grouping gets closer to 0.469, at the price of a table that grows exponentially with the group size. For general data this is a poor trade. For one specific kind of data it is the trick of the century.

That kind is data where the dominant symbol is zero and the zeros come in runs. Take a 4×4 block of quantized DCT coefficients, the kind the [block transforms](../block-transforms/) article produces and JPEG quantizes:

$$\begin{pmatrix} 12 & 3 & 0 & 0 \\ -2 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Thirteen zeros out of sixteen; $p(0) > 0.8$; Huffman on individual coefficients would pay a bit for every one of them. Instead, read the block along the zigzag diagonals, low frequency first, so the zeros collect at the end, and replace each nonzero coefficient with the pair (number of zeros before it, its value), ending with a single end-of-block symbol.

```python
block = np.array([[12, 3, 0, 0], [-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
zig = [(i, s - i) for s in range(7) for i in (range(s, -1, -1) if s % 2 == 0 else range(s + 1)) if 0 <= i < 4 and 0 <= s - i < 4]
seq = [int(block[i, j]) for i, j in zig]
pairs, run = [], 0
last = max(i for i, v in enumerate(seq) if v != 0)
for v in seq[:last + 1]:
    if v == 0: run += 1
    else: pairs.append((run, v)); run = 0
print("zigzag order:", seq)
print("run/value pairs:", pairs, "+ EOB  ->", len(pairs) + 1, "symbols instead of 16")
# zigzag order: [12, 3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# run/value pairs: [(0, 12), (0, 3), (0, -2)] + EOB  -> 4 symbols instead of 16
```

Sixteen symbols became four, and the thirteen zeros cost one EOB codeword between them. The run-length stage has not compressed anything by itself; it has manufactured a new alphabet in which no symbol has $p > 0.5$, so that Huffman, which follows it, is back in the regime where it is nearly optimal. This is exactly JPEG's baseline entropy stage: zigzag, (run, size) pairs, a Huffman table for the pairs, EOB. Every JPEG you have ever opened pays the one-bit tax this way, by refusing to pay it.

{{< lab id="rle" >}}

## A code with no table at all

Now the second place prefix codes live, which looks nothing like the first. The [linear prediction](../linear-prediction/) article produces residuals: the difference between a sample and the predictor's guess. Good predictors produce residuals that are mostly small, occasionally large, with a histogram that falls off roughly geometrically: $P(n) \propto \theta^n$ for some $\theta < 1$. Here are seven of them (signed values folded to non-negative first, as the zigzag mapping in that article does):

$$[3,\ 5,\ 2,\ 37,\ 1,\ 0,\ 8].$$

You could build a Huffman tree for them. But a FLAC decoder on a 2005 music player has no interest in reading a tree out of the file for every block of 4096 samples, and the encoder has no interest in building one. What would be ideal is a prefix code with a closed form: give me the distribution's one parameter and I can write the codeword for any integer directly.

For a geometric distribution that code exists, and Golomb found it in 1966. The Rice special case, which is what everyone implements, works like this: pick a parameter $k$. For an integer $n$, write $q = \lfloor n / 2^k \rfloor$ in unary ($q$ ones then a zero), then write the low $k$ bits of $n$ in plain binary. The unary part says roughly how big $n$ is; the binary part says exactly where. A number costs $q + 1 + k$ bits.

```python
vals = [3, 5, 2, 37, 1, 0, 8]
def rice(n, k):
    q, r = n >> k, n & ((1 << k) - 1)
    return "1" * q + "0" + format(r, f"0{k}b") if k else "1" * q + "0"
for k in (1, 2, 3):
    codes = [rice(v, k) for v in vals]
    print(f"k={k}: {' '.join(codes)}  -> {sum(map(len, codes))} bits (fixed 6-bit: {6*len(vals)})")
print(f"mean residual {np.mean(vals):.1f}, log2(mean) = {np.log2(np.mean(vals)):.2f}")
# k=1: 101 1101 100 11111111111111111101 01 00 111100  -> 40 bits (fixed 6-bit: 42)
# k=2: 011 1001 010 111111111001 001 000 11000  -> 33 bits (fixed 6-bit: 42)
# k=3: 0011 0101 0010 11110101 0001 0000 10000  -> 33 bits (fixed 6-bit: 42)
# mean residual 8.0, log2(mean) = 3.00
```

With $k = 1$ the small values are cheap but the 37 becomes a nineteen-bit unary groan. With $k = 3$ every value costs at least four bits but the 37 shrinks to eight. The sweet spot is where the unary part is short for typical values, which happens when $2^k$ is about the mean: here the mean is 8, $\log_2 8 = 3$, and $k = 2$ and $k = 3$ tie at 33 bits against 42 for a fixed six-bit field. That rule, $k \approx \log_2(\text{mean residual})$, is the whole parameter-selection procedure. FLAC computes the mean of each partition of residuals, picks $k$ from it, writes $k$ in a few bits, and never transmits a table. The code is a prefix code, it is near-optimal for the geometric shape, and the decoder is a handful of shifts and counts. (Golomb's general form uses a divisor $m$ that need not be a power of two, and Gallager and van Voorhis showed exactly which $m$ is optimal for a given $\theta$; Rice's power-of-two restriction loses a few hundredths of a bit and buys hardware simplicity.)

The 37 also shows the Rice code's one real weakness. A single huge residual costs a unary run proportional to its size. Real formats add an escape: if $q$ exceeds some limit, write the limit in unary and then the value in plain binary. The [scalar quantization](../scalar-quantization/) article's escape codes are the same idea one chair over.

{{< lab id="rice" >}}

## Shipping the tree

Huffman's code is only useful if the decoder has the same tree, so somebody has to pay for its transmission. Three options, in order of how much they trust the data. Standard tables, fixed in the specification: JPEG's Annex K tables, MP3's thirty-two tables, DEFLATE's "fixed Huffman" block type, chosen once for typical data and never sent. Semi-static tables: the encoder counts the block, builds the tree, and writes it in the block header; DEFLATE's "dynamic" blocks do this and it costs tens of bytes per block, which is why tiny inputs compress badly. Adaptive Huffman, which rebuilds the tree as symbols arrive, exists and nobody uses it; when you want adaptivity you have left the world of prefix codes anyway.

The header cost has a clever reduction called **canonical Huffman**. Given only the code *lengths*, you can reconstruct a valid prefix code deterministically: sort the symbols by length, then alphabetically within a length, and assign codewords as consecutive binary integers, shifting left each time the length increases. The bit patterns the merge produced are discarded; only the lengths matter, and lengths are small integers, cheap to transmit (DEFLATE even Huffman-codes the lengths). Canonical codes also decode by arithmetic comparison rather than tree walking, which is faster. So the thing shipped in a DEFLATE header is not a tree but a list of lengths, one per symbol, and there are two such lists: one for literals and match lengths together, one for match distances. Two alphabets with different statistics get two codes. That is the "group like statistics" principle from the [channel and layout](../channel-and-layout/) article, applied to the token stream of LZ77.

## The deliberate empty chair

One last option, which is not a defeat. A fixed-length code is a prefix code too: all codewords of length $l$, Kraft sum $N \cdot 2^{-l} \le 1$. It pays the full one-bit tax and more. But it has a property no variable-length code can have: you can find the $i$-th symbol without decoding the $i - 1$ symbols before it. It is at bit $i \cdot l$. For random access, for a GPU thread that needs element 4,097 of a weight matrix and nothing else, for SIMD lanes that must all finish together, that property is worth more than the ratio. The [block float formats](../block-float-formats/) article is about systems that chose the empty chair on purpose, and the [anatomy of a job](../anatomy-of-a-job/) article is about when that is the right call. Every choice of code is a choice about the job.

## Connections

**Morse code** is a prefix-ish code tuned by hand to letter frequency a century before Huffman: E is a single dot, T a single dash, Q is dash-dash-dot-dash. It is not a true prefix code (E's dot is the start of many letters), so it needs gaps between letters, which is why operators' timing mattered and why a telegraph line's human throughput was a research question in the first place.

**Twenty questions.** A prefix code's tree is a decision tree, and reading the codeword is answering yes/no questions about the symbol. The expected number of questions to identify a random symbol is at least its entropy, and Huffman's tree is the best possible question strategy when each question must be answered fully. Arithmetic coding, in the [next article](../arithmetic-and-ans/), is what happens when you are allowed to ask a question and only hear part of the answer.

**DEFLATE's two trees** are the same idea as separate Huffman tables for DC and AC coefficients in JPEG, and as separate codes per residual partition in FLAC: whenever two streams have different histograms, two codes beat one, because one code would be a model mismatched to both and would pay the KL tax from the [first article](../modeling-and-coding/) on each.

**In the game:** the [Woodcutter](/compressionfight/#fighter=huffman), the [Ruler](/compressionfight/#fighter=rice) and the [Repeater](/compressionfight/#fighter=rle) are this family, with the Empty Chair as the fixed-length option. They sit in the Archive Job (DEFLATE), Pixel Perfect (PNG), Studio Master (FLAC: Ruler), the Gallery (JPEG: Repeater then Woodcutter) and Walkman (MP3).
