---
title: "Arithmetic coding and ANS: fractional bits"
slug: arithmetic-and-ans
chair: entropy
family: "Fractional-bit coders"
summary: "How a whole message becomes one number so that a symbol can cost 0.03 bits, why that is what makes adaptive and context modeling possible, and how ANS packs the same information into a single integer that pops like a stack."
fighters: [arith, ans]
related: [prefix-codes, context-models, modeling-and-coding]
reading_time: 15
---

## A symbol that costs less than a bit

The [prefix codes](../prefix-codes/) article ended on a wall. A source that sends A ninety percent of the time has an entropy of 0.47 bits per symbol, and the best prefix code pays 1.0, because a codeword cannot be shorter than one bit. The desert weather station from the [first article](../modeling-and-coding/) hit the same wall at 0.62 versus 1.15. Grouping symbols into pairs and triples chips at the wall but never removes it, and the tables grow exponentially while you chip.

So here is the question, and it should sound slightly absurd: *how can a symbol cost 0.03 bits?* A bit is the smallest thing there is. You cannot write three hundredths of a zero into a file. Whatever the answer is, it cannot be "a shorter codeword". It has to be a different idea of what a codeword is.

The idea was in the air at MIT in the early 1960s; Peter Elias is usually credited with the observation, though he never published it, and Shannon's 1948 paper contains a version of it as a proof device. It became an algorithm at IBM in 1976, in a paper by Jorma Rissanen and a Stanford thesis by Richard Pasco, and a practical one in Rissanen and Langdon's 1979 paper that gave it its name: arithmetic coding. The name is apt. The message is not a sequence of codewords at all. It is one number.

## One number for the whole message

Take the smallest example that shows the mechanism. Three symbols, $P(A) = 1/2$, $P(B) = 1/4$, $P(C) = 1/4$, and the message ABAC. Lay the unit interval $[0, 1)$ on a table and give each symbol a slice proportional to its probability, in a fixed order: A owns $[0, 0.5)$, B owns $[0.5, 0.75)$, C owns $[0.75, 1)$.

To code the first symbol, A, keep A's slice and throw the rest away. The interval is now $[0, 0.5)$. To code the next symbol, B, treat the current interval as if it were the whole table: subdivide $[0, 0.5)$ in the same proportions, so A gets $[0, 0.25)$, B gets $[0.25, 0.375)$, C gets $[0.375, 0.5)$, and keep B's. Then A again: keep the first half of $[0.25, 0.375)$, which is $[0.25, 0.3125)$. Then C: the last quarter of that, $[0.296875, 0.3125)$.

```python
import numpy as np
model = {"A": (0.0, 0.5), "B": (0.5, 0.75), "C": (0.75, 1.0)}   # [low, high) share of each symbol
lo, hi = 0.0, 1.0
for s in "ABAC":
    width = hi - lo
    lo, hi = lo + width * model[s][0], lo + width * model[s][1]
    print(f"after {s}: [{lo:.6f}, {hi:.6f})  width {hi - lo:.6f} = 1/{round(1 / (hi - lo))}")
print(f"bits = -log2(width) = {-np.log2(hi - lo):.1f}")
# after A: [0.000000, 0.500000)  width 0.500000 = 1/2
# after B: [0.250000, 0.375000)  width 0.125000 = 1/8
# after A: [0.250000, 0.312500)  width 0.062500 = 1/16
# after C: [0.296875, 0.312500)  width 0.015625 = 1/64
# bits = -log2(width) = 6.0
```

{{< fig src="img/arithmetic-and-ans/interval_zoom.png" alt="Five rows of intervals, each zooming into the slice of the symbol coded" caption="ABAC as one number. Each symbol keeps its slice of the current interval and the next symbol subdivides that slice. The final interval has width 1/64; the six-bit fraction 0.010011 lies inside it, and six bits is exactly 1 + 2 + 1 + 2." >}}

The message is now any number inside $[0.296875, 0.3125)$. The decoder, holding the same model, finds that number in A's slice of $[0,1)$, so the first symbol was A; zooms into A's slice and finds the number in B's part of it; and so on, until it has read four symbols and stops (the length is sent, or a terminating symbol is). We have replaced "a codeword per symbol" with "a region per message". How many bits does it take to name a point in that region? A binary fraction with $b$ digits after the point can land on any multiple of $2^{-b}$, and the interval has width $1/64$, so some six-digit fraction must fall inside it: $0.010011_2 = 19/64 = 0.296875$ does. Six bits for ABAC.

Now the thing to notice, the one the whole article hangs on. The width of the final interval is the product of the probabilities of the symbols: $\tfrac12 \cdot \tfrac14 \cdot \tfrac12 \cdot \tfrac14 = 1/64$. The number of bits needed is $-\log_2$ of that width, and the logarithm of a product is a sum:

$$\text{bits} = -\log_2 \prod_i p_i = \sum_i -\log_2 p_i = 1 + 2 + 1 + 2 = 6.$$

Each symbol narrowed the interval by its probability, so each symbol contributed exactly $-\log_2 p$ to the length of the final number. Nothing in that argument needed $-\log_2 p$ to be an integer. If $P(A)$ were 0.9, coding an A would narrow the interval by a factor of 0.9, which lengthens the final fraction by $-\log_2 0.9 = 0.152$ bits. Code nine A's and the interval shrinks by $0.9^9$, costing 1.37 bits for all nine together. **The cost is additive and fractional because the message is one number, and lengths of numbers add when their widths multiply.** The one-bit tax does not exist here; there is no codeword to be at least one bit long. The desert station gets its 0.62.

Try AAAA with the same model: four halvings, $[0, 1/16)$, four bits, which is what a prefix code would also pay. Arithmetic coding never does worse than $-\log_2 p$ per symbol plus a couple of bits for the whole message; it simply stops rounding up.

{{< lab id="arith" >}}

{{< fig src="img/arithmetic-and-ans/bits_vs_p.png" alt="Bits charged versus probability for arithmetic coding against the one-bit floor, and average bits on a binary source" caption="Left: the price of a likely symbol, minus log2 p, versus the one bit a prefix code must charge. Right: averaged over a two-symbol source, Huffman pays 2.1 times the entropy at P(A) = 0.9 and 12 times at 0.99." >}}

## The model may change its mind every symbol

The second thing to notice is quieter and matters more. Look at the loop again. At each step, the encoder takes the current interval and subdivides it according to a model. Nothing in the mechanism requires the model to be the same at every step. It only requires that the decoder, which replays the same steps, use the same model at the same step. And the decoder, by the time it reaches step $k$, has decoded symbols $1$ through $k-1$. So the model at step $k$ may be any function whatsoever of the symbols so far.

This is the property that changed what compression is. A prefix code is a table built before coding starts, and swapping tables per symbol is impractical (you would ship a tree per symbol). An arithmetic coder consumes a probability distribution per symbol as naturally as it consumes a symbol, so a model may adapt: count frequencies as they arrive, condition on the previous two bytes, run a neural network. The [context models](../context-models/) article is entirely about models of this kind, and every one of them plugs straight into the loop above. PPM, CABAC, PAQ and "a language model plus an arithmetic coder" are all the same two lines of code with a different function supplying `model`. The Oracle chair in the game has a socket, and this is the socket.

## Finite precision, in words

The toy above used Python floats and would die after a few dozen symbols, when the interval width underflows. Real coders keep the interval as two integers, `low` and `range`, in 32 or 64 bits, and renormalize: whenever the top bits of `low` and `low + range` agree, those bits are settled forever, so shift them out to the output and shift the interval up to use the freed precision. The interval is thereby kept from collapsing, at the cost of a small, bounded rounding loss per symbol. There is one nuisance: a later carry can propagate into bits already emitted (the interval straddles a boundary like 0.0111… / 0.1000…), and every implementation has a scheme for holding those bits back until the straddle resolves. The **range coder**, Martin's 1979 variant, does the same thing renormalizing a byte at a time instead of a bit, which is faster and is what LZMA and most "range coders" you meet actually are. Arithmetic coder and range coder are one algorithm with two bookkeeping styles.

A second simplification runs the other way. If the alphabet is binary, the "subdivide" step is one multiply (or, with the probability quantized to a few bits, one table lookup), and the symbol's slice is just "the bigger part or the smaller part". **Binary arithmetic coders** of this kind run inside H.264 and HEVC as CABAC, and inside JPEG 2000 and JBIG2 as the MQ coder. Every multi-valued syntax element is first *binarized* into a string of decisions, each decision gets a context (a small adaptive probability chosen by what neighbouring blocks did), and the binary coder charges $-\log_2 p$ for each. CABAC's range update is a 64-state × 4-bucket lookup table and no multiplier at all, which is what it took to put adaptive arithmetic coding into a 200 mW hardware decoder.

## The thicket

Here the story leaves mathematics for twenty years. Arithmetic coding in practical form was patented by IBM, and related patents followed from Mitsubishi, AT&T and others, around the Q-coder and its relatives. The JPEG standard of 1992 includes an arithmetic-coded mode (the QM-coder) that produces files several percent smaller than the baseline Huffman mode; almost no encoder has ever emitted it and almost no decoder supports it, because implementing it meant a licence. When Julian Seward wrote bzip2 in 1996 he deliberately replaced the arithmetic coder of his earlier bzip with Huffman coding to stay clear of the patents, and bzip2's ratio was capped by that choice from the day it shipped. For two decades, the best entropy coder known was one that free software could not use, and prefix codes stayed in every format that mattered. The patents have all expired now, but they shaped the installed base, and the installed base is what formats are made of.

The constraint also drove a technical question that turned out to have a good answer: arithmetic coding pays a multiply and a renormalization per symbol and is strictly serial; Huffman decoding is a table lookup. Even without the patents, could you have the ratio of one at the speed of the other?

## ANS: the same information, one integer

In a series of papers between 2007 and 2014, Jarek Duda answered yes with a construction he called asymmetric numeral systems, and he put it in the public domain. It starts from a reframing. Arithmetic coding keeps a *sub-interval* and narrows it. ANS keeps a *single natural number* $x$, the state, and grows it. Coding a symbol of probability $p$ should multiply the state by about $1/p$, so that $\log_2 x$ grows by $-\log_2 p$; after the whole message, $\log_2 x$ is the number of bits the message cost, and the bits of $x$ are the output.

Here is the cheapest form, **rANS**. Choose a denominator $M$ and integer frequencies $f_s$ summing to $M$, so $P(s) = f_s / M$; also the cumulative frequencies $c_s$. Take $M = 4$, $f_A = 3$, $f_B = 1$, $c_A = 0$, $c_B = 3$. To push a symbol $s$ onto a state $x$:

$$x' = \left\lfloor \frac{x}{f_s} \right\rfloor \cdot M + c_s + (x \bmod f_s).$$

Read it as: divide $x$ into $f_s$-sized chunks, spread the chunk count across $M$ (that is the $\times M / f_s \approx 1/p$ growth), and store the leftover $x \bmod f_s$ together with the symbol's offset $c_s$ in the low digit. The low digit, $x' \bmod M$, lands in $[c_s, c_s + f_s)$, which is $s$'s slice of $M$. So the state's last base-$M$ digit tells the decoder which symbol was pushed last, and everything else in $x'$ reconstructs the previous $x$.

```python
M, f, cum = 4, {"A": 3, "B": 1}, {"A": 0, "B": 3}
x = 4
for s in "AAB":
    x_new = (x // f[s]) * M + cum[s] + (x % f[s])
    print(f"push {s}: x {x} -> {x_new}   log2 grew by {np.log2(x_new / x):.3f} (ideal {-np.log2(f[s] / M):.3f})")
    x = x_new
out = []
while x > 4:
    slot = x % M
    s = "A" if slot < 3 else "B"
    x = f[s] * (x // M) + slot - cum[s]
    out.append(s)
    print(f"pop {s}: x -> {x}")
print("decoded (reversed):", "".join(out))
# push A: x 4 -> 5   log2 grew by 0.322 (ideal 0.415)
# push A: x 5 -> 6   log2 grew by 0.263 (ideal 0.415)
# push B: x 6 -> 27   log2 grew by 2.170 (ideal 2.000)
# pop B: x -> 6
# pop A: x -> 5
# pop A: x -> 4
# decoded (reversed): BAA
```

Encoding A, A, B from $x = 4$ gives $4 \to 5 \to 6 \to 27$. The state grew from $\log_2 4 = 2$ to $\log_2 27 = 4.75$ bits, 2.75 bits for three symbols, against the ideal $0.415 + 0.415 + 2 = 2.83$; with a state this tiny the rounding is visible, and in real use the state is 32 or 64 bits and the per-symbol loss is thousandths of a bit. Decoding reads $27 \bmod 4 = 3$, which is in B's slice $[3, 4)$, so the last symbol was B, and inverts the formula to get back to 6; then $6 \bmod 4 = 2$ is in A's slice, back to 5; then 4. **The symbols come out in reverse.** That is not a bug to fix; it is the price of the construction. The state is a stack: the last symbol pushed is the first popped. Encoders therefore run through the block backwards, so that the decoder reads it forwards, which means ANS wants its data in blocks and sits awkwardly with a model that must adapt in decoding order. Static-per-block tables suit it perfectly, and that is where it lives.

{{< lab id="ans" >}}

Two refinements turn the toy into the fastest entropy coder in deployment. First, renormalization, as before: keep $x$ inside a window like $[2^{31}, 2^{63})$ by shifting whole bytes or 32-bit words out before each push and back in before each pop; the state never overflows and output comes in word-sized chunks. Second, in **tANS** (Duda's tabled variant; Yann Collet's FSE in zstd is the famous implementation), the state is kept small, say in $[L, 2L)$ with $L = 4096$, and the push function for every (state, symbol) pair is precomputed into a table. Encoding and decoding become "look up the next state, emit some bits", with no division and no multiplication, which is Huffman's cost. The compression is arithmetic coding's, within a few hundredths of a bit of the entropy. The trade-off between the two coders that had stood since 1976 simply disappeared.

rANS keeps the multiply but gains something tANS cannot: because the state update depends only on the state and the symbol, two or more independent states can be interleaved, symbol by symbol, in one stream, and a decoder can advance all of them in parallel in SIMD lanes. Fabian Giesen's interleaved rANS from 2014 is why the entropy stage of a modern codec can run at gigabytes per second on a CPU and why CacheGen can decode KV-cache streams on a GPU. tANS is the choice when there is one stream and a small alphabet per table (zstd's literal lengths, match lengths and offsets; Apple's LZFSE); rANS is the choice when there are many symbols per table or many lanes to fill (JPEG XL, Draco's 3D meshes, the CRAM genomics format). AV1, for comparison, kept a multi-symbol adaptive range coder, because it wanted per-symbol adaptation more than it wanted lanes.

## Where each one is used, and why

Lay the three families on the job axes from the [anatomy of a job](../anatomy-of-a-job/) article. A prefix code is the cheapest decoder and the only one that is stateless enough to resynchronize cheaply; it stays where symbols are nearly uniform after a front end (DEFLATE after LZ77, JPEG after run-length). An adaptive arithmetic or range coder is where the model must change every symbol and the decoder may be serial: context modeling, CABAC, LZMA's literal coder, PAQ, learned codecs, an LLM on each end. ANS is where the distribution is static per block and throughput matters: zstd, JPEG XL, the entropy stage of any new format designed after 2015. The ratio differences among the three on well-modeled data are small. The differences in speed, parallelism and what kind of model they can be attached to are the whole story, and they are differences in what the job allows.

## Connections

**A bijection between messages and numbers.** Arithmetic coding is the same construction as enumerative coding (Cover, 1973): order all possible messages lexicographically by their probability intervals, and the code for a message is its index among them. Every message of a given length maps to a disjoint sub-interval of $[0, 1)$ and the sub-intervals tile it exactly, which is the Kraft equality from the [prefix codes](../prefix-codes/) article with real-valued lengths. The coder is a bijection between strings and reals in which likely strings get short reals.

**Compounding.** The interval width after $n$ symbols is $\prod p_i$, and the bits are $\sum -\log_2 p_i$. That is the same arithmetic as a Kelly gambler's wealth after $n$ rounds, which multiplies by a factor each round and whose logarithm therefore adds. A coder narrowing an interval and a gambler compounding a bankroll are the same process read in opposite directions, which is why log-loss shows up in both and why a prediction market's prices are a probability model you could compress with.

**Exactness at both ends.** The decoder must reproduce the encoder's model bit for bit at every step; the moment it does not, every later symbol is wrong. For a counting model this is trivial. For a neural network it is not: floating-point summation order differs between GPUs, between kernels, between batch sizes, and a probability that differs in the last ulp can move a symbol across a slice boundary. That is why LLM-based compressors quantize the model's output probabilities to integers before coding, and why "symmetric" in a fighter card means more than "same cost": it means *identical*.

**In the game:** the [Fencer](/compressionfight/#fighter=arith) and the [Stacker](/compressionfight/#fighter=ans) are this family. The Fencer is ideal in Prime Time (CABAC), the Library (PAQ) and the Courier (CacheGen); the Stacker in the Log Pipe (zstd's FSE), and both are accepted substitutes for the Woodcutter almost everywhere else.
