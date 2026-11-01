---
title: "Modeling and coding: bits are surprise"
slug: modeling-and-coding
chair: foundations
family: "Foundations"
summary: "Why every file has a floor it cannot shrink below, why the floor belongs to the probabilities rather than the coder, and why compressing with the wrong model costs exactly the KL divergence."
fighters: [cm, neural_prior, huffman]
related: [prefix-codes, arithmetic-and-ans, context-models, side-information]
reading_time: 14
---

## The question nobody could price

Imagine a telephone company's traffic department in the spring of 1948. A junior engineer, call her Harriet Lund, has been handed a question by her manager that sounds like bookkeeping and turns out to be philosophy: *how many telegraph symbols per second can this line really carry, if the operators on it "talk like humans"?* The company wants to know how many conversations to sell per wire. The physics is settled: the line can carry so many pulses per second before the pulses smear into each other. What Harriet needs is a different number. Human messages are not random pulses. They are full of habits. How much of a message is habit, and how much is news?

Her first answer is the one everybody gives. There are 26 letters plus a space; twenty-seven things; five bits each, as the Baudot teleprinter already does. Count the letters, multiply by five, done. She knows this is wrong before she writes it down, because the telegraph operators down the hall do not transmit "THE" letter by letter in their heads. They know the next letter after "T-H" is almost always "E". A code that charges "E" five bits there is charging for something the receiver could have guessed.

Her second answer is cleverer. Operators use perhaps two thousand distinct words; eleven bits name one of two thousand; English words average around four and a half letters, so that is about two and a half bits per letter instead of five. Better. But it has the same disease one level up. "THE" is one word in two thousand, yet it is a tenth of everything they send. Charging it eleven bits is as wasteful as charging "E" five. And some words she has never seen, so the two-thousand-word table is a lie about the future.

Each time, the naive answer fails the same way: it prices every symbol the same, while the symbols are not the same. Some are nearly certain and some are shocks. If there is a correct price, it has to depend on how surprising the symbol was. That is as far as Harriet gets on her own. In July her library receives the *Bell System Technical Journal* with a long paper by a mathematician named Shannon, and it contains the price.

## A weather station and a code that fits it

Put the telegraph aside and take the smallest case that shows everything. A weather station in a valley sends one of four reports every hour: SUN, RAIN, SNOW, HAIL. Over years of logs the frequencies settle at

$$p = (0.5,\ 0.25,\ 0.125,\ 0.125).$$

The obvious code is fixed: two bits per report, `00 01 10 11`, forty-eight bits a day, no thought required. Now try a code that matches the frequencies: SUN `0`, RAIN `10`, SNOW `110`, HAIL `111`. It still decodes without separators, because no codeword is the start of another (the [prefix codes](../prefix-codes/) article is about why that property is the whole game). Its average length is

$$0.5\cdot 1 + 0.25\cdot 2 + 0.125\cdot 3 + 0.125\cdot 3 = 1.75\ \text{bits per report},$$

forty-two bits a day instead of forty-eight. So far this is just a good code. The interesting thing is a question: *could a cleverer code do better than 1.75?* Three-bit codewords for the rare symbols look wasteful; could SNOW take `11` if we rearranged? Try a few and you will find that every rearrangement that shortens one word lengthens another, and the average never drops below 1.75. Something is holding the floor there, and it is not the cleverness of the code.

Here is what is holding it. Look at the lengths we chose next to the probabilities: SUN has $p = 1/2$ and got 1 bit; RAIN has $p = 1/4$ and got 2; SNOW and HAIL have $p = 1/8$ and got 3. In every case the length is $-\log_2 p$. The code is not merely "shorter for common symbols"; it charges each symbol exactly the logarithm of its improbability. And the average of that charge is

$$H(p) = -\sum_i p_i \log_2 p_i = 1.75.$$

{{< fig src="img/modeling-and-coding/cost_curve.png" alt="The curve of minus log2 p against p, with the two stations' symbols marked on it" caption="The price of a symbol. A symbol with probability 1/2 costs one bit, 1/4 costs two, 1/8 costs three. On the desert station SUN at 0.9 would cost 0.15 bits and HAIL at 0.02 would cost 5.6; a fixed code charges two for everything." >}}

## Why the price is a logarithm

Here is the question that matters, the one the whole field hangs from: *why that function?* Why should the right charge for a symbol be $-\log_2 p$ and not, say, $1/p$ or $1 - p$?

Because of what happens to sequences. The valley station's day is twenty-four reports. If the reports are independent, the probability of a specific day's log is the product of twenty-four probabilities. A day of all SUN has probability $(1/2)^{24}$. A day with twelve SUN, six RAIN, three SNOW, three HAIL has probability $(1/2)^{12}(1/4)^{6}(1/8)^{3}(1/8)^{3} = 2^{-42}$. Now, how many bits does it take to name one specific thing among all the things that could have happened? If the possibilities were equally likely you would need $\log_2(\text{count})$ bits. They are not equally likely, but the typical day has probability about $2^{-42}$, so there are, roughly, $2^{42}$ "typical days" and you need about 42 bits to say which one happened. Forty-two is exactly $24 \times 1.75$. The logarithm is forced on us by one fact: probabilities of sequences multiply, and lengths of codes add. The only function that turns a product into a sum is a logarithm. So the price of a symbol must be $-\log_2 p$, prices are additive over a message, and the average price per symbol is the entropy.

This is the first thing to take away, and it is worth stating plainly because it is so often misread. **The floor is a property of the probabilities, not of the coder.** Shannon's source coding theorem (we state it, we do not prove it) says: for a source of independent symbols with distribution $p$, any lossless code must spend at least $H(p)$ bits per symbol on average, and codes exist that come arbitrarily close to $H(p)$ when symbols are coded in long blocks. No ingenuity moves the floor. Ingenuity can only find the floor and stand on it.

```python
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log2(p)).sum())

stations = {"valley": [0.5, 0.25, 0.125, 0.125],
            "desert": [0.9, 0.05, 0.03, 0.02],
            "hills":  [0.25, 0.5, 0.125, 0.125]}
for name, p in stations.items():
    print(f"{name:7s} H = {entropy(p):.3f} bits/report, prices = {np.round(-np.log2(p), 2)}")
# valley  H = 1.750 bits/report, prices = [1. 2. 3. 3.]
# desert  H = 0.618 bits/report, prices = [0.15 4.32 5.06 5.64]
# hills   H = 1.750 bits/report, prices = [2. 1. 3. 3.]
```

Three lines of NumPy, and the valley's floor comes out at the 1.75 we found by hand. Two other stations have appeared in that listing; we will need both.

{{< lab id="huffman" >}}

## The desert station, where the coder becomes the bottleneck

The second station sits in a desert: $p = (0.9, 0.05, 0.03, 0.02)$. The entropy function says $H = 0.618$ bits per report. SUN, at $p = 0.9$, should cost $-\log_2 0.9 = 0.15$ bits. But a prefix code assigns whole codewords, and the shortest possible codeword is one bit long. The best prefix code for this station is SUN `0`, RAIN `10`, SNOW `110`, HAIL `111` again, and it averages $0.9 + 0.1 + 0.09 + 0.06 = 1.15$ bits, nearly double the floor. No prefix code of any kind can beat 1.0 here, because SUN alone, at one bit each, already costs 0.9 bits per report on average.

So for the desert, the floor is 0.62 and the best whole-bit code sits at 1.15. The gap is not in the probabilities; we know them perfectly. The gap is in the coder, which cannot charge a fraction of a bit. The way out is to charge for several symbols at once, so that a run of nine SUNs can cost nine times 0.15 bits, a bit and a half in total, rather than nine bits. Arithmetic coding and ANS do precisely that, and the [fractional bits](../arithmetic-and-ans/) article is where the desert station gets its 0.62. For now the lesson is the shape of the problem: *entropy is the target, and a coder is judged by how close it gets.*

## The split

Step back from the stations and notice what we did. We never asked how the weather works. We asked only for the probability of each report, and once we had it, the price of each report followed mechanically. That division is the organizing idea of the entire field, and it deserves its own vocabulary.

A **model** is anything that assigns a probability $P(\text{symbol} \mid \text{everything seen so far})$ to the next symbol. The valley model is a four-entry table; it ignores the past entirely. A better model would notice that SNOW is likelier in January, or that the hour after RAIN is rarely SUN, and would assign sharper probabilities.

A **coder** is a machine that, handed a symbol and the probability its model gave it, emits about $-\log_2 P$ bits, and whose decoder, handed the same probabilities, recovers the symbol. Huffman is a coder, arithmetic coding is a coder, ANS is a coder.

**Every compressor is these two halves.** The coder half is close to solved: modern coders sit within a fraction of a percent of $-\log_2 P$. What is not solved, and never will be, is the model half, because a model is a claim about the world. LZ77 is a model ("the next bytes are the ones that followed this phrase last time"). A DCT plus a quantizer plus a run-length stage is a model of photographs. A context mixer is a model of text. A language model is a model of text with a few billion parameters. And when you plug a large language model into an arithmetic coder, with nothing else changed, you get a compressor that takes English to roughly 0.8 bits per character, where zstd manages about 2.5 and the raw file was 8. A 1 GB text archive: 310 MB with zstd, around 100 MB with the LLM. None of that improvement came from the coder. All of it came from the model knowing more about English.

That is the claim this site rests on, and you can see it in the [map](../../): one chair for the coder (the Packer), and nearly every other chair is some species of model. "How well can you compress $X$" and "how well can you predict $X$" are the same question, because the compressed size *is* the sum of the log-losses of the predictions.

{{< lab id="cm" >}}

## The hills station, and what it costs to be wrong

Now the third station, in the hills, where it rains more than it shines: $P = (0.25, 0.5, 0.125, 0.125)$. Its entropy is 1.75, the same as the valley's. But suppose the hill station was built by the same contractor, who installed the valley's code (SUN `0`, RAIN `10`, …) without re-measuring. The code is the one appropriate to $Q = (0.5, 0.25, 0.125, 0.125)$ and the weather is $P$. How much does the hill station pay?

Each report costs the length the valley code gives it, and the lengths are $-\log_2 Q$. The average under the true frequencies $P$ is

$$H(P, Q) = -\sum_i P_i \log_2 Q_i = 0.25\cdot 1 + 0.5\cdot 2 + 0.125\cdot 3 + 0.125\cdot 3 = 2.0\ \text{bits}.$$

Two bits per report. The clever code, applied to the wrong station, does exactly as well as the fixed code that never had an opinion about the weather at all. The floor was 1.75; the mismatch costs a quarter of a bit on every report, six bits a day, forever.

That quarter bit has a name and a formula. Write the actual cost minus the floor:

$$H(P,Q) - H(P) = \sum_i P_i \log_2 \frac{P_i}{Q_i} = D_{\mathrm{KL}}(P \,\|\, Q).$$

The term on the right is the Kullback–Leibler divergence of the truth from the model, and the identity

$$H(P, Q) = H(P) + D_{\mathrm{KL}}(P \,\|\, Q)$$

is the single most useful equation for thinking about compressors. It says the bill always splits into two parts: the part that is genuinely unpredictable, $H(P)$, which no one can avoid, and the part you pay for not knowing $P$, which is exactly the KL divergence of your model from the truth.

```python
P = np.array([0.25, 0.5, 0.125, 0.125])   # the hills, what really happens
Q = np.array([0.5, 0.25, 0.125, 0.125])   # the valley code we shipped
cross = float(-(P * np.log2(Q)).sum())      # what we actually pay per report
kl = float((P * np.log2(P / Q)).sum())      # the mismatch tax
print(f"H(P) = {entropy(P):.3f}   H(P,Q) = {cross:.3f}   D_KL(P||Q) = {kl:.3f}")
print(f"H(P) + D_KL = {entropy(P) + kl:.3f}  (equals H(P,Q))")
# H(P) = 1.750   H(P,Q) = 2.000   D_KL(P||Q) = 0.250
# H(P) + D_KL = 2.000  (equals H(P,Q))
```

{{< fig src="img/modeling-and-coding/three_stations.png" alt="Grouped bars for the three stations: the fixed code, the entropy, the best prefix code, and the cost under the valley's model" caption="Three stations, one coder. In the valley the matched code hits the floor exactly. In the desert the floor is 0.62 but no whole-bit code can get under 1.0. In the hills the entropy is 1.75 again, but the valley's code charges 2.0: the extra 0.25 bits per report is the KL divergence." >}}

Read that equation once more with a different emphasis. The KL divergence is usually introduced as an abstract measure of distance between distributions. Here it is a byte count. "How much did you know about the data" is not a figure of speech; it is the difference between two file sizes. A model trained on the wrong corpus, a dictionary built from last year's logs, a quantization table tuned for a different kind of image: each pays $D_{\mathrm{KL}}$, measurably, in the output. The [side information](../side-information/) article is largely about engineering that term down.

## Compressed output should look like noise

There is a practical test hiding in the theorem. If a coder is charging $-\log_2 P$ per symbol and the model is right, the output has entropy equal to its length: one bit of information per bit. A stream with one bit of information per bit has no structure left in it at all. Its zeros and ones are equally frequent, its pairs are equally frequent, nothing predicts anything. It is indistinguishable from coin flips.

```python
rng = np.random.default_rng(0)
code = {0: "0", 1: "10", 2: "110", 3: "111"}
reports = rng.choice(4, size=100_000, p=[0.5, 0.25, 0.125, 0.125])
bits = "".join(code[r] for r in reports)
print(f"bits per report: {len(bits) / len(reports):.4f}   (H = 1.75)")
print(f"fraction of 1s: {bits.count('1') / len(bits):.4f}")
pairs = [bits[i:i + 2] for i in range(0, len(bits) - 1, 2)]
print("2-bit blocks:", {k: round(pairs.count(k) / len(pairs), 3) for k in ("00", "01", "10", "11")})
# bits per report: 1.7482   (H = 1.75)
# fraction of 1s: 0.4983
# 2-bit blocks: {'00': 0.251, '01': 0.248, '10': 0.254, '11': 0.247}
```

A hundred thousand valley reports through the valley code: 1.748 bits each, half the bits are ones, every two-bit block appears a quarter of the time. The output is noise. This is the sanity check you can run on any format: gzip a `.zip` file and it does not shrink, because DEFLATE's entropy stage already left nothing. If a compressor's output *does* compress further, some redundancy survived, which means some probability was mis-estimated, which means there is a KL term still being paid.

## What Harriet's answer looks like

Harriet's two naive answers were both models, and both were bad ones. "Five bits per letter" is the uniform model, $Q = 1/27$ for every letter; the KL divergence from real English is enormous, about three bits per letter. "Eleven bits per word" is a better model that still ignores context. Shannon's answer was not a number but a method: measure $P$ as carefully as you can, with as much context as you can afford, and the capacity of the line in human sentences is the line's bits per second divided by the entropy per letter. In 1951 he ran a guessing game with human subjects to estimate that entropy and put it near one bit per letter. The [context models](../context-models/) article picks that game up, because it is exactly where modern text compression starts.

## Connections

**Perplexity.** Language models report perplexity, and perplexity is $2^{\text{bits per token}}$ when the bits are the model's cross-entropy on held-out text. The desert station has perplexity $2^{0.618} = 1.53$: on average the next report is as uncertain as a choice among 1.53 equally likely options. A model with perplexity 20 on English is one that spends $\log_2 20 = 4.3$ bits per token, which, divided by about four characters per token, is a bit per character. Perplexity leaderboards are compression leaderboards wearing a different unit.

**Kelly betting.** A gambler who believes the next report will be SUN with probability 0.9 and bets a fraction of her wealth accordingly grows her wealth at a rate set by the same sum, $\sum P_i \log_2 (\text{odds})$. The optimal bet fractions are the probabilities, the growth rate per round is the log-score of the forecast, and a forecaster who uses the wrong $Q$ grows slower by exactly $D_{\mathrm{KL}}(P\|Q)$ per round. Compression and compounding are the same arithmetic because both multiply probabilities and only care about the logarithm.

**Occam and MDL.** If a model's quality is its cross-entropy on the data, then "the best explanation is the shortest" becomes a theorem instead of a slogan. Rissanen's minimum description length principle says to pick the model that minimizes (bits to describe the model) + (bits to describe the data under the model). A model that compresses the data is a model that has found structure in it; a model that merely memorizes it is as long as the data. The second term in that sum is this article; the first is the [side information](../side-information/) article.

**In the game:** the [Woodcutter](/compressionfight/#fighter=huffman) is the first coder; the [Mentalist](/compressionfight/#fighter=cm) and [Oracle Prime](/compressionfight/#fighter=neural_prior) are the extreme of the model half. The split appears in every heist, and the Library heist (bzip2 / xz / PAQ) is the one where only the model is allowed to matter.
