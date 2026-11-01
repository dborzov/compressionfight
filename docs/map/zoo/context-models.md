---
title: "Context models: predicting the next symbol"
slug: context-models
chair: predict
family: "Statistical context models"
summary: "After 'th', how sure are you the next letter is 'e'? How a model turns that certainty into fewer bits, what it does when a context has never seen a symbol, and why any predictor, from a count table to a transformer, is a compressor."
fighters: [cm, neural_prior, arith]
related: [arithmetic-and-ans, modeling-and-coding, side-information]
reading_time: 15
---

## A guessing game in a living room

In 1951 Claude Shannon published a short paper called *Prediction and Entropy of Printed English*, and the experiment in it was done at home. He would take a passage of text his subject had not read, cover it, and ask her to guess the next letter. If she was wrong, she guessed again, and again, until she got it. His principal subject was his wife, Betty, and she was good at it. Given a hundred letters of context she often needed only one guess; English, it turned out, carries about one bit of information per letter, when the reader is a person who knows the language. The 4.75 bits a 27-symbol alphabet nominally needs, or the 4.1 you get from letter frequencies alone, were mostly habit, not news.

The experiment measured something no code of the time could cash in. Shannon's guesser was a *model*: a function from everything seen so far to a probability distribution over the next letter. The [first article](../modeling-and-coding/) argued that a compressor is a model plus a coder, and that the coder charges $-\log_2 P(\text{actual} \mid \text{context})$. So the question this article answers is Betty's question, stated for machines: *after "th", how sure are you the next letter is "e", and how do you turn that certainty into fewer bits?* The second half of the question has a short answer, the [arithmetic coder](../arithmetic-and-ans/), which accepts a fresh distribution for every symbol. The first half is the whole of this article, because it is where every bit of ratio in the lossless world is actually won.

## A sentence, a context, a count

Take a text short enough to keep in your head, with spaces written as underscores:

```
the_theme_of_the_thesis
```

Twenty-three characters. Suppose we are about to code the "e" after the final "th" (position 15), and the model is allowed to look at the two previous characters: an **order-2** model. What does it know? It scans the past for every occurrence of "th" and counts what followed. There were two earlier ones, at "the" and "theme", and both were followed by "e". Two for two.

The naive move is to say $P(e \mid th) = 1$. That is a catastrophe waiting to happen, because the coder charges $-\log_2 P$, and $-\log_2 0 = \infty$: if the next character after some "th" is ever anything else, the file cannot be written. Every count-based model needs to leave probability for things it has not seen. The oldest way is Laplace's: add one to every count, as if each of the 27 symbols had been seen once before the data started.

```python
import numpy as np
text = "the_theme_of_the_thesis"
A = 27
def counts_after(text, ctx):
    out = {}
    for i in range(len(ctx), len(text)):
        if text[i - len(ctx):i] == ctx:
            out[text[i]] = out.get(text[i], 0) + 1
    return out
for ctx in ("th", "he", "e_"):
    c = counts_after(text, ctx)
    n = sum(c.values())
    print(f'after "{ctx}": {c}  P(e) = ({c.get("e",0)}+1)/({n}+{A}) = {(c.get("e",0)+1)/(n+A):.3f}'
          f'  -> {-np.log2((c.get("e",0)+1)/(n+A)):.2f} bits')
# after "th": {'e': 4}  P(e) = (4+1)/(4+27) = 0.161  -> 2.63 bits
# after "he": {'_': 2, 'm': 1, 's': 1}  P(e) = (0+1)/(4+27) = 0.032  -> 4.95 bits
# after "e_": {'t': 2, 'o': 1}  P(e) = (0+1)/(3+27) = 0.033  -> 4.91 bits
```

(The listing counts over the whole string, four "th"s in all, which is what the game's lab does; the arithmetic is the same.) After "th", with four sightings of "e" and nothing else, Laplace gives $P(e \mid th) = 5/31 = 0.161$, and the coder charges $2.63$ bits for an "e" that any English reader would have bet the house on. After ten more sightings, $15/41 = 0.366$, 1.45 bits. It takes twenty-five consecutive "e"s before the model will even say fifty-fifty. Laplace smoothing is honest about the unseen but absurdly slow to trust the seen, because it spreads 27 phantom counts over an alphabet in which, after "th", twenty-five of the symbols will never appear.

{{< lab id="cm" >}}

Real context models fix this in two ways, and both are worth knowing. The first is to replace the counts with a **state that moves at a fixed rate**: keep $P$ for the symbol (or, in a binary model, for the bit), and after each observation move it a fixed fraction of the way toward what happened.

```python
p = 0.5                      # P(next is 'e' | th), binary, starts undecided
for hit in range(1, 15):
    p += (1.0 - p) / 8       # every 'e' moves p an eighth of the way to 1
    if hit in (1, 4, 14):
        print(f"after {hit:2d} hits: P = {p:.3f}, cost of the next e = {-np.log2(p):.2f} bits")
# after  1 hits: P = 0.562, cost of the next e = 0.83 bits
# after  4 hits: P = 0.707, cost of the next e = 0.50 bits
# after 14 hits: P = 0.923, cost of the next e = 0.12 bits
```

Four sightings and the "e" costs half a bit instead of 2.63; fourteen and it costs 0.12. The rate, here $1/8$, is a dial between how fast the model trusts and how badly it is fooled by a fluke, and real models often start with a fast rate and slow it as the count grows (which is what CABAC's probability-state machine and PAQ's adaptive counters both do under the hood). The second fix is a better estimate of how much probability the unseen deserve, which is the subject of the next section, because it turns out to be the central engineering problem of the whole family.

## Longer contexts, and the cliff

An order-2 model knows "th" is followed by "e". An order-3 model knows "the" is followed by a space or an "m"; an order-5 model knows "thesi" is followed by "s". More context, sharper predictions, fewer bits. So why not order 20?

Because of the table. An order-$k$ model over an alphabet of $A$ symbols has up to $A^k$ contexts, and every context has to be *seen* before it can predict anything. On real text the tables grow until memory runs out, and worse, the long contexts are mostly empty, so the model spends its time with no evidence at all. Here is the effect measured on a 55 KB English sample (this site's primer), coding adaptively in one pass with a Laplace-smoothed order-$k$ table and nothing else:

{{< fig src="img/context-models/bits_vs_order.png" alt="Bits per character versus context order for an order-k model alone and for PPM with escapes" caption="Bits per character against context order on a 55 KB English sample. An order-k table alone (red) improves from order 0 to order 1 and then falls off a cliff: by order 4 it is worse than sending raw bytes, because nearly every context is new. The same tables with PPM's escape mechanism (green) keep improving to 2.81 bits per character at order 4, below xz at 3.05 and zlib at 3.25 on the same bytes." >}}

The red curve is the cliff. Order 1 is the best the lone table manages, 4.44 bits per character; by order 4 it is paying 6.7 bits, nearly a raw byte, because almost every four-character context in a 55 KB file is being seen for the first time and Laplace has 256 phantom counts spread over it. The sharper the model's *potential* knowledge, the less *actual* knowledge it has per context. A long-context model on its own is a confident ignoramus.

## Escapes: paying for what you did not know

So the question that unlocks the family: *what should a model do when its longest context has never seen this symbol?* It should not pretend; it should admit it and ask something less specific. Cleary and Witten's 1984 answer, Prediction by Partial Matching, made the admission a symbol in its own right. Every context's distribution contains, besides the symbols it has seen, an **escape** symbol meaning "not one of these; consult a shorter context". The coder codes the escape at its probability, drops to order $k-1$, and tries again; if that fails too, another escape, and so on down to order 0, and finally to an order $-1$ model that is uniform over the alphabet and cannot fail.

The cost of a symbol is then the *sum over the ladder*: every escape it took, plus the final hit. Watch it on the toy text. Code the first "s" of "thesis", at position 20, after the context "the". Nothing in the past has an "s" in it at all.

{{< fig src="img/context-models/escape_ladder.png" alt="The escape ladder for the first s of thesis: four escapes and a uniform fallback" caption="Coding the first 's' after 'the'. Order 3 has seen a space and an 'm' after 'the', never an 's': escape, 1.32 bits. Order 2 after 'he': the same, 1.32. Order 1 after 'e': 1.58. Order 0: 's' has never occurred at all, 1.95. Order -1: uniform over 27 symbols, 4.75. Total 10.93 bits for one genuinely new character." >}}

Nearly eleven bits, for a character that would have cost eight raw. That is the right price: the model had no evidence, and a model with no evidence should pay for its ignorance, but it should pay *once*, at the rung where the ignorance actually lives, rather than a little at every rung forever. The escape probability itself is the delicate part. In the figure it is the PPMC rule, $u/(n+u)$ where $u$ is the number of distinct symbols seen in the context and $n$ the total count: a context that has seen many different followers is more likely to see a new one. Compare that to Laplace. For the "e" after the final "th", PPMC's order-2 rung says $P(e) = 2/3$ (two sightings, one distinct symbol, so $2/(2+1)$), 0.58 bits, where Laplace said 2.63. Same counts, a different theory of the unseen, four and a half times fewer bits. Add **exclusion** (once order 3 has ruled out space and "m", the lower orders should not spend probability on them either) and you have PPMC, the compression-ratio champion of the early 1990s. Dmitry Shkarin's PPMd, inside 7-Zip and RAR, is its descendant and is still what you get when you ask either for "best text compression". The green curve in the figure is a PPM of this kind, and at order 4 it reaches 2.81 bits per character, under both xz and zlib on the same file, in forty lines of Python with no tuning.

The ladder also explains why PPM beats the LZ family on text and why it costs what it costs. LZ77 only sees exact repeats; a context model sees that after "th" comes "e" sixty percent of the time whether or not this particular "the" was ever followed by this particular word. The price is memory for the tables and strict symmetry: the decoder must rebuild every context and every escape probability from decoded history, so it does exactly the encoder's work, and neither side can skip ahead.

## Mixing: when two models disagree

PPM uses one context at a time: the longest that has anything to say. A different idea, and the one that owns the records, is to ask *all* of them at once and combine their answers. The order-5 model says the next bit is a 1 with probability 0.9; the order-1 model says 0.3; a word-level model says 0.6. What is the right combination?

Not the average. Averaging probabilities can never be more confident than the most confident model, and models that are right for different reasons should reinforce. The move that works, which Matt Mahoney's PAQ series settled on around 2005, is to combine in the **logistic domain**: stretch each probability to a log-odds, take a weighted sum, and squash back.

$$\mathrm{st}(p) = \ln\frac{p}{1-p}, \qquad p_{\text{mix}} = \mathrm{sq}\!\left(\sum_i w_i\, \mathrm{st}(p_i)\right), \qquad \mathrm{sq}(x) = \frac{1}{1+e^{-x}}.$$

```python
stretch = lambda p: np.log(p / (1 - p))
squash = lambda x: 1 / (1 + np.exp(-x))
p1, p2 = 0.9, 0.3            # two models' P(bit = 1)
for w in ((0.5, 0.5), (1.0, 1.0), (2.0, 0.2)):
    pm = squash(w[0] * stretch(p1) + w[1] * stretch(p2))
    print(f"weights {w}: mix = {pm:.3f}   (linear average would be {(w[0]*p1 + w[1]*p2)/sum(w):.3f})")
w = np.array([0.5, 0.5]); lr = 0.02
pm = squash(w @ [stretch(p1), stretch(p2)])
w += lr * (1 - pm) * np.array([stretch(p1), stretch(p2)])   # bit was 1: reward the model that said so
print(f"after one '1': weights -> {np.round(w, 3)}, mix -> {squash(w @ [stretch(p1), stretch(p2)]):.3f}")
# weights (0.5, 0.5): mix = 0.663   (linear average would be 0.600)
# weights (1.0, 1.0): mix = 0.794   (linear average would be 0.600)
# weights (2.0, 0.2): mix = 0.986   (linear average would be 0.845)
# after one '1': weights -> [0.515 0.494], mix -> 0.671
```

Two things in that listing. With weights that have learned to trust the first model, the mix is 0.986, more confident than either input, which a linear average cannot do. And the weight update is one line: $w_i \mathrel{+}= \eta\,(y - p_{\text{mix}})\,\mathrm{st}(p_i)$, the gradient of the coding cost $-\log_2 p_{\text{mix}}(y)$ with respect to $w_i$. The mixer is a single-layer neural network trained online to minimize bits, and the bits it minimizes are the bits in the file. PAQ8 runs hundreds of such models (orders 1 through 6, word contexts, sparse contexts that skip bytes, models specialized for images, executables, columns of tables) through a few layers of mixers selected by context, and feeds the result to a binary arithmetic coder. It runs at tens of kilobytes per second. Every byte updates hundreds of tables. It is the best lossless compression there is, and nobody uses it for anything but winning.

## The Hutter Prize and the rule that keeps it honest

What it wins is a prize. In 2006 Marcus Hutter offered money for compressing the first 100 MB of English Wikipedia, enwik8, and in 2020 raised the stakes to the first gigabyte, enwik9, with half a million euros on the table paid out in proportion to improvement. The PAQ lineage and its successor cmix have held the record throughout; the current entries land around 110 MB for the gigabyte, roughly 0.9 bits per character.

The prize has one rule that matters more than the number. **The size that counts is the compressed file plus the decompressor.** You could "compress" enwik9 to zero bytes by shipping a decompressor with enwik9 inside it. You could compress it very well indeed by shipping a decompressor with a 70-billion-parameter language model inside it. Neither counts, because the whole point of a compressor is that what the receiver already holds is part of the bill. This is the [side information](../side-information/) stage stated as a contest rule, and it is why the Hutter Prize is a prize for *learning*, in Hutter's own framing: a decompressor of size $S$ that reproduces the data from $C$ bytes has found $S + C$ bytes of description, and the shortest description is the best model of what Wikipedia is.

## Any predictor is a compressor

Now the step that makes this article the hinge of the modern map. The mixer in PAQ is a tiny neural network producing $P(\text{next bit} \mid \text{context})$. A transformer language model is a large neural network producing $P(\text{next token} \mid \text{context})$. There is no difference in kind. Attach an arithmetic coder to a 7B-parameter LLM, give it English, and it compresses to about 0.8 bits per character, below cmix, far below zstd's 2.5, with zero code written for compression. The model's cross-entropy on the text *is* the compressed size; that was the content of the [first article](../modeling-and-coding/), and here it is as a product.

The price is what the context mixing's algorithm card lists as weaknesses, and every one of them follows from the ladder and the mixer. The decoder must run the identical model, so decompression costs a forward pass per token: symmetric and slow. The model's state is the entire context: gigabytes of memory, no random access, no seeking into the middle of the stream. And the probabilities must be bit-exact on both machines, because an arithmetic coder that receives $P = 0.4999$ on one side and $0.5001$ on the other decodes garbage from that symbol onward; floating-point reductions on two different GPUs do not agree in the last bit, so practical LLM compressors quantize the model's probabilities to integers before they touch the coder.

{{< lab id="neural_prior" >}}

The same socket appears in miniature inside every video decoder. CABAC, the entropy stage of H.264 and HEVC, is a context model: each binary decision (is this block skipped? is this coefficient nonzero? is the motion vector's sign negative?) chooses one of several hundred probability states based on what the neighbouring blocks did, and the state adapts at a fixed rate after every decision, exactly like the counter listing above. It is order-1-ish, with "context" meaning spatial neighbours rather than previous bytes, and it is why CABAC saves roughly a tenth of the bitrate against the older Huffman-style CAVLC. And in learned image codecs, the "hyperprior" of Ballé and colleagues is a small network that predicts the distribution of each latent coefficient from a few side bits, feeding an arithmetic coder; the [learned transforms](../learned-transforms/) article has the rest of that pipeline.

{{< lab id="arith" >}}

## Connections

**n-gram language models** are order-$k$ context models under another name, and the problem of unseen n-grams is the escape problem. Kneser–Ney smoothing, the standard answer in speech recognition for twenty years, backs off from a trigram to a bigram to a unigram with absolute discounting and a continuation probability that is, in substance, a more careful estimate of $u/(n+u)$. PPM and Kneser–Ney were invented in different departments for the same reason and converged on the same shape.

**Prequential and MDL.** Coding a file adaptively, one symbol at a time with the model updated after each, produces a total of $\sum_t -\log_2 P_t(x_t \mid x_{<t})$. Dawid called that quantity the prequential (predictive-sequential) log-loss and proposed judging a statistical model by it; Rissanen's minimum description length principle says the same from the coding side. A context model's output length is a model-selection criterion you can read off a file size.

**Solomonoff induction**, in one sentence: weight every computable hypothesis by $2^{-(\text{its length in bits})}$, and predict by the weighted mixture; it is the uncomputable limit of a context mixer whose models are all possible programs, and the Hutter Prize is its budget-constrained cousin.

**In the game:** the [context mixing](/compressionfight/#fighter=cm) is this family, [learned probability model](/compressionfight/#fighter=neural_prior) its learned extreme, and the [arithmetic coding](/compressionfight/#fighter=arith) the partner neither can work without. They headline the Maximum-ratio text archive task, where ratio is everything and nobody is in a hurry.
