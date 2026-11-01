---
title: "Sorting transforms: the Burrows-Wheeler idea"
slug: sorting-transforms
chair: transform
family: "Sorting transforms"
summary: "The Burrows-Wheeler transform is a reversible permutation that groups letters by the context that follows them, turning context into locality, and the same sorted table is a search index."
fighters: [bwt, rle]
related: [lz-family, context-models, prefix-codes]
reading_time: 14
---

# The shuffler: sorting transforms

*How can sorting a block of text, a reversible permutation, make it compress like a context model?*

## The question

Palo Alto, 1994. Michael Burrows and David Wheeler at DEC's Systems Research Center publish a technical report with a modest title, "A Block-sorting Lossless Data Compression Algorithm", describing something Wheeler had found years earlier and nobody had quite believed was useful. The claim was strange. Take a block of text. Permute its characters, in a particular reversible way. Do nothing else clever. Then feed the permuted block to the cheapest, dumbest back end you have, and get compression ratios close to the best context models of the day, which were slow, memory-hungry adaptive beasts.

Here is why that should sound impossible. An entropy coder charges by the histogram. If the block contains 500 `e`s and 20 `z`s, a Huffman or arithmetic coder working from those counts charges a fixed price per `e` and per `z`, and *a permutation does not change the counts*. Shuffle the letters any way you like; the order-0 entropy, the bits-per-symbol floor for a coder that sees only frequencies, is identical before and after. On a 75 KB sample of this site's own documentation, the raw text has an order-0 entropy of 4.868 bits per character, and after the Burrows-Wheeler transform it has an order-0 entropy of 4.868 bits per character. By the only number the entropy-coding stage can see, nothing happened.

So what did?

## Seven rotations of a banana

Take `banana$`, where `$` is an end marker that sorts before every letter. Write down all seven rotations of the string, sort them, and read off the *last* character of each row:

```python
def bwt(s):
    rots = sorted(s[i:] + s[:i] for i in range(len(s)))   # every rotation, sorted
    return "".join(r[-1] for r in rots), rots.index(s)     # last column, and where the original landed
L, idx = bwt("banana$")
print(L, idx)
```
```
annb$aa 4
```

{{< fig src="img/sorting-transforms/rotations.png" alt="The sorted rotation tables of banana$ and abracadabra$ with the first column F and last column L highlighted and the original string's row marked" caption="Every rotation, sorted. The first column F is the sorted characters; the last column L is the transform. Row 4 of the banana table is the original string, and that row number is the primary index." >}}

Look at the table. The sorted first column, call it $F$, is just the characters of `banana$` in order: `$aaabnn`. The last column, $L$, is `annb$aa`. Both have the same letters, as a permutation must. But the two `n`s in $L$ are adjacent, and three of the four `a`s are too. The transform output is `a`, then `nn`, then `b$`, then `aa`: four runs where the input `banana$` had seven.

Why did the `n`s land together? Read along their rows. Row 2 is `ana$ban` and row 3 is `anana$b`. Both rotations *begin* with `an`, because that is what the sort put next to each other, and each one's last character is the character that *precedes* that beginning in the original string. Both `n`s in `banana` are followed by `a`, so when the rotations are sorted by what follows, those two `n`s are neighbours. Rows 5 and 6 begin with `na`; their last characters are the letters before each `na`, which are `a` and `a`. Rows 1 to 3 begin with `a`; their last characters are what precedes each `a`: `n`, `n`, `b`.

That is the whole trick, and it is worth saying it in the general case before going on. The **Burrows-Wheeler transform** of a string $s$ of length $n$ is the last column $L$ of the lexicographically sorted matrix of its $n$ rotations, together with the **primary index**, the row at which the original string appears. Row $i$ of the matrix begins with some context $c_i$ (the rotation's prefix) and ends with the character $L[i]$ that precedes that context in $s$. So

$$L[i] = s\big[(\mathrm{SA}[i] - 1) \bmod n\big],$$

where $\mathrm{SA}$ is the **suffix array**, the starting positions of the rotations in sorted order. $L$ is the string's characters *sorted by the context that follows each of them*.

## Aha 1: context becomes locality

Now the question from the opening has an answer. A [context model](../context-models/) like PPM gets its ratio by conditioning: after `th`, `e` is very likely, so `e` in that context costs a fraction of a bit. It pays for that with a table of statistics per context, updated adaptively, and an arithmetic coder that is handed a different distribution for every symbol.

The BWT does the conditioning by *sorting*. Every character that is followed by `he_` in a block of English lands in one contiguous stretch of $L$, and that stretch is nearly all `t`. Every character followed by `ing_` lands in another stretch, mostly `r`, `k`, `d`, `t`. Every character followed by `u` is mostly `q` and `o`. The transform has not changed what the frequencies are over the block. It has arranged for the frequencies to be *wildly different from one neighbourhood of the output to the next*. The information a context model keeps in a table, the BWT keeps in the *position* of each character.

This is the aha moment of the family and it is worth sitting with: context correlation, a kind of redundancy that an order-0 coder cannot see at all, has been converted into **locality**, which is a kind of redundancy that is easy to see but that an order-0 coder *also* cannot use directly, because it still sums over the whole block. The entropy is 4.868 bits before and 4.868 bits after. A reversible permutation cannot lower it. What it can do is set up the next stage.

{{< fig src="img/sorting-transforms/runs.png" alt="Left: histogram of run lengths in 75 KB of Markdown before and after the BWT, runs falling from 73,513 to 36,824. Right: bits per character after BWT and move-to-front falling from 4.53 to 3.06 as the block size grows from 1 KB to 64 KB" caption="Left: the same 75 KB of text has 73,513 runs raw and 36,824 after the BWT, with runs of forty or more appearing where there were none. Right: order-0 entropy after BWT and move-to-front, against block size; the dashed line is the raw text's 4.87 bits." >}}

The left panel is the measurement on the 75 KB sample: the raw text has 73,513 runs of identical characters, the longest being 85 (a line of dashes); the transformed text has 36,824, the longest being 259. Nothing was added or removed. The characters were lined up by context, and contexts in English have strong opinions about what precedes them.

{{< lab id="bwt" >}}

{{< viz src="viz/sorting-transforms-bwt.html" height="520" >}}

## Three links deep: MTF, runs, and only then a coder

A back end that can cash in locality has to turn "the symbol I just saw is likely to recur soon" into "small numbers are common". That is **move-to-front**: keep the alphabet in a list, output each symbol's current position in the list, then move that symbol to the front.

```python
def mtf(s, alphabet):
    table, out = list(alphabet), []
    for c in s:
        i = table.index(c); out.append(i)
        table.insert(0, table.pop(i))                          # move it to the front
    return out
print(mtf("annb$aa", "$abn"))
print(mtf("ard$rcaaaabb", "$abcdr"))
print(mtf("aaaaaabbbbbbaaaaaa", "ab"))
```
```
[1, 3, 0, 3, 3, 3, 0]
[1, 5, 5, 3, 2, 5, 4, 0, 0, 0, 5, 0]
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
```

On seven characters of banana there is little to show: a run of two `n`s becomes `3, 0`, and the rest is noise. On `abracadabra$`, whose transform is `ard$rcaaaabb`, the run of four `a`s becomes `4, 0, 0, 0`, and the final `bb` becomes `5, 0`. The third line is the point: a long run of anything becomes a run of zeros after its first symbol, and a block that alternates between a few symbols in long stretches becomes almost entirely zeros no matter *which* symbols they were. On the 75 KB sample, 52 percent of the MTF output is zero, and the order-0 entropy of the MTF output is 3.05 bits per character against the 4.87 of the raw text. *Now* the frequencies are skewed, and a plain Huffman coder can see it.

Between MTF and the coder, one more stage: **run-length encoding** of the zeros, because a run of 200 zeros is 200 symbols at a bit or so each to a Huffman coder and a single (run, length) token to RLE. bzip2 codes zero runs with a pair of special symbols that spell the run length in a bijective base-2, so the alphabet the Huffman stage sees is "a run of zeros of some length" plus the nonzero MTF indices.

This is the composition rule of the primer, three links deep, and it is the second aha of the family. The BWT converts context into locality; MTF converts locality into small numbers; RLE converts runs of small numbers into a handful of tokens; and only then does an entropy coder see the global skew it needs. Each stage eats what the previous one produced and emits what the next one wants. Remove any one and the chain breaks: BWT then Huffman compresses nothing, because the histogram is unchanged; MTF without BWT does little on raw text, because raw text has only weak locality; RLE on raw text finds almost no runs. It is a pipeline of matched interfaces, not a bag of tricks.

{{< lab id="rle" >}}

## Undoing it with one index

It would all be useless if the sort threw information away, and a sceptic should ask how `annb$aa` plus the number 4 can possibly be enough to get `banana$` back. The brute-force argument is short enough to run. We know $L$. We know $F$, because $F$ is $L$ sorted. Write each $L[i]$ in front of each row, sort the rows, and you have the sorted list of all *two*-character prefixes of the rotations. Repeat $n$ times and you have rebuilt the whole matrix:

```python
def ibwt(L, idx):
    rows = [""] * len(L)
    for _ in range(len(L)):
        rows = sorted(L[i] + rows[i] for i in range(len(L)))   # prepend L, re-sort, repeat
    return rows[idx]
print(ibwt("annb$aa", 4))
```
```
banana$
```

That is $O(n^2 \log n)$ and nobody does it, but it proves the point: $L$ determines the matrix, and the primary index picks the row. The real inverse is linear time and rests on one observation, the **LF mapping**. Take the $k$-th occurrence of some character `a` counting down column $L$, and the $k$-th occurrence of `a` counting down column $F$. They are the *same character of the original string*. The reason: the `a`s in $L$ appear in the order of their following contexts (because rows are sorted by what follows), and the `a`s in $F$ appear sorted by what follows *them*, which is the same ordering. So the rank of a character within its letter is preserved between the two columns, and "the row in $F$ where this $L$-character lives" is a lookup. Hop from $L[i]$ to its row in $F$, read that row's $L$, hop again; each hop moves one character backwards through the string.

```python
def ibwt_lf(L, idx):
    order = sorted(range(len(L)), key=lambda i: (L[i], i))     # L's characters in F order, ties by position
    LF = {i: f for f, i in enumerate(order)}                   # row i of L -> row of the same character in F
    out, i = [], idx
    for _ in range(len(L)):
        out.append(L[i]); i = LF[i]
    return "".join(reversed(out))
print(ibwt_lf("annb$aa", 4), ibwt_lf("ard$rcaaaabb", 3))
```
```
banana$ abracadabra$
```

Seven hops for banana, twelve for abracadabra, one array of $n$ integers, and the original is back. Encoding needs a sort of $n$ rotations (a suffix array, built in $O(n)$ or $O(n \log n)$ with the right algorithm, and in practice the most expensive part of bzip2); decoding needs one counting pass and $n$ hops. The transform is a little asymmetric and the asymmetry points the wrong way for an archive, which is one of the reasons the LZ family won the general-purpose war.

## The dials: block size, and the one thing you cannot have

Everything above operates on a **block**. The sort is over the rotations of a block, the LF mapping is within a block, and a character's context is only visible to the sort if the context is in the same block. So the block size is a dial, and the right panel of the figure above turns it: on the sample, 1 KB blocks leave 4.53 bits per character after BWT and MTF, 8 KB blocks 3.72, 64 KB blocks 3.06. Bigger blocks mean more characters sharing each context and longer, purer runs. bzip2's block size is its only user-facing parameter, 100 KB to 900 KB (the `-1` to `-9` flags), and the memory cost of the sort is several bytes per input byte on top of the block itself, so a 900 KB block wants a few megabytes of RAM, a lot in 1996 and nothing now.

What the dial cannot buy is streaming or random access. The first byte of the output cannot be emitted until the last byte of the block has been read and sorted, so latency is a whole block; and no byte of a block can be decoded without the LF walk from the primary index through every byte before it. Every block is one indivisible unit, which puts the BWT on the opposite side of the seek dial from the [motion compensation](../motion-compensation/) article's GOP, where the whole design is about how often to cut the chain. It is the right shape for a file compressed once and read whole, and the wrong shape for a log that is appended to, a socket, or a column a query wants to skip into.

## bzip2: a transform in a lossless codec

Imagine Julian Seward in 1996, with the DEC report on his desk and a problem that is not technical. He has a BWT compressor that works and that uses an arithmetic coder as its back end, which is the natural choice, since an adaptive arithmetic coder can follow the shifting local statistics of the transformed block almost perfectly. But in 1996 arithmetic coding is covered by patents held by IBM and others, and a free compressor that infringes a patent is not free. JPEG's arithmetic mode exists in the standard and is used by nobody for exactly this reason. So he takes the arithmetic coder out and replaces it with Huffman, the one coder everyone agrees is unencumbered, accepting a small loss of ratio to ship something anyone can use; that trade is the documented reason bzip2 uses Huffman codes, and the scene around it is imagined. The result is the stack the primer calls canonical: an initial run-length pass to tame worst-case inputs for the sort, the BWT over blocks of up to 900 KB, move-to-front, run-length coding of zeros, and Huffman with several tables switched every fifty symbols so that the coder can still follow the local statistics a little. It beat gzip's ratio handily on text, cost several times the CPU, and sat in every Unix distribution for twenty years until xz and zstd displaced it.

The point to carry away is in the map's name for the family: a *transform* in a lossless codec. The transform stage is usually described as the setup move for a quantizer, a DCT that piles energy into a few coefficients so the rest can be discarded. The BWT is in the same stage for the same reason, a rearrangement that saves nothing itself and makes the next stages' job possible, with no quantizer anywhere in the pipeline. The transform's job is to change the *shape* of the redundancy, and shape is what the downstream stages are sensitive to.

## Aha 3: the same table is an index

There is a third thing the sorted matrix is, and it is the reason the BWT appears in the "used in" line of the game's [Burrows-Wheeler (BWT)](/compressionfight/#fighter=bwt) alongside genome aligners. The LF mapping walks backwards through the string one character at a time. Run it on a *pattern* instead of on the whole text and it counts the pattern's occurrences.

```python
def fm_count(L, pattern):
    alphabet = sorted(set(L))
    C = {c: sum(x < c for x in L) for c in alphabet}           # rows of F where c starts
    occ = lambda c, i: L[:i].count(c)                          # how many c in L before row i
    lo, hi = 0, len(L)
    for c in reversed(pattern):                                # read the pattern backwards
        if c not in C: return 0
        lo, hi = C[c] + occ(c, lo), C[c] + occ(c, hi)
        if lo >= hi: return 0
    return hi - lo
L, _ = bwt("banana$")
print([(p, fm_count(L, p)) for p in ("ana", "an", "nan", "ban", "nab")])
```
```
[('ana', 2), ('an', 2), ('nan', 1), ('ban', 1), ('nab', 0)]
```

Read the pattern from its last character. The rows of the sorted matrix that begin with `a` are a contiguous range, $[C(\text{a}), C(\text{b}))$. Among those rows, the ones whose *preceding* character is `n` are the ones whose $L$ entry is `n`; by the LF property, those map to a contiguous range of the rows beginning with `na`. Among those, the ones preceded by `a` map to the rows beginning with `ana`. Each step shrinks the range by one character of context and costs one rank query, "how many `n`s in $L$ before row $i$", so the number of steps is the length of the pattern, not the length of the text. `ana` occurs twice in `banana` (overlapping, at positions 1 and 3), and the range after three steps has width 2.

This is the **FM-index** of Ferragina and Manzini, 2000: store $L$ in a form that answers rank queries quickly (a wavelet tree or sampled counts), plus the small $C$ table, and you have a structure that both *is* the compressed text and answers "how many times, and where, does this pattern occur" in time proportional to the pattern. A human genome is three billion characters over a four-letter alphabet; its FM-index fits in a few gigabytes, and the aligners that map sequencing reads to it, BWA and Bowtie from 2009 onwards, locate each read by backward search. Every genome sequenced in the last fifteen years has passed through a Burrows-Wheeler transform, not to compress it, but to search it. The same sort, the same column, the same LF hop.

## Connections

The suffix array is the BWT with the last step left off, and it is also the limit object of an LZ match finder: sorted suffixes let you find the longest previous occurrence of any string in time proportional to its length, which is what the [LZ family](../lz-family/)'s encoders are approximating with hash chains and binary trees. The two families, so different at the output, share their most expensive data structure.

The relation to PPM is the one to keep. PPM conditions on the preceding $k$ characters and stores statistics in a table; the BWT groups by the *following* characters, in the position of the output, with no table. Transforming a block and running an adaptive order-0 coder over it approximates a context model of unbounded order whose statistics are local rather than global, which is why BWT-based compressors sit between gzip and PPM on the ratio axis and why the best of them (bsc, the BWT mode of some research compressors) pair the transform with an adaptive arithmetic coder rather than with bzip2's Huffman tables. [Context models](../context-models/) is the article about what the table-based approach buys for the extra cost.

And the FM-index is a reminder that a compressed representation can be a *better* representation of the data than the raw one. A sorted, indexed, reversible permutation of a genome answers questions the genome itself cannot answer without a scan. That theme, a compressed form that serves computation directly, returns in the Billion-scale vector search task with product quantization, where distances are computed on the codes.

**In the game:** [Burrows-Wheeler (BWT)](/compressionfight/#fighter=bwt) and [Run-length coding](/compressionfight/#fighter=rle) are accepted in the Maximum-ratio text archive task (bzip2 against xz and PAQ), where there is no latency pressure and a patient decoder; Run-length coding also serves in the Photographs task, running JPEG's zigzag zeros ahead of Huffman. The [codec dissection](../codec-dissections/) of bzip2 shows the five-stage stack stage by stage.
