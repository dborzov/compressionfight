---
title: "Say it again: the LZ family"
slug: lz-family
chair: predict
family: "Dictionary matchers: the LZ family"
summary: "A back-reference is a prediction made with certainty; the window it points into can be the last 32 KB, a trained dictionary, or a file on another machine."
fighters: [lz77, lzw, dictionary]
related: [side-information, prefix-codes, sorting-transforms, codec-dissections]
reading_time: 14
---

# Say it again: the LZ family

*The data repeats itself. How do you say "the same as before" cheaply, and how far back is "before"?*

## The question

Haifa, 1977. Jacob Ziv and Abraham Lempel at the Technion are circling a problem that Shannon's theory answers only in principle. Shannon says the cost of a message is its surprise under the right probability model, and that the model is where all the cleverness lives (see [modeling and coding](../modeling-and-coding/)). Fine. But *which* model? A model of English is not a model of Fortran listings, and neither is a model of the accounting tapes a bank wants to shrink. Every probability model anyone had built was hand-fitted to one kind of data. Was there a compressor that needed no model at all, one that would learn whatever it was fed and still approach the entropy of the source?

The two of them noticed something that is obvious the moment you say it and was not obvious before. Real data does not merely have skewed letter frequencies. It *repeats itself*. A program uses the same identifier a hundred times. A log says `connection refused` on every third line. A JSON document says `"user_id":` once per record. If the next twenty bytes are a copy of twenty bytes you have already sent, you do not need a model of those bytes. You need a pointer.

That is the whole family in one sentence, and the rest of this article is about the two questions it leaves open: what, exactly, does a pointer cost, and what may it point into?

## A cat, a mat, and four pointers

Take the string `the_cat_sat_on_the_mat_the_cat_sat`, 34 bytes. Scan it left to right. At each position, look back through everything already emitted and ask: is the longest string starting *here* also sitting somewhere *back there*? If it is, and it is at least three bytes long, emit a pair (distance back, length) instead of the bytes. Otherwise emit the byte itself and move on by one.

```python
def lz77(s, window, minlen=3):
    i, out = 0, []
    while i < len(s):
        best = (0, 0)                                  # (distance, length)
        for d in range(1, min(i, window) + 1):         # every start in the window
            L = 0
            while i + L < len(s) and s[i + L - d] == s[i + L]:
                L += 1                                 # may run past the start: overlap is fine
            if L > best[1]: best = (d, L)
        if best[1] >= minlen: out.append(best); i += best[1]
        else:                  out.append(s[i]); i += 1
    return out

s = "the_cat_sat_on_the_mat_the_cat_sat"
toks = lz77(s, window=len(s))
print(toks)
lits = sum(isinstance(t, str) for t in toks)
print(len(s), "bytes ->", lits, "literals +", len(toks) - lits, "matches")
```
```
['t', 'h', 'e', '_', 'c', 'a', 't', '_', 's', (4, 3), 'o', 'n', '_', (15, 4), 'm', (11, 3), (23, 11)]
34 bytes -> 13 literals + 4 matches
```

Read the tokens against the string. The first eight bytes are new, so they go out as literals. At position 9 the parser sees `at_` and finds the same three bytes four positions back (the tail of `cat_`): token `(4,3)`. `on_` is new. At position 15, `the_` was seen at position 0: `(15,4)`. The `m` of `mat` is new, but `at_` is again three bytes that occurred eleven back: `(11,3)`. And then the big one: the last eleven bytes, `the_cat_sat`, are a verbatim copy of the first eleven, twenty-three positions back: `(23,11)`.

{{< fig src="img/lz-family/matches.png" alt="The toy string with the four back-references drawn as arrows from each copy to its source" caption="The greedy parse of the toy string. Thirteen bytes are literals; four pointers cover the remaining twenty-one." >}}

Four pointers cover 21 of the 34 bytes. If a pointer cost two bytes, the message would go out as 13 + 8 = 21 bytes, a saving of 13. Real formats do better on the pointers (we will see how), but the bookkeeping is already the right shape: *literals plus copies*.

The decoder has an easier job than the encoder. It never searches. It reads a literal and appends it, or reads `(d, L)` and copies `L` bytes from `d` positions back in what it has already written:

```python
def lz77_decode(toks):
    out = []
    for t in toks:
        if isinstance(t, str): out.append(t)
        else:
            d, L = t
            for _ in range(L): out.append(out[-d])   # byte by byte, so overlaps work
    return "".join(out)
print(lz77_decode(lz77("abababababab", 12)), lz77("abababababab", 12))
```
```
abababababab ['a', 'b', (2, 10)]
```

Notice the second example. The pointer `(2, 10)` says "go back two bytes and copy ten", which means copying bytes that are being written *during* the copy. That is legal, and it is how LZ77 does run-length encoding for free: `(1, 1000)` is a thousand repeats of the previous byte.

Now the proper names. The thing we just built is **LZ77**: the data already emitted is the **window** (or sliding window; the encoder may only point back at most $W$ bytes), a pointer is a **back-reference** or **match** $(d, L)$ with distance $d \le W$ and length $L$, and the sequence of literals and matches is the **parse**. The decoder's state is the last $W$ bytes of its own output, nothing more. There is no table, no tree, no model to transmit. The dictionary is the data itself.

## Aha 1: a pointer is a prediction with certainty

Look at what `(23, 11)` actually says. It says: *the next eleven bytes are exactly those eleven*. In the language of [modeling and coding](../modeling-and-coding/), a predictor hands the entropy coder a probability for each symbol and the coder charges $-\log_2 P$. A back-reference is the degenerate case where the predictor assigns $P = 1$ to each of the eleven bytes. Each of them then costs $-\log_2 1 = 0$ bits. The whole of `the_cat_sat`, 88 bits as literals, costs nothing. What you pay for is the *pointer*: the bits needed to say which prediction you are making, i.e. $d$ and $L$.

That reframing explains two things people find odd about LZ.

First, why it is in the Oracle's chair on the map and not the Packer's. LZ77 saves no bits by itself; it rewrites a stream of bytes into a stream of literals and pointers. In our toy, 13 literals and 4 pairs. Whether that is smaller than 34 bytes depends entirely on how the pairs are written down, and that is an entropy coder's job.

Second, why there is always an entropy coder *after* it. The token stream is not noise. Short matches are far more common than long ones; distances cluster (recent bytes are the most likely source of a copy); among literals, `e` and space still dominate. All of that is statistical skew, and a prefix code or an ANS coder eats it. DEFLATE, the format inside zip, gzip and PNG, codes its tokens with two Huffman trees per block: one over an alphabet that merges the 256 literals with the possible match lengths (3 to 258), and a second over distance buckets (1 to 32,768). A match in DEFLATE typically costs a dozen-odd bits for the pair, which is how `(23, 11)` turns 88 bits into about 15. zstd keeps the architecture and upgrades each part: literals through a fast Huffman, the (literal-length, match-length, offset) triples through FSE, a table-driven ANS; see [fractional bits](../arithmetic-and-ans/). LZ4 goes the other way and uses *no* entropy coder: byte-aligned tokens, pointers written as raw 16-bit offsets, a ratio around 2:1 instead of 3:1, and decoding at multiple gigabytes per second because decoding is nothing but `memcpy`.

That last word is the one to hold on to.

## Aha 2: two dials, and a decoder that does not care

Shrink the window and run the parser again.

```python
for w in (4, 3):
    toks = lz77(s, window=w)
    print(w, [t for t in toks if isinstance(t, tuple)])
```
```
4 [(4, 3)]
3 []
```

With a four-byte window the parser can only still see `at_` four bytes back; the three longer matches are out of reach and become literals. With three bytes it finds nothing at all, and the "compressor" emits the input unchanged. The window is **reach**: a repeat farther back than $W$ does not exist as far as the encoder is concerned. It is also **memory**: the decoder must hold the last $W$ bytes of output to resolve pointers. DEFLATE's $W = 32{,}768$ was chosen in 1990 for machines with a megabyte of RAM, and it is frozen in every gzip file ever written. LZMA (xz, 7z) lets the window run to hundreds of megabytes; zstd's `--long` mode reaches 128 MB and beyond, which is what lets it find the second copy of a file inside a tarball.

{{< fig src="img/lz-family/window.png" alt="Compression ratio rising from 4.6 to 7.2 as the LZ window grows from 4 KB to 4 MB on a synthetic log" caption="One codec (LZMA), one 3.7 MB synthetic log, the window swept. Each of the 3,000 line templates recurs about every 300 KB; the ratio climbs steeply once the window can reach that far and then saturates." >}}

The figure is a real measurement on a synthetic service log: 40,000 lines drawn from 3,000 templates, so a given template recurs roughly every 300 KB. Below a 32 KB window the codec only catches the short-range repeats (timestamps, `INFO`, `path=/api/`) and manages 4.8:1. By 1 MB every template is in reach and the ratio is 7.1:1. Nothing about the codec changed except how far back it was allowed to look.

The second dial is **parse effort**. Our toy parser is *greedy*: at each position it takes the longest match available and moves on. Greedy is not optimal. Taking a 3-byte match now can cost you an 8-byte match that would have started one byte later. *Lazy* matching (gzip levels 4 to 9) checks whether deferring by one byte gives a longer match before committing. *Optimal parsing* (xz, zstd at high levels, Brotli at level 11) treats the problem as what it is: a shortest-path problem over positions, where each literal and each possible match is an edge weighted by its coded cost, and the cheapest parse is the cheapest path from the first byte to the last. Formally, for a parse $\pi$ into tokens $t_1, \dots, t_m$,

$$\text{bits}(\pi) = \sum_{k=1}^{m} -\log_2 P(t_k), \qquad \pi^\star = \arg\min_\pi \text{bits}(\pi),$$

and the minimisation is dynamic programming over at most $n$ nodes, once you know the costs the entropy coder will charge. Expensive, because the costs depend on the token statistics, which depend on the parse. The encoder iterates.

Here is the property that made this family the substrate of everything that gets distributed. Both dials, window and parse effort, live on the encoder. The decoder's work per output byte is the same whether the encoder spent a microsecond or a minute choosing the tokens: read a token, copy some bytes. `gzip -1` and `gzip -9` produce files that decompress at the same speed. `zstd -1` through `zstd -19` change the match finder (hash chains, then binary trees, then optimal parsing) and nothing else. The compression-level slider that every user of these tools knows is purely an encoder-side knob, which is exactly what an *encode once, decode a billion times* job wants (see [the anatomy of a job](../anatomy-of-a-job/)). Spend hours in a datacenter finding the best parse; every phone decodes it with a loop of `memcpy`.

## The other flavour: writing the dictionary down

There is a second way to build a pointer, and in 1978 Lempel and Ziv published it too. Instead of pointing into the raw window, both sides maintain an explicit table of *phrases*, numbered as they are discovered. Terry Welch's 1984 variant, **LZW**, starts the table with the 256 single bytes and grows it by one entry per output code:

```python
def lzw(s):
    table = {chr(i): i for i in range(256)}
    w, out = "", []
    for c in s:
        if w + c in table: w += c
        else:
            out.append(table[w]); table[w + c] = len(table); w = c
    out.append(table[w])
    return out
print(lzw("ABABABA"))
```
```
[65, 66, 256, 258]
```

Walk it. `A` is in the table; `AB` is not, so emit 65 (`A`), add `AB` as entry 256, keep `B`. `BA` is not in the table: emit 66, add `BA` as 257, keep `A`. Now `AB` *is* in the table, so the parser extends to `ABA`, which is not: emit 256, add `ABA` as 258, keep `A`. Then `AB`, `ABA` are both known, the string ends, emit 258. Seven bytes became four codes, and the dictionary (`AB`, `BA`, `ABA`) was never transmitted; the decoder rebuilds it by replaying the same rule on the codes it receives.

LZW was simple, symmetric and fast, and it is the compressor inside GIF, TIFF and Unix `compress`. It is also the family's cautionary tale about where a dictionary lives. Welch's employer, Sperry (later Unisys), held a patent on it; in late 1994 Unisys announced it would collect royalties from software that wrote GIFs, and within a year a group of volunteers had designed PNG around LZ77 and DEFLATE precisely to route around the patent. Phil Katz's PKZIP had made DEFLATE the de facto standard in 1993, and he published the format so that anyone could implement it. The lineage you use today, zip, gzip, PNG, HTTP, git, zstd, Brotli, LZ4, is all LZ77. The explicit-table branch mostly died with its patent, and not only for legal reasons: the window *is* a dictionary, one that needs no table management and never has to be reset when it fills.

{{< lab id="lz77" >}}

{{< lab id="lzw" >}}

## The small-message problem

Imagine a message-queue team in 2016. Ten million events a second, each a 300-byte JSON blob, each one compressed and decompressed on its own because consumers read them one at a time. Someone switches on gzip per message and watches the dashboard. The savings are a fraction of what the same events get when they are bulk-compressed in a nightly file, and the smallest messages come out *larger* than they went in.

Our toy parser explains why without needing any further theory. On the first byte of a message, the window is empty. The parser cannot point at `"user_id":` because it has never seen it; the first occurrence of everything is a literal. A 300-byte message has room for at most a few short self-repeats. And then the container costs: a gzip header and trailer are 18 bytes, and a DEFLATE block that stores literals uncompressed adds its own few bytes of framing, so a message with nothing to match goes out heavier than it came in.

```python
import zlib, json, random
random.seed(1)
def event(i):                                           # a checkout event: fixed schema, random values
    return json.dumps({"timestamp": 1700000000 + 60 * i, "level": "info", "service": "checkout-api",
        "region": random.choice(["eu-west-1", "us-east-1", "ap-south-1"]), "user_id": "u_%06x" % random.randrange(16**6),
        "event_type": random.choice(["cart_add", "cart_remove", "checkout_start", "page_view"]),
        "sku": "SKU-%05d" % random.randrange(100000), "quantity": random.randrange(1, 5),
        "price_cents": random.randrange(100, 20000), "currency": "EUR", "session_id": "s_%08x" % random.randrange(16**8),
        "latency_ms": random.randrange(5, 500), "client": random.choice(["ios", "android", "web"])}).encode()
msg = event(0)
dic = (event(1) + event(2))[:600]                       # two earlier events, both sides keep them
def deflate(data, zdict=b""):
    co = zlib.compressobj(9, zlib.DEFLATED, -15, zdict=zdict) if zdict else zlib.compressobj(9, zlib.DEFLATED, -15)
    return co.compress(data) + co.flush()
print(len(msg), "raw ->", len(deflate(msg)), "alone,", len(deflate(msg, dic)), "with a 600-byte dictionary")
```
```
287 raw -> 191 alone, 62 with a 600-byte dictionary
```

The event has thirteen fields with random user ids, SKUs and prices, and the same schema every time. Alone, DEFLATE gets it from 287 bytes to 191: the field names repeat a little, the quotes and colons code cheaply, and that is all. Hand DEFLATE a *preset dictionary* of 600 bytes made of two earlier events, something zlib has supported since the 1990s, and the same message goes out as 62 bytes. Nothing about the algorithm changed. The window was simply non-empty when the first byte arrived: `"timestamp":` and `"service":"checkout-api"` and `"currency":"EUR"` were all already there to point at.

{{< fig src="img/lz-family/dictionary.png" alt="Compressed bytes per JSON message falling from 193 to 43 as the preset dictionary grows from nothing to 32 KB" caption="Fifty ~300-byte events, DEFLATE, mean compressed size against the size of the preset dictionary. The first 300 bytes of shared context do most of the work." >}}

Sweep the dictionary size and the curve is steep and then flat: the first few hundred bytes of shared context cut the message by two thirds, and 32 KB of context only shaves another third off *that*. The information that matters is the schema, and a schema is small. This is zstd's `--train`: pick the segments of a sample corpus that are most often matched against (zstd's COVER algorithm), pack them with pretrained entropy tables into a dictionary of around 100 KB, give it an ID, and have the frame header name the ID so the decoder knows which window to preload. Facebook built the feature for exactly this job. Brotli ships a fixed dictionary of about 120 KB of web English and HTML in the spec itself, which is why it beats gzip on small pages. The game's [PHRASEBOOK](/compressionfight/#fighter=dictionary) is this move, and [side information](../side-information/) is the article about what it costs to keep both sides holding the same 100 KB.

{{< lab id="dictionary" >}}

## Aha 3: the window can be anything both sides share

Step back and look at what the dictionary trick really did. It replaced "the window is the last $W$ bytes of this stream" with "the window is some bytes we both already have". Once you see the window as *any shared context*, three systems that look nothing like gzip turn out to be LZ77 with a different window.

**rsync.** You have a new version of a file; a remote machine has the old one. The receiver splits its old copy into blocks and sends back a weak rolling checksum and a strong hash per block. The sender slides the rolling checksum over the new file one byte at a time, and whenever it hits a block the receiver already holds, it sends "copy block 17" instead of the bytes. Literals plus pointers; the window is the old file on the other side of the network.

**Deduplicating backup.** A storage system holds terabytes; tonight's snapshot is mostly yesterday's. Split every stream into chunks, hash each chunk, store each distinct chunk once, and represent a file as a list of chunk hashes. Literals plus pointers again, where the window is the entire store and a pointer is a 32-byte hash. The sharp idea is *where to cut*. Fixed-size chunks fail the moment one byte is inserted, because every later boundary shifts by one and nothing after the edit matches:

```python
import numpy as np
rng = np.random.default_rng(0)
data = rng.integers(0, 256, 20000, dtype=np.uint8).tobytes()
edited = data[:7000] + b"X" + data[7000:]              # insert one byte
def fixed_chunks(b, size=1000):
    return {b[i:i + size] for i in range(0, len(b), size)}
def cdc_chunks(b, win=16, mask=(1 << 10) - 1):         # ~1 boundary per 1024 bytes
    out, start = set(), 0
    for i in range(win, len(b)):
        h = zlib.crc32(b[i - win:i]) & mask            # a stand-in for a rolling hash
        if h == 0: out.add(b[start:i]); start = i
    out.add(b[start:]); return out
for name, f in (("fixed", fixed_chunks), ("content-defined", cdc_chunks)):
    a, b2 = f(data), f(edited)
    print(f"{name:16s} chunks {len(a):3d}  unchanged after insert: {len(a & b2):3d}")
```
```
fixed            chunks  20  unchanged after insert:   7
content-defined  chunks  20  unchanged after insert:  19
```

Content-defined chunking cuts wherever a hash of the last few bytes hits a pattern, so boundaries are decided by the content, not by the byte count. Insert one byte and only the chunk containing it changes; the other nineteen still hash to the same values and cost nothing to store. Real systems use a true rolling hash (Rabin fingerprints, buzhash) that updates in $O(1)$ per byte, the same primitive rsync uses to find matches at arbitrary alignment.

**git packfiles.** Objects that look similar (the same path, similar size, so successive versions of one file) are sorted next to each other, each is delta-encoded as copies from and insertions relative to its neighbour, and the result is passed through zlib. Two layers of LZ: a pointer into another object, then pointers within the window.

Each of these is the Oracle's chair with the window pointed somewhere new, and each one keeps the asymmetry: finding the matches (hashing, searching, chunking) is the expensive side, and reassembling from pointers is a copy. The map's [side information](../side-information/) article treats the question these raise, of how both ends come to hold the same window, as the Inside Man's problem; here it is enough to see that it is the *same* algorithm.

## Connections

Match finding is a string-searching problem, and the data structures of that field are the engines inside every LZ encoder. zstd's fast levels use hash chains; its slow levels use binary search trees over the window; the suffix tree and the suffix array, which index every suffix of a text so that the longest previous occurrence of any string can be found in time proportional to its length, are the theoretical limit of the match finder. The same suffix array, sorted the same way, is the Burrows-Wheeler transform in another guise, which is why the [sorting transforms](../sorting-transforms/) article can promise context-model-grade ratios from a sort.

The output side connects the other way. The reason DEFLATE is "LZ77 then Huffman" and not "Huffman then LZ77" is the composition rule from the primer: the matcher manufactures a skewed token stream, and the [prefix code](../prefix-codes/) cashes it in. Reverse the order and Huffman would see flat byte frequencies in `abcabcabc` and the matcher would see noise.

And the window-as-shared-context view connects to the whole Inside Man column. A trained zstd dictionary, Brotli's built-in web dictionary, the receiver's old file in rsync, the chunk store in a backup system: these are points on the primer's spectrum of *where the prior lives*, from "learned from the stream" to "installed on both ends". LZ77 with an empty window is the adaptive extreme; LZ77 with a 100 KB trained window is a shared prior; and the fact that the same decoder loop serves both is the reason the dictionary feature was a few hundred lines in zstd rather than a new codec.

**In the game:** [ECHO](/compressionfight/#fighter=lz77), [SCRIBE](/compressionfight/#fighter=lzw) and [PHRASEBOOK](/compressionfight/#fighter=dictionary) sit in the Oracle's and the Inside Man's chairs of The Archive Job (DEFLATE), Pixel Perfect (PNG), The Log Pipe (zstd with a trained dictionary) and The Cold Storage (KVTC's DEFLATE back end); see the [codec dissections](../codec-dissections/) for each squad in full.
