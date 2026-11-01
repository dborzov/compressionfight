---
title: "Channel and layout transforms: group like statistics"
slug: channel-and-layout
chair: transform
family: "Channel and layout transforms"
summary: "RGB to YCbCr, stereo to mid/side, byte shuffling and columnar storage are one principle at three scales: put bytes that share statistics next to each other and separate the ones that do not, so the coders downstream see clean distributions instead of interleaved mush."
fighters: [prism]
related: [block-transforms, perceptual-models, lz-family, codec-dissections]
reading_time: 13
---

## The colour that had to hide

Imagine an engineer at a television manufacturer in 1952, a year or so before the American colour standard is settled, and imagine the constraint sheet on his desk. Colour television must be broadcast on the same channels, in the same bandwidth, as the black-and-white service, and every black-and-white set already in a living room must keep showing a correct picture from the colour signal, without modification. Fifteen million of them. He cannot send three pictures, red, green and blue, in the space of one. And he cannot send red, green and blue at all, because the old sets would not know what to do with any of them.

The first answer anyone proposes is to send brightness, which is what the old sets want, and then somehow squeeze the colour in beside it. The question that stops the room is what "the colour" is. If you have sent brightness, how much of red, green and blue is left to send? The camera produces three full signals. Intuition says you have sent one of three and two remain. The engineer who answers that question correctly, and the NTSC committee did, has discovered the subject of this article. The same discovery was made again by the people who designed stereo audio broadcasting, again by whoever first compressed a float array and wondered why it would not shrink, and again by the database engineers who built columnar stores. Each time the question is the same. *Why does storing the same numbers in a different order make them compress several times better?* The primer quotes a factor of 5 to 10 for a table in Parquet against the same table as gzipped CSV. Nothing about the numbers changed. Only where they sit.

## Four pixels

Here are four pixels from a patch of skin going into shadow, as a camera produces them, three bytes each:

```python
import numpy as np
px = np.array([[200, 192, 188], [172, 166, 160], [141, 137, 134], [112, 106, 104]], float)
R, G, B = px.T
Y  = 0.299*R + 0.587*G + 0.114*B
Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
for name, ch in [("R", R), ("G", G), ("B", B), ("Y", Y), ("Cb", Cb), ("Cr", Cr)]:
    print(f"{name:2s} {ch.round(1)}   range {ch.max()-ch.min():5.1f}")
```
```
R  [200. 172. 141. 112.]   range  88.0
G  [192. 166. 137. 106.]   range  86.0
B  [188. 160. 134. 104.]   range  84.0
Y  [193.9 167.1 137.9 107.6]   range  86.4
Cb [124.7 124.  125.8 126. ]   range   2.0
Cr [132.3 131.5 130.2 131.2]   range   2.1
```

Look at the red, green and blue rows. They are three nearly identical sequences. Each one goes from about 200 down to about 105 in the same four steps, and each one would cost a coder about the same number of bits, say seven bits per sample for a range of 88. Three channels, twenty-one bits per pixel, and most of those bits are saying the same thing three times: *this patch is getting darker*. The colour, the fact that this is skin and not grey, is the tiny difference between the rows, R a little above G a little above B, and it is drowned in the brightness.

The three lines under them are JPEG's **YCbCr** transform, the same mixing that the television committee arrived at. $Y$ is a weighted average of the three, with the weights 0.299, 0.587 and 0.114 chosen to match how bright each primary looks to the eye, and it carries the darkening: range 86, the same as before. $C_b$ and $C_r$ are the differences of blue and red from that brightness, offset by 128 so they can be stored as bytes, and their ranges are 2.0 and 2.1. The brightness was stored three times; now it is stored once, and the two leftover channels say "a bit warm" in about one bit each. Nothing was discarded. The transform is a 3×3 matrix with an inverse, and the pixels come back exactly in floating point.

{{< fig src="img/channel-and-layout/planes.png" alt="The R, G, B planes and the Y, Cb, Cr planes of a drawn portrait as grey images, with a bar chart of each plane's standard deviation" caption="The same move on a whole 256×256 picture. The three RGB planes are three near-copies of the brightness, each with a standard deviation around 60 to 70; after the transform Y keeps it and Cb and Cr fall to 26 and 12." >}}

This is the first aha, and it answers the 1952 question. *R, G and B are three copies of brightness plus a little colour.* Send brightness once, and what is left to send is not two more pictures but two small corrections, which is exactly how much bandwidth was spare. The black-and-white sets take $Y$ and ignore the rest, which was the whole point.

{{< lab id="prism" >}}

For a coder the win is a matter of distributions. Three channels of standard deviation 60 to 70 cost about $3 \times \tfrac12 \log_2 \sigma^2$ bits per pixel at a fixed error; one channel of 59 and two of 26 and 12 cost noticeably less, because $\log_2$ of a small variance is small. It is the same arithmetic as the [transform coding gain](../block-transforms/) of the DCT, applied across the colour axis instead of the spatial one, and for the same reason: the geometric mean of the variances dropped while the arithmetic mean stayed about where it was. On its own the saving is modest, the fighter card calls it "a tiny win", and the real reason every image and video codec starts with this transform is what it makes possible next. Once colour sits in two channels of its own, those channels can be treated differently from brightness, and the eye, which resolves colour at about half the resolution it resolves brightness, does not mind if they are stored at half the resolution. That is chroma subsampling, the cheapest 2:1 in all of image coding, and it belongs to the [perceptual](../perceptual-models/) chair, which decides what the viewer will not miss. The Lens only made the two channels separable.

{{< lab id="chroma" >}}

## Integers in, integers out

The transform above has a catch for lossless work. The coefficients are irrational, so $Y$ comes out as 193.9 rather than an integer, and rounding it to a byte loses a fraction of a bit of information that the inverse cannot recover. PNG, JPEG XL and FLAC all need the same decorrelation *without* the loss, and the trick that gives it is called lifting: build the transform as a sequence of steps, each of which adds a rounded function of some channels to another channel, and each step is reversible by subtracting the same rounded function.

```python
import numpy as np
def ycocg_r(r, g, b):                      # lifting steps: integer in, integer out
    co = r - b;  t = b + (co >> 1)
    cg = g - t;  y = t + (cg >> 1)
    return y, co, cg
def ycocg_r_inv(y, co, cg):
    t = y - (cg >> 1);  g = cg + t
    b = t - (co >> 1);  r = b + co
    return r, g, b
print("YCoCg-R(200,192,188) =", ycocg_r(200, 192, 188), "->", ycocg_r_inv(*ycocg_r(200, 192, 188)))

L, R = 1000, 980                            # stereo sample pair
mid, side = (L + R) >> 1, L - R             # FLAC: mid keeps the floor, side keeps the parity
total = 2*mid + (side & 1)                  # the lost bit of L+R is the parity of L-R
print("mid/side =", (mid, side), " back to", ((total + side) >> 1, (total - side) >> 1))
```
```
YCoCg-R(200,192,188) = (193, 12, -2) -> (200, 192, 188)
mid/side = (990, 20)  back to (1000, 980)
```

**YCoCg-R** is a luma and two chroma channels, "orange" and "green", built from four lifting steps using only shifts and adds. The halving `>> 1` throws away a bit each time, and the inverse recovers it because the inverse undoes the steps in reverse order with the *same* truncated values available to it. Three bytes went in as (200, 192, 188), became (193, 12, −2), and come back as (200, 192, 188): the transform is a bijection on integers.

The second half is stereo. The left and right channels of a recording are two microphones a few centimetres apart hearing the same room, so like R and G they are near-copies: 1000 and 980. **Mid/side** stores the average and the difference, 990 and 20, and the difference is small and quiet wherever the stereo image is narrow, which is most of the time. FLAC's version keeps `side = L − R` in full and `mid = (L + R) >> 1`, dropping the lowest bit of the sum, which is safe because the sum and the difference of two integers always have the same parity: the dropped bit of the sum is the lowest bit of the side channel, so the decoder reads it back from there. Twenty is cheaper to code than 980; the [linear predictor](../linear-prediction/) and the Rice coder that follow in FLAC see one loud channel and one quiet one instead of two loud ones.

These two examples are the second aha in its narrowest form. **Mix the channels so one carries the shared part and the others carry the small differences.** The shared part was being paid for once per channel; now it is paid for once.

## The same idea, one level down: inside a float

Now a case where nobody is mixing anything, and the principle still applies. A sensor logs a slowly drifting voltage as 32-bit floats. The values are nearly identical, differing in the fourth decimal, and yet a general-purpose compressor barely touches the file. Here are eight of them as bytes:

```python
import numpy as np
vals = np.array([1.9950, 1.9975, 2.0000, 2.0025, 2.0050, 2.0075, 2.0100, 2.0125], np.float32)
by = vals.view(np.uint8).reshape(-1, 4)[:, ::-1]      # columns: byte3 byte2 byte1 byte0
for v, b in zip(vals, by): print(f"{v:.4f}  " + " ".join(f"{x:02x}" for x in b))

def H(a):                                              # zeroth-order entropy in bits/symbol
    _, c = np.unique(a, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum()
t = np.linspace(0, 1, 4096, dtype=np.float32)          # 4096 samples of a slow sine near 2.0
sig = (2.0 + 0.5*np.sin(2*np.pi*3*t) + 0.002*np.random.default_rng(0).standard_normal(4096)).astype(np.float32)
planes = sig.view(np.uint8).reshape(-1, 4)[:, ::-1]
print("entropy per byte plane (b3..b0):", [round(float(H(planes[:, j])), 2) for j in range(4)])
print("entropy of the interleaved bytes :", round(float(H(sig.view(np.uint8))), 2))
import zlib
print("zlib bytes, interleaved:", len(zlib.compress(sig.tobytes(), 9)),
      " shuffled:", len(zlib.compress(np.ascontiguousarray(planes.T).tobytes(), 9)), " raw:", sig.nbytes)
```
```
1.9950  3f ff 5c 29
1.9975  3f ff ae 14
2.0000  40 00 00 00
2.0025  40 00 28 f6
2.0050  40 00 51 ec
2.0075  40 00 7a e1
2.0100  40 00 a3 d7
2.0125  40 00 cc cd
```
```
entropy per byte plane (b3..b0): [1.0, 6.29, 7.95, 7.96]
entropy of the interleaved bytes : 6.87
zlib bytes, interleaved: 14102  shuffled: 9817  raw: 16384
```

Read the hex columns. The top byte, sign and most of the exponent, is `3f` or `40` for every value: the numbers are all near 2, so the exponent is one of two things. The next byte holds the last exponent bit and the top seven mantissa bits, and it is `ff` or `00`. The bottom two bytes are the fine mantissa, and they are different for every value, with no pattern a coder could use; they are, in effect, noise. A float32 is a struct of four fields with four completely different statistics, and in memory the fields of consecutive floats are interleaved: constant, nearly constant, noise, noise, constant, nearly constant, noise, noise. A byte-oriented compressor walking that stream sees an entropy of 6.87 bits per byte and manages 14102 bytes out of 16384, a saving of 14%.

The fix is called **byte shuffle**, and Blosc and HDF5 apply it before every general-purpose compressor for exactly this reason. Transpose the array: all the top bytes first, then all the second bytes, then the third, then the fourth. The numbers are untouched, and the inverse is another transpose. On 4096 samples of a slow sine near 2.0, the entropies of the four planes come out as 1.0, 6.3, 7.95 and 7.96 bits per byte: the exponent plane is a one-bit decision, the low mantissa planes are incompressible, and now they are in separate rooms. The same zlib on the shuffled bytes gives 9817, a saving of 40% instead of 14%, and almost all of it comes from the exponent plane, which collapses to a few dozen bytes of run lengths while the mantissa planes stay where they were. Shuffling did not make the mantissa compressible. It stopped the mantissa from poisoning the exponent.

{{< fig src="img/channel-and-layout/byteplanes.png" alt="A slow sine of 4096 float32 samples, its four byte planes shown as rows, and a bar chart of the entropy of each plane against the interleaved stream" caption="Byte shuffle on a float32 array. The top byte plane is almost constant and the two low mantissa planes are noise; interleaved, the mixture has an entropy of 6.87 bits per byte, separated, the exponent plane costs 1 bit and the noise stays where it was." >}}

## The same idea, one level up: a table

The last scale is a database table. Six rows of a web log, five columns: a date, a method, a status, a path, a latency. Stored as text, row by row, the byte stream reads a date, then a word, then a number, then a path, then a small integer, then the next date, and the compressor at the end of the pipe sees a mixture of five languages. Stored column by column, each column is a stream in one language, and the statistics of each are simple enough to count by hand: the date column has 1 distinct value in 6, method and status and path have 2 or 3, and only the latency column has 6.

{{< fig src="img/channel-and-layout/layout.png" alt="The same six-row, five-column log laid out row-wise and column-wise, with distinct-value counts per column" caption="Same thirty values. Row-wise, consecutive bytes belong to different columns with different distributions. Column-wise, each column is homogeneous: one has a single value, three have two or three, and each can be given the coder that fits it." >}}

Scale that up to twenty thousand rows and measure. Here the measurement surprises in a useful way:

```python
import numpy as np, zlib
rng = np.random.default_rng(0); n = 20000
cols = {"date":   np.full(n, "2024-03-01"),
        "method": rng.choice(["GET", "GET", "GET", "POST", "PUT"], n),
        "status": rng.choice(["200", "200", "200", "201", "404", "500"], n),
        "path":   rng.choice(["/api/users", "/api/users", "/api/orders", "/api/items", "/api/login"], n),
        "ms":     rng.integers(1, 60, n).astype(str)}
H = lambda a: (lambda p: float(-(p*np.log2(p)).sum()))(np.unique(a, return_counts=True)[1] / len(a))
row_wise = "\n".join(",".join(r) for r in zip(*cols.values())).encode()
col_wise = b"\n".join("\n".join(c).encode() for c in cols.values())
z = lambda b: len(zlib.compress(b, 9))
print("raw:", len(row_wise), "B   zlib row-wise:", z(row_wise), "B   zlib column-wise:", z(col_wise), "B")
for name, c in cols.items(): print(f"  {name:7s} {len(set(c)):3d} distinct   entropy {H(c):.2f} bits/row")
print("sum of column entropies:", round(sum(H(c) for c in cols.values()), 2), "bits/row ->",
      round(n*sum(H(c) for c in cols.values())/8), "B if each column is coded at its own entropy")
```
```
raw: 664798 B   zlib row-wise: 53830 B   zlib column-wise: 45701 B
  date      1 distinct   entropy -0.00 bits/row
  method    3 distinct   entropy 1.37 bits/row
  status    4 distinct   entropy 1.79 bits/row
  path      4 distinct   entropy 1.92 bits/row
  ms       59 distinct   entropy 5.88 bits/row
sum of column entropies: 10.96 bits/row -> 27394 B if each column is coded at its own entropy
```

Handing the two layouts to the same general-purpose compressor gains only 18%. That is not a failure of the principle; it is a lesson about which coders care about layout. An [LZ matcher](../lz-family/) with a 32 KB window finds the string `/api/users` whether it is next to a date or next to another path, so for *repeats* the row layout costs little. The number to look at is the last line. Each column, considered alone, has a tiny alphabet and a skewed distribution, and the sum of the five per-column entropies is 10.96 bits per row, 1.4 bytes, against 2.7 bytes per row for zlib on the rows: a further factor of two that no byte-oriented matcher will find, because it lives in *per-column distributions*, not in repeated strings. That is what Parquet's column encoders go after. A column with four distinct values gets a **dictionary** and two-bit indices; a column of one repeated value gets **run-length encoding** and costs nothing; a column of timestamps or IDs gets **delta** coding; and then a general-purpose compressor runs over each column's page, seeing one clean distribution at a time. The 5 to 10× against gzipped CSV is mostly these encoders, and they are only available once the layout has put each column in a room of its own.

This is the second aha at its full scale. **Group bytes that share statistics, and separate the ones that do not.** YCbCr does it across three channels of a pixel. Byte shuffle does it across the four fields of a float. Struct-of-arrays, which game engines use for the same reason their physics loops want all the positions together, does it across the fields of a record. Columnar storage does it across the columns of a table. It is one transform at three scales, and in every case the reason it works is that the coders downstream, whether an entropy coder, a delta coder or an LZ window, do better on a clean distribution than on an interleaved mixture of several.

## Blocking is not compression

There is a transform in every codec's layout that does the opposite of everything above, and it is worth naming because it is so often mistaken for a compression technique. Blocks. bzip2 sorts in blocks of 100 to 900 kB; FLAC codes in frames of a few thousand samples; video codecs close a group of pictures every couple of seconds; Parquet splits a column into row groups; zstd frames, GGUF blocks and CacheGen's anchor tokens are the same thing in other clothes. In each case the stream is cut into pieces that are coded independently, with the context, the dictionary and the adaptive statistics reset at every cut.

Every one of those resets costs ratio. Here is the price on the shuffled float array from above, compressed as one piece and in independent pieces:

```python
import numpy as np, zlib
t = np.linspace(0, 1, 4096, dtype=np.float32)
sig = (2.0 + 0.5*np.sin(2*np.pi*3*t) + 0.002*np.random.default_rng(0).standard_normal(4096)).astype(np.float32)
planes = np.ascontiguousarray(sig.view(np.uint8).reshape(-1, 4).T)      # shuffled layout
whole = len(zlib.compress(planes.tobytes(), 9))
for blocks in (1, 4, 16, 64):
    chunks = np.array_split(np.arange(4096), blocks)
    total = sum(len(zlib.compress(np.ascontiguousarray(planes[:, c]).tobytes(), 9)) for c in chunks)
    print(f"{blocks:3d} independent block(s): {total} bytes  (+{100*(total/whole-1):.0f}%)")
```
```
  1 independent block(s): 9816 bytes  (+0%)
  4 independent block(s): 10369 bytes  (+6%)
 16 independent block(s): 10834 bytes  (+17%)
 64 independent block(s): 11454 bytes  (+17%)
```

Sixty-four blocks of 64 samples cost 17% more than one block, and every byte of that is paid for something other than ratio. What it buys is plumbing: a decoder can start at block 40 without reading blocks 0 to 39, which is random access; eight cores can take eight blocks each, which is parallelism; a corrupted byte destroys one block rather than everything after it, which is containment; and the encoder needs memory for one block rather than the whole stream, which is a bounded footprint. The third aha of this article is the one the [anatomy of a job](../anatomy-of-a-job/) article turns into a rule: **blocking is not compression, it is plumbing, and the block size is a dial set by the job, not by the data.** A streaming service sets its keyframe interval for seeking; a message queue frames each message so it can be decoded alone; a KV cache is paged in blocks of sixteen tokens so memory can be reclaimed. None of them would choose the block size by asking what compresses best, because the answer to that question is always "one block".

## Connections

**Decorrelation, again.** Run PCA on the RGB values of a natural photograph and the first principal component comes out close to a weighted average of the three channels, roughly luma, with the other two components carrying colour differences: YCbCr is a fixed, eye-weighted approximation to the data-adaptive rotation of the [rotations article](../rotations-and-rank/), in the same way the DCT is a fixed approximation to the KLT of a smooth row. The television committee derived it from colorimetry and compatibility; statistics would have handed them nearly the same matrix.

**Data-oriented design.** The struct-of-arrays layout that game and simulation programmers adopt for cache efficiency is the column layout for the same reason: a loop that touches only positions should not drag velocities and names through the cache. "Group what is accessed together" and "group what compresses together" are the same instruction, because both are about what sits in the same window.

**Database compression.** The reason an analytical database is both smaller and faster than the same data in a row store is that both properties come from one layout. Decoding only the columns a query touches is random access by construction, and the per-column encoders are the per-column coders of this article. Parquet's dictionary pages are a shared prior the [side information](../side-information/) article would recognise.

**In the game:** PRISM (channel decorrelation) sits in the Lens chair in THE GALLERY (JPEG), PRIME TIME (H.264 / AV1), THE SHOWROOM (WebP) and STUDIO MASTER (FLAC mid/side), and is paired with ILLUSIONIST for the COLOR CON combo; byte shuffle, columnar layout and framing have no fighter of their own because they are the plumbing every squad stands on.
