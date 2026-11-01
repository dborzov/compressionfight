---
title: "Motion compensation: predicting across frames"
slug: motion-compensation
chair: predict
family: "Temporal prediction: motion compensation"
summary: "Consecutive frames are the same picture shifted; the encoder pays to find the shift, the decoder copies for free, and the distance between keyframes is the dial between ratio and seeking."
fighters: [motion, spatial, anchor_delta]
related: [linear-prediction, block-transforms, anatomy-of-a-job, codec-dissections]
reading_time: 14
---

# The tracker: motion compensation

*Consecutive frames are nearly the same picture, shifted. How do you say "shifted" in a few bits, and what does it cost you to seek?*

## The question

Imagine a broadcast engineer in 1988, call him Teodor, in a lab that has been handed a brief by the telephone company: put a moving picture of a person's face down an ISDN line. The line carries 64 kilobits per second. The picture is the smallest format anyone will accept for a videophone, 176 by 144 pixels, with colour at a quarter of that resolution, around 38,000 samples a frame. At ten frames a second he has 6,400 bits per frame, which is about *a sixth of a bit per sample*. A single raw frame, eight bits a sample, is 300 kilobits and would take nearly five seconds to send.

Teodor knows the still-image toolbox, because the committee down the hall is finishing what will become JPEG: a block transform, a quantizer tuned to the eye, a Huffman coder; see [block transforms](../block-transforms/). Pushed hard, it gets a face down to about a bit per sample before it falls apart. He needs six times better than that, and the eye will not forgive six times coarser quantization. Where could another factor of six possibly come from?

He looks at two consecutive frames of his test clip. They are almost the same picture. The head has moved, a little, to the right.

## The difference, and why it is not enough

The obvious move is the one from [linear prediction](../linear-prediction/): predict that frame 2 equals frame 1, and send the miss. Here is a toy to make that precise. An $8 \times 8$ frame, dark except for a bright $3 \times 3$ square, and a second frame in which the square has moved one pixel to the right.

```python
import numpy as np
f1 = np.zeros((8, 8), int); f1[2:5, 2:5] = 200          # a bright 3x3 square at (row 2, col 2)
f2 = np.zeros((8, 8), int); f2[2:5, 3:6] = 200          # the same square, one pixel to the right
print(f2[1:6, 1:7])
print("plain difference, nonzero cells:", np.count_nonzero(f2 - f1))
```
```
[[  0   0   0   0   0   0]
 [  0   0 200 200 200   0]
 [  0   0 200 200 200   0]
 [  0   0 200 200 200   0]
 [  0   0   0   0   0   0]]
plain difference, nonzero cells: 6
```

The frame difference has six nonzero cells: a column of $-200$ where the square *was* and a column of $+200$ where it *is*. Nine bright pixels moved one step and the residual has six large values in it, with the full dynamic range of the picture, two-thirds of the content that moved. On Teodor's clip the effect is worse: a head is full of edges, and an edge moved by two pixels produces a residual that is a pair of edges, one negative and one positive, each as sharp as the original. "Same as last frame" is a poor predictor of anything that moves, and in a video call *everything* moves, slightly, all the time.

What is wrong with the prediction is not its magnitude. It is its *position*. The content of frame 2 is in frame 1, one pixel to the left of where it is needed.

## Aha 1: pay to find the shift, copy for free

So let the predictor say *where to copy from*. For a block of the current frame, search the previous frame over a small range of displacements and find the one whose contents match best. The standard measure of "best" is the sum of absolute differences, **SAD**: cheap, needs no multiplies, and good enough.

```python
def shifted(ref, dx, dy):
    """The reference moved by (dx, dy); pixels that come from outside are 0."""
    out = np.zeros_like(ref); H, W = ref.shape
    ys, xs = np.mgrid[0:H, 0:W]; sy, sx = ys - dy, xs - dx
    ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
    out[ok] = ref[sy[ok], sx[ok]]; return out
sad = np.array([[np.abs(f2 - shifted(f1, dx, dy)).sum() for dx in range(-2, 3)] for dy in range(-2, 3)])
print(sad)
dy, dx = np.unravel_index(sad.argmin(), sad.shape)
print("best vector (dx, dy) =", (int(dx) - 2, int(dy) - 2), " residual nonzero:", np.count_nonzero(f2 - shifted(f1, dx - 2, dy - 2)))
```
```
[[3600 3200 2800 2400 2800]
 [3600 2800 2000 1200 2000]
 [3600 2400 1200    0 1200]
 [3600 2800 2000 1200 2000]
 [3600 3200 2800 2400 2800]]
best vector (dx, dy) = (1, 0)  residual nonzero: 0
```

Twenty-five candidate displacements, from $(-2,-2)$ to $(+2,+2)$. The SAD surface has a single zero in it, at $(dx, dy) = (+1, 0)$: shift the reference one pixel right and it *is* frame 2. The residual is all zeros. Instead of six cells of $\pm 200$, the encoder sends one vector, which with 25 possibilities costs $\log_2 25 \approx 4.6$ bits, and a residual whose every entry is zero, which an entropy coder charges almost nothing for.

{{< fig src="img/motion-compensation/frames.png" alt="The two toy frames, the plain difference with six nonzero cells, the SAD search surface with a zero at (+1,0), and the all-zero motion-compensated residual" caption="The toy in one row: frame 1, frame 2 with the vector drawn, the plain difference, the SAD over all 25 candidates, and the residual after copying with (+1, 0)." >}}

Names, now that the toy has made them concrete. The displacement is a **motion vector** $\mathbf{v} = (dx, dy)$, chosen per block. Finding it is **motion estimation**: the search over candidate vectors, which an encoder runs for every block of every frame. Applying it is **motion compensation**: reading the block of the **reference frame** at the displaced position and using it as the prediction. For a block $B$ at position $\mathbf{p}$ in the current frame $F_t$, the prediction and residual are

$$\hat{F}_t(\mathbf{p}) = F_{t-1}(\mathbf{p} - \mathbf{v}), \qquad E_t(\mathbf{p}) = F_t(\mathbf{p}) - F_{t-1}(\mathbf{p} - \mathbf{v}) \quad \text{for } \mathbf{p} \in B,$$

$$\mathbf{v}^\star = \arg\min_{\mathbf{v} \in \mathcal{S}} \sum_{\mathbf{p} \in B} \left| F_t(\mathbf{p}) - F_{t-1}(\mathbf{p} - \mathbf{v}) \right|,$$

with $\mathcal{S}$ the search range. In H.261 the block was a $16 \times 16$ **macroblock** and the range a few pixels at integer precision. The vector itself is cheap to send; neighbouring blocks tend to move together, so codecs predict each vector from its neighbours' and code only the difference, the predictor's stage applied to its own output.

Now look at the asymmetry, because it is the cleanest example of one anywhere on the map. Motion estimation is a *search*. In the toy it is 25 SADs over 64 pixels; in a real encoder it is per block, per reference frame, at sub-pixel precision, with a smart search pattern rather than an exhaustive one, and it is by a wide margin the most expensive thing a video encoder does. Motion compensation is a *copy*: read 64 numbers from an address. The decoder never searches. It never sees the SAD surface. It reads a vector and copies. That is the asymmetry chosen by the job described in [the anatomy of a job](../anatomy-of-a-job/): one encode in a datacenter, a billion decodes on phones and televisions, so every hour spent searching for better vectors is paid once and saved a billion times. The [LZ family](../lz-family/) has the same shape, searching for matches versus copying from a window, and for the same reason; motion compensation is that idea with a two-dimensional window that happens to be the previous picture.

Three refinements make the copy almost always available. Objects do not move by whole pixels, so modern codecs search at **sub-pixel** positions: the reference is interpolated between its samples (H.264 uses a six-tap filter for half-pixel positions and averages for quarter-pixel), and a vector such as $(+1.25, -0.5)$ copies from a picture that never existed as a frame but is a better prediction than any whole-pixel shift. Objects also uncover what was behind them and are hidden by what passes in front, so a block may copy from any of several **reference frames** held in a buffer, not just the previous one; H.264 allows up to sixteen. That buffer is the decoder's memory bill: a 1080p frame at 4:2:0 is $1920 \times 1080 \times 1.5 \approx 3.1$ MB, so four references are 12 MB that a 200 mW decoder chip must keep and read at frame rate, which is why the reference count is capped by the profile and level a device advertises. And because the residual is quantized, the reconstructed reference has coding noise and faint block edges in it; an **in-loop deblocking filter** smooths those *inside* the prediction loop, so the next frame predicts from the cleaned picture, and encoder and decoder apply the identical filter so the loop stays closed.

{{< lab id="motion" >}}

## Aha 2: the GOP is a dial, and everything with a seek point turns it

The chain of predictions has a consequence Teodor noticed the first time a bit error hit his line: the picture did not merely glitch, it *stayed* wrong, the glitch copied forward from frame to frame until he restarted. Every frame was predicted from the last, so an error in one lived in all its descendants. The same chain has a second consequence for anyone who wants to start watching in the middle: there is no frame you can decode without first decoding the one before it, all the way back to the beginning.

The cure is to break the chain on purpose. Code some frames without any reference, as a still image: an **I-frame** (intra). Predict the others from earlier frames: **P-frames**. The run from one I-frame to the next is a **group of pictures**, a GOP.

```python
kinds = "IPPPIPPP"                                       # frame types in display order
def needed(k):                                           # frames to decode before frame k shows
    i = k
    while kinds[i] != "I": i -= 1
    return list(range(i, k + 1))
def corrupted_by_loss(k):                                # frames wrong if frame k's packet is lost
    j = k + 1
    while j < len(kinds) and kinds[j] != "I": j += 1
    return list(range(k, j))
print("seek to frame 3: decode", needed(3))
print("seek to frame 6: decode", needed(6))
print("lose frame 1: wrong frames", corrupted_by_loss(1))
print("lose frame 5: wrong frames", corrupted_by_loss(5))
```
```
seek to frame 3: decode [0, 1, 2, 3]
seek to frame 6: decode [4, 5, 6]
lose frame 1: wrong frames [1, 2, 3]
lose frame 5: wrong frames [5, 6, 7]
```

With a GOP of four, seeking to frame 3 costs decoding four frames, and a lost packet corrupts the picture until the next I-frame, three frames at most. Shorten the GOP and both numbers shrink. But look at what an I-frame costs.

{{< fig src="img/motion-compensation/gop.png" alt="Bars of estimated bits per frame across sixteen frames of a panning texture: I-frames at 21 thousand bits, P-frames at about 7 thousand" caption="A 64×64 texture panning two pixels a frame, coded by a toy codec with a GOP of eight: block matching over ±4 pixels, residual quantized with step 8, bits estimated as the order-0 entropy of what is left plus the uncoded vectors. The I-frames cost three times what the P-frames do." >}}

The figure is a toy codec on a panning texture: $8 \times 8$ blocks, a $\pm 4$ search, a quantizer, and bits estimated from the residual's entropy. Each I-frame costs about 21,000 bits; each P-frame about 6,900. The I-frame pays for the whole picture; the P-frame pays for the motion and the sensor noise. A GOP of eight on this clip averages 8,600 bits a frame; a GOP of two would average 14,000; an I-frame every frame is a stream of JPEGs at 21,000. Real codecs on real footage see a wider gap, because real P-frames are mostly *skipped* blocks whose vector and residual are both predicted perfectly, and B-frames widen it again.

**B-frames** are the dial turned the other way. A B-frame is predicted from a frame before it *and* a frame after it in display order, which means the encoder must hold the future frame, code it first, and transmit frames out of order; the decoder reorders them. An object that appears from behind another is in the future frame and not the past one, so bidirectional prediction finds copies that forward-only prediction cannot, and B-frames are the cheapest frames in a stream. The price is **latency**: to code a B-frame you must have already captured the frame after it. A streaming service with seconds of buffer pays that gladly. A video call, with a mouth-to-ear budget of around 150 ms, cannot.

Here is the aha. The GOP length sits on a single axis: *ratio against random access*, and the same axis runs through systems that have nothing to do with pictures. A zstd stream is split into frames that each start with an empty window, so a reader can begin at a frame boundary; a FLAC file is cut into blocks of 4,096 samples each with its own predictor, so a player can seek to any second; Parquet files have row groups that decode independently, so a query touches only the groups it needs; GGUF stores weights in blocks of 32 with their own scales, so a GPU thread can dequantize any block without its neighbours; CacheGen codes KV tokens in groups of ten against an anchor, so a CUDA thread can reconstruct any token on its own ([linear prediction](../linear-prediction/) ends on exactly that example). Every one of these is an I-frame by another name: a point at which the chain of context is cut, paid for in ratio, so that someone can start there. Where you set the interval is dictated by the job, not by the data. Two seconds between keyframes for streaming video; a keyframe only on demand, with a slow *intra refresh* that re-codes a stripe of each frame without reference, for a video call that cannot afford the bitrate spike of a whole I-frame and cannot wait for one when a packet is lost. In the strictest low-latency configurations the receiver acknowledges frames and the sender predicts only from frames it knows arrived, so a loss never propagates at all.

## Aha 3: the miss still has structure

Teodor's vectors got him most of his factor of six. The residual was the rest of the problem, and it is where the motion-compensation stage hands off to the next one.

In the toy, the residual was identically zero because the square moved by exactly one pixel and nothing else changed. In a real frame, the residual after compensation is small but it is not noise: lighting shifted a little across a whole block, so the residual has a DC offset; an edge moved by a fraction of a pixel the interpolation did not quite capture, so the residual has a faint edge in it; the camera's sensor noise is spatially white but the block's content is not. Small values with spatial correlation are precisely what a block transform is for. So the residual block goes through a DCT (in H.264, a $4 \times 4$ or $8 \times 8$ integer transform), its coefficients are quantized with a step chosen by rate control, and the quantized coefficients are entropy-coded with CABAC. The *decoder* then reverses that, adds the reconstructed residual to its motion-compensated prediction, runs the deblocking filter, and stores the result as the reference for the next frame.

That loop, predict from a reference, transform the miss, quantize, entropy code, reconstruct, store, is the **hybrid video codec**, and it is the design of H.261 in 1988. It is also the design of MPEG-1, MPEG-2, H.264, HEVC, VP9, AV1 and VVC. Thirty-five years of video codecs changed the parts: block sizes from fixed $16 \times 16$ to trees of $4 \times 4$ through $128 \times 128$; vector precision from integer to eighth-pixel; one reference to many; nine intra modes to dozens; Huffman to CABAC. The skeleton is Teodor's. Each generation buys thirty to fifty percent of bitrate at equal quality for five to ten times the encoder complexity, and nearly all of that complexity is in the search for better predictions, because the search is the encoder's to spend and the decoder's to ignore.

Rate control is where the dial of [rate and distortion](../rate-and-distortion/) meets this skeleton. The quantizer step is the only lossy knob, and the encoder turns it per block to hit a bitrate (constant bitrate for broadcast, where the channel is fixed) or a quality (CRF for files, where the step is held and the bits land where the content needs them). A budget of 5 Mbit/s for 1080p30 is a 300:1 ratio from the 1.5 Gbit/s raw picture, and the hybrid loop reaches it only because temporal prediction has already removed most of each frame before the transform ever sees it.

{{< lab id="spatial" >}}

{{< lab id="anchor_delta" >}}

## Connections

Motion estimation is a special case of **optical flow**, the computer-vision problem of estimating a per-pixel displacement field between two images. Flow methods from 1981 (Horn-Schunck, Lucas-Kanade) solve it densely and with sub-pixel precision; a codec's block matcher solves it coarsely, per block, and with a different objective, since it does not want the *true* motion but the displacement that yields the cheapest residual, which may copy from a place where the same texture happens to recur. Modern encoders and modern flow networks have been borrowing from each other ever since.

The configuration split between calls and streaming is the article on [the anatomy of a job](../anatomy-of-a-job/) in miniature: the same codec with B-frames and two-second GOPs for Netflix, and with no B-frames, intra refresh, one slice per packet and acknowledgement-driven references for a call. Opus against MP3 is the same story in audio.

And the GOP generalises past media entirely. A chain of deltas that must be decoded in order, cut at intervals by self-contained anchors, is the structure of a KV cache streamed by CacheGen, of a git packfile's delta chains (which have a maximum depth for exactly the seek reason), and of every snapshot-and-journal storage design. The engineers in those fields rarely call it a GOP. It is one.

**In the game:** [Motion compensation](/compressionfight/#fighter=motion) holds the prediction stage in the Streaming video task (H.264 / AV1), with [Spatial prediction](/compressionfight/#fighter=spatial) beside it for intra blocks; [Anchor delta coding](/compressionfight/#fighter=anchor_delta) is the same GOP idea in the KV cache over a network task (CacheGen). The [codec dissection](../codec-dissections/) of H.264 shows the whole hybrid loop as a pipeline.
