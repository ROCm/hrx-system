# Native media conformance

These cases exercise the optional media extension without activating a GPU
compute device. They use direct `amdf_memory_t` backing and finish at independent
CPU observation of the decoded samples. Media/compute interoperability lives in
[`../interop/gpu/media`](../interop/gpu/media/README.md).

The JPEG corpus covers complete eight-bit baseline 4:2:0 streams decoded into
NV12 on the Linux JPEG engine. Both native image identities and direct plane
operands run at 128×96 and 80×72. Sixteen pictures reuse four recordings across
array layers; two separately activated media domains alternate original and
inverse images into shared backing through native timeline dependencies. The
application joins each eight-picture group, checks both queues' original
execution outcomes, and reuses its fixed slots. Eventfd delivery, old-point
notification, independent duplicate imports, subrange re-export, invalid plane
extents, exact configuration rejection, and checked destruction exercise the
same public lifetime as a pipelined consumer.

Every visible Y/U/V sample must differ from the independent CPU decode by at
most one. The complete returned plane extent belongs to native decoding,
including pitch and MCU padding; padding contents are unspecified. Bytes
outside those extents remain guarded. These cases make no throughput or
hardware-reset claim.

## Fixture provenance

The eight files are four original synthetic inputs and their decoded samples,
created offline with FFmpeg 8.1.2. They introduce no codec dependency into the
library or CTS executable. For each width/height, the source is planar full-range
YUV420. Plane `p` (Y=0, U=1, V=2) uses coordinates local to that plane:

```python
sample = 32 + ((x * 7 + y * 11 + p * 43 + ((x // 8) ^ (y // 8)) * 17) % 192)
```

The `_inverse` inputs replace each source sample with `255 - sample`. Encode
one frame using:

```sh
ffmpeg -f rawvideo -pixel_format yuvj420p -video_size WIDTHxHEIGHT \
  -i source.yuv -frames:v 1 -c:v mjpeg -q:v 3 -threads 1 -f image2 input.jpg
ffmpeg -i input.jpg -frames:v 1 -pix_fmt yuvj420p -f rawvideo decoded.yuv
```

The `.nv12` fixture preserves decoded Y verbatim, then interleaves decoded U,V
without scaling or color conversion. Fixture lookup uses embedded filenames,
independent of build-system input ordering.

## Video conformance

H264 and HEVC use the same resolved syntax and independent sample fingerprints
as the [GPU interop corpora](../interop/gpu/media/README.md). The Linux cases
exercise native image identities and direct plane operands with identical
visible results, explicit session BEGIN/END, bounded recording reuse, and
rejected preparation preserving the prior executable recording.

The VP9 corpus adds profile 0/NV12 and profile 2/P010, each with twelve 520×200
pictures. Two tile columns and partial bottom blocks exercise native reference
padding. Eight pictures use adaptive probabilities and segmentation; a second
four-picture keyframe group disables segmentation. The fixture deliberately
retains nonzero disabled filter features, which must not affect the pixels.
Logical reference-map entries alias unique application-owned physical slots.
Twenty-four pictures reuse four recordings and nine reference slots. Additional
16×16 streams exercise the native minimum with a single partial superblock,
using eight-bit native images and ten-bit direct planes on Linux. Invalid
map indices and header partition lengths must reject before replacing a live
recording. Every visible stored sample bit participates in the independent
fingerprint, including P010's six zero low bits.

### VP9 fixture provenance

`vp9_8.packet` and `vp9_10.packet` were generated with FFmpeg 8.0.1 and
libvpx-vp9, then independently decoded with FFmpeg's native `vp9` decoder into
NV12/P010. Each keyframe group starts with the same synthetic source. For sample
range `R = 1 << depth`, frame number `N`, and plane-local coordinates:

```text
Y = (3*X + 5*Y + 31*N) % R
U = (7*X + 11*Y + 17*N) % R
V = (5*X + 13*Y + 23*N) % R
```

Encoder settings are two threads, `cpu-used=4`, `crf=30`, zero target bitrate,
keyframe interval eight, no lookahead or alternate-reference frames,
`frame-parallel=0`, and `tile-columns=1`. The eight-picture group uses
`aq-mode=2`; the four-picture group uses `aq-mode=0`. `tile-rows=1` is requested;
the emitted headers have one tile row at this height. The two groups are joined
at their independent keyframes before software decoding.

Each packet has a little-endian `VP9CTSF1` header, precision, dimensions, and
picture count. Per-picture records hold resolved header/filter/segmentation
metadata, stable reference bindings, an independent per-pixel Y/U/V FNV-1a
fingerprint, and the complete VP9 picture padded to 128 bytes. The metadata
comes from `trace_headers` with parser state resolved offline; no parser or
codec library participates in CTS execution. Fixture SHA-256 values are
`f355a0d861ada0484fbee23acc27da3303ef8e5a9cbd61ee5f331b8660c78685`
and `f5114bec683367d7eecef4cc622933ca776bee83809b83998840e270c355d924`.
The minimum-size variants use the same procedure at 16×16, emitting one tile
column. Their SHA-256 values are
`a1379b0a975b7d0b71a96e9ef0c0a372d61c4968e629ebe58cb791a3cfd017f0`
and `e0dae4268e54a5dd3f86cb5eb1b9b87ea86cd56caec05df306f1d2a320fed953`.

### AV1 fixture provenance

The AV1 profile 0 corpus covers NV12 and P010 at both 64×64 and 128×128
superblock sizes. Baseline and extended streams each contain twelve 520×200
pictures and four tiles. Extended streams enable super-resolution with
denominator ten, restoration, and film grain; displayed samples differ from
the separately retained reconstructed references. Additional streams exercise
enabled segmentation with quantization matrices, nonidentity global motion,
and 16×16 lossless output.
The GPU interop corpus also uses 128×128 lossless variants for its 64×64 crop.
All streams are independently decoded with dav1d through FFmpeg 8.0.1.

Baseline inputs use the same plane equations as VP9 above, encoded with
`libaom-av1`, `cpu-used=6`, `crf=30`, two threads, keyframe interval eight,
zero lookahead, no alternate references, `aq-mode=1`, and a 2×2 tile grid.
Extended streams use aomenc 3.13.1 with `cpu-used=4`, `cq-level=30`,
`superres-mode=1`, both super-resolution denominators ten,
`film-grain-test=1`, and restoration enabled. Lossless variants use the same
source at their stated size with `lossless=1`, `aq-mode=0`, and one tile.

The active-tool streams use `cpu-used=0`, `aq-mode=3`, global motion, and
quantization matrices with levels three through seven. Their warped texture, for
sample range `R`, uses plane-local coordinates:

```text
A = floor((X + N*Y/32) / 8)
B = floor((Y + N*X/64) / 8)
Y = (37*A + 61*B + 17*((A*B) % 29)) % R
U = (7*(X + 2*N) + 11*Y) % R
V = (5*(X + 2*N) + 13*Y) % R
```

Each `AV1CTSF1` packet holds precision, dimensions, twelve resolved picture
records, tile ranges, stable physical slots and logical maps, a per-pixel
Y/U/V FNV-1a fingerprint, and compressed low-overhead OBUs padded to 128 bytes.
Syntax comes from FFmpeg `trace_headers`, with inferred and inherited state
resolved offline. Fixtures are selected by embedded name. No runtime parser
or software codec participates in either CTS or libamdf.

Fixture | SHA-256
--- | ---
`av1_baseline_10.packet` | `0afd5d4b2c5291aeb6fb6f6171e6b08f94674b45fb857242fdef2f7cff8bfb34`
`av1_baseline_8.packet` | `98b368ae69b42e33a7ca4e29ed10d6a6bb793acf9fee9044ec9c20e86feb1890`
`av1_extended_10.packet` | `939d1b7b5655bd056d0c4dac2ef48bd0bff91b8c1db9878ae7db40f7d0aec609`
`av1_extended_8.packet` | `f8c7aab9a42f1b8c809a6fa3d877d3ffda082081537b074a956edf4eba7db45d`
`av1_lossless_10.packet` | `16acbbb7f35e5af97baf23341248175496fc1e7f47541f7539bb39a67fd366e3`
`av1_lossless_8.packet` | `54bd666e92c836de8adc894cc660c75f4359cbfa645c5639a2379b56b222cf01`
`av1_lossless_crop_10.packet` | `a2e77cf9bee5b5b2116f4462025cabe6deffcc0bf8c0e3b2e8273129f7b5256d`
`av1_lossless_crop_8.packet` | `ccf0acb583e2f7c7dee6844090f372cdbe6915ab06298513fac4906c9181e06a`
`av1_tools_10.packet` | `7280ed4b86fc9793405e048e845fa2e070cbf9c4614019486d6be9348acf7672`
`av1_tools_8.packet` | `b25a5a9efa5d59e95b3da59c9e872328979b3d52d33f932ba6a3daad3c1ccad5`

## Running

Build `//libamdf/cts/media:jpeg_dynamic_bin` and
`//libamdf:amdf_shared_artifact` with `--//libamdf/config:media=true`. The normal
`//libamdf/cts/media:jpeg_dynamic` target declares the AMD GPU resource. An
explicit `--amdf_require_test` per case distinguishes successful execution from
an unavailable optional extension or engine. Tests obey the CTS runner's native
endpoint selection. ASAN and optimized executions establish separate correctness
records; neither is a benchmark.
