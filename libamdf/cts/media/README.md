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

## Running

Build `//libamdf/cts/media:jpeg_dynamic_bin` and
`//libamdf:amdf_shared_artifact` with `--//libamdf/config:media=true`. The normal
`//libamdf/cts/media:jpeg_dynamic` target declares the AMD GPU resource. An
explicit `--amdf_require_test` per case distinguishes successful execution from
an unavailable optional extension or engine. Tests obey the CTS runner's native
endpoint selection. ASAN and optimized executions establish separate correctness
records; neither is a benchmark.
