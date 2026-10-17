# Composable real FFT experiment

This package contains allocation-free Loom C++ helpers for real forward and
inverse transforms. The 400-point audio transform and 1024-point spectral
transform are the performance workloads. Exact length, packet width, and storage
types specialize at import; the number of contiguous frames remains dynamic.
The implementation uses direct real radix-2/4/5 butterflies across frame lanes,
with a packed complex transform for the single-frame tail.

`fft.h` composes the stages; `geometry.h` owns radix order and storage extents.
`real_packet.h` and `frequency_packet.h` own the two
execution shapes; their stage and storage helpers can also be called from a
larger kernel. `frame_transpose.h` and `deinterleave.h` declare shuffle
families; their matching `.cxx` files provide semantic implementations and x86
AVX-512-profile specializations. A caller declares `data = [":headers"]` and
`deps = [":layout"]`, as the `example` library does. The shared provider library
is linked once even when several independently imported callers use the helpers.
The radix formulas in `real_radix.h` are adapted from
[BSD-licensed PocketFFT](https://github.com/mreineck/pocketfft/blob/c90e55b3d529f8efa40ed01a20de22405f45fc65/pocketfft_hdronly.h); its copyright and license
are retained in that file.

## Caller contract

`example.cxx` imports the helpers into ordinary Loom functions:

```cpp
fft_experiment::cascade_transform<false, 400, 8>(
    input, output, table, workspace, frames);
```

The first template argument selects inverse execution when true. Length must be
positive, even, and contain only factors two and five. Packet width is a power of
two. Eight lanes followed by 4/2/1 tails is the batched x86 policy; other widths
express a caller-selected schedule, without an implied performance guarantee.

Input and output independently use `float`, `std::float16_t`, or
`std::bfloat16_t`. Arithmetic, tables, and scratch are FP32. Conversion occurs
while loading or storing the frame, with no separate conversion pass. The
forward sign is negative and unnormalized; inverse divides by the length.

| Region | Forward extent | Inverse extent | Ownership |
|---|---:|---:|---|
| Input | `frames * length` scalars | `frames * (length + 2)` scalars | Borrowed, immutable |
| Output | `frames * (length + 2)` scalars | `frames * length` scalars | Borrowed, writable |
| Tables | `table_elements<length>()` floats | Same | Immutable, shareable |
| Scratch | `workspace_elements<length, width>()` floats | Same | Exclusive until completion |

Regions are disjoint and contiguous, and each scalar has its natural alignment.
The caller provides the complete extents before invoking the function. A forward
spectrum contains `length/2 + 1` interleaved real/imaginary pairs; DC and Nyquist
imaginary components are positive zero. Inverse ignores those two imaginary
inputs. Zero frames performs no loads or stores. Scratch contents are disposable
and can be reused by a subsequent invocation after completion. The implementation
does not allocate, synchronize with a host, or retain any borrowed storage.

Tables contain stage-major real twiddles followed by the single-frame complex
and reconstruction twiddles. `tables.py` constructs them once; its CLI writes
little-endian FP32 data:

```sh
iree-bazel-run //experimental/fft:tables -- 400 /tmp/fft400-table.bin
iree-bazel-build --config=loom-importer-cxx //experimental/fft:example
```

The 400-point table contains 1,185 floats; the 1024-point table contains 3,056.
For a separately specialized one-frame function, `frequency_packet.h` exposes
`forward` and `inverse` with `T=float` and `Width=1`; these use the
`frequency_table(length)` suffix alone and `2*length` scratch floats. Keeping
this latency entry separate lets a caller select its shape bucket before JIT
compilation instead of paying for the complete packet cascade.

This is an experimental source package. It does not select a general FFT API or
a device-placement policy. A consuming pipeline owns framing, windowing,
normalization beyond the stated inverse scaling, scheduling, and buffer
lifetimes. Integrating those operations with the stage helpers requires the
pipeline's numerical and memory-traffic qualification.

## Small checks

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx //experimental/fft/...
```

The default VM checks compare tiny transforms with independent direct-DFT
fixtures. They cover all nine storage pairs, zero and single frames, 8/4/2/1
packet tails, wider packet tails, radix composition, and input/table/scratch
ownership. Each direction is a separate small test. Fixtures are generated at
build time; generated spectra and large qualification workloads are not embedded
in compiler unit tests.

Each check also has an explicit `_native` target for an AVX-512-capable x86
qualification machine, for example `//experimental/fft:f16_bf16_forward_native`.
Those checks compare native execution with the VM and independent fixture using
two task workers. Their hardware requirement keeps them out of generic CPU CI.
