# Loom image-codec research archive

This branch preserves an independent GPU JPEG experiment for possible use as
an example in `hrx-solutions`. **It is not intended for hrx-system main or
loom-serve integration.** The compiler changes beneath the archive are a
separate upstream workstream. There is no codec pull request.

The experiment demonstrated compressed input, parallel entropy work,
reconstruction, bounded row storage, exact bicubic filtering, and isolated
per-image failure. Its final persistent schedule missed the performance goal:
2,140 ms on the GPU versus 517–520 ms on 16 CPU workers for 128 decode/resize
operations. Image buffer backing fell from 425.5 MB to 94.4 MB, but CPU memory
parity and complete libjpeg replacement were not established. [The result and
measurement boundary](RESULTS.md) explain the stopping decision.

## Contents and ownership

* `jpeg/` contains the native header parser, metadata/codebook construction,
  compressed-upload preparation, and independent positive/negative tests.
* `kernels/staged/` preserves the full-frame decoder. `kernels/streaming/`
  preserves the connected row producer. `kernels/resize/` holds the exact
  baseline and polling variants from the final bounded-memory comparison.
* `fixtures/` contains three small synthetic compressed-input cases with
  independent expected bytes, poisoned initial storage, and explicit check
  regions: positive, malformed marker, and malformed terminal padding.
* `reference/` contains the independently written sequential entropy decoder,
  counterexamples to unbounded speculative resynchronization, and numerical
  reference tools. `cpu/` retains the task-pool libjpeg comparators.
* `source-snapshots.tar.gz` preserves earlier kernel variants, source generators,
  scan algorithms, comparators, and experiment scripts under `research/`, plus
  exact pre-formatting qualified sources under `qualified/`. These are frozen
  source snapshots, not maintained entry points. Machine staging scripts refer
  to the original fleet and temporary fixture layouts; the commands below
  avoid them. The bundle contains source text only and can be extracted into
  any temporary directory for investigation.
* `evidence/` contains structured observations, native instruction listings,
  source/resource pins, and the final chart. Historical hashes identify the
  measured artifacts; they do not claim a fresh measurement on another build.

`manifest.json` records original and archived SHA-256 identities. Exact measured
kernel bytes remain in the source bundle; the runnable copies follow repository
formatting. Fixture bytes are unchanged. A few historical path strings were
relocated, with both identities retained. No third-party codec implementation,
model weights, compiler executable, object file, or toolchain is vendored here.
The parser's libjpeg dependency belongs to its independent tests; the CPU
comparison deliberately links libjpeg. This archive does not remove the
server's dependency.

## Bounded reproduction

The archive pins compiler commit
`f60115c770812cc3844fd8938b163f54c812f6dd`. Its parent stack is preserved in
this branch. Source changes in later Loom versions need ordinary requalification;
the recorded timings remain tied to their original compiler and hardware.
The standalone libjpeg oracle recipes use Bazel directly. They do not widen
the serving package's dependency facade or add a CMake codec dependency.

From the repository root:

```sh
python3 -B experimental/codec_reference/prepare.py --verify-only
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/codec_reference/jpeg:all \
  //experimental/codec_reference/reference:check_test
build_tools/bin/iree-bazel-build --config=asan \
  //loom/src/loom/tools/loom-import-cxx \
  //loom/src/loom/tools/loom-link \
  //loom/src/loom/tooling/target/amdgpu/serve:codec_reference
python3 -B experimental/codec_reference/prepare.py \
  --tools bazel-bin/loom/src/loom/tools \
  --output /tmp/codec-reference-positive
```

On a configured AMDGPU runner, immediately after building that exact caller:

```sh
bazel-bin/loom/src/loom/tooling/target/amdgpu/serve/codec_reference \
  --jpeg_fixture=/tmp/codec-reference-positive \
  --jpeg_initial=. --jpeg_regions=regions.txt \
  --jpeg_access=true --amdgpu_asan=true \
  --amdgpu_asan_report_policy=fail-device \
  --gtest_filter=JpegCommandTest.CompressedToRgb
```

`prepare.py --case small-r3-w128-invalid` and
`--case small-r3-w128-invalid-padding` prepare the two rejection cases in new
output directories. `--variant baseline` selects the original synchronization
policy. The caller verifies final RGB bytes, validity, peer outputs, and guards
through the actual command JIT and execution owner. It needs no temporary
source overlay or edits to production serving code.

The small fixtures are correctness witnesses, not substitutes for the large
comparison. A performance reproduction rebuilds the caller and CPU comparator
with `--config=opt`, regenerates the recorded large cohort, verifies outputs
first, and uses exclusive hardware with the same completion boundary. The
historical generators and source receipts retain that procedure; model weights
and external photographic inputs are intentionally not part of this archive.

## Scope of a later example

The qualified device family is restart-free, 8-bit sequential Huffman JPEG,
with grayscale, RGB, and the recorded YCbCr sampling profiles. Header parsing
recognizes more cases than the GPU path implements. Progressive, arithmetic,
lossless, complete restart support, and general multiscan decoding are not
provided by this device implementation. The independent sequential oracle has
its own, broader test scope. Rejected processes remain explicit errors.

A later example can extract the entropy algorithm, row-buffer ownership, and
command-program composition without presenting them as a competitive decoder.
Any renewed production proposal needs new measured evidence for the execution
shape, full input coverage, complete memory accounting, and final consumer
ownership. The previous compiler wins and passing checks do not supply that
evidence.
