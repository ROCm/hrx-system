# Source-defined models for loom-serve

This directory contains learned model packages, not another server. The supported
`loom-serve` executable loads them at runtime through deployment JSON, JITs their
Loom programs, and owns their shared device resources and HTTP lifetimes. There
are no per-model C implementations or compiler-internal dependencies here.

- [Qwen3.8-27B](models/qwen/README.md): retained text, mixed prefill/decode,
  paged KV, MTP and DFlash2 source programs.
- [Qwen3.8 Flash-Next](models/qwen_flash_next/README.md): resident MoE text,
  recurrent GDN, paged sparse attention, and n-gram embeddings over packed epochs.
- [Krea 2 Turbo](models/krea/README.md): text-to-image, BF16 or INT8 ConvRot DiT,
  and a separately loaded LoRA with request-time strength.
- [Pocket TTS](models/pocket_tts/README.md): streaming speech, registered
  reference voices, retained conditioning, and independent request batching.
- [Shared motifs](motifs/README.md): reusable arithmetic receiving SSA operands
  and template arguments, independent of a model's configuration namespace.

These numerical programs target GFX11 AMD GPUs. A server built with other
backends does not translate their AMDGPU-specific kernels into CPU or NPU code.
The server and model packages have independent lifetimes: changing a compatible
package needs no server rebuild. The packages remain experimental and separate
from the supported serving implementation.

## Build and load

From the repository root, build the supported tool with AMDGPU enabled in the
normal project configuration:

```sh
iree-bazel-build --config=opt //loom/src/loom/tools/loom-serve:loom-serve
```

The model guides produce deployment documents that bind local asset paths.
The same binary accepts several model documents:

```sh
bazel-bin/loom/src/loom/tools/loom-serve/loom-serve \
  --device=amdgpu --listen=127.0.0.1:8080 --workers=8 \
  --model=/tmp/qwen-deployment.json --model=/tmp/krea-deployment.json
```

With no `--model`, the server starts empty. The documents can instead be posted
to `/v1/models/register`; their absolute `base_directory` locates programs on
the server machine. Listener readiness is not model readiness: registration
starts asynchronous preparation, and `/v1/status` reports progress or a model
error. Inference can queue while preparation completes.

Deployment capacities belong in JSON, not command-line model flags. Process
flags select devices, listeners, workers and memory policy. The
[model-document reference](../../loom/docs/src/serving/model-documents.md),
[HTTP reference](../../loom/docs/src/serving/http-profiles.md), and
[operator guide](../../loom/docs/src/serving/workflows/operate-model-server.md) define
those shared contracts.

## Copy a package

Copy the selected model and the shared sources listed in its guide, preserving
their relative directory layout. Each `model.json` names its preparation and control
programs and the numerical provider list; preparation also names parameter
policies such as `weights.loom`. Checkpoints and tokenizers are separate assets,
not included in this repository.

All program roles accept `.loom` or `.loombc`. Bytecode is portable IR that still
JITs for the selected device. Converting a provider requires updating its JSON
reference; converting a parameter policy also requires updating its source-owned
reference in preparation. Native `.hsaco` files are not a replacement for this
source package contract.

## Check and measure

The numerical cases exercise real kernels against independent arithmetic oracles
or narrower implementations. Chat-policy checks invoke the public request tool.
No checkpoint download is needed for this suite:

```sh
iree-bazel-test --config=asan //experimental/loom_serve/...
```

Cases that depend on an authored GFX11 target declare
`check.requires<hal.amdgpu.target> {target = "gfx11-generic"}`. They execute on
compatible RDNA3 and RDNA3.5 devices and report an explicit skipped requirement
elsewhere. Portable arithmetic, text, and control cases retain their wider
execution scope; incompatible compiler input still fails when no requirement
excludes it.

The [mixed-model check](models/krea/README.md#mixed-model-lifecycle-check) uses
real assets and the supported HTTP server. It verifies exact outputs across
weight eviction and retained-session restoration, rather than merely checking
that a model loads. Python prepares assets and provides test and benchmark
tooling; learned inference executes through the source programs.

Performance measurements use an optimized server, exclusive device access and
an explicitly recorded workload. ASan results establish correctness, not speed.
The Qwen guide provides a client-visible multi-session benchmark; the server's
`/metrics`, `/logz` and `/v1/status` expose the corresponding scheduling and memory
behavior. Run logs, machine details and comparisons are experiment artifacts,
not model documentation.

For a new model, start with the supported
[model-authoring guide](../../loom/docs/src/serving/model-programs.md) and
[small ABI fixtures](../../loom/src/loom/serve/fixture/README.md). These larger
packages show specialized kernels and composition, not a requirement to copy
their architecture or native server internals.
