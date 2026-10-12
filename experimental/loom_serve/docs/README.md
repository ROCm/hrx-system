# Building and serving a model with Loom

This packet explains a working source-driven model runner and the boundaries a
second model port can reuse. The implementation is experimental, intentionally
small in scope, and available on
[`users/benvanik/loom-serve`](https://github.com/ROCm/hrx-system/tree/users/benvanik/loom-serve).
Compiler, runtime and model examples travel together on that branch.

| Question | Guide |
| --- | --- |
| How do I start empty, load a model and operate the server? | [Operator guide](OPERATING.md) |
| Where does a fresh agent start, and what counts as success? | [Agent entry point](AGENT_START.md) |
| Can I exercise multiple modality ABIs without downloading a model? | [Three-model source-only catalog](../models/fixtures/README.md) |
| What are the catalog, endpoint and lifetime contracts? | [Shared serving reference](../serving/README.md) |
| How does model math become runnable source? | [Model authoring](MODEL_AUTHORING.md) |
| Which exact source entries and storage records does a text port implement? | [Text package contract](../text/README.md) |
| Where do compiler, VM, HAL, scheduler, and sessions meet? | [Runner architecture](RUNNER.md) |
| What stays single-node now, and where can remote execution fit? | [Execution scope and ownership](RUNNER.md#single-node-scope-and-remote-execution) |
| How do we find substantial gains and establish that they are real? | [Performance experiments](PERFORMANCE.md) |

## Serving capabilities

One shared HTTP server now loads text, image, embedding and decision source
packages. The [weight-free catalog](../models/fixtures/catalog.json) exercises
real image, embedding and decision VM/GPU/HTTP paths, including raw tokenization,
through one executable. These deterministic fixtures prove ABI and ownership,
not learned embedding/decision quality or NPU execution. The native services
share task workers, parameter admission and command workspace; new compatible
packages do not require model-specific C or relinking. Real Qwen and Krea remain
the learned-model examples.

The generic text runner runs the authored Qwen3.8-27B model from Loom text,
weights, and tokenizer data. Startup compiles command programs, their reachable
GPU kernels, and VM control in process through the public `loomc` API. There
is no prerequisite compiler subprocess or prepared model-artifact directory.

One residency shares target weights, cached command variants, a VM process,
scratch storage, and an arena containing independently retained session rows.
Packed epochs mix prompt input with decode or MTP verification. Greedy sampling
and MTP proposal/acceptance happen on device. FFN gate/up weights are permuted
in place once during startup, without another resident weight image; all target
and MTP projection variants share that layout.

The text runner supports 1–16 resident rows with an automatic source-JIT
token/span catalog; the current Qwen package caps epochs at 512 packed input
tokens. Private F16 attention pages grow from one shared pool rather than
reserving a full context per row.
Admission reserves each request's complete high-water credit and queues excess
work. The real HTTP differential covers single, odd, and sixteen-row cohorts;
the [model guide](../models/qwen/README.md#pooled-kv-and-reserved-admission)
describes those checks and capacity-pressure/reuse coverage. Explicit
[shared endpoints](../text/README.md#explicit-shared-endpoints) retain immutable
KV/recurrent state; advancing branches receive private writers and partial-tail
COW without a recurrent-state clone. Explicit
[suspension](RUNNER.md#retained-state-and-physical-capacity)
can retain target/draft/recurrent state in DRAM while releasing device backing;
native checkpoint images survive reset and reuse of the originating row.
HTTP admission does not yet choose sessions for this transition. Optional
[automatic token-prefix reuse](../text/README.md#automatic-token-prefix-reuse)
shares bounded intermediate endpoints across requests; its native snapshots
remain in device memory. Disk paging still requires a storage/policy consumer.
The host still schedules and reads completion records between epochs.
The [runner guide](RUNNER.md#extension-boundaries) identifies the exact changes
needed to move those boundaries. The intended 20–40-agent deployment, learned
NPU models and autonomous device scheduler are not demonstrated by this branch.
The [resident embedding fixture](../models/fixtures/embedding/README.md#verification)
does prove source-JIT XDNA execution through HTTP, including first output,
cancellation and retirement; it has no learned weights.

## Fresh-machine prerequisites

This reproduction route is Linux x86-64 with a working C/C++ toolchain, Python
3.12 including `venv`, Git, and network access to the repository's pinned build
dependencies. [BUILDING.md](../../../BUILDING.md) owns the general build setup.
The runner does not require PyTorch, Transformers, a Python inference server,
or a model-specific compiler executable.

| Gate | Execution requirement |
| --- | --- |
| HTTP/chat/scheduler unit tests | CPU only |
| Source-JIT text, embedding and decision fixtures | Linux x86-64 CPU with VM/x86 targets and task HAL; no GPU or downloaded model |
| Small source-to-VM-to-GPU tests | AMD gfx11 GPU with working ROCr; tested on gfx1100 and gfx1151 |
| Vulkan image/text/embedding/decision fixtures | Vulkan device with the SPIR-V compiler and Vulkan HAL enabled |
| Resident embedding fixture | Physical XDNA device with matching compiler/HAL; no learned parameters |
| Full Qwen recipe below | gfx1151; some authored kernel targets explicitly require it |
| A new model/target | Its own real-device numerical qualification; the Qwen result is not proof for another target |

ROCr's `libhsa-runtime64.so.1`, its dependencies, kernel driver, and device
permissions belong on the execution host. An installed `rocminfo` should
enumerate the intended GPU before model debugging starts. Headers obtained by
the build are not a runtime installation. Linux/ROCr is the demonstrated learned
model route here, not a claim of learned-model support on Windows, NVIDIA or
the NPU. The separate source fixtures exercise additional backend contracts.

The example GGUF needs roughly 19 GiB of storage before runtime state. The
packed server defaults to 65,536 shared KV positions (4 GiB target plus 256 MiB
draft with MTP) and sixteen recurrent slots (149.625 MiB each), before shared
workspace, compiler, and runtime overhead. `--rows` and `--pool_capacity`
control these distinct memory costs; `--context_capacity` is a logical
per-session limit. There is no automatic CPU/disk spill. The startup
`Streaming` and `Residency` lines expose major allocations, not total peak
memory. The weight-free tests avoid this large-model capacity requirement.
Build storage and the model download need separate budgets; model files belong
on persistent storage with over 21 GB free, not a temporary RAM filesystem.

For a new checkout, the branch already includes the required compiler fixes:

```sh
git clone --branch users/benvanik/loom-serve https://github.com/ROCm/hrx-system.git
cd hrx-system
git rev-parse HEAD
python3.12 -B dev.py bazel setup
python3.12 -B dev.py bazel doctor
```

All subsequent commands use repository-root paths. There is no dependency on
another worktree, private source files, or a pre-populated build cache. A source
archive needs the whole repository tree, not just `experimental/loom_serve`.
Setup installs the pinned developer tools; it does not install the GPU driver.
Managed build environments retain their normal remote-build and hardware-runner
policies.

### CPU first output without a model download

The [operator walkthrough](OPERATING.md#first-output-starting-with-no-models)
works on Linux x86-64 using native x86 JIT kernels, not just protocol mocks.
After setup, this invocation builds only VM/x86 compiler targets and the CPU
task HAL, then starts an empty server:

```sh
build_tools/bin/iree-bazel-run --config=opt \
  --//loom/config/target:enable=vm,x86 \
  --//loom/config/execute:enable=iree_hal \
  --//runtime/config/hal:drivers=task \
  //experimental/loom_serve/serving:server -- \
  --device=task --pool_backing=fixed --listen=127.0.0.1:8080 --workers=8
```

These complete-list build overrides apply to this invocation; they do not
rewrite the worktree configuration. A managed hardware environment retains its
normal execution policy. The process needs neither ROCr nor GPU access.
Register the operator guide's embedding fixture, undrain it and request
`["a","bbb"]` at two dimensions. The independently checkable vectors are
`[1,1]` and `[6,3]`; changing the request to `"ab"` gives `[3,2]`. This proves
source loading, JIT, device execution and HTTP output, not learned embeddings.
The image-containing three-model catalog still requires a GPU.

The corresponding CPU source-service checks are:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  --//loom/config/target:enable=vm,x86 \
  --//loom/config/execute:enable=iree_hal \
  --//runtime/config/hal:drivers=task \
  //experimental/loom_serve/models/fixtures/text:service_cpu_test \
  //experimental/loom_serve/models/fixtures/embedding:service_cpu_test \
  //experimental/loom_serve/models/fixtures/decision:service_cpu_test
```

### GPU model configuration

For the Qwen/Krea examples, pinned ROCm headers suffice: Loom emits its own
native GPU code and this recipe needs no HIP device compiler or ROCm SDK path.

```sh
build_tools/bin/iree-bazel-configure \
  -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_VM=ON \
  -DLOOM_EXECUTE_IREE_HAL=ON -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned
```

An already configured worktree retains its complete target/driver choices rather
than replacing them with this fresh-checkout example. A separately configured
ROCm SDK can still supply device tooling for other repository tests.

## Reproduce the compiler/runtime boundary without weights

For final user-visible results first, the
[three-model walkthrough](../models/fixtures/README.md) supplies a tracked
catalog, exact requests and expected vectors/probabilities/pixels. The smaller
tests below isolate compiler/runtime boundaries when debugging a port.

The CPU gate is independent of model files and GPU availability:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/http:request_test \
  //experimental/loom_serve/http:server_test \
  //experimental/loom_serve/models/qwen:chat_test \
  //experimental/loom_serve/storage:block_pool_test \
  //experimental/loom_serve/text:schedule_test \
  //experimental/loom_serve/models/qwen:weights_test \
  //experimental/loom_serve/models/qwen:benchmark_service_test
```

Then, with the GPU runner available:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/runtime:jit_test \
  //experimental/loom_serve/runtime:weights_test \
  //experimental/loom_serve/runtime:control_test
```

Success means all named GPU targets pass without hardware skips. The JIT cases
exercise one and four shared workers. The CPU program test exercises Qwen's
actual source weight policy without loading the model. The loader tests use a small
safetensors file, distinct source-selected preparers, shared roots prepared
exactly once, unchanged tensors, failing policy/file inputs, and retained
consumers across activation, eviction and cancellation.
Test output is under `bazel-testlogs/experimental/loom_serve/`; adding
`--test_output=all` shows individual cases. The `JitTest` source-to-VM case checks
that two independently specialized eight-kernel stages change shared device
values from 100 to 112, 114, ..., 126 using source default 5 and explicit
override 7. It also verifies successful reuse after a native compilation
failure, and the separate cases cover invalid configuration and a command with
no native requests. A compiler that merely accepts the source does not satisfy
this gate.

[`jit_test.cc`](../runtime/jit_test.cc) is the smallest complete embedding example:
source catalog, source defaults and binding overrides, actual device-profile
specialization, native code loading, JIT VM invocation, and shared GPU state.
It destroys compiler storage before executing the prepared commands.
[`control_test.cc`](../runtime/control_test.cc) adds retained rows, explicit timeline
edges, feedback branching, and
device-produced indirect counts. Neither test needs model downloads.

## Reproduce full Qwen and retained HTTP output

The adapter expects the specific `Qwen3.8-27B-UD-Q5_K_XL.gguf` tensor layout
and its matching Hugging Face `tokenizer.json`. MTP additionally needs the
block-64 tensors in that weight file. Other quantization mixes or model sizes
are different ports, not interchangeable filenames. The model's
[source guide](../models/qwen/README.md) describes the tensor and kernel
contracts.

The download recipe uses the [Unsloth
GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/408fcc1807ab264d8cd644a7c4c0f58fbd32eebc/Qwen3.8-27B-UD-Q5_K_XL.gguf)
and [matching Qwen
tokenizer](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json).
Upstream has published a different GGUF under the same filename on `main`; the
download revision preserves the tensor layout expected by this source package.
A newer file needs its layout checked before substitution.

Set a persistent destination once, then fetch only these two files. This keeps
one checkpoint copy, without cloning the publisher's entire model repository:

```sh
model_dir=/path/to/models/qwen38-27b
mkdir -p "$model_dir"
curl --fail --location --continue-at - \
  --output "$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/408fcc1807ab264d8cd644a7c4c0f58fbd32eebc/Qwen3.8-27B-UD-Q5_K_XL.gguf
curl --fail --location --continue-at - \
  --output "$model_dir/tokenizer.json" \
  https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json
```

Weights are external assets, not part of the branch; the small integration
tests above remain runnable without obtaining them.

Build the exact executable targets before running or transferring them:

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/models/qwen:epoch_check \
  //experimental/loom_serve/text:server
```

The following direct invocations belong on a qualified GPU runner. That runner
needs the binaries, this model source directory, weights, and tokenizer; it does
not need `loom-compile`, Python model frameworks, or generated kernel files.

```sh
bazel-bin/experimental/loom_serve/models/qwen/epoch_check \
  --model=experimental/loom_serve/models/qwen \
  --prefill_capacity=128 --context_capacity=2048 --epoch=32:8 --mtp \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json"
```

This check allocates eight retained rows. It must exit zero with
`PASS: packed outputs, omitted outputs, row continuation and schedule transitions.`
With `--mtp` it also exercises speculative verification. That differential
compares implementations in this runner; it does not replace the independent
numerical oracle required for a new model port.

The resident comparison rows are independent of each recipe's span capacity.
The checker partitions its fixed cohort into legal epochs, retaining the same
isolated-row oracle across compact-slot changes. These complementary catalogs
exercise one-, two-, four- and eight-span verification recipes and all four
token capacities:

```sh
check_shapes() {
  bazel-bin/experimental/loom_serve/models/qwen/epoch_check \
    --model=experimental/loom_serve/models/qwen \
    --prefill_capacity=512 --context_capacity=2048 --mtp \
    --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
    --tokenizer="$model_dir/tokenizer.json" "$@"
}
check_shapes --epoch=32:1 --epoch=128:2 --epoch=256:4 --epoch=512:8
check_shapes --epoch=32:2 --epoch=128:1 --epoch=256:8 --epoch=512:4
check_shapes --epoch=5:1
```

Each call owns one residency and releases it before the next starts. The last
call exercises a single narrow recipe at the fixture's five-token minimum;
this is correctness coverage, not a suggested performance setting. The server's
smaller-token boundary checks below have a different purpose.

After the check exits and releases the residency, start one server:

```sh
bazel-bin/experimental/loom_serve/text/server \
  --model=experimental/loom_serve/models/qwen \
  --context_capacity=2048 --pool_capacity=8192 --mtp --mtp_depth=3 \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json" \
  --rows=4 --port=8080 --heartbeat_ms=1000
```

Startup emits a copied `compilation` report, allocation diagnostics, then a JSON `ready`
event on stderr. `GET /healthz` then reports readiness. The server binds loopback
and supplies neither authentication nor TLS; it is a local experiment, not a
publicly exposed endpoint. In another terminal on that host, this
standard-library-only client verifies distinct codewords and cached history
over two turns per client:

```sh
python3.12 -B experimental/loom_serve/models/qwen/benchmark_service.py \
  --url=http://127.0.0.1:8080 --clients=4 --long-lines=8 --max-tokens=32
```

Success is four `client_complete` records with distinct `verified_codeword`
values and a final `summary` with `clients: 4`, `requests: 8`, plus exit zero.
The client fails on a wrong codeword, missing retained-prefix reuse, incomplete
SSE, or empty response. A healthy HTTP port alone does not establish model
correctness. The service can now be inspected or used by another client; the
larger [runner README](../README.md#shared-text-execution) describes session and
chat behavior.

SIGINT/SIGTERM drains accepted model work and shuts down transport ownership.
Host ASAN here qualifies correctness; its timings are not optimized performance
results. Device access instrumentation is a separate option described in the
model guide. Shared hardware runs use the environment's benchmark lease around
execution, not around the build.

The [server lifecycle check](../models/qwen/check_service.py) automates startup and shutdown
around a retained four-client witness. It also verifies that depth-three MTP
rejects shape tables unable to fit four inputs, accepts the four-token boundary,
and accepts narrow shapes alongside a wider one. Depth zero also accepts a
one-token shape.
After building `text_server` as above, this runs sequential residencies, never
multiple weight copies at once:

```sh
python3.12 -B -m experimental.loom_serve.models.qwen.check_service \
  --server=bazel-bin/experimental/loom_serve/text/server \
  --model=experimental/loom_serve/models/qwen \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json" \
  --output=/path/to/run-evidence/service-check
```

Success ends with an `event: pass` record after eight retained requests and
clean server retirement. Per-case stderr is kept in a new output directory;
an existing directory is rejected to preserve previous evidence.

The [pooled row-family check](../models/qwen/check_rows.py) compares exact streaming output,
usage, and finish reasons against a sequential dense control. Its candidate
uses no manual epoch flags and checks that the automatic catalog follows the
ready cohort. Run with `--rows=1`, `--rows=3`, and `--rows=16` in separate
output directories; the last case requires a full sixteen-row verifier and a
512-token shape. [check_capacity.py](../models/qwen/check_capacity.py) separately checks
reserved admission, queued cancellation, idle eviction, and retained reuse
under an intentionally small shared pool. Both take the same model/server/path
arguments as the lifecycle check above.

## What another model author can take away

The text and image executables contain no built-in model package. A compatible
port supplies source bootstrap, control, commands, kernels and weight policy
through `--model`; it adds no per-model C or native registration entry.
The [text contract](../text/README.md) makes the current token/span bounds,
storage lanes, numerical entry points and chat policy explicit. The
[diffusion contract](MODEL_AUTHORING.md#source-defined-diffusion-package)
describes the separate finite-image flow.

A raw Hugging Face checkpoint is not executable Loom source. Its port must
establish actual tensor interpretation, numerical kernels and retained-state
semantics. The independent plain-text/JSON chat fixture proves that the same
native text caller accepts another policy; it does not prove another model's
math. A new modality outside these two contracts needs a bounded ownership
slice through the shared runtime, not a model-name branch in the server.
