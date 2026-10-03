# Building and serving a model with Loom

This packet explains a working source-driven model runner and the boundaries a
second model port can reuse. The implementation is experimental, intentionally
small in scope, and available on
[`users/benvanik/loom-serve`](https://github.com/ROCm/hrx-system/tree/users/benvanik/loom-serve).
The branch includes main's final compiler-correctness fixes through `1618350bab`
(PR 1216). Commit `f076aa0955` adds shared, source-JIT weight preparation during
loading. Earlier performance checkpoints identify their measured historical
commits; the branch's serving changes were replayed unchanged onto this main
base before the prepared-weight comparison.

| Question | Guide |
| --- | --- |
| Where does a fresh agent start, and what counts as success? | [Agent entry point](AGENT_START.md) |
| How does model math become runnable source? | [Model authoring](MODEL_AUTHORING.md) |
| Where do compiler, VM, HAL, scheduler, and sessions meet? | [Runner architecture](RUNNER.md) |
| How do we find substantial gains and establish that they are real? | [Performance experiments](PERFORMANCE.md) |
| What is an approachable next model, and what exactly needs implementing? | [First independent port](FIRST_PORT.md) |

## What is demonstrated

The Qwen3.8-27B adapter runs its authored quantized model from Loom text,
weights, and tokenizer data. Startup compiles command programs, their reachable
GPU kernels, and VM control in process through the public `loomc` API. There
is no prerequisite compiler subprocess or prepared model-artifact directory.

One residency shares target weights, cached command variants, a VM process,
scratch storage, and an arena containing independently retained session rows.
Packed epochs mix prompt input with decode or MTP verification. Greedy sampling
and MTP proposal/acceptance happen on device. The full-model differential and
four-client, eight-turn retained HTTP witness passed after the JIT conversion
on gfx1151. The small JIT/control integration tests also passed on gfx1100.
These are correctness witnesses, not a newly established performance lead.

FFN weights are now permuted in place once during startup, without another
resident weight image. All target and MTP projection variants share that layout.
The [prepared-weight checkpoint](PERFORMANCE.md#prepared-weight-checkpoint)
records its small measured throughput gain, startup cost, and rejected mapped
memory placements; file loading still uses bounded staging.

The concrete adapter currently admits up to eight rows and 512 packed input
tokens, with startup-selected shapes and a common context capacity. Its caches
are contiguous F16 attention storage plus recurrent state, not a paged prefix
pool. The host still schedules and reads completion records between epochs.
The [runner guide](RUNNER.md#extension-boundaries) identifies the exact changes
needed to move those boundaries. The intended 20–40-agent deployment, NPU
execution, autonomous device scheduler, and cross-session prefix sharing are
not demonstrated by this branch.

## Fresh-machine prerequisites

This reproduction route is Linux x86-64 with a working C/C++ toolchain, Python
3.12 including `venv`, Git, and network access to the repository's pinned build
dependencies. [BUILDING.md](../../../BUILDING.md) owns the general build setup.
The runner does not require PyTorch, Transformers, a Python inference server,
or a model-specific compiler executable.

| Gate | Execution requirement |
| --- | --- |
| HTTP/chat/scheduler unit tests | CPU only |
| Small source-to-VM-to-GPU tests | AMD gfx11 GPU with working ROCr; tested on gfx1100 and gfx1151 |
| Full Qwen recipe below | gfx1151; some authored kernel targets explicitly require it |
| A new model/target | Its own real-device numerical qualification; the Qwen result is not proof for another target |

ROCr's `libhsa-runtime64.so.1`, its dependencies, kernel driver, and device
permissions belong on the execution host. An installed `rocminfo` should
enumerate the intended GPU before model debugging starts. Headers obtained by
the build are not a runtime installation. Linux/ROCr is the demonstrated runner
route here, not a claim that this service already runs on Windows, NVIDIA, or
the NPU.

The full-model witness used a 128-GB unified-memory gfx1151 machine. The GGUF
alone occupies 18.830 GiB on disk. At 2K context, eight target rows add about
2.169 GiB of retained device storage, before shared workspace, MTP weights/cache,
compiler, and runtime overhead. There is no automatic CPU/disk spill. The
startup `Streaming` and `Residency` lines expose major allocations, not total
peak memory. The small tests and the proposed 135M port avoid this large-model
capacity requirement. Build storage and the model download need separate
budgets; model files belong on persistent storage with over 21 GB free, not a
temporary RAM filesystem.

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

For a fresh configuration, pinned ROCm headers suffice: Loom emits its own
native GPU code and this example needs no HIP device compiler or ROCm SDK path.

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

The CPU gate is independent of model files and GPU availability:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve:http_request_test \
  //experimental/loom_serve:http_server_test \
  //experimental/loom_serve:qwen_chat_test \
  //experimental/loom_serve:qwen_schedule_test \
  //experimental/loom_serve:benchmark_service_test
```

Then, with the GPU runner available:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve:jit_test \
  //experimental/loom_serve:control_test
```

Success means both GPU targets pass, not skip: two JIT cases and eight control
cases. Test output is under `bazel-testlogs/experimental/loom_serve/`; adding
`--test_output=all` shows individual cases. The `JitTest` source-to-VM case checks
that independently specialized increments of three and seven change shared
device state from 100 to 110. A compiler that merely accepts the source does
not satisfy this gate.

[`jit_test.cc`](../jit_test.cc) is the smallest complete embedding example:
source catalog, two configurations, actual device-profile specialization, native
code loading, JIT VM invocation, and shared GPU state. It destroys compiler
storage before executing the prepared commands. [`control_test.cc`](../control_test.cc)
adds retained rows, explicit timeline edges, feedback branching, and
device-produced indirect counts. Neither test needs model downloads.

## Reproduce full Qwen and retained HTTP output

The adapter expects the specific `Qwen3.8-27B-UD-Q5_K_XL.gguf` tensor layout
and its matching Hugging Face `tokenizer.json`. MTP additionally needs the
block-64 tensors in that weight file. Other quantization mixes or model sizes
are different ports, not interchangeable filenames. The model's
[source guide](../models/qwen38/README.md) describes the tensor and kernel
contracts.

The qualified GGUF is 20,218,178,624 bytes. The actual inputs used for the
source-only witness have these SHA-256 identities:

```text
176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab  Qwen3.8-27B-UD-Q5_K_XL.gguf
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3  tokenizer.json
```

The matching public inputs are the [Unsloth GGUF at revision
`408fcc1807ab264d8cd644a7c4c0f58fbd32eebc`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/408fcc1807ab264d8cd644a7c4c0f58fbd32eebc/Qwen3.8-27B-UD-Q5_K_XL.gguf)
and [Qwen tokenizer at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json).
Upstream has published a different GGUF under the same filename on `main`;
the immutable revision and checksum are load-bearing. A newer file is not a
substitute for this reproduction input.

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
(
  cd "$model_dir" && sha256sum --check <<'CHECKSUMS'
176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab  Qwen3.8-27B-UD-Q5_K_XL.gguf
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3  tokenizer.json
CHECKSUMS
)
```

Both checksum lines must report `OK` before model execution. Weights are
external assets, not part of the branch; the small integration tests above
remain runnable without obtaining them.

Build the exact executable targets before running or transferring them:

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve:qwen_epoch_check \
  //experimental/loom_serve:qwen_server
```

The following direct invocations belong on a qualified GPU runner. That runner
needs the binaries, this model source directory, weights, and tokenizer; it does
not need `loom-compile`, Python model frameworks, or generated kernel files.

```sh
bazel-bin/experimental/loom_serve/qwen_epoch_check \
  --model=experimental/loom_serve/models/qwen38 \
  --prefill_capacity=128 --context_capacity=2048 --epoch=32:8 --mtp \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json"
```

This check allocates eight retained rows. It must exit zero with
`PASS: packed outputs, omitted outputs, row continuation and schedule transitions.`
With `--mtp` it also exercises speculative verification. That differential
compares implementations in this runner; it does not replace the independent
numerical oracle required for a new model port.

After the check exits and releases the residency, start one server:

```sh
bazel-bin/experimental/loom_serve/qwen_server \
  --model=experimental/loom_serve/models/qwen38 \
  --prefill_capacity=128 --context_capacity=2048 \
  --epoch=32:8 --epoch=128:8 --mtp --mtp_depth=3 \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json" \
  --rows=4 --port=8080 --heartbeat_ms=1000
```

Startup emits `jit_stage` events, allocation diagnostics, then a JSON `ready`
event on stderr. `GET /healthz` then reports readiness. The server binds loopback
and supplies neither authentication nor TLS; it is a local experiment, not a
publicly exposed endpoint. In another terminal on that host, this
standard-library-only client verifies distinct codewords and cached history
over two turns per client:

```sh
python3.12 -B experimental/loom_serve/benchmark_service.py \
  --url=http://127.0.0.1:8080 --clients=4 --long-lines=8 --max-tokens=32
```

Success is four `client_complete` records with distinct `verified_codeword`
values and a final `summary` with `clients: 4`, `requests: 8`, plus exit zero.
The client fails on a wrong codeword, missing retained-prefix reuse, incomplete
SSE, or empty response. A healthy HTTP port alone does not establish model
correctness. The service can now be inspected or used by another client; the
larger [runner README](../README.md#shared-qwen-execution) describes session and
chat behavior.

SIGINT/SIGTERM drains accepted model work and shuts down transport ownership.
Host ASAN here qualifies correctness; its timings are not optimized performance
results. Device access instrumentation is a separate option described in the
model guide. Shared hardware runs use the environment's benchmark lease around
execution, not around the build.

The [server lifecycle check](../check_service.py) automates startup and shutdown
around a retained four-client witness. It also verifies that depth-three MTP
rejects shape tables unable to fit four inputs, accepts the four-token boundary,
and accepts narrow shapes alongside a wider one. Depth zero also accepts a
one-token shape.
After building `qwen_server` as above, this runs sequential residencies, never
multiple weight copies at once:

```sh
python3.12 -B -m experimental.loom_serve.check_service \
  --server=bazel-bin/experimental/loom_serve/qwen_server \
  --model=experimental/loom_serve/models/qwen38 \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json" \
  --output=/path/to/run-evidence/service-check
```

Success ends with an `event: pass` record after eight retained requests and
clean server retirement. Per-case stderr is kept in a new output directory;
an existing directory is rejected to preserve previous evidence.

## What another model author can take away

The reusable unit is the source-to-command JIT and coarse queue/timeline
boundary. The Qwen C adapter still owns concrete dimensions, buffer meanings,
weight interpretation, and retained-state rules. Supplying another directory
to `--model` does not make an arbitrary Hugging Face model run. A new adapter
and its model source establish those semantics; the runtime below that boundary
already has a real caller and lifetime tests.
