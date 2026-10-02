# Building and serving a model with Loom

This packet explains a working source-driven model runner and the boundaries a
second model port can reuse. The implementation is experimental, intentionally
small in scope, and available on
[`users/benvanik/loom-serve`](https://github.com/ROCm/hrx-system/tree/users/benvanik/loom-serve).
The first published in-process JIT implementation is commit `6e26aa8eda`, based
on mask-safety commit `47682d1bc30944f4ddd863cd1f613f95dcf898bb`.

| Question | Guide |
| --- | --- |
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

The concrete adapter currently admits up to eight rows and 512 packed input
tokens, with startup-selected shapes and a common context capacity. Its caches
are contiguous F16 attention storage plus recurrent state, not a paged prefix
pool. The host still schedules and reads completion records between epochs.
The [runner guide](RUNNER.md#extension-boundaries) identifies the exact changes
needed to move those boundaries. The intended 20–40-agent deployment, NPU
execution, autonomous device scheduler, and cross-session prefix sharing are
not demonstrated by this branch.

## Reproduce the compiler/runtime boundary without weights

Commands use repository-root paths. [BUILDING.md](../../../BUILDING.md) owns
tool installation, runtime loading, and target configuration. A new build needs
Loom AMDGPU and VM compilation, IREE HAL execution, and the AMDGPU runtime
driver. For a fresh configuration with an installed ROCm SDK:

```sh
build_tools/bin/iree-bazel-configure \
  -DLOOM_BUILD=ON -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_VM=ON \
  -DLOOM_EXECUTE_IREE_HAL=ON -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_PATH=/opt/rocm
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve:jit_test \
  //experimental/loom_serve:control_test
```

The SDK path is installation-specific. An already configured worktree retains
its complete target/driver choices rather than replacing them with this example.
ROCr and device access belong on the execution machine; compiler headers on a
build worker do not supply either. Managed build environments keep their normal
remote-build and hardware-runner policies.

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

Weights are external assets, not part of the branch. These identities establish
the reproduction inputs; the small integration tests above remain runnable
without obtaining them.

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
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json

bazel-bin/experimental/loom_serve/qwen_server \
  --model=experimental/loom_serve/models/qwen38 \
  --prefill_capacity=128 --context_capacity=2048 \
  --epoch=32:8 --epoch=128:8 --mtp --mtp_depth=3 \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --rows=4 --port=8080 --heartbeat_ms=1000
```

The first command checks packed versus independent target execution, retained
continuation, and speculative acceptance/commit behavior. Successful startup of
the second emits a JSON `ready` event; `GET /healthz` then reports readiness.
In another terminal, this standard-library-only client verifies distinct
codewords and cached history over two turns per client:

```sh
python -B experimental/loom_serve/benchmark_service.py \
  --url=http://127.0.0.1:8080 --clients=4 --long-lines=8 --max-tokens=32
```

SIGINT/SIGTERM drains accepted model work and shuts down transport ownership.
Host ASAN here qualifies correctness; its timings are not optimized performance
results. Device access instrumentation is a separate option described in the
model guide. Shared hardware runs use the environment's benchmark lease around
execution, not around the build.

## What another model author can take away

The reusable unit is the source-to-command JIT and coarse queue/timeline
boundary. The Qwen C adapter still owns concrete dimensions, buffer meanings,
weight interpretation, and retained-state rules. Supplying another directory
to `--model` does not make an arbitrary Hugging Face model run. A new adapter
and its model source establish those semantics; the runtime below that boundary
already has a real caller and lifetime tests.
