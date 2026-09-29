# Qwen3.8-27B GPU stages

These sources implement the UD-Q5_K_XL GGUF layout for the experimental runner.
The prefill and greedy decode roots use ordinary command programs and kernels;
their artifacts are independent of the host scheduler. Parameter placement must
match across roots so all stages share one resident weight slab.

`compile.py` is a cold, offline driver of the normal Loom tools. It links each
root, emits a portable command plus kernel requests, and compiles exactly those
requests into HSACO images. The generated JSON contains configuration and an
artifact index, not executable host policy. Compilation stops on the first
failure. An output directory is usable only after the whole command succeeds.

From the repository root:

```sh
build_tools/bin/iree-bazel-build -c opt \
  //loom/src/loom/tools/loom-link //loom/src/loom/tools/loom-compile
python -B experimental/loom_serve/models/qwen38/compile.py \
  --output=/path/to/artifacts --context-capacity=512 --prefill-capacity=512
```

The prefill capacity selects a fixed dispatch schedule. Its control buffer holds
the actual count (1 through capacity), absolute input position, and EOS ID.
Stateful kernels only consume active rows. Stateless work may cover padding.
The 512-row schedule is reusable, not necessarily optimal for tiny suffixes.
The host caller must allocate attention storage for the same compiled context
capacity and enforce the input/count/context contract before submission.

GDN token-parallel convolution reads an immutable history snapshot. A preceding
copy kernel captures the old history into packed transient scratch; the final
active token publishes the next history into persistent state. A workgroup's
completion cannot order other workgroups' old-history reads. This separation is
required even when a particular launch order happens to hide the race.

## Correctness and kernel handoff

The focused checks need no weights or baked numerical fixtures:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/gdn_convolution.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_prefill.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@convolution_history

build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_prefill.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 \
  --case=@qwen38_gdn_conv_parallel_differential_case
```

The scenario compares an ordinary GPU function against an independent serial
VM function, using double accumulation and VM F32 exponential for activation.
The HAL profile observes result buffers. History and input preservation are
bitwise checks; activation permits 2e-7 absolute plus 2e-6 relative error.
Eight deterministic trials cross the old-history/input boundary. The complete
parallel-kernel differential separately checks all channels, Q/K normalization,
history publication and untouched padding at lengths 1, 2, 3, 5, 511 and 512.

Full-model qualification additionally requires fresh artifacts and actual
generated text through the runner. Kernel differential success alone does not
establish retained-session or end-to-end model correctness. The shared-row CLI
and retained pi service checks are described in the parent README.

For tuning, hold the GGUF, tokenizer, prompts, capacity and generated length
fixed. Recompile changed requests and rerun both numerical checks and the
full-model witness. Performance runs use optimized host binaries without ASAN,
the machine benchmark lock, and separate cold loading, prefill and completed
decode timing. ASAN correctness runs are not a throughput baseline. Changing
kernel math or dispatch geometry needs no runner or HAL change when the stage's
buffer/state and parameter-placement contracts remain intact.
