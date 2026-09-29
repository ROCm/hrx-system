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

## Packed-span state ownership

`kernels/qwen38/spans.loom` defines experiment-private model data, not a command
ABI. An epoch describes packed token spans with their resident row, consumed
position and selected output. A separate row table locates persistent GDN and
KV storage in one fixed arena. Compact activation placement is independent of
resident state placement. The caller supplies validated, nonoverlapping spans
and distinct resident rows; kernels consume those invariants directly.

The GDN span kernels reuse the ordinary serial convolution and recurrent math.
Each workgroup carries one span's state through its causal token sequence and
publishes only that resident row. No state compaction or history snapshot is
needed by this serial-within-span schedule. The ordinary token-parallel
convolution still requires its immutable snapshot.

The packed-versus-isolated differential advances lengths 5, 3, 1 and 1 in
resident rows 4, 1, 5 and 2, then advances only the first two spans again using
the same dispatch capacity. Distinct inputs and initial state, inactive rows,
row guards and unused activation rows are checked bit-for-bit after both issues.
The metadata initializer belongs only to the fixture; a model caller supplies
those buffers directly.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/gdn_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_prefill.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/gdn_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@mixed_gdn_spans
```

Repeat with `--sanitizer='access|operation'` for GPU access/operation checks.
This qualifies GDN state routing, not full-model mixed execution or serving
throughput. The service still invokes one retained row at a time.

## Four-row projection reuse

The Q5 source contains independent-row and four-row weight-reuse schedules.
Its differential uses distinct activation rows, a 65-channel output tail and
K values 256, 768, 5120 and 17408. Fixed backing covers the largest sample;
each sample consumes its own contiguous active prefix. This checks the same
packed computation under both schedules, not an independent accuracy oracle.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/kernels/ggml/linear_q5k_q8_1_x4.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_qk_common.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/quantize_q8_1_x4.loom \
  --device=amdgpu --target=amdgpu:gfx1151 \
  --case=@ggml_linear_q5k_q8_1_x4_m4_differential_case \
  --config=ggml.linear_q5k_q8_1_x4.token_capacity=4 \
  --config=ggml.linear_q5k_q8_1_x4.output_capacity=65 \
  --config=ggml.quantize_q8_1_x4.group_capacity=544
```

Adding `--sanitizer='access|operation'` checks GPU accesses and operations as
well as host ASAN. The performance pair uses M=4, K=5120, N=17408 with the same
packed buffers. Both arms are single dispatches: one assigns separate work to
each activation row; the other shares each weight packet across four dot chains.
Neither benchmark includes activation quantization or the rest of the model.

Build the linker and benchmark tool with optimized flags before invoking them:

```sh
build_tools/bin/iree-bazel-build \
  //loom/src/loom/tools/loom-link //loom/src/loom/tools/iree-benchmark-loom \
  -c opt --features=thin_lto \
  --copt=-O3 --cxxopt=-O3 --host_copt=-O3 --host_cxxopt=-O3 \
  --copt=-march=native --cxxopt=-march=native \
  --host_copt=-march=native --host_cxxopt=-march=native
```

The benchmark consumes one module, so merge the source and providers first:

```sh
bazel-bin/loom/src/loom/tools/loom-link/loom-link \
  experimental/loom_serve/models/qwen38/kernels/ggml/linear_q5k_q8_1_x4.loom \
  experimental/loom_serve/models/qwen38/kernels/ggml/linear_qk_common.loom \
  experimental/loom_serve/models/qwen38/kernels/ggml/quantize_q8_1_x4.loom \
  --mode=merge --to=bc --output=/path/to/q5-linked.loombc
```

Once build activity has stopped, the following is a bounded comparison. The
eight device-local binding sets keep the 61 MB weight matrix from becoming a
hot-cache-only result. A batch is 64 repeated dispatches, not 64 model tokens.

```sh
benchmark-lock -- bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom \
  /path/to/q5-linked.loombc \
  --device=amdgpu --target=amdgpu:gfx1151 --measure=dispatch_complete \
  --compare=@ggml_linear_q5k_q8_1_x4_m4_gate_up_baseline,@ggml_linear_q5k_q8_1_x4_m4_gate_up \
  --interleave=ABABA --repetitions=2 --input-ring-count=8 \
  --batch-size=64 --iterations=3 --warmup-iterations=1 \
  --min-time-ms=0 --max-batches=3 --stable-p90-to-p50-ppm=0 \
  --profile-final-batch=false \
  --config=ggml.linear_q5k_q8_1_x4.token_capacity=4 \
  --config=ggml.linear_q5k_q8_1_x4.output_capacity=17408 \
  --output=/path/to/result.json
```

Comparison mode reports normalized host queue-completion time and suppresses
profiling inside its windows. For device duration, run separate A/B/A/B/A
windows under one benchmark lease: replace `--compare`, `--interleave` and
`--repetitions` with the selected `--benchmark=@...` name, enable
`--profile-final-batch=true`, and give each window a distinct
`--artifact-bundle-dir`. Keep all other flags and the linked input identical.
The profiled replay is separate from the uninstrumented completion measurement;
its dispatch distribution contains 64 device samples. Compare each candidate
with both adjacent controls and inspect profile warnings and sample counts.

Kernel reuse is not evidence that the server batches model math: its stateful
GDN/KV work and retained-row mapping need their own complete model witness
before changing the service schedule.
