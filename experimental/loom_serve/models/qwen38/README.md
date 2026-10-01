# Qwen3.8-27B GPU stages

These sources implement the UD-Q5_K_XL GGUF layout for the experimental runner.
The prefill, greedy decode and packed epoch roots use command programs and kernels;
their artifacts are independent of the host scheduler. Parameter placement must
match across roots so all stages share one resident weight slab.

`compile.py` is a cold, offline driver of the normal Loom tools. It links each
root, emits a portable command plus kernel requests, and compiles exactly those
requests into HSACO images. The generated JSON contains configuration and an
artifact index, not executable host policy. Compilation stops on the first
failure. An output directory is usable only after the whole command succeeds.

`--stage=mtp` prepares block-64 warm/carry/proposal commands. It emits `draft`
and `begin` at the fixed 32-token/eight-span proposal shape and
`warm<prefill-capacity>` for a matching target epoch. Each loaded target shape
needs its warm variant. Shared embedding, target normalization and full-vocabulary
output roots resolve to existing target weight views; the extra block-64 groups
load once. There is no command ABI or HAL extension.

The model's optional MTP bundle warms its private cache after committed target
epochs. `loom_serve_qwen_model_propose` returns three candidates without changing
target positions or predictions. Its private hidden chain is discarded before
the next proposal; committed target hidden supplies the starting carry. This is
proposal execution, not accepted speculative decoding: the service does not yet
select or publish these candidates. Target verification and accepted recurrent
commit must own that publication boundary.

The real-weight `qwen_epoch_check --mtp=/path/to/bundle` checks proposal isolation,
duplicate resident histories, compact-row permutation, and subsequent retained
target continuations. The focused carry check additionally compares exact copy
identities and untouched rows/padding without loading model weights:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/mtp_prepare.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/mtp_prepare.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/spans.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@mtp_carry_spans
```

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

### Device sanitizer diagnostics

Host `--config=asan` does not instrument GPU kernels. `compile.py` forwards
`--sanitizer` and `--sanitizer-reporting` to each kernel compilation; for example,
`--sanitizer='access|operation' --sanitizer-reporting=default` enables device
access/operation checks and structured reports. Instrumented artifacts belong in
a separate output directory from performance artifacts.

The runner uses the HAL stderr event sink for device diagnostics. Access checks
also require runtime shadow state: add `--amdgpu_asan=true` and
`--amdgpu_asan_report_policy=fail-device` to the runner invocation to report and
stop on an address-sanitizer failure. `--amdgpu_asan_shadow_mode=premapped`
maps poisoned shadow across the covered reservation, allowing covered stray
addresses to report instead of faulting on an unmapped shadow page.

Stages can be instrumented independently with `--stage`; record exactly which
artifacts were instrumented when interpreting a result. Compilation failure
leaves an incomplete stage, not a usable partially sanitized artifact set.
Sanitizer runs diagnose correctness, not throughput.

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

The attention differential uses the same packed metadata contract and shared
query normalization, RoPE, cache publication and causal WMMA bodies. Lengths
17, 3, 1 and 1 at distinct consumed positions exercise query-tile and cache-block
tails. Resident rows 4, 1, 5 and 2 select a nonzero layer in a guarded two-layer
arena; a second issue changes lengths and active cardinality. Complete query,
gate, output and cache buffers agree bit-for-bit with isolated execution,
including inactive rows, layers and padding.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/attention_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/attention_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/attention_prefill.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/attention_prefill_wmma.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/attention_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@mixed_attention_spans \
  --config=qwen38.attention.cache_capacity=257
```

The same GPU sanitizer flags apply. These comparisons qualify state routing
against the same math, not independent model accuracy or full-model mixed
execution. Metadata remains immutable until the epoch's completion edge permits
reuse. The service's packed scheduler gathers credited ready rows into this
entry; isolated and matched-math controls use the same ready-span partition.

## Complete packed model witness

`epoch.loom` advances a flat token matrix through the same dense layer programs
as prefill. Only stateful GDN/attention dispatches select resident rows. An
epoch's output-to-packed-row map gathers requested final span rows into the
vocabulary head, which shares Q6 weights across groups of four predictions.
The map is produced with the spans, not recovered by a device-side search.
Padded output rows are zeroed; the fixed head still computes padded groups,
including when no prediction is requested. Stateless dense work also covers
the selected token capacity. These are explicit schedule costs to measure.

The runner's `loom_serve_qwen_model_epoch` accepts distinct resident row indices,
input spans and output-selection flags. One VM process, weight slab, workspace
and state arena serve every row. Cold setup creates the immutable row-origin
table and reusable epoch buffers. Completion commits consumed positions and
selected predictions. A span that omits output selection invalidates its old
prediction but preserves the consumed state for a subsequent explicit input.
There are no per-epoch state copies or model-storage allocations. HAL allocation
behavior is a separate property; this interface makes no zero-allocation claim
about queue implementation internals.

The manual real-weight witness uses packed rows 6/1/7/3 and disjoint isolated
reference rows in the same residency. It compares mixed 5/3/1/1 inputs, changing
cardinality, paused-row resumption, omitted predictions and return to ordinary
decode. Length-one reference inputs use the same prefill dense schedule until
the explicit transition to the separate decode family. A 16384-token context
makes some resident origins exceed 4 GiB. Full-model output/continuation checks
complement the whole-state kernel differentials above.

```sh
python -B experimental/loom_serve/models/qwen38/compile.py \
  --stage=all --output=/path/to/artifacts \
  --context-capacity=16384 --prefill-capacity=32 --span-capacity=4
build_tools/bin/iree-bazel-run --config=asan \
  //experimental/loom_serve:qwen_epoch_check -- \
  --prefill=/path/to/artifacts/prefill --decode=/path/to/artifacts/decode \
  --epoch=/path/to/artifacts/epoch \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json
```

Build the linker/compiler first as shown above. `--stage=both` retains the two
isolated roots; `--stage=epoch` emits only the packed root. Stage placement and
context must agree. The optional packed VM configuration is selected once at
model creation, not instantiated per session. The HTTP service currently uses
the isolated configuration; this witness exercises the packed public model API.

The same manual executable has bounded completed-work comparisons. Build its
exact target with the optimized flags used below, stop build activity, then run
under `benchmark-lock` with the same model arguments and one comparison flag:

| Flag | Ready inputs | Separate baseline |
| --- | --- | --- |
| `--compare=single` | One five-token span | One ordinary prefill |
| `--compare=mixed` | 5/3/1/1 tokens | Two prefills plus two ordinary decodes |
| `--compare=full` | 17/13/1/1 tokens | Two prefills plus two ordinary decodes |
| `--compare=decode` | Four one-token spans | Four ordinary decodes |

Each comparison warms both arms, then emits three JSONL samples per ABABA
window. Prefixes are reset/replayed outside timing for every sample; physical
rows, histories, weights and workspace are identical between arms. Timing
includes input upload, VM/queue submission, execution and result completion.
Every sample checks final positions and selected tokens against the baseline.
No sanitizer or profiling belongs in these timings. Report window medians and
compare each packed window with both neighboring controls. Tiny-context model
comparisons establish neither long-context performance nor service/vLLM parity.

### Retained coding-work replay

`qwen_workload` accepts one already-rendered chat prompt file per resident row
(up to eight). It replays the same retained prefixes before each measurement
window, then consumes the remaining prompt text and generates until EOS or a
per-row output cap. Rows initially decoding have their complete prompts seeded;
the first `--prefill_rows` rows begin at `--retained_tokens`. Seeding is reported
separately and excluded from the measured window.

The bounded workload reserves one input for each ready decode row and divides
the remaining token capacity among ready prefill rows. Both execution arms
receive this same partition. `A` executes each span independently with ordinary
prefill or decode; `B` submits all spans as one epoch. This compares completed
model work, not HTTP transport, asynchronous arrivals, or an optimized serving
policy. The independent baseline uses the supplied fixed-capacity prefill
program; it does not select smaller shapes for partially filled spans.

Compile equal prefill/epoch token capacities and sufficient span capacity. For
example, after building the exact executable with the optimized flags below:

```sh
benchmark-lock -- bazel-bin/experimental/loom_serve/qwen_workload \
  --prefill=/path/to/artifacts/prefill --decode=/path/to/artifacts/decode \
  --epoch=/path/to/artifacts/epoch \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --prompt_file=/path/to/review-0.txt --prompt_file=/path/to/review-1.txt \
  --prompt_file=/path/to/review-2.txt --prompt_file=/path/to/review-3.txt \
  --retained_tokens=1024 --prefill_rows=2 --max_tokens=64 \
  --order=ABABA --baseline=practical
```

JSONL records each completed epoch's spans, starting positions, prefill/decode
input counts and selected outputs. Window records include total execution time,
wall time including planning/logging, output IDs, EOS and final positions. Text
decoding occurs afterward on stderr. A decode row's first prediction was made
during seeding and is printed with its continuation but excluded from timed
output counts. No cold model loading or prefix replay is hidden in a throughput
number. The first packed invocation is included, not discarded as warmup.

The practical decode family can change greedy trajectories relative to the
prefill/packed math. The report preserves differences; equal-work comparisons
require matching actual counts, not merely matching requested caps. For a
correctness run, `--baseline=matched --order=AB` uses the prefill math for
length-one isolated inputs too and fails on any selected-token mismatch. This
matched baseline is an ownership oracle, not a decode-performance baseline.

Selected-output routing and the Q6 four-row projection have focused checks:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/output_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/output_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/layer_decode.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@selected_output_rows \
  --sanitizer='access|operation'

build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/output_spans.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_q6k_q8_1_x4.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/quantize_q8_1_x4.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@selected_output_projection \
  --config=ggml.linear_q6k_q8_1_x4.token_capacity=4 \
  --config=ggml.linear_q6k_q8_1_x4.output_capacity=9 \
  --config=ggml.quantize_q8_1_x4.group_capacity=160 \
  --sanitizer='access|operation'
```

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
