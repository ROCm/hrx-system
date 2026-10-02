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

`--stage=mtp` prepares block-64 warm/carry/proposal and target verification
commands. It emits `propose` at the fixed 32-token/eight-span proposal shape and
`warm<prefill-capacity>` and `verify<prefill-capacity>` for a matching target
epoch. Each loaded target shape needs both variants. Shared embedding, target
normalization and full-vocabulary
output roots resolve to existing target weight views; the extra block-64 groups
load once. There is no command ABI or HAL extension.

The model's optional MTP bundle warms its private cache after committed target
epochs. `loom_serve_qwen_model_verify` packs four-input verifiers with ordinary
known spans. A `PROPOSE` span supplies just its pending anchor. One cached
command compacts the proposal rows, gathers committed carry, and runs three
draft rounds whose sampled tokens feed both the next round and their reserved
verifier slots on device. Target verification and catch-up follow on the same
execution timeline without any intermediate host readback or wait. The private
hidden chain lives in the shared command workspace; only committed target
carry persists between epochs. Supplied candidates can use the same verifier
without requesting proposal generation. Device greedy acceptance stops at
mismatch, EOS or output credit. Known
spans publish directly; speculative GDN transitions occupy a 63,504,384-byte
capture, and only the accepted prefixes replay into retained state. Attention
tails beyond the accepted position stay unreachable. MTP catch-up then consumes
accepted inputs paired with committed target hidden. All buffers and commands
are prepared once. The HTTP scheduler selects this optional path with
`--mtp=/path/to/bundle --mtp_depth=3`; depth zero keeps MTP warm without
proposing. Whole verifier spans share a target epoch with known prompt input.

Both MTP input projections consume the normalized embedding/hidden matrix
directly with the token-reusing Q8_0/F16 WMMA contraction and F32 accumulation.
Canonical Q8_0 weights stay in the shared residency. There is no intermediate
Q8 activation pack or per-token weight traversal for the 10240-by-5120
concatenation projection. The projection's active token bound comes from the
device descriptor header, including compact generated cohorts.

Catch-up stops after normalization, K/V projection and cache publication. It
does not compute a query, read attention, project an attention output or run
the feed-forward block. Carry still records the committed target endpoint.
The K/V kernels and shape-dependent contraction choices are shared with
proposal execution. Four immutable parameter roots bind views of that same
proposal block's normalization and projection weights, without a second copy.

The full-sized `qwen38_mtp_projection_8_case` and
`qwen38_mtp_projection_512_case` checks in `linear_q6k_f16_wmma.loom` compare
against independent F32 products over channel-varying finite weights and
distinct activation rows. Their numerical tolerance accounts for F16 operand
rounding. Target verification, not draft arithmetic agreement, determines
committed outputs; full-model qualification also observes draft acceptance.

The real-weight `qwen_epoch_check --mtp=/path/to/bundle` compares distinct
resident histories against ordinary target continuations: compact-row
permutation, zero/partial/full draft
acceptance, output-credit truncation, mixed known/speculative epochs, and
subsequent retained target continuations. Forced proposals come from ordinary
target execution, not baked token fixtures. Natural device-generated proposals
are checked through their accepted outputs and subsequent retained continuation.
`tests/mtp_proposal.loom` additionally checks compacted anchors, all three
publication positions, reversed resident rows, one/eight generated spans,
interleaved supplied candidates and known input, and untouched token storage.
The focused carry check additionally compares exact copy
identities and untouched rows/padding without loading model weights:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/mtp_prepare.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/mtp_prepare.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/spans.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@mtp_carry_spans
```

`tests/mtp_accept.loom` compares acceptance records and both derived descriptor
tables against an independent minimum-frontier oracle across four fixtures,
including EOS and untouched storage. `tests/gdn_speculation.loom` compares the
entire eight-row recurrent/history slab bitwise before and after accepted
publication: known work commits once, speculative work remains unpublished
until replay, and inactive rows remain unchanged. Both support device access
sanitization. They use the corresponding kernels plus `gdn_spans.loom`,
`gdn_prefill.loom`, `gdn_common.loom`, and `spans.loom` for the GDN check.

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

### Fused feed-forward projection

The owned-output Q5 gate/up kernel views F16 activations as complete 256-channel
blocks. Each four-element packet is wholly inside one block, so its load needs
no K-axis mask. Active-token and output-channel tails remain masked. This view
preserves the canonical contiguous activation layout and does not add padding
or alter the contraction or F32 SwiGLU calculation.

Each workgroup stages complete canonical Q5 blocks for both projections once
per 256 input channels. Metadata and code decoding then read the local slab
across its eight quantization groups. After the final contraction barrier
retires all block readers, the same allocation holds the per-wave output
transpose. The model keeps one unchanged global weight residency; encoded
staging and publication share storage only within the workgroup.

`tests/ffn_gate_up.loom` compares complete outputs with the paired-wave schedule,
including inactive rows, partial output channels and four token tiles at the
model's full projection shape. Distinct activation rows and channel-varying
gate/up weights exercise the indexing paths. These are staging and layout
differentials, not independent model-accuracy oracles.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/ffn_gate_up.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/ffn_gate_up_q5k_f16_wmma_wave32.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/linear_q5k_f16_wmma.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_q5k_q8_1_x4.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access
```

The narrow 32-token Q5 projection uses the same encoded-block staging lifetime
for a single matrix. Four wave64 subgroups share a 64-channel tile; the final
barrier allows its encoded slab to hold the output transpose. The command
program and canonical global weight layout are unchanged.

`tests/linear_q5k_f16_wmma.loom` checks every output against the generic
global-decoding schedule at the three production projection widths and a
48-channel tail, with distinct activation rows and channel-varying finite
weights. Run it with the Q5 libraries and device access instrumentation:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/linear_q5k_f16_wmma.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/linear_q5k_f16_wmma.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_q5k_q8_1_x4.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_qk_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access
```

The wide Q5 projection pipelines the eight quantization groups inside each
256-channel block. Global weight-code and activation acquisition runs one
group ahead of LDS publication and matrix consumption. Scale/minimum metadata
stays in the ordered consumer. Padded channels read a valid source row and use
their zero metadata; padded activations are selected to zero after acquisition.
This keeps load results out of guarded reconvergence before the preceding
group's WMMA work. F32 and F16 activation storage share the same F16 operands,
F32 accumulation order, and unchanged resident weight representation.

`tests/linear_q5k_f16_wmma_wave32.loom` compares both activation storage types
bitwise with the wave64 global-decoding schedule, including single-token,
partial-token and partial-channel tiles, untouched output padding, and full
5120-channel projections. The check uses the same Q5 libraries as above plus
`kernels/qwen38/linear_q5k_f16_wmma_wave32.loom`.

### Q6 metadata contractions

The C64/C128 metadata-staged Q6 contractions acquire codes and activations one
K32 group ahead of matrix consumption. Metadata scaling stays in the ordered
consumer; shared storage is not double-buffered. Padded channels and tokens
read valid source rows and are selected to positive zero after acquisition.
The six projection/residual and F32/F16-input entries preserve their F16
operands, F32 accumulation order, canonical weights, dispatch mapping and
output layout.

`tests/linear_q6k_f16_wmma_metadata.loom` compares all six entries bitwise with
the independent global-decoding schedule and separate residual addition.
Seeded channel-varying weights, single/partial/full token tiles, zero grids,
channel tails and untouched output padding exercise the shared contract.
On a qualified GPU test runner:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/linear_q6k_f16_wmma_metadata.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/linear_q6k_f16_wmma_metadata.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/linear_q6k_f16_wmma.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/layer_prefill.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_q6k_q8_1_x4.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_qk_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access
```

### Vocabulary projection

Epoch and MTP roots pass their complete padded output cohort to the shared
greedy head. Packing and argmax cover that cohort once. The contraction uses
32-row groups, then 8-row groups, then four-row or single-row tails. The
model-private 8/32-row kernels specialize the canonical 5120-channel Q6_K
layout; a wave reuses decoded weights across independent activation rows.
This changes neither vocabulary coverage nor output ordering. No additional
retained state or resident weight copy is required.

`tests/output_projection.loom` checks every vocabulary output against independent
wave64 row contractions using three seeds, distinct activation rows and
channel-varying finite packed weights. This qualifies grouping and addressing
against the established math, not independent model accuracy. The MTP model
witness above also covers accepted-state continuation through the new head.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen38/tests/output_projection.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/qwen38/output_projection.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_q6k_q8_1_x4.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/quantize_q8_1_x4.loom \
  --library=experimental/loom_serve/models/qwen38/kernels/ggml/linear_qk_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 \
  --config=ggml.linear_q6k_q8_1_x4.token_capacity=512 \
  --config=ggml.linear_q6k_q8_1_x4.output_capacity=248320 \
  --config=ggml.quantize_q8_1_x4.group_capacity=69632 \
  --sanitizer=access --sanitizer-reporting=trap
```

Trap reporting checks accesses and terminates on a violation without the
additional structured-reporting register footprint. These full-vocabulary
checks need roughly a GiB for each live weight fixture, independently of model
weights; they belong in a bounded GPU run, not a concurrent small-test sweep.

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
vocabulary head, which shares Q6 weights across grouped predictions.
The map is produced with the spans, not recovered by a device-side search.
Padded output rows are zeroed; the fixed head still computes padded rows,
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
