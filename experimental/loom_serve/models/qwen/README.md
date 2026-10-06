# Qwen3.8-27B GPU stages

These sources consume UD-Q5_K_XL GGUF weights for the experimental runner.
The prefill, greedy decode and packed epoch roots use command programs and kernels;
their artifacts are independent of the host scheduler. Parameter placement must
match across roots so all stages share one resident weight slab.

The runner consumes this directory directly with `--model`. Its
`sources.txt` catalog indexes configuration, command and kernel providers once.
`prepare.loom` is the cold VM entry: it declares command roots, run-dependent
specialization and checkpoint bindings. `config.loom` owns fixed specialization
bounds; `control.loom` contains isolated and packed inference VM entries;
`weights.loom` owns cold tensor preparation policy. `--prefill_capacity` bounds
the automatic packed token catalog; `--rows` bounds its independent span axis.
Repeated `--epoch=tokens:spans` replace that catalog with explicit JIT specializations.
`--context_capacity` sets the logical attention ceiling. Native code is
generated for the actual HAL device, not selected from precompiled directories.

`--mtp` prepares block-64 proposal, cache catch-up, and target verification.
The proposal uses a 32-token projection tile and the configured resident-row
count (1–16); each target epoch gets matching catch-up and verifier variants.
Shared embedding, target normalization, and full-vocabulary output roots reference the existing weight slab. Extra block-64
parameter groups load once. There is no command ABI or HAL extension.

## Source startup and control

The native constructor calls `prepare` with the requested prefill, context,
pool and resident-row capacities, MTP selection, packed shapes and checkpoint
path. Shapes cross as little-endian i64 token/span pairs, not a native struct.
The source declares stages through `prepare.stage`, `prepare.config_i64` and
`prepare.parameter`. It returns the effective prefill capacity, the number of
leading stages sharing target parameter placement, an opaque control buffer,
the terminal token spelling, and storage descriptors. The constructor JITs
those declarations and checks shared placement before streaming one parameter
bank.

The temporary startup process is destroyed before device setup. Its returned
control buffer survives in the residency's VM environment and is passed to
both warm entries; only source interprets its stage indices. There is no native
command-name/configuration table or per-row VM. Source computes retained buffer
sizes, arena views, initial target/draft row tables, page size and map origin.
Native allocation consumes those records without reconstructing model geometry;
workspace size/alignment comes from command reflection. Initial table payloads
survive cold VM teardown and accepted uploads, including failed construction.
One initialization join retires them before their references are released.

The storage result uses little-endian i64 records, not native structs. Its
eleven allocation records are residual, arena, packed metadata, target origins,
inputs, selected outputs, draft carry, committed metadata, verification results,
draft cache and draft origins. Each contains byte length, alignment and initial
zero extent. Zero length omits an inactive packed/MTP resource. Row views contain
five offset/length pairs per row in control, recurrent, attention, input and
progress order. Two initial origin payloads and a page-size/map-origin/carry-
stride/feedback-split record complete the cold result. These roles remain a
private adapter contract, not a universal storage language.

`control.loom:encode_epoch` maps semantic host spans into opaque device metadata
and padded token banks. Each host record carries input length, original position,
resident row, known-ID start/count, selection/proposal flags, output credit and
the precomputed first-plan index. The second plan occupies the next sixteen
records. The input stream contains only known IDs; speculative continuation
anchors come from device results. Native code owns the semantic table and ID
stream but contains no device descriptor, selection or verification-bank layout.

After both execution frontiers retire, `publish_epoch` folds ordinary or verified
feedback into `{consumed, known, outputs, verifications, tokens[8]}` records.
This preserves caller order across compacted continuation rows and distinguishes
prompt progress from speculation. All host buffers and VM wrappers are allocated
once; opaque upload/readback sizes come from source-declared device storage.
Encoding, submission and publication share one process and invocation. The
native owner retains all backing through partial submission failure and teardown.

`render_tool` in that same source program owns canonical tool-call framing and
literal identifier rules. Incoming assistant history and generated tool results
use this one formatter, preserving the checkpoint spelling independently of the
original generated token sequence in device state. Native `json.members` returns
typed offsets into an input buffer, and `json.unescape` decodes into source-owned
storage. Those utilities contain no chat-template or model policy. Formatting is
request-scoped; it creates no additional VM process, JIT compilation or GPU work.

The native adapter still manages page IDs, validates semantic spans, owns role
templates, generated XML/schema interpretation and retained-session policy, and
joins each bounded cohort before applying semantic progress. Those remaining
contracts are not yet a model-neutral text ABI.

`check_tools.py` exercises real generated typed calls and client-serialized
assistant/tool follow-ups through HTTP. A malformed intervening history must be
rejected without losing the completed session, and the valid follow-up must
answer from its tool result with nonzero retained-cache usage. The harness
returns deterministic text; it does not execute model-selected code. Client
transcripts and server heartbeats are recorded as JSONL:

```sh
python -B -m experimental.loom_serve.models.qwen.check_tools \
  --server=/path/to/qwen_server --model=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json --output=/path/to/new-results --clients=3
```

## Bounded device-fed continuation

`--mtp --mtp_depth=3 --continuation_epochs=2` lets the source VM enqueue two
target/MTP epochs before returning to admission and transport. The default is
one. The first epoch accepts mixed prompt/known and speculative spans normally.
The host pre-issues a second mixed plan from those same admitted rows: remaining
prompt chunks, continuing verifiers, and completed prompt tails ready to begin
generation. After catch-up, `continue.loom` resolves actual accepted positions
and output credit, compacts surviving spans, copies queued known tokens and
routes final predictions directly into speculative anchors. Each known chunk
is consumed once. The second shape fits its pre-issued worst-case cohort; a
prompt-only plan skips proposal work entirely.
There is no intermediate host wait, accepted-token download dependency or
token re-upload between those epochs.

EOS, exhausted output credit and insufficient context remove a span before
the second epoch can mutate its state. Two result banks and original-span tags
keep feedback immutable while later work proceeds. The result payload grows
by 448 bytes. A second fixed input/plan bank adds 4176 bytes, shared by all rows.
The host provisions the speculative high-water mark against
reserved capacity before submission and releases rejected pages
only after both work and feedback retire. At most eight outputs per original
span return in generation order. Cancellation and new arrivals are observed
at the next cohort boundary, not delayed behind an unbounded queue.

An empty second cohort still runs padded stateless math in the current fixed
commands; it mutates no retained state. This explicit experiment is not a
resident worker or a claim of zero idle weight traffic. The default remains
one epoch until controlled endpoint evidence supports a scheduling policy.
JSONL distinguishes `device_epochs`/`traversals` from host scheduling events;
per-row `verification_epochs` records actual advances and keeps draft acceptance
accounting correct. `known_tokens` separates caller-supplied inputs from
speculation, including a prompt-to-generation transition in one cohort;
`continued_prefill_tokens` identifies prompt work carried by the second epoch.
The HTTP comparison tools accept `--continuation-epochs=2`
to compare device-fed pooled execution against one-epoch dense output.

## Pooled KV and reserved admission

`--pool_capacity=N` selects a shared physical token budget for packed target
and MTP KV. `N` is a multiple of 64; `--context_capacity` remains each row's
logical ceiling. The packed server defaults to 65,536 shared positions and
sixteen resident slots. The isolated CLI and experiment tools default to dense
KV; explicit zero selects dense addressing in any caller. Pooled execution
generates cached shapes automatically when no `--epoch` is supplied and has no
failure fallback to the dense layout. Single-row calls use the same packed
addressing path.

Physical pages are assigned only when a span first writes them. Every target
layer uses the same page IDs in disjoint layer-major K/V planes. Draft KV uses
those IDs in its own allocation. New mappings upload before their consuming
commands on the existing timeline; existing mappings need no per-epoch update.
Verification provisions its full speculative extent, then releases rejected
full pages after target execution, catch-up, and feedback retire. Reset returns
the row's pages without clearing or copying KV. Recurrent/history state stays
private and exact, with its own fixed per-row cost.

For example, 262,144 pooled positions cost 16 GiB of target F16 KV and 1 GiB
of MTP KV, independent of their distribution across rows. Recurrent/history
state additionally costs 149.625 MiB per resident row. These are storage
formulas, not a long-context throughput or quality result.

The HTTP service reserves page-rounded credit for retained input, appended
input, requested output, and the speculative high-water mark before admission.
It assigns physical pages only during growth. A request that fits alone but
cannot fit alongside active guarantees waits in a bounded FIFO without a GPU
row. An impossible-alone request returns 400 with the required and available
capacity; rejected admission preserves an existing checkpoint.

Idle retained rows yield pages in LRU order when needed for admission. Active
requests keep their completion guarantee. Completion returns surplus credit
before terminal output drains; finish and cancellation release the active
reservation. Idle pages remain available for retained follow-ups until displaced.
`--pending_requests`, `--connections`, and `--request_body_bytes` bound host-side
waiting storage separately. Same named-session overlap, active or queued,
returns 409; an exhausted pending queue returns 503.

Logical reservations and physical page ownership are distinct. Overcommitting
active guarantees requires held/offloaded residency and is not enabled by this
policy. Direct model API callers still submit bounded epochs; exceeding free
pages rejects the whole epoch before device submission.

`check_capacity.py` is the complete HTTP witness. Given a built server, model
directory, weights, tokenizer, and a new output directory, it compares sequential
dense responses with concurrent requests using a deliberately constrained shared
pool. It exercises queued cancellation, same-session conflict, idle eviction,
an impossible request, and retained continuation. Its client-visible text,
usage, and completion reasons must match; admission and heartbeat accounting
must remain within the pool. Run it on the qualified model execution host:

```sh
python -B -m experimental.loom_serve.models.qwen.check_capacity \
  --server=/path/to/qwen_server --model=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json --output=/path/to/new-results
```

`check_rows.py` exercises the complete resident-row family through TCP. It
compares sequential dense output with a concurrent pooled cohort, requires every
configured row to execute, and requires a full-cohort MTP epoch. Unequal prompts
also exercise mixed prefill/decode for multirow runs. Set `--rows` to any value
from 1 through 16; 1, 3, and 16 cover the single, non-power-of-two, and maximum
residencies. The dense control uses explicit 64/128-token shapes; the candidate
uses the automatic catalog and verifies cohort-sized selection. At sixteen
rows it also requires a 512-token shape to execute. `--catalog=explicit` holds
both arms to the same shapes for addressing/row-family comparisons. It takes
the same model/server/path arguments as `check_capacity.py`.

## Online weight residency

The generic `weights.c` resolves the complete target/MTP placement before submitting
I/O. Shared roots retain views of the original allocation; only new roots
allocate storage and contribute file reads. The normal `file` parameter mode
uses asynchronous file handles and HAL queue reads targeting device-local
storage. The current placement uses HAL's bounded staging path on the qualified
GPU, including unified-memory hardware. This preserves the measured inference
performance but still incurs a staging-to-final transfer. Scoped-mappable
dual-local placements eliminated that transfer in experiments but regressed
serving throughput; they are not the default. Host visibility alone also does
not justify placing streaming weights across PCIe on a discrete device.

`weights.loom` exports `prepare_weight(buffer key) -> (buffer root, i64 bytes)`.
It selects `qwen38_prepare_ffn` and 61,276,160 bytes for block FFN gate/up keys;
all other keys return an empty root and zero to retain their original bytes.
Those name and shape rules live only in source. The loader validates the
returned size, JITs each distinct preparer once, and releases the cold policy
process before inference begins. The command name is ordinary UTF-8 rodata,
not an encoded host pointer or a new command-program ABI.

FFN gate/up tensors use a lossless block permutation:
`[channel][K256][176 bytes]` becomes
`[channel / 8][K256][channel % 8][176 bytes]`. All block headers and codes remain
unchanged. Other tensors keep their original encoding. This gives adjacent
channels a compact block stream without padding, expanded scales, or another
resident image. The isolated decode, four-token verifier, narrow 32-token,
generic small-batch, and wide fused FFN entries all consume the same format.
Their `_channel8` entry points specialize shared arithmetic implementations;
canonical entries remain useful for other model tensors and exact comparisons.

`weights_prepare.loom` is an ordinary source-JIT command. Each workgroup captures
eight complete K5120 rows in 28,160 bytes of local memory, then publishes the
permutation into that same owned range. No workgroup overwrites another's
unread input. Target and MTP tensors are prepared once during creation, never
on an inference path.

Four loading lanes each carry read-ready and prepared-ready timelines. A lane's
next read waits for its previous preparation; different lanes can overlap.
Consecutive canonical spans share a readiness group, while each prepared tensor
has its own read-to-dispatch edge. One terminal join makes all weights ready.
Queue submission order alone establishes no dependency. A failed lane aborts
the model, propagates failure to unsubmitted dependents, and leaves accepted
operations' resources retained until queue retirement and device teardown.

The existing profiling flags also cover startup reads and preparation, so their
overlap can be inspected separately from inference timings. A layout's kernel
gain is not an endpoint gain, and a warm filesystem-cache load does not measure
cold NVMe throughput.

`queue-events,dispatch-events,memory-events` provides host queue records,
preparation dispatch timing, and final allocation accounting. Device-queue
timing uses a distinct timestamp path on AMDGPU and disables host-clock fitting
when selected alongside those events. A read event marks completion publication,
not the duration of file transfer. Startup copy payload is a separate ledger:
a single final weight image can still receive every byte through staging.

`tests/prepared_q5.loom` evaluates canonical projections, permutes their weights
in place, and compares every output bit through each prepared consumer. It
covers channel tails, partial active spans, all production wide shapes, and
zero grids without model downloads or baked numerical outputs. On a qualified
GPU runner, select its owned cases with these providers and specializations:

```sh
model=experimental/loom_serve/models/qwen
for case in narrow narrow_zero generic_1_1 generic_4_4 generic_8_8 \
            generic_64_33 generic_64_64 generic_0_1 q8_1 q8_4 \
            wide_128_1 wide_128_128 wide_256_129 wide_256_256 \
            wide_512_511 wide_512_512 wide_0_1; do
  build_tools/bin/iree-bazel-run --config=asan \
    //loom/src/loom/tools/iree-test-loom -- \
    "$model/tests/prepared_q5.loom" \
    --library="$model/weights_prepare.loom" \
    --library="$model/kernels/qwen38/linear_q5k_f16_wmma.loom" \
    --library="$model/kernels/qwen38/ffn_gate_up_prefetch.loom" \
    --library="experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom" \
    --library="experimental/loom_serve/motifs/ggml/linear_qk_common.loom" \
    --library="experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom" \
    --config=qwen38.ffn.input_size=5120 --config=qwen38.ffn.output_size=17408 \
    --config=ggml.linear_q5k_q8_1_x4.token_capacity=4 \
    --config=ggml.linear_q5k_q8_1_x4.output_capacity=17408 \
    --config=ggml.quantize_q8_1_x4.group_capacity=160 \
    --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access \
    --case="@qwen38_prepared_$case" || exit
done
```

## Model execution

The serving path uses `loomc` in process for command products, native kernels,
and VM bytecode. Kernel experiments use `iree-test-loom`,
`iree-benchmark-loom`, and native compile reports independently of serving.

`control.loom` owns the packed submission sequence: optional proposal,
target/verification, a feedback fork, then MTP cache catch-up. The host supplies
validated descriptors, the selected catalog index, and retained buffers to one
VM invocation. Native imports submit indexed commands or bounded downloads;
they contain no Qwen routing. The host joins accepted work and feedback even
when a later native call fails, then publishes row frontiers only on success.

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
spans publish directly; speculative GDN capture uses 7,938,048 bytes per compiled
span, independently of token capacity. Only accepted prefixes replay into
retained state; capture and replay scratch both scale with the span shape.
Attention tails beyond the accepted position stay unreachable. MTP catch-up then consumes
accepted inputs paired with committed target hidden. All buffers and commands
are prepared once. The HTTP scheduler selects this optional path with
`--mtp --mtp_depth=3`; depth zero keeps MTP warm without
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

The real-weight `qwen_epoch_check --mtp` compares distinct
resident histories against ordinary target continuations: compact-row
permutation, zero/partial/full draft
acceptance, output-credit truncation, mixed known/speculative epochs, and
subsequent retained target continuations. Forced proposals come from ordinary
target execution, not baked token fixtures. Natural device-generated proposals
are checked through their accepted outputs and subsequent retained continuation.
`tests/mtp_proposal.loom` additionally checks compacted anchors, all three
publication positions, reversed resident rows, all counts from one through
sixteen, interleaved supplied candidates and known input, and untouched token storage.
The focused carry check additionally compares exact copy
identities and untouched rows/padding without loading model weights:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/tests/mtp_prepare.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/mtp_prepare.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/spans.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@mtp_carry_spans
```

`tests/mtp_accept.loom` compares acceptance records and both derived descriptor
tables against an independent minimum-frontier oracle across all counts from
one through sixteen and four mismatch/EOS/credit patterns, including untouched
inactive storage. `tests/gdn_speculation.loom` compares the entire sixteen-row recurrent/history slab bitwise before and after accepted
publication: known work commits once, speculative work remains unpublished
until replay, and rejected transitions never publish. Both support device access
sanitization. They use the corresponding kernels plus `gdn_spans.loom`,
`gdn_prefill.loom`, `gdn_common.loom`, and `spans.loom` for the GDN check.

From the repository root:

```sh
build_tools/bin/iree-bazel-build --config=asan //experimental/loom_serve/models/qwen:generate
# Run on a qualified GPU host with the source directory available.
bazel-bin/experimental/loom_serve/models/qwen/generate \
  --model=experimental/loom_serve/models/qwen \
  --prefill_capacity=512 --context_capacity=2048 \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json --prompt='What is 2+2?' --max_tokens=16
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
  experimental/loom_serve/models/qwen/tests/gdn_convolution.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/gdn_prefill.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/gdn_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@convolution_history

build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/kernels/qwen38/gdn_prefill.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/gdn_common.loom \
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

Full-model qualification additionally requires source-only JIT and actual
generated text through the runner. Kernel differential success alone does not
establish retained-session or end-to-end model correctness. The shared-row CLI
and retained pi service checks are described in the parent README.

### Fused feed-forward projection

The Q5 gate/up kernels view F16 activations as complete 256-channel blocks.
Each four-element packet is wholly inside one block, so its load needs no
K-axis mask. This view preserves the canonical contiguous activation layout
and does not add padding or alter the contraction or F32 SwiGLU calculation.

Each workgroup stages complete canonical Q5 blocks for both projections once
per 256 input channels. Metadata and code decoding then read the local slab
across its eight quantization groups. After the final contraction barrier
retires all block readers, the same allocation holds the per-wave output
transpose. The model keeps one unchanged global weight residency; encoded
staging and publication share storage only within the workgroup.

Both wide model paths use
[`ffn_gate_up_prefetch.loom`](kernels/qwen38/ffn_gate_up_prefetch.loom).
`qwen_compile_stage` binds `qwen38.ffn.input_size=5120` and
`qwen38.ffn.output_size=17408` through the existing JIT configuration. Concrete
kernel wrappers resolve these dimensions and pass them as SSA operands to the
shared template; its body has no model configuration dependency. Template
application retains those compile-time facts, unlike ordinary launch arguments
that remain runtime values in the device body. The
kernel requires complete 256-input/64-output-channel tiles. Token capacity
sets the grid while the actual token count stays in its device control buffer.

The outer block loop pipelines encoded weight and activation acquisition over
eight unrolled quantization groups. Of each projection's 704 sixteen-byte
packets, the partial 192-thread round precedes the complete 512-thread round.
This ordering leaves the complete next-block loads outstanding across current
matrix work; it does not fetch padding packets. The workgroup barriers retain
ownership of the single LDS stage. Padded activation lanes load bounded row
zero and then select positive zero. Read-ahead adds neither a second LDS stage
nor another global weight residency.

`tests/ffn_gate_up_prefetch.loom` compares complete output arrays bitwise with
the independently staged paired-wave implementation at the production K/N.
Three seeds exercise active counts 1, 128, 129, 256, 511 and 512, plus a zero
grid. Inactive output rows retain their original bytes. Explicit selection
keeps checks embedded in the provider libraries outside this suite. On a
qualified GPU test runner:

```sh
for shape in 128_1 128_128 256_129 256_256 512_511 512_512 0_1; do
  build_tools/bin/iree-bazel-run --config=asan \
    //loom/src/loom/tools/iree-test-loom -- \
    experimental/loom_serve/models/qwen/tests/ffn_gate_up_prefetch.loom \
    --library=experimental/loom_serve/models/qwen/kernels/qwen38/ffn_gate_up_prefetch.loom \
    --library=experimental/loom_serve/models/qwen/kernels/qwen38/ffn_gate_up_q5k_f16_wmma_wave32.loom \
    --library=experimental/loom_serve/models/qwen/kernels/qwen38/linear_q5k_f16_wmma.loom \
    --library=experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom \
    --config=qwen38.ffn.input_size=5120 --config=qwen38.ffn.output_size=17408 \
    --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access \
    --case="@ffn_prefetch_${shape}_case" || exit
done
```

`tests/ffn_prefetch_specialization.loom` also uses the shared motif at K256/N64
and K768/N128 in the same module, without model configuration. It checks
single/odd block counts and active-token tails against the independently staged
paired-wave implementation with three seeds:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/tests/ffn_prefetch_specialization.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/ffn_gate_up_prefetch.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/ffn_gate_up_q5k_f16_wmma_wave32.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/linear_q5k_f16_wmma.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access \
  --case=@ffn_independent_specializations
```

The generic owned-output entry remains available for other channel geometry.
Its inner quantization loop pipelines activation acquisition one group ahead
of matrix consumption. Local weight and metadata rows are initialized even for
padded channels, so decoding can remain converged and select positive zero at
the consumer. Its active-token and output-channel tails remain masked.

`tests/ffn_gate_up.loom` compares complete outputs with the paired-wave schedule,
including inactive rows, partial output channels and four token tiles at the
model's full projection shape. Single active rows, one and odd block counts,
and zero grids exercise pipeline fill, drain and padded acquisition. Distinct
activation rows and channel-varying gate/up weights exercise the indexing paths.
These are staging and layout differentials, not independent model-accuracy
oracles.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/tests/ffn_gate_up.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/ffn_gate_up_q5k_f16_wmma_wave32.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/linear_q5k_f16_wmma.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access
```

The narrow 32-token Q5 projection pipelines complete encoded weight blocks for
a single matrix. Four wave64 subgroups share a 64-channel tile. Partial packet
rounds retire before complete rounds, whose reads overlap the current block's
eight unrolled matrix groups. Complete and partial channel tiles specialize
the same body; tails never fetch nonexistent channels. The final barrier allows
the encoded slab to hold the output transpose. Canonical and prepared global
weight layouts specialize the same local contraction schedule.

`tests/linear_q5k_f16_wmma.loom` checks every output against the generic
global-decoding schedule at the three production projection widths and every
16-channel tail class, with distinct activation rows and channel-varying finite
weights. A zero-grid case checks disabled dispatches. Select the two owned cases
with the Q5 libraries and device access instrumentation:

```sh
for case in qwen38_narrow_q5_block_staging qwen38_narrow_q5_zero_grid; do
  build_tools/bin/iree-bazel-run --config=asan \
    //loom/src/loom/tools/iree-test-loom -- \
    experimental/loom_serve/models/qwen/tests/linear_q5k_f16_wmma.loom \
    --library=experimental/loom_serve/models/qwen/kernels/qwen38/linear_q5k_f16_wmma.loom \
    --library=experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom \
    --library=experimental/loom_serve/motifs/ggml/linear_qk_common.loom \
    --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access \
    --case="@$case" || exit
done
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
The projection/residual and F32/F16-input entries share F16 operands, F32
accumulation order, canonical weights and output layout. The C128/T256
F16-input residual entry gives each subgroup eight output fragments instead
of four, reusing the decoded weights across twice as many token rows. Its
43,008-byte LDS allocation retains the four shared result transpose slots.
The attention and GDN command programs select this entry for the 256- and
512-token FFN-down recipes; smaller and intermediate recipes retain their
own tile geometry. This is a model-source specialization, with no additional
weight image, global scratch allocation or host dispatch policy.

`tests/linear_q6k_f16_wmma_metadata.loom` compares all seven entries bitwise with
the independent global-decoding schedule and separate residual addition.
Seeded channel-varying weights, single/partial/full token tiles, zero grids,
channel tails and untouched output padding exercise the shared contract.
On a qualified GPU test runner:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/tests/linear_q6k_f16_wmma_metadata.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/linear_q6k_f16_wmma_metadata.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/linear_q6k_f16_wmma.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/layer_prefill.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_q6k_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_q8_0_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_qk_common.loom \
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
  experimental/loom_serve/models/qwen/tests/output_projection.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/output_projection.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_q6k_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_qk_common.loom \
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

Host `--config=asan` does not instrument GPU kernels. The runner's
`--kernel_sanitizer='access|operation' --kernel_sanitizer_reporting=default`
selects device assertions in the JIT pipeline. All stages in that residency use
the selected checks; performance runs use `--kernel_sanitizer=none`. The offline
inspection tool separately exposes `--sanitizer`/`--sanitizer-reporting`.

The runner uses the HAL stderr event sink for device diagnostics. Access checks
also require runtime shadow state: add `--amdgpu_asan=true` and
`--amdgpu_asan_report_policy=fail-device` to the runner invocation to report and
stop on an address-sanitizer failure. `--amdgpu_asan_shadow_mode=premapped`
maps poisoned shadow across the covered reservation, allowing covered stray
addresses to report instead of faulting on an unmapped shadow page.

The startup compiler reports failures before generation begins. Host and device
sanitizer configurations are separate evidence; record both when interpreting a
run. Sanitizer runs diagnose correctness, not throughput.

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
  experimental/loom_serve/models/qwen/tests/gdn_spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/gdn_spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/gdn_prefill.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/gdn_common.loom \
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
  experimental/loom_serve/models/qwen/tests/attention_spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_prefill.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_prefill_wmma.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@mixed_attention_spans \
  --config=qwen38.attention.cache_capacity=257 \
  --config=qwen38.attention.pool_capacity=0
```

The same GPU sanitizer flags apply. These comparisons qualify state routing
against the same math, not independent model accuracy or full-model mixed
execution. Metadata remains immutable until the epoch's completion edge permits
reuse. The service's packed scheduler gathers credited ready rows into this
entry; isolated and matched-math controls use the same ready-span partition.

`tests/attention_pages.loom` runs the same production packed kernels against
dense preparation/attention, using a fixture-owned permutation of physical
pages. It crosses a page boundary at absolute position 61, selects a nonzero
layer and row, and compares every output and physical cache element, including
unused pages and guards. With the same libraries above, select
`--case=@scattered_attention_pages`,
`--config=qwen38.attention.cache_capacity=192`, and
`--config=qwen38.attention.pool_capacity=256`. Repeat with `--sanitizer=access`.

The shared KV-map helpers, KV preparation and WMMA body take logical capacity
as an explicit operand. Concrete wrappers resolve configuration; the logical
extent sizes per-row page maps while physical capacity sizes the K/V planes.
`tests/attention_specialization.loom` composes two independent logical extents
(81 and 145) and pools (128 and 192) without any configuration bindings. Its
known-value case checks every physical cache element, different map strides
and permutations, full blocks, partial blocks and untouched output padding:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/tests/attention_specialization.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_prefill.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_prefill_wmma.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/attention_common.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access \
  --case=@attention_independent_specializations
```

The retained `qwen_epoch_check --mtp` witness places unequal prefixes directly
before page boundaries, derives accepted candidates from target execution,
injects mismatches, resets/reuses rows, and continues untouched histories.
Run the identical catalog with `--pool_capacity=0` and a nonzero capacity and
compare both selected tokens and committed-frontier records. A large physical
pool also exercises layer byte addresses above 4 GiB even with short occupied
histories. Host-ASAN runs qualify lifetime correctness, not performance.

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
build_tools/bin/iree-bazel-run --config=asan \
  //experimental/loom_serve/models/qwen:epoch_check -- \
  --model=experimental/loom_serve/models/qwen \
  --prefill_capacity=32 --context_capacity=16384 --epoch=32:4 \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json
```

The server and this witness use the same source-driven model creation path.
Each JIT stage uses the residency's context and parameter layout. The HTTP
service's default packed scheduler invokes the same packed API exercised here;
`--scheduler=isolated` provides the independent-row scheduling comparison.

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
benchmark-lock -- bazel-bin/experimental/loom_serve/models/qwen/workload \
  --model=experimental/loom_serve/models/qwen \
  --prefill_capacity=32 --context_capacity=16384 --epoch=32:4 \
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
  experimental/loom_serve/models/qwen/tests/output_spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/output_spans.loom \
  --library=experimental/loom_serve/models/qwen/kernels/qwen38/layer_decode.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@selected_output_rows \
  --sanitizer='access|operation'

build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/models/qwen/tests/output_spans.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_q6k_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom \
  --device=amdgpu --target=amdgpu:gfx1151 --case=@selected_output_projection \
  --config=ggml.linear_q6k_q8_1_x4.token_capacity=4 \
  --config=ggml.linear_q6k_q8_1_x4.output_capacity=9 \
  --config=ggml.quantize_q8_1_x4.group_capacity=160 \
  --sanitizer='access|operation'
```

## Four-row projection reuse

Both Q5 body templates receive token/output capacity as explicit operands,
separately from live token count and K/N. The four concrete kernels resolve
configuration for the single/four-row schedules and canonical/channel8 layouts.
Neither reusable body reads config or adds capacity arguments to the dispatch
ABI. The shared [composition test](../../motifs/ggml/tests/q5_specialization.loom)
applies both bodies at capacity pairs 4/65
and 32/96 in one composition without config bindings. It checks one/odd block
depths, a 65-channel tail, and an untouched fifth output row:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  "experimental/loom_serve/motifs/ggml/tests/q5_specialization.loom" \
  --library="experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom" \
  --library="experimental/loom_serve/motifs/ggml/linear_qk_common.loom" \
  --library="experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom" \
  --device=amdgpu --target=amdgpu:gfx1151 --sanitizer=access \
  --case=@q5_independent_specializations
```

The Q5 source contains independent-row and four-row weight-reuse schedules.
Its differential uses distinct activation rows, a 65-channel output tail and
K values 256, 768, 5120 and 17408. Fixed backing covers the largest sample;
each sample consumes its own contiguous active prefix. This checks the same
packed computation under both schedules, not an independent accuracy oracle.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //loom/src/loom/tools/iree-test-loom -- \
  experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom \
  --library=experimental/loom_serve/motifs/ggml/linear_qk_common.loom \
  --library=experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom \
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
  experimental/loom_serve/motifs/ggml/linear_q5k_q8_1_x4.loom \
  experimental/loom_serve/motifs/ggml/linear_qk_common.loom \
  experimental/loom_serve/motifs/ggml/quantize_q8_1_x4.loom \
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
