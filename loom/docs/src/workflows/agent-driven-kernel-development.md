# Agent-driven kernel development

Loom makes kernel search auditable. Source, correctness cases, benchmark
workloads, compiler decisions, native artifacts, and physical measurements can
all retain one experiment identity. An agent can therefore explore a large
space without asking a benchmark number to stand in for understanding.

The agent is not an oracle. Each edit is a bounded experiment whose expected
compiler consequence is stated before compilation and whose winner is selected
by matched physical evidence after correctness passes.

```text
production witness
    -> semantic cut
    -> checked source and workloads
    -> compiler hypothesis
    -> compiler and native evidence
    -> controlled physical experiment
    -> integrated result
```

This workflow coordinates the focused formatting, correctness, benchmark,
artifact, and report workflows. Those pages own their complete command and
result contracts; this page owns the evidence and decision loop around them.

## Freeze one production witness

Kernel work begins from an exact operation in a real program. A convenient
synthetic matrix can prove a compiler mechanism, but it cannot establish the
storage, routing, tail, cache, or publication contract that matters in
production.

The witness record gives every later artifact a join key:

| Identity | Evidence retained |
| --- | --- |
| Reference | Model and parameter format, reference revision, backend, executable identity, and target software. |
| Workload | Input or token sequence, batch and sequence shape, routing distribution, active extents, and important tails. |
| Semantic boundary | Named input and output tensors, logical layouts, physical encodings, included transforms, and all dispatches between them. |
| Executable | Entry point, target, configuration, launch geometry, artifact hash, and compiler version. |
| Data | Captured boundary tensors or deterministic fixtures, parameter bytes, residency policy, and cache-thwarting policy. |
| Timing | Host or device time domain, measured region, instrumentation, batching, warmup, aggregation, and machine-ownership policy. |

Correctness and performance can use different oracles. A scalar
implementation or high-fidelity framework may own numerical truth while an
optimized runtime supplies the performance reference. Agreement between two
backends that share the same graph or quantization mistake is not an
independent numerical proof.

Several workload buckets normally share the witness. Decode, small prefill,
large prefill, non-multiple tails, and both sides of a routing or algorithm
cliff expose different physical schedules. The first bucket keeps iteration
cheap; the others prevent a locally winning schedule from becoming a one-shape
artifact.

## Keep three evidence classes independent

Every candidate produces three kinds of evidence. None substitutes for
another.

| Evidence | Question answered | Typical producers | What it does not prove |
| --- | --- | --- | --- |
| Numerical | Does this cut implement the intended operation? | Small `check.case` differentials, access-sanitized execution, captured boundary tensors, independent scalar references, end-to-end output. | Speed or emitted mechanism. |
| Compiler | What did specialization and lowering infer and emit? | Compile reports, provider-selection details, IR traces, manifests, Low assembly, native code objects, disassembly. | Device latency, bandwidth, or useful overlap. |
| Physical | What did the selected device program do? | Controlled uninstrumented timing, calibrated device timestamps, hardware counters, complete command spans. | Numerical correctness or source intent. |

A green check does not make a kernel fast. Fewer VGPRs do not make a kernel
faster. A short timestamp does not make a kernel correct. The workflow advances
only when the evidence class required by the current gate is present.

## Define the semantic cut before the schedule

A semantic cut names model-level inputs and outputs rather than copying a
reference runtime's current dispatch boundaries. Examples include routed
activations plus packed expert weights to SwiGLU output, or query/key/value
state plus a mask to normalized attention output.

The cut records:

- logical shapes and element types;
- packed storage and alignment contracts;
- dynamic extents and legal tails;
- exact or tolerated numerical behavior;
- overwrite, accumulation, and aliasing behavior; and
- facts known when the JIT specializes the program.

Writes consumed only by the next operation, pure epilogues, and transforms
required only by a generic library ABI identify possible contractions. The cut
remains stable while its implementation changes. It may use one fused kernel,
several kernels, or a command program; physical dispatch count is not part of
the mathematical contract.

## Keep cheap falsifiers beside the source

The authored module retains enough evidence for another agent to change it
without rediscovering the operation:

- algorithm, layout, and ownership comments;
- launch configuration and provider-selection intent;
- small nonzero correctness cases;
- realistic benchmark buckets and tails; and
- target requirements only where the algorithm genuinely depends on them.

Large captured tensors can live in a separate integration fixture. Small
in-source cases make every edit locally falsifiable; captured fixtures prove
the production boundary.

Distinct bindings use distinct values. Gate and up weights, Q and K weights,
routes, scales, and residuals sharing one convenient zero or identity pattern
can allow a swapped binding to pass. All-zero performance inputs are especially
misleading: they can select zero-scale branches, collapse nonlinear work,
reuse zero pages, and turn gathers into cache-hot row-zero probes. Fresh device
allocations also have undefined contents, so correctness never depends on them
happening to contain zero.

The cheap front of the loop uses the public tools:

```shell
loom-format --check kernel.loom

iree-benchmark-loom kernel.loom \
  --benchmark=@production_bucket \
  --dry-run \
  --output=plan.json

iree-test-loom kernel.loom \
  --case=@production_cut_case \
  --device=amdgpu \
  --sanitizer=access
```

Planning catches source, parameter, selection, and workload mistakes without
compiling or touching a device. Access instrumentation checks the physical
memory path after target lowering; the numerical oracle still owns the
mathematical result. The focused [correctness](test-correctness.md) and
[benchmark](benchmark.md) workflows describe case selection, external
libraries, samples, and sanitizer policy.

## Bound the performance regime before hill climbing

Tile and provider searches become useful only after the experiment has a
performance envelope. For throughput, the existing oracle is a proven
achievable floor and a relevant compute or memory roofline is an upper bound.
For latency, the inequalities reverse: the roofline supplies an idealized lower
bound and the oracle is a concrete point to beat. State the unit and direction
instead of saying only that a candidate is “between the bounds.”

The envelope is measured for the exact semantic cut and physical request shape:

| Bound or probe | Question answered |
| --- | --- |
| Selected oracle | What has already been achieved for this production boundary? |
| Traffic accounting plus sustainable bandwidth | What latency floor follows from the minimum required bytes under this access pattern? |
| Operation accounting plus relevant compute throughput | What latency floor follows from the required matrix, dot, vector, or scalar work? |
| Load-only proxy | How much of the current schedule remains when the same addresses and publication path perform only enough observable arithmetic to keep the traffic live? |
| Cache-resident proxy | How much remains when the same compute and schedule reuse operands from the intended cache level instead of streaming them from the production memory tier? |
| Dispatch-only or publication proxy | How much command-processor, launch, synchronization, and output-publication cost exists without the main body? |

The proxy's native artifact must prove that it retained the mechanism it claims
to isolate. A load-only source that the compiler deletes, a cache-resident case
whose working set spills from cache, or an empty kernel that omits the real
publication barrier establishes no bound. Hardware counters and measured
sustainable rates are stronger than a device data-sheet peak whose instruction
type, request form, or residency differs from the kernel.

These probes classify the regime before broad search. If a load-only proxy is
already near the candidate, compute rewrites cannot recover much. If the
cache-resident proxy remains slow, blaming external bandwidth is unsupported.
If the candidate is below the known oracle and far from every relevant
roofline, it is probably missing a structural schedule rather than a locally
better integer tile.

Optimization then has two coupled phases:

1. **Create headroom.** Shorten live ranges, reduce VGPR or LDS allocation,
   remove spills and unnecessary barriers, or reduce unavoidable traffic until
   another residency or issue regime becomes possible.
2. **Fill the headroom.** Add useful independent work, accumulator chains,
   coalesced vector requests, staged prefetch, or producer-consumer overlap so
   the newly available waves and issue slots hide latency.

The first phase can make a report look cleaner while making the kernel slower.
The second can raise pressure while improving throughput. Register count, LDS,
and occupancy are therefore transition constraints, not standalone objectives.

Many schedule changes form one causal bundle. Changing subgroup size to match
an oracle also changes lane identity, fragment ownership, collective scope,
workgroup topology, LDS exchange, and the amount of independent work per wave.
A mechanical subgroup edit tests an incomplete schedule. The candidate record
names the complete bundle required to enter the intended regime and uses
selected partial variants or ablations to expose interactions. “One variable”
means one causal hypothesis, not necessarily one textual edit.

## Workgroup traversal is a schedule

Memory locality has three distinct scales:

| Scale | Authoring choice | Evidence |
| --- | --- | --- |
| Within a subgroup | Lane-to-address mapping and packet width | Coalescing, unique bytes, gaps, and native requests |
| Within a workgroup | Tile shape, shared staging, and fragment ownership | Operand reuse, LDS, registers, barriers, and overlap |
| Between workgroups | Mapping physical workgroup IDs to logical tiles | Repeated operand footprints, cache behavior, and measured traversal variants |

A kernel can be coalesced, spill-free, and well pipelined while repeatedly
evicting the data another workgroup needs next. Changing traversal can improve
cache reuse without reducing FLOPs, issued load bytes, or workgroup count.
Aggregate instruction and memory counts therefore cannot rule out a traversal
win. Higher register use from coordinate arithmetic can also be worthwhile.

For independent `BM x BN` output tiles of `C[M,N] = A[M,K] B[N,K]^T`, a group
of `G` row tiles can sweep output columns while revisiting an input panel of
roughly `min(G * BM, M) * K * sizeof(A)` bytes. The opposite orientation favors
reuse of B. These panel sizes generate hypotheses; other operands, concurrent
work, cache associativity, and actual hardware dispatch order still affect
residency. Mapping nearby physical IDs to nearby logical tiles does not
guarantee their execution order and provides no synchronization contract.

A bounded experiment compares traversal orientations or group sizes while
holding tile geometry, arithmetic, bindings, and launch count fixed. The
mapping remains a bijection, including a final incomplete group and partial
matrix tiles; padded workgroups and duplicate stores would change the
experiment. Exact output comparisons suit an arithmetic-preserving mapping.
Numerical changes retain their separate oracle/error-envelope decision.

Native evidence checks the coordinate transform, tails, register pressure,
spills, and unchanged inner work. Controlled interleaved timing establishes
the benefit at each production shape; the complete consumer then establishes
whether it survives real cache competition. A winning group size on one shape
does not select it for every shape or target. The [benchmark reuse policy](benchmark.md#control-data-reuse)
distinguishes hot-input experiments from streaming and integrated workloads.

Current subgroup-access reports describe lane geometry, not inter-workgroup
reuse distance or physical cache misses. A traversal experiment can therefore
be warranted even when `loom-compile-report suggest` reports no finding and
issued-byte counts are unchanged.

The [Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)
demonstrates grouped program ordering, and [CUTLASS's efficient GEMM guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
discusses threadblock rasterization. They supply algorithmic candidates; their
chosen parameters and cache-level assumptions are not portable tuning results.

## Search laterally across dispatch boundaries

Fusion is one of Loom's highest-leverage searches because the semantic cut is
not tied to an oracle runtime's operator catalog. A producer can publish the
consumer's representation while values are live, eliminating an intermediate
round trip, a command-processor dispatch, host scheduling, and synchronization.
Command programs make the fused and split forms comparable under one authored
boundary.

Dispatch count is not the objective. Fusion can extend live ranges, raise VGPR
or LDS allocation, introduce workgroup barriers, duplicate computation, reduce
residency, or serialize work that previously overlapped. Conversely, when the
runtime and dependency graph permit concurrent execution, an LDS-heavy kernel
that occupies only part of the machine may overlap with an LDS-free kernel and
fill resources that a monolithic fusion leaves idle.

Compare at least these quantities over the same semantic endpoints:

- intermediate and parameter bytes transferred;
- command-processor packets, host submissions, and synchronization edges;
- registers, LDS, spills, residency, and independent work for every kernel;
- complete dependent device span rather than the sum of isolated medians; and
- measured overlap between split dispatches when overlap is part of the
  hypothesis.

The search includes fused, deliberately split, and pipelined forms. A trivial
epilogue fusion may win immediately; a large contraction may be faster as
several complementary kernels. The selected form is the smallest schedule that
uses the machine well across the complete semantic cut, not the one with the
fewest launch records.

## Write the candidate record first

An agent's source edit begins with a compact experiment record:

```text
Production boundary:  the exact witness and shape
Independent variable: the one source, config, provider, compiler, or target change
Hypothesis:           the physical mechanism expected to improve
Compiler consequence: the report or native delta that must appear first
Correctness gates:    local, sanitized, captured, and integrated checks
Discriminator:        the cheapest production shape that executes the new body
Success and stop:     thresholds selected before observing the result
```

Examples of falsifiable compiler consequences include removing two global
materializations, selecting a matrix provider instead of scalar dots,
eliminating full wait drains, reducing a live range enough to cross a residency
tier, or changing the cross-lane transaction shape. “Try another tile size” is
not a hypothesis until it names the mechanism the tile changes.

One causal hypothesis makes the result interpretable. It may be a single edit
or a coupled schedule bundle whose pieces are not independently useful.
Necessary partial variants identify interactions, while a compiler repair and
the kernel candidate remain separate changes whenever the candidate can consume
the repaired compiler through its normal source contract.

## Express loop schedules in the source

Loom's `scf.for` already carries author-selected unrolling and ordinary
read-ahead. Reusable motifs take depth and unroll factor as template arguments,
so each instantiation can choose its schedule. The values can also be calculated
from specialized arguments or target properties. Inside such a motif, the
logical loop uses the supplied `%depth` and `%factor`:

```loom
%result = scf.for %row = [%begin to %count step %step](%sum = %initial : f32) -> (f32) pipeline(%depth) unroll(%factor) {
  %value = view.load %values[%row, %lane] : view<64x32xf32> -> f32
  %next = scalar.addf %sum, %value : f32
  scf.yield %next : f32
}
```

`unroll` expands an exact finite loop; `unroll(%factor)` groups iterations and
handles runtime tails. `pipeline(%depth)` moves ordinary loads and their
prerequisites ahead of ordered computation. Pipelining runs before unrolling,
so depth counts original iterations. Each control also works independently.

Loop-carried views may change extent or layout from one iteration to the next.
Carry the dynamic extent or encoding beside the view and make the loop result
type refer to its sibling result. The initial operand, body argument, yield,
and final result then state the same relationship with their local SSA values.
The [dependent carried-view guide](../guide/functions-and-control.md#change-a-carried-views-shape)
shows the complete spelling and a checked shrinking-view witness. This keeps
ragged windows and progressively consumed pages in structured IR, where normal
unrolling, pipelining, and report workflows can preserve and inspect them.

When the physical pitch itself changes, select or carry an
`encoding.layout.strided` value rather than expanding each candidate into
manual address arithmetic. The [address-layout
guide](../guide/functions-and-control.md#select-and-carry-address-layouts)
shows runtime pitches through selections and loop recurrences.

Streaming reductions, guarded ragged rows, and packed dequantization/dot loops
are useful candidates. Ordinary loads and pure operations may contain nested
`scf.if` and `scf.for`; each structured operation normally stays intact in its
stage. A top-level conditional with reads in one branch may instead keep the
natural guarded recurrence in source: the condition and independent read
closure become a guarded producer, while the carried update remains the
consumer. Read addresses and guards must still be independent of outer carried
state. Pure inner loops may stay in the consumer and use that state. The
[checked guarded-row motif](tune-loop-schedules.md#keep-guards-and-inner-loops-in-the-source)
demonstrates independent inner and outer policies, while the
[cooperative paged-attention motif](tune-loop-schedules.md#pipeline-cooperative-paged-attention)
shows guarded K/V loads, a subgroup reduction, and online-softmax state in one
authored conditional. Fixed-bound tiles can also
read global inputs ahead of ordered workgroup stores, shared reads, and
workgroup-memory barriers. A requested full linear inner unroll can expose
mixed load/store units; read-only reductions and independent schedules retain
their existing shape. The [workgroup-staging example](tune-loop-schedules.md#read-ahead-across-workgroup-staging)
covers publication and reuse of one shared allocation. Global or unknown
writes, global barriers, source-order fences, `scf.while`, and explicit async
groups have different scheduling requirements. Fixed-bound tiles can pipeline
reads into a subgroup or workgroup reduction; the collective stays in the
memory-pure consumer even when it shares a guarded source branch with the loads.
See the [participation contract](../guide/functions-and-control.md#pipeline-reads-ahead-of-ordered-computation)
when the tile contains collectives.
Unannotated loops receive no read-ahead transformation.

An experiment driver may bind these choices with global config keys to make
benchmark sweeps convenient. A library-wide depth key couples every motif
instance; production callers carry the selected values or target-derived
calculation instead. The first experiment compares depth one with a larger depth
while keeping the unroll factor and workload fixed. Correctness includes empty
and short loops, startup boundaries, and partial-unroll remainders. Detailed compile reports
retain the chosen depth and producer/consumer schedule; `suggest` exposes
`scf.compare_pipeline_depth` with available final resource costs. Registers,
spills, occupancy, code size, compile time, and measured runtime decide whether
the extra live state is useful.

On AMDGPU, `amdgpu.pipeline_copy_waits` points to full waits at materialized
branch-payload copies. Check whether those blocks are steady backedges before
changing the schedule. An explicit larger unroll factor with
`schedule(recurrence)` can expose register reuse that carries pending loads
across the backedge; confirm the native moves and wait counts as well as the
source schedule. The cited outstanding count is block-local; zero can still
represent a required residual counter-epoch or control-flow hazard.

The [loop-tuning walkthrough](tune-loop-schedules.md) supplies a vector-row motif
with per-instance policies, checked row-sum and packed-dot experiment harnesses,
configuration sweeps, and actual `show`/`suggest` output. The
[per-instance search](search-loop-schedules.md) composes two independent motifs,
checks candidate identity, and inspects resource cliffs before device execution.
For shared-input kernels, the
[grouping and depth comparison](tune-loop-schedules.md#choose-grouping-and-depth-independently)
separates load reuse from available subgroup parallelism, with different policy
winners across workload sizes and GPU targets.
The [control-flow guide](../guide/functions-and-control.md#unrolling-is-a-loop-policy)
owns the exact policy and schedule semantics.

For dependent route/page lookups, compare metadata and payload lead distances
separately. The checked
[routed-row example](tune-loop-schedules.md#separate-route-and-payload-lookahead)
uses ordinary SSA queues to keep a row ID farther ahead than its weighted
payload. Its matched unrolled control, numerical cases and named benchmark rows
make source distance, native overlap and measured speed separate decisions.
For cooperative attention, the
[paged-attention example](tune-loop-schedules.md#pipeline-cooperative-paged-attention)
pipelines K/V fragments into subgroup score reductions while preserving shared
page identity, ordered softmax/PV state and ragged tail guards. Its independent
analytic checks and matched benchmarks provide a starting point for schedule
searches without hand-building the input queue.
For top-k token lists, the
[sparse attention example](tune-loop-schedules.md#pipeline-sparse-token-attention)
separates the selected-prefix guard from physical-ID validity, with independent
numerical checks, short allocations and poisoned inactive payloads. Preserve
both boundaries when searching schedules for an indexer-selected workload.
For grouped query heads, the
[shared-loading example](tune-loop-schedules.md#share-kv-loads-across-query-heads)
reuses K/V fragments while keeping separate queries, lengths and softmax states.
Its independent, shared-serial and shared-pipelined callers separate reuse from
scheduling: fewer loads can cost more live state and less subgroup parallelism.

For authored native motifs, give a Low helper
[`schedule(phased)`](../guide/functions-and-control.md#compose-independently-scheduled-helpers)
and place `low.schedule.phase` separators between its instruction phases. The
first phase is implicit, and SSA values remain visible across separators.
Callers use ordinary `low.invoke`; inlining and loop cloning preserve each
invocation's independent schedule. These authored phases impose instruction
order without memory waits. Compare native waits and register use along with
`scope_count`; a larger overlap window may cost more live registers. Native
AMDGPU and x86 enforce the contract, while intermediate representations reject
it when they cannot guarantee final instruction order.

Repeated shared-memory reuse has a separate authoring choice. A full
`kernel.barrier` keeps rendezvous at one source point; a matching
`kernel.barrier.arrive` and `kernel.barrier.wait` pair can expose independent
private work after the last shared read and before the next overwrite. Keep the
portable algorithm behind one template contract, select a split provider only
for targets with a native realization, and retain a complete-barrier fallback.
The [checked split-barrier workflow](tune-loop-schedules.md#overlap-private-work-with-shared-tile-release)
shows the source shape. In `loom-compile-report show`, confirm the complete,
arrive, and wait plan keys and their dynamic counts. In native output, confirm
read completion, signal, useful work, wait, and overwrite in that order, then
compare registers, spills, residency, code size, and measured runtime with the
full-barrier control.

## Ask the compiler before asking the GPU

The baseline and candidate compile under the same root, workload,
configuration, backend, and target identity:

```shell
loom-compile kernel.loom \
  --root=@production_cut \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx1151 \
  --output=candidate.hsaco \
  --compile-report=summary \
  --compile-report-output=candidate.report.json

loom-compile-report show candidate.report.json
loom-compile-report suggest candidate.report.json
loom-compile-report diff baseline.report.json candidate.report.json
```

The report answers whether the candidate changed the intended mechanism:

- which provider and source-to-Low plan were selected;
- which loop-carried aggregates were decomposed, deliberately preserved, or
  rejected at their source boundary;
- which packed, vector, matrix, memory, and synchronization families remain;
- scheduled pressure and final register allocation;
- LDS, private memory, spills, and materialized reloads;
- modeled residency thresholds and limiting resources; and
- code size, request shape, waits, and barriers.

Residency is constrained by all tied resources, not whichever scalar limiter
appears first. Use the per-resource explanation and joint next-tier requirements
before prioritizing a VGPR or LDS reduction. Unknown counts or launch geometry
are missing evidence, not an assurance that a resource is nonlimiting. A higher
modeled tier earns a benchmark experiment rather than proving a performance win.

An empty suggestion list means only that registered target diagnostics found
no issue. It does not prove that the schedule matches an external oracle or
that the hardware will prefer it.

For a loop carrying a logical vector bank, inspect **Source boundary
projections** before manually expanding the state. A selected row proves that
the compiler already split fixed components; a preserved whole-value row may be
the intended native fragment representation; and a rejected row names the
access or transport condition that blocked decomposition. Source suggestions
for dynamic, mixed-shape, and incompatible whole-bank uses are experiment
proposals. Recompile and compare their final resource and runtime evidence
before retaining the rewritten form.

When the expected delta is absent, the candidate returns to source or becomes
a standalone compiler reproducer. Repeated physical timing cannot make a
missing compiler mechanism appear. Detailed reports, source-to-Low rows, IR
snapshots, and native disassembly enter only after `show`, `diff`, or `suggest`
has narrowed the question. The [compile-report workflow](compile-reports.md)
owns those evidence paths and the strict comparison identity.

## Earn physical measurement

A candidate whose correctness and compiler gates pass receives one short pilot
at the cheapest production shape that executes its changed static body. Wider
shape sweeps and long locked runs are earned by entering the predeclared parity
band, proving a substantial traffic reduction, or resolving a noisy boundary
near the stop threshold.

Two candidates in one module can run under an interleaved policy:

```shell
iree-benchmark-loom candidates.loom \
  --device=amdgpu \
  --compare=@baseline,@candidate \
  --sample=0 \
  --interleave=ABABA \
  --repetitions=2 \
  --measure=dispatch_complete \
  --output-format=jsonl \
  --artifact-bundle-dir=candidate-evidence \
  --output=comparison.jsonl
```

The experiment retains the optimized executable identity, exact workload,
batch size, input-binding ring, data-residency policy, warnings, raw process
rows, and machine-ownership state. Rotating enough independent bindings to
exceed the relevant cache prevents a small model slice from masquerading as a
streaming workload. A deliberate cache-hot experiment is valid when it is
labeled and compared with the same reuse contract.

The score's time domain is part of its identity:

- synchronous host completion includes submission, execution, and completion;
- an enclosing device interval measures one explicitly timestamped replay;
- per-dispatch profiling attributes work inside an instrumented replay; and
- hardware-counter collection can perturb both scheduling and elapsed time.

An instrumented replay is not automatically the device portion of the ordinary
host-timed replay. Retained profile metadata may add timestamp packets,
barriers, flushes, fixups, or harvest work. Same-replay device time must be
contained by the synchronous host interval that encloses it. Separate replays
have no subtraction or containment relationship and require a paired
perturbation calibration before supplying a cross-backend score.

The [benchmark workflow](benchmark.md) defines batching, data reuse,
interleaving, structured results, and profile interpretation in detail.

## Reject attractive non-evidence

Several recurring observations suggest experiments but cannot select a winner:

| Observation | Missing mechanism |
| --- | --- |
| The reference uses wave64. | Wave ownership, lane cohorts, physical workgroup shape, collectives, fragment layout, LDS exchange, and independent work still need derivation. |
| The candidate uses fewer registers or has higher modeled occupancy. | Occupancy is a constraint and cliff detector; latency hiding and useful independent work may have fallen with pressure. |
| The native listing is shorter. | Dynamic requests, waits, dependencies, and cache behavior determine whether removed instructions were limiting. |
| Tile geometry and issued bytes are unchanged. | A different workgroup traversal can change cross-workgroup cache reuse and physical memory traffic. |
| One lane issues wider packets. | Neighboring lanes may now touch distant addresses and destroy wave-level coalescing. |
| The source resembles the oracle's loop. | Address forms, clauses, allocation, waits, and the runtime-selected dispatch can still differ. |
| A profiler reports precise timestamps. | Precision does not establish replay identity or low perturbation. |
| One member of a coupled schedule loses. | An incomplete subgroup, ownership, fragment, LDS, or overlap change may not enter the intended regime; test the declared bundle and discriminating ablations. |
| The fused form launches fewer kernels. | Pressure, barriers, lost concurrency, and the complete device span determine whether removed dispatches were useful. |

A losing candidate still has value when it falsifies one of these mechanisms.
The experiment record states the corrected explanation and removes the more
complex schedule from production source. A clean report metric is never a
reason to retain source that did not improve the physical objective.

## Integrate before making the model claim

Independent kernel wins are not automatically compositional. The selected cut
is replayed with real captured activations, production parameter bytes and
layouts, explicit dependency barriers, and a timestamp spanning its first
dispatch through final publication. This exposes transient traffic, hidden
transposes, route imbalance, missing synchronization, and gaps between
dispatches.

The integrated result distinguishes:

- the sum of individual dispatch times;
- the complete dependent device span;
- host completion and submission time; and
- an end-to-end application result such as first-token latency.

Holding unported work at reference parity can produce an Amdahl projection.
That remains a projection until the complete layer or model executes. Command
programs make this integration boundary authorable in the same language as the
kernels, but they do not relax the evidence contract.

## Reduce compiler blockers to standalone packets

Model context is usually noise at the compiler boundary. A durable blocker
packet contains:

- one minimized `.loom` reproducer;
- the exact public command;
- expected and observed behavior;
- compiler and target provenance;
- the relevant report, IR snapshot, native artifact, or disassembly; and
- a correctness case or benchmark only when execution is required.

The desired source survives beside any temporary model workaround. It becomes
the regression witness for the compiler repair. Unsupported generic behavior
is fixed in the compiler or target family rather than hidden by a model-only
mnemonic or silent fallback.

## Hand off evidence, not a transcript

An agent handoff records the production witness, selected source and artifact
identities, passing correctness gates, exact benchmark commands, baseline
reports and results, rejected mechanisms, the active hypothesis, and the next
decision gate. The source, cases, reports, and raw result bundle carry the
proof; the handoff explains how to navigate it.

This makes resumption cheap without promoting an agent's prose into evidence.
Another agent can reproduce the selected state, falsify the active hypothesis,
and continue from the same boundary instead of trusting a narrative summary.

## Use external oracles for bounded questions

External compilers and runtimes can make one part of the loop independently
observable. Each oracle keeps its own selection, provenance, and evidence
boundary:

- [RADV and Vulkan](oracles/radv.md) join an optimized Vulkan dispatch to its
  selected SPIR-V, ACO native program, launch geometry, and score;
- [LLVM MC](oracles/llvm-mc.md) independently qualifies selected native packet
  encodings;
- [GGML and llama.cpp](oracles/ggml-llama-cpp.md) preserve model, quantized
  storage, graph, runtime-selection, and semantic-cut contracts during a port;
  and
- [Native schedule reconstruction](oracles/native-schedule.md) preserves an
  executable Low oracle while raising its behavior into default-pipeline source
  and minimizing the first compiler divergence.

An oracle answers the question named by its page. It does not become Loom's
source language, force its dispatch boundaries into a model program, or waive
the numerical, compiler, and physical gates around the answer.

## Compact loop

For each candidate:

1. Name the production semantic cut, workload, oracle, and relevant rooflines.
2. Use mechanism probes to classify the current compute, memory, or control
   regime.
3. Run in-source correctness and captured boundary checks.
4. State one causal hypothesis, including every coupled change required to
   enter the intended regime.
5. Compile and inspect `show`, `suggest`, strict `diff`, and the expected native
   delta.
6. Follow evidence into detailed IR or ISA only when the mechanism differs.
7. Benchmark one discriminating production shape.
8. Search across dispatch boundaries and sweep realistic shapes, tails, and a
   second target only after the pilot wins.
9. Integrate the cut and measure the complete dependent span, including useful
   overlap.
10. Keep the simplest schedule supported by all three evidence classes and
    preserve rejected mechanisms and standalone compiler blockers.

The result is more than a fast artifact. It is a source family another agent
can understand, specialize, validate, and improve without rediscovering its
numerical contract or inventing a new measurement policy.
