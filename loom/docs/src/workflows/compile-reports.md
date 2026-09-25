# Read compile reports

A Loom compile report joins emitted artifact facts with compiler analysis for
one compilation identity. It is the first place to answer questions about code
size, instruction mix, memory traffic, register pressure, residency, spills,
and the source-to-Low decisions that produced them.

`loom-compile-report` provides three bounded views:

- `show` presents one report without requiring knowledge of its JSON schema;
- `diff` compares one controlled change under an explicit identity contract;
- `suggest` uses source policies and target evidence to propose experiments.

Raw JSON remains available after one of those views identifies the exact
evidence that needs deeper inspection. The scenario-indexed field and `jq`
recipes live in [Query compile evidence](compile-report-queries.md).

## Capture one report

Request a structured summary beside the emitted artifact:

```shell
loom-compile kernel.loom \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx11-generic \
  --output=kernel.hsaco \
  --compile-report=summary \
  --compile-report-output=kernel.report.json
```

`summary` is the routine tuning product. `details` adds row-level provenance
for source selections, schedule bands, pressure origins, allocations, spills,
waits, target capabilities, and other focused investigations. Text report modes
exist for direct terminal output, but structured JSON is the composable input
to report tools and automation.

Compile reports are version-zero compiler evidence. Analyze and compare reports
with the same Loom tool version that produced them. A schema mismatch fails
with an instruction to regenerate the report rather than guessing at an old
layout.

## Show one report

Start with the target-neutral human view:

```shell
loom-compile-report show kernel.report.json
```

The header states artifact, target, specialization, compiled launch geometry,
configuration, and compile status. Launch fields appear only when the compiled
artifact fixes them; a missing workgroup count is the normal result for a
workload-dependent launch function, not missing benchmark evidence. Each entry
then separates two evidence classes:

- **Artifact facts** come from the emitted code or target-owned artifact
  inspection: code bytes, instruction counts, instruction families, and
  physical memory usage.
- **Compiler analysis** comes from the prepared program and target model:
  scheduled pressure, compiled execution economics, residency, allocation, and
  scheduling evidence.

Compiled execution economics never require a static dispatch size. Exact
fixed-trip multiplicities are reported per workitem and, when the artifact
fixes local workgroup size, for one workgroup. They count target-Low operation
effects and descriptor-effect widths, not native instruction issues, memory
transactions, or executed branch paths. Control flow is a statically reachable
block envelope: data-dependent alternatives may both contribute. Whole-
dispatch geometry belongs to a concrete benchmark launch unless the compiled
artifact genuinely specializes it.

Register-move analysis includes compiler-classified causes. Summary reports
preserve the aggregate packet and register-unit counts; detailed reports let
`show` enumerate causes and let `diff` distinguish added, removed, changed, and
unchanged causes. This separates a source-level repacking change from a
target-created operand-bank repair instead of treating both as generic moves.

Residency analysis in both summary and detailed reports explains each resource
constraint, not just the first limiting resource. It shows final usage, rounded
allocation, allocation granularity, pool scope, and each resource's independent
wave ceiling. A resource can use many registers without currently constraining
residency; a fixed workgroup-slot ceiling cannot be removed by reducing register
or LDS usage at the same launch shape.

When resources tie, the next-tier requirements are joint. For example, on
gfx1151 a wave64 kernel with 256 workitems per workgroup, 88 VGPRs and 15,616
LDS bytes requires **both** at most 84 VGPRs and at most 14,336 LDS bytes to
move from eight to nine modeled waves.
Reducing only one leaves the other limit in place. Missing final register counts
or an unknown workgroup size make exact residency unavailable; the report keeps
the known resource facts without inventing a complete transition. These are
target-model constraints, not a throughput prediction. Recompilation checks the
candidate and its new constraints; benchmarking establishes whether it helps.

Detailed reports also retain target-neutral native contraction and fragment-
transition facts selected for each source-to-Low row. `show` presents the
native tile and participant count, per-role register and payload placement,
logical ownership multiplicity, and any source-owner movement required by a
fragment transition. Movement counts remain unscaled structural facts: a
cross-participant result such as `256/512 (50%)` describes destination
positions in one selected native transition, not dispatch executions or a
hardware cycle estimate. The selected target recipe and descriptor are shown
separately from the coordinate evidence that explains them.

Coordinate-bearing physical positions are reported separately from payload
elements. Packed or strided carriers need not assign an independent logical
coordinate to every payload slot, so multiplying payload elements by
participants can overstate the ownership domain.

Exact transitions include readable equations over the target-neutral physical
dimensions `participant` and participant-local `position`. A relation such as
`source.participant = participant % 16 + (position % 2) * 16` states precisely
which source owner supplies each destination position without exposing a
backend instruction mnemonic or requiring the viewer to load target tables.
Metadata-dependent, parametric, and opaque placements retain their evidence
class and do not claim exact ownership.

Fragment memory reports also preserve the producer-proven address relationship
across a complete subgroup. `show` prints each authored memory source and its
packet variants with the per-lane packet width, exact lane-offset formula, and
the byte-interval union for the subgroup. Requested bytes count every lane's
request, unique bytes count their union, and span bytes cover the lowest through
highest requested address. Dense, gapped, and overlapping are exact geometry
facts; they are deliberately not presented as cache-line transaction or cycle
predictions. When the compiler cannot prove a complete subgroup, uniform
dynamic base, uniform control, or packet width, the missing proof and its reason
remain visible instead of manufacturing geometry.

Explicit loop pipeline policies appear under **Source loop pipelines**. Each
policy records its compiled function, loop ordinal, applied depth, queue shape,
and ordinary read count. Detailed reports include each source operation's
producer or consumer stage and its lookahead in original iterations. Depth one
records the author's serial policy. Unannotated loops produce no pipeline rows.
Loop ordinals distinguish applied policies within a function; they are not
source locations or stable identifiers across arbitrary source edits.

The [loop-tuning walkthrough](tune-loop-schedules.md) follows runnable row-sum
and packed-dot kernels from `pipeline(...) unroll(...)` source through these
schedule rows, `suggest`, and matched serial controls.

Loop-carried aggregate decisions appear under **Source boundary projections**.
Each row identifies the source function, loop ordinal, recurrence-column
ordinal, logical type, and one of three outcomes:

- `selected` means the compiler replaced the aggregate recurrence with
  homogeneous scalar or tail-vector components;
- `preserved` means the authored whole value remains the right representation,
  including native fragments and untouched banks; and
- `rejected` means component accesses attempted a decomposition but could not
  form one valid boundary representation.

Summary reports retain the three outcome counts. Detailed reports add the
source and component shapes plus a stable reason such as
`static_component_accesses`, `whole_value_use`,
`non_static_component_access`, or `inconsistent_component_access`:

```text
Source boundary projections
  selected=1 preserved=0 rejected=0
  update_vector_bank scf.for[0] loop_state[0]: loop-vector-bank selected vector<4x4xf32> -> 4 x vector<4xf32> reason=static_component_accesses
```

The operation and state ordinals identify the compiled source structure; they
are not source locations and may change after an arbitrary source edit. The
[vector-bank guide](../guide/vectors-and-structured-compute.md#carry-logical-vector-banks-through-loops)
shows the source form that produces this decision.

Unavailable fields are omitted instead of rendered as zero. That distinction
matters: zero instructions is a measurement; no target inspector for that
metric is an absence of evidence.

Automation can request the same bounded view as deterministic JSON:

```shell
loom-compile-report show kernel.report.json --format=json \
  >kernel.view.json
```

The view is smaller and more stable for dashboards and agents than the complete
compiler report.

## Diagnose LDS bank conflicts

For an AMDGPU kernel using workgroup memory, capture a `details` report and run
`loom-compile-report show kernel.report.json`. The **Bank service** view connects
source loads and stores to the selected LDS instruction and its lane-service
model. Here, a report "packet" is one selected load/store instruction, such as
`ds_write_b128`. Static totals count each instruction site once, regardless
of how often its enclosing loop executes. Coverage divides those instructions
into three categories:

- **Exact** instructions have proven addresses and active lanes under a named
  model.
- **Unknown** instructions have a model, but an address, alignment, or
  participation proof is missing. Each source group names the missing proof.
- **Unmodeled** instructions have no model for the selected target, access width,
  and wave size. A missing model is not evidence of conflict-free access.

Model selection respects the function's execution width. Silicon-calibrated
models cover `ds_read_u16`, `ds_write_b16`, and b32/b64/b128 reads and writes on
gfx1100/gfx1151 in wave32 and wave64, and gfx942 in wave64. The gfx1100/gfx1151
models also cover the partial-register `ds_load_u16_d16` and
`ds_load_u16_d16_hi` reads in both wave sizes. Documented CDNA3 b128 wave64
models cover gfx940/gfx941. The gfx1250 wave32 model is explicitly an
unvalidated vendor software model. Other gfx11 processors, gfx1200/gfx1201,
unsupported wave modes, and other access widths report unmodeled coverage. A
shared bank count alone does not establish shared service rules.

Reads and writes can have different lane-service groups, and repeated reads
can broadcast. AMD's [LDS bank-conflict explanation](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
describes the CDNA3 b128 groups; the
[ROCm programming guide](https://rocm-handbook.amd.com/_/downloads/amd-rocm-programming-guide/en/docs-7.2.3/pdf/)
describes identical-address broadcast. Wide-access analysis requires proven
alignment and full-subgroup participation. Fragment accesses use their compiled
lane/register layout, including repeated lane addresses.

Source accesses can combine multiple workitem coordinates and subgroup-uniform
offsets. The analysis uses the native X-fastest workitem order and checks every
wave in the workgroup. For example, on gfx1100/gfx1151, a b128 store at
`16*x + 512*y` is conflict-free for a `32×2` wave32 workgroup. Changing the shape
to `4×8` puts two rows in each write-service phase and doubles the required
rounds. A `12×8`
shape has different profiles across waves and reports
`address-wave-profiles-differ`; no single wave's profile represents it exactly.

Constant division, remainder, shift, and mask can also describe tiled
coordinates. For example, the b128 store address
`144*(x/8) + 16*(x%8)` is conflict-free across a 128-thread wave32 workgroup
on gfx1100/gfx1151. Replacing `x` with `x+1` inside both digits doubles the
required rounds. An offset inside division changes lane grouping; it is not
just a common translation of the final addresses. These proofs require
nonnegative, nonwrapping arithmetic and constant divisors. Unproved varying
terms, runtime coordinate strides, and relationships lost across control-flow
arguments remain unknown.

The b64 models use contiguous 16-lane service groups on the qualified devices.
For example, a wave32 b64 access at `8*lane` needs two uncontended rounds.
Changing the lane stride to 128 bytes maps all sixteen lanes in each group to
the same two banks, requiring 32 rounds: 30 extra rounds per instruction.
Repeated reads of the same address still need only the uncontended rounds.

Halfword and word accesses use contiguous 32-lane service groups.
Halfword reads to either half of a bank word share a request; writes to disjoint
halves also combine. Distinct words mapping to the same bank still conflict.
The model reports `packet_bytes` separately from `bank_word_bytes` so a two-byte
access retains its subword identity.

Packed fragment loads can fill a register with separate low- and high-half
reads. Each instruction gets its own model and address proof. For example,
halfword reads at `2*(lane%16)` repeat eight bank words and need only two
uncontended rounds per instruction in wave64. Filling the other register half
does not change which LDS banks serve the read.

Subword placement matters even with a fixed lane layout. For example, two
16-halfword spans separated by 96 bytes are conflict-free at a four-byte-aligned
base. Moving the base by two bytes makes the spans touch distinct words of bank
zero. The report retains static offsets and dynamic divisibility, evaluates
compatible byte-base residues, and reports `address-base-residue-unproven` when
the possible placements have different phase profiles. Full-word translations
only rotate bank indices and need no enumeration.

Required and extra **service rounds** describe proven static instructions under
the reported model. They are not measured cycles, wall-clock time, or a predicted
speedup. Dynamic totals include only instructions with proven execution counts;
unresolved loop counts remain unknown. The model's provenance is separate from
the address proof: `exact` under an unvalidated model is still experimental.

Use `loom-compile-report suggest kernel.report.json` to find proven conflicting
groups, ordered by extra static service rounds. A finding names the source
buffer and instruction, compares required and uncontended service, and gives
the current bytes per lane. This order prioritizes structural layout
experiments; a rarely executed tail can rank above a frequently executed loop.
Runtime frequency and measured time determine which experiment matters most.

Findings also identify unknown and unmodeled accesses to the same buffer,
including other load and store forms. When the compiler records an LDS growth
limit, the finding states how much padding fits before the next modeled
residency drop, holding the launch and other resource counts fixed. Missing
growth limits remain unavailable. Unvalidated-model suggestions require
`--include-experimental`.

For example, 32 wide reads might require 2,048 service rounds versus 256
uncontended: an eightfold structural service requirement, with 1,792 extra
rounds. If the report also records 27,904 bytes of LDS and a residency drop at
32,769 bytes, there are 4,864 bytes of growth before that cliff. This supports
trying a padded row pitch while preserving the 16-byte instruction width and
updating both producer and consumer views. It does not establish a particular
pitch: the physical layout, every access direction, and alignment still need
qualification. Saving space in another staging buffer can also make room for
padding, so compare the combined footprint as well as each individual change.

Compare an authored pitch, padding, or lane-mapping change with
`loom-compile-report diff baseline.report.json candidate.report.json`. The diff
reports service changes and proof loss independently. A lower conflict count
accompanied by more unknown or unmodeled instructions does not demonstrate an
improvement. Evaluate both producer stores and consumer loads, then check
register pressure, LDS footprint, residency, and native execution time before
selecting the layout. Instruction scheduling can overlap service with other
work; it cannot remove a conflict within one LDS instruction.

## Diff one controlled change

Capture the same root, configuration, target, and workload before and after one
source or compiler change, then compare them:

```shell
loom-compile-report diff \
  baseline.report.json \
  candidate.report.json
```

The default `exact` contract requires the recorded compilation identity to
match: schema, mode, artifact kind and format, backend, module, function,
target, specialization, configuration, workload, and entry set. A mismatch is
an error rather than a misleading delta. These fields do not fingerprint source
contents or the compiler binary; a reproducible experiment records those
identities alongside the reports.

```shell
loom-compile-report diff \
  baseline.report.json \
  candidate.report.json \
  --format=json \
  >candidate.diff.json
```

An intentional configuration sweep changes that identity, as can historical or
externally produced artifacts. `--force` can compare reports containing exactly
one compiled entry each:

```shell
loom-compile-report diff \
  historical-baseline.report.json \
  historical-candidate.report.json \
  --force
```

The forced view preserves both report and entry identities, lists every field
that violates the exact comparison contract, and labels the deltas as
observational rather than causal. Reports with more than one entry remain an
error because entry order is not a semantic pairing contract. Use the strict
default when the recorded context is held fixed. A
[per-instance schedule search](search-loop-schedules.md) uses `--force` to inspect
an intentional policy change while retaining the fixed source, compiler, numeric
contract, and workload. The tool exposes the mismatch; the experiment establishes
which variable changed.

Wave memory diffs group packet and strategy variants under the stable authored
source identity. A scalar-to-wide lowering therefore appears as removed and
added variants of one source even when the number of packets changes. The tool
does not invent one-to-one matches between ambiguous variants, and it retains
the complete before and after geometry so widening, lane dispersion, and
overlap can be evaluated together.

For a forced single-entry comparison, the explicit entry pair also aligns wave
memory sources whose function was renamed. Packet variants still require exact
semantic identity; the forced pair does not authorize guessed variant matches.

Native-layout diffs use the retained function and `source_low` row index as the
source-row identity. They compare normalized placement, movement, and factor
fields directly; recipe-key spelling is reported as a separate selection
change. If an edit shifts or removes the source row, the evidence appears as
removed and added instead of being paired heuristically with another operation.

Wait-reason diffs compare counter/reason groups independently of the aggregate
entry totals. A schedule change can therefore keep the same number of wait
actions while replacing full drains with partial waits or moving a dependency
between ordinary, loop-entry, and loop-carried reasons.

## Compare target specialization

Target comparison deliberately relaxes only the selected target specialization
fields while retaining the backend and target family:

```shell
loom-compile-report diff \
  gfx11-generic.report.json \
  gfx1151.report.json \
  --comparison=target
```

This answers how the same program and configuration change when specialized
within one target family. It does not make AMDGPU and SPIR-V reports comparable,
or allow different workloads and configuration bindings to masquerade as a
target delta.

## Choose an experiment

Suggestions interpret the applied source policies and the selected target's
report evidence:

```shell
loom-compile-report suggest kernel.report.json
```

Each finding names an action and cites the exact evidence paths and values that
motivated it. The output is a prioritized experiment queue, not an assertion
that a transformation will improve performance. Recompile, retest, and measure
each accepted experiment.

For an explicitly pipelined loop, `scf.compare_pipeline_depth` suggests comparing
a smaller depth while holding the unroll factor and workload fixed. It cites
the applied policy and available final registers, spill count, occupancy,
private memory, and code size for the matching compiled entry. The queue size
counts SSA values, which can occupy several physical registers or share storage;
it is not a physical register count. A single report establishes pipeline use
and final resource consumption. A matched compilation and runtime comparison
establishes the change in cost and performance.

On AMDGPU, `amdgpu.pipeline_copy_waits` identifies full global-load waits whose
native consumers are materialized branch-payload copies. Its cited
`outstanding_before` values count packets in each scheduled block rather than
the whole hardware counter. A zero local count can still represent a required
residual counter-epoch or control-flow hazard; it does not establish that the
wait is redundant.

Rejected vector-bank projections produce source experiments only when the
reason identifies a concrete rewrite:

- `vector.compare_static_bank_access` compares a statically unrolled or
  explicitly carried form when a small bounded component is selected
  dynamically;
- `vector.compare_uniform_bank_components` compares one component shape when
  the source mixes prefix ranks; and
- `vector.compare_componentwise_bank_state` compares a componentwise consumer
  when one use still requires the whole active bank.

Each suggestion asks the author to verify that projection became selected and
then compare registers, spills, occupancy, code size, compile time, and measured
runtime with the workload and schedule held fixed. Selection enables that
experiment; it does not establish a gain. Expected whole-value fragments,
untouched banks, very large decompositions, peer rejection, and generic
transport rejection remain visible in `show` without speculative advice.

Nested `scf.if` and `scf.for` appear as intact operations in the reported
producer/consumer schedule. The read count includes static load operations
inside their regions, including alternative branches; it is not a count of
dynamic memory transactions. Independently pipelined inner loops have their
own policy rows. Their transformed bodies then participate in the enclosing
loop's schedule.

This source advice works across target families. When target-specific advice is
unavailable, the result retains its reason in `target_unavailable_reason` while
still showing source findings. A source helper's schedule remains inspectable
without assigning it an entry's resource costs unless the report supplies the
matching function identity.

Residency suggestions cite every required resource reduction with its units and
allocation scope. A joint suggestion appears only when the compiler retained
all limiting requirements for that next tier. No footprint-reduction suggestion
is produced at a fixed launch ceiling or when exact transition evidence is
missing.

The experimental AMDGPU fragment-packet finding requires exact wave geometry
for every cited scalar packet row. It reports per-lane width together with
cross-lane coverage, unique bytes, span, maximum gap, and maximum adjacent-lane
delta. Packet width is not treated as a monotonic objective: wider candidates
that worsen dispersion require a hardware timing win, rather than receiving a
recommendation merely because they reduce static packet count. When the same
packets coincide with target-created operand-bank materialization, the finding
leads with allocator placement and asks the reader to verify that the repairs
belong to those loads before changing the memory hierarchy.

```shell
loom-compile-report suggest kernel.report.json --format=json \
  >kernel.suggestions.json
```

Default findings require the provider's high-confidence evidence tier.
Exploratory policies and experiments derived from structurally exact but
hardware-unvalidated models are opted into explicitly:

```shell
loom-compile-report suggest kernel.report.json \
  --include-experimental
```

The output preserves the confidence class and evidence for every finding, so
automation can keep high-confidence and exploratory searches distinct.

## Follow one finding into details

Regenerate a detailed report when `show`, `diff`, or `suggest` identifies a
specific question whose answer needs row-level provenance:

```shell
loom-compile kernel.loom \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx1151 \
  --output=kernel.hsaco \
  --compile-report=details \
  --compile-report-output=kernel.details.json
```

For example, inspect the values live at a register-pressure peak:

```shell
jq '.pressure_origin_rows.rows[]? |
    {function, register_class, peak_block, peak_operation,
     origin, semantic_tag, live_units, live_values} |
    with_entries(select(.value != null))' kernel.details.json
```

Or group materialized spill evidence by source origin and semantic tag:

```shell
jq '[.spill_rows.rows[]?] |
    group_by(.origin + ":" + (.semantic_tag // ""))[] |
    {function: .[0].function,
     origin: .[0].origin,
     semantic_tag: .[0].semantic_tag,
     spill_count: length,
     bytes: (map(.byte_size) | add),
     stores: (map(.store_count) | add),
     reloads: (map(.reload_count) | add)}' kernel.details.json
```

The focused tool supplied the question; the JSON path retrieves its provenance.
This is materially different from scanning an undifferentiated disassembly and
trying to recover which source decision produced it.

Pressure, allocation, scheduling, wait, memory, target-capability, and IR-trace
queries are collected in [Query compile evidence](compile-report-queries.md)
instead of being repeated in every tool's help.

## Relate reports to benchmark results

`iree-benchmark-loom` can retain the same compile-report tree beside timing
evidence. Debug and full artifact bundles write per-candidate report sidecars;
JSONL compile rows identify candidates so report, correctness, profile, and
benchmark events can be joined.

Timing answers whether a candidate changed performance. A report diff answers
which emitted or modeled compiler properties changed. Neither substitutes for
the other: lower register pressure without a timing improvement is useful
compiler evidence, not a performance claim.

The productive loop is:

1. establish a passing checked case and a representative benchmark;
2. capture a baseline score and compile report;
3. change one source, configuration, provider, pass, or target variable;
4. retest correctness and collect a candidate score;
5. diff comparable reports and inspect the timing interpretation;
6. use suggestions or detailed rows to choose the next experiment.

That loop preserves the path from source fact to compiler decision to native
artifact to measured behavior without making every ordinary compile dump every
intermediate.
