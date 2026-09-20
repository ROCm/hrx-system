# Checked Benchmark Corpus

This corpus contains complete Loom programs with correctness cases and
benchmark rows. Target providers compose these data-only sources into
capability-coherent modules and execute them through production-facing
infrastructure. A case belongs here when the important risk is the interaction
between multiple compiler and runtime contracts, and that interaction cannot
be represented faithfully by a smaller source-low, op, pass, tool, or authoring
test.

Each checked benchmark source is expected to contain:

- One or more kernels or callable entry points under test.
- One or more `check.case` records with deterministic input generation and
  meaningful expectations.
- One or more `check.benchmark<@case>` rows selecting the checked workload for
  timing.
- Comments that identify the represented behavior class and any non-obvious
  numeric, layout, target, or launch contract.

Fast-math flags are part of the benchmark contract. Cases that model ML-fast
math should spell that policy in source and mark the contractable add/sub/reduce
operations that backend and oracle comparisons are allowed to fuse. Cases that
intentionally have no contraction opportunity should say that too, so missing
FMA-style instructions are not misclassified as backend losses.

The source stays more test-focused than the authoring corpus. Authoring
examples are comment-rich recipes for people and agents learning how to write
Loom. Checked benchmark cases are regression and benchmark assets: they can
have comments, but their main job is to preserve end-to-end signal.

## Case Template

```loom
kernel.def @semantic_behavior_class() {
  // Launch geometry that is part of the represented behavior.
} launch(%input: buffer, %output: buffer) {
  // Full source pattern under test.
  kernel.return
}

check.case public @semantic_behavior_class_case {
  // Deterministic inputs that distinguish the represented behavior.
  kernel.launch @semantic_behavior_class(%input, %output)
      : [](tensor<...>, tensor<...>)
  check.expect.close actual(...) expected(...) atol(...) rtol(...) nan(...) : ...
  check.return
}

check.benchmark<@semantic_behavior_class_case> @semantic_behavior_class_smoke
```

The template is intentionally small. A real case should add parameters and
multiple benchmark rows only when those shape classes answer distinct compiler
or runtime questions.

## Seed Candidate Policy

The packed-field dequantize plus scaled dot workload that motivated this suite
is a good candidate only when it is reduced to a behavior-class name and
nontrivial oracle:

- storage fields: signed 8-bit fields packed in a 32-bit load or byte storage;
- decode contract: sign extension to integer lanes, conversion to f32, scale
  application with the intended scale type;
- compute contract: f32 dot accumulation with nonzero positive and negative
  lanes;
- launch contract: enough lanes/workgroups to exercise the address and store
  behavior being claimed;
- benchmark rows: at least one smoke row and any larger row only if the larger
  shape changes compiler behavior.

Seed names based on a model, product, local report, or vague quantization
nickname do not meet the suite contract. If the exact GGML block schema matters,
the name says so. If the case only uses raw signed byte fields, the name says
that instead.

## Streaming Pipeline Controls

`streaming_packed_s8_dot.loom` carries packed fields, f16 scales, activation
vectors, and an ordered f32 accumulator through runtime-count loops. Its
serial and handwritten depth-two forms establish the basic recurrence. A
second pair compares serial recurrence-unroll two with depth-four lookahead
at the same unroll factor, separating the benefit of lookahead from the
benefit of unrolling. Pipeline depth describes the retained input records;
unroll describes how many original iterations share a loop body.

The depth-four form starts three records, issues a future record while
consuming the oldest, and drains the remaining records in order. Counts below
three take the serial path. Analytic cases use a closed-form exact result;
the shared varied-input case changes every carried input independently and
compares all forms bitwise across startup, steady-state remainders, and drain
boundaries. Different output sentinels make missing stores observable.

The matched factor-two benchmark rows are
`@streaming_packed_s8_dot_serial_unroll2_n128_time` and
`@streaming_packed_s8_dot_pipelined_depth4_unroll2_n128_time`. Their device
timings belong alongside emitted load/compute order, wait causes, registers,
code size, and JIT cost: numerical agreement alone does not establish overlap,
and deeper lookahead increases the amount of live state and generated code.
These programs express their pipelines with ordinary SSA and SCF, using the
existing unroll policy independently.

## Dependent Route Lookahead

`routed_row_combine_f32.loom` combines selected expert-output rows into
1024-channel token rows. Its two kernels use the same unroll factor of two.
The serial control reads and consumes one route per logical iteration;
the explicit pipeline reads route IDs two iterations ahead of consumption and
weights/payloads one ahead. Carrying only the ID in the metadata stage avoids
making the weight an immediate consumer of that lookup.

The varied-input checks preserve row/weight identity across startup, drain,
holes, duplicates and tails. Independent analytic cases check missing routes,
floating-point recurrence order, and the benchmark outputs. Access testing uses
undersized backing storage for inactive routes. The `n8_t1`, `n8_t16`, and
`n8_t256` benchmark suffixes cover decode and batched combines; `n32` rows stress
a longer recurrence. Each benchmark case launches exactly one kernel.

Compare the serial and pipelined rows under the same batching and cache policy.
Native partial waits show overlap inside the unrolled body; edge copies may
still require completion at the backedge. Registers, code size, JIT cost and
device time are separate evidence. The website's
[dependent-load workflow](../../../../../docs/src/workflows/tune-loop-schedules.md#separate-route-and-payload-lookahead)
walks through that comparison.

## Cooperative Paged Attention

`cooperative_paged_attention_f32.loom` assigns one subgroup to each 128-channel
query. A runtime page loop shares one table lookup across K/V rows; a fixed
sixteen-row inner loop pipelines guarded fragments into a subgroup QK reduction
and ordered online softmax/PV state. Target subgroup width sets each lane's
channel fragment. One template takes the per-caller depth and unroll factor,
with depth-one and depth-three callers both unrolled by two.

Independent analytic cases check maximum, denominator and normalized output
for empty, absent, repeated, short and ragged pages. Varied inputs distinguish
K/V row identity and lane fragments across shared pages and different query
lengths. Undersized inactive backing makes accidental tail reads observable
under the access sanitizer. Timing cases launch one kernel each, with exact
expectations and the full K/V cache footprint.

Matched benchmark names are
`@cooperative_paged_attention_serial_n128_i256` and
`@cooperative_paged_attention_pipelined_n128_i256`. The `n128`/`n1024` and
`i1`/`i16`/`i256` suffixes select token and query counts. The
[cooperative pipeline workflow](../../../../../docs/src/workflows/tune-loop-schedules.md#pipeline-cooperative-paged-attention)
connects these workloads to native resource reports and controlled timings.

## Sparse Token Attention

`sparse_token_attention_f32.loom` gathers an ordered prefix of physical token
IDs for each 128-channel query. The prefix guard protects the index read;
an unsigned range test separately protects dependent K/V reads. Negative and
out-of-range IDs leave all online state unchanged, while duplicates contribute
once per occurrence. Causal selection belongs to the caller's index list.

One template takes the caller's depth and unroll factor. A fixed sixteen-entry
tile permits read-ahead into the subgroup score reduction while the outer
prefix count stays dynamic. Matched depth-one/depth-three callers both unroll
by two. Independent analytic cases check maximum, denominator and normalized
output; varied queries cover shared lists and differing prefixes. Minimal
backing and NaN-poisoned inactive payloads exercise both access boundaries.

The `@sparse_token_attention_serial_n128_i256` and
`@sparse_token_attention_pipelined_n128_i256` benchmarks launch one kernel
each. The `n128`/`n1024` and `i1`/`i16`/`i256` suffixes select prefix and query
counts with the same full-cache footprint. The
[sparse attention workflow](../../../../../docs/src/workflows/tune-loop-schedules.md#pipeline-sparse-token-attention)
connects this source to access checking, reports and schedule comparisons.

## Review Questions

Before adding a source file here, the review answers:

- What reusable behavior class does this case represent?
- Which plausible compiler/runtime bug would the correctness sample catch?
- Which existing narrower test would lose the failure mode?
- Which benchmark row will continuous benchmarking track?
- Which targets should run the case, skip it, or report unavailable?
- What evidence would let us retire this case later?
