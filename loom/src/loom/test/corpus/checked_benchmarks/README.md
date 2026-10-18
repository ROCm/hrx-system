# Checked Benchmark Corpus

This corpus contains complete Loom programs with correctness scenarios and
companion benchmark campaigns. Target providers compose these data-only
sources into capability-coherent modules and execute them through
production-facing infrastructure. A workload family belongs here when the
important risk is the interaction between multiple compiler and runtime
contracts, and that interaction cannot be represented faithfully by a smaller
source-low, op, pass, tool, or authoring test.

Each checked benchmark family is expected to contain:

- One or more kernels or callable entry points under test.
- One or more `check.scenario` records that batch deterministic runtime trials
  against an independent oracle. A focused `check.case` remains appropriate
  when one integration program must coordinate several subjects.
- A `_benchmark.loom` companion whose target-only scenarios select the timed
  workload without placing benchmark scale in ordinary correctness execution.
- One or more `check.benchmark<@scenario> @name` rows selecting those campaigns
  for timing.
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

## Scenario Template

```loom
kernel.def @semantic_behavior_class() {
  // Launch geometry that is part of the represented behavior.
} launch(%input: buffer, %output: buffer) {
  // Full source pattern under test.
  kernel.return
}

func.def @semantic_behavior_class_oracle(%input: buffer, %output: buffer) {
  // Independent scalar reference.
  func.return
}

check.scenario @semantic_behavior_class_values {
  check.trial[...](%trial: index, %entropy: check.entropy) {
    // Runtime inputs that distinguish the represented behavior.
    check.compare<@semantic_behavior_class, @semantic_behavior_class_oracle>(%input, %output) : (tensor<...>, tensor<...>) -> () {
      check.expect.close actual(...) expected(...) atol(...) rtol(...) nan(...) : ...
    }
  }
  check.return
}
```

The companion benchmark scenario uses `check.invoke<@semantic_behavior_class>`
with the same production shape and no oracle execution in the timed path. The
template is intentionally small. A real family adds trial domains only when
those shapes answer distinct compiler or runtime questions.

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
three take the serial path. One seven-trial scenario checks the serial
recurrence against a closed-form exact oracle. The shared varied-input case
changes every carried input independently and compares all forms bitwise across
startup, steady-state remainders, and drain boundaries. Different output
sentinels make missing stores observable. Benchmark-only count sweeps live in
the companion source, outside ordinary correctness execution.

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
holes, duplicates and tails. Correctness scenarios use an independent,
route-ordered oracle and analytic expectations for missing routes and
floating-point recurrence order. Access testing uses undersized backing storage
for inactive routes. The `@routed_row_combine_serial_workloads` and
`@routed_row_combine_pipelined_workloads` benchmarks each generate six samples:
one, sixteen and 256 tokens with eight routes, followed by the same token counts
with 32 routes. `--sample=2` selects eight routes and 256 tokens. Each sample
launches one kernel, with input generation outside the measured invocation.

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
under the access sanitizer. Benchmark samples launch one kernel each, with the
same full K/V cache footprint.

The `@cooperative_paged_attention_serial_workloads` and
`@cooperative_paged_attention_pipelined_workloads` benchmarks each generate six
samples: 128-token prefixes at 1, 16, and 256 queries, followed by 1024-token
prefixes at the same query counts. `--sample=2` selects 128 tokens and 256
queries. Input generation stays outside the measured invocation. The
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

The `@sparse_token_attention_serial_workloads` and
`@sparse_token_attention_pipelined_workloads` benchmarks each generate six
samples: 128-token prefixes at 1, 16, and 256 queries, followed by 1024-token
prefixes at the same query counts. Each sample launches one kernel with the
same full-cache footprint; `--sample=2` selects 128 tokens and 256 queries. The
[sparse attention workflow](../../../../../docs/src/workflows/tune-loop-schedules.md#pipeline-sparse-token-attention)
connects this source to access checking, reports and schedule comparisons.

## Shared K/V Across Query Heads

`grouped_paged_attention_f32.loom` compares two independent query subgroups
with one subgroup sharing K/V fragments across two distinct query heads.
A pair owns its page table; each head retains its own length, query, online
state and normalized output. The union prefix guards shared loads, while
separate consumer guards preserve unequal tails and either head being empty.

Independent and shared callers both use depth two and unroll two. A shared
depth-one caller isolates pipelining. The online update is a reusable template
whose vector width comes from the target's subgroup size. Analytic checks
anchor both heads' states and outputs; varied pairs distinguish query and
table ownership. Minimal backing covers ragged tails, holes and repeated pages.

The `@grouped_paged_attention_independent_workloads` and
`@grouped_paged_attention_shared_workloads` benchmarks each generate six
samples: 128-token prefixes at 1, 128, and 1024 pairs, followed by 1024-token
prefixes at the same pair counts. `--sample=2` compares 2048 query heads over
the same 64 MiB K/V pool. Sharing halves issued K/V loads for
equal-length heads but retains more state and launches fewer subgroups. The
[shared-loading workflow](../../../../../docs/src/workflows/tune-loop-schedules.md#share-kv-loads-across-query-heads)
compares that tradeoff separately from pipeline depth.

## Review Questions

Before adding a source file here, the review answers:

- What reusable behavior class does this case represent?
- Which plausible compiler/runtime bug would the correctness sample catch?
- Which existing narrower test would lose the failure mode?
- Which benchmark row will continuous benchmarking track?
- Which targets should run the case, skip it, or report unavailable?
- What evidence would let us retire this case later?
