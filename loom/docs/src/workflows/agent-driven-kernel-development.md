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

One independent variable makes the result interpretable. A necessary compiler
repair and the kernel candidate remain separate changes whenever the candidate
can consume the repaired compiler through its normal source contract.

## Ask the compiler before asking the GPU

The baseline and candidate compile under the same root, workload,
configuration, backend, and target identity:

```shell
loom-compile kernel.loom \
  --root=@production_cut \
  --backend=amdgpu-hal \
  --target=gfx1151 \
  --output=candidate.hsaco \
  --compile-report=summary \
  --compile-report-output=candidate.report.json

loom-compile-report show candidate.report.json
loom-compile-report suggest candidate.report.json
loom-compile-report diff baseline.report.json candidate.report.json
```

The report answers whether the candidate changed the intended mechanism:

- which provider and source-to-Low plan were selected;
- which packed, vector, matrix, memory, and synchronization families remain;
- scheduled pressure and final register allocation;
- LDS, private memory, spills, and materialized reloads;
- modeled residency thresholds and limiting resources; and
- code size, request shape, waits, and barriers.

An empty suggestion list means only that registered target diagnostics found
no issue. It does not prove that the schedule matches an external oracle or
that the hardware will prefer it.

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
| One lane issues wider packets. | Neighboring lanes may now touch distant addresses and destroy wave-level coalescing. |
| The source resembles the oracle's loop. | Address forms, clauses, allocation, waits, and the runtime-selected dispatch can still differ. |
| A profiler reports precise timestamps. | Precision does not establish replay identity or low perturbation. |

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
  encodings; and
- [GGML and llama.cpp](oracles/ggml-llama-cpp.md) preserve model, quantized
  storage, graph, runtime-selection, and semantic-cut contracts during a port.

An oracle answers the question named by its page. It does not become Loom's
source language, force its dispatch boundaries into a model program, or waive
the numerical, compiler, and physical gates around the answer.

## Compact loop

For each candidate:

1. Name the production semantic cut and workload.
2. Run in-source correctness and captured boundary checks.
3. State the expected compiler and native change.
4. Compile and inspect `show`, `suggest`, and strict `diff`.
5. Follow evidence into detailed IR or ISA only when the mechanism differs.
6. Benchmark one discriminating production shape.
7. Sweep realistic shapes, tails, and a second target after the pilot wins.
8. Integrate the cut and measure the complete dependent span.
9. Keep the simplest schedule supported by all three evidence classes.
10. Preserve rejected mechanisms and standalone compiler blockers.

The result is more than a fast artifact. It is a source family another agent
can understand, specialize, validate, and improve without rediscovering its
numerical contract or inventing a new measurement policy.
