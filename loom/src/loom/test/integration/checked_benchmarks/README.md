# Checked Benchmark Integration Suite

This suite is for full Loom programs that must compile, execute, validate, and
benchmark through production-facing infrastructure. A case belongs here when the
important risk is the interaction between multiple compiler and runtime
contracts, and that interaction cannot be represented faithfully by a smaller
source-low, op, pass, tool, or authoring test.

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
Loom. Checked benchmark integration cases are regression and benchmark assets:
they can have comments, but their main job is to preserve end-to-end signal.

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
  func.call @semantic_behavior_class(...)
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

## Review Questions

Before adding a source file here, the review answers:

- What reusable behavior class does this case represent?
- Which plausible compiler/runtime bug would the correctness sample catch?
- Which existing narrower test would lose the failure mode?
- Which benchmark row will continuous benchmarking track?
- Which targets should run the case, skip it, or report unavailable?
- What evidence would let us retire this case later?
