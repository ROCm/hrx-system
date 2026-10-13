# Deterministic source decision model

This package exercises the [SystemOne adapter](../../service/finite/decision/README.md)
and [finite model ABI](../../service/finite/README.md). It compiles ordinary Loom
VM preparation/control and two native kernels on x86 CPU, AMDGPU and Vulkan.
No downloaded model, native canned answer or model-specific C is involved.

The complete [model document](model.json) names its prepare/control programs,
native providers and asset base. Start it with
`--model=loom/src/loom/serve/fixture/decision/model.json`, or
register the same object over HTTP with an absolute `base_directory`.
Its `definition.options.batch_units` ranges from 1 through 256 and defaults
to 64. Each question costs one unit plus its
option count. Requests that individually exceed the limit return 422; compatible
requests already waiting can share one cohort. This byte-oriented fixture has
no tokenizer and reports zero input and output tokens.

For each option, source preparation adds the raw JSON byte lengths of state,
instructions and criterion to the first byte of a choice ID (or the zero-based
implicit score/noul ordinal). An absent noul criterion contributes zero bytes;
choice null contributes its four literal bytes. The first kernel computes:

```
weight = 1 + ((seed + cohort_question_count) & 3)
```

The second kernel normalizes these positive weights per question, using
compiler-reflected transient workspace between kernels. Its output rows are
stored in reverse physical order and described by the source result plan.
The adapter restores original question/option identities. Cohort size
deliberately affects this oracle: a per-request command loop cannot accidentally
pass the packed-dispatch test. These values have no semantic/calibration meaning.

## Verification

On supported x86 CPU, AMDGPU and Vulkan runners using the repository's execution
policy:

```sh
iree-bazel-test --config=asan \
  //loom/src/loom/serve/service/finite/decision:protocol_test \
  //loom/src/loom/tooling/target/cpu/serve:source_decision_service_test \
  //loom/src/loom/tooling/target/amdgpu/serve:source_decision_service_test \
  //loom/src/loom/tooling/target/spirv/serve:source_decision_service_test
```

The actual native-device/HTTP cases run with one and four shared task workers.
They cover a four-question cohort spanning three requests, independent peer
reset, choice/score/noul results, reversed storage, deterministic ties, source
capacity rejection followed by valid work, and shared routing alongside another
source embedding model and raw tokenization. CPU protocol tests cover escaped
IDs, structured criteria, normalized-result validation and exact confidence
formulas. Non-dyadic probabilities are compared numerically against the
floating-point division contract, not against one backend's decimal rounding.
Exactly representable cohort results retain exact wire checks. The source uses
the selected device target; a server using `--device=task --pool_backing=fixed` or
`--device=vulkan --pool_backing=fixed` needs no package edits. CPU device kernels
are JIT-compiled to native x86 and share the application executor with model
preparation and scheduling.

A real decision port supplies its own source formatter, parameter declarations,
backbone/head kernels and numerical evidence. This fixture proves the serving
boundary, not learned-model quality, throughput or NPU support.
