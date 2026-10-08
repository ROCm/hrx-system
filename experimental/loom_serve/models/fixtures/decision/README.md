# Deterministic source decision model

This package exercises the [SystemOne adapter](../../../decision/README.md)
and [finite model ABI](../../../finite/README.md). It compiles ordinary Loom
VM preparation/control and two GPU kernels. No downloaded model, native canned
answer or model-specific C is involved.

Catalog entry:

```json
{
  "kind": "decision",
  "name": "judge",
  "source": "experimental/loom_serve/models/fixtures/decision",
  "assets": "experimental/loom_serve/models/fixtures/decision",
  "options": {"batch_units": 64},
  "pending_requests": 16
}
```

Use this entry in the shared server's `models` catalog. `batch_units` ranges
from 1 through 256 and defaults to 64. Each question costs one unit plus its
option count. Requests that individually exceed the limit return 422; compatible
requests already waiting can share one cohort. This byte-oriented fixture has
no tokenizer and reports zero input and output tokens.

For each option, source preparation adds the raw JSON byte lengths of state,
instructions and criterion to the first byte of a choice ID (or the zero-based
implicit score/noul ordinal). An absent noul criterion contributes zero bytes;
choice null contributes its four literal bytes. The first GPU kernel computes:

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

On a supported AMDGPU runner using the repository's normal execution policy:

```sh
iree-bazel-test --config=asan \
  //experimental/loom_serve/decision:protocol_test \
  //experimental/loom_serve/models/fixtures/decision:service_test
```

The actual GPU/HTTP cases run with one and four task workers. They cover a
four-question cohort spanning three requests, independent peer reset,
choice/score/noul results, reversed storage, deterministic ties, source capacity
rejection followed by valid work, and shared routing alongside another source
embedding model and raw tokenization. CPU protocol tests cover escaped IDs,
structured criteria, normalized-result validation and exact confidence formulas.

A real decision port supplies its own source formatter, parameter declarations,
backbone/head kernels and numerical evidence. This fixture proves the serving
boundary, not learned-model quality, throughput or NPU support.
