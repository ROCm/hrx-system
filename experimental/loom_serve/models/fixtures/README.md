# Three source models, one server, no downloaded weights

[catalog.json](catalog.json) loads deterministic image, embedding and decision
packages into the ordinary [shared server](../../serving/README.md). Each uses
real source-JIT VM programs and two GPU kernels, compiler-reflected private
workspace, normal admission and the actual HTTP response encoder. The native
server has no fixture switch or canned-response path.

This is a small first-output and authoring witness. It establishes model-package
and transport contracts, not learned-model quality, memory pressure, neural
throughput or NPU support. The separate [retained-text fixture](text/README.md)
uses the actual [retained-row contract](../../text/README.md) and a generated
76-byte checkpoint. It can be added to this catalog; text generation is not
simulated through a finite-vector endpoint.

## Launch

The [fresh-machine setup](../../docs/README.md#fresh-machine-prerequisites)
enables the VM/AMDGPU compiler and HAL. On a supported Linux AMDGPU runner,
from the repository root:

```sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/serving:server -- \
  --models=experimental/loom_serve/models/fixtures/catalog.json \
  --port=8080 --workers=1
```

Managed build and hardware-runner policies still apply. When execution uses a
separate runner, build the exact `serving:server` target first, transfer its
runtime closure and this source directory, and preserve the root-relative
catalog paths. No compiler CLI, generated model binaries or downloaded weights
are required by the running process. One worker intentionally exercises shared
CPU scheduling; it is not the recommended configuration for measuring speed.

The `ready` event announces the listening address. In another terminal:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/readyz
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

Readiness returns `{"status":"ready"}`. Discovery returns `pixels` as image,
`vectors` as embedding, and `judge` as decision, each with its primary endpoint.
The first two have `loom.tokenize:true`; `judge` has no tokenizer. Idle status
reports zero queued/active requests. Status describes scheduler ownership, not
GPU utilization; queries do not activate weights or wait for model completion.

## Exact first results

The following commands are independent and may overlap. Requests to the same
finite-vector model can coalesce; the chosen two-dimensional embedding result
does not depend on cohort size.

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"vectors","content":"aba"}'
curl --fail --silent --show-error http://127.0.0.1:8080/detokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"pixels","tokens":[1,0,1]}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"vectors","input":["a","bbb"],"dimensions":2}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"judge","state":"a","questions":{"truth":{"type":"noul","instructions":"???","criteria":{"false":"","true":"x"}}}}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"pixels","prompt":"aba","seed":1,"response_format":"b64_json"}'
```

Expected results:

| Request | Independently checkable output |
| --- | --- |
| Tokenize | `{"tokens":[0,1,0]}` |
| Detokenize | `{"content":"bab"}` |
| Embeddings | Input indices 0/1 contain `[1,1]` / `[6,3]`; prompt/total usage is 4 |
| Decision | `truth` is `{"type":"noul","noul":0.666666687}` for this single-question cohort; token usage is zero |
| Image | `data[0].b64_json` decodes to a 7×5 RGB PNG. R/G are 255; B alternates 0/255 by row-major pixel index |

The decision fixture intentionally changes its probabilities with cohort size.
Its [oracle](decision/README.md) defines the exact calculation; another concurrent
decision request can therefore change this example's expected number. The image
changes red with seed parity and green with prompt-token parity. These are
request-dependent GPU results, not fixed HTTP strings. SIGINT/SIGTERM stops
admission and drains accepted work before model and device destruction.

## From a fixture to a real port

| Model behavior | Source starting point | Native contract |
| --- | --- | --- |
| Text-conditioned finite image | [Image package](image/README.md), then [Krea](../krea/README.md) | [Image model](../../image/model.h) |
| Ragged inputs to final embedding vectors | [Embedding package](embedding/README.md) | [Finite model](../../finite/README.md), [embedding records](../../embedding/protocol.h) |
| Structured questions to choice/score/noul probabilities | [Decision package](decision/README.md) | [Finite model](../../finite/README.md), [SystemOne records](../../decision/protocol.h) |
| Retained autoregressive sessions | [Tiny text package](text/README.md), then [Qwen](../qwen/README.md) | [Text model and source entries](../../text/README.md) |

Copying a source package establishes its ABI skeleton, not its mathematics.
The port replaces kernel math, tensor names/layouts, source preparation and
request/task framing, and supplies an independent numerical oracle. Native
modality code stays unchanged for compatible packages. The embedding fixture's
[weighted variant](embedding/variants/weights.loom) adds an actual checkpoint
parameter and its [policy](embedding/variants/policy.loom) through source only;
its service test exercises load/admission/eviction rather than hiding weights
in a native callback. A real model supplies its actual checkpoint file.

`prepare.loom` declares model capacities, tokenizer and commands. `control.loom`
owns request preparation, batch packing and result mapping. `sources.txt` lists
the numerical providers and configuration. Compatible `.loom` edits take effect
on the next server launch without relinking. Shapes, pooling, normalization,
learned heads and device command selection remain source responsibilities;
the native finite scheduler uses only declared cost and compatibility tags.

An endpoint for embeddings or decisions does not imply that EmbeddingGemma2 or
CLEF weights can already be loaded. Those ports still need their source package
and real numerical evidence. XDNA execution additionally requires the matching
compiler-target/HAL command path. This catalog proves the GPU serving boundary.
Audio, image-input conditioning and graph/workflow composition require their
own concrete input/output ownership contracts rather than repurposed fields.

## Focused verification

CPU protocol and catalog checks run without model files or a GPU:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/serving:configuration_test \
  //experimental/loom_serve/http:router_test \
  //experimental/loom_serve/embedding:protocol_test \
  //experimental/loom_serve/decision:protocol_test
```

The source ABI checks execute actual GPU commands and HTTP results with one and
four workers, including input rejection followed by reuse, original result
association and lifetime across client reset:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures/image:service_test \
  //experimental/loom_serve/models/fixtures/embedding:service_test \
  //experimental/loom_serve/models/fixtures/decision:service_test
```

The [held-device service check](../../image/service_test.cc) separately verifies
that readiness/status and heartbeat progress while a real HAL predecessor is
held, then releases it and checks final output. It is not a timeout-based speed
assertion. Real model quality and controlled performance runs follow the
[authoring](../../docs/MODEL_AUTHORING.md) and
[measurement](../../docs/PERFORMANCE.md) guides.
