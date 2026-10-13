# Three source models, one server, no downloaded weights

The `model.json` documents in [image](image/model.json),
[embedding](embedding/model.json) and [decision](decision/model.json) load deterministic image, embedding and decision
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
  --device=amdgpu \
  --model=experimental/loom_serve/models/fixtures/image/model.json \
  --model=experimental/loom_serve/models/fixtures/embedding/model.json \
  --model=experimental/loom_serve/models/fixtures/decision/model.json \
  --listen=127.0.0.1:8080 --workers=1
```

Managed build and hardware-runner policies still apply. When execution uses a
separate runner, build the exact `serving:server` target first, transfer its
runtime closure and this source directory, and preserve the root-relative
provider paths. No compiler CLI, generated model binaries or downloaded weights
are required by the running process. One worker intentionally exercises shared
CPU scheduling; it is not the recommended configuration for measuring speed.

The same catalog also runs on Vulkan: enable the SPIR-V compiler and Vulkan HAL,
then replace `--device=amdgpu` with `--device=vulkan --pool_backing=fixed`. No source
package edits are needed. The complete image-containing catalog requires a GPU;
the separate
[CPU recipe](../../docs/OPERATING.md#first-output-starting-with-no-models) runs
text, embedding and decision packages on native x86.

The `ready` event announces the listening address. In another terminal:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/readyz
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

Readiness returns `{"status":"ready"}` even while packages are preparing.
After preparation, discovery returns `pixels` as image, `vectors` as embedding,
and `judge` as decision, each with its primary endpoint and input/output
modalities. Registrations still preparing or retiring remain visible in
`/v1/status`, not `/v1/models`. The first two prepared models have
`loom.tokenize:true`; `judge` has no tokenizer. Inference and tokenizer
requests arriving earlier wait within the holding budget. Idle status reports
zero queued/active requests. Status describes scheduler ownership, not GPU
utilization; queries do not activate weights or wait for model completion.

## Independently checkable first results

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
| Decision | `truth` has type `noul` and probability approximately 2/3 for this single-question cohort; token usage is zero |
| Image | `data[0].b64_json` decodes to a 7×5 RGB PNG. R/G are 255; B alternates 0/255 by row-major pixel index |

The decision fixture intentionally changes its probabilities with cohort size.
Its [oracle](decision/README.md) defines the calculation and floating-point
precision contract; another concurrent decision request can therefore change
this example's expected number. The image changes red with seed parity and
green with prompt-token parity. These are
request-dependent GPU results, not fixed HTTP strings. SIGINT/SIGTERM stops
admission and drains accepted work before model and device destruction.

## From a fixture to a real port

| Model behavior | Source starting point | Native contract |
| --- | --- | --- |
| Text-conditioned finite image | [Image package](image/README.md), then [Krea](../krea/README.md) | [Image model](../../../../loom/src/loom/serve/service/image/model.h) |
| Ragged inputs to final embedding vectors | [Embedding package](embedding/README.md) | [Finite model](../../../../loom/src/loom/serve/service/finite/README.md), [embedding records](../../../../loom/src/loom/serve/service/finite/embedding/protocol.h) |
| Structured questions to choice/score/noul probabilities | [Decision package](decision/README.md) | [Finite model](../../../../loom/src/loom/serve/service/finite/README.md), [SystemOne records](../../../../loom/src/loom/serve/service/finite/decision/protocol.h) |
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
owns request preparation, batch packing and result mapping. The model document's `definition.sources` lists
the numerical providers and configuration. Compatible `.loom` edits take effect
on the next registration without relinking or restarting the server. A prepared
registration retains its compiled code until removal. Shapes, pooling,
normalization, learned heads and device command selection remain source responsibilities;
the native finite scheduler uses only declared cost and compatibility tags.

An endpoint for embeddings or decisions does not imply that EmbeddingGemma2 or
CLEF weights can already be loaded. Those ports still need their source package
and real numerical evidence. The separate
[resident XDNA variant](embedding/README.md#verification) proves first and
repeated HTTP output through direct `queue_dispatch`, with no command buffer.
It requires an enabled XDNA compiler target/HAL and physical NPU, and has no
learned weights. This catalog instead proves the GPU command-stage boundary.
Audio, image-input conditioning and graph/workflow composition require their
own concrete input/output ownership contracts rather than repurposed fields.

## Focused verification

CPU protocol and catalog checks run without model files or a GPU:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //loom/src/loom/serve/server:configuration_test \
  //loom/src/loom/serve/server:router_test \
  //loom/src/loom/serve/service/finite/embedding:protocol_test \
  //loom/src/loom/serve/service/finite/decision:protocol_test
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

The [held-device service check](../../../../loom/src/loom/serve/service/image/service_test.cc) separately verifies
that readiness/status and heartbeat progress while a real HAL predecessor is
held, then releases it and checks final output. It is not a timeout-based speed
assertion. Real model quality and controlled performance runs follow the
[authoring](../../docs/MODEL_AUTHORING.md) and
[measurement](../../docs/PERFORMANCE.md) guides.

## Mixed lifecycle churn

The [catalog workload](catalog_test.py) launches the actual shared executable
with all four source-model families and a temporary 76-byte text checkpoint.
It copies the source packages into an external deployment root, dereferencing
runfiles symlinks, and moves numerical providers into sibling library directories
referenced by each model document. The separate VM preparation/control programs,
weight policy and tokenizer assets remain in their packages. The child runs
from that new root with model-relative providers/assets and an ordinary
startup flagfile. It cannot satisfy those references through the original
repository cwd. The CLI test also combines flagfile and command-line model
entries, then unregisters and re-registers the same document over HTTP.

The workload removes each model's `control.loom` in turn, placing that deployment
last in the catalog, and separately removes a sibling numerical provider. The
server stays ready, exposes that registration as `failed` with the missing-file
diagnostic, and produces exact output from a peer model. The failed deployment
remains inspectable through normal shutdown. Restoring the files permits the
complete workload below; no alternate source directory or precompiled artifact
can satisfy a missing leaf.

The dynamic catalog starts empty and registers twelve initially drained models.
Its one-worker run uses inline process options; the four-worker run loads the
same options through an IREE flag file. A separate flag-file launch combines
TCP and Unix listeners with spaces and quotes in the socket path. These execute
the [operator configuration flow](../../docs/OPERATING.md), not a test-only
configuration parser.

Twenty concurrent client operations share two resident text rows and a
128-position logical pool. Variable-length conversations run alongside image,
embedding, decision and tokenizer requests; partial-prefix forks and rewind
are checked against independent dense arithmetic. Invalid requests and reset
peers must leave the other work usable.

After releasing the explicit checkpoint, each round admits a fresh request
reserving all 128 positions with an incompatible prefix. Idle retention is
evictable, but leaked pins or reservations would prevent that request fitting.
Session/checkpoint names are reused across rounds, final status and metrics
must agree on idle, and shutdown joins normally. The default runs four rounds
each with one and four shared workers, emitting one compact JSON result per
round. No model download or production test hook is involved.

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures:catalog_test \
  //experimental/loom_serve/models/fixtures:catalog_vulkan_test
```

Both backend targets run the same workload, including twelve dynamic models and
256 parked clients. The first real embedding request activates native queue
machinery; its thread cost is recorded separately. Loading the remaining models,
parking clients and replacing model registrations must add no userspace threads.
Non-dyadic decision probabilities use a numeric bound for F32 division rather
than a backend-specific decimal string. Text, embedding and image results retain
their exact oracles.

This is a bounded logical-ownership workload, not a latency assertion or a
particular interleaving/batch-size requirement. Fixed backing isolates it from
physical remapping; real-model and native checks retain responsibility for VMM
trim/growth, offload, MTP, numerical quality and measured throughput. The
fixture-specific held-work checks cover exact cancellation and dependency
boundaries that an external client cannot force merely by resetting a socket.
