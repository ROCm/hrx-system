# Deterministic source embedding model

This package implements the [finite-vector ABI](../../../finite/model.h) using
real VM preparation and native device commands on x86 CPU, AMDGPU and Vulkan.
It needs no downloaded model. Its purpose is to expose batching, row association
and ownership errors through normal HTTP requests; its numbers have no neural
meaning.

The tokenizer recognizes `a`, `b` and `x` as IDs 0, 1 and 2. Source preparation
accepts text or token IDs, 1–31 tokens per input, and dimensions 2 or 4 (default
4). The source-owned `batch_units` option ranges from 1 through 256 and defaults
to 64. Every row costs its token count plus one unit. A request must fit; the
scheduler combines compatible requests up to that capacity.

The first kernel pools each input into four values:

```
[sum(token_id + 1), token_count, first_token_id + 1, cohort_row_count]
```

The second kernel selects the requested dimensions and writes rows in reverse
physical order. The source result plan restores the client's original indices.
The fourth component deliberately exposes the actual dispatch cohort, making a
scalar loop masquerading as batching observable. Two-dimensional results omit
that component. Both kernels communicate through compiler-reflected transient
workspace.

## Run through the shared server

Use a catalog with this deployment (paths are relative to the server's working
directory):

```json
{
  "models": [{
    "kind": "embedding",
    "name": "vectors",
    "source": "experimental/loom_serve/models/fixtures/embedding",
    "assets": "experimental/loom_serve/models/fixtures/embedding",
    "options": {"batch_units": 64},
    "pending_requests": 16
  }]
}
```

On an x86 CPU runner, using the repository's normal build/execution policy:

```sh
iree-bazel-run //experimental/loom_serve/serving:server -- \
  --models=/path/to/models.json --listen=127.0.0.1:8080 \
  --device=task --pool_backing=fixed
```

`--device=amdgpu` or `--device=vulkan` runs the identical source through that
backend. Each compiler target and HAL driver must be enabled in the build.
The CPU task device shares the server workers; device kernels compile to native
x86 code while request/control programs execute in the VM.

`POST /v1/embeddings` with
`{"model":"vectors","input":["a","bbb"],"dimensions":2}` returns vectors
`[1,1]` and `[6,3]`, indices 0 and 1, and four input tokens. The equivalent
token input is `[[0],[1,1,1]]`. `encoding_format:"base64"` encodes the same
little-endian IEEE754 F32 values. `/tokenize` and `/detokenize` are available
through the package's tokenizer without activating its model commands.

## Verification

```sh
iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures/embedding:service_cpu_test \
  //experimental/loom_serve/models/fixtures/embedding:service_test \
  //experimental/loom_serve/models/fixtures/embedding:service_vulkan_test
```

The tests cover ragged cross-request coalescing, distinct output dimensions,
reversed physical rows, reset peers, source-capacity splitting, input errors,
float/base64 responses, raw tokenizer utilities and heartbeat shutdown. They run
with one and four shared workers. CPU and Vulkan cover fixed-backed model
storage; AMDGPU additionally covers elastic group-backed admission.

`variants/` supplies a checkpoint-backed scaling head and its weight policy.
The test writes a separate source directory and a tiny safetensors checkpoint
containing F32 2.0, then loads that package with the same native binary. A second
actual model pins the only available parameter slab. The accepted request stays
prepared while admission waits; after the pin releases, the source's scaling
head produces doubled vectors through the HTTP endpoint. The test exercises
real checkpoint indexing, loading, LRU eviction and compact feedback, not a
native canned response or mock model vtable.
