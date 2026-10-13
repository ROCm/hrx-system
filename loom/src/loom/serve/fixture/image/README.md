# Deterministic source image model

This small package implements the same source ABI as learned image models.
It needs no downloaded weights. `prepare.loom` declares one command stage;
`control.loom` tokenizes the prompt and constructs a request payload. Two device
kernels exchange a palette through compiler-reflected private workspace and
write F32 CHW pixels. The ordinary image service encodes those pixels as PNG.

The tokenizer recognizes `a`, `b`, and `x`, with one token per character. Red
depends on seed parity, green on token-count parity, and blue on pixel parity.
Pixel extents are independently configurable from 1 through 64. The output is
a transport/ownership oracle, not a neural quality or throughput benchmark.
The source rejects prompts beyond the configured token limit, exercising
request validation in the VM rather than in the HTTP parser.

The integration test runs real source JIT, VM control, command submission and
HTTP responses with one and four shared CPU workers, both whole-command and
one-command-per-part execution. The latter preserves the palette across queue
submissions and exercises cancellation before the pixel consumer. CPU task,
AMDGPU and Vulkan use the same source and output oracle:

```sh
iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures/image:service_cpu_test \
  //experimental/loom_serve/models/fixtures/image:service_test \
  //experimental/loom_serve/models/fixtures/image:service_vulkan_test
```

The ordinary server can load this package without any model-specific native
code. From the repository root, on an x86 CPU runner:

```sh
iree-bazel-run //loom/src/loom/tools/loom-serve -- \
  --device=task --pool_backing=fixed \
  --model=loom/src/loom/serve/fixture/image/model.json \
  --listen=127.0.0.1:8080
```

`POST /v1/images/generations` with
`{"model":"pixels","prompt":"aba","seed":1}` returns a base64 PNG
through the normal endpoint. Parameter paging, learned image quality
and real model performance have separate model tests.
