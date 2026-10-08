# Deterministic source image model

This small package implements the same source ABI as learned image models.
It needs no downloaded weights. `prepare.loom` declares one command stage;
`control.loom` tokenizes the prompt and constructs a request payload. Two GPU
kernels exchange a palette through compiler-reflected private workspace and
write F32 CHW pixels. The ordinary image service encodes those pixels as PNG.

The tokenizer recognizes `a`, `b`, and `x`, with one token per character. Red
depends on seed parity, green on token-count parity, and blue on pixel parity.
Pixel extents are independently configurable from 1 through 64. The output is
a transport/ownership oracle, not a neural quality or throughput benchmark.
The source rejects prompts beyond the configured token limit, exercising
request validation in the VM rather than in the HTTP parser.

The integration test runs real source JIT, VM control, command submission and
HTTP responses with one and four shared CPU workers:

```sh
iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures/image:service_test
```

On an admitted AMDGPU runner, the standalone image server can load this package
without any model-specific native code:

```sh
iree-bazel-run //experimental/loom_serve/image:server -- \
  --model=experimental/loom_serve/models/fixtures/image \
  --checkpoint=experimental/loom_serve/models/fixtures/image \
  --width=7 --height=5 --text_tokens=32
```

`POST /v1/images/generations` with `{"prompt":"aba","seed":1}` returns a
base64 PNG through the normal endpoint. Parameter paging, learned image quality
and real model performance have separate model tests.
