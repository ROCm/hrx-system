# A tiny retained-text model

This package implements the same pooled autoregressive ABI as Qwen, with two
tiny GPU kernels and a real four-byte checkpoint parameter. It uses ordinary
source-JIT VM programs, page ownership, recurrent slots, named checkpoints and
HTTP streaming. There is no native mock model or canned response path.

This is deterministic arithmetic, not learned language generation or a
throughput proxy. It makes retained-session failures observable without a model
download. Other modalities have their own [source fixtures](../README.md).

## Definition and independent oracle

Token IDs are `a=0`, `b=1`, `x=2`, `#=3`. `#` is the terminal token and the
assistant-message delimiter. Starting with `h=k=position=0`, each consumed
token performs unsigned 32-bit arithmetic:

```text
h = (5*h + token + bias) % 97
k = (7*k + h + position) % 101
history[position] = k
position += 1
prediction = #                         if position % 11 == 0
             (h + sum(history)) % 3     otherwise
```

Selection reads the complete paged history; the next chunk reads both its
recurrent anchor and last cached value. The CPU oracle in
[service_test.cc](service_test.cc) maintains a dense running sum instead of
interpreting device packets or physical page IDs. A pending prediction remains
unconsumed until the next step.

With `bias=2`, prompt `abxa` generates `bxaaaxb` followed by EOS, or `bx` with
`max_tokens=2`. The latter leaves consumed position 5 and pending token `x`:
a deliberately partial four-token page for checkpoint tests.

## Source boundaries

[prepare.loom](prepare.loom) declares one command, one checkpoint binding,
private row views, four-token pages and independent four-byte recurrent slots.
Every host token/span bucket uses that command; a bucket does not inherently
require another compiled kernel. Supported configurations use pooled capacity,
packed epochs and no MTP. Unsupported configurations fail through the cold
source program's `input.require` capability.

[advance.loom](advance.loom) consumes known spans, then selects output. The
command has one parameter root and the text ABI's six dynamic bindings. Its
numerical helper receives dimensions as arguments; configuration is read at
kernel entry points. A restored row reads its shared recurrent anchor at the
fork frontier, writes its assigned slot, and uses the writer thereafter.
Logical positions resolve through the host-owned physical page map.

[control.loom](control.loom) owns packets, feedback and chat framing. It
concatenates system/user content and terminates supplied assistant messages
with `#`. Completed canonical text excludes that next delimiter. A retained
append consumes the original pending token: if it was EOS, input preparation
removes the duplicated leading delimiter from the suffix. Full replay and
retained execution therefore consume identical histories. Tool and reasoning
features are rejected.

The integration test uses fixed physical backing while exercising the actual
logical pool and copy-on-write sharing. It does not establish live virtual
remapping, offload, MTP, neural accuracy or NPU execution. Those contracts have
separate real-model/backend witnesses.

## First output without a download

Create the 76-byte checkpoint at an explicit temporary path:

```sh
python3 - /tmp/fixture-text.safetensors <<'PY'
import struct
import sys

header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
header += b' ' * (-len(header) % 8)
with open(sys.argv[1], 'wb') as output:
    output.write(struct.pack('<Q', len(header)))
    output.write(header)
    output.write(struct.pack('<I', 2))
PY

build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/text:server -- \
  --model=experimental/loom_serve/models/fixtures/text \
  --weights=/tmp/fixture-text.safetensors \
  --tokenizer=experimental/loom_serve/models/fixtures/text/tokenizer.json \
  --rows=4 --context_capacity=96 --pool_capacity=256 \
  --prefill_capacity=32 --checkpoint_capacity=2 --max_tokens=16 \
  --pool_backing=fixed --workers=1 --port=8080
```

Repository build and hardware-runner policy still applies. A separate runner
needs the exact built server, its runtime closure, these sources and the tiny
checkpoint. No compiler CLI runs in the server.

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Loom-Session: example' \
  -d '{"model":"fixture-text","stream":true,"max_tokens":2,"messages":[{"role":"user","content":"abxa"}]}'
curl --fail --silent --show-error -X POST \
  http://127.0.0.1:8080/v1/checkpoints/base -H 'X-Loom-Session: example'
```

The stream contains `bx` with finish reason `length`; the next request pins
position 5. Subsequent requests send full history and can select
`X-Loom-Checkpoint: base` to fork or rewind. The
[text HTTP contract](../../../text/README.md#named-http-checkpoints) details
ownership and error behavior.

The shared server accepts the same package alongside [catalog.json](../catalog.json):

```json
{
  "kind": "text",
  "name": "fixture-text",
  "source": "experimental/loom_serve/models/fixtures/text",
  "weights": "/tmp/fixture-text.safetensors",
  "tokenizer": "experimental/loom_serve/models/fixtures/text/tokenizer.json",
  "rows": 4,
  "context_capacity": 96,
  "pool_capacity": 256,
  "prefill_capacity": 32,
  "checkpoint_capacity": 2,
  "max_tokens": 16,
  "mtp_depth": 0
}
```

## Verification

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures/text:request_test \
  //experimental/loom_serve/models/fixtures/text:service_test
```

The qualified AMDGPU test uses one and four shared workers. It compares full
and chunked input, row identities, simultaneous branch writers, pinned prefixes,
final pool credit, HTTP rewind/continuation, input rejection recovery, EOS
framing, queued cancellation and changed checkpoint weights against independent
dense arithmetic. The CPU-only request test exercises source framing, exact
token IDs and rejection before any device or checkpoint is needed.
Exact output, not compilation alone, establishes the boundary. Real Qwen
remains the numerical/performance reference for its architecture.
