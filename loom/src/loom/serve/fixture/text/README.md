# A tiny retained-text model

This package implements the pooled autoregressive serving ABI with two
tiny device kernels and a real four-byte checkpoint parameter. It uses ordinary
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
[source_test_oracle.h](../../service/text/source_test_oracle.h) maintains a dense running sum instead of
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
The source sizes its packet storage and specializes the token view for the
largest requested shape, through the runner's 262144-token context ceiling.
Tests exercise 513-, 1024- and 4097-token shapes, mixed decode/prefill, and
4089-token HTTP history reuse against the dense oracle. The same two-kernel
command serves every smaller bucket; no native capacity edit is required.

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

Copy the package to a temporary directory, generate its 76-byte checkpoint,
and enable two named checkpoint slots in its model document:

```sh
fixture_directory=$(mktemp -d)
python3 - "$fixture_directory" <<'PY'
import json
import shutil
import struct
import sys
from pathlib import Path

directory = Path(sys.argv[1])
shutil.copytree("loom/src/loom/serve/fixture/text", directory, dirs_exist_ok=True)
header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
header += b' ' * (-len(header) % 8)
(directory / "weights.safetensors").write_bytes(
    struct.pack('<Q', len(header)) + header + struct.pack('<I', 2)
)
document = json.loads((directory / "model.json").read_text())
document["definition"]["checkpoint_capacity"] = 2
(directory / "model.json").write_text(json.dumps(document))
PY

build_tools/bin/iree-bazel-run --config=opt \
  //loom/src/loom/tools/loom-serve -- \
  --model="$fixture_directory/model.json" \
  --device=task --pool_backing=fixed --workers=1 --listen=127.0.0.1:8080
```

The CPU task device shares the server's worker pool and JITs native x86 code;
it does not interpret the device kernels in the VM. `--device=amdgpu` or
`--device=vulkan` runs the same source on the corresponding GPU backend with
fixed backing. Each backend must be enabled in the build configuration.
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
[text HTTP contract](../../../../../../experimental/loom_serve/text/README.md#named-http-checkpoints) details
ownership and error behavior.

For wrappers expecting a complete JSON response, omit `stream` or set it to
false. The same model and scheduler produce the same tokens:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fixture-text","max_tokens":2,"messages":[{"role":"user","content":"abxa"}]}'
```

`choices[0].message` is `{"role":"assistant","content":"bx"}`, finish reason
is `length`, and usage reports four prompt tokens and two completion tokens.
With `max_tokens:16`, content is `bxaaaxb`, finish reason is `stop`, and usage
counts eight completion tokens including EOS. Streaming and nonstreaming turns
can share one `X-Loom-Session`; neither changes the retained transcript.

The server accepts this [model document](model.json) alongside other models
using repeated `--model` flags. The copied document resolves its source and
checkpoint paths relative to its own directory. The same complete document can
be registered over HTTP with an explicit absolute `base_directory` naming that
directory. Source programs may also be stored as `.loombc`; changing their
format does not change this path or ownership contract.

## Verification

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/fixtures/text:request_test \
  //experimental/loom_serve/models/fixtures/text:client_test \
  //loom/src/loom/tooling/target/cpu/serve:source_text_service_test \
  //loom/src/loom/tooling/target/amdgpu/serve:source_text_service_test \
  //loom/src/loom/tooling/target/spirv/serve:source_text_service_test
```

The native CPU, AMDGPU and Vulkan service tests use one and four shared workers.
They compare full and chunked input, row identities, simultaneous branch writers,
pinned prefixes, final pool credit, HTTP rewind/continuation, input rejection recovery, EOS
framing, queued cancellation and changed checkpoint weights against independent
dense arithmetic. Elastic virtual-memory cases run separately on AMDGPU; the
CPU and Vulkan suites exercise fixed-backed logical pooling. The CPU-only
request test exercises source framing, exact token IDs and rejection before any
device or checkpoint is needed.
Exact output, not compilation alone, establishes the boundary. Learned models
need their own numerical and performance evidence.

The CPU-only client test starts the public shared-server executable and checks
actual HTTP JSON/SSE parsing, usage, concurrent output modes, retained turns
across those modes, EOS, unsupported options and malformed source output. It
creates only the tiny checkpoint above. The service tests also disconnect a
nonstreaming request with a real epoch accepted, verify retirement and reuse,
and check that no successful headers escaped before completion.
