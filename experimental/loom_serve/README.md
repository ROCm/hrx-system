# Loom model runner experiment

This branch-local runner connects compiled model-control VM code to prepared
HAL commands. Its private interfaces are intended to change with real models.
It is not a general HAL VM module or a serving framework.

`command.c` combines a parsed portable command program with loaded executable
reflection. Recording retains fixed buffers and code. Rebindable slots include
model state and explicit scratch; the materializer allocates no device backing.
There is no semantic-resource extension to the command artifact format.

`module.c` publishes prepared stages as ordinary fixed-signature VM imports:
`(hal.buffer, ... slots ...) -> i64`. Model code selects the stage and buffers;
the native call submits and returns without waiting. HAL captures bindings and
retains their buffers independently of the VM invocation. One process can serve
many rows; its preallocated native binding scratch is reused between calls.

`execution.c` is the shared execution capability used by both the model and
host I/O. Each accepted submission waits on the preceding submission and signals
the next timeline value, including transfers across different exact queues.
This deliberately serializes shared scratch use. The returned integer is scoped
to that execution object, not a global completion handle. Queue rejection does
not advance the accepted frontier. The host drains accepted work before
recycling borrowed host payloads and observes asynchronous failures explicitly.

The GPU integration test compiles Qwen generation-state kernels, command programs,
and a VM entry from source. Two retained rows share code, commands, and one VM
process while one row pauses and resumes. Checked outputs cover token history,
position, padding, and EOS. A rejected native binding must leave the execution
domain usable. This is an ownership/control witness, **not a full Qwen model
run or a performance result**. The test artifacts currently target gfx1151.

From the worktree root:

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve:control_test
```

Weights, KV pools, model forward stages, scheduling policy, and transport are
separate from this coarse execution boundary. None is inferred from buffer
contents or command names by the native submission code.

## Shared Qwen execution

`qwen_model.{h,c}` owns a concrete Qwen3.8-27B UD-Q5_K_XL residency: one GGUF
parameter slab, prepared prefill/decode commands, model VM process, residual
buffer and packed workspace. One preallocated arena partitions private retained
state among up to eight rows. Rows are data, not VM processes. A single host
owner multiplexes their stages through the shared execution timeline.

`qwen` is the CLI caller, not an HTTP service. It round-robins active input
chunks and decode steps, supports retained follow-up turns, and can reset/reuse
the same rows for repeated runs without reloading weights. This is stage-level
interleaving, not batched model math. Greedy selection and device position/history
updates are compiled stages. The host stops on EOS or its requested token limit.

Model math is supplied as separately compiled artifacts, not built into this
binary. Each stage directory contains:

```text
manifest.json                 Version 2 loom-command-set manifest, one root.
config.json                   Compile-time model capacities from the stage driver.
commands/<artifact>           Portable command named by that root.
kernels/<request-stem>.hsaco   Compiled image for each manifest entry.
```

The manifest supplies entry names and the root-local entry mapping. It is not
an execution language: command order, arguments, parameter placement, and
transient requirements come from the portable command artifact. Both stages
must place every parameter identically; the loader compares their keys,
fixed-buffer indices, offsets, lengths, and alignments before sharing weights.
Allocation satisfies both stages' published slab size/alignment requirements.

The [model sources and compiler driver](models/qwen38/README.md) reproduce these
artifacts. Configuration stays with its compiled stage; the loader checks that
both stages declare the same context capacity. Prefill accepts active chunks
up to the compiled launch capacity (at most 512); the caller enforces remaining
context capacity. The seven rebindable slots are:

| Slot | Buffer contract |
| --- | --- |
| 0 | Shared residual rows: 512 × 5120 f32 values. |
| 1 | Three i32 values: prefill count/base/EOS, then decode position/count/EOS. |
| 2 | GDN convolution and recurrent state: 156,893,184 bytes. |
| 3 | Sixteen attention layers' KV state: context capacity × 65,536 bytes. |
| 4 | 512 i32 token IDs; after prefill, current token at 0 and a 128-token history ring from 1. |
| 5 | Eight i32 progress values; generated count at 0 and EOS flag at 7. |
| 6 | Shared scratch sized/aligned for the larger requirement of both stages. |

Slots 1 through 5 are private arena spans; 0 and 6 are shared. The current roots
use contiguous per-row KV, not paged KV. At a 16K context, two retained rows use
2.292 GiB, alongside the 18.504 GiB parameter slab and shared scratch. The host
does not infer these model semantics from kernel reflection.

The selected token returned by a stage has not yet entered the recurrent/KV
state. Decode consumes it; another prefill can instead discard it. A retained
follow-up consumes the last delivered token before appending new role markers
and input. This distinction is essential at EOS and tool-result boundaries.
The CLI retains output tokens on the host while the fixed device ring wraps.

```sh
build_tools/bin/iree-bazel-run --config=asan //experimental/loom_serve:qwen -- \
  --prefill=/path/to/compiled/prefill \
  --decode=/path/to/compiled/decode \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --prompt='The secret word is MAPLE. Remember it and reply only READY.' \
  --prompt='The secret word is COBALT. Remember it and reply only READY.' \
  --followup='What is the secret word? Answer with only that word.' \
  --chunk_size=7 --iterations=2 --max_tokens=16
```

The example exercises distinct retained rows, short/padded chunks, follow-ups
and reset/reuse. Each iteration should produce READY/READY then MAPLE/COBALT.
Omitting `--chunk_size` uses the compiled capacity and avoids artificial tiny
chunks. `--prompt_file` accepts an already rendered chat transcript for replay
comparisons. `--rows` repeats the supplied prompt set across a fixed row count.

JSON metric lines on stderr report consumed input, selected output (including
EOS), prefill/decode step counts and completed-stage durations. Model TTFT runs
from round admission through the final input chunk's first prediction; it
includes other rows' intervening work. Round-robin duration covers the whole
set. Cold model creation and final text printing are outside those durations.
For a warm decode rate, divide decode steps by decode seconds; aggregate rate
uses the whole set's elapsed time, not the sum of per-row rates.

All device backing is allocated before generation. The current host stage API
waits for readback and is single-owner; transport can run independently. These
checks do not establish zero allocations inside VM/HAL submission or batched
throughput. Controlled performance uses an optimized binary and benchmark lock,
not ASAN correctness timings.

## Local TCP transport

`http_server.{h,c}` runs raw `iree/net` TCP carriers on a standard proactor
thread, independently of blocking model stages. Its application owner claims
fully framed requests, sends copied response bytes with bounded send credit,
and finishes or aborts each connection. The transport has no model or chat
semantics. A connection carries one HTTP/1.1 request and a close-delimited
response; retained sessions belong above this connection lifetime.

The listener binds loopback only. Request framing bounds headers and body
storage, rejects ambiguous framing and pipelining, and retains request views
until application release. Sixteen connection slots bound admission; excess
connections are diagnosed and closed. Send completion returns credit, and
deactivation joins outstanding operations before a slot is reused. Shutdown
joins both the accept target and its cancellation receipt before stopping the
poll owner. An unrecoverable proactor failure aborts the experimental process
instead of reclaiming storage whose I/O retirement cannot be established.

The executable blocks handled signals before creating any threads. SIGINT and
SIGTERM request application shutdown, allowing the model owner to finish its
current stage before releasing the transport and model. Client errors are
local to that peer; the model scheduler chooses its safe cancellation boundary.

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve:http_request_test \
  //experimental/loom_serve:http_server_test
```

The loopback checks use real sockets/carriers and cover fragmented requests,
ordered streaming bytes, half-close, repeated slot reuse, peer reset and
shutdown with receive/accept work pending. They establish transport ownership,
not a working chat endpoint or a pi session.
