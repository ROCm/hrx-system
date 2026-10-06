# Source-defined autoregressive serving

`//experimental/loom_serve/text:server` and `:generate` are model-independent
executables. They have no dependency on a model package or embedded source
catalog. `--model=/path/to/package` is required. Model files are JIT inputs,
not build inputs to these binaries.

The [Qwen package](../models/qwen/README.md) is the real-weight implementation
of this contract. Its numerical kernels, stage composition, storage geometry,
checkpoint policy, tokenizer framing and chat policy live in Loom source.
Its remaining C programs are test-only numerical/replay observers.
The [image runner](../image/model.h) provides a separate source-defined
diffusion contract over the same JIT, weight and execution infrastructure.

## First output and first port

From the repository root:

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/text:server -- \
  --model=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --rows=4 --context_capacity=32768 --pool_capacity=65536 \
  --prefill_capacity=512 --mtp --mtp_depth=3
```

The real Qwen sources currently target gfx1151. Repository build/execution
policy still applies; a successful build on another host is not a GPU check.
The [reproduction packet](../docs/README.md) supplies assets, request examples
and the generated-tool/retained-session witnesses.

A new text model supplies `prepare.loom`, `control.loom`, `sources.txt`,
the providers named by that catalog, and its weight-policy source. Catalog
paths are relative to the package; shared motif paths must travel with it.
There is no native registration list. Editing the package changes the next
process without rebuilding the server.

The first numerical slice is one real-weight block through its actual command
root and source catalog, then a complete retained continuation. After that,
the same source runs under HTTP admission and mixed scheduling. Copying the
Qwen package is a starting scaffold, not a model conversion: its dimensions,
layer graph, layouts and checkpoint names must be replaced with the target
model's actual contract.

## Execution contract

This is a bounded packed-autoregressive runner, with optional three-token
speculation. It is not an arbitrary model-graph interpreter.

* One residency shares a VM process, cached commands, weights and workspace.
  Sessions are data rows. Calls to the shared invocation are serialized.
* The host chooses ready spans and a cached token/span shape. Source creates
  device packets and chooses commands. Attention and recurrent state layout
  are source concerns; the host owns physical page IDs and completion credit.
* Known input, one-token decode and speculative verification can share an
  epoch. The runner currently supports 1–16 resident rows, up to 512 input
  tokens per epoch, and a logical context ceiling of 262144 positions.
  Physical pool capacity is independent, up to 4194304 positions, divisible
  by the source-declared page size.
* MTP is optional. Depth three consumes one anchor and three proposals;
  the source must implement proposal, target verification, accepted-state
  publication and draft catch-up. A model without this path runs without
  `--mtp`; merely declaring MTP does not implement it.
* `--continuation_epochs=2` pre-issues a second device-fed cohort. The host
  still joins each cohort; this is not an autonomous persistent device loop.
* Pooled execution reserves stable device addresses at startup and backs
  row/page ranges on demand. Admission reserves completion high-water credit
  and queues excess work. These are private pages, not shared prefixes or
  offloaded session checkpoints. Fixed backing remains an explicit comparison
  and device-sanitizer configuration.

The automatic shape catalog crosses token classes up to `--prefill_capacity`
with span classes up to `--rows`, including exact odd endpoints. Repeated
`--epoch=tokens:spans` replaces that catalog. Source may reject unsupported
specializations; the runner never substitutes another model or stale code.

## Bootstrap and storage

The cold entry in `prepare.loom` is:

```text
prepare(prefill:i64, context:i64, pool:i64, rows:i64, mtp:i32,
        shapes:buffer, shape_count:i64, weights:buffer)
  -> (prefill_capacity:i64, shared_stage_count:i32, state:buffer,
      terminal_spelling:buffer, allocations:buffer, row_views:buffer,
      target_origins:buffer, draft_origins:buffer, geometry:buffer)
```

`shapes` contains little-endian i64 `{token_capacity, span_capacity}`
pairs. The source declares command roots, configuration overrides and
checkpoint bindings through the [preparation capability](../runtime/preparation.h).
The first `shared_stage_count` stages have identical target parameter
placement and seven dynamic bindings. All stages use one checkpoint domain,
with at most eight dynamic bindings and workspace last when present.
Auxiliary roots may share existing tensors or add draft tensors.

`state` is opaque to C and survives cold-VM teardown. The terminal spelling
is looked up in the supplied tokenizer. The current token protocol has one
terminal token ID. Geometry is not inferred from checkpoint filenames.

Storage records are little-endian i64:

| Result | Records and meaning |
| --- | --- |
| `allocations` | Eleven `{byte_length, alignment, initial_zero_length}` records, in the order below |
| `row_views` | `rows * 5` pairs `{offset, length}` into the state arena: control, recurrent, dense attention, input IDs, progress |
| `target_origins` | 16 pairs of source-defined byte origins, uploaded to the target row table |
| `draft_origins` | 16 pairs uploaded to the draft row table when enabled |
| `geometry` | Header `{page_tokens, page_map_byte_origin, draft_carry_stride, feedback_split}`, then zero or more cache-region records described below |

The eleven allocation slots are residual, state arena, packed metadata,
target origins/page map, packed input IDs, ordinary selected IDs, draft carry,
committed metadata, verification feedback, draft cache, and draft origins/page
map. The first two are present; packed slots 2–5 exist with epoch shapes;
draft slots 6–10 exist with MTP. Workspace size and alignment come from command
reflection, independently of this list.

Pooled sources append cache regions, each five little-endian i64 values:
`{allocation_slot, byte_origin, plane_count, plane_stride, block_bytes}`.
A physical block ID selects `block_bytes` consecutive bytes in every plane:
`byte_origin + plane * plane_stride + block_id * block_bytes`. Regions describe
actual kernel addressing; native code has no layer count or KV element type.
Dense sources have no regions. Region extents must fit their allocation and
each plane's block range must fit its stride.

Pooled CLI execution defaults to `--pool_backing=elastic`: source allocations
containing cache regions reserve stable addresses and commit physical slabs as
rows/pages grow. Private row views initialize on first use. `--slab_bytes`
selects coarse physical granularity (2 MiB by default; zero queries the
allocator recommendation), independent of logical token-page size. The explicit
`--pool_backing=fixed` comparison path backs all source state at startup and
supports device address sanitization, which excludes user VMM. Unsupported VMM
is an error, not a silent change of storage policy. Host ASAN remains usable.
Elastic backing also reserves stable parameter roots in the same physical
allocation domain. Heartbeat `elastic_state` and `elastic_parameters` account
mutable state and weights independently; transient workspace is separate.

`loom_serve_text_model_deactivate` retires consuming work and releases parameter
backing without discarding retained rows or rebuilding commands. The next
inference call activates on demand; `loom_serve_text_model_activate` can warm
explicitly before admission. Activation reuses the checkpoint provider and
preparation commands, streaming original file bytes into the same final roots
and preparing them in place. It neither JITs nor holds a second weight copy.
An already active model does no loading/preparation. Fixed backing rejects
deactivation explicitly. The `models/qwen:epoch_check --reload_weights` witness
checks real target/MTP continuation across compaction and parameter eviction.

`loom_serve_text_model_trim` compacts owned blocks into a live ID prefix and
returns empty physical slabs. Source regions drive bounded device-copy batches;
KV never reads back to the host. Copies retire before host and device maps
change, and map uploads retire before old backing is unmapped. Live recurrent
state and partially occupied slabs stay backed. Reset rows release their private
backing at this maintenance cut and initialize again on reuse. The operation
does not evict live history, rebuild commands, or change virtual addresses.
Its result separates relocated blocks, copied bytes, and physically released
bytes. The real-weight `models/qwen:epoch_check --trim` witness compares an
uncompacted continuation, including MTP, then checks full trim and regrowth.
The HTTP service coalesces evicted and cancelled rows into one maintenance cut
before the next cohort. `state_trim` JSONL records expose the copied/released
bytes and elapsed maintenance time. Active and retained idle histories survive;
normal epochs without ownership loss do not perform this maintenance.

Each row's control/input/progress and recurrent views are present. Dense
attention is present only without a physical pool; pooled kernels find it
through source origins. The recurrent view is the zero-resettable private
state region, not a hardcoded Gated DeltaNet layout. The current materializer
requires a nonempty recurrent view even for an attention-only model. Such a
port must explicitly account for that resettable span in its storage plan;
the contract does not allocate a Qwen-sized state behind the model's back.
This exact contract is implemented by [model.c](model.c) and exercised with
the real [bootstrap](../models/qwen/prepare.loom) and its
[layout checks](../models/qwen/prepare_test.cc).

Initialization uploads and returned source buffers remain owned through
queue retirement. A failed partial creation drains accepted work before
releasing backing.

## Warm numerical entries

The signatures and complete source implementation are in
[control.loom](../models/qwen/control.loom). The host-to-source semantic plan
is independent of the model's device packet format:

* A plan has 32 little-endian i32 records of eight fields:
  `{length, position, row, input_begin, input_count, flags, output_credit,
  first_span}`. The first cohort occupies records 0–15 and the continuation
  starts at 16. Flags are SELECT=1 and PROPOSE=2.
* `encode_epoch` receives that plan, contiguous known token IDs, both span
  counts and the terminal ID. It fills the opaque metadata/input upload
  buffers and returns ordinary output count.
* `epoch` receives opaque state, shape/selection/speculation facts and the
  thirteen retained device buffers. It invokes `runner.execute_N` and
  `runner.feedback`; it does not wait or allocate new device storage.
* After retirement, `publish_epoch` translates opaque feedback into
  `{consumed, known, outputs, verifications, tokens[8]}` i32 records in
  original span order. Only accepted state contributes to consumed progress.
* The isolated `step` receives state, a prefill/decode selector and seven
  buffers: residual, control, recurrent, attention, IDs, progress, workspace.
  Control holds `{input_count, absolute_position, terminal_id}`.
  Selected ID is at input slot zero; progress slot zero is selected count
  and slot seven is the terminal flag.

The last selected token has not yet been consumed. Retained HTTP appends
preserve that original ID separately from canonical text. Re-tokenizing a
canonical checkpoint and treating it as already consumed corrupts that
invariant when tools or whitespace were normalized.

## Chat policy entries

All text policy lives in the same warm program as numerical execution:

| Export | Inputs → result |
| --- | --- |
| `model_name` | none → UTF-8 identity buffer, retained by the native policy |
| `chat_begin` | original request JSON, raw tools array → prefix buffer, opaque i64 rendering state |
| `chat_message` | original message JSON, raw role spelling, decoded content, decoded reasoning, rendered calls, state, ordinal, presence bits → fragment buffer, next state |
| `chat_end` | state → final generation-prefix buffer |
| `render_tool` | name, decoded argument JSON, call ordinal → canonical buffer and used byte length |
| `parse_tools` | raw schemas, generated suffix → JSON array of function-call records |
| `prepare_input` | text, format:i32, boundary:i32, capacity:i64 → little-endian i32 IDs and complete count:i64 |
| `text_end` | accumulated response, previous published extent:i64, phase:i32 → safe byte extent:i64 |
| `complete_text` | canonical prompt, final ordinary content, canonical calls → owned canonical checkpoint buffer |

Unmarked types above are buffers; rendering state, ordinal and presence bits
are i64. Presence bits are reasoning=1, tool_calls=2 and tool_call_id=4.
Raw JSON remains available so a policy can inspect names, IDs or fields not
represented in the decoded text. `chat_begin` receives the whole request;
a policy requiring whole-history traversal can render there and return empty
per-message fragments. C does not interpret its rendering state.

The current HTTP protocol is streaming greedy text, with up to 256 messages
and 16 function tools. Text content may be a string, null, or an array of text
parts. Unsupported sampling, strict constrained output, stored responses and
image parts fail explicitly. `model_options` is an opaque JSON object for
model policy; `enable_thinking` is forwarded too. The Qwen policy rejects
thinking and nonempty model options; another source policy may implement them.

Tool parsing returns records of the form
`{"type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}`.
The host assigns IDs and serializes SSE; source owns generated grammar and
canonical separators. `tools.parse_xml` is an optional bulk codec selected
explicitly by the Qwen program, not the default interpretation of text.
The independent [plain-text/JSON policy](../models/qwen/testdata/portable_chat.loom)
is tested through the same native caller without linking that codec.

Input format is rendered=0 or single-user=1; boundary is fresh=0, open=1 or
terminal=2. Open/terminal refers to the separately retained pending token.
The input capability returns a bounded token prefix; complete encoding needs
the source's capacity-plus-one check. Output phase is streaming=0 or complete=1.
Published extents are monotonic and within the accumulated response.

Request fragments borrow input bytes only during invocation and are copied
before references release. Completed checkpoint arguments instead own their
backing, so the returned buffer may alias any argument safely. Cancellation,
replacement and shutdown release the retained result. No session owns a VM.

## Checks that matter for a port

The CPU-only request inspector JITs the actual chat/input/completion entries
without loading model weights or creating a device:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //experimental/loom_serve/text:request_check -- \
  --model=/path/to/source/package --tokenizer=/path/to/tokenizer.json \
  --request=/path/to/request.json --capacity=8192
```

The request file uses the same body as the HTTP endpoint, for example
`{"model":"qwen3.8-27b","stream":true,"messages":[{"role":"user","content":"Hello."}]}`
with the package's actual public name. Standard output is one JSON record with
`model`, the complete rendered `prompt`, and its `tokens`. These can be compared
directly with the model's reference tokenizer/template before any numerical
work. An input exceeding capacity fails, rather than returning a truncated
prompt as success.

Adding `--response=/path/to/raw-generated-text.txt` also reports `text_end`,
the canonical `checkpoint`, and the structured `tool_calls` delta. The file is
raw model text, not a serialized OpenAI response. This exercises the same source
completion and tool parsing used by HTTP; the observer neither generates tokens
nor claims to test GPU cache retention. Invalid input produces a nonzero exit
with diagnostics on stderr and no successful JSON record.

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve/...
```

The serving suite covers shared native mechanisms; model-local checks cover
actual source policy, parameter layout and packet semantics. A new model adds
its own real-weight numerical witness and retained HTTP continuation, including
a malformed intervening request and an independently paced second row.
The Qwen checks demonstrate this path, not another model's numerical accuracy.

The [authoring guide](../docs/MODEL_AUTHORING.md) covers motifs, source catalogs,
JIT specialization and compiler reports. The [performance guide](../docs/PERFORMANCE.md)
separates controlled throughput from instrumentation and correctness runs.
