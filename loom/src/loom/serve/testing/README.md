# Serving test support

These `testonly` libraries support native lifecycle tests and model-authoring
checks. The server has no Python dependency, and this package is not a client
SDK. Integration tests stay beside the server or service they exercise.

`model.h` owns decoded model documents for native tests. `stalled_stderr.h`
reproduces a full Linux stderr pipe so a lifecycle test can prove that a blocked
diagnostic sink cannot stop inference on its only worker.

The test-only native text request observer loads a source VM policy and
tokenizer, prepares a real chat request and reports its tokens and completed
transcript. Both the deterministic text fixture and external model checks use
this path without loading numerical weights or acquiring a device.

The local `loom_serve_testing` Python package contains these helpers:

- `http_harness` owns child-process startup, HTTP control calls, joined shutdown,
  and explicitly fallible telemetry history. Readiness and completion come from
  server state and counters, not sleeps or promised event retention.
- `logs` owns an observer's cursors. Gaps, model removal and stream reincarnation
  remain explicit; a reader never owns server resources.
- `chat` consumes complete chat/SSE responses, validates terminal outcomes and
  records monotonic request, first-text and output-chunk times. An optional
  content observer receives answer deltas with their receipt timestamps while
  the response is live, allowing a workload to feed a downstream consumer.
  Reasoning and tools remain separate, and observer time is part of the sample.
- `workload` replays frozen requests with independent arrivals, completion-linked
  followups, and bounded client concurrency. Its `replay` command writes JSONL
  observations and per-model/endpoint summaries.

Bazel callers depend on the required library; its `imports` attribute supplies
the Python package root. CMake carries the same import directory through the
dependency. A standalone model-check environment can add this directory to
`PYTHONPATH`, without installing or modifying the Loom compiler's Python
authoring package. Each helper's unit tests live beside its implementation;
the real command and modality tests exercise the same helpers end to end.

## Replaying a mixed workload

The replay tool targets an already-running HTTP(S) server. It neither launches
models nor resets their caches, so the same input can test a cold deployment,
retained multi-turn work, or alternate execution variants. Server configuration,
cache state, and hardware exclusion belong in the experiment record.

A JSONL workload specifies complete request bodies, not prompt templates. Each
record has a unique `id`, an inference `path`, a `body` containing the model, and
optional `headers`. Arrivals use either `at_ms` relative to workload start, or
`after` naming an earlier request plus an optional `delay_ms` after that request's
HTTP completion. These examples can be saved together as `workload.jsonl`:

```json
{"id":"agent-a-1","at_ms":0,"path":"/v1/chat/completions","headers":{"X-Loom-Session":"agent-a"},"body":{"model":"coder","messages":[{"role":"user","content":"Say hello."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":32,"temperature":0}}
{"id":"search-1","at_ms":100,"path":"/v1/embeddings","body":{"model":"search","input":["find the tokenizer","explain its tests"]}}
{"id":"agent-a-2","after":"agent-a-1","delay_ms":250,"path":"/v1/chat/completions","headers":{"X-Loom-Session":"agent-a"},"body":{"model":"coder","messages":[{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Now say goodbye."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":32,"temperature":0}}
```

```bash
iree-bazel-run //loom/src/loom/serve/testing:replay -- \
  --server=http://127.0.0.1:8080 --workload="$PWD/workload.jsonl" --concurrency=16 \
  > results.jsonl
```

The same command is available as `python -m loom_serve_testing.replay` in an
environment with this test package on `PYTHONPATH`. It supports streaming
single-choice chat, embeddings, image generations, and System One JSON requests.
The full manifest is validated before sending anything. Bodies and policy fields
are preserved; the replay does not replace the recorded assistant turn with a
newly generated one. A live agent run is a different experiment.

Client concurrency bounds submitted HTTP requests. Requests delayed by that cap
remain queued, with `client_lag_ms` measuring scheduled arrival to actual launch.
`request_ms` measures the launched HTTP lifetime, and
`arrival_to_completion_ms` includes both. A client-limited run is identifiable
rather than silently interpreted as server throughput. Each stream also reports
`first_output_ms` (reasoning, answer text, or a tool fragment), `first_text_ms`
(answer text only), and output-chunk times. Transport chunks are not tokens;
token totals and rates use only server-reported usage. Missing usage remains
missing, including for image and decision endpoints.

The output starts with a workload fingerprint, emits each terminal request as
it is collected, and ends with per-model/endpoint summaries. Summaries use
nearest-rank percentiles; completion latencies include successful requests only,
while launch lag includes all attempted requests. Rates use the entire workload
duration, including declared idle periods. All client timestamps use the same
high-resolution performance clock. When a run resolves no elapsed interval,
counts remain available and rates are `null`; the tool does not invent a
minimum duration. A nonzero exit means at least one request failed or was blocked
by a failed predecessor. Independent requests keep
running; requests are never retried. The outer job owns any deadline.

Request hashes identify body/header changes without retaining their contents.
Response hashes omit transport timestamps, IDs, and usage; they expose divergent
outputs without retaining generated images or transcripts. They are not quality
oracles. Different valid outputs, including a different encoded image with the
same pixels, can have different hashes. A model-specific oracle still judges
quality, and token-work changes must be considered when comparing times.

The native mixed-catalog tests exercise this runner against source-JIT text,
image, embedding, and decision fixtures, including a cached text continuation.
Their timings establish no learned-model performance claim. Measured comparisons
use optimized server builds, matched request/settings and retained-state policy,
and an excluded measurement runner; sanitizer runs establish correctness only.
