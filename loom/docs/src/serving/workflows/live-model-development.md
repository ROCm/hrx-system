# Develop models on a live server

`loom-serve` can JIT new execution programs while retaining a model's weights
and state. An author can check and benchmark a change, upload it from another
machine, compare outputs, and select or roll back the executable without
restarting the server. This works for text, image, embedding and decision
services, including command programs, their kernels and direct-dispatch
implementations. The client needs HTTP access, not write access to the server
filesystem. The text example below compares continuations from one pinned
checkpoint; finite requests use the same staging and selection endpoints.

This is the serving half of [agent-driven kernel development](../../workflows/agent-driven-kernel-development.md).
Kernel checks establish numerical behavior; compiler reports explain the
emitted implementation; live requests establish whether the change improves
the model workload. Keeping those results together makes the loop repeatable
by another person or agent.

## Establish the resource contract

Start with a working [model package](../model-programs.md), its
matching parameter checkpoint, installed Loom tools, `curl` and `jq`.
A checkpoint alone does not supply a model implementation. The examples use
a text deployment named `scout`; its package has `prepare.loom`, `control.loom`,
`weights.loom`, `tokenizer.json` and two native providers, `config.loom` and
`advance.loombc`. Substitute your package's complete file list and paths.

Source uploads and live native selection work across these serving modalities:
prepared deployments advertise `loom.variants` when enabled.
It preserves preparation declarations, the stage/shape catalog, control VM,
tokenizer, parameter interpretation and retained-state layout. The author
asserts compatibility; binding checks do not prove mathematical equivalence.
Changes to those preserved contracts require a new registration, not a native
variant. A compatible optimization can change instructions, traversal, fusion
or scratch requirements within the existing stage binding contract. An image
candidate can also change the number of command submission parts without
replacing the control VM. Each image keeps one execution choice through its
last part and CPU encoding; finite-vector cohorts likewise finish before
selection. Prepared but unsubmitted requests can use the new choice.

In the model document, set `name` to `scout`,
`definition.checkpoint_capacity` to at least one and
`definition.variant_capacity` to at least four. Variant slots retain the original,
candidates and failed attempts. Context and pool capacity must also accommodate
the prompt and continuation. The [model document reference](../model-documents.md#text-definitions)
owns these deployment settings, independent of server process lifetime.

## Register a model from uploaded sources

Start an empty server on the inference machine with an explicit device:

```sh
loom-serve --device=legacy-amdgpu --listen=127.0.0.1:8080
```

The device must support the model's programs. Other device and backing choices
are in the [command reference](../index.md). Use an existing
server instead when one is already running. HTTP administration assumes trusted
clients and code; a tunnel or deployment-owned boundary supplies access control.

On the authoring machine, `baseline/model.json` is the complete deployment
document. Its `definition.weights` field names a file on the inference machine,
either absolutely or relative to an explicit server-local `base_directory`.
Relative program and metadata references instead name members of the uploaded
package:

```sh
server=http://127.0.0.1:8080

curl --fail-with-body --silent --show-error -H 'Expect:' \
  "$server/v1/models/register" \
  -F 'request=@baseline/model.json;type=application/json;filename=scout/model.json' \
  -F 'file=@baseline/prepare.loom;filename=scout/prepare.loom' \
  -F 'file=@baseline/control.loom;filename=scout/control.loom' \
  -F 'file=@baseline/weights.loom;filename=scout/weights.loom' \
  -F 'file=@baseline/config.loom;filename=scout/config.loom' \
  -F 'file=@baseline/advance.loombc;filename=scout/advance.loombc' \
  -F 'file=@baseline/tokenizer.json;filename=scout/tokenizer.json'

curl --fail --silent --show-error "$server/v1/status"
```

HTTP 202 accepts registration; preparation and JIT continue asynchronously.
`/v1/status` reports progress and errors. An inference request can wait for
preparation within the admission budget. A drained registration retains its
upload without preparing until activated or undrained.

`filename` defines a logical package path, not a destination on server disk.
The request's directory anchors relative references, including references to
shared providers elsewhere inside the package. Missing members fail rather
than finding a similarly named file on the server. Absolute program or metadata
paths explicitly select server-local files. This also permits keeping a large
tokenizer beside the weights instead of uploading it.

Both `.loom` and `.loombc` are JIT inputs; bytecode is portable IR, not an AOT
device binary. Uploaded bytes are retained in memory through their owning
registration or compilation. They are not a persistent package catalog.
The complete multipart body must fit `--request_body_bytes` (8 MiB by default).
`-H 'Expect:'` suppresses curl's optional continue handshake; ordinary file
parts supply the required content length. Parameter checkpoints stay on the
server. The [upload reference](../http-profiles.md#uploaded-source-packages)
owns exact framing and path rules.

## Save a repeatable continuation

Use a deterministic request policy while checking the live loop. For a model
supporting greedy sampling, set `temperature` to zero and preserve the same
thinking, tool and template options for every trial. First complete a seed turn:

```sh
jq -n '{model:"scout", temperature:0, max_tokens:16,
        messages:[{role:"user", content:"Explain briefly why the sky is blue."}]}' \
  >seed-request.json

curl --fail-with-body --silent --show-error "$server/v1/chat/completions" \
  -H 'Content-Type: application/json' -H 'X-Loom-Session: seed' \
  --data-binary @seed-request.json >seed-response.json

curl --fail-with-body --silent --show-error -X POST \
  "$server/v1/checkpoints/comparison" \
  -H 'X-Loom-Model: scout' -H 'X-Loom-Session: seed' >checkpoint.json

jq -n --slurpfile request seed-request.json --slurpfile response seed-response.json \
  '$request[0] | .max_tokens = 32 |
   .messages += [$response[0].choices[0].message,
                {role:"user", content:"Continue the explanation in one sentence."}]' \
  >continuation.json

original=$(curl --fail --silent --show-error "$server/v1/models/variants" \
  -H 'X-Loom-Model: scout' | jq -er '.selected')
```

The pin captures an immutable consumed frontier and its pending token. Each
trial still supplies full history; `X-Loom-Checkpoint` requires it to extend
that endpoint. A fresh destination session forks the checkpoint rather than
continuing whatever a previous trial wrote. The pin stays protected until
explicitly released. Its state cost depends on the model, including recurrent
state; it is not necessarily just a small KV reference.

The next request's chat template must preserve that pinned token prefix.
Removing earlier reasoning or rewriting prior messages can break this contract;
a strict checkpoint request rejects a mismatch instead of silently replaying.

## Check and measure a candidate before uploading

Keep the baseline inputs unchanged and author the replacement providers in
`candidate/`. State one hypothesis: for example, a traversal change should
reduce repeated weight traffic without changing parameter or KV layout.
The check module below links the candidate implementation and its independent
oracle and declares `production_cut` and `production_bucket`:

```sh
loom-format --check candidate/cut.loom
loom-test candidate/cut.loom --case=@production_cut --device=legacy-amdgpu \
  >candidate.test.json
loom-benchmark candidate/cut.loom \
  --benchmark=@production_bucket --device=legacy-amdgpu \
  --measure=dispatch_complete --output=candidate.benchmark.json
```

These symbols belong to the authored checks, not to the serving ABI. Functions
can be checked directly; kernel trials use the appropriate independent oracle.
Representative tails, span counts, aliasing and packed storage matter as much
as the convenient full tile. [Correctness](../../workflows/test-correctness.md) and
[benchmarking](../../workflows/benchmark.md) describe selection, libraries and measurement
policy. Access-sanitized checks and uninstrumented performance runs are
separate evidence.

[Capture compile reports](../../workflows/compile-reports.md#capture-one-report) for the
same production root, target and specialization, then inspect the change:

```sh
loom-compile-report show candidate.report.json
loom-compile-report diff baseline.report.json candidate.report.json
```

Look for the mechanism behind the score: memory requests, spills, matrix
instructions, register/LDS pressure, dispatch count and scheduling. A reduced
instruction count or improved occupancy estimate is a hypothesis about speed,
not a substitute for measurement. A faster isolated kernel also needs a live
model comparison with representative prompt/decode mixtures.

## Upload a candidate without selecting it

Staging names a ready base and the **complete** native provider set, not a patch
to the base's source catalog. It does not upload or rerun preparation/control.
For the example's two-provider catalog:

```sh
jq -n --argjson base "$original" \
  '{model:"scout", base:$base, sources:["config.loom","advance.loombc"]}' \
  >candidate.json

curl --fail-with-body --silent --show-error -H 'Expect:' \
  "$server/v1/models/variants" \
  -F 'request=@candidate.json;type=application/json;filename=scout/candidate.json' \
  -F 'file=@candidate/config.loom;filename=scout/config.loom' \
  -F 'file=@candidate/advance.loombc;filename=scout/advance.loombc' \
  --output=staged.json

candidate=$(jq -er 'select(.variant.phase == "ready") | .variant.id' staged.json)
jq '{selected, variant}' staged.json
```

The request waits asynchronously for JIT by default. Other inference can
progress, and `selected` remains unchanged. `wait:false` instead returns
HTTP 202 with an ID; a later query observes `compiling`, `ready` or `failed`:

```sh
curl --fail --silent --show-error "$server/v1/models/variants" \
  -H 'X-Loom-Model: scout' -H "X-Loom-Variant: $candidate"
```

A failed waiting JIT returns 422 with retained diagnostics, including package
source names and available line/column locations. `--fail-with-body` preserves
that JSON in `staged.json` while failing the command. It never promotes failed
code. Disconnecting does not cancel an accepted compilation; inventory reveals
its outcome. Failed attempts occupy variant credit until removed.

## Compare from the same checkpoint

Selection is model-wide. Subsequent epochs, including those of already-admitted
requests, may use the new code. Coordinate with other authors and isolate the
evaluation workload; there is no request-local variant selector here. A
selection response completes at a safe execution boundary, without reloading
weights or copying KV state. The epoch boundary described here is specific to
text; image and finite-vector services use the whole-invocation boundaries
described above.

In a Bash session, these helpers retain complete responses and use
expected-current protection on each selection:

```bash
selected=$original

select_variant() {
  local next=$1
  curl --fail-with-body --silent --show-error \
    "$server/v1/models/variants/select" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"scout\",\"variant\":$next,\"expected\":$selected}" || return
  selected=$next
}

continue_from_pin() {
  local label=$1
  curl --fail-with-body --silent --show-error "$server/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "X-Loom-Session: trial-$label" -H 'X-Loom-Checkpoint: comparison' \
    --data-binary @continuation.json --output="$label.json" || return
  jq -S '{choices, usage}' "$label.json" >"$label.comparison.json"
}

continue_from_pin A1 &&
select_variant "$candidate" && continue_from_pin B1 &&
select_variant "$original" && continue_from_pin A2 &&
select_variant "$candidate" && continue_from_pin B2 &&
select_variant "$original" && continue_from_pin A3

diff -u A1.comparison.json B1.comparison.json
diff -u A1.comparison.json A2.comparison.json
diff -u A1.comparison.json B2.comparison.json
diff -u A1.comparison.json A3.comparison.json
```

Exact comparison is useful for a semantics-preserving deterministic change.
Response IDs and timestamps differ, so the comparison keeps choices and usage.
`prompt_tokens_details.cached_tokens` should show the common pinned prefix.
Equal decoded text is not proof of identical logits or raw generated token
IDs; the HTTP completion exposes neither. Numerical checks own the acceptable
error envelope. Stochastic or intentionally approximate changes need a quality
criterion beyond byte equality.

A 409 selection response means the expected choice is stale or another
selection conflicts. Inventory identifies the current choice; blindly retrying
with a new expected ID would overwrite another author's work. An interrupted
selection response also requires inspection before the next change.

## Measure the serving benefit

The same alternating sequence can compare performance after JIT has finished.
Run on an optimized, uninstrumented server with controlled device ownership.
Keep input histories, output limits, request arrival pattern, sampling, cache
policy and background load matched. Repeat both warmup and measured workloads
for each choice, using fresh destination session IDs for every trial. Report
the spread, not just the fastest sample.

For text, measure first emitted token, completion latency and aggregate delivered
tokens per second across the concurrent cohort. HTTP time-to-first-byte can be
a header or role-only event; it is not automatically token TTFT. Rejected draft
tokens are work but not useful output. A cached checkpoint comparison measures
continuation behavior, so separate fresh-session trials are needed to establish
prefill gains.

For images, replay identical prompts, seeds, dimensions, step counts and adapter
settings against each choice. Compare decoded pixels or a stated image-quality
criterion, not response IDs or timestamps. Include output encoding in end-to-end
latency. For embeddings and decisions, replay the same inputs and batching
conditions and compare returned vectors or decisions under the model's numerical
tolerance. Report both completion latency and aggregate useful outputs per
second. These finite requests need no conversation checkpoint; the retained
weights and unchanged request contract provide their common starting point.

The [observation workflow](observe-model-server.md) connects client results to
`/logz/stream`, counters, epoch packing and memory accounting. Comparing
`weight_activity` snapshots before and after a swap can show whether additional
activation, file reads or preparation occurred. Compiler JIT diagnostics are
not kernel performance reports, and serving-managed memory is not total process
or device memory. Keep these measurement boundaries explicit.

## Promote or roll back and release the experiment

After the comparison above, the original remains selected. Accept the candidate
with `select_variant "$candidate"`; reject it by leaving the original selected.
Keep both ready choices while rollback is valuable. Rollback is
`select_variant "$original"` using the last acknowledged selection as the
expected ID. Switching code does not undo emitted output or state already
written; the pinned checkpoint supplies the earlier state for a new branch.

Once no trial needs its pin or unused candidate, release them explicitly:

```sh
curl --fail-with-body --silent --show-error -X DELETE \
  "$server/v1/checkpoints/comparison" -H 'X-Loom-Model: scout'

# This example rejects B after the A/B/A/B/A comparison.
curl --fail-with-body --silent --show-error "$server/v1/models/variants/remove" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"scout\",\"variant\":$candidate}"
```

Removing a selected, compiling or borrowed choice returns a conflict instead of
invalidating active work. Keeping an accepted candidate selected permits
removing the unused original in the same way. IDs belong to the current server
process, not a durable package version.

The author's durable result is the accepted source package and deployment
document, its correctness and benchmark evidence, and the workload used for
the decision. Updating that package makes a later registration reproduce the
winner. The running server supplies a place to try the change with live weights
and state; the retained evidence lets the next agent improve it again.
