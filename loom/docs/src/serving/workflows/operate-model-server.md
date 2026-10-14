# Operate a model server

`loom-serve` separates process lifetime, model registration, request admission
and parameter residency. A model can stay registered while drained or cold;
changing one of these states need not restart the server or discard other models.
The [first-output walkthrough](serve-models.md) establishes a running server and
a `vectors` deployment for the commands below.

## Find the state you need

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

`/v1/status` includes every registration and the server's actual resources.
`/v1/models` lists prepared deployments and their capabilities; a prepared but
drained model remains listed. `/readyz` describes the server, not readiness of
every model. The status fields answer different questions:

| Field on each model | Question |
| --- | --- |
| `loom.registration` | Is this registration declared, preparing, ready, retiring, failed or awaiting a release retry? |
| `loom.status.admission` | Is the gate open, draining, drained or removing? |
| `loom.status.phase` | What is the modality scheduler doing? |
| `loom.parameter_operation` | Is an explicit activation/deactivation waiting or in progress? |
| `loom.parameters` | What reloadable weight plans are active and how much retained backing do they account for? |
| `loom.variants` | Which native executable choice is preferred, and how many choices are retained? |
| `loom.error` | Did preparation or a lifecycle operation fail? |

The registration wire values are `registered`, `preparing`, `ready`, `retiring`,
`failed` and `release_failed`. Scheduler phases are modality-specific and are
not measurements of device utilization.

## Register without immediately consuming model memory

A model document with `"admission":"drained"` registers metadata without
reading programs or weights or allocating model storage. It still occupies one
registration slot and has an observation stream. This is useful for a catalog
larger than the set of models that can be resident simultaneously.

The normal default is `"open"`. Registration returns HTTP 202 after accepting
server-owned intent. Preparation and JIT happen asynchronously. A client
disconnect does not cancel that registration; `/v1/status` reports its outcome.
A failed source package remains inspectable without stopping healthy peers.
After correcting its inputs, unregistering and registering again starts an
explicit new attempt with a new incarnation.

The [model-document reference](../model-documents.md) applies equally
to startup `--model` arguments and HTTP registration. HTTP changes live state;
the client owns persistence of those changes into its launch configuration.

## Drain, resume or remove

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
```

Drain closes admission when the owner processes the request. Accepted model
work, including queued requests, continues. New inference, tokenizer and text
checkpoint requests wait outside the model within the holding budget. Control
and observation routes remain available. A successful drain response means the
gate changed; `admission:"drained"` confirms accepted work has finished.

Drain is reversible and does not evict weights, erase sessions or remove the
registration. Undrain reopens admission; held requests enter that modality's
normal bounded admission policy. Repeated drain/undrain is idempotent.

Unregister is irreversible:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/unregister \
  -H 'Content-Type: application/json' -d '{"model":"vectors","mode":"drain"}'
```

`mode:"drain"` is the default and lets accepted requests finish.
`mode:"abort"` cancels their intent. Both wait for actual CPU/device borrowers
to retire before releasing memory, and return HTTP 202 for accepted removal.
Graceful removal can be escalated to abort but cannot be undone by undrain.

The name stays occupied during retirement; a duplicate registration returns
409. The model leaves `/v1/models` as soon as removal is accepted, but remains
in `/v1/status` until retirement finishes and its name is released. Absence
from `/v1/status` confirms completion. With concurrent administrators, compare
`loom.incarnation`: the same name may already identify a new registration.

Held arrivals receive 410 during removal, and requests for the absent name
receive 404 after retirement. A quiescent platform release failure remains
visible in `/v1/status` as `release_failed`, retaining the resources needed to
retry. Another unregister explicitly retries that release.

## Control weight residency independently

Explicit activation requires a healthy registration with admission closed:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/activate \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

This also prepares a metadata-only registration without opening admission.
HTTP 202 accepts intent; `parameter_operation` advances through `waiting`
and `activating` to `idle`. Checking `loom.error` and waiting for `idle` establishes
the result before undrain. `parameters.active` alone means weight activations
have been enqueued, not that device work has completed.

With elastic backing, `/v1/models/deactivate` releases reloadable weights from
a healthy, drained, unpinned model while retaining compiled programs, sessions
and checkpoints. It accepts the same `{"model":"name"}` body and returns 202.
Wait for `parameter_operation:"idle"`, check `loom.error`, and inspect
`parameters.active` and `committed_bytes`. Prepared fixed-backed weights cannot
deactivate and return 409; a metadata-only drained registration is already cold.

Undrain and conflicting residency operations return 409 while an operation is
pending. Ordinary inference can reload evicted weights on demand without
re-JIT. Successful activation does not itself pin weights against later pressure.

`/v1/models/pin` and `/v1/models/unpin` accept the same model-only body and
change eviction eligibility. Pinning is not activation: a cold registration
can remember the pin without loading its weights. Status exposes this policy
as `loom.parameter_pinned`. A pending deactivation cannot be pinned.

## Replace native code without reloading weights

A prepared text deployment can JIT replacement command programs and kernels
while keeping its weights, control VM, tokenizer, sessions and checkpoints.
Staging and selection are separate: several candidates can compile without
changing inference. Explicit selection makes a ready choice preferred at a
completed execution boundary. There is no unregister/register cycle or
required admission drain.

The author asserts that replacements preserve the meaning and layout of retained
state and parameters. The server checks the native binding contract, not
mathematical equivalence. Preparation declarations, shape catalog, control VM
and tokenizer remain fixed; changing them requires a new registration.
For a text deployment named `scout` with variant capacity enabled, these commands
use `jq` to carry server-issued IDs:

```sh
original=$(curl --fail --silent --show-error \
  http://127.0.0.1:8080/v1/models/variants -H 'X-Loom-Model: scout' |
  jq -er '.selected')

candidate=$(curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8080/v1/models/variants \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"scout\",\"base\":$original,\"base_directory\":\"/srv/models/scout/candidate\",\"sources\":[\"config.loom\",\"advance.loom\"]}" |
  jq -er 'select(.variant.phase == "ready") | .variant.id')

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8080/v1/models/variants/select \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"scout\",\"variant\":$candidate,\"expected\":$original}"
```

These JSON-only commands use provider files already on the server. A
[multipart source package](../http-profiles.md#uploaded-source-packages)
instead carries client-local providers in the request. Staging waits
asynchronously by default and returns retained compiler diagnostics on failure;
`"wait":false` instead returns an ID immediately for later inspection.
Existing inference can progress during JIT. Selection briefly quiesces that
model's accepted CPU/device work before rebinding native stages; it neither
reloads weights nor copies KV state. Subsequent text epochs use the new code,
including epochs advancing already-admitted requests. This is a model-wide
preference, not a request-local A/B selector.

Keeping the original choice makes rollback another selection:

```sh
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8080/v1/models/variants/select \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"scout\",\"variant\":$original,\"expected\":$candidate}"
```

Expected-current protection prevents another author's update from being silently
overwritten. After an interrupted response, inventory establishes the actual
preference; disconnect does not cancel accepted work. Rollback restores code,
not a conversation's earlier state. Named checkpoints provide independent
state forks when comparing continuations. Native-update compatibility must hold
for every retained session and checkpoint that either choice will consume.

Unused terminal candidates can be explicitly removed to reclaim cache credit.
Pending and failed attempts count against `variant_capacity`; capacity exhaustion
rejects new staging instead of evicting an arbitrary rollback choice. The
[HTTP reference](../http-profiles.md#native-executable-variants)
defines inspection, selection and removal responses.
The [live development workflow](live-model-development.md) combines these controls
with remote source uploads, checked kernel changes and pinned continuation trials.

## Bound memory and admission

Process limits and model capacities constrain different resources:

| Limit | Resource it bounds |
| --- | --- |
| `--connections` | HTTP connections shared by all listeners. |
| `--held_requests` | Requests waiting outside closed/unprepared model gates. |
| `--model_capacity` | All registered names, including failed and retiring models. |
| Model `pending_requests` | Modality request credit; text/image count waiting requests, finite services include the active cohort. |
| Text `rows` | Concurrently advancing execution rows. |
| Text `context_capacity` | Logical token ceiling of one context. |
| Text `pool_capacity` | Shared logical KV capacity, independent of maximum context times row count. |
| Text `variant_capacity` | Native choices, including the original, pending JITs and failed attempts. |
| `--memory_bytes` | Physical retained parameter/state backing under elastic policy on one device. |

Holding capacity stays below connection capacity so held inference cannot
consume every control connection. Live log streams and waiting variant-management
requests share an additional combined bound with held requests. Full credit
returns 429. Incomplete requests and
active response streams still consume transport credit; these bounds are not
a promise that arbitrary slow clients leave connections free.

Text admission reserves a request's completion high-water. Short contexts grow
by blocks rather than eagerly reserving each row's maximum physical slab.
When an oldest request cannot fit, `admission_overtakes` permits a bounded number
of smaller fitting requests to pass; its default is the execution-row count.
Zero selects FIFO. This bounds successful overtakes, not elapsed waiting time:
a slow active reader can retain its reservation.

Elastic pressure can reclaim idle unpinned weights and eligible idle text
history across models sharing the same device pool. Accepted work, prepared
continuations and explicit checkpoint references remain protected. An evicted
session can replay its full input history; a session name is not durable storage.
Pool pressure does not automatically serialize conversations to disk. If all
capacity is protected and no accepted work can release it, admission reports
overload rather than reclaiming live storage.

## Trim without changing the budget

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/devices/trim \
  -H 'Content-Type: application/json' -d '{"device":0,"target_bytes":0}'
```

This requests reduction of eligible serving-managed commitment toward the
target. It releases idle workspace and unpinned reloadable parameters, not
active work or retained conversations. HTTP 200 returns actual accounting;
protected storage can prevent reaching the target. Fixed weights are not
reclaimable. Trim does not change the configured limit or discard compiled code.

The [observation workflow](observe-model-server.md) explains retained versus
workspace accounting. Their sum is not total VRAM or process RAM, and reported
free bytes are not a guarantee that the next request can be admitted.

## Shut down safely

SIGINT/SIGTERM stops application admission and joins accepted asynchronous work
before destruction. Cancellation does not make an already-submitted device
operation disappear. There is no timeout that authorizes freeing live buffers.

Invalid requests and isolated source failures stay scoped to their requests or
registrations. A native failure affecting shared backing instead stops admission
and drains accepted work before reporting the terminal error. Completion
callbacks publish outcomes; they do not tear down shared devices or pools.

HTTP and source loading assume a trusted deployment. TCP/Unix listeners share
the same authority; socket permissions, tunnels or an external proxy supply
access control. [Command options](../index.md) describe binding
and resource configuration.
