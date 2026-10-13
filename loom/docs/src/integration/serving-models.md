# Author a model package for loom-serve

A serving package combines a model's numerical programs with its input,
parameter and state contracts. The server supplies shared workers, JIT,
device queues, memory ownership, request scheduling and HTTP encoding. Adding
a model that fits an existing modality changes the package, not the server's C
code or executable.

The [first-output walkthrough](../workflows/serve-models.md) runs a complete
arithmetic package before weights or neural accuracy enter the picture. That
is a useful starting point for a port: changing its math preserves a working
path through registration, execution, output and retirement.

## Select the execution contract

The important distinction is the shape of state and output, not the model name:

| Contract | Source responsibilities | Starting package |
| --- | --- | --- |
| Retained text | Token framing, paged state geometry, mixed prompt/decode math, progress, sampling and chat policy. | [Tiny text model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/text) |
| Staged image | Prompt preparation, parameter domains, numerical stages, invocation continuation and final RGB. | [Image model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/image) |
| Finite embeddings | Input preparation, cohort cost/compatibility, pooling/projection and final vectors. | [Embedding model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/embedding) |
| Finite decisions | Structured question framing and compact normalized answer distributions. | [Decision model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/decision) |

These packages use actual source VM programs, native kernels and production
HTTP services, not canned native replies. They establish the serving contract,
not learned quality or competitive neural performance. Each learned port needs
its own numerical oracle and representative workload.

An audio stream, image-conditioned request or arbitrary workflow graph needs an
input/output ownership contract beyond these HTTP profiles. Such a port is a
modality extension, not an embedding vector with overloaded fields. The
[HTTP reference](../reference/serving-http.md) defines the existing boundary.

## Keep the package complete and relocatable

A typical package contains:

```text
model.json           deployment, source roles, assets and capacities
prepare.loom         cold declarations and parameter interpretation
control.loom         warm request and execution policy
config.loom          model/target specialization inputs
kernel.loom          numerical providers and reusable helpers
tokenizer.json       tokenizer asset, when needed
weights.safetensors  checkpoint selected by preparation policy
```

These filenames are conventions, not hardcoded server paths.
The [model document](../reference/serving-models.md) names the
actual files. GGUF and safetensors inputs follow the selected preparation
policy; a file extension or model name does not convert an arbitrary checkpoint.
Its keys, encodings and layouts must agree with the authored source.

All program roles and numerical providers accept `.loom` or `.loombc`.
Bytecode is portable IR and still JITs for the selected device. References in a
file document resolve from its directory, including references to shared motif
libraries elsewhere. An HTTP document supplies its absolute server-local base.

`definition.sources` is a numerical provider catalog, not an ordered dispatch
list. Roots and imported symbols determine dependency closure through the
normal Loom linker. Reusable helpers take SSA values or template arguments;
model-specific configuration belongs at program/kernel boundaries. This keeps
a motif usable without importing another model's global configuration namespace.

Preparation and control VM files compile separately and link with native
capability modules. Their imports are not resolved from the numerical provider
catalog. The cold program names numerical roots; the warm program invokes the
resulting prepared stages. These are explicit boundaries, not one combined
host/device link graph.

## Separate preparation from execution

The cold `prepare` VM program receives modality inputs and the source-owned
`options` object. It validates architecture constraints and declares stages,
configuration and checkpoint domains. Shared preparation capabilities include:

| Import in module `prepare` | Meaning |
| --- | --- |
| `stage(root, tag)` | Declare a command-program root and a source-defined selection tag. |
| `dispatch(root, tag)` | Declare a buffer-only native kernel root with its compiled launch configuration. |
| `config(stage, key, value)` | Bind a public Loom configuration value in textual syntax. |
| `config_i64(stage, key, value)` | Bind a signed integer configuration value. |
| `parameter(stage, binding, directory, path, policy)` | Associate a reflected fixed binding with a checkpoint and preparation policy. |

The exact signatures and ownership are in the
[preparation contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/runtime/program/preparation.h).
Declarations copy their strings; opaque returned state must own its storage or
retain its inputs. Preparation is not a place to submit device work.

The server resolves these declarations, compiles through its shared compiler
task queue, and retains the selected stages for the model. Configuration can
specialize stage shapes or layouts without giving every session its own program.
Compatible requests share the prepared code and parameters. Warm work does not
invoke a compiler subprocess or redo unchanged model preparation.

The warm `control` VM imports the runner-local `runner.execute_N` and
`runner.feedback` capabilities. `execute_N` submits a prepared stage with its
buffers; `feedback` transfers a bounded final result into a registered host slot.
Both enqueue work without waiting. The native model owner schedules completion
on work/feedback timelines and preserves buffers until their actual last use.
The [runner-module contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/runtime/program/module.h)
defines arity, buffer types and lifetime. This is a coarse serving capability,
not a source program's ownership of the entire device or process scheduler.

## Trace one finite request end to end

The embedding package is the smallest complete example. Its cold entry is
`prepare(asset_root, options)`. It returns a batch-unit limit, upload/output/
feedback capacities, an optional tokenizer asset, a source name and opaque state.
Its control program exports:

```text
prepare_request(state, input) -> (prepared_bytes, units, compatibility_tag)
prepare_batch(state, stage_tags, ranges, prepared_bytes)
    -> (stage_index, upload_payload, result_plan)
run(stage_index, feedback_bytes, input_buffer, output_buffer)
```

The HTTP parser provides indexed protocol records, not a source-private JSON
guess. `prepare_request` performs model framing and validation. `units` is a
positive additive bound on real work/storage; it need not mean tokens. Equal
compatibility tags mean requests can share a cohort. Source state is immutable
across request preparation, and returned buffers retain any aliased input.

The native scheduler starts with the oldest prepared request and fills the
cohort from compatible waiting requests. It does not wait for hypothetical
arrivals. Batch preparation receives stable request-order ranges over concatenated
prepared bytes, selects a declared stage and maps physical output back to requests.

The result plan describes physical output bytes separately from compact feedback
bytes, then each request's rows and each row's offset/width. Device padding and
activation storage need not be downloaded. Projection and normalization happen
in the numerical program; the host encodes final F32 vectors. Decision encoding
similarly consumes final probabilities, not hidden activations.

The example deliberately reverses physical output rows. Correct client indices
therefore prove that result mapping works rather than merely agreeing with a
convenient device layout. Its independent arithmetic oracle also makes ragged
packing and cohort association observable. See the
[finite-source contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/finite/README.md)
for exact record layouts and ownership.

## Retained text and staged images

Text uses one prepared model and cached stage set. Sessions are rows and storage
references, not independent VM processes. Its source declares page/recurrent
geometry and consumes packed spans that can mix prompt tokens, decode anchors
and speculative verifier work. Token capacity and independently advancing span
capacity are separate. The source reports completed progress; native ownership
commits it after readiness.

Shared prefixes and explicit checkpoints keep immutable endpoint references.
A branch needs a private recurrent writer without copying the entire KV prefix.
The tiny text package exercises a partial-page frontier, an unconsumed pending
token, fork/rewind, cancellation and full-replay equivalence. Those boundaries
matter more than getting one unretained prompt to emit plausible text. The
[text service](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/text/README.md)
links the storage, epoch and chat-policy declarations beside their owners.

Image preparation selects stages and parameter domains for configured geometry.
Warm source prepares opaque request bytes from prompt, seed and adapter strength,
then drives command parts to final RGB. Image control returns a continuation
cursor; another invocation occurs only after the previous accepted frontiers
retire. Compiler-reflected workspace remains live across barrier-aligned parts.
Final output is F32 CHW RGB in `[-1,1]` before native image encoding. The
[image source contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/image/model.h)
defines entry signatures, alignment and cancellation.

## Make parameters and temporaries visible

Command stages declare fixed checkpoint domains and preparation policies.
Stages sharing a weight bank must agree on reflected placement. Preparation
indexes checkpoint metadata; activation streams and transforms weight payloads
into final storage. Warm requests reuse those plans. Deactivation can release
reloadable backing without discarding compiled code or retained session state.

The embedding package's
[weighted variant](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/fixture/embedding/variants/weights.loom)
and [policy](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/fixture/embedding/variants/policy.loom)
demonstrate adding a checkpoint-backed head without native model code.
Comparing the checkpoint's actual shapes/types with the numerical oracle is
part of the port; current key/extent checks do not establish model identity.

Private command workspace comes from compiler reflection and queue-ordered
allocation. It is not a permanently resident per-session tensor arena.
Escaping output remains live through its final device or transfer consumer;
copied host feedback survives CPU encoding. A slow network reader receives
copied response storage rather than retaining device workspace.

Finite direct-dispatch stages instead take input/output buffers and own their
device-local working storage. The resident XDNA example uses this path. These
direct stages have no checkpoint-domain binding in the current source contract;
weighted direct programs need an explicit source-owned parameter-placement
contract, not a guess from native argument reflection. The ordinary weighted
command example and unweighted resident example prove different boundaries.

## Establish correctness before throughput

Start with the selected package's deterministic output and replace one numerical
stage with the intended model operation. Checks can compare functions directly
with independent VM oracles; kernel checks specify an appropriate oracle rather
than copying candidate output into a golden. [Correctness workflows](../workflows/test-correctness.md)
cover functions, cases and scenarios.

A complete port then establishes final outputs through the same server binary
using an external package directory. The useful cases include ragged inputs,
different request sizes, cancellation during accepted device work, a failed
request followed by successful reuse, and unregister/re-register. Retained text
also compares append/fork/rewind against full replay. A compilation-only success
does not establish these ownership or numerical contracts.

[Agent-driven kernel development](../workflows/agent-driven-kernel-development.md)
and [compile reports](../workflows/compile-reports.md) provide the optimization
loop once correctness is measurable. Real model shapes, memory traffic and
independent error bounds guide layout, traversal, fusion and specialization.
Microkernel speed is one input; [server observations](../workflows/observe-model-server.md)
and end-to-end useful throughput determine whether the change helps the workload.
