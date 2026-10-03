# Author a model as a program

Loom model code describes math, storage, specialization, and launch structure.
The embedding supplies a live device, model configuration, parameter storage,
and mutable state. This runner uses text sources directly; source changes are
picked up by the next process without rebuilding its executable.

## Start from a concrete numerical and storage contract

For a new model, the input evidence is its pinned configuration, checkpoint,
tokenizer/chat template, and reference implementation. The contract includes
layer order, projection orientation, normalization epsilon, positional encoding,
attention masking, tied weights, special-token IDs, and every persistent state
update. A plausible sentence is weaker evidence than correct intermediate
values and a matching retained continuation.

The checked-in Qwen example has 64 target layers in sixteen groups: three Gated
DeltaNet layers then one full-attention layer. That structure is explicit in
[`model_prefill.loom`](../models/qwen38/programs/qwen38/model_prefill.loom),
not inferred by the runtime. Its quantized tensor layouts and fixed dimensions
are model-specific. Reusing a contraction motif does not imply reusing its
Qwen parameter offsets or attention convention.

The first useful port milestone is one real-weight block through the actual
JIT/command/runtime path, compared against an independent reference, followed
by an end-to-end model continuation. This crosses weight interpretation,
specialization, device execution, and ownership before HTTP or scheduling work
expands around the new model.

## Four source layers

| Layer | Meaning | Existing example |
| --- | --- | --- |
| Ordinary functions and templates | Reusable arithmetic, layouts, target-selected implementation motifs | [Authoring corpus](../../../loom/src/loom/test/corpus/authoring/README.md) |
| Kernels | Workload-to-launch mapping and one device invocation | [Token embedding](../models/qwen38/kernels/qwen38/token_embedding.loom) |
| Command programs | Parameter roots, transient lifetimes, layer composition, dispatch dependencies | [Packed epoch](../models/qwen38/epoch.loom) |
| VM control | Coarse model orchestration through typed runner imports | [Qwen control](../models/qwen38/control.loom) |

A kernel's workload arguments establish launch geometry and specialization;
its launch arguments supply the values and buffers consumed by that invocation.
Command specialization arguments fix one composition, while command bindings
identify persistent and transient storage at execution. The canonical guides
cover [kernels](../../../loom/docs/src/guide/kernels-and-launch.md),
[command programs](../../../loom/docs/src/guide/command-programs.md), and
[storage views](../../../loom/docs/src/guide/buffers-views-memory.md).

`command.parameter` names a tensor and its required view. The compiler lays out
parameter roots and publishes their offsets, sizes, and alignment. Semantic
`buffer.alloca<global>` inside a command program becomes planned transient
storage; the runner allocates its backing during setup. It is not a per-token
HAL queue allocation. Workgroup allocations inside kernels have a different
lifetime and represent local shared storage.

## Configuration and live target facts

[`qwen_compile_stage`](../qwen_model.c) supplies a `loomc_config_options_t`
containing model dimensions and shape capacities. The same configuration is
applied while materializing the command and its native source requests. The
compiler retains the relationship between a kernel's launch math and body.

Values fixed for a specialization belong in `config.decl`/`config.get` or
specialization arguments. Current active lengths, token IDs, row origins, and
positions belong in workload arguments or device descriptors. Capacity and
active count are different facts: padding can exist without advancing the
persistent state of inactive rows. The [facts guide](../../../loom/docs/src/guide/facts-and-specialization.md)
explains both domains and path-dependent refinement.

The fused [FFN block read-ahead](../models/qwen38/kernels/qwen38/ffn_gate_up_prefetch.loom)
is a concrete example: `qwen38.ffn.input_size` and `qwen38.ffn.output_size`
configure the body, allowing an exact block loop to pipeline weight acquisition.
Its workload carries token capacity, and its five launch buffers carry live count,
activations, two weight views and output. There is no runtime K/N scalar to
rediscover or a host readback to learn the active count. The
[model guide](../models/qwen38/README.md#fused-feed-forward-projection) includes
the matching configuration and seeded full-array checks.

An authored target contract fixes algorithmic requirements such as subgroup
width. `loomc_target_profile_create_amdgpu_iree_hal` supplies the actual device
facts for native specialization. The Q4/Q5/Q8 wave32 entries explicitly name
that width because it also changes their command launch counts. Replacing those
queries with guessed host constants would split one contract into two.

For large slabs, logical dimensions can remain bounded indices while byte
origins use `offset`. A row starting beyond 4 GiB must retain that full origin
through all subviews and address arithmetic. The packed attention sources show
that distinction; changing an index width is a measured compiler/kernel
experiment, not a blanket optimization rule.

## Source catalog to executable commands

[`sources.txt`](../models/qwen38/sources.txt) contains relative source paths,
one per line. The runner indexes providers once and requests a named command
root. The minimal corresponding catalog and programs are in
[`testdata/jit`](../testdata/jit/sources.txt).

The real embedding sequence in [`jit.c`](../jit.c) is:

1. Create the target environment, context, prepared compiler and pipeline;
   obtain the live HAL profile and freeze the source index. A standard
   `loomc_task_pool_t` supplies up to four physical-core workers by default,
   each with a reusable `loomc_workspace_t`. Product construction uses separate
   caller scratch.
2. Call `loomc_cmd_program_product_build` with the root and configuration. Its
   request sink takes ownership of reachable native source requests.
3. Submit each request through a `loomc_task_queue_t` on that shared pool.
   Its binding/root ordinals and product requirements supply the native
   specialization records. Each task deserializes, compiles, and emits from the
   same private module, preserving prepared compiler facts. Compiler, pipeline,
   profile and product are immutable and shared; mutable scratch is selected by
   the task's mutually exclusive worker ordinal.
4. Each task loads its executable through the selected queue family and resolves
   its disjoint requirement slots. Drain the stage's queue before binding those
   entries in command-local order or releasing partial results after a failure.
   The worker population and scratch survive for subsequent stages.
5. Allocate/load fixed parameter roots. `loom_serve_jit_stage_record` creates
   reusable commands retaining those buffers and executables.
6. JIT the VM source, transfer its image to the trusted in-process VM loader,
   and link its imports against the runner's prepared stages.

This is cold-path setup. Already recorded commands do not depend on the source
index or compiler workspace remaining alive. A compile failure returns source
diagnostics and releases partial state; it never substitutes a stale artifact.
The [task-pool embedding example](../../../loom/binding/c/example/jit_task_pool.c)
shows the public ownership protocol without a model dependency. In this runner,
stage calls remain synchronous; their native kernel requests compile in parallel.
No task queue or compiler work is introduced into the serving epoch path.

The Qwen adapter validates shared parameter placement across its independently
compiled roots before assigning one weight slab. Auxiliary MTP roots either
view existing tensors or own additional tensors. IREE's parameter index/provider
loads bytes into those destinations. Another checkpoint format can reuse the
IO machinery, but its model adapter must establish names, encoding, orientation,
and size rather than treating a matching byte count as numerical equivalence.

Checkpoint encoding and inference layout need not be identical.
[`qwen_weights.c`](../qwen_weights.c) loads each unique tensor into final
residency and schedules the ordinary source-JIT
[`prepare.loom`](../models/qwen38/prepare.loom) command after its read. Each
workgroup captures eight complete Q5 rows before rewriting its disjoint range;
the permutation requires only workgroup-local storage. The target and MTP
consumers then use that same prepared encoding. An alternative model owns its
own format and preparation contract, not Qwen's tensor-name predicate or
dimensions. Layout qualification covers every consuming shape and the complete
startup ownership path, including byte copies and peak residency. A faster
projection alone does not establish a better serving configuration.

## Correctness that survives optimization

[`gdn_convolution.loom`](../models/qwen38/tests/gdn_convolution.loom) is a small
`check.scenario`/`check.trial` example: an ordinary function runs on the GPU and
an independent serial function runs through the VM oracle. No test kernel
wrapper is needed. It checks the activation with an explicit numerical
tolerance and input/history preservation bitwise. Kernel-level checks can name
an explicit oracle function with the intended serial semantics. The
[checks guide](../../../loom/docs/src/guide/checks-and-benchmarks.md) owns the
syntax and observation contract.

`iree-test-loom --library` merges the libraries' embedded check records too.
An explicit `--case=@name` selects the intended case and all its samples;
an empty selection runs every loaded record. The FFN recipe iterates its
complete owned case list so provider benchmarks do not silently broaden the
suite or require additional fixture-only dependencies.

Useful cases exercise boundaries that realistic batching creates: one active
token, full tiles, odd tails, inactive rows, distinct resident origins, repeated
continuation, and reset/reuse. Shared-state comparisons cover untouched regions
as well as the active outputs. A scratch staging optimization can be numerically
correct in isolation yet corrupt another session or a later turn.

Kernel differentials establish an implementation relationship, not necessarily
model accuracy. Independent reference outputs establish checkpoint/math
interpretation. [`qwen_epoch_check.c`](../qwen_epoch_check.c) then checks full
model retained histories, isolated versus packed execution, shape transitions,
and MTP publication. The HTTP witness adds canonical-history reuse and visible
multi-turn responses. These checks protect different boundaries.

## Shape and layout evolution

JIT makes specialization cheap to request; it does not remove authored storage
bounds. The current adapter generates token and span classes independently at
startup, shares a maximum workspace, and resolves immutable native exports once.
Its sixteen-row bound comes from four-input verification fitting a 64-entry
selected-output table, not from assigning sixteen full contexts. The private
KV pool separates shared physical capacity from per-row logical context.
Expanding the 512-token/sixteen-row envelope requires updating host arrays,
source/view bounds, descriptor producers, workspace sizing, and numerical tail
coverage together. Adding runtime variants additionally requires publishing new callable
stages without invalidating in-flight command or VM references.

The next model does not need a general serving framework first. A small
model-specific adapter with explicit storage contracts can reuse `jit`,
`command`, `execution`, and the native module. Once two real adapters share a
semantic boundary, that evidence can justify extracting a common model API.
