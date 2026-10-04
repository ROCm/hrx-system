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
| VM control and policy | Coarse orchestration through typed imports; source queries returning ordinary values and buffers | [Qwen control](../models/qwen38/control.loom), [weight policy](../models/qwen38/weights.loom) |

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

Transient reservation begins at its source allocation, not its first memory
access. Placing an allocation immediately before its first producing command
lets the planner reuse storage retired by earlier commands. Krea's
[VAE attention](../models/krea2/vae_attention.loom) allocates its attended
features after QKV projection; the preceding normalization buffer is then
available for reuse. Its full-shape composition checks both output bits and
the four-feature-plane scratch bound.

Schedule regions express dependencies, not hints. A command-program call
preserves its implicitly serial body when expanded inside `command.concurrent`.
A template expands into its caller's region instead; a multi-command template
branch needs an explicit `command.serial` scope when its operations depend on
one another. Krea's [denoising input phase](../models/krea2/denoise.loom) uses
that nesting to overlap text copying with image projection while keeping the
LoRA B/add consumer after its base and A producers. Buffer aliasing does not
create that edge. Exact composition against separately executed native stages
checks the dependency contract independently of floating-point oracle error.

## Configuration and live target facts

The model's [`config.loom`](../models/qwen38/config.loom) supplies fixed typed
`config.def` values through the ordinary source catalog.
[`qwen_compile_stage`](../qwen_model.c) supplies run-dependent overrides in
`loomc_config_options_t`. Both feed the same command materialization and native
source requests; the compiler retains the relationship between a kernel's
launch math and body. Kernel declarations still constrain accepted values with
their range and multiple predicates. The small JIT caller exercises a source
default, a different explicit override, and an invalid override followed by
successful compiler reuse.

Ten shape-dependent or derived bindings remain in the adapter, including Q8
output bounds and quantizer group capacity. Stage selection and state geometry
also remain native model policy; fixed source defaults alone are not a complete
model-owned bootstrap.

Configuration is a model entry/specialization boundary. Reusable functions,
templates, and motifs receive dimensions and layout facts as explicit SSA
operands, rather than reading ambient model-named configuration symbols.
A concrete wrapper can resolve configuration and apply the same template with
different constant operands; the reusable body retains compile-time facts
without depending on that model's keys. Launch arguments that remain runtime
values are a different contract from template operands fixed by that wrapper.

Current active lengths, token IDs, row origins, and positions belong in workload
arguments or device descriptors. Capacity and active count are different facts:
padding can exist without advancing the persistent state of inactive rows.
The [facts guide](../../../loom/docs/src/guide/facts-and-specialization.md)
explains both domains and path-dependent refinement.

The fused [FFN block read-ahead](../models/qwen38/kernels/qwen38/ffn_gate_up_prefetch.loom)
resolves `qwen38.ffn.input_size` and `qwen38.ffn.output_size` in its concrete
kernel wrappers and passes K/N into the body template. The exact block loop
still pipelines weight acquisition; the template itself has no config reads.
The [independent-specialization case](../models/qwen38/tests/ffn_prefetch_specialization.loom)
applies that same motif at K256/N64 and K768/N128 in one module without Qwen
config bindings, comparing both against independently staged contractions.
Its workload carries token capacity, and its five launch buffers carry live count,
activations, two weight views and output. There is no runtime K/N scalar to
rediscover or a host readback to learn the active count. The
[model guide](../models/qwen38/README.md#fused-feed-forward-projection) includes
the matching configuration and seeded full-array checks.

The [cache-map helpers](../models/qwen38/kernels/qwen38/spans.loom) and shared
KV preparation/WMMA templates likewise receive logical capacity explicitly.
That capacity sizes each row's logical page map and bounds absolute positions;
physical pool capacity independently sizes the shared K/V planes. The concrete
dense and packed wrappers resolve configuration without adding launch arguments
or device metadata loads. The
[attention specialization case](../models/qwen38/tests/attention_specialization.loom)
combines 81/145-position logical caches with 128/192-position physical pools
in one composition without any model configuration bindings. It checks different
map strides, scattered pages, complete cache contents, and masked output tails.
These motifs still implement Qwen's fixed head geometry and RoPE; explicit
cache operands alone do not make them arbitrary-model attention.

The [packed Q5 contraction bodies](../models/qwen38/kernels/ggml/linear_q5k_q8_1_x4.loom)
take token/output capacity independently of live token count and K/N. Their
single-row and four-row schedules accept the same explicit bounds and weight
ordering; canonical and channel-interleaved wrappers resolve the configuration.
The [reuse case](../models/qwen38/tests/q5_specialization.loom) instantiates both
schedules at two capacity pairs in one module without config bindings, including
an odd channel tail and an untouched output row. Capacity constrains the body;
it does not become the amount of work executed.

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
   `loomc_task_pool_t` supplies physical-core workers, each with a reusable
   `loomc_workspace_t`. The runner requests up to eight with
   `loomc_task_pool_options_t.max_worker_count`; processor affinity and topology
   may provide fewer. Product construction uses separate caller scratch.
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
[`weights.c`](../weights.c) loads each unique tensor into final
residency. It calls the model's source-JIT weight policy with the tensor name as
a read-only, non-NUL-terminated VM buffer. `prepare_weight` returns a command
root buffer and its required tensor byte length; an empty root and zero length
request unchanged file bytes. The loader validates the result against actual
reflection and caches each distinct command. A model needing no transformation
can return the same empty root and zero for every key. Preparers expose one
mutable tensor binding, no fixed buffers, and no global scratch; their source
wrappers fix any configuration needed for compilation.

A model whose kernels consume the checkpoint representation directly can start
with this complete `weights.loom`; it needs no preparation kernels or C policy:

```text
global.rodata.def @unchanged = bytes("")

func.def public @prepare_weight(%key: buffer) -> (buffer, i64) {
  %root = global.load @unchanged : buffer
  %bytes = scalar.constant 0 : i64
  func.return %root, %bytes : buffer, i64
}
```

Qwen's [`weights.loom`](../models/qwen38/weights.loom) owns the tensor-name
predicate and Q5 dimensions and selects the ordinary source-JIT
[`prepare.loom`](../models/qwen38/prepare.loom) command after its read. Each
workgroup captures eight complete Q5 rows before rewriting its disjoint range;
the permutation requires only workgroup-local storage. The target and MTP
consumers then use that same prepared encoding. An alternative model owns its
own format and preparation contract, not Qwen's tensor-name predicate or
dimensions. Layout qualification covers every consuming shape and the complete
startup ownership path, including byte copies and peak residency. A faster
projection alone does not establish a better serving configuration.

This is a cold policy query, not a graph language interpreted by C. The same
[`program`](../program.h) owner handles pure startup queries and the inference
VM linked against native submission exports. The startup query process is gone
before inference begins. Common native pipelines can consume model-owned query
results while additional control moves into VM source; modality-specific state
and scheduling contracts still need an actual caller before being generalized.

The inference program exposes both `step` and `epoch` from one source-JIT image.
[`control.loom`](../models/qwen38/control.loom) is a concrete example of typed
command selection, optional proposal, verification, and cache catch-up. Its
`runner.execute_N` calls select cached commands by residency-local index;
`runner.feedback` forks a download into a cold-registered host slot. The source
contains the command order and buffer routing, while native execution owns
timeline assignment and lifetime. A later source/native failure still requires
joining accepted work before the host recycles payloads. The
[runner guide](RUNNER.md#queues-failure-and-reclaim) describes that boundary.

## Image and audio entry points

The reusable embedding accepts command programs and buffer bindings, not text
tokens. [`jit_test.cc`](../jit_test.cc) exercises it without a tokenizer, chat
request, KV cache, or Qwen adapter. That is the smaller starting point for an
image or audio port. The [Krea component port](../models/krea2/README.md) adds
real checkpoint and adapter evidence: one source-JIT command now composes
batched time conditioning, image projection, all 28 transformer layers, the
velocity head, and eight Euler updates. Base, zero-strength, and active LoRA
trajectories match their independently staged native components bit-for-bit;
parameter roots stay immutable and one planned workspace serves every step.
Its reproduction guide separates primitive numerical checks, exact composition,
and accumulated image differences. A separate source command now runs the full
still-image VAE from packed BF16 latent to F32 RGB, with folded spatial
upsampling and in-place residuals. Its final pixels agree with the independent
CPU decoder to at most one 8-bit level on the qualified base/LoRA images.
Text conditioning remains external. The existing HTTP
service and packed scheduler remain concrete Qwen consumers; changing their
model directory does not turn them into an image or audio endpoint. No complete
image or audio model is qualified by this packet.

A first tensor-in/tensor-out adapter has this ownership flow:

1. It creates a [`loom_serve_device_t`](../device.h), which owns the device,
   asynchronous services, exact queues, and execution timelines. Borrowed
   device/queue handles create a source catalog with `loom_serve_jit_create`
   and specialize named roots with `loom_serve_jit_compile`. Source owns tensor
   names, shapes, arithmetic, and fixed configuration; the caller supplies
   run-dependent specialization.
2. It passes the stages' reflected parameter roots to
   `loom_serve_weights_load`, with its own source weight policy. The adapter
   establishes identical parameter placement for the shared-root prefix;
   equal allocation sizes alone do not establish that contract. An unchanged
   checkpoint layout uses the empty preparation root shown above. Each load
   selects explicit roots from one checkpoint domain; base and adapter files
   load their own roots before the composed command is recorded. Equal tensor
   keys across different domains never imply shared residency.
3. It allocates model input, output and persistent-state buffers. Reflected
   transient requirements supply workspace size and alignment. It records the
   stages with `loom_serve_jit_stage_record`; recorded commands retain their
   executables and fixed buffers after compiler storage is destroyed.
4. Using the device owner's execution context, it registers the immutable stage
   table and any host feedback spans with `loom_serve_module_create`, then links its
   source VM control through `loom_serve_program_create`. The HAL type provider
   and borrowed feedback storage outlive all accepted work. Requests carry
   bindings into this shared process, not their own VM instances.
5. It uploads request data through `loom_serve_execution_transfer`, invokes its
   VM entry using the retained invocation, and joins the relevant completion
   frontiers before publishing output. `runner.execute_N` returns an accepted
   submission value, not a completed tensor. An error after an earlier accepted
   submission still requires draining both execution timelines before reuse or
   teardown. [`control_test.cc`](../control_test.cc) exercises that failure path.
   The device owner is destroyed only after model commands, buffers, VM state,
   and accepted host payloads have retired; destroying it is not an implicit
   completion wait.

The model's storage contract determines the pipeline shape. Krea's finite
denoising trajectory retains its latent and conditioning buffers inside one
source command with a specialized `scf.for` loop and a supplied device schedule.
No host loop, per-step allocation, or device-to-host step-counter query is
needed. A model with a different request/feedback boundary can instead retain
state across coarse command invocations. An audio encoder could have one finite
input/output transform; streaming audio additionally needs an explicit carried
state, sample position, and publication rule for each chunk. Those audio shapes
are candidates, not implemented adapters. Their first witness includes real
checkpoint bytes and independently checked output through the ownership flow
above.

Iteration counts and command selection can live in ordinary source VM control.
The current native imports expose coarse submission and feedback, not arbitrary
HAL allocation, scalar kernel-argument mutation, or a full device API. A model
that needs changing per-iteration values can carry them in its bound device
state and consume them in kernels. That keeps the existing submission contract
intact; an additional native capability needs a concrete caller and lifetime
contract rather than a model-specific command-program ABI change.

Image encoding, waveform framing, tokenization and request transport remain
frontend concerns for the initial port. The first visible result is a checked
image/tensor or audio chunk from the actual model, before generalizing network
protocols or batching policy. Autoregressive audio may eventually share token
scheduling mechanisms; that requires evidence about its state and output
contracts, not merely that it generates discrete values.

## Correctness that survives optimization

[`gdn_convolution.loom`](../models/qwen38/tests/gdn_convolution.loom) is a small
`check.scenario`/`check.trial` example: an ordinary function runs on the GPU and
an independent serial function runs through the VM oracle. No test kernel
wrapper is needed. It checks the activation with an explicit numerical
tolerance and input/history preservation bitwise. Kernel-level checks can name
an explicit oracle function with the intended serial semantics. The
[checks guide](../../../loom/docs/src/guide/checks-and-benchmarks.md) owns the
syntax and observation contract.

For real checkpoint tensors, [`component_check`](../component_check.c) loads
named source commands and records two complete executions. `--output_type`
selects raw little-endian `f16`, `bf16`, `f32`, or `f64` observation without
converting device output; its default and default tolerances describe BF16.
An F32 component supplies its own `--atol` and `--rtol`. JSON distinguishes raw
bit differences, element-envelope violations, and relative L2; nonfinite pairs
fail even in report-only mode. The small queued-copy contract test exercises
all four representations, signed zero, fatal nonfinite values, invalid byte
lengths, and finite F64 values whose squares overflow or underflow:

```sh
iree-bazel-build --config=asan //experimental/loom_serve:component_check
python -B experimental/loom_serve/component_check_test.py \
  --checker bazel-bin/experimental/loom_serve/component_check
```

The command runs on the selected AMDGPU runner. Model-specific drivers supply
their independent numerical oracles; interpreting an F32 tensor as BF16 to
fit a qualification tool would invalidate that comparison.

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
startup, shares a maximum workspace, and publishes an immutable command table.
The shared VM selects entries by index; model entry points resolve once.
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
