# Capture PyTorch values for Loom replay

Capture the inputs and outputs of a selected model boundary, then use those
files in an authored Loom correctness case or scenario. The reference supplies
real values; Loom source supplies the operation, numerical acceptance, mutable
state, and benchmark workload.

**Example files:** [`capture.py` and `replay.loom`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/workflows/oracles/pytorch).
The example runs a real PyTorch module, saves its observations, compares a Loom
function with the saved output, and reads Loom's own captures back in a second
invocation. Its eight-element inputs demonstrate capture and replay mechanics;
a production performance workload needs representative dimensions and data.

## Pin the reference and select the boundary

A useful capture belongs to one reproducible model invocation. Record the model
repository and revision, weight identities, tokenizer, input bytes or token IDs,
model configuration, PyTorch and framework versions, device, evaluation mode,
autocast policy, and numerical settings such as TF32. A random seed complements
captured inputs; it does not identify all of that state.

`model.eval()` selects evaluation behavior. `torch.inference_mode()` controls
autograd bookkeeping. Neither replaces the other. Choose the numerical oracle
and acceptance independently of the fastest implementation's rounding choices;
a higher-precision reference and an optimized production reference can serve
different purposes.

The boundary may be a projection, an attention block, or a complete layer.
Capture its needed inputs, including routing IDs, masks, cache contents, and
positions. Immutable weights can remain in their identified checkpoint rather
than being duplicated for every call. A module hook identifies an eager module
call; it does not identify the backend kernels or establish their timing.

## Save each observation before its storage is reused

The example's writer publishes one NPY file at a time and records the original
dtype, shape, strides, storage offset, device, and file hash beside it:

```python
--8<-- "examples/workflows/oracles/pytorch/capture.py:save_tensor"
```

The NPY file contains dense logical values in row-major order. The sidecar
describes the source view; it does not recreate shared storage or an alias
graph. An authored replay that depends on aliasing represents that relationship
explicitly, for example with `check.tensor.view`.

Memory scales with active observations and normal framework workspace, rather
than a dictionary of all preceding layer outputs. The serial example writes
one observation at a time; concurrent calls can each have a copy in flight.
A single large tensor can still require a large host copy. Select a useful
boundary or slice when the whole value is unnecessary. File errors propagate,
and existing files are not silently overwritten.

PyTorch's [pre-hook and forward-hook APIs](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)
place input capture before `forward` and output capture after it. The example
saves the input in the pre-hook so an in-place operation cannot overwrite the
reference input first. Each invocation gets a separate `call-N` directory,
including repeated calls to a shared module. Hooks return no replacement values
and are removed when the capture scope exits:

```python
--8<-- "examples/workflows/oracles/pytorch/capture.py:hooks"
```

Calls complete within the scope, including any worker threads started by the
caller. An invocation interrupted by a model or capture error leaves partial
evidence; the scope rejects it even if the caller caught that error. The
example publishes `run.json` only after the complete run succeeds.

For a model block with keyword inputs and tuple outputs, selection stays
explicit at the semantic boundary:

```python
with capture_module(
    block,
    "run-001/layer-7",
    select_input=lambda args, kwargs: kwargs["hidden_states"],
    select_output=lambda result: result[0],
):
    with torch.inference_mode():
        model(**inputs)
```

Additional input state needs its own explicit capture. A single selected tensor
is sufficient for the ReLU example; it is not sufficient for attention or a
stateful decoder. A capture from `torch.compile`, a fused operation, or a device
graph also needs a boundary whose execution semantics have been established.
An eager hook alone does not prove what a compiled runtime executed.

Saving a detached tensor for later is insufficient: a NumPy view may still
[share its storage](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.numpy.html)
with the original. The example finishes each write before returning from the
hook. Device transfers and file writes perturb execution, so collect performance
from a separate uninstrumented run with the same selected implementation.

## Run the capture and replay example

Place the example files together in your reference environment with PyTorch and
NumPy installed. No checkpoint download is needed for this example:

```shell
python capture.py captures
loom-test replay.loom --case=@relu_capture --device=amdgpu --file-output-root=outputs
loom-test replay.loom --case=@replay_outputs
loom-test replay.loom --case=@bf16_carrier
```

Use `--device=task` for the CPU execution provider or `--device=vulkan` for a
compatible Vulkan device. PyTorch capture is on the CPU in this example;
changing the Loom provider does not change the saved reference values.

The authored scenario loads the captured input and PyTorch golden. It checks
the target's complete output against both that golden and Loom's independent
oracle execution:

```loom
--8<-- "examples/workflows/oracles/pytorch/replay.loom:scenario"
```

The exact comparison is appropriate for these finite, exactly representable
ReLU values. A reduction or approximation normally uses an explicit
`check.expect.approximate` contract derived from the workload. Changing the
capture does not authorize relaxing the acceptance threshold.

The scenario's write saves the completed target and Loom oracle observations
under `outputs/scenario-relu_capture/configuration-0/trial-0-0/actual/` and
`expected/`. The original PyTorch golden remains in `captures/`. A failed
comparison can use the default `mode(on_failure)` to publish only its evidence.
The [checks guide](../../guide/checks-and-benchmarks.md#a-scenario-compares-independent-executions)
defines path identity, output policies, and memory ownership.

The same authored scenario can supply a benchmark. This command is a single
iteration smoke that checks the workflow, not a performance measurement:

```shell
loom-benchmark replay.loom --benchmark=@captured_relu --device=amdgpu \
  --scenario-correctness=true --iterations=1 --warmup-iterations=0 \
  --min-time-ms=0 --warmup-time-ms=0 --max-batches=1 \
  --file-output-root=outputs --output=measurements.jsonl
```

Capture belongs to the separate correctness phase. Timed repetitions do not
write the observations. For a production benchmark, preserve the full semantic
cut, representative dimensions, parameter residency, input-ring policy, and
completion boundary from the [benchmark workflow](../benchmark.md).

## Preserve BF16 and other physical carriers explicitly

Standard NPY has no BF16 or float8 descriptor. The example saves BF16 bits in
an `i16` array and float8 bits in an `i8` array, with the semantic dtype in the
sidecar. These are bit views, not integer conversions. Native NPY types keep
their own descriptors.

Loom reads the physical carrier with its matching type, then reinterprets it
explicitly. The example verifies BF16 values against the corresponding F32
reference without changing the captured bits:

```loom
--8<-- "examples/workflows/oracles/pytorch/replay.loom:bf16"
```

Packed and quantized formats require their actual format contract: block bytes,
scales, zero points, logical dimensions, and encoding identity. The example
rejects quantized and sparse tensor objects rather than treating a convenience
dequantization as the original storage. A port chooses which physical operands
to capture and how its Loom motif interprets them.

## Carry the evidence into the model

One captured input proves one execution. A maintained port adds representative
shapes, tails, routing distributions, and state transitions, then reaches a
complete layer or model output. Preserve the capture's provenance beside the
authored cases so another agent can regenerate it from the reference.

Compiler reports explain selection and emitted cost. Runtime captures supply
observed values. Neither infers a stable kernel ABI, reconstructs device-side
control flow, or replaces the author's choice of benchmark boundary. The
[agent workflow](../agent-driven-kernel-development.md) connects those distinct
evidence classes to optimization and integration gates.
