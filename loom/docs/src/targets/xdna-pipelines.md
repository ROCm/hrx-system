# Resident pipelines on XDNA

A resident pipeline keeps data moving between independently progressing strands
inside one device invocation. DMA can fill an input while a compute worker
processes another record. Channels govern when each record becomes readable and
when its storage can be reused; the host does not dispatch every stage.

This walkthrough follows a BF16 gate/up projection from authored Loom to a
native XDNA executable. The same source is a maintained compiler workload.
It exercises ordinary functions, owned channels, adjacent tile memory, DMA,
and specialization. It computes one projection pair and a quadratic gate,
not a complete FFN or an optimized matrix multiplication.

The current native realization requires explicit singleton worker selections,
specialized storage sizes, and channel protocols supported by the target.
Each worker image must fit its tile's instruction budget. GPU and CPU strands
need their own execution and progress mechanisms; selecting a different target
does not yet turn this example into a GPU or CPU executable.

## Compile a complete program

Save [gated-projection.loom](../generated/examples/targets/xdna-pipelines/gated-projection.loom).
It includes all helper functions and the public
`@ffn_gate_up_quadratic_bf16` pipeline. With an installed Loom toolchain that
includes XDNA, compile for Strix Halo NPU5:

```shell
loom-format --check gated-projection.loom
loom-compile gated-projection.loom \
  --root=@ffn_gate_up_quadratic_bf16 \
  --target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11 \
  --config=ffn_gate_up.input_size=512 \
  --format=xdna --output=projection.xdna
```

For Strix NPU4, use `amd.xdna.aie2p:amd.xdna.strix.17f0_10`.
Compilation needs no attached NPU. Execution requires a device matching the
exact profile; these images are not interchangeable.

The caller supplies three buffers in source argument order:

| Argument | Contents at width K | Access |
| --- | --- | --- |
| Input | One row of K BF16 values. | Read. |
| Weights | Two rows of K BF16 values, gate then up. | Read. |
| Output | One F32 value carrying the rounded BF16 result. | Write. |

All three bindings satisfy the source's 64-byte base-alignment contract. K is
a compile-time configuration from 512 through 4096, divisible by 32. Dot
products accumulate in F32. The epilogue computes
`gate * (0.5 + 0.25 * gate) * up`, rounding each step to BF16. This explicitly
chosen quadratic gate is different from SiLU or SwiGLU.

## Construct storage and independently progressing strands

Here is the complete pipeline body from that source. Its helper definitions
are in the downloadable file; the transfer helpers appear below.

```loom
--8<-- "generated/examples/targets/xdna-pipelines/pipeline.loom"
```

`pipeline.def<kernel>` requires one directly loadable artifact. Its leading
parentheses contain specialization arguments, empty here; `run` declares the
values supplied for each invocation. The selected target provides hardware
facts. The target symbol is not a live device or a shared mutable execution
instance.

A `pipeline.strand` declares work rather than executing inline at its position
in the construction region. Each selected worker enters its strand once.
Ordinary `scf.for` loops express repetition. Reaching the closing brace completes
that strand instance; it does not implicitly restart it whenever input arrives.
Strands capture lexical SSA values, including buffers and channel identities.

The three coordinate lists are origins, counts, and strides. On this target,
`workers([0, 2], [1, 1], [1, 1])` selects column 0, row 2, and
`workers([0, 3], [1, 1], [1, 1])` selects its adjacent compute tile. Coordinate
`i` within a dimension maps to `origin + i * stride`.

A strand need not consume a compute core. After specialization, this example's
transfer strand consists entirely of fixed DMA transfers and channel actions.
The device invocation command stream issues its copies and preserves its waits
and publications. Only the arithmetic strand needs tile instructions. A transfer
strand and a compute strand can also select the same tile when communication
has this realization.

The command stream is one sequential engine, so the compiler assigns at most
one strand to it. This realization requires an acyclic unconditional sequence,
fixed transfer endpoints, and at most one admission at each channel endpoint.
Bounded ingress loops have a separate realization: the DMA engine itself can
reserve a free channel slot, copy the next record, and publish it. Independent
DMA streams keep separate engines and routes rather than sharing the command
sequencer's execution order. Communication that needs instruction execution
still runs on its selected core.

`pipeline.memory<workgroup>[0, 3]` selects the invocation's tile-local pool.
It allocates nothing and introduces no synchronization. The three `buffer.alloca`
operations create distinct allocation roots in that pool. Their `buffer.view`
operations describe how slots and payload elements occupy those allocations.
Compute accesses and DMA routes must have legal mappings to that memory.
Adjacency permits direct shared-memory access between these selected tiles.

The leading view dimension is the number of storage slots. For example,
`view<2x[%input_size]xbf16>` supplies two K-element records to
`channel<tile<[%input_size]xbf16>>`. The channel type describes one record, while the bound
view and capacity describe its storage. Binding creates a fresh protocol
identity: equal addresses do not imply equal channels or control dependencies.

At K=512, the authored channel storage totals 4104 bytes: 2048 for input slots,
2048 for retained weights, and 8 for output slots. The projection also uses a
64-byte private reduction buffer. Alignment and transport/control storage are
additional physical requirements accounted for during lowering. The compiler
checks the complete placement against the device's available memory.

## Publication and reuse surround the actual transfer

The ingress helper reserves writable storage, launches a copy into it, waits
for that copy, then publishes the initialized record:

```loom
--8<-- "generated/examples/targets/xdna-pipelines/gated-projection.loom:fill-input"
```

`write<tile<[%input_size]xbf16>>` owns the reservation. The view is a borrowed way to access
its payload, so ordinary vector operations and functions can use it.
Publishing consumes the write obligation. Publishing before the DMA completes
would let the consumer read incomplete data.

The output helper follows the other side of the lifetime:

```loom
--8<-- "generated/examples/targets/xdna-pipelines/gated-projection.loom:drain-output"
```

`channel.acquire` returns an owned read plus a ready payload view. The read
stays owned through the outgoing DMA; only `channel.release` permits recycling
that record's storage. Waiting for the DMA is a storage-lifetime edge, not a
global barrier across the pipeline.

The compute strand holds its weight read across its entire loop. Each input
and output record has a shorter lifetime inside that loop. More records can
reuse the same weights and slots without a new host submission. This maintained
workload sets the record count to one; its two-slot storage and prefill/drain
loops express the streaming schedule even when specialization removes that
repetition. A larger stream also needs input and output bindings sized for its
record count.

This schedule preloads at most two inputs and then drains an output before
refilling an input. That order matters: a service worker that blocks filling
an input while the compute worker blocks on a full output can deadlock. Buffer
capacity and control flow are part of the author's schedule in this native
cut; channel ownership alone is not a proof that every authored network makes
progress.

## Repeating ingress without a feeder core

A regular input stream can move its entire reserve/copy/wait/publish loop into
DMA descriptors. For example, eight external records can cycle through two
tile-local slots while the compute strand retains its weights. Each DMA
iteration acquires a free credit and publishes a ready credit only after the
local write completes. The consumer returns free credit through its ordinary
`channel.release`. A full input channel stalls that DMA stream without making a
compute core poll for space.

This realization preserves supported strided record layouts. Descriptor
iteration advances the external record address and rotates the local slot;
it does not gather each record into another transient buffer. On the current
profiles, one task supports up to 256 executions, with up to 64 distinct external
record offsets and 64 local slots. A repeatedly read fixed external record can
use all 256 executions. Memory, descriptor, route, and lock limits still apply.

Read-ahead needs an ownership proof. An external DMA can fetch bytes before the
corresponding local slot becomes free. The compiler therefore requires disjoint
caller-buffer facts and verifies that the complete pipeline cannot write the
source while those reads are in flight. Distinct buffer arguments alone do not
establish disjointness; an author's `buffer.assume.noalias` is a caller contract.
Imported channels or opaque effects that prevent this proof keep instruction
execution. The compiler also requires the consumer to drain the complete
produced prefix before invocation completion, so no task still owns its backing
storage when the caller reuses it.

Output lifetime remains separate. A compute strand can drain each result with
the output helper above, holding its read until the external write completes.
Ingress offload does not authorize releasing that output after only a local
source read. Configuration starts autonomous input streams before potentially
blocking command-stream work, and resets the local iteration cursor between
invocations when the record count ends partway around the slot ring.

The resulting compile report counts instruction images and channel storage
separately, so eliminating a feeder image cannot hide its storage cost.
Successful lowering proves that the program fits this realization; device
execution and timing remain separate qualification steps.

## Inspect the real compilation boundaries

Add IR tracing to the same successful compile when investigating placement or
generated worker code:

```shell
loom-compile gated-projection.loom \
  --root=@ffn_gate_up_quadratic_bf16 \
  --target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11 \
  --config=ffn_gate_up.input_size=512 --format=xdna \
  --dump-ir-after=outline-pipeline-strands \
  --dump-ir-after=aie2p-lower-pipeline \
  --dump-ir-output=trajectory/ \
  --compile-report=summary --compile-report-output=projection.report.json \
  --output=projection.xdna
```

The trace directory contains complete modules, with a trace index identifying
their boundaries. The generated examples below come from those compiler
outputs, with trace comments removed from the downloadable modules.

### Explicit callable boundaries

The [outlined module](../generated/examples/targets/xdna-pipelines/outline-pipeline-strands.loom)
still contains the pipeline's storage and placement. Each strand calls an
ordinary function with explicit captured arguments. The public three-buffer
invocation ABI has not acquired those internal channel arguments.

??? example "Complete compiler-produced outlined module"

    ```loom
    --8<-- "generated/examples/targets/xdna-pipelines/outline-pipeline-strands.loom"
    ```

Shapes and target environments are visible in this IR. The normal compiler has
also simplified the single-record loop and lowered remaining structured loops
to CFG. A helper that accepts only borrowed views still performs ordinary
memory computation. Channel-aware helpers expose ownership effects through
their definitions before worker realization.

### Tile code and configuration code

The [native realization module](../generated/examples/targets/xdna-pipelines/aie2p-lower-pipeline.loom)
contains the arithmetic worker and configuration functions. The worker implements
its channel credits and arithmetic. Configuration code establishes routes,
storage and bindings, loads the worker image, and starts execution. It also
implements the finite transfer strand's ordered DMA and channel actions before
joining the final completion obligations. No transfer image is loaded.

An input publication still follows its DMA completion. An output acquisition
can become a descriptor's credit acquire when its copy and wait immediately
follow it; otherwise it remains an explicit ordered wait. The final command
waits for the external output write before returning. Channel storage remains
allocated even when communication requires no tile instructions, and the
compile report accounts for storage and worker images separately.

Both are ordinary target IR. In particular, `program.load` is an AIE2P
**configuration instruction**, inside a Low function targeting
`amd.xdna.aie2p.configuration`. It selects
the column, row, and tile-code symbol to load. It is not a generic pipeline
operation, a host callback between records, or a separate executable category.

| Term | Meaning in this example |
| --- | --- |
| Role | The logical transfer or projection computation authored by functions and control flow. |
| Strand | One independently progressing source body, entered once per selected worker instance. |
| Worker program | Tile instructions for the computation and communication assigned to a core. A DMA-realized strand needs no worker program. |
| Tile image | Linked instructions and data admitted against one tile's physical budgets. |
| Exported entry | The named invocation ABI in the containing ELF. |

The `.xdna` ELF can contain multiple exports and code contributions. A tile
program is not the ELF itself. Sharing immutable code does not merge invocation
storage or channel identities. This lowering emits resident images that fit
the instruction budget; it does not yet insert in-invocation image replacement.
That requires a target realization which preserves live storage and ownership
while transferring instruction residency. Splitting a large program into
functions alone does not remove the 16 KiB tile instruction limit.

### Resume compilation from text or bytecode

Save either complete generated module, for example the native realization as
`native.loom`, and compile it without the original source or analysis process:

```shell
loom-format native.loom --to=bc --output=native.loombc
loom-compile native.loombc --root=@ffn_gate_up_quadratic_bf16 \
  --format=xdna --output=resumed.xdna
```

The IR already contains its specialized target and program semantics. Analysis
is reconstructed from that module; no saved compiler-side planning object is
required. The walkthrough exercises both boundaries on both profiles and
compares the native artifacts produced by their text and bytecode forms.

Use the [compile-report workflow](../workflows/compile-reports.md) to inspect
the native artifact's code, allocation and scheduling evidence. Successful
compilation establishes target admission; it does not establish numerical
correctness or measured device throughput.

## The caller owns the invocation lifetime

The normal `iree-test-loom` path can compile and execute buffer-bound XDNA
pipelines through the
[XDNA HAL](https://github.com/ROCm/hrx-system/tree/main/runtime/src/iree/hal/drivers/amd/xdna):

```shell
iree-test-loom checked-pipeline.loom --device=xdna
```

The HAL selects the exact compiler profile from immutable device facts, loads
the emitted ELF, stages the scenario buffers, submits the entry, and preserves
their lifetimes through completion and readback. The current image ABI accepts
buffer launch bindings. Leading specialization values must already be bound
into compiler SSA before image emission; runtime scalar bindings and invocation
results require a future image ABI.

The HAL queries each entry's backing and binding requirements and prepares its
establishing command. libamdf submits the prepared instruction range. It does
not interpret channels, invent a work schedule, or patch each pipeline record.

The establishing command loads/configures the workers once for that invocation.
All record processing and input/output handoffs occur inside it. The emitted
completion path waits for the program's worker and transfer obligations before
the caller can reclaim their resources. A separate device consumer of the
output still needs its own synchronization and lifetime edge.

The current time-sliced native context does not guarantee tile state survives
between independent submissions. Reusing the same context or immutable command
allocation does not establish that guarantee. Each independent invocation uses
the establishing command; retaining state within one invocation is a different
contract from retaining it between submissions.

This is the boundary a complete pipeline launcher must own: materialize
transient allocations, bind the caller's I/O, submit work, and track completion.
The XDNA HAL implements those steps for finite buffer-bound pipelines. General
heterogeneous launch lowering, GPU participation/progress, and code-image
replacement each need a consuming realization before they become executable
capabilities of the source model.
