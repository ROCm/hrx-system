# Embed command and kernel product frontiers

**Example source:** [`loom/binding/c/example/product_frontier.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/product_frontier.c)

A command product can finish before the executable kernels it references. The
parent owns portable command bytes and an explicit requirement table; each
source-backed child crosses the boundary as an independently owned ordinary
Loom bytecode request. A host can compile those requests locally, send their
ordinary bytecode and durable roots through its own compiler-service protocol,
or satisfy the same requirements from prebuilt executables. The product
descriptor that selects a local pipeline is deliberately process-local and is
not part of the serialized source.

The checked C example uses only the public `loomc` API. It deliberately runs
the parent and child synchronously so the ownership boundary remains visible;
an application worker pool uses the same callback to enqueue transferred
requests without changing the compiler operations.

## Prepare immutable process state

The example creates one context, one worker-local workspace, and one frozen
link index over the source universe. It resolves the human-readable command
root once and keeps the index-wide ordinal for the hot path. A compiler and
prepared pass program are also created once for child requests:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:prepare"
```

Frozen indexes, compilers, pass programs, and immutable sources can be shared
by workers. A workspace is mutable invocation scratch and belongs to one active
worker. The example reuses one workspace only because its parent and child
operations run sequentially.

## Publish only source-backed child work

The parent source contains one local `kernel.def` and one bodyless
`kernel.entry.decl`. Command construction creates two executable-entry
requirements, but it publishes a request only for the local definition. The
external entry remains a real requirement for the embedding to bind from HIP,
SPIR-V, HSACO, or another provider.

The callback accepts ownership of each immutable child request:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:publish-request"
```

Publication is provisional until command construction returns an OK status
with a successful result. An asynchronous host may begin a cache lookup or
compilation immediately, but it commits the child-to-parent binding only after
the parent succeeds. Already transferred request references remain owned by
the receiver if the parent later fails.

Supplying the callback enables source publication:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:build-command"
```

Leaving `request_sink.publish` null is the body-blind path. It returns the same
command programs and entry requirements without opening, classifying, cloning,
or serializing implementation bodies.

## Compile children as ordinary requests

Every published child carries a normal `.loombc` source, exact source-local
root ordinals, the required process-local product descriptor, and provisional
bindings to its parent requirements. It retains no mutable module, workspace,
link plan, or analysis state.

The example compiles its child through the generic prepared compiler and asks
for a bytecode module product:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:compile-request"
```

This target-independent pipeline keeps the example available in builds that
disable AMDGPU, SPIR-V, and LLVM emission. A target integration selects its
normal target pipeline and executable emitter for the same exact request. The
[kernel JIT guide](jit-kernel.md) shows target specialization, native artifact
emission, and runtime loading for AMDGPU and SPIR-V.

The child product preserves request-root order as export order and has no
unresolved requirements. Its parent binding is separate from compilation
identity, so two parents can reuse one compiled product while binding its
exports into different requirement ordinals.

## Run the checked example

```shell
python dev.py bazel run //loom/binding/c/example:product_frontier
```

The output summarizes the mixed frontier:

```text
--8<-- "generated/examples/integration/product-frontier/summary.txt"
```

The one local kernel produces one child product. The external kernel accounts
for the second parent requirement without producing a fake source request.

## Place caching and concurrency outside compilation

The low-level flow has four independent decisions:

| Decision | Interface |
| --- | --- |
| Which command roots to build | Exact ordinals in a frozen link index or ordinary request |
| Whether source-backed children are needed | Presence of `loomc_request_sink_t::publish` |
| Which pipeline produces a requested representation | `loomc_request_product_descriptor` plus embedding policy |
| Where work and reuse live | The embedding's queue, worker-local workspaces, and process-local cache |

No task runtime or cache implementation is required by the compiler API. A
single-threaded application can run the example literally. A concurrent host
retains each request, checks its cache before leasing a workspace, and schedules
only misses. Product artifacts are immutable byte sequences, so successful
results can pass directly to executable loaders or packaging code without a
filesystem round trip.
