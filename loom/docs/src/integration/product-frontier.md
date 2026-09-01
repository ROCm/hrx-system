# Embed command and kernel product frontiers

**Example source:** [`loom/binding/c/example/product_frontier.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/product_frontier.c)

A command product can finish before the executable kernels it references. The
parent owns portable command bytes and an explicit requirement table; each
source-backed child crosses the boundary as an independently owned executable
kernel request over ordinary Loom bytecode. A host can compile those requests
locally, send their bytecode and durable roots through its own compiler-service
protocol, or satisfy the same requirements from prebuilt executables. The
product descriptor that selects the kernel-product operation is deliberately
process-local and is not part of the serialized source.

The checked C example uses only the public `loomc` API. It deliberately accepts
the child synchronously so the ownership boundary remains visible; an
application worker pool uses the same callback to enqueue transferred requests
without changing command-product construction.

## Prepare immutable process state

The example creates one context, one worker-local workspace, and one frozen
link index over the source universe. It resolves the human-readable command
root once and keeps the index-wide ordinal for the hot path:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:prepare"
```

Frozen indexes and immutable sources can be shared by workers. A workspace is
mutable invocation scratch and belongs to one active worker.

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

## Route typed child requests

Every published child carries a normal `.loombc` source, exact source-local
root ordinals, the kernel-product descriptor, and provisional bindings to its
parent requirements. The executable-entry goal says the command program
already owns physical launch geometry, so the kernel product must not produce a
second host launch configuration. The request retains no mutable module,
workspace, link plan, or analysis state.

The example validates the request identity and parent binding before handing it
to target-specific scheduling:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:inspect-request"
```

Keeping the example at the request boundary makes it available in builds that
disable AMDGPU, SPIR-V, and LLVM emission. A target integration passes the same
request to `loomc_kernel_product_build_request` with its prepared compiler,
target pipeline, and executable emitter. The [kernel JIT guide](jit-kernel.md)
shows target specialization, native artifact emission, and runtime loading for
AMDGPU and SPIR-V.

The parent binding is separate from compilation identity, so two parents can
reuse one compiled product while binding its exports into different
requirement ordinals.

## Run the checked example

```shell
python dev.py bazel run //loom/binding/c/example:product_frontier
```

The output summarizes the mixed frontier:

```text
--8<-- "generated/examples/integration/product-frontier/summary.txt"
```

The one local kernel produces one child request. The external kernel accounts
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
filesystem round trip. [Schedule concurrent JIT work](jit-task-pool.md) shows
that composition with caller-owned completion and the optional standard task
pool.
