# XDNA HAL

The `xdna` driver runs native Loom array programs through libamdf. It owns
device discovery, native contexts, executable storage, buffers, direct queue
operations, and completion. The [image loader](image/README.md) admits `.xdna`
ELFs and applies their declared storage and binding contracts.

## Running a Loom scenario

Enable the native provider, HAL registration, and Loom XDNA emitter:

```sh
iree-bazel-configure -DAMDF_BUILD=ON -DIREE_HAL_DRIVER_XDNA=ON \
  -DLOOM_TARGET_XDNA=ON -DLOOM_EMIT_XDNA=ON -DLOOM_TARGET_VM=ON
iree-bazel-run //loom/src/loom/tools/iree-test-loom -- \
  loom/src/loom/tooling/target/amd/xdna/test/hal_execution.loom --device=xdna
```

The scenario compiles a finite multiply pipeline for the selected device and
compares its outputs with an independent VM implementation. Trial inputs vary;
tensor views exercise nonzero binding offsets, shared input storage, and output
guards. The device's exact target profile selects the compiler's array profile.

`xdna://0?columns=1` selects discovery ordinal zero and a one-column native
context. The default is one column. The context width must equal the compiled
array's width; a wider context does not implicitly pad the program. Endpoint
identity, firmware ABI, and geometry are checked when an executable is loaded.

The normal tool accepts finite `pipeline.def<kernel>` subjects with buffer
launch bindings and no leading specialization arguments. Specialization
arguments need a compiler preparation step that binds the scenario's
configuration values before emission; the adapter reports `UNIMPLEMENTED` for
that form. Runtime scalar bindings and invocation results are absent from the
current native image ABI.

## Execution and ownership

A device provisions one dispatch/transfer queue and one time-sliced native
context. Each accepted operation captures its transient arguments and retains
its resources. Semaphore dependencies express ordering. A consumer submitted
before its producer waits without occupying a native execution slot.

Queue-owned fixed-block arenas capture operation metadata, placed semaphore
timepoints, transfer descriptors, and dispatch bindings. UPDATE data is split
across reusable payload blocks and copied through one target mapping. The pools
grow for a new workload shape and then reuse those blocks; repeating a warmed
shape and depth performs no host allocation or free.

Each dispatch first attempts a nonwaiting claim on native publication. When the
persistent observer is active, the pending ring has capacity, and every wait is
proved by an exact accepted frontier, the calling thread prepares and submits
the command directly. A placed proactor receipt commits the accepted point and
transfers the claim to older deferred work before releasing it. Full capacity,
unsatisfied waits, inexact causal state, and claim contention use the private
publisher instead. libamdf reports its own full publication window as `BUSY`;
the HAL retains the prepared invocation and retries only after checked native
progress. The underlying OS submission can still wait for device wake or native
credits, so a direct caller can pay that latency. The private publisher isolates
that cost on every queued route.

The shared proactor owns queued causal admission, native acceptance commits,
and checked retirement. Accepted invocations occupy a bounded ring sized to the
native queue's prepared capacity. Each pending invocation owns exclusive mutable
command and binding storage; checked retirement returns that storage for reuse.
Immutable backing is shared when its static address references also point to
shared allocations. Binding updates publish only their patched ranges. Every
dispatch executes invocation zero to establish the array state; time slicing
does not promise resident tile state across submissions.

Eligible host operations run on an independent private worker and never advance
the native queue's completion frontier. Neither mapped transfer work nor native
submission can block the shared proactor. Both services return results through
placed proactor operations, preserving one owner for causal state, completion
publication, and terminal reclamation. A dependent dispatch whose producer is
still entering the native queue retries after that producer's acceptance; other
unresolved waits register their ordinary semaphore timepoints immediately. The
public queue call always captures its arguments before either direct or queued
publication takes ownership.

Direct fill, update, copy, upload, download, and queue barriers are supported.
Execution/access dependencies and global system visibility need no additional
XDNA command. Ranged memory effects return `UNIMPLEMENTED` because this queue
family has no native range-maintenance action and cannot promote a prepared
range without weakening its contract. Transfers use mapped native storage with
the required cache operations. Allocated buffers report the provider's actual
coherence properties. Loom correctness scenarios keep canonical host data in
coherent heap storage and stage device copies; buffer offsets and aliases
survive upload and readback.

Queue notifications are wake hints. One cold-registered persistent proactor
source consumes each event, refreshes the native queue's checked retirement and
terminal outcome, and requests a new one-shot native notification for the
oldest remaining point. No per-completion wait operation or source registration
is required. Ordinary retired work releases payload resources before signaling
HAL semaphores. An operation-held device reference keeps the queue's capture
pools alive through inline signal callbacks, then the arenas are returned before
the final device release.

Final device release places shutdown on the proactor owner. It joins the two
idle services, destroys the native queue, and begins terminal source
unregistration outside the source callback. The unregistration receipt places a
second operation before releasing the event, queue, context, and proactor entry.
If observation fails without proving retirement, the queue reports the error,
fails completion edges, and retains the unsafe-to-release ownership graph.
Teardown failures similarly report and retain ownership without aborting the
application or maintaining a live-object registry.

Reusable command buffers, device memory pools, external memory import/export,
file operations, host calls, and atomic queue operations return explicit
unsupported statuses. The direct path has no dependency on those mechanisms.

## Verification

Native correctness suites declare an XDNA hardware requirement:

```sh
iree-bazel-test --config=asan \
  //runtime/src/iree/hal/drivers/amd/xdna/cts:native_test \
  //loom/src/loom/tooling/target/amd/xdna:hal_execution_test
```

The native suite exercises submission capture, rebinding, guarded buffer views,
consumer-before-producer dependencies, direct transfers, failure propagation,
and allocation properties. The Loom suite exercises the public tool, compiler,
HAL, physical execution, readback, and numerical oracle together.
