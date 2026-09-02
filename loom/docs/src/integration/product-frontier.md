# Parallelize kernel JIT compilation

**Complete example:** [`loom/binding/c/example/product_frontier.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/product_frontier.c)

Loading a model or specializing a workload can make dozens or hundreds of
kernels ready to compile at once. Compiling them serially extends startup
latency, while giving every kernel its own compiler or thread pool repeats
setup and oversubscribes the host.

The `loomc` API turns the live kernel set into small, independent build
requests. An application checks each request against its cache, submits only
the misses to a bounded scheduler, and binds the completed kernel products back
to the command program that requested them. Immutable compiler state is shared
across the process; each worker owns only its mutable scratch workspace.

The scheduler boundary is intentionally generic. Applications with an existing
executor implement `loomc_task_sink_t`. Applications without one can use
Loom's optional `loomc_task_pool_t` and `loomc_task_queue_t`. The compile task
and its ownership rules are identical in both cases.

This guide covers:

- when independent kernel compilation improves JIT latency;
- how command construction discovers only the kernels that remain live after
  linking and specialization;
- where cache lookup, scheduling, compilation, and completion belong; and
- how to use either an application scheduler or Loom's standard worker pool.

## Choose this path for batches and continuous specialization

Independent scheduling helps when one application event creates several
compilation opportunities:

| Application event | Useful scheduling policy |
| --- | --- |
| Model load | Compile the model's live kernel set concurrently and retain the loaded results with the model plan. |
| Shape or workload specialization | Reuse cached versions and schedule only newly selected semantic classes. |
| Continuous JIT | Keep a bounded compile queue alive and attach each invocation to its own completion object. |
| Autotuning or search | Submit independent configuration candidates without constructing a compiler per candidate. |

A single kernel compiled once during process startup does not need this
machinery. Ahead-of-time packaging remains a better fit when all relevant
target and workload facts are already known at build time.

## Split command construction from kernel compilation

A command program describes host-side orchestration and names the executable
kernel entries it will dispatch. Building that command program can determine
which kernel implementations are actually needed without compiling every
kernel body in the source catalog.

```text
command roots + configuration
             |
             v
loomc_cmd_program_product_build
             |
             +----> portable command bytes
             |
             +----> executable-entry requirements
                         |
                         +----> source-backed request -> cache -> compile
                         |
                         +----> external entry -------> application binding
```

The API reference calls this split a **product frontier**. A product is the
immutable output of one successful compiler operation. Its frontier is simply
the set of requirements that another product or the application must satisfy.
There is no frontier object to manage and no special source format: every
published kernel request contains ordinary Loom bytecode plus exact roots.

Command construction applies linking, configuration, dead-code elimination,
and schedule selection before publishing requests. Dead branches create no JIT
work. Launch sites that select the same semantic kernel class can share one
request. A bodyless `kernel.entry.decl` remains an external requirement and
does not create a fake compile job.

## Prepare a long-lived compile service

Prepare expensive immutable state once:

- one context and target environment;
- source catalogs frozen into link indexes;
- compilers and pass programs for the supported product routes; and
- a bounded scheduler with one `loomc_workspace_t` per worker.

The checked example uses Loom's standard four-worker pool and attaches one
compile queue:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:prepare-scheduler"
```

`loomc_task_pool_worker_count` reports the actual worker count after host CPU
affinity and topology are applied. The example allocates exactly that many
workspaces. Task callbacks receive a dense, mutually exclusive worker ordinal,
so selecting scratch is an array lookup rather than a lock or workspace lease.

An application-owned executor replaces only the sink and worker setup. It
implements `loomc_task_sink_t::submit`, assigns each execution lane a stable
ordinal, and invokes `loomc_task_execute` exactly once for every accepted task.
The application does not link `loomc/task_pool.h`, `loomc/task_queue.h`, or the
IREE task runtime in that configuration.

## Publish compile requests while building the command

Pass a `loomc_request_sink_t` when building the selected command roots:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:build-command"
```

`loomc_cmd_program_product_build` is synchronous with respect to the command
product, but calls `request_sink.publish` as independent kernel requests become
available. The callback may submit work immediately, allowing kernel
compilation to overlap the remainder of command construction.

The example command uses three source-backed kernels and one externally
provided entry. Command construction therefore publishes three requests while
returning four executable-entry requirements. Omitting `request_sink.publish`
builds the same command bytes and requirements without opening kernel
implementation bodies; that is useful when all executables are already
available from another provider.

Publication is provisional until command construction returns both an OK
status and a succeeded result. A latency-oriented service can schedule requests
immediately and discard their parent bindings if the command later fails. A
service that cannot tolerate speculative work can retain the requests in the
callback and submit them only after the parent succeeds.

## Check the cache before crossing the scheduler

The request callback is the cache boundary. A production key normally includes:

- request bytecode contents and exact root ordinals;
- the required product route;
- target profile and invocation configuration;
- selected pass pipeline and artifact options; and
- compiler and executable-format compatibility versions.

A cache hit records the existing product against the request's parent bindings
without allocating a task. A miss wraps the immutable request in an
application-owned task record and submits it. Cache policy stays outside
`loomc`: an in-process JIT, a compiler service, and a persistent artifact cache
can use different storage and eviction policies over the same requests.

The checked callback omits a cache because every process run starts empty. Its
scheduler-facing path is the same path used after a real cache miss:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:schedule-request"
```

The publication callback owns `request` at entry. Successful task submission
transfers the task to the sink, which executes and destroys it exactly once. A
rejected submission leaves task ownership with the caller. The output slot
retains the request until its completed product has been bound or discarded.

The request's process-local product descriptor selects the compilation route.
This example accepts the compiled-module route used by
`loomc_compile_request`. A service supporting additional product families uses
the descriptor as a dispatch key rather than guessing from symbol names,
filenames, or target strings.

## Compile with worker-local scratch

Each task shares the prepared compiler and pass program and selects its
workspace by worker ordinal:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:compile-task"
```

The example requests compiled `.loombc` so it runs in target-independent
builds. A native JIT gives the task its prepared target pipeline and continues
through the compile-and-emit sequence shown in [Embed kernel JIT
compilation](jit-kernel.md). The scheduling contract does not change when the
result becomes an HSACO, SPIR-V module, or another runtime executable.

Task execution writes its `loomc_status_t`, `loomc_result_t`, and product into
caller-owned completion state. A non-OK status reports API misuse or an
infrastructure failure. An OK status may still accompany a failed compiler
result with diagnostics. Both must be checked before caching or loading a
product.

## Bind completed kernels to the command

Every request carries provisional mappings from requirement ordinals in the
parent command product to root ordinals in the child request. Successful child
products preserve request-root order as export order, so the same mapping
identifies the compiled export that satisfies each command requirement:

```c
--8<-- "generated/examples/integration/product-frontier/product_frontier.c:bind-results"
```

The example verifies three compiled bindings and leaves the bodyless
`@external` entry for the application. A real runtime uses those mappings to
join loaded executable exports to command-program entry slots. Two command
products may reuse one cached child product while carrying different parent
requirement ordinals.

Completion belongs to the application request, not to the worker pool. The
short-lived example shuts down its queue to drain one finite batch. A
continuous JIT keeps the pool and queues alive, increments a per-command
pending count for each accepted request, resolves cache hits immediately, and
signals its future, callback, or event-loop message when every required binding
has completed. Observing an empty queue is not a completion protocol because
another producer may submit work at any time.

## Run the checked example

```shell
python dev.py bazel run //loom/binding/c/example:product_frontier
```

The output distinguishes compiled source kernels from the executable that the
application must provide:

```text
--8<-- "generated/examples/integration/product-frontier/summary.txt"
```

For the generic task protocol and standard pool lifecycle in isolation, see
[Use the JIT task pool](jit-task-pool.md). For target profiles, native artifact
emission, launch configuration, and runtime loading, continue with [Embed
kernel JIT compilation](jit-kernel.md). The generated [`loomc` C API
reference](../reference/c-api/index.md) defines the exact ownership and
thread-safety contract for every handle used here.
