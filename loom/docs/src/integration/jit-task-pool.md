# Use the JIT task pool

**Example source:** [`loom/binding/c/example/jit_task_pool.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/jit_task_pool.c)

LoomC separates a queueable unit of work from the scheduler that executes it.
An embedding can submit the same `loomc_task_t` records to its own event loop,
deterministic executor, or an optional standard `loomc_task_queue_t`. The core
compiler API does not link the task runtime. The standard implementation splits
its shared worker population (`loomc_task_pool_t`) from each independent
task domain (`loomc_task_queue_t`), and neither object owns compiler policy,
caches, workspaces, requests, products, or completion state.

This separation keeps cache hits on the shortest path. A host first derives the
identity of the requested product and checks its process-local cache. Only a
miss allocates a task and crosses a scheduler boundary:

```text
request -> classify -> cache lookup -> hit  -> reuse product
                                    -> miss -> submit task -> compile -> publish
```

The checked example loads eight immutable typed configuration modules, then
compiles eight independently configured program modules through the public
LoomC API. Configuration parsing stays outside the worker callback; each task
borrows its prepared configuration module for the compile invocation. The
example is intentionally target-independent; a native JIT uses the same task
shape with its prepared target pipeline and emitter.

## Pipeline independent work domains

One process-wide pool can drive compiler queues, HAL materializers, and
application processes without forcing their work through one FIFO:

```text
                             +-> compile queue --------+
request -> cache miss -------+                         +-> publish product
                             +-> command queue --------+
                                                       |
shared worker pool <------ HAL executable loader <-----+
                   <------ command-buffer recorder <---+
                   <------ application processes
```

A queue owns readiness and scheduling only within its domain; callbacks from
one queue may execute concurrently and complete in any order. A compile
callback that publishes an executable-load task therefore makes that task
immediately eligible on the shared workers instead of placing it behind the
remaining compilation roots. The worker population remains bounded while
compilation, executable loading, and command-buffer recording pipeline
naturally.

Applications using the IREE task runtime include `loomc/iree/task_pool.h` to
allocate a pool from an existing `iree_task_executor_t` or borrow the executor
created by a LoomC pool. HAL integrations attach their own cooperative
processes to that executor and do not depend on LoomC task queues. Applications
using another scheduler continue to implement `loomc_task_sink_t` directly.

## Split process, worker, and request state

The scheduler exposes a dense, mutually exclusive worker ordinal on every task
callback. That ordinal indexes caller-owned mutable scratch directly, without a
workspace lease, hash lookup, or lock:

| Lifetime | Example state |
| --- | --- |
| Process | Context, source catalog, frozen link indexes, compiler, pass programs, cache, shared task pool, and independent queues. |
| Worker | One `loomc_workspace_t` and any target-specific scratch indexed by worker ordinal. |
| Request | Immutable request/configuration, task record, terminal result, and application completion state. |

The example prepares the shared compiler state, requests a four-worker pool,
attaches one compilation queue, and allocates exactly one workspace per actual
worker:

```c
--8<-- "generated/examples/integration/jit-task-pool/jit_task_pool.c:configure"
```

Four workers are the default and suit common latency-oriented JIT workloads.
Eight is another normal application choice when independent compilations have
enough work to amortize scheduling. Search and autotuning services can request
wider pools when many parameter variants produce independent reports. The
actual worker count may be lower when the process affinity or container CPU set
exposes fewer physical cores.

## Keep the task record application-owned

A concrete task embeds `loomc_task_t` at offset zero and carries only the state
needed to terminalize that operation:

```c
--8<-- "generated/examples/integration/jit-task-pool/jit_task_pool.c:task-record"
```

Successful submission transfers the task reference to the sink. The sink runs
the callback exactly once and then invokes its destructor exactly once. A
rejected submission leaves ownership with the caller, which may retry another
sink or destroy the task immediately.

The callback selects worker-local scratch by ordinal and otherwise uses shared
immutable compiler state:

```c
--8<-- "generated/examples/integration/jit-task-pool/jit_task_pool.c:execute"
```

Task execution has no scheduler-facing status return. The concrete task writes
its compiler result or infrastructure status into caller-owned completion state
before returning. This prevents an application compilation failure from
poisoning unrelated tasks or turning the scheduler into a second compiler API.

## Compose completion separately from scheduling

The compact example submits one finite batch and uses queue shutdown as its
join:

```c
--8<-- "generated/examples/integration/jit-task-pool/jit_task_pool.c:submit"
```

A long-lived compiler service instead allocates one pool and its attached queues
for the service lifetime, then uses request completion objects, callbacks,
futures, or event-loop messages. Neither a queue nor the pool exposes a global
idle wait: another producer may publish work at any time, and observing a
transiently empty domain does not mean an application request is complete. The
request owner knows exactly which products it is waiting for and terminalizes
its completion object when all of them have resolved.

Queue shutdown is drain-only. It stops that domain from accepting work, executes
every accepted task, and releases its cooperative process without affecting
other domains attached to the pool. Work that recursively publishes child tasks
therefore completes its request before its queue begins teardown; publication
attempted after shutdown is correctly rejected. Attached queues
retain the shared executor, so they may outlive the convenience pool handle and
the final owner joins the worker threads.

## Run the checked example

```shell
python dev.py bazel run //loom/binding/c/example:jit_task_pool
```

The output confirms that every scheduled variant produced its requested
bytecode artifact:

```text
--8<-- "generated/examples/integration/jit-task-pool/summary.txt"
```

Applications that already own a scheduler include `loomc/task.h` and implement
`loomc_task_sink_t::submit`. Applications wanting the standard implementation
link `//loom/binding/c:task_pool` and `//loom/binding/c:task_queue`, then include
`loomc/task_pool.h` and `loomc/task_queue.h`. Both paths use the same concrete
task records and preserve the same ownership contract.
