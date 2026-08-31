# Schedule concurrent JIT work

**Example source:** [`loom/binding/c/example/jit_task_pool.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/jit_task_pool.c)

LoomC separates a queueable unit of work from the scheduler that executes it.
An embedding can submit the same `loomc_task_t` records to its own event loop,
deterministic executor, or the optional standard `loomc_task_pool_t`. The core
compiler API does not link the task runtime, and the standard pool does not own
compiler policy, caches, workspaces, requests, products, or completion state.

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

## Split process, worker, and request state

The scheduler exposes a dense, mutually exclusive worker ordinal on every task
callback. That ordinal indexes caller-owned mutable scratch directly, without a
workspace lease, hash lookup, or lock:

| Lifetime | Example state |
| --- | --- |
| Process | Context, source catalog, frozen link indexes, compiler, pass programs, cache, and task pool. |
| Worker | One `loomc_workspace_t` and any target-specific scratch indexed by worker ordinal. |
| Request | Immutable request/configuration, task record, terminal result, and application completion state. |

The example prepares the shared compiler state, requests a four-worker pool,
and allocates exactly one workspace per actual worker:

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

The compact example submits one finite batch and uses pool shutdown as its
join:

```c
--8<-- "generated/examples/integration/jit-task-pool/jit_task_pool.c:submit"
```

A long-lived compiler service instead allocates one pool for the service
lifetime and uses its own request completion objects, callbacks, futures, or
event-loop messages. The pool intentionally has no global idle wait: another
producer may publish work at any time, and observing a transiently empty queue
does not mean an application request is complete. The request owner knows its
actual frontier and terminalizes it when every required product has resolved.

Shutdown is drain-only. It stops accepting work, executes every accepted task,
and releases the worker population. Work that recursively publishes child tasks
therefore completes its application frontier before service teardown begins;
publication attempted after shutdown is correctly rejected.

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
`loomc_task_sink_t::submit`. Applications wanting the standard pool additionally
link `//loom/binding/c:task_pool` and include `loomc/task_pool.h`. Both paths use
the same concrete task records and preserve the same ownership contract.
