# Loom model runner experiment

This branch-local runner connects compiled model-control VM code to prepared
HAL commands. Its private interfaces are intended to change with real models.
It is not a general HAL VM module or a serving framework.

`command.c` combines a parsed portable command program with loaded executable
reflection. Recording retains fixed buffers and code. Rebindable slots include
model state and explicit scratch; the materializer allocates no device backing.
There is no semantic-resource extension to the command artifact format.

`module.c` publishes prepared stages as ordinary fixed-signature VM imports:
`(hal.buffer, ... slots ...) -> i64`. Model code selects the stage and buffers;
the native call submits and returns without waiting. HAL captures bindings and
retains their buffers independently of the VM invocation. One process can serve
many rows; its preallocated native binding scratch is reused between calls.

`execution.c` is the shared execution capability used by both the model and
host I/O. Each accepted submission waits on the preceding submission and signals
the next timeline value, including transfers across different exact queues.
This deliberately serializes shared scratch use. The returned integer is scoped
to that execution object, not a global completion handle. Queue rejection does
not advance the accepted frontier. The host drains accepted work before
recycling borrowed host payloads and observes asynchronous failures explicitly.

The GPU integration test compiles Qwen generation-state kernels, command programs,
and a VM entry from source. Two retained rows share code, commands, and one VM
process while one row pauses and resumes. Checked outputs cover token history,
position, padding, and EOS. A rejected native binding must leave the execution
domain usable. This is an ownership/control witness, **not a full Qwen model
run or a performance result**. The test artifacts currently target gfx1151.

From the worktree root:

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve:control_test
```

Weights, KV pools, model forward stages, scheduling policy, and transport are
separate from this coarse execution boundary. None is inferred from buffer
contents or command names by the native submission code.
