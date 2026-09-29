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

## Single-row Qwen execution

`qwen` is a concrete Qwen3.8-27B UD-Q5_K_XL caller, not an HTTP service. It loads
the canonical GGUF into one resident parameter slab, prepares prefill and decode
commands once, and invokes the compiled `qwen_model.loom` through one VM process.
Device row state survives each invocation. Host uploads, stage execution, and
readback share explicit timeline ordering. Greedy selection and position/history
updates are part of the compiled stages; the host stops on their EOS result or
the requested output length. Text output excludes special tokens.

Model math is supplied as separately compiled artifacts, not built into this
binary. Each stage directory contains:

```text
manifest.json                 Version 2 loom-command-set manifest, one root.
commands/<artifact>           Portable command named by that root.
kernels/<request-stem>.hsaco   Compiled image for each manifest entry.
```

The manifest supplies entry names and the root-local entry mapping. It is not
an execution language: command order, arguments, parameter placement, and
transient requirements come from the portable command artifact. Both stages
must place every parameter identically; the loader compares their keys,
fixed-buffer indices, offsets, lengths, and alignments before sharing weights.
Allocation satisfies both stages' published slab size/alignment requirements.

This caller's explicit model configuration is one fixed weight root, a
512-token context, at most 128 generated tokens, and seven rebindable slots:

| Slot | Buffer contract |
| --- | --- |
| 0 | Residual rows: 512 × 5120 f32 values. |
| 1 | Three i32 values: prefill count/base/EOS, then decode position/count/EOS. |
| 2 | GDN convolution and recurrent state: 156,893,184 bytes. |
| 3 | Sixteen attention layers' KV state: 512 × 65,536 bytes. |
| 4 | 512 i32 token IDs; after prefill, current token at 0 and history from 1. |
| 5 | Eight i32 progress values; generated count at 0 and EOS flag at 7. |
| 6 | Shared scratch sized/aligned for the larger requirement of both stages. |

The stage sources must use this state layout and specialize prefill to the exact
encoded prompt length. `--prefill_tokens` checks the encoded length against the
caller-supplied specialization; it does not recover model semantics from kernel
reflection. The default no-thinking arithmetic prompt is 24 tokens with the
model's tokenizer. Different prompt lengths require matching prefill artifacts.

```sh
build_tools/bin/iree-bazel-run --config=asan //experimental/loom_serve:qwen -- \
  --prefill=/path/to/compiled/prefill \
  --decode=/path/to/compiled/decode \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --prefill_tokens=24 --max_tokens=16
```

All row buffers and scratch are allocated before generation. The current host
loop waits for each token's readback; it provides a full-model correctness
witness, not batching, a zero-allocation measurement, or a throughput claim.
