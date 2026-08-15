# Qwen Experimental Runtime Guide

Read `experimental/qwen/README.md` before editing this experiment.

This tree is a standalone, non-shipping experiment. It may study and fork
small reusable mechanisms from other experimental trees, but it never depends
on, links against, includes, or exposes symbols from them.

Loom command programs own model scheduling, parameter names, tensor views,
transient packing, synchronization, fusion, and kernel selection. C owns the
generic GGUF/tokenizer sources, HAL resources, compilation cache, command
package materialization, and queue issue. Adding or changing a model operation
must not require a C routing change.

The first Qwen3.8 vertical slice is one complete GDN layer over exact
UD-Q5_K_XL parameter payloads, including its recurrent-state transition. A
synthetic scheduler, isolated kernel, or host-side tensor schema is supporting
evidence rather than completion of that milestone.

Every program uses explicit semaphore dependencies. HAL queues are not FIFO,
and a reusable command buffer is not issued concurrently unless its materialized
contract explicitly permits it.

Compiler and kernel failures fail loud. Preserve a minimal reproducer for the
owning compiler/runtime boundary and fix the contract; never silently select a
fallback implementation.

Python and source-rewriting patchers are not Loom integration machinery. The
runtime links canonical authored Loom source and never generates, rewrites, or
intercepts it as a workaround.
