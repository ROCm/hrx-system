# Qwen Owned Runtime Agent Guide

Read `experimental/qwen/README.md` before editing this experiment.

This tree is a standalone, non-shipping experiment. It may privately fork a
small mechanism after studying `experimental/id4`, but it never depends on,
links against, includes, or exposes symbols from that tree. Forked mechanisms
are renamed around Qwen's contracts and simplified to the behavior this model
actually needs.

The independently moving Qwen kernel corpus lives under
`experimental/qwen_moe`. Runtime code consumes its generated linked modules
through Qwen-owned export declarations. It does not move model ownership,
command recording, parameter gathering, or benchmark policy into the kernel
tree.

The first vertical slice is a complete layer-0 prefill program over the real
GGUF payloads. Host-only schema tests and isolated kernel tests are supporting
evidence; the milestone requires the same runtime to record and submit the
layer from both the CLI and the Google Benchmark executable.

Keep the CLI and benchmark as separate process owners. Neither executable
contains kernel names, weight offsets, scratch layout, or mathematical
scheduling. Layer rows may upload captured hidden states because layer-level
execution is a durable optimization surface, but fixture formats and expected
values remain in the executable or testing layer rather than the runtime.

Every program uses explicit semaphore dependencies. HAL queues are not FIFO,
and command-buffer reuse is limited to one in-flight issue per program object.
Persistent model and request allocations use indeterminate-lifetime queue
allocation; transient storage has an explicit alloca, execute, and dealloca
chain.

Compiler or kernel failures do not silently select another implementation.
Create a matched bug bead and local reproducer packet, select any bounded
workaround explicitly, and record which milestone claim the workaround cannot
authorize.

Python or other source-rewriting patchers are not sanctioned Loom integration
machinery. Runtime modules selectively link the canonical authored sources;
compiler failures are fixed at their owning boundary or isolated in a minimal
Loom function fork. Do not generate, rewrite, or intercept linked Loom source
as a workaround.

Any Loom file named `*_bringup_workaround.loom` is likewise a temporary,
minimal function fork rather than a new kernel variant. It carries only the
blocked function and declarations required to verify it, and is deleted rather
than evolved after the upstream fix.
