# Start here: a working handoff, not a framework specification

The first outcome is an observed execution, not a new serving abstraction. This
branch has one working Qwen adapter and reusable JIT/queue ownership code. A
fresh agent can establish that boundary without a model download, reproduce the
qualified Qwen run when the hardware fits, then implement a smaller model using
the same embedding path. No private working history is required.

An initial assignment can be this concrete:

> Establish the source-only JIT tests using this packet's README and record the
> actual target, build configuration, and passing case counts. Read the minimal
> JIT caller before the Qwen adapter. Then implement the pinned SmolLM2-135M
> candidate through one real-weight block and a retained greedy continuation,
> with an independent numerical reference. Keep the model-specific adapter
> small, reuse the queue/JIT ownership boundary, and expose unsupported compiler
> behavior with the smallest real reproducer. HTTP and performance expansion
> follow a correct retained continuation, not the other way around.

That is a new model implementation assignment. Changing Qwen's `--model` path
alone cannot load SmolLM2. [FIRST_PORT.md](FIRST_PORT.md) pins the candidate and
separates existing infrastructure from the model work still required.

## Establish a baseline before editing

The [README](README.md#fresh-machine-prerequisites) supplies bootstrap,
configuration, public model downloads, exact commands, and success output.
CPU tests establish parsing/scheduling; the small GPU tests establish actual
source compilation, VM imports, queue submission, and completion. Full Qwen
adds real checkpoint interpretation, retained state, MTP, and multi-turn HTTP.
Each result establishes that boundary only. A machine without the required
device can complete the CPU gate but cannot certify the GPU gate.

The recorded baseline consists of the checkout commit and dirty diff, compiler
and build flags, execution device/ROCr identity, commands and exit statuses,
test cases actually executed, and model revisions/hashes when applicable.
Skipped hardware tests remain unqualified. A cached test result represents the
same declared inputs; a fresh-device qualification records a real execution.

The authoring loop is source-driven. Editing model `.loom` files changes the
next runner process without rebuilding the C executable; editing the C adapter
requires building its exact target again. `sources.txt` lists source providers,
while `control.loom` and the cold `weights.loom` policy are loaded separately.
An added command/kernel library must be present in
that catalog; a generated HSACO or compiler subprocess is not part of this
deployment path.

## Read in this order

| Evidence | What to recover from it |
| --- | --- |
| [`testdata/jit/stage.loom`](../testdata/jit/stage.loom), [`control.loom`](../testdata/jit/control.loom), [`jit_test.cc`](../jit_test.cc) | One complete source/configuration/VM/GPU call and its cleanup, including compiler storage dying before command execution |
| [`jit.h`](../jit.h), [`jit.c`](../jit.c) | Public `loomc` embedding, source indexing, live device facts, native request ownership, reusable command recording |
| [`command.h`](../command.h), [`execution.h`](../execution.h), [`module.h`](../module.h), [`control_test.cc`](../control_test.cc) | Buffer borrowing/retention, exact queues and timelines, accepted work, feedback lifetime, drain after failure |
| [`program.h`](../program.h), model [`control.loom`](../models/qwen38/control.loom) | One shared source-JIT process with isolated and packed entries; indexed command selection, proposal/verify/catch-up routing, and bounded feedback without intermediate host waits |
| [`qwen_model.h`](../qwen_model.h), [`qwen_model.c`](../qwen_model.c), [`epoch.loom`](../models/qwen38/epoch.loom) | A concrete residency: weight placement, row origins, mutable state, scratch, descriptors, packed traversal and progress |
| [`weights.h`](../weights.h), [`weights.loom`](../models/qwen38/weights.loom), [`weights_test.cc`](../weights_test.cc) | Model-owned VM policy selecting cached preparers, actual file bytes into shared final storage, and transformation exactly once per unique tensor |
| [`qwen_schedule.h`](../qwen_schedule.h), [`qwen_service.c`](../qwen_service.c) | Ready spans versus model rows, canonical history, output credit, admission and scheduling policy |

The detailed [authoring](MODEL_AUTHORING.md) and [runner](RUNNER.md) guides
explain the invariants behind those callers. Canonical Loom documentation wins
over guessed MLIR syntax or remembered compiler behavior. The
[authoring corpus](../../../loom/src/loom/test/corpus/authoring/README.md) gives
small checked examples for individual numerical mechanisms.

## The engineering loop this template is meant to transfer

One live experiment record carries the numerical/storage contract, observed
baseline, one proposed mechanism, expected evidence, and the next decision.
Measurements and code are evidence; an earlier explanation is a hypothesis
until its callers or results agree. A contradiction changes the record rather
than gaining a workaround beside it.

A new model first crosses the hard boundary: public checkpoint bytes through
the actual loader, JIT, command, and retained GPU state into independently
checked outputs. Matching isolated arithmetic is not enough if a second token
or interleaved row corrupts state. A convincing sentence is not an oracle.
The port expands after one real block and its cache updates agree, then after
the whole model's retained continuation agrees.

Performance starts from a controlled workload and a falsifiable bottleneck.
Compiler reports explain register pressure, spills, waits and memory access;
completed device work establishes speed. Changing one mechanism at a time
keeps regressions interpretable. A narrow kernel win returns to the retained
multi-agent workload before it becomes a serving claim. The
[performance guide](PERFORMANCE.md) supplies controls and measurement details.

The code is an experimentation platform. A model-specific second adapter earns
an abstraction by exposing a genuinely shared contract, not by anticipating
every modality. One compiler/JIT residency, shared weights and code, data rows
instead of per-session VMs, explicit dependency edges, and no steady-state
device allocation are the intended invariants. HAL and the command-program ABI
are not extension slots for model bookkeeping. The current host epoch wait and
fixed row/shape bounds are explicit in the runner guide, not hidden behind a
promise of a device-owned loop.

Small coherent commits keep a working boundary available while the next one is
investigated. A continuation note records what actually passed, which mechanism
failed, exact reproduction inputs, and the next concrete action. Completed
work stays reproducible from tracked sources; large generated artifacts and
downloaded weights remain separate from source history.

## Failure triage by boundary

| Observation | First discriminating evidence |
| --- | --- |
| Setup/configuration fails | Python 3.12, managed tool diagnostics, [BUILDING.md](../../../BUILDING.md); build-service failures belong to that service's repair path |
| Target is incompatible or tests skip | Enabled AMDGPU/VM compiler, HAL substrate and runtime driver, plus the actual test runner's GPU capability |
| `hsa_init`, loader, or device creation fails | ROCr dependency loading, device enumeration/permissions and available resources on the **execution** machine; model source has not run yet |
| JIT cannot resolve a symbol or configuration | Catalog entry, named root, `config.decl` constraints and supplied bindings; stderr identifies the failing stage |
| Target contract rejects the GPU | Authored target declaration versus live device profile; full Qwen contains gfx1151-specific entries |
| Tensor missing, wrong size, or bad output at the first layer | Exact checkpoint hash, tensor name/shape/orientation/encoding, tokenizer IDs and reference intermediate values |
| Single row works; retained or packed work fails | Absolute row origins/byte offsets, cache publication, inactive-row masking, input/output lifetime and explicit semaphore edges |
| A GPU fault or unstable output | Smallest retained reproducer, target/profile, shape and source; `--kernel_sanitizer=access` instruments device accesses separately from host `--config=asan` |
| HTTP accepts requests but the witness fails | Client exception and server `admit`/`epoch`/`complete` records; `ready` proves initialization, not correct retained history |

An unsupported operation or compiler failure is useful transfer evidence when
it comes with exact source, command/configuration, device and first divergence.
Relaxing the numerical check, substituting stale code, or moving orchestration
outside the JIT would lose the property being tested. If the next step needs a
different device, checkpoint, or architecture decision, that concrete missing
input is the handoff back to the human.
