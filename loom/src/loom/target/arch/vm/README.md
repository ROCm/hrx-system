# Loom VM Target

The Loom VM target lowers ordinary Loom programs to the portable Core VM ISA.
It is a normal ISA target alongside AMDGPU, SPIR-V, LLVM IR, and WebAssembly;
the VM does not own a parallel compiler pipeline or a private scheduler and
register allocator.

The compilation path is:

1. Target selection chooses the Core VM provider and establishes the target
   contracts used by common legalization.
2. Source operations lower to target Low through compact generated rule tables
   plus family-specific materialization where an operation needs more than an
   opcode mapping.
3. Common Low scheduling, liveness, register allocation, and spilling produce
   explicit machine operands and stack storage.
4. VM module selection records the exact functions, constants, globals,
   callable types, imports, exports, and reflection metadata required by the
   artifact.
5. The emitter serializes one segmented bytecode image in a single pass. It
   writes into 32 KiB byte-sequence blocks and patches reserved directory and
   function rows after their payloads are known; it performs no sizing pass,
   contiguous staging allocation, or grow-and-copy loop.
6. A kernel-product build may emit the same specialized function version both
   as a target-native entry and as a VM launch-configuration function. Emitter
   projections join those artifacts once by stable compiler function version,
   and the immutable product publishes only artifact-local ordinals to runtime
   consumers.

The target consumes the authoritative ISA declarations from
[`runtime/src/iree/vm/bytecode/spec/`](../../../../../../runtime/src/iree/vm/bytecode/spec/).
Python generation produces purpose-built compiler tables from that source;
Loom does not maintain a second handwritten opcode catalog. The generated
tables remain build outputs. Checked-in wire headers belong to the bytecode
package because runtime-only builds must compile without running Python.

## Code map

- [`provider.c`](provider.c) registers the Core target provider and its target
  pipeline.
- [`legalization.c`](legalization.c) installs generated source and Low
  legalization rules.
- [`lower/`](lower/) owns source-to-Low matching, planning, value mapping, and
  family-specific emission. Files are split by invariant cluster rather than
  by generated opcode.
- [`abi/`](abi/) materializes source-ordered callable signatures into VM
  register, stack, and overflow carriers.
- [`constants/`](constants/), [`contracts/`](contracts/), and
  [`records/`](records/) materialize artifact-wide state established during
  lowering.
- [`launch_config_program.c`](launch_config_program.c) extracts the pure host
  regions of requested kernels into one shared VM companion artifact.
- [`../../emit/vm/`](../../emit/vm/) selects and serializes final VM modules.
- [`../../../../binding/c/target/vm/`](../../../../binding/c/target/vm/) loads
  launch companions and exposes the allocation-free launch invocation API.

## Test boundaries

Shared `source_low` template tests prove that target-neutral Loom programs lower
through this target. VM-specific `.loom-test` files cover ABI and artifact
contracts that are not shared with another target. Runtime instruction tests
exercise individual Core records without inventing a second Loom encoder, and
compiler-to-runtime execution tests load actual Loom-emitted modules. The
kernel-product HAL tests are the production boundary for coupled native and VM
launch artifacts.

The optional HAL ISA page, faithful multi-workitem kernel simulation, command
program execution, and foreign-language wrapper generation are independent
extensions. None is required to compile and execute Core host programs or
kernel launch configurations.
