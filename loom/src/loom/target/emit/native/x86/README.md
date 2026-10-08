# Scalar x86 native functions and task kernels

`loom-compile` can compile ordinary High functions into an ELF relocatable
object for an independent C or C++ linker input:

```sh
loom-compile functions.loom --target=x86:scalar --output=functions.o
cc caller.c functions.o -o caller
```

For example, this function exports the C signature
`uint64_t mix(uint64_t input, uint64_t delta)`:

```loom
func.def public @mix(%input: i64, %delta: i64) -> (i64) {
  %sum = scalar.addi %input, %delta : i64
  %result = scalar.xori %sum, %input : i64
  func.return %result : i64
}
```

The target selects SysV AMD64 by default, independently of the compiler host.
The explicit function spelling is
`abi(object_function, {calling_convention = "sysv"})`. The ELF product accepts
scalar arguments and one result: `i32`, `i64`, and 64-bit buffer addresses,
indices, or offsets. The first six arguments use registers; remaining arguments
use the caller's stack. Unused arguments retain their ABI positions. Private
functions, recursion, and direct calls to external C symbols remain outlined.
Unsupported calling conventions or payload classes are diagnosed before an
object is returned.

Module-internal definitions may return several flattened values because every
caller and callee is emitted together. Integer-class results use RAX and RDX;
SSE-class results independently use vector registers 0 and 1. Additional
results use a compact caller-owned block after the complete stack-argument
area. A callee retains that block across nested calls, while allocation plans
all register results as one parallel exit. Public and imported functions remain
limited to one result until their source aggregate layout supplies a real
platform ABI instead of an ambiguous flat value list.

The shared allocator preserves values across call clobbers, inserts spill
storage, and resolves register permutations. When a signature exceeds the
register bank, proven entry stores and pre-call reloads become direct transfers
between ABI locations and invocation storage. A shared plan retains their
locations and memory ordering, so emission consumes the completed decisions.
Ordinary instructions continue to receive allocated registers. Synchronous
native and VM consumers use this transport contract; asynchronous device
storage keeps its explicit scheduled instructions.

Authored Low uses the same contract. When a logical boundary is not determined
by its physical register carriers, `abi_layout({signature = (...) -> (...)})`
retains that distinction across serialization. A Boolean in GPR32 consequently
cannot be admitted as an ordinary C `uint32_t` parameter by accident.

The architecture provider owns target identity and selects the object emitter.
ABI classification supplies entry argument locations and call clobbers, and
reserves RSP before the shared scheduler and allocator run. These constrain
boundary transfers without pinning the argument's entire SSA lifetime. Native
instruction preparation consumes their final packets, edge transport, and CFG
block ordinals, retaining actual GPR writes, including implicit writes and
allocation-generated moves. This
determines the minimal callee-preserved register envelope before byte encoding.
All returns reach its common restore sequence.

The encoder consumes concrete operands and generated opcode facts. It has no
IR, allocation, or calling-convention decisions to recover. The native object
writer joins code contributions and owns ELF tables. Only emitted bytes survive
between functions; scheduling and allocation scratch is reclaimed immediately.
The returned artifact is independent of all compiler storage.

`test/function.loom-test` checks which registers require preservation, including
writes introduced by allocation transport. Focused writer tests verify save and
restore order and branch destinations from prepared native instructions;
instruction encoding has its own byte-level tests. `callable_link_test` links a
Loom-generated object into a normal host test with independent arithmetic and
memory references; the compiler exits before linking begins.

## Task HAL kernels

A source kernel can also become a task HAL executable library:

```sh
loom-compile fill.loom --target=x86:scalar --output=fill.so
```

```loom
kernel.def export("fill") @fill(%groups: index) {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%output: buffer, %value: i32) {
  %x = kernel.workgroup.id<x> : index
  %count = kernel.workgroup.count<x> : index
  %buffer_base = index.constant 0 : offset
  %view = buffer.view %output[%buffer_base] : buffer -> view<[%count]xi32>
  view.store %value, %view[%x] : i32, view<[%count]xi32>
  kernel.return
}
```

The command-line target is optional when the source binds its kernels to an
explicit target, such as `x86.target<scalar> @cpu` and
`kernel.def target(@cpu)`. Both forms use the same task ABI and emitter.

Kernel roots select the task library format automatically; ordinary function
roots select a relocatable object. `--format=x86-hal` also selects the library
explicitly. Each workgroup invokes the body once. Workgroup size and subgroup
size are one; explicit loops within the body own additional work. Global
invocation coordinates equal workgroup coordinates. XYZ IDs and counts come
from the dispatch's state. The body can call ordinary scalar helpers and use
aligned invocation-private stack storage across those calls.

The retained logical signature describes binding ordinals and byte offsets in
the dispatch constant segment. Bindings are dense in declaration order, while
constants align to their scalar element width, capped at eight bytes. Unused
parameters retain their public positions. The physical entry receives three
state pointers, imports its parameters, and returns an explicit success value.
These operations are ordinary Low instructions before allocation; the native
encoder receives the same prepared form as an ordinary function.

The task contract and its lowering have shared compiler owners:

| Component | Responsibility |
| --- | --- |
| `target/abi/task` | Logical parameter layout, reflection records, task builtins, and invocation-state fields. |
| `codegen/low/lower/task_abi` | Select and import invocation-constant source queries through the target's type mapper. |
| `codegen/low/transforms/task_abi` | Parameter accesses, predicates, entry/return rewriting, queries, and concrete function-version lifetime. |
| `target/emit/native/module` and `task_library` | Symbol admission, native contributions, library metadata, artifact fixups, and detached output ownership. |
| x86 materialization and native emission | Physical carriers, instruction sequences, SysV calls and frames, byte encoding, and relocation facts. |

The task entry interface describes semantic actions, such as importing a
parameter or a workgroup ID. Targets can realize each action with several
instructions. Native module assembly calls the target once per function and
reclaims its scheduling and allocation scratch after retaining output bytes
and fixups. The implemented task data model and native artifact writers are
64-bit little-endian; these shared components do not imply an instruction
encoder or runtime execution support for another architecture.

The compiler also emits the versioned library query as an ordinary Low function.
Its address reference to the readonly library object becomes a native relocation.
The artifact contains code, reflection records, and internal pointer fixups; it
contains no pointers into compiler memory. Public `iree_hal_executable_load`
loads the artifact on a compatible task device, and the queue dispatch APIs
consume its reflected interface. Input artifact bytes can be released once
loading completes.

The shared `target/abi/task/dispatch_test.cc` consumer exercises that public
boundary with an explicitly targeted grid kernel and an unbound byte kernel in
the same library: the compiler exits before loading, artifact bytes are released before dispatch,
and two semaphore-ordered 3D grids update a nonzero-offset binding. The kernel
passes aligned private storage to a retained helper. Exact buffer contents and
guard words check the invocation, call, storage, and binding contracts together.
Another export checks byte-sized constants and a zero-work dispatch. The x86
fixtures supply CLI and C API artifacts to the same consumer. The public C
embedding example produces the latter library:

```sh
iree-bazel-run //loom/binding/c/example:compile_artifact -- fill.loom x86:scalar fill.so
```

That example creates a configured target environment, selects `x86:scalar`,
and calls `loomc_compile_artifact` with inferred roots and the target's default
pipeline. It releases the compiler, module, and workspace before writing the
result-owned executable bytes. Both entry points use the core x86 provider;
neither owns ABI policy or loads a runtime while compiling.

The shared scenario runner selects its native profile from CPU facts published
by the task device. For example, on an x86-64 task device:

```sh
iree-test-loom loom/src/loom/test/corpus/control/loop/scalar_state.loom --device=task
```

The runner compiles and executes the subject through the public task loader and compares
its runtime observations against the independent VM oracle. The same HAL
candidate, buffer staging, submission, and comparison path serves GPU targets.
The task adapter only projects device capabilities and selects an eligible core
emitter; it owns no instruction lowering or ABI materialization. The scalar
native profile requires no optional CPU features. SIMD profiles are available
for source-to-Low compilation, but native execution selection requires their
vector ABI transport and instruction encoding before admitting them.
