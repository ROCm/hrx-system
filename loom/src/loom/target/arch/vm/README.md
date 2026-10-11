# VM target

The VM target compiles ordinary Loom functions into standalone `.vm` modules.
It uses the same legalization, control-flow lowering, scheduling, and register
allocation infrastructure as native targets. The VM package supplies the
machine description and builds a physical VM program plan; the target writer
serializes that plan. There is no separate compiler pipeline.

## Compile a function

```mlir
func.def public @arithmetic(%a: i32, %b: i32, %c: i32) -> (i32) {
  %sum = scalar.addi %a, %b : i32
  %scaled = scalar.muli %sum, %c : i32
  func.return %scaled : i32
}
```

With a VM-enabled compiler, save this as `arithmetic.loom` and run:

```sh
loom-compile arithmetic.loom --target=vm:core --output=arithmetic.vm
vm-dis arithmetic.vm
```

`--target=vm:core` selects the VM profile and its default `.vm` output format.
The source needs no target declaration or attributes. The shared compiler
specializes public functions and their callees for the selected profile, which
supplies the calling convention during lowering. A public function exports its
symbol name unless an explicit export name overrides it. Internal callees are
compiled into the same module without becoming host-callable exports.

The output contains the instruction stream, logical VM signatures, sorted
callable and export tables, and referenced read-only data. It is independent of
the source module and compiler lifetime. A host loads it through the bytecode
module API, links a program, creates a process, and invokes an export. The
[runtime guide](../../../../../../runtime/src/iree/vm/README.md) explains those
objects, argument/result ownership, and reusable invocation storage.

## From source to bytes

Source operations first use the shared target specialization and legalization
pipeline. Supported scalar operations select VM descriptors; vector bodies can
use shared scalarization where their operations and control-flow boundaries
permit it. Source functions then become `low.func.def target<vm.core>` with
typed value and reference registers. Low assembly uses the VM mnemonics, while
retaining compiler constructs such as SSA values and block arguments.

The common frame builder schedules those Low operations, assigns registers and
local storage, and plans edge copies and spills. The target-owned
[function planner](function_plan.c) consumes each frame once to produce final
instruction packets, direct calls, transfers, and branches. Branch
displacements are resolved before the physical program is published. Program
planning does not rerun whole-module verification or reconstruct a second
allocation plan.

The [program plan builder](program_build.c) collects VM functions from the
target-low module and assigns their final table ordinals. It
consumes shared function-version information instead of performing its own
callgraph specialization, and retains only referenced read-only payloads. The
result contains final wire rows and immutable function bytecode, with no IR,
analysis, or diagnostic surface. The target-owned
[binary writer](../../emit/vm/module_binary.c) lays out that physical program;
it cannot reach back into compiler state or reject a compiler invariant. The
returned byte sequence retains the planned instruction storage between its
writer-owned prefix and suffix instead of copying function bytes. Consumers can
enumerate that segmented storage or explicitly clone it when they need
contiguous bytes.

## Where the contracts live

| Area | Owner |
| --- | --- |
| Opcodes, packet fields, selector semantics, and wire records | [Runtime ISA specification](../../../../../../runtime/src/iree/vm/bytecode/spec/) |
| Projection into shared Low descriptors | [descriptors.py](../../../../../py/loom/target/arch/vm/descriptors.py) |
| Source-operation correspondence and selection constraints | [contracts.py](../../../../../py/loom/target/arch/vm/contracts.py) |
| Target registration and profile selection | [provider.c](provider.c) |
| Type mapping and symbolic read-only data lowering | [lower.c](lower.c) |
| Selection of native math forms or shared recipes | [math_policy.c](math_policy.c) |
| Physical program planning | [function_plan.h](function_plan.h), [program_build.h](program_build.h) |
| VM image layout | [program.h](program.h), [module_binary.h](../../emit/vm/module_binary.h) |

The Python projection imports the runtime specification. It maps the existing
ISA into Loom's descriptor and constraint schemas instead of maintaining a
parallel set of opcode numbers or packet layouts. Generated contract indices
and descriptor tables are build outputs, consumed as immutable data by the
shared lowering machinery and physical program planning. The binary writer
only consumes the resulting target-owned plan.

## Supported source boundary

This compiler surface supports synchronous ordinary functions with scalar,
buffer, and declared managed reference arguments/results, direct internal calls,
reference selection, branches and loops, local storage, typed memory operations,
and referenced read-only data. Scalars include narrow integer and floating
formats; `index` and `offset` use the
profile's 64-bit carrier. Internal vector programs rely on shared legalization,
not a public vector calling convention. Supported math modes are selected by
the target math policy rather than by substituting a host platform's libm.

Runtime ISA availability is distinct from source-lowering support. This target
does not provide kernel/workgroup execution, HAL command programs, indirect
calls, process-global mutation, or suspension.
Unsupported source representations and instructions fail compilation; the
emitter does not fall back to another execution engine. Kernel launch-config
evaluation is a separate integration boundary, not part of compiling an
ordinary `.vm` module.

Managed reference types declare an explicit provider namespace and type name.
For example, `hal.buffer` and `hal.buffer_view` preserve their native HAL
identities through calls and control flow; they are distinct from the Core
`buffer` type used for CPU byte storage. A host registers the corresponding
native type provider before loading the image. Unknown opaque types fail
compilation instead of being treated as untyped pointers.

The optional [HAL reference provider](../../../../../../runtime/src/iree/module/hal/README.md)
supplies canonical descriptors and typed host adapters for the two HAL types.
Their VM references share the original HAL owner count, including buffer
recycling and view destruction. Neither the compiler nor the generic VM links
this provider automatically; the embedding composes the providers its modules
use.

Low reference registers retain their source types. The module planner binds
each distinct type declaration once and emits a sorted namespace/type table;
signatures compare both field kinds and reference identities. Runtime image
construction resolves those keys to the host's canonical descriptors. Calls
then move ordinary VM references without name lookup or wrapper allocation.

## Native imports and captured source

A runtime import states the providing module's namespace independently of the
local symbol. The VM linker resolves it against the native module supplied by
the embedding:

```loom
func.decl import("diagnostics") @observe(%site: buffer)

func.def public @report() {
  %site = func.location [#func.location.file<"example.cc", range = [12, 3, 12, 28]>] : buffer
  func.call @observe(%site) : (buffer)
  func.return
}
```

The import uses the ordinary scalar/reference calling convention, including
overflow arguments and results. Returned references transfer their ownership
through the call ABI; a native failure unwinds reference arguments normally.
An explicit import alias is needed only when its external symbol differs from
the local symbol.

### Structured synchronous native entries

A Low import with `abi(vm_native)` names a synchronous provider entry rather
than an ordinary VM callable. Its `abi_layout` states a static aggregate record
and the positions occupied by its logical argument and result selectors:

```loom
low.func.decl import(native, "sum", module = "host") target<vm.core> abi(vm_native) abi_layout({arguments = [3, 4], record = [2, 3, 0, 0, 0], results = [2], version = 0}) @sum(%a: reg<vm.value : i64>, %b: reg<vm.value : i64>) -> (reg<vm.value : i64>)
```

Here the provider's record has a two-word range header, a result selector, and
two input selectors in the range payload. `record` contains u32 literal words
and zero selector placeholders. `arguments` and `results` index those words in
logical signature order; each placeholder is distinct. `version` selects the
exact provider schema. This layout is a public calling convention, independent
of physical register assignment. Source dialect legalization can construct it
from static operand segments without introducing a provider export per shape.

The shared scheduler and allocator retain the normal SSA lifetimes, with no
fixed ABI-register prefixes or native clobbers. Final emission patches each
placeholder from those allocation facts, including direct selectors for spilled
values and references. The module carries `control.call.native` instructions
and immutable records, not ordinary import signatures or argument packets.

At program linking, the provider schema verifies range layout, selector bounds,
and result distinctness, and computes invocation scratch. Execution borrows the
caller frame directly. Entries cannot suspend or reenter the VM; all input use
precedes result publication, so dying inputs and new outputs may share storage.
Native asynchronous APIs retain their own work-lifetime contracts. The VM
neither tracks nor waits for their completion. The
[native call contract](../../../../../../runtime/src/iree/vm/native_call.h)
defines the provider and ownership boundary.

`func.location` materializes an immutable source value as a normal VM buffer.
Its semantic nodes retain file ranges, source field ranges, optional original
text, and fused/tagged/opaque provenance even when debug locations are stripped.
Compiler clients can [capture these nodes from admitted source](../../../tooling/input/README.md)
before source storage expires. Native consumers decode the versioned
[`LLOC` span](../../../format/location.h) and retain the buffer when observations
must outlive the invocation or executable. Runtime decoding needs no compiler
tables or source files.

## Testing

[test/](test/) contains Low assembly, emission, and source-lowering checks.
Portable lowering inputs come from the shared `source_low` TEMPLATE corpus;
VM expectations specify the target's selected representation.

The [semantic correctness corpus](../../../test/corpus/) contains target-neutral
programs and their `check.case` inputs and expectations. The target-owned VM
suite links and executes each source independently through `iree-test-loom`,
using the [VM testbench](../../../tooling/target/vm/testbench.h). The testbench
compiles one module for all cases in that source, releases the compiler copy and
scratch before loading the emitted bytes, and reuses a runtime process and
invocation storage. This exercises artifact ownership, dynamic execution, and
returned buffer aliases rather than only comparing disassembly.

The [tool integration suite](../../../tooling/target/vm/test/vm.test.json) compiles
source with `loom-compile` and reads the resulting file with `vm-dis` in a
separate process. Runtime instruction and malformed-image tests remain under
`runtime/src/iree/vm/`; they do not require the compiler.
