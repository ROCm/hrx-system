# Portable Command Programs

Command programs compile reusable, command-buffer-shaped work from the same
Loom source that defines the kernels they launch. They preserve model-level
structure long enough for Loom to specialize control flow, share kernel
dependencies, derive launch counts, plan transient storage, and encode explicit
concurrency. The result is a target-neutral command artifact accompanied by
independently compilable kernel units and a pure host launch-count function.

The command compiler is independent of HAL. A materializer maps the portable
artifact to its command system, such as a HAL command buffer, a CUDA graph, or
a native queue packet sequence.

Four boundaries keep the feature composable:

- Source command programs remain open Loom IR. Normal linking,
  specialization, and canonicalization apply before portability is required.
- The prepared plan is the sole owner of root, launch-count, dependency, and
  storage decisions. Later stages consume those indexed results instead of
  reconstructing them from source IR.
- Command roots, launch-count functions, and kernel units are peer compilation
  products. No product kind is implicitly primary or nested inside another.
- The serialized artifact contains logical resources and typed arguments, not
  HAL objects, process addresses, native calling conventions, or target code.

## Source Model

`command.program.def` divides its signature into two roles:

- Specialization arguments precede `launch`. They are program-invocation
  values whose facts participate in staged specialization. Uses that control
  program topology, allocation, or executable selection must resolve before
  materialization. Residual values used only by launch arithmetic remain
  inputs to the aggregate launch-count function, so they can change without
  rerecording the command program.
- Buffer bindings follow `launch`. They occupy stable slots whose concrete
  buffers may be supplied each time the materialized program is issued.

For example, this program specializes over a bounded token count, resolves one
immutable parameter view, and records three projections that may execute
concurrently after the prepare dispatch completes:

```loom
kernel.def @prepare(%token_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%token_count, %one, %one)
      workgroup_size(%one, %one, %one) : index
} launch(%input: buffer, %scratch: buffer) {
  kernel.return
}

kernel.def @project(%token_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%token_count, %one, %one)
      workgroup_size(%one, %one, %one) : index
} launch(%weight: view<4096x4096xbf16, #dense>,
         %scratch: buffer, %output: buffer) {
  kernel.return
}

command.program.def public @attention(%token_count: index) launch(
    %parameters: buffer,
    %input: buffer,
    %query: buffer,
    %key: buffer,
    %value: buffer) where [range(%token_count, 1, 1024)] {
  %weight = command.parameter %parameters, "attn.weight"
      : view<4096x4096xbf16, #dense>
  %scratch_bytes = index.constant 32768 : offset
  %scratch = buffer.alloca<global> align(256) %scratch_bytes : buffer
  kernel.launch @prepare[%token_count](%input, %scratch)
      : [index](buffer, buffer)
  command.concurrent {
    kernel.launch @project[%token_count](%weight, %scratch, %query)
        : [index](view<4096x4096xbf16, #dense>, buffer, buffer)
    kernel.launch @project[%token_count](%weight, %scratch, %key)
        : [index](view<4096x4096xbf16, #dense>, buffer, buffer)
    kernel.launch @project[%token_count](%weight, %scratch, %value)
        : [index](view<4096x4096xbf16, #dense>, buffer, buffer)
  }
  command.return
}
```

Ordinary function calls, program calls, loops, conditionals, templates, and
value-fact refinement remain source constructs. Preparation specializes them
until the command schedule is a closed sequence of kernel launches and explicit
`command.serial` or `command.concurrent` regions. A residual construct without
a portable command meaning fails at that boundary instead of acquiring an
implicit runtime interpretation.

`command.parameter` names immutable content within an explicit source buffer.
It returns a typed logical view and contributes a placement requirement; it
does not allocate, load, transfer, or synchronize the content.

## Compilation Products

Preparation accepts one or more selected program roots and constructs one
owned plan:

```text
selected command roots
          |
          v
 link their union dependency closure
          |
          v
 compose programs and specialize root structure
          |
          +-------------------------+--------------------------+
          |                         |                          |
          v                         v                          v
 portable Low roots       aggregate launch functions   shared kernel units
          |                                                    |
          v                                                    v
 command artifacts                              independent target lowering
```

The plan products are peers:

- Each selected root owns a symbol-closed `cmd.core` Low function and can be
  serialized into one portable command artifact.
- Each root has one pure host function returning its unique residual XYZ
  launch-count tuples.
- The plan owns a deduplicated set of selectively linked kernel units. Each
  root references the units it uses through dense dependency indices.

Kernel launch sites share a unit when their linked kernel identity and
boundary-projected scalar facts match. Prefill, decode, and other roots can
therefore share compiled kernels without coupling their command artifacts.
Each unit remains an ordinary target-compilation input and can be lowered or
retrieved from a caller-owned cache independently.

Source preparation requires every scheduled `kernel.launch` to resolve to a
`kernel.def`: launch-configuration extraction and dependency-unit
specialization both consume that definition. A bodyless `kernel.decl` is
diagnosed at this boundary instead of acquiring guessed launch semantics. The
artifact ABI remains agnostic to executable provenance, so a materializer can
supply any executable whose entry reflection satisfies the recorded logical
schema.

The source module may be released after preparation. The plan owns every
module, parameter key, requirement table, and dependency mapping reachable
from its roots.

## Launch Counts

The compiler represents each linked kernel configuration as a private pure
helper in the plan's shared launch module. Scheduled launches become ordinary
pure calls from one aggregate function per root. Common subexpression
elimination merges calls with the same helper and ordered workload values
before the surviving calls are inlined, so repeated dispatches do not expand
the same launch arithmetic. Canonicalization and common subexpression
elimination then simplify the aggregate function, and equal residual dynamic
XYZ tuples produce one result slot regardless of how many dispatches use them.

Launch counts have two portable placements:

- Exact counts are embedded in a direct dispatch command.
- Counts derived from residual specialization arguments are returned by the
  aggregate host function on each program invocation. The caller stores the
  tuples in one rebindable buffer, and dispatches reference them through static
  indirect commands. Updating this table does not rerecord the command program.

The Low ISA and artifact format also represent dynamic indirect dispatches,
whose count tuple is produced in buffer storage while the command program is
executing.

## Storage

Command-program buffers have two materialization roles:

- A fixed root retains the same concrete buffer while the materialized program
  lives. Immutable parameter roots use this role.
- A rebindable root is a stable issue-time binding slot. Inputs, outputs,
  transient storage, and host launch-count storage use this role.

Every operational range is a root index, byte offset, and byte length. A length
of `UINT64_MAX` denotes the remaining root range. This keeps command artifacts
independent of process addresses and lets one recorded program run against
different issue-time buffers.

`buffer.alloca<global>` roots are packed into one rebindable transient slab.
The planner derives live intervals from command waves, preserves requested
alignment, and aliases ranges whose lifetimes do not overlap. Allocations
defined or used in the same concurrent wave remain distinct.

Parameters are packed independently within each fixed source root. The
artifact reports every substituted key and concrete placement along with each
root's required length and alignment; parameter loading remains owned by the
embedding application.

## Scheduling

The source schedule describes dependency intent:

- Lexical commands and children of `command.serial` are ordered.
- Siblings in `command.concurrent` have no dependency edges between them and
  join before later work.

Preparation flattens that structure into contiguous waves. Commands within one
wave retain source traversal order but may execute concurrently. Successive
waves are separated by a full execution barrier. When a barrier immediately
precedes a fill, copy, or dispatch, the Low ISA and artifact encode a dedicated
barrier form of that command so the common join-and-record sequence remains one
portable instruction.

## Portable Low ISA

Prepared roots use `low.func.def target<cmd.core> abi(command_program)`. Their
zero-argument function signature is intentional: external resources enter
through dense ABI tables declared by `abi_layout`.

```loom
low.func.def target<cmd.core> abi(command_program)
    abi_layout({entry_count = 1, executable_count = 1,
                fixed_buffer_count = 1, rebindable_binding_count = 2})
    @project() asm {
  %parameters = resource<command_input> {index = 0, source_type = buffer}
      : reg<cmd.buffer>
  %input = resource<command_input> {index = 0, source_type = buffer}
      : reg<cmd.binding>
  %output = resource<command_input> {index = 1, source_type = buffer}
      : reg<cmd.binding>
  %executable = resource<command_input> {index = 0, source_type = index}
      : reg<cmd.executable>
  %entry = resource<command_input> {index = 0, source_type = index}
      : reg<cmd.entry>
  %zero = cmd.constant.u64 0
  %remaining = cmd.constant.u64 -1
  %one = cmd.constant.u32 1
  cmd.dispatch.direct<%executable, %entry>[%one, %one, %one](%parameters, %zero, %remaining, %input, %zero, %remaining, %output, %zero, %remaining)
  return
}
```

Executable and entry values are resources rather than symbol references. This
separates the command artifact from target binaries and permits an embedding
application to supply Loom-produced or external executables through the same
table ABI.

Dispatch arguments form a logical typed payload. Buffer arguments are encoded
as fixed or rebindable root ranges; scalar arguments use exact `b8`, `b16`,
`b32`, or `b64` bits. The artifact does not choose native argument offsets,
padding, or calling convention. A materializer combines each logical entry
schema with executable reflection for its command system.

## Artifact Boundary

Serialization is the closed portability boundary. It accepts a prepared root
only when every remaining Low descriptor and operation has a command encoding,
then writes canonical little-endian tables with no compiler pointers, symbol
references, or borrowed strings.

`loom_cmd_program_parse` is the external byte boundary. It validates the
canonical table layout and cross-table indices once without allocating, then
returns an immutable view borrowing the supplied bytes. Typed accessors and the
barrier-wave iterator consume that verified view without rescanning source IR.

The artifact records:

- fixed buffers, rebindable bindings, executables, and entry requirements;
- operational buffer ranges and logical entry argument schemas;
- tagless dispatch argument bytes and ordered command records;
- parameter roots, keys, and concrete placements;
- transient slab and host launch-count table requirements.

## Ownership Map

| Object | Owns | Borrows |
| --- | --- | --- |
| Source module | Authored IR | Nothing |
| Prepared plan | Root and launch modules, kernel units, root tables, parameter keys | Compiler context and arena block pool |
| Serialized bytes | Complete portable program | Nothing |
| Parsed program view | Nothing | Serialized bytes |

`loom_cmd_program_plan_deinitialize` releases the complete prepared plan.
Serialized bytes are caller-owned and may outlive that plan. A parsed view is
valid only while its source byte span remains live.

## Implementation Map

| Path | Contract |
| --- | --- |
| `loom/py/loom/dialect/command/defs.py` | Source operations and canonical text syntax |
| `loom/src/loom/ops/command/verify.c` | Source-level program, launch, and parameter verification |
| `loom/src/loom/target/arch/cmd/lower/program_plan.*` | Multi-root ownership and product preparation |
| `loom/src/loom/target/arch/cmd/lower/launch_graph.*` | Aggregate launch-count factoring |
| `loom/src/loom/target/arch/cmd/lower/schedule.*` | Structured schedules and execution waves |
| `loom/src/loom/target/arch/cmd/lower/parameters.*` | Immutable parameter enumeration and placement |
| `loom/src/loom/target/arch/cmd/lower/transients.*` | Transient live ranges and slab packing |
| `loom/py/loom/target/arch/cmd/descriptors.py` | Portable `cmd.core` Low ISA |
| `loom/src/loom/target/arch/cmd/lower/serialize.*` | Closed Low-to-artifact serialization |
| `loom/src/loom/target/arch/cmd/format.*` | Canonical binary layout |
| `loom/src/loom/target/arch/cmd/program.*` | Parsed artifact view and typed accessors |

Production-backed `loom-check` emitters expose prepared root programs,
aggregate launch functions, and independently compilable kernel units in text.
The `.loom-test` corpus beside this implementation uses those emitters, so its
expected IR exercises the same preparation and serialization path as an
embedding application.
