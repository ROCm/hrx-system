# Portable Command Programs

Command programs compile reusable, command-buffer-shaped work from the same
Loom source that defines the kernels they launch. They preserve model-level
structure long enough for Loom to specialize control flow, share kernel
dependencies, derive launch counts, plan transient storage, and encode explicit
concurrency. The result is a target-neutral command artifact plus explicit
executable-entry requirements. Kernel implementations remain independent
compilation products and are never opened by command planning.

The command compiler is independent of HAL. A materializer maps the portable
artifact to its command system, such as a HAL command buffer, a CUDA graph, or
a native queue packet sequence.

Four boundaries keep the feature composable:

- Source command programs remain open Loom IR. Normal linking,
  specialization, and canonicalization apply before portability is required.
- The command plan is the sole owner of root, dispatch-count, entry, and
  storage decisions. Later stages consume those indexed results instead of
  reconstructing them from source IR.
- Logical launch configuration remains ordinary deterministic computation in
  the caller until generic inlining, canonicalization, CSE, and folding have
  run. It may load explicit workload buffers or views without opening a device
  implementation.
- The serialized artifact contains logical resources and typed arguments, not
  HAL objects, process addresses, native calling conventions, or target code.
- Optional kernel request publication transfers independently owned ordinary
  Loom modules to its recipient. The plan retains only their atomic entry
  requirements, not source products or classification state.

## Source Model

`command.def` divides its signature into two roles:

- Specialization arguments precede `launch`. They are program-invocation
  values whose facts participate in staged specialization. Uses that control
  program topology, allocation, or executable selection must resolve before
  materialization. Direct dispatch counts and scalar device-ABI arguments must
  likewise become exact before portable command lowering; values intended to
  remain dynamic use explicit buffer or view storage.
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

command.def public @attention(%token_count: index) launch(
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
value-fact refinement remain source constructs. Planning specializes them
until the command schedule is a closed sequence of kernel launches and explicit
`command.serial` or `command.concurrent` regions. A residual construct without
a portable command meaning fails at that boundary instead of acquiring an
implicit runtime interpretation.

`command.parameter` names immutable content within an explicit source buffer.
It returns a typed logical view and contributes a placement requirement; it
does not allocate, load, transfer, or synchronize the content.

## Compilation Products

Planning accepts one or more selected program roots and constructs one
owned plan:

```text
selected command implementation facets
                 |
                 v
 selectively link command code, kernel contracts,
 and pure launch-configuration facets
                 |
                 v
 replace launches in caller CFG with ordinary calls + dispatches
                 |
                 v
 generic specialization, inlining, CSE, and folding
                 |
                 v
 schedule configured dispatches and lower portable Low roots
                 |
                 v
 command artifacts + executable-entry requirements
                 |
                 +---- optional independent kernel source requests
```

The plan owns:

- Each selected root owns a symbol-closed `cmd.core` Low function and can be
  serialized into one portable command artifact.
- A plan-wide table contains each distinct configured executable-entry
  declaration. Roots reference that table through dense local slots so an
  executable object and its entry token are resolved atomically.
- Parameter, transient, and physical dispatch-count placement is retained as
  plan data consumed directly by Low conversion and serialization.

Command planning requires every scheduled `kernel.launch` to resolve to a
selected launch-configuration facet. Selective materialization reconstructs a
private `kernel.decl` and a pure or deterministic read-only configuration
function without decoding the kernel implementation body. A bodyless
`kernel.decl` with no configuration is diagnosed instead of acquiring guessed
launch semantics. An authored `kernel.entry.decl` can be dispatched directly
when the embedding application provides the matching executable entry.

The source providers and index may be released after planning. The plan owns
the shared lowered root module, parameter keys, requirement tables, and
root-local entry mappings. It owns no kernel source request: optional requests
transfer to their recipient during planning and remain provisional until the
parent operation succeeds.

## Launch Counts

Selective linking projects each required kernel launch configuration as an
ordinary private function carrying its proven pure or read-only contract.
Launch resolution replaces `kernel.launch` at its caller-owned CFG site with an
inline call and a `kernel.dispatch` that uses the call results. Pure calls may
cross launch-schedule wrappers while all workloads are defined outside them.
Read-only calls remain beside the lexical launch so they cannot move ahead of a
command that produces their storage. Ordinary control flow keeps both calls
branch- and loop-local.

The normal function pipeline owns canonicalization, CSE, unrolling, inlining,
and dead-symbol elimination. Equal calls can therefore collapse before command
scheduling, and exact results become ordinary dispatch facts without a second
launch evaluator or result-compaction pass.

Configured dispatch counts have three portable placements:

- Exact counts are embedded in a direct dispatch command.
- An exact aligned `view<3xi32>` rooted in immutable parameters becomes a
  static indirect dispatch: its contents stay unchanged throughout execution.
- An exact aligned `view<3xi32>` rooted in invocation buffers or transient
  command storage becomes a dynamic indirect dispatch. The device fetches its
  counts immediately before dispatch issue, after any preceding producer wave.
  Invocation buffers may already contain valid counts at program entry;
  transient storage requires a preceding execution wave.

Residual deterministic read-only count expressions become explicit generated
device producers writing transient XYZ tuples. Compatible consumers can share
a producer, while intervening possible writes separate their observations.
The serialized artifact contains those producers and indirect consumers; it
never delegates count evaluation to an implicit host callback.

The generated producer must preserve each captured load's observation. The
shared read-motion analysis checks that a load can reach its originating
consumer without crossing a memory or execution boundary. A scalar loaded
before an intervening write instead gets a count producer at its source
observation. If one count expression combines incompatible observation epochs,
retain those values in explicit device storage before consuming them together.
All count clusters are planned and validated before any source root is rewritten.

## Storage

Command-program buffers have two materialization roles:

- A fixed root retains the same concrete buffer while the materialized program
  lives. Immutable parameter roots use this role.
- A rebindable root is a stable issue-time binding slot. Inputs, outputs,
  transient storage, and explicit indirect-count views use this role.

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

Planning flattens that structure into contiguous waves. Commands within one
wave retain source traversal order but may execute concurrently. Successive
waves are separated by a full execution barrier. When a barrier immediately
precedes a fill, copy, or dispatch, the Low ISA and artifact encode a dedicated
barrier form of that command so the common join-and-record sequence remains one
portable instruction.

## Portable Low ISA

Planned roots use `low.func.def target<cmd.core> abi(command_program)`. Their
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
as fixed or rebindable root ranges. Fixed-width scalar arguments use exact
`b8`, `b16`, `b32`, or `b64` bits, while logical `index` and `offset` arguments
retain their distinct carrier semantics and exact canonical 64-bit values. The
artifact does not choose a target carrier width, native argument offset,
padding, or calling convention. A materializer must combine each logical entry
schema with executable reflection, validate representability, and repack
address scalars to the reflected width. A fixed-width scalar requires an exact
reflected width match.

## Artifact Boundary

Serialization is the closed portability boundary. It accepts a planned root
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
- exact dispatch argument bytes and ordered command records;
- parameter roots, keys, and concrete placements;
- transient slab and optional launch-count storage requirements.

## Ownership Map

| Object | Owns | Borrows |
| --- | --- | --- |
| Source module | Authored IR | Nothing |
| Command plan | Root module, entry requirements, root tables, parameter keys | Compiler context and arena block pool |
| Kernel request recipient | Independently compilable request modules | Nothing |
| Serialized bytes | Complete portable program | Nothing |
| Parsed program view | Nothing | Serialized bytes |

`loom_cmd_program_plan_deinitialize` releases the complete command plan.
Serialized bytes are caller-owned and may outlive that plan. A parsed view is
valid only while its source byte span remains live.

## Implementation Map

| Path | Contract |
| --- | --- |
| `loom/py/loom/dialect/command/defs.py` | Source operations and canonical text syntax |
| `loom/src/loom/ops/command/verify.c` | Source-level program, launch, and parameter verification |
| `loom/src/loom/transforms/kernel/resolve_launches.*` | Caller-owned launch-to-call-and-dispatch rewriting |
| `loom/src/loom/target/arch/cmd/lower/program_plan.*` | Multi-root ownership and plan construction |
| `loom/src/loom/target/arch/cmd/lower/dispatch_counts.*` | Direct and explicit indirect count placement |
| `loom/src/loom/target/arch/cmd/lower/schedule.*` | Structured schedules and execution waves |
| `loom/src/loom/target/arch/cmd/lower/parameters.*` | Immutable parameter enumeration and placement |
| `loom/src/loom/target/arch/cmd/lower/transients.*` | Transient live ranges and slab packing |
| `loom/py/loom/target/arch/cmd/descriptors.py` | Portable `cmd.core` Low ISA |
| `loom/src/loom/target/arch/cmd/lower/serialize.*` | Closed Low-to-artifact serialization |
| `loom/src/loom/target/arch/cmd/format.*` | Canonical binary layout |
| `loom/src/loom/target/arch/cmd/program.*` | Parsed artifact view and typed accessors |

The production-backed `loom-check` emitter exposes planned root programs in
text and round-trips each root through command serialization and the untrusted
artifact parser. The `.loom-test` corpus beside this implementation therefore
exercises the same planning and serialization path as an embedding
application.
