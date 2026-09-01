# IREE VM Runtime

The VM runtime executes small compiler-produced host programs without making
the compiler, a bytecode module, or a particular native-module mechanism part
of the public composition model. The same interfaces compose bytecode, native
C, and future module implementations.

The object hierarchy separates work that can be amortized from state that must
remain caller-exclusive:

1. An `iree_vm_environment_t` is a thread-safe construction-time registry of
   host-owned reference-type provider tables. Module creation resolves the
   types it uses; process creation and invocation perform no environment
   lookup.
2. An `iree_vm_module_t` is one immutable implementation-defined module. Its
   public declarations expose imports, exports, callable types, and reflection
   metadata. Its private implementation owns bytecode images, native tables,
   or any other representation.
3. An `iree_vm_program_t` links one executable module with zero or more library
   modules. Program creation resolves imports, canonicalizes callable types,
   assigns opaque per-process state slices, and retains the immutable modules.
   Libraries have no implicit initialization.
4. An `iree_vm_process_t` is one mutable instance of a program. Creation
   attaches every module's opaque state slice and invokes only the executable
   module's optional `initialize` export. A process contains no execution lock;
   the host serializes one process or creates independent processes for
   concurrent execution.
5. An `iree_vm_invocation_t` is reusable caller-owned execution storage. The
   caller may place it in fixed storage or allocate it with the convenience
   API. One invocation drives either synchronous `iree_vm_invoke` or the
   asynchronous `iree_vm_invocation_start`/`resume` state machine.

Arguments and results use source-ordered `iree_vm_variant_t` arrays. Argument
counts and result counts are exact. Once boundary validation succeeds, an
invocation consumes every input argument on every return. A failing or
suspended drive leaves result bytes untouched; terminal success initializes
every result and transfers its owners to the caller. These rules let callers
zero-initialize result storage and keep cleanup independent from failure
control flow.

Suspension is an ordinary execution outcome, not an `iree_status_t`. Status is
reserved for terminal failure. The synchronous wrapper drives the same async
core and may block or deadlock if the embedding cannot make provider progress;
executors with coroutine support use `start` and `resume` directly.

## Bytecode implementation

[`bytecode/`](bytecode/) is one module implementation. Module creation validates
external bytes once, resolves reference types, and builds the immutable runtime
tables needed for execution. The interpreter then executes those same stable
bytes directly and performs only checks that depend on live state.

The authoritative module-format and ISA model lives in
[`bytecode/spec/`](bytecode/spec/). Its Python declarations generate the wire
projection, verifier and interpreter tables, disassembly data, and published
human reference. [`bytecode/README.md`](bytecode/README.md) explains that
generation boundary and the no-second-codec conformance strategy.

## Where to start

- [`module.h`](module.h) defines the generic module ABI and reflection model.
- [`program.h`](program.h) defines immutable composition and link-time work.
- [`process.h`](process.h) defines mutable instantiation and initialization.
- [`invocation.h`](invocation.h) defines synchronous and asynchronous driving.
- [`variant.h`](variant.h) and [`ref.h`](ref.h) define values and ownership.
- [Loom's VM target](../../../../loom/src/loom/target/arch/vm/README.md)
  describes how the first compiler producer lowers and emits these modules.
