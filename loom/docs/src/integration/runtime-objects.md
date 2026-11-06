# Runtime-provided objects

Loom programs can hold typed handles to objects supplied by a native runtime
provider. The provider owns the payload layout and operations; Loom owns the
program's types, calls, control flow, and handle lifetimes. A stateful decoder
or image frame can therefore appear in an ordinary source library without an
application-specific compiler extension.

The executable boundary described here is VM bytecode calling native providers.
The same source can be parsed, formatted, serialized, and linked using the
ordinary Loom tools. Loading the executable requires the native providers named
by its type and function imports.

## Declare an external identity

A [`type.decl`](../reference/dialects/type/ops/decl.md) gives an external payload
type a local name:

```loom
type.decl import("images", "frame") @Frame
```

The namespace and payload name form the exact external identity. Another module
may call the same identity `@ImageFrame`; its handles still match. Using the
same local spelling for a different external identity does not make the types
equivalent. Declarations precede their uses, and both key strings must be
nonempty UTF-8 without embedded NUL characters.

Two handle types refer to the declaration:

| Type | Contract |
| --- | --- |
| [`ref<@Frame>`](../reference/types/ref.md) | An owning handle. Passing or returning it transfers that owner. Every source owner must be released or transferred on each normal path. |
| [`ptr<@Frame>`](../reference/types/ptr.md) | A borrowed handle. Passing it preserves the caller's handle; the runtime carrier may hold protection needed to keep the payload alive. |

Both admit null. A `ptr` is a managed handle, with no numerical address, pointer
arithmetic, or access to native fields. Payload operations are typed function
calls. Buffer and view types remain the separate interface for memory access.

Ownership mode is part of type identity, including inside function signatures.
A provider exporting an owning parameter cannot bind an import declaring a
borrowed parameter, even though both values use the VM's managed-reference bank.

## Keep a returned pointer safe

Suppose an image provider exports `capture`, `consume`, and `width` with these
signatures:

```loom
type.decl import("images", "frame") @Frame

func.decl import("images", "capture") @capture() -> (ref<@Frame>, ptr<@Frame>)
func.decl import("images", "consume") @consume(%frame: ref<@Frame>)
func.decl import("images", "width") @width(%frame: ptr<@Frame>) -> (i32)

func.def public @capture_width() -> (i32) {
  %owner, %saved = func.call @capture() : () -> (ref<@Frame>, ptr<@Frame>)
  func.call @consume(%owner) : (ref<@Frame>)
  %width = func.call @width(%saved) : (ptr<@Frame>) -> (i32)
  func.return %width : i32
}
```

The provider publishes the borrowed result while its payload is live. The native
result handoff acquires protection before that anchor can disappear. Consuming
`%owner` therefore leaves `%saved` valid. The final `width` call ends the local
pointer's protection; if no other owners remain, the payload's native finalizer
runs before the function returns its scalar result.

This example uses a native borrowed-result handoff. It does not require a
source-level reference-to-pointer conversion. A borrowed return has a borrowed
signature even when the physical carrier holds an owner for safety.

Source owners and pointer protection have different endpoints. An explicit
source owner remains live until release or transfer, even after its last payload
read. Compiler-added pointer protection ends after the final reachable use,
including branch-local deaths and unused results. Register allocation and spill
storage preserve those boundaries rather than keeping accidental owners until
frame teardown.

Authored Low exposes `low.copy`, `low.move`, and `low.release` for explicit
carrier operations. Copying an owning handle creates another owner; moving it
transfers the owner and ends use of the source. Releasing a pointer carrier
retires any acquired protection. Releasing null has no payload effect.

## Supply the native implementation

The runtime provider publishes an `iree_vm_ref_type_table_t` for its namespace.
Each payload descriptor supplies its type name and native finalizer; each
payload begins with the intrusive `iree_vm_ref_object_t` prefix. Callable
metadata uses exact `REF` or `PTR` fields and refers to the provider's payload
type. The VM resolves those identities when binding modules.

Native implementations use the ordinary call-packet helpers to borrow or move
arguments and publish results. The provider promises that a returned handle is
null or live at the handoff. A ref count protects payload lifetime; it does not
synchronize mutable payload state or independently keep provider code loaded.
The embedding retains the provider and its descriptor storage for the full
period in which those handles can be used or finalized.

The [VM embedding guide](https://github.com/ROCm/hrx-system/blob/main/runtime/src/iree/vm/README.md#values-and-ownership)
describes carrier ownership, type-table resolution, suspension, and synchronous
host calls. The compiler does not need those native tables to understand a
source declaration; the runtime needs them to execute its imports.

## Normal exit, failure, and artifact admission

An owning SSA value cannot be consumed twice or used after transfer. Reaching a
normal exit with an unconsumed source owner is a compilation error. The source
or frontend expresses that owner's release; the compiler does not invent a
normal function-exit destructor for it.

Terminal VM failure releases every initialized managed owner held by the
invocation and runs native finalizers, including initialized results from a
partially failing native call.
It does not run arbitrary guest destructor bodies. Native finalizer order on
terminal abort is not reverse source construction order.

The default lowering pipeline prepares pointer cleanup before executable
emission. Public artifact APIs also check prepared ownership when the caller
supplies an empty or custom pass program. Direct emission requires prepared Low
IR: it diagnoses missing owner consumption or pointer cleanup without changing
the input IR. Text and serialized sources, and reachable linked library bodies,
receive the same checks.

These checks establish a compiler-input ownership boundary. They do not validate
arbitrary native provider behavior or independently prove the lifetime safety
of externally supplied VM executable bytes. Source conversion primitives,
C++ managed wrappers, and native machine-code lowering require their own
frontend and execution mappings; declaring an opaque payload does not supply
those mappings implicitly.
