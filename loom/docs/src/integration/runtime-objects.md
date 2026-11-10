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

## Represent absence and compare identity

[`ref.null`](../reference/dialects/ref/ops/null.md) creates a typed empty handle
without calling a provider or allocating a payload:

```loom
func.def @empty_frame() -> (ref<@Frame>) {
  %empty = ref.null : ref<@Frame>
  func.return %empty : ref<@Frame>
}
```

The same operation accepts `ptr<@Frame>`. Each owning null handle has its own
source lifetime: it must be released or transferred on every normal path, just
like a live owner. Null release and duplication have no payload effect.

[`ref.compare.null`](../reference/dialects/ref/ops/compare-null.md) returns `i1`
indicating absence. [`ref.compare.eq`](../reference/dialects/ref/ops/compare-eq.md)
checks whether handles refer to the same object:

```loom
func.def @same_frame(%owner: ref<@Frame>, %frame: ptr<@Frame>) -> (i1) {
  %same = ref.compare.eq %owner, %frame : ref<@Frame>, ptr<@Frame>
  ref.release %owner : ref<@Frame>
  func.return %same : i1
}
```

Equality accepts owning and borrowed handles in either position, provided both
name the same external payload type. Two null handles compare equal; null and
live compare unequal. Two live handles compare equal only for the same object,
even when distinct objects contain identical data. These operations neither
retain nor dereference the payload. They observe a live handle and cannot use
an SSA identity after its explicit release. Their predicate results can drive
ordinary conditionals; scalar boolean negation expresses non-null or inequality.

## Borrow an owning handle

[`ref.borrow`](../reference/dialects/ref/ops/borrow.md) obtains a pointer to the
same payload without consuming the owner:

```loom
type.decl import("images", "frame") @Frame

func.decl import("images", "capture") @capture() -> (ref<@Frame>)
func.decl import("images", "width") @width(%frame: ptr<@Frame>) -> (i32)

func.def public @capture_width() -> (i32) {
  %owner = func.call @capture() : () -> (ref<@Frame>)
  %frame = ref.borrow %owner : ref<@Frame> -> ptr<@Frame>
  %width = func.call @width(%frame) : (ptr<@Frame>) -> (i32)
  ref.release %owner : ref<@Frame>
  func.return %width : i32
}
```

The owner covers the pointer's lifetime in this function. The compiler proves
that coverage across the pointer's copies and emits a borrowed carrier without
another reference-count acquisition.
[`ref.release`](../reference/dialects/ref/ops/release.md) ends this source owner's
lifetime and runs the native finalizer if it holds the last reference. The
released SSA value cannot be used again; independent handles remain valid.
Releasing null has no payload effect.

A saved pointer also remains valid when its owner ends first:

```loom
func.def public @saved_width(%owner: ref<@Frame>) -> (i32) {
  %saved = ref.borrow %owner : ref<@Frame> -> ptr<@Frame>
  ref.release %owner : ref<@Frame>
  %width = func.call @width(%saved) : (ptr<@Frame>) -> (i32)
  func.return %width : i32
}
```

Here the compiler acquires protection at `ref.borrow`, while the owner is still
live. That protection retires after `width`, before the function returns its
scalar result. Borrowing after the owner was consumed is a compilation error;
protection does not resurrect a consumed handle. Null projects to null.

Coverage is currently proved within a scheduled basic block. A pointer crossing
control flow, entering local storage, or escaping through a return retains
protection when coverage is unproved. Passing an owner and its pointer to the
same consuming call also requires protection: the callee may release the owner
before reading the pointer. These cases have the same safe source semantics.

The structured `DOMINANCE/021` remark identifies acquired protection and its
reason. For a local pointer-copy chain it also identifies the owner boundary
and final use; for control-flow and storage paths it identifies where coverage
becomes unproved. [Compilation reports](../workflows/compile-reports.md) retain
these diagnostics alongside source locations.

A provider can also return a `ptr` directly. Native result publication acquires
protection while the payload is live, and an escaping VM return acquires or
transfers protection before its frame retires. The callable signature stays
borrowed even when its physical carrier holds an owner for safety.

Source owners and pointer protection have different endpoints. An explicit
source owner remains live until release or transfer, even after its last payload
read. Compiler-added pointer protection ends after the final reachable use,
including branch-local deaths and unused results. Register allocation and spill
storage preserve those boundaries rather than keeping accidental owners until
frame teardown.

`ref.release` also accepts a `ptr` to end that handle's lifetime explicitly,
retiring any protection it holds. Ordinary pointers need no explicit release:
the compiler already retires them after their final reachable use.

Authored Low exposes `low.borrow`, `low.copy`, `low.move`, and `low.release`
for explicit carrier operations. `low.borrow` has the same safe projection
contract as the source operation; it cannot request an unchecked machine borrow.
Copying an owning handle creates another owner; moving it
transfers the owner and ends use of the source. Releasing a pointer carrier
retires any acquired protection. Releasing null has no payload effect.

## Share an object between consumers

[`ref.duplicate`](../reference/dialects/ref/ops/duplicate.md) creates an
independent handle with the same payload and ownership mode:

```loom
type.decl import("images", "frame") @Frame

func.decl import("images", "display") @display(%frame: ref<@Frame>)
func.decl import("images", "save") @save(%frame: ref<@Frame>)

func.def public @display_and_save(%frame: ref<@Frame>) {
  %saved = ref.duplicate %frame : ref<@Frame>
  func.call @display(%frame) : (ref<@Frame>)
  func.call @save(%saved) : (ref<@Frame>)
  func.return
}
```

Each consumer receives its own owner. Consuming either handle leaves the other
valid, and the payload finalizes when the last owner releases. Duplication of
null produces another null handle.

The same operation accepts `ptr<@Frame>` and produces another `ptr<@Frame>`.
Each pointer's protection ends at its own final use; duplication does not change
its public signature to `ref`. An unused owning duplicate is still an explicit
ownership obligation and is diagnosed if it reaches a normal exit unconsumed.

When a native caller lends an object for a synchronous invocation, its owner
anchors the borrowed argument for that invocation. Duplicating that argument
keeps it borrowed through register copies, local storage, and internal argument
transport, without acquiring extra owners. A pointer that already carries
protection duplicates that protection instead. Returning either kind of pointer
acquires or transfers an owner so the result can outlive the invocation.

## Carry request state between invocations

A host can retain an object between calls into a compiled program. Each call
receives that request's state, borrows it for native operations, and returns
the owner for the next turn. For example, a provider implementing an incremental
decoder could expose this contract:

```loom
type.decl import("decode", "state") @Decoder

func.decl import("decode", "append") @append(%decoder: ptr<@Decoder>, %input: buffer) -> (buffer)

func.def public @decode_chunk(%state: ref<@Decoder>, %input: buffer) -> (ref<@Decoder>, buffer) {
  %decoder = ref.borrow %state : ref<@Decoder> -> ptr<@Decoder>
  %text = func.call @append(%decoder, %input) : (ptr<@Decoder>, buffer) -> (buffer)
  func.return %state, %text : ref<@Decoder>, buffer
}
```

The provider supplies the state payload and `append` implementation. The source
program needs no knowledge of their layout. Its owning SSA value remains live
through `append`, and the return transfers that owner back to the host.
The host can interleave calls for independent decoders using the same compiled
code. Each request keeps its own returned handle; neither compiled-code
ownership nor invocation reuse establishes shared mutable request state.
Reference counting protects lifetime; the host and provider still coordinate
concurrent access to a payload.

A source turn can instead release its input owner and return a replacement.
Previously saved `ptr` values continue to name the old payload, with protection
through their final use. An escaped pointer result keeps that protection until
the host releases its result carrier. Replacing one request's state does not
change another request's state or a reader's saved payload.

An accepted owning argument transfers into the invocation even if a native
call later fails. Terminal cleanup releases the invocation's owners, and a
failed host invocation leaves its result cells unchanged. A caller that needs
to retain the input independently keeps another owner before calling; this
preserves the payload's lifetime, without rolling back native mutations.

Returned owners and protected pointer results can outlive the compiled
program. Their provider's type descriptors and finalizer code must remain live
until the final result is released.

## Eliminate unused lifetimes

A pure factory can allocate an object without making that allocation observable.
When its result is used only for handle bookkeeping, the compiler can remove
the factory, its input computation, and the complete unused lifetime:

```loom
type.decl import("images", "frame") @Frame

func.decl import("images", "create") pure @create(%width: i32) -> (ref<@Frame>)

func.def @discard_frame(%width: i32) {
  %frame = func.call pure @create(%width) : (i32) -> (ref<@Frame>)
  %copy = ref.duplicate %frame : ref<@Frame>
  ref.release %frame : ref<@Frame>
  ref.release %copy : ref<@Frame>
  func.return
}
```

The optimized body contains only `func.return`. There is no allocation,
reference-count acquisition, or finalizer invocation. Typed nulls and unused
copy, borrow, move, and release graphs follow the same rule, including
ownership forwarded through branches and loops. A pure obtain call borrowing
an existing object can disappear when its returned handles are unused.

Purity describes payload effects; it does not make two live owners
interchangeable. Repeated pure factory calls still produce independent handles.
Returning a handle, reading its payload, or using an identity comparison keeps
the required creation and cleanup. If a call produces a used scalar result
alongside an otherwise unused owner, that owner still needs its release.

Observable resource acquisition belongs in an effectful factory. Such a call
and its cleanup remain even when no payload is read. Incoming owners retain
their cleanup as well: a pure call consuming an already-materialized owner
cannot disappear on its own. Elimination removes a complete dead lifetime;
it does not shorten the explicit lifetime of a surviving object.

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

### Source calls and asynchronous native work

The current Loom VM compiler emits synchronous callable contracts. A compiled
import cannot bind to a native export declared `MAY_YIELD`; linking rejects
that mismatch before invoking the provider. The VM runtime's resumable
invocation API does not by itself make a compiled source function resumable.

A synchronous native operation can submit asynchronous device work and return
a completion object or updated request state. The native submission owns every
resource that the accepted work still needs. Source return, handle replacement,
and compiled-program destruction do not establish device completion. The host
retires that work before releasing any provider or host storage it borrows.
This allows bounded source turns to participate in an asynchronous service
while their own calls remain synchronous.

## Normal exit, failure, and artifact admission

An owning SSA value cannot be consumed twice or used after transfer. Reaching a
normal exit with an unconsumed owner in the optimized program is a compilation
error. Optimization may eliminate a complete unused lifetime before this check.
The source or frontend expresses that owner's release; the compiler does not
invent a normal function-exit destructor for it.

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
of externally supplied VM executable bytes. C++ managed wrappers and native
machine-code lowering require their own frontend and execution mappings;
declaring an opaque payload does not supply those mappings implicitly.
