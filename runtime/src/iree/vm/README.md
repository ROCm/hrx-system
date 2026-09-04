# IREE VM Runtime

The IREE VM executes small compiler-produced host programs while keeping the
compiler, bytecode, and native implementation details outside of the public
composition model. Bytecode modules, native C modules, and future module kinds
all implement the same immutable module interface and link into the same
programs.

The design moves every reusable operation out of the invocation path. Module
creation validates one implementation, program creation links a composition,
and process creation materializes mutable state. A warm invocation uses an
already-bound function and caller-owned storage without allocation, locking,
name lookup, type lookup, or module-shape queries.

## Object Model

```text
environment ----------borrows----------> host-owned ref-type provider tables
immutable module(s) --borrow-----------> ref-type descriptors
immutable program ----retains----------> module(s)
mutable process -------retains----------> program
reusable invocation ---temporarily borrows--> process
```

An `iree_vm_environment_t` is a thread-safe construction registry for
host-owned reference-type provider tables. Module factories that need
executable ref types use it to resolve their exact descriptors. Modules store
those pointers as borrows and do not retain the environment; the host keeps
each provider table, its descriptors and strings, and the code containing its
destroy callbacks live for as long as any module or value can name them.

A ref-type provider is simply the host library or static component that owns
one immutable namespace table, such as the core `vm` table containing
`vm.buffer`. It is a lifetime boundary, not an execution object or per-process
allocation.

An `iree_vm_module_t` is one immutable implementation-defined module. Its
public declarations contain imports, exports, structural callable types, and
optional reflection metadata. Its implementation may own bytecode, native
tables, or another representation without exposing that representation through
the generic API.

An `iree_vm_program_t` links one executable module with zero or more library
modules. Program creation retains and name-sorts the modules, canonicalizes
callable types, resolves imports, assigns opaque process-state slices, and
selects the executable module's optional `initialize` export. Library modules
have no implicit initialization.

`iree_vm_import_t`, `iree_vm_export_t`, and `iree_vm_callable_type_t` are
borrowed identities for static declarations in one module. An
`iree_vm_function_ref_t` is a resolved callable bound to a linked program. An
`iree_vm_function_t` binds that callable to one process and is the value passed
to host invocation APIs. None of these small identity values owns its module,
program, or process.

An `iree_vm_process_t` is one mutable instance of a program. Process creation
attaches every module's opaque state, runs the executable initializer, seals the
state for publication, and only then returns the process. A process contains no
execution lock. The host serializes use of one process and creates independent
processes when it wants concurrent execution.

An `iree_vm_invocation_t` is reusable fixed-capacity execution storage. The
caller may place it in an existing byte span or use the allocation convenience
API. Capacity never grows and there is no size-probe/retry protocol: the host
chooses a policy size suitable for its programs, can pool equal-sized spans, and
receives `RESOURCE_EXHAUSTED` if an operation exceeds that capacity.

## Values and Ownership

Host calls exchange source-ordered arrays of 16-byte `iree_vm_variant_t`
values. Each value is a scalar, a ref, a function ref, canonical null, or empty.
The ownership suffix on constructors and extractors is the complete lifetime
contract:

- `_borrowed` creates no owner. The source anchor remains live for every use.
- `_retained` creates an independent owner. The destination is eventually
  reset or released.
- `_move` transfers an owner and clears the source. Moving a borrowed value
  first promotes it so the result may safely escape its original anchor.

Function refs are non-owning program-bound values. Bound functions are
non-owning process-bound values. The referenced program or process remains live
for every use, including the complete duration of a suspended invocation.

### Binding Shape

A language binding normally represents environments, modules, programs, and
processes as owning objects. Function refs and bound functions are anchored
views whose wrapper keeps the corresponding program or process owner live.
Invocation storage is a separately reusable resource: a synchronous wrapper
may borrow it for one call, while an asynchronous operation object owns or
borrows it exclusively until completion, failure, or cancellation retirement.

Bindings preserve `_borrowed`, `_retained`, and `_move` as distinct ownership
operations instead of inspecting a ref count. They also initialize result
carriers to empty and release them through the same unconditional cleanup path
used by C callers. The 16-byte ref, function, and variant layouts are native
in-process ABI values; they are not stable serialized values or cross-process
handles.

A borrowed ref argument is accepted only when the bound function is declared
non-yielding. A possibly yielding function rejects it before module entry so a
host never discovers the lifetime restriction only on a rare suspend path. If
incorrect module metadata allows execution to reach an actual yield with a
borrowed root argument, the driver returns `FAILED_PRECONDITION` instead of
letting the borrow escape. Retained or moved arguments have no such restriction.

### Resolving Ref-Type Families

Ref types have structured `(namespace, type_name)` identities and canonical
process-local descriptor pointers. The environment ensures that one registered
provider owns each namespace. A consumer asks for the provider's generic table
and copies the append-only prefix it knows into a named structure:

```c
const iree_vm_ref_type_table_t* table =
    iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"));
iree_vm_ref_types_t types = {0};
IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(table, &types));

iree_vm_variant_t argument =
    iree_vm_buffer_variant_from_ptr_borrowed(&types, input_buffer);
```

A newer provider may append types without changing the prefix copied by an
older host. An older provider fails resolution for a newer consumer that needs
more fields. Carrying the provider's canonical descriptor pointers avoids
relying on global-symbol pointer equality across dynamic-library boundaries.

## Synchronous Host Flow

The following C flow starts with modules returned by their implementation
factories. The executable in this example accepts no initialization arguments
and exports `launch_config` with `(i32) -> i32`.

```c
iree_vm_module_t* executable_module = /* module factory result */;
iree_vm_program_modules_t modules = {
    .executable = executable_module,
    .libraries = iree_vm_module_span_empty(),
};

iree_vm_program_t* program = NULL;
iree_vm_process_t* process = NULL;
iree_vm_invocation_t* invocation = NULL;
iree_status_t status =
    iree_vm_program_create(modules, host_allocator, &program);

// Program creation retained its modules on success. Drop the factory owners
// after the call regardless of its status.
iree_vm_module_release(executable_module);

if (iree_status_is_ok(status)) {
  status = iree_vm_invocation_allocate(16 * 1024, host_allocator, &invocation);
}
if (iree_status_is_ok(status)) {
  status = iree_vm_process_create(program, invocation,
                                  iree_vm_variant_span_empty(), host_allocator,
                                  &process);
}

iree_vm_function_t function = iree_vm_function_null();
if (iree_status_is_ok(status)) {
  status = iree_vm_process_lookup_function(
      process, IREE_SV("launches"), IREE_SV("launch_config"), &function);
}

iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(1024)};
iree_vm_variant_t results[1] = {0};
if (iree_status_is_ok(status)) {
  status = iree_vm_invoke(invocation, function,
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results));
}

// Reset arguments unconditionally. A structurally valid invocation consumed
// them; an earlier rejection left them owned by this scope.
iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
int32_t workgroup_count = 0;
if (iree_status_is_ok(status)) {
  status = iree_vm_i32_from_variant(results[0], &workgroup_count);
}
// Results began empty and are touched only when invoke returns OK, so the same
// unconditional cleanup is valid on every path.
iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));

iree_vm_process_release(process);
iree_vm_invocation_free(invocation);
iree_vm_program_release(program);
return status;
```

Argument and result counts are always exact. The VM never truncates extra
values or partially fills a short result span. After structural boundary
validation, every invocation return consumes all arguments. Results are touched
only on terminal success and then contain a complete initialized result set.
This makes pre-zeroed result arrays and unconditional cleanup the normal safe
caller pattern.

`iree_vm_invoke` drives the asynchronous core by waiting on the operation's
wake callback. It creates no workers and pumps no event loop. It may deadlock if
the provider can make progress only on the blocked calling thread. Hosts with a
thread pool, event loop, or coroutine executor use the asynchronous API.

## Asynchronous Driving

`iree_status_t` reports failure of a drive and never represents suspension.
Once a drive enters execution, a non-OK status is terminal for that operation.
Input-boundary rejection occurs before execution and preserves the prior
operation state. Successful drives report either
`IREE_VM_EXECUTION_OUTCOME_COMPLETED` or
`IREE_VM_EXECUTION_OUTCOME_SUSPENDED`:

| Drive result | Results | Outcome | Invocation |
| --- | --- | --- | --- |
| `OK + COMPLETED` | Initialized and transferred | `COMPLETED` | Idle |
| `OK + SUSPENDED` | Untouched | `SUSPENDED` | Active |
| Non-OK after begin | Untouched | Untouched | Idle |
| Invalid `start` storage | Untouched | Untouched | Unchanged |
| Invalid `resume` | Untouched | Untouched | Suspended |

`Idle` describes invocation storage ownership only. A terminal failure unwinds
VM frames and releases their owners, but it does not roll back mutable process
state or automatically taint the process. The VM makes no general claim about
whether a process is meaningful to invoke again after a failed program call;
the program contract and host recovery policy define that scope.

`iree_vm_invocation_start` copies the level-triggered wake callback for the
operation. A provider publishes its readiness state before calling it. The wake
only tells the host that another `resume` may make progress; wake calls may be
coalesced, arrive before `start` returns, or be followed by a spurious poll.

Cancellation is the only operation allowed concurrently with a drive. The
first valid cancellation reason becomes sticky and wakes the host without
entering a provider on the cancelling thread. Before reusing the invocation or
reclaiming callback data, the host joins all threads authorized to cancel and
retires every stale queued wake from the completed operation.

## Reflection

Import, export, and callable descriptions use caller-owned storage so generic
modules may return fixed read-only data or synthesize presentation strings
without allocating. Queries follow the normal IREE size-probe/fill pattern:

```c
iree_host_size_t storage_size = 0;
iree_status_t status = iree_vm_export_query_description(
    export_value, iree_byte_span_empty(), &storage_size, NULL);

void* storage_data = NULL;
if (iree_status_is_ok(status) && storage_size != 0) {
  status = iree_allocator_malloc(host_allocator, storage_size, &storage_data);
}

iree_vm_export_description_t description = {0};
if (iree_status_is_ok(status)) {
  status = iree_vm_export_query_description(
      export_value, iree_make_byte_span(storage_data, storage_size),
      &storage_size, &description);
}
if (iree_status_is_ok(status)) {
  // Consume description.arguments/results/documentation/authored_type here.
}
iree_allocator_free(host_allocator, storage_data);
```

The allocator's normal alignment satisfies the query. Insufficient storage is
not an error: the query returns OK, reports the exact required size, and leaves
the storage and output description untouched. Field arrays point into caller
storage. Strings may point into either immutable module storage or the caller's
transient storage, so both remain live while the description is used.

Metadata is attached only to the public module, import, and export surfaces.
Entries have sorted stable keys and typed byte values. Ordinal enumeration is
available for tools, while exact-key lookup returns `out_found = false` for
ordinary absence without constructing a status.

## Implementing a Module

A module implementation embeds `iree_vm_module_t` at offset zero and owns an
immutable `iree_vm_module_descriptor_t`, vtable, declaration data, and private
representation. Its factory calls `iree_vm_module_initialize` only after those
objects are complete. Successful initialization publishes the first ref owner.
On failure the VM does not call `destroy`; the factory still owns and cleans up
the unpublished object.

```c
typedef struct app_module_t {
  iree_vm_module_t base;
  iree_allocator_t host_allocator;
  // Immutable implementation storage follows.
} app_module_t;

static void app_module_destroy(iree_vm_module_t* base_module) {
  app_module_t* module = (app_module_t*)base_module;
  iree_allocator_t host_allocator = module->host_allocator;
  iree_allocator_free(host_allocator, module);
}

static iree_status_t app_module_create(iree_allocator_t host_allocator,
                                       iree_vm_module_t** out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_module is required");
  }
  *out_module = NULL;
  app_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  iree_status_t status = iree_vm_module_initialize(
      &app_module_vtable, &app_module_descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
```

Here `app_module_vtable.destroy` is `app_module_destroy`. A real factory may
place descriptors and declaration arrays in the same allocation or borrow them
from static read-only storage; either way they remain immutable and live until
the final module release.

Program creation queries the immutable declaration tables and performs all
generic linking once. Process creation allocates one zeroed slab containing an
opaque slice for each module. `attach_state` constructs a slice, the executable
`initialize` function may mutate it, `seal_state` validates it for publication,
and `detach_state` releases it in reverse module order. Common code never
interprets the slice. Nullable lifecycle callbacks are identity operations; a
present callback is still invoked for a zero-sized slice.

`function_start` and `function_resume` are the trusted execution ABI for module
implementations, not host invocation entry points. They receive linked targets,
validated physical call packets, and the module's process slice. Yielding code
pushes a durable frame before returning `SUSPENDED`. Nested local, import, and
indirect calls are published through execution helpers and entered later by the
iterative driver, avoiding native-stack recursion.

Each callback executes with floating-point exceptions masked and
round-to-nearest selected. The driver restores the host floating-point control
state before every return, including suspension and terminal failure.

Optional presentation and metadata callbacks expose only public declarations.
Internal functions and private process state remain implementation details.

## Core Buffers

`iree_vm_buffer_t` is the core CPU-addressable byte object. Rights are READ,
WRITE, or both and may only be attenuated by subspans. Proper subspans cache
their byte pointer and retain the flattened root, keeping access fast.

Ordinary constructors create open roots. A private storage provider such as a
HAL mapping may close its root during explicit unmap by clearing root access and
data before native cleanup. Existing refs remain valid tombstone objects, but
the root and every view report no effective access and byte access fails. There
is intentionally no public generic close operation.

## Where to Start

- [`environment.h`](environment.h) and [`ref.h`](ref.h) define the construction
  registry and host-managed ref-type system.
- [`module.h`](module.h) and [`execution.h`](execution.h) define the generic
  module provider ABI.
- [`program.h`](program.h) and [`process.h`](process.h) define immutable
  composition and mutable instantiation.
- [`invocation.h`](invocation.h) and [`sync.h`](sync.h) define asynchronous and
  synchronous driving.
- [`variant.h`](variant.h) defines the host call value and typed adapter model.
- [`reflection.h`](reflection.h) defines public descriptions and metadata.
- [`bytecode/spec/`](bytecode/spec/) is the authoritative module-format and ISA
  model. The bytecode loader and interpreter are a later module implementation,
  not part of the generic API reconstructed here.
