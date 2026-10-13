# HAL VM module

The optional `iree/module/hal` library exposes HAL operations through statically
shaped native VM calls. Applications register the HAL reference family, create
the module with `iree_hal_module_create`, and include it in the libraries passed
to `iree_vm_program_create`. The VM core has no HAL dependency.

The current operation is `hal.queue.copy`: independent wait and signal lists,
buffer ranges, and optional before/after barrier lists with nested memory
barriers, buffer barriers, and transition recipes. It calls
`iree_hal_queue_copy` directly. Other HAL methods are not exports of this
module.

## Source authoring and lowering

HAL source operations do not name a VM ABI. For example:

```loom
func.def @copy(%queue: hal.queue, %ready: hal.semaphore,
               %done: hal.semaphore, %source: hal.buffer, %target: hal.buffer,
               %offset: i64, %length: i64, %value: i64, %flags: i64) {
  hal.queue.copy (%queue, [(%ready, %value)], [(%done, %value)],
                  (%source, %offset), (%target, %offset), %length,
                  (none, []), %flags)
  func.return
}
```

Tuples are fixed records; lists are statically shaped ranges; `none` means an
absent optional value. For a barrier boundary, `none` preserves HAL's
conservative default, while `[]` explicitly supplies no extra payload barriers.
Python builders accept the same structure as tuples, lists and `None`:

```python
builder.hal.queue_copy(arguments=(
    queue, [(ready, value)], [(done, value)],
    (source, offset), (target, offset), length, (None, []), flags,
))
```

`loom/py/loom/dialect/hal/methods.py` owns the semantic records and exact method
versions. Generation projects those declarations into source verification
tables, native ABI layouts and field positions, and VM lowering data. The C
compiler consumes linked-in tables; there is no runtime Python dependency or
sidecar Loom library.

VM lowering walks each verified source tree once and retains a native record
prototype and selector positions. It emits a typed native `low.func.decl` and
ordinary Low call; common allocation supplies the physical registers. The
artifact contains both the typed declaration and callsite record. Different
list lengths use the same exact-version native export, not different C entry
points. Fixed records flatten in the native ABI; optional values become
zero-or-one ranges, preserving absent versus present-empty groups.

Linking matches exact versions and typed layouts, checks optional cardinality,
and sizes descriptor scratch. Execution does not traverse those schema tables:
one `control.call.native` reaches the C implementation, which reads borrowed
register values while filling HAL descriptors once into invocation scratch.
The HAL captures the descriptors before returning. The module has no mutable
process state, pending-work ledger, or completion callback.

## Reference ownership

`iree/module/hal/types.h` lets a VM host pass managed HAL handles
through compiled calls. Link the optional `types` library and register
the reference family once when composing the environment:

```c
const iree_vm_ref_type_table_t* table = NULL;
IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment, &table));
iree_hal_module_types_t types;
IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types));

// Retain one argument owner while keeping the caller's HAL owner.
iree_vm_variant_t argument =
    iree_hal_buffer_variant_from_ptr_retained(&types, buffer);
```

Other native modules resolve the already registered `hal` table from the
environment. The consumer-owned `iree_hal_module_types_t` contains canonical
handles in append-only family ordinal order.
Bytecode images use their own sorted namespace/type ordinals; loading binds
those keys to the same canonical handles used by native signatures.

References preserve the original object pointer, buffer range and access rights,
and view shape, element type and encoding. VM and HAL owners share the existing
intrusive count. The final VM buffer owner invokes HAL buffer destruction; the
final view owner destroys the view and releases its buffer. No wrapper object is
allocated.
The Core VM `buffer` remains a separate CPU byte-storage type.

Device groups, queues, command buffers, semaphores and pools use cold wrapper
objects. `iree_hal_vm_queue_wrap(group, queue, allocator, &ref)`, for example,
retains both the native queue and its group. The wrapper releases the resource
before its group, preserving borrowed device/queue-family pointers even when a
VM register was the last owner. The same contract applies to command buffers,
semaphores and pools. Device-group wrappers retain the group directly.

These wrappers add no work-retirement handling. A caller still keeps native
pools alive until all returned buffers are destroyed and all operations using
the pool are terminal. Backing storage, mapping lifetimes and synchronization
remain native HAL contracts. Releasing the last wrapper does not wait for work.

The environment borrows the type table and can be freed after module
construction. Type descriptors and finalizer code must remain loaded until
all referring modules and escaped objects are gone. Statically linked tools
naturally provide that lifetime; a dynamic host keeps the provider library live.

The `types` library remains independently linkable for applications that pass
HAL references through their own modules without exposing the native HAL
operation module. Neither library introduces a compiler dependency.
