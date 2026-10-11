# HAL reference types

`iree/module/hal/types.h` lets a VM host pass managed HAL handles
through compiled calls. Link the optional `types` library and register
its provider once when composing the environment:

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
handles in append-only provider ordinal order.
Bytecode images use their own sorted namespace/type ordinals; loading binds
those keys to the same canonical handles used by native signatures.

References preserve the original object pointer, buffer range and access rights,
and view shape, element type and encoding. VM and HAL owners share the existing
intrusive count. The final VM buffer owner follows HAL recycling; the final view
owner destroys the view and releases its buffer. No wrapper object is allocated.
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

The environment borrows the provider table and can be freed after module
construction. Provider descriptors and finalizer code must remain loaded until
all referring modules and escaped objects are gone. Statically linked tools
naturally provide that lifetime; a dynamic host keeps the provider library live.

This library supplies types and ownership adapters. It does not add native HAL
operations or a compiler dependency to either the VM or HAL core libraries.
