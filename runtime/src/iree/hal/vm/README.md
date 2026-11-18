# HAL VM integration

This package adapts HAL resources to the VM. It depends on the native HAL and
Core VM libraries; both remain independent of this package and each other.
Native ownership and driver behavior stay with HAL. The type provider preserves
native object identity and shares its reference counts with VM callers.

## Reference types

`iree/hal/vm/types.h` lets a VM host pass actual HAL buffers, buffer views,
fences, and files through ordinary compiled calls. Link the optional `types`
library and register its provider once when composing the environment:

```c
const iree_vm_ref_type_table_t* table = NULL;
IREE_RETURN_IF_ERROR(iree_hal_vm_register_types(environment, &table));
iree_hal_vm_types_t types;
IREE_RETURN_IF_ERROR(iree_hal_vm_types_resolve(table, &types));

// Retain one argument owner while keeping the caller's HAL owner.
iree_vm_variant_t argument =
    iree_hal_buffer_variant_from_ptr_retained(&types, buffer);
```

Other native modules resolve the already registered `hal` table from the
environment. The consumer-owned `iree_hal_vm_types_t` contains canonical
handles for `buffer`, `buffer_view`, `fence`, and `file`, in append-only provider
ordinal order. Bytecode images use their own sorted namespace/type ordinals;
loading binds those keys to the same canonical handles used by native signatures.

References preserve the original object pointer, buffer range and access rights,
and view shape, element type and encoding. VM and HAL owners share the existing
intrusive count. The final VM buffer owner follows HAL recycling; the final view
owner destroys the view and releases its buffer. Fences retain their semaphore
timepoints and use the same shared owner count. Dropping a fence neither cancels
accepted work nor establishes retirement of the resources it uses. No wrapper
object is allocated. The Core VM `buffer` remains a separate CPU byte-storage type.

The environment borrows the provider table and can be freed after module
construction. Provider descriptors and native destructor code must remain loaded
until all referring modules and escaped objects are gone. Statically linked tools
naturally provide that lifetime; a dynamic host keeps the provider library live.

## Finite file transfers

The optional `file` library exposes the HAL's finite indirect reads and writes
to compiled VM callers. The host creates `iree_hal_vm_file_module_create` with
the registered types and an exact transfer queue, then includes the returned
`hal.file` module when linking. Neither the VM core nor the HAL core acquires a
dependency on this adapter or on a compiler.

`read_indirect` and `write_indirect` accept borrowed arrays of actual file and
buffer handles, with a corresponding array of offset/length pairs for each.
These windows bound the operation independently of a checkpoint tensor, cache,
or model format. A records buffer window and a count buffer describe the
native HAL's finite list; the adapter does not enumerate or decode that list.
Both linear and strided records can address any entry in the captured tables.

The caller supplies a producer fence and receives an independent completion
fence. For example, an application can express this dependency graph:

```text
produce records ------> read_indirect -----> consume loaded bytes
         \                                  ^
          +------------> independent work ---+
```

Submission borrows the arrays only until the native call returns. HAL captures
their resource handles and windows before returning, and retains the buffers
and files through accepted accesses. The caller can immediately release its
arrays, handles, or native module; the hosting device group, pools, and external
backing keep their ordinary lifetime obligations. Successful completion orders
consumers and descriptor reuse. Dropping the returned fence does not cancel
the operation, and an independently failed semaphore does not prove retirement.

No execution timeline is implicitly joined or advanced by the adapter. The
host's existing execution layer composes returned fences with its dispatches.
This permits transfers to overlap independent work without a host wait or an
application-specific transfer engine. Small binding tables use stack storage;
larger tables use one temporary allocation proportional to bound objects,
independent of the number of records.
