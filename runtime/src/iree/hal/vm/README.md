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
through `experimental.hal.file`. This module and its signatures are temporary
and are expected to be replaced as HAL/VM integration changes. The host calls
`iree_hal_vm_file_module_create` with the registered types and a transfer queue,
then includes the returned module when linking.

The exported read has this machine signature (`ptr` borrows an object, `ref`
returns an owned object, and the four spans are read-only):

```text
experimental.hal.file.read_indirect(
  files:          span<ptr<hal.file>>,
  file_windows:   span<i64>,           // offset, length per file
  buffers:        span<ptr<hal.buffer>>,
  buffer_windows: span<i64>,           // offset, length per buffer
  wait:           ptr<hal.fence>,      // nullable producer dependency
  records:        ptr<hal.buffer>,
  records_offset: i64,
  records_length: i64,
  count:          ptr<hal.buffer>,     // contains a little-endian u64 count
  count_offset:   i64
) -> ref<hal.fence>
```

`write_indirect` has the same signature and copies from buffers to files.
The queue comes from module creation, so it is absent from each call. The
`hal.file`, `hal.buffer`, and `hal.fence` types retain their canonical identities.

For example, bind one checkpoint file window and one cache buffer window. A
device can write two native records selecting these byte ranges, relative to
those windows:

```text
file[0] offset 4096  -> buffer[0] offset   0, length 128
file[0] offset 8192  -> buffer[0] offset 128, length 128
count = 2
```

Submit their buffers once with the device producer's fence. The adapter forwards
the native record list without reading or decoding it. HAL waits for the
producer, performs the reads, and signals the returned fence after the loaded
bytes are visible. Linear and strided records share this path; file and buffer
windows bound every access. Independent dispatches can overlap the reads:

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

The host's execution layer composes returned fences with its dispatches; the
adapter does not implicitly join or advance that timeline. Small binding tables
use stack storage; larger tables use one temporary allocation proportional to
bound objects, independent of the number of records.

Neither VM core nor HAL core depends on this adapter. Its argument and ownership
tests live beside it and use the real task driver through shared CTS utilities.
Native HAL CTS tests remain VM-independent and exercise transfers across drivers.
