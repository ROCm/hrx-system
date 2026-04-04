// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_RUNTIME_H_
#define PYRE_RUNTIME_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Export macros
//===----------------------------------------------------------------------===//

#if defined(PYRE_BUILDING_SHARED)
#define PYRE_API __attribute__((visibility("default")))
#elif defined(PYRE_STATIC)
#define PYRE_API
#else
#define PYRE_API
#endif

//===----------------------------------------------------------------------===//
// Host allocator (type + system getter only — functions after Status)
//
// Two-word value type, layout-compatible with iree_allocator_t.
// Used for host-side memory allocation (not device memory).
// pyre_host_allocator_system() returns the process-wide system allocator,
// backed by mimalloc when available.
//===----------------------------------------------------------------------===//

typedef struct pyre_host_allocator_t {
  void* self;
  void* ctl;
} pyre_host_allocator_t;

// Pre-initialized system allocator. Valid as soon as libpyre.so is loaded
// (no pyre_*_initialize() required). On ELF this is a direct data export;
// Windows will need __declspec(dllimport) when we get there.
PYRE_API extern pyre_host_allocator_t pyre_host_allocator_system_value;

static inline pyre_host_allocator_t pyre_host_allocator_system(void) {
  return pyre_host_allocator_system_value;
}

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

#define PYRE_VERSION_MAJOR 0
#define PYRE_VERSION_MINOR 1
#define PYRE_VERSION_PATCH 0

PYRE_API void pyre_runtime_version(int* major, int* minor, int* patch);

//===----------------------------------------------------------------------===//
// Status
//
// Values match iree_status_code_t. Verified by _Static_assert in
// implementation.
//===----------------------------------------------------------------------===//

typedef enum pyre_status_code_t {
  PYRE_STATUS_OK = 0,
  PYRE_STATUS_CANCELLED = 1,
  PYRE_STATUS_UNKNOWN = 2,
  PYRE_STATUS_INVALID_ARGUMENT = 3,
  PYRE_STATUS_DEADLINE_EXCEEDED = 4,
  PYRE_STATUS_NOT_FOUND = 5,
  PYRE_STATUS_ALREADY_EXISTS = 6,
  PYRE_STATUS_PERMISSION_DENIED = 7,
  PYRE_STATUS_OUT_OF_MEMORY = 8,  // IREE: RESOURCE_EXHAUSTED
  PYRE_STATUS_FAILED_PRECONDITION = 9,
  PYRE_STATUS_ABORTED = 10,
  PYRE_STATUS_OUT_OF_RANGE = 11,
  PYRE_STATUS_UNIMPLEMENTED = 12,
  PYRE_STATUS_INTERNAL = 13,
  PYRE_STATUS_UNAVAILABLE = 14,
  PYRE_STATUS_DATA_LOSS = 15,
} pyre_status_code_t;

// Status is an opaque pointer. NULL = OK, non-NULL = error with payload.
typedef struct pyre_status_s* pyre_status_t;

static inline bool pyre_status_is_ok(pyre_status_t status) {
  return status == NULL;
}

static inline pyre_status_t pyre_ok_status(void) { return NULL; }

PYRE_API pyre_status_t pyre_make_status(pyre_status_code_t code,
                                        const char* message);

PYRE_API pyre_status_code_t pyre_status_code(pyre_status_t status);

PYRE_API pyre_status_t pyre_status_to_string(pyre_status_t status,
                                             char** out_message,
                                             size_t* out_length);

PYRE_API void pyre_status_free_message(char* message);

PYRE_API void pyre_status_ignore(pyre_status_t status);

//===----------------------------------------------------------------------===//
// Host allocator functions
//===----------------------------------------------------------------------===//

// Allocates zeroed memory.
PYRE_API pyre_status_t pyre_host_allocator_malloc(
    pyre_host_allocator_t allocator, size_t byte_length, void** out_ptr);

// Allocates uninitialized memory. Only use when immediately overwriting.
PYRE_API pyre_status_t pyre_host_allocator_malloc_uninitialized(
    pyre_host_allocator_t allocator, size_t byte_length, void** out_ptr);

// Reallocates to |byte_length|. Extended bytes are undefined.
PYRE_API pyre_status_t pyre_host_allocator_realloc(
    pyre_host_allocator_t allocator, size_t byte_length, void** inout_ptr);

// Duplicates a byte block.
PYRE_API pyre_status_t pyre_host_allocator_clone(
    pyre_host_allocator_t allocator, const void* src, size_t byte_length,
    void** out_ptr);

// Frees memory. NULL is a no-op.
PYRE_API void pyre_host_allocator_free(pyre_host_allocator_t allocator,
                                       void* ptr);

// Allocates zeroed, aligned memory. The byte at |offset| within the
// allocation will be aligned to at least |min_alignment|.
PYRE_API pyre_status_t pyre_host_allocator_malloc_aligned(
    pyre_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** out_ptr);

// Reallocates aligned memory.
PYRE_API pyre_status_t pyre_host_allocator_realloc_aligned(
    pyre_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** inout_ptr);

// Frees memory from pyre_host_allocator_malloc_aligned. NULL is a no-op.
PYRE_API void pyre_host_allocator_free_aligned(pyre_host_allocator_t allocator,
                                               void* ptr);

//===----------------------------------------------------------------------===//
// Opaque handle types
//===----------------------------------------------------------------------===//

typedef struct pyre_device_s* pyre_device_t;
typedef struct pyre_allocator_s* pyre_allocator_t;
typedef struct pyre_semaphore_s* pyre_semaphore_t;
typedef struct pyre_stream_s* pyre_stream_t;
typedef struct pyre_buffer_s* pyre_buffer_t;
typedef struct pyre_module_s* pyre_module_t;
typedef struct pyre_function_s* pyre_function_t;
typedef struct pyre_value_list_s* pyre_value_list_t;
typedef struct pyre_fence_s* pyre_fence_t;
typedef struct pyre_buffer_view_s* pyre_buffer_view_t;
typedef struct pyre_executable_s* pyre_executable_t;
typedef struct pyre_compiler_output_s* pyre_compiler_output_t;
typedef struct pyre_physical_memory_s* pyre_physical_memory_t;

//===----------------------------------------------------------------------===//
// Enums and flags
//
// All values match their IREE counterparts. Verified by _Static_assert
// in the implementation. Bitfield types use typedef + #define.
//===----------------------------------------------------------------------===//

typedef enum pyre_accelerator_type_t {
  PYRE_ACCELERATOR_GPU = 0,
  PYRE_ACCELERATOR_CPU = 1,
} pyre_accelerator_type_t;

typedef enum pyre_device_property_t {
  PYRE_DEVICE_PROPERTY_NAME = 0,
  PYRE_DEVICE_PROPERTY_ARCHITECTURE,
  PYRE_DEVICE_PROPERTY_TOTAL_MEMORY,
  PYRE_DEVICE_PROPERTY_COMPUTE_UNITS,
  PYRE_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE,
  PYRE_DEVICE_PROPERTY_WARP_SIZE,
  PYRE_DEVICE_PROPERTY_MAX_SHARED_MEMORY,
  PYRE_DEVICE_PROPERTY_CLOCK_RATE,
  PYRE_DEVICE_PROPERTY_PCI_BUS_ID,
} pyre_device_property_t;

// Memory type bitfield. Values match iree_hal_memory_type_t.
typedef uint32_t pyre_memory_type_t;
#define PYRE_MEMORY_TYPE_NONE            0x00000000u
#define PYRE_MEMORY_TYPE_OPTIMAL         0x00000001u
#define PYRE_MEMORY_TYPE_HOST_VISIBLE    0x00000002u
#define PYRE_MEMORY_TYPE_HOST_COHERENT   0x00000004u
#define PYRE_MEMORY_TYPE_HOST_CACHED     0x00000008u
#define PYRE_MEMORY_TYPE_HOST_LOCAL      0x00000046u
#define PYRE_MEMORY_TYPE_DEVICE_VISIBLE  0x00000010u
#define PYRE_MEMORY_TYPE_DEVICE_LOCAL    0x00000030u

// Memory access bitfield. Values match iree_hal_memory_access_t.
typedef uint16_t pyre_memory_access_t;
#define PYRE_MEMORY_ACCESS_NONE    0x00
#define PYRE_MEMORY_ACCESS_READ    0x01
#define PYRE_MEMORY_ACCESS_WRITE   0x02
#define PYRE_MEMORY_ACCESS_DISCARD 0x04
#define PYRE_MEMORY_ACCESS_ALL     0x07

// Buffer usage bitfield. Values match iree_hal_buffer_usage_t.
typedef uint32_t pyre_buffer_usage_t;
#define PYRE_BUFFER_USAGE_NONE             0x00000000u
#define PYRE_BUFFER_USAGE_TRANSFER_SOURCE  0x00000001u
#define PYRE_BUFFER_USAGE_TRANSFER_TARGET  0x00000002u
#define PYRE_BUFFER_USAGE_TRANSFER         0x00000003u
#define PYRE_BUFFER_USAGE_DISPATCH_STORAGE_READ  0x00000400u
#define PYRE_BUFFER_USAGE_DISPATCH_STORAGE_WRITE 0x00000800u
#define PYRE_BUFFER_USAGE_DISPATCH_STORAGE       0x00000C00u
#define PYRE_BUFFER_USAGE_MAPPING_SCOPED         0x01000000u
#define PYRE_BUFFER_USAGE_MAPPING_PERSISTENT     0x02000000u
#define PYRE_BUFFER_USAGE_DEFAULT                0x00000C03u

// Map flags for pyre_buffer_map. Values match iree_hal_memory_access_t.
typedef uint16_t pyre_map_flags_t;
#define PYRE_MAP_READ    PYRE_MEMORY_ACCESS_READ
#define PYRE_MAP_WRITE   PYRE_MEMORY_ACCESS_WRITE
#define PYRE_MAP_DISCARD PYRE_MEMORY_ACCESS_DISCARD

// Dispatch flags (pyre-specific, no IREE equivalent).
typedef enum pyre_dispatch_flags_t {
  PYRE_DISPATCH_FLAG_NONE = 0,
  PYRE_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS = 1 << 0,
  PYRE_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION = 1 << 1,
} pyre_dispatch_flags_t;

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

typedef struct pyre_timeline_point_t {
  pyre_semaphore_t semaphore;
  uint64_t value;
} pyre_timeline_point_t;

typedef struct pyre_semaphore_list_t {
  pyre_semaphore_t* semaphores;
  uint64_t* values;
  size_t count;
} pyre_semaphore_list_t;

typedef uint64_t pyre_queue_affinity_t;

// Buffer parameters for allocator operations.
typedef struct pyre_buffer_params_t {
  pyre_memory_type_t type;
  pyre_memory_access_t access;
  pyre_buffer_usage_t usage;
  pyre_queue_affinity_t queue_affinity;
} pyre_buffer_params_t;

//===----------------------------------------------------------------------===//
// GPU accelerator lifecycle
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_gpu_initialize(uint32_t flags);
PYRE_API pyre_status_t pyre_gpu_shutdown(void);
PYRE_API pyre_status_t pyre_gpu_device_count(int* count);
PYRE_API pyre_status_t pyre_gpu_device_get(int index, pyre_device_t* device);

//===----------------------------------------------------------------------===//
// CPU accelerator lifecycle
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_cpu_initialize(uint32_t flags);
PYRE_API pyre_status_t pyre_cpu_shutdown(void);
PYRE_API pyre_status_t pyre_cpu_device_count(int* count);
PYRE_API pyre_status_t pyre_cpu_device_get(int index, pyre_device_t* device);

//===----------------------------------------------------------------------===//
// Device operations (generic, works for any accelerator type)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_device_get_property(pyre_device_t device,
                                                pyre_device_property_t prop,
                                                void* value,
                                                size_t value_size);

PYRE_API pyre_status_t pyre_device_synchronize(pyre_device_t device);

PYRE_API pyre_status_t pyre_device_get_type(pyre_device_t device,
                                            pyre_accelerator_type_t* type);

PYRE_API void pyre_device_retain(pyre_device_t device);
PYRE_API void pyre_device_release(pyre_device_t device);

//===----------------------------------------------------------------------===//
// Allocator
//
// Each device owns an allocator. pyre_device_allocator() returns a
// borrowed reference — valid for the lifetime of the device. Do NOT
// call pyre_allocator_release() unless you first called
// pyre_allocator_retain().
//===----------------------------------------------------------------------===//

// Returns borrowed reference. Always succeeds (every device has an allocator).
PYRE_API pyre_allocator_t pyre_device_allocator(pyre_device_t device);

PYRE_API void pyre_allocator_retain(pyre_allocator_t allocator);
PYRE_API void pyre_allocator_release(pyre_allocator_t allocator);

// Allocate buffer with explicit params. No stream ordering.
PYRE_API pyre_status_t pyre_allocator_allocate_buffer(
    pyre_allocator_t allocator, pyre_buffer_params_t params, size_t size,
    pyre_buffer_t* buffer);

// Import external host pointer as a pyre buffer.
PYRE_API pyre_status_t pyre_allocator_import_buffer(
    pyre_allocator_t allocator, pyre_buffer_params_t params,
    void* host_ptr, size_t size, pyre_buffer_t* buffer);

//===----------------------------------------------------------------------===//
// Semaphores (timeline synchronization primitives)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_semaphore_create(pyre_device_t device,
                                             uint64_t initial_value,
                                             pyre_semaphore_t* semaphore);

PYRE_API void pyre_semaphore_retain(pyre_semaphore_t semaphore);
PYRE_API void pyre_semaphore_release(pyre_semaphore_t semaphore);

PYRE_API pyre_status_t pyre_semaphore_query(pyre_semaphore_t semaphore,
                                            uint64_t* value);

PYRE_API pyre_status_t pyre_semaphore_wait(pyre_semaphore_t semaphore,
                                           uint64_t value,
                                           uint64_t timeout_ns);

PYRE_API pyre_status_t pyre_semaphore_signal(pyre_semaphore_t semaphore,
                                             uint64_t value);

//===----------------------------------------------------------------------===//
// Streams (high-level execution contexts)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_stream_create(pyre_device_t device, uint32_t flags,
                                          pyre_stream_t* stream);

PYRE_API void pyre_stream_retain(pyre_stream_t stream);
PYRE_API void pyre_stream_release(pyre_stream_t stream);

PYRE_API pyre_status_t pyre_stream_synchronize(pyre_stream_t stream);

PYRE_API pyre_status_t pyre_stream_query(pyre_stream_t stream, bool* complete);

PYRE_API pyre_status_t pyre_stream_flush(pyre_stream_t stream);

PYRE_API pyre_status_t pyre_stream_get_semaphore(
    pyre_stream_t stream, pyre_semaphore_t* semaphore);

PYRE_API pyre_status_t pyre_stream_get_timeline_position(
    pyre_stream_t stream, pyre_timeline_point_t* position);

// Advances the stream's owned timeline to the next value and returns it.
// Use when out-of-band work (e.g. VM invocation with a signal fence) will
// eventually signal the stream semaphore at the returned timepoint.
PYRE_API pyre_status_t pyre_stream_advance_timeline(
    pyre_stream_t stream, uint64_t* value);

PYRE_API pyre_status_t pyre_stream_wait_on(pyre_stream_t stream,
                                           pyre_timeline_point_t position);

//===----------------------------------------------------------------------===//
// Buffer
//===----------------------------------------------------------------------===//

// Stream-ordered allocation (convenience over allocator).
PYRE_API pyre_status_t pyre_buffer_allocate(pyre_stream_t stream, size_t size,
                                            pyre_memory_type_t mem_type,
                                            pyre_buffer_usage_t usage,
                                            pyre_buffer_t* buffer);

PYRE_API void pyre_buffer_retain(pyre_buffer_t buffer);
PYRE_API void pyre_buffer_release(pyre_buffer_t buffer);

PYRE_API pyre_status_t pyre_buffer_map(pyre_buffer_t buffer,
                                       pyre_map_flags_t flags, size_t offset,
                                       size_t size, void** mapped_ptr);

PYRE_API pyre_status_t pyre_buffer_unmap(pyre_buffer_t buffer);

PYRE_API pyre_status_t pyre_buffer_get_device_ptr(pyre_buffer_t buffer,
                                                  void** device_ptr);

PYRE_API pyre_status_t pyre_buffer_get_size(pyre_buffer_t buffer,
                                            size_t* size);

//===----------------------------------------------------------------------===//
// Synchronous data transfer
//
// These block until the transfer completes. For async transfers, use
// stream operations (pyre_stream_copy_buffer, etc).
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_synchronous_h2d(pyre_device_t device,
                                            const void* host_src,
                                            pyre_buffer_t dst,
                                            size_t dst_offset, size_t size);

PYRE_API pyre_status_t pyre_synchronous_d2h(pyre_device_t device,
                                            pyre_buffer_t src,
                                            size_t src_offset, void* host_dst,
                                            size_t size);

//===----------------------------------------------------------------------===//
// Stream operations (batched via pending command buffer)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_stream_fill_buffer(pyre_stream_t stream,
                                               pyre_buffer_t buffer,
                                               size_t offset, size_t size,
                                               const void* pattern,
                                               size_t pattern_size);

PYRE_API pyre_status_t pyre_stream_copy_buffer(
    pyre_stream_t stream, pyre_buffer_t src, size_t src_offset,
    pyre_buffer_t dst, size_t dst_offset, size_t size);

PYRE_API pyre_status_t pyre_stream_update_buffer(pyre_stream_t stream,
                                                 const void* host_data,
                                                 size_t host_data_size,
                                                 pyre_buffer_t dst,
                                                 size_t dst_offset);

//===----------------------------------------------------------------------===//
// Direct queue operations (single-op, no command buffer)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_queue_fill(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores, pyre_buffer_t buffer,
    size_t offset, size_t size, const void* pattern, size_t pattern_size);

PYRE_API pyre_status_t pyre_queue_copy(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores, pyre_buffer_t src,
    size_t src_offset, pyre_buffer_t dst, size_t dst_offset, size_t size);

PYRE_API pyre_status_t pyre_queue_barrier(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores);

// TODO(pyre): Stubs — declared for streaming rebase, not yet implemented.

// Dispatch config for kernel launch.
typedef struct pyre_dispatch_config_t {
  uint32_t workgroup_count[3];
  uint32_t workgroup_size[3];
  uint32_t subgroup_size;
} pyre_dispatch_config_t;

// Buffer binding reference for dispatch.
typedef struct pyre_buffer_ref_t {
  pyre_buffer_t buffer;
  size_t offset;
  size_t length;
} pyre_buffer_ref_t;

PYRE_API pyre_status_t pyre_queue_dispatch(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_executable_t executable, uint32_t export_ordinal,
    const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags);

PYRE_API pyre_status_t pyre_stream_dispatch(
    pyre_stream_t stream, pyre_executable_t executable,
    uint32_t export_ordinal, const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags);

// Host callback in queue execution order.
typedef pyre_status_t (*pyre_host_call_fn_t)(void* user_data);

PYRE_API pyre_status_t pyre_queue_host_call(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_host_call_fn_t callback, void* user_data);

PYRE_API pyre_status_t pyre_stream_execution_barrier(pyre_stream_t stream);

//===----------------------------------------------------------------------===//
// VM modules and invocation
//===----------------------------------------------------------------------===//

// Loads a VMFB bytecode module and instantiates an invocation context with the
// device-backed HAL module. |vmfb_data| must outlive |module|.
PYRE_API pyre_status_t pyre_module_load_vmfb(pyre_device_t device,
                                             const void* vmfb_data,
                                             size_t vmfb_size,
                                             pyre_module_t* module);

// Loads a VMFB bytecode module from compiler-owned output. The module retains
// |compiler_output| and releases that backing store when the bytecode module is
// destroyed. The original |compiler_output| handle may be released as soon as
// this returns successfully.
PYRE_API pyre_status_t pyre_module_load_compiler_output(
    pyre_device_t device, pyre_compiler_output_t compiler_output,
    pyre_module_t* module);

PYRE_API void pyre_module_retain(pyre_module_t module);
PYRE_API void pyre_module_release(pyre_module_t module);

// Resolves an exported function by fully-qualified name. The returned function
// retains |module|, so it remains valid until pyre_function_release().
PYRE_API pyre_status_t pyre_module_lookup_function(
    pyre_module_t module, const char* name, pyre_function_t* function);

PYRE_API void pyre_function_retain(pyre_function_t function);
PYRE_API void pyre_function_release(pyre_function_t function);

PYRE_API pyre_status_t pyre_function_invoke(pyre_module_t module,
                                            pyre_function_t function,
                                            pyre_value_list_t args,
                                            pyre_value_list_t rets);

//===----------------------------------------------------------------------===//
// VM value lists
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_value_list_create(size_t capacity,
                                              pyre_value_list_t* list);

PYRE_API void pyre_value_list_retain(pyre_value_list_t list);
PYRE_API void pyre_value_list_release(pyre_value_list_t list);

PYRE_API pyre_status_t pyre_value_list_size(pyre_value_list_t list,
                                            size_t* size);

PYRE_API pyre_status_t pyre_value_list_push_i64(pyre_value_list_t list,
                                                int64_t value);

PYRE_API pyre_status_t pyre_value_list_get_i64(pyre_value_list_t list,
                                               size_t index,
                                               int64_t* value);

PYRE_API pyre_status_t pyre_value_list_push_null_ref(pyre_value_list_t list);

PYRE_API pyre_status_t pyre_value_list_push_buffer(pyre_value_list_t list,
                                                   pyre_buffer_t buffer);

PYRE_API pyre_status_t pyre_value_list_push_buffer_view(
    pyre_value_list_t list, pyre_buffer_view_t buffer_view);

PYRE_API pyre_status_t pyre_value_list_push_fence(pyre_value_list_t list,
                                                  pyre_fence_t fence);

//===----------------------------------------------------------------------===//
// Fences
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_fence_create(size_t capacity,
                                         pyre_fence_t* fence);

PYRE_API pyre_status_t pyre_fence_create_at(pyre_semaphore_t semaphore,
                                            uint64_t value,
                                            pyre_fence_t* fence);

PYRE_API void pyre_fence_retain(pyre_fence_t fence);
PYRE_API void pyre_fence_release(pyre_fence_t fence);

PYRE_API pyre_status_t pyre_fence_insert(pyre_fence_t fence,
                                         pyre_semaphore_t semaphore,
                                         uint64_t value);

PYRE_API pyre_status_t pyre_fence_extend(pyre_fence_t into_fence,
                                         pyre_fence_t from_fence);

PYRE_API pyre_status_t pyre_fence_signal(pyre_fence_t fence);

PYRE_API pyre_status_t pyre_fence_wait(pyre_fence_t fence,
                                       uint64_t timeout_ns);

//===----------------------------------------------------------------------===//
// Buffer views
//===----------------------------------------------------------------------===//

typedef uint32_t pyre_element_type_t;
#define PYRE_ELEMENT_TYPE_NONE      0x00000000u
#define PYRE_ELEMENT_TYPE_OPAQUE_8  0x00000008u
#define PYRE_ELEMENT_TYPE_BOOL_8    0x13000008u
#define PYRE_ELEMENT_TYPE_INT_16    0x10000010u
#define PYRE_ELEMENT_TYPE_UINT_8    0x12000008u
#define PYRE_ELEMENT_TYPE_INT_32    0x10000020u
#define PYRE_ELEMENT_TYPE_INT_64    0x10000040u
#define PYRE_ELEMENT_TYPE_SINT_8    0x11000008u
#define PYRE_ELEMENT_TYPE_SINT_16   0x11000010u
#define PYRE_ELEMENT_TYPE_SINT_32   0x11000020u
#define PYRE_ELEMENT_TYPE_SINT_64   0x11000040u
#define PYRE_ELEMENT_TYPE_FLOAT_16  0x21000010u
#define PYRE_ELEMENT_TYPE_FLOAT_32  0x21000020u
#define PYRE_ELEMENT_TYPE_FLOAT_64  0x21000040u
#define PYRE_ELEMENT_TYPE_BFLOAT_16 0x22000010u

typedef uint32_t pyre_encoding_type_t;
#define PYRE_ENCODING_TYPE_OPAQUE          0u
#define PYRE_ENCODING_TYPE_DENSE_ROW_MAJOR 1u

PYRE_API pyre_status_t pyre_buffer_view_create(
    pyre_buffer_t buffer, size_t shape_rank, const int64_t* shape,
    pyre_element_type_t element_type, pyre_encoding_type_t encoding_type,
    pyre_buffer_view_t* buffer_view);

PYRE_API void pyre_buffer_view_retain(pyre_buffer_view_t buffer_view);

PYRE_API void pyre_buffer_view_release(pyre_buffer_view_t buffer_view);

PYRE_API pyre_status_t pyre_buffer_view_rank(pyre_buffer_view_t buffer_view,
                                             size_t* rank);

PYRE_API pyre_status_t pyre_buffer_view_dim(pyre_buffer_view_t buffer_view,
                                            size_t dim,
                                            int64_t* value);

//===----------------------------------------------------------------------===//
// Virtual memory (allocator methods)
//===----------------------------------------------------------------------===//

PYRE_API pyre_status_t pyre_allocator_query_virtual_memory(
    pyre_allocator_t allocator, pyre_memory_type_t mem_type, bool* supported,
    size_t* min_page_size, size_t* recommended_page_size);

PYRE_API pyre_status_t pyre_allocator_virtual_memory_reserve(
    pyre_allocator_t allocator, pyre_queue_affinity_t affinity, size_t size,
    pyre_buffer_t* virtual_buffer);

PYRE_API pyre_status_t pyre_allocator_virtual_memory_release(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer);

PYRE_API pyre_status_t pyre_allocator_physical_memory_allocate(
    pyre_allocator_t allocator, pyre_memory_type_t mem_type, size_t size,
    pyre_physical_memory_t* physical);

PYRE_API pyre_status_t pyre_allocator_physical_memory_free(
    pyre_allocator_t allocator, pyre_physical_memory_t physical);

PYRE_API pyre_status_t pyre_allocator_virtual_memory_map(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer,
    size_t virtual_offset, pyre_physical_memory_t physical,
    size_t physical_offset, size_t size);

PYRE_API pyre_status_t pyre_allocator_virtual_memory_unmap(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size);

// Memory protection bitfield. Values match iree_hal_memory_protection_t.
typedef uint64_t pyre_memory_protection_t;
#define PYRE_MEMORY_PROTECTION_NONE       0ull
#define PYRE_MEMORY_PROTECTION_READ       (1ull << 0)
#define PYRE_MEMORY_PROTECTION_WRITE      (1ull << 1)
#define PYRE_MEMORY_PROTECTION_READ_WRITE \
    (PYRE_MEMORY_PROTECTION_READ | PYRE_MEMORY_PROTECTION_WRITE)

PYRE_API pyre_status_t pyre_allocator_virtual_memory_protect(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, pyre_memory_protection_t protection);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_RUNTIME_H_
