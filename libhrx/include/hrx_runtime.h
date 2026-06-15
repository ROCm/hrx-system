// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_RUNTIME_H_
#define HRX_RUNTIME_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Export macros
//===----------------------------------------------------------------------===//

#if defined(_WIN32) && defined(HRX_BUILDING_SHARED)
#define HRX_API __declspec(dllexport)
#elif defined(_WIN32) && !defined(HRX_STATIC)
#define HRX_API __declspec(dllimport)
#elif defined(HRX_BUILDING_SHARED)
#define HRX_API __attribute__((visibility("default")))
#elif defined(HRX_STATIC)
#define HRX_API
#else
#define HRX_API
#endif

//===----------------------------------------------------------------------===//
// Host allocator (type + system getter only — functions after Status)
//
// Two-word value type, layout-compatible with iree_allocator_t.
// Used for host-side memory allocation (not device memory).
// hrx_host_allocator_system() returns the process-wide system allocator,
// backed by mimalloc when available.
//===----------------------------------------------------------------------===//

typedef struct hrx_host_allocator_t {
  void* self;
  void* ctl;
} hrx_host_allocator_t;

// Returns the process-wide system allocator. Valid as soon as libhrx is loaded
// (no hrx_*_initialize() required).
HRX_API hrx_host_allocator_t hrx_host_allocator_system(void);

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

#define HRX_VERSION_MAJOR 0
#define HRX_VERSION_MINOR 1
#define HRX_VERSION_PATCH 0

HRX_API void hrx_runtime_version(int* major, int* minor, int* patch);

//===----------------------------------------------------------------------===//
// Status
//
// Values match iree_status_code_t. Verified by _Static_assert in
// implementation.
//===----------------------------------------------------------------------===//

typedef enum hrx_status_code_t {
  HRX_STATUS_OK = 0,
  HRX_STATUS_CANCELLED = 1,
  HRX_STATUS_UNKNOWN = 2,
  HRX_STATUS_INVALID_ARGUMENT = 3,
  HRX_STATUS_DEADLINE_EXCEEDED = 4,
  HRX_STATUS_NOT_FOUND = 5,
  HRX_STATUS_ALREADY_EXISTS = 6,
  HRX_STATUS_PERMISSION_DENIED = 7,
  HRX_STATUS_OUT_OF_MEMORY = 8,  // IREE: RESOURCE_EXHAUSTED
  HRX_STATUS_FAILED_PRECONDITION = 9,
  HRX_STATUS_ABORTED = 10,
  HRX_STATUS_OUT_OF_RANGE = 11,
  HRX_STATUS_UNIMPLEMENTED = 12,
  HRX_STATUS_INTERNAL = 13,
  HRX_STATUS_UNAVAILABLE = 14,
  HRX_STATUS_DATA_LOSS = 15,
} hrx_status_code_t;

// Status is an opaque pointer. NULL = OK, non-NULL = error with payload.
typedef struct hrx_status_s* hrx_status_t;

static inline bool hrx_status_is_ok(hrx_status_t status) {
  return status == NULL;
}

static inline hrx_status_t hrx_ok_status(void) { return NULL; }

HRX_API hrx_status_t hrx_make_status(hrx_status_code_t code,
                                     const char* message);

HRX_API hrx_status_code_t hrx_status_code(hrx_status_t status);

HRX_API hrx_status_t hrx_status_to_string(hrx_status_t status,
                                          char** out_message,
                                          size_t* out_length);

HRX_API void hrx_status_free_message(char* message);

HRX_API void hrx_status_ignore(hrx_status_t status);

//===----------------------------------------------------------------------===//
// Host allocator functions
//===----------------------------------------------------------------------===//

// Allocates zeroed memory.
HRX_API hrx_status_t hrx_host_allocator_malloc(hrx_host_allocator_t allocator,
                                               size_t byte_length,
                                               void** out_ptr);

// Allocates uninitialized memory. Only use when immediately overwriting.
HRX_API hrx_status_t hrx_host_allocator_malloc_uninitialized(
    hrx_host_allocator_t allocator, size_t byte_length, void** out_ptr);

// Reallocates to |byte_length|. Extended bytes are undefined.
HRX_API hrx_status_t hrx_host_allocator_realloc(hrx_host_allocator_t allocator,
                                                size_t byte_length,
                                                void** inout_ptr);

// Duplicates a byte block.
HRX_API hrx_status_t hrx_host_allocator_clone(hrx_host_allocator_t allocator,
                                              const void* src,
                                              size_t byte_length,
                                              void** out_ptr);

// Frees memory. NULL is a no-op.
HRX_API void hrx_host_allocator_free(hrx_host_allocator_t allocator, void* ptr);

// Allocates zeroed, aligned memory. The byte at |offset| within the
// allocation will be aligned to at least |min_alignment|.
HRX_API hrx_status_t hrx_host_allocator_malloc_aligned(
    hrx_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** out_ptr);

// Reallocates aligned memory.
HRX_API hrx_status_t hrx_host_allocator_realloc_aligned(
    hrx_host_allocator_t allocator, size_t byte_length, size_t min_alignment,
    size_t offset, void** inout_ptr);

// Frees memory from hrx_host_allocator_malloc_aligned. NULL is a no-op.
HRX_API void hrx_host_allocator_free_aligned(hrx_host_allocator_t allocator,
                                             void* ptr);

//===----------------------------------------------------------------------===//
// Opaque handle types
//===----------------------------------------------------------------------===//

typedef struct hrx_device_s* hrx_device_t;
typedef struct hrx_allocator_s* hrx_allocator_t;
typedef struct hrx_semaphore_s* hrx_semaphore_t;
typedef struct hrx_stream_s* hrx_stream_t;
typedef struct hrx_buffer_s* hrx_buffer_t;
typedef struct hrx_module_s* hrx_module_t;
typedef struct hrx_function_s* hrx_function_t;
typedef struct hrx_value_list_s* hrx_value_list_t;
typedef struct hrx_fence_s* hrx_fence_t;
typedef struct hrx_buffer_view_s* hrx_buffer_view_t;
typedef struct hrx_executable_s* hrx_executable_t;
typedef struct hrx_physical_memory_s* hrx_physical_memory_t;
typedef struct hrx_event_s* hrx_event_t;
typedef struct hrx_mem_pool_s* hrx_mem_pool_t;
typedef struct hrx_graph_s* hrx_graph_t;
typedef struct hrx_graph_exec_s* hrx_graph_exec_t;
typedef struct hrx_graph_node_s* hrx_graph_node_t;

//===----------------------------------------------------------------------===//
// Enums and flags
//
// All values match their IREE counterparts. Verified by _Static_assert
// in the implementation. Bitfield types use typedef + #define.
//===----------------------------------------------------------------------===//

typedef enum hrx_accelerator_type_t {
  HRX_ACCELERATOR_GPU = 0,
  HRX_ACCELERATOR_CPU = 1,
} hrx_accelerator_type_t;

typedef enum hrx_device_property_t {
  HRX_DEVICE_PROPERTY_NAME = 0,
  HRX_DEVICE_PROPERTY_ARCHITECTURE,
  HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
  HRX_DEVICE_PROPERTY_COMPUTE_UNITS,
  HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE,
  HRX_DEVICE_PROPERTY_WARP_SIZE,
  HRX_DEVICE_PROPERTY_MAX_SHARED_MEMORY,
  HRX_DEVICE_PROPERTY_CLOCK_RATE,
  HRX_DEVICE_PROPERTY_PCI_BUS_ID,
} hrx_device_property_t;

// Memory type bitfield. Values match iree_hal_memory_type_t.
typedef uint32_t hrx_memory_type_t;
#define HRX_MEMORY_TYPE_NONE 0x00000000u
#define HRX_MEMORY_TYPE_OPTIMAL 0x00000001u
#define HRX_MEMORY_TYPE_HOST_VISIBLE 0x00000002u
#define HRX_MEMORY_TYPE_HOST_COHERENT 0x00000004u
#define HRX_MEMORY_TYPE_HOST_CACHED 0x00000008u
#define HRX_MEMORY_TYPE_HOST_LOCAL 0x00000046u
#define HRX_MEMORY_TYPE_DEVICE_VISIBLE 0x00000010u
#define HRX_MEMORY_TYPE_DEVICE_LOCAL 0x00000030u

// Memory access bitfield. Values match iree_hal_memory_access_t.
typedef uint16_t hrx_memory_access_t;
#define HRX_MEMORY_ACCESS_NONE 0x00
#define HRX_MEMORY_ACCESS_READ 0x01
#define HRX_MEMORY_ACCESS_WRITE 0x02
#define HRX_MEMORY_ACCESS_DISCARD 0x04
#define HRX_MEMORY_ACCESS_ALL 0x07

// Buffer usage bitfield. Values match iree_hal_buffer_usage_t.
typedef uint32_t hrx_buffer_usage_t;
#define HRX_BUFFER_USAGE_NONE 0x00000000u
#define HRX_BUFFER_USAGE_TRANSFER_SOURCE 0x00000001u
#define HRX_BUFFER_USAGE_TRANSFER_TARGET 0x00000002u
#define HRX_BUFFER_USAGE_TRANSFER 0x00000003u
#define HRX_BUFFER_USAGE_DISPATCH_STORAGE_READ 0x00000400u
#define HRX_BUFFER_USAGE_DISPATCH_STORAGE_WRITE 0x00000800u
#define HRX_BUFFER_USAGE_DISPATCH_STORAGE 0x00000C00u
#define HRX_BUFFER_USAGE_MAPPING_SCOPED 0x01000000u
#define HRX_BUFFER_USAGE_MAPPING_PERSISTENT 0x02000000u
#define HRX_BUFFER_USAGE_DEFAULT 0x00000C03u

// Map flags for hrx_buffer_map. Values match iree_hal_memory_access_t.
typedef uint16_t hrx_map_flags_t;
#define HRX_MAP_READ HRX_MEMORY_ACCESS_READ
#define HRX_MAP_WRITE HRX_MEMORY_ACCESS_WRITE
#define HRX_MAP_DISCARD HRX_MEMORY_ACCESS_DISCARD

// Dispatch flags (hrx-specific, no IREE equivalent).
typedef enum hrx_dispatch_flags_t {
  HRX_DISPATCH_FLAG_NONE = 0,
  HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS = 1 << 0,
  HRX_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION = 1 << 1,
} hrx_dispatch_flags_t;

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

typedef struct hrx_timeline_point_t {
  hrx_semaphore_t semaphore;
  uint64_t value;
} hrx_timeline_point_t;

typedef struct hrx_semaphore_list_t {
  hrx_semaphore_t* semaphores;
  uint64_t* values;
  size_t count;
} hrx_semaphore_list_t;

typedef uint64_t hrx_queue_affinity_t;

// Buffer parameters for allocator operations.
typedef struct hrx_buffer_params_t {
  hrx_memory_type_t type;
  hrx_memory_access_t access;
  hrx_buffer_usage_t usage;
  hrx_queue_affinity_t queue_affinity;
} hrx_buffer_params_t;

// Export metadata for direct executable dispatch. |name| is owned by the
// executable and remains valid until hrx_executable_release() drops the last
// reference. |constant_byte_length| is the byte length of the constants buffer
// expected by hrx_queue_dispatch()/hrx_stream_dispatch().
typedef struct hrx_executable_export_info_t {
  const char* name;
  uint32_t flags;
  uint32_t constant_byte_length;
  uint32_t binding_count;
  uint32_t parameter_count;
  uint32_t workgroup_size[3];
} hrx_executable_export_info_t;

//===----------------------------------------------------------------------===//
// GPU accelerator lifecycle
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_gpu_initialize(uint32_t flags);
HRX_API hrx_status_t hrx_gpu_shutdown(void);
HRX_API hrx_status_t hrx_gpu_device_count(int* count);
HRX_API hrx_status_t hrx_gpu_device_get(int index, hrx_device_t* device);

//===----------------------------------------------------------------------===//
// CPU accelerator lifecycle
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_cpu_initialize(uint32_t flags);
HRX_API hrx_status_t hrx_cpu_shutdown(void);
HRX_API hrx_status_t hrx_cpu_device_count(int* count);
HRX_API hrx_status_t hrx_cpu_device_get(int index, hrx_device_t* device);

//===----------------------------------------------------------------------===//
// Device operations (generic, works for any accelerator type)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_device_get_property(hrx_device_t device,
                                             hrx_device_property_t prop,
                                             void* value, size_t value_size);

// Deprecated: devices do not own a single implicit timeline. Synchronize
// explicit streams, semaphores, or fences instead.
HRX_API hrx_status_t hrx_device_synchronize(hrx_device_t device);

HRX_API hrx_status_t hrx_device_get_type(hrx_device_t device,
                                         hrx_accelerator_type_t* type);

HRX_API void hrx_device_retain(hrx_device_t device);
HRX_API void hrx_device_release(hrx_device_t device);

//===----------------------------------------------------------------------===//
// Allocator
//
// Each device owns an allocator. hrx_device_allocator() returns a
// borrowed reference — valid for the lifetime of the device. Do NOT
// call hrx_allocator_release() unless you first called
// hrx_allocator_retain().
//===----------------------------------------------------------------------===//

// Returns borrowed reference. Always succeeds (every device has an allocator).
HRX_API hrx_allocator_t hrx_device_allocator(hrx_device_t device);

HRX_API void hrx_allocator_retain(hrx_allocator_t allocator);
HRX_API void hrx_allocator_release(hrx_allocator_t allocator);

// Allocate buffer with explicit params. No stream ordering.
HRX_API hrx_status_t hrx_allocator_allocate_buffer(hrx_allocator_t allocator,
                                                   hrx_buffer_params_t params,
                                                   size_t size,
                                                   hrx_buffer_t* buffer);

// Import external host pointer as a hrx buffer.
HRX_API hrx_status_t hrx_allocator_import_buffer(hrx_allocator_t allocator,
                                                 hrx_buffer_params_t params,
                                                 void* host_ptr, size_t size,
                                                 hrx_buffer_t* buffer);

//===----------------------------------------------------------------------===//
// Semaphores (timeline synchronization primitives)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_semaphore_create(hrx_device_t device,
                                          uint64_t initial_value,
                                          hrx_semaphore_t* semaphore);

HRX_API void hrx_semaphore_retain(hrx_semaphore_t semaphore);
HRX_API void hrx_semaphore_release(hrx_semaphore_t semaphore);

HRX_API hrx_status_t hrx_semaphore_query(hrx_semaphore_t semaphore,
                                         uint64_t* value);

HRX_API hrx_status_t hrx_semaphore_wait(hrx_semaphore_t semaphore,
                                        uint64_t value, uint64_t timeout_ns);

HRX_API hrx_status_t hrx_semaphore_signal(hrx_semaphore_t semaphore,
                                          uint64_t value);

//===----------------------------------------------------------------------===//
// Streams (high-level execution contexts)
//
// hrx_stream_t is the canonical owner of a stream timeline/semaphore inside
// libhrx. Frontend stream objects should retain a hrx_stream_t and query the
// owning device or timeline via accessors instead of creating a second backend
// timeline universe.
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_stream_create(hrx_device_t device, uint32_t flags,
                                       hrx_stream_t* stream);

HRX_API void hrx_stream_retain(hrx_stream_t stream);
HRX_API void hrx_stream_release(hrx_stream_t stream);

HRX_API hrx_status_t hrx_stream_synchronize(hrx_stream_t stream);

// Waits for the stream's current timeline position without flushing pending
// command buffer work. Use hrx_stream_synchronize() to flush and then wait.
HRX_API hrx_status_t hrx_stream_wait(hrx_stream_t stream);

HRX_API hrx_status_t hrx_stream_query(hrx_stream_t stream, bool* complete);

HRX_API hrx_status_t hrx_stream_flush(hrx_stream_t stream);

HRX_API hrx_status_t hrx_stream_get_semaphore(hrx_stream_t stream,
                                              hrx_semaphore_t* semaphore);

// Returns the device that owns this stream. The returned pointer is borrowed
// and remains valid while |stream| is retained. Call hrx_device_retain() if a
// longer-lived reference is needed.
HRX_API hrx_status_t hrx_stream_get_device(hrx_stream_t stream,
                                           hrx_device_t* device);

HRX_API hrx_status_t hrx_stream_get_timeline_position(
    hrx_stream_t stream, hrx_timeline_point_t* position);

// Advances the stream's owned timeline to the next value and returns it.
// Use when out-of-band work (e.g. VM invocation with a signal fence) will
// eventually signal the stream semaphore at the returned timepoint.
HRX_API hrx_status_t hrx_stream_advance_timeline(hrx_stream_t stream,
                                                 uint64_t* value);

HRX_API hrx_status_t hrx_stream_wait_on(hrx_stream_t stream,
                                        hrx_timeline_point_t position);

//===----------------------------------------------------------------------===//
// Buffer
//===----------------------------------------------------------------------===//

// Stream-ordered allocation (convenience over allocator).
HRX_API hrx_status_t hrx_buffer_allocate(hrx_stream_t stream, size_t size,
                                         hrx_memory_type_t mem_type,
                                         hrx_buffer_usage_t usage,
                                         hrx_buffer_t* buffer);

HRX_API void hrx_buffer_retain(hrx_buffer_t buffer);
HRX_API void hrx_buffer_release(hrx_buffer_t buffer);

HRX_API hrx_status_t hrx_buffer_map(hrx_buffer_t buffer, hrx_map_flags_t flags,
                                    size_t offset, size_t size,
                                    void** mapped_ptr);

HRX_API hrx_status_t hrx_buffer_unmap(hrx_buffer_t buffer);

HRX_API hrx_status_t hrx_buffer_get_device_ptr(hrx_buffer_t buffer,
                                               void** device_ptr);

HRX_API hrx_status_t hrx_buffer_get_size(hrx_buffer_t buffer, size_t* size);

//===----------------------------------------------------------------------===//
// Synchronous data transfer
//
// These block until the transfer completes. For async transfers, use
// stream operations (hrx_stream_copy_buffer, etc).
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_synchronous_h2d(hrx_device_t device,
                                         const void* host_src, hrx_buffer_t dst,
                                         size_t dst_offset, size_t size);

HRX_API hrx_status_t hrx_synchronous_d2h(hrx_device_t device, hrx_buffer_t src,
                                         size_t src_offset, void* host_dst,
                                         size_t size);

//===----------------------------------------------------------------------===//
// Stream operations (batched via pending command buffer)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_stream_fill_buffer(hrx_stream_t stream,
                                            hrx_buffer_t buffer, size_t offset,
                                            size_t size, const void* pattern,
                                            size_t pattern_size);

HRX_API hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream,
                                            hrx_buffer_t src, size_t src_offset,
                                            hrx_buffer_t dst, size_t dst_offset,
                                            size_t size);

HRX_API hrx_status_t hrx_stream_update_buffer(hrx_stream_t stream,
                                              const void* host_data,
                                              size_t host_data_size,
                                              hrx_buffer_t dst,
                                              size_t dst_offset);

//===----------------------------------------------------------------------===//
// Direct queue operations (single-op, no command buffer)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_queue_fill(
    hrx_device_t device, hrx_queue_affinity_t affinity,
    const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_buffer_t buffer,
    size_t offset, size_t size, const void* pattern, size_t pattern_size);

HRX_API hrx_status_t hrx_queue_copy(
    hrx_device_t device, hrx_queue_affinity_t affinity,
    const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_buffer_t src,
    size_t src_offset, hrx_buffer_t dst, size_t dst_offset, size_t size);

HRX_API hrx_status_t
hrx_queue_barrier(hrx_device_t device, hrx_queue_affinity_t affinity,
                  const hrx_semaphore_list_t* wait_semaphores,
                  const hrx_semaphore_list_t* signal_semaphores);

// TODO(hrx): Stubs — declared for streaming rebase, not yet implemented.

// Dispatch config for kernel launch.
typedef struct hrx_dispatch_config_t {
  uint32_t workgroup_count[3];
  uint32_t workgroup_size[3];
  uint32_t subgroup_size;
} hrx_dispatch_config_t;

// Buffer binding reference for dispatch.
typedef struct hrx_buffer_ref_t {
  hrx_buffer_t buffer;
  size_t offset;
  size_t length;
} hrx_buffer_ref_t;

HRX_API hrx_status_t hrx_queue_dispatch(
    hrx_device_t device, hrx_queue_affinity_t affinity,
    const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_executable_t executable,
    uint32_t export_ordinal, const hrx_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const hrx_buffer_ref_t* bindings, size_t binding_count, uint32_t flags);

HRX_API hrx_status_t hrx_stream_dispatch(
    hrx_stream_t stream, hrx_executable_t executable, uint32_t export_ordinal,
    const hrx_dispatch_config_t* config, const void* constants,
    size_t constants_size, const hrx_buffer_ref_t* bindings,
    size_t binding_count, uint32_t flags);

// Host callback in queue execution order.
typedef hrx_status_t (*hrx_host_call_fn_t)(void* user_data);

HRX_API hrx_status_t
hrx_queue_host_call(hrx_device_t device, hrx_queue_affinity_t affinity,
                    const hrx_semaphore_list_t* wait_semaphores,
                    const hrx_semaphore_list_t* signal_semaphores,
                    hrx_host_call_fn_t callback, void* user_data);

HRX_API hrx_status_t hrx_stream_execution_barrier(hrx_stream_t stream);

//===----------------------------------------------------------------------===//
// Direct executables
//
// Load native executable packages and inspect export metadata for direct
// dispatch. Kernel tensor/storage arguments should be passed as bindings;
// scalars, strides, and sizes should be packed into the constants block.
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_executable_load_data(hrx_device_t device,
                                              const void* executable_data,
                                              size_t executable_data_size,
                                              const char* executable_format,
                                              hrx_executable_t* executable);

HRX_API hrx_status_t hrx_executable_load_file(hrx_device_t device,
                                              const char* path,
                                              const char* executable_format,
                                              hrx_executable_t* executable);

HRX_API void hrx_executable_retain(hrx_executable_t executable);
HRX_API void hrx_executable_release(hrx_executable_t executable);

HRX_API hrx_status_t hrx_executable_export_count(hrx_executable_t executable,
                                                 size_t* count);

HRX_API hrx_status_t
hrx_executable_export_info(hrx_executable_t executable, uint32_t export_ordinal,
                           hrx_executable_export_info_t* out_info);

HRX_API hrx_status_t hrx_executable_lookup_export_by_name(
    hrx_executable_t executable, const char* name, uint32_t* export_ordinal);

//===----------------------------------------------------------------------===//
// VM modules and invocation
//===----------------------------------------------------------------------===//

// Loads a VMFB bytecode module and instantiates an invocation context with the
// device-backed HAL module. |vmfb_data| must outlive |module|.
HRX_API hrx_status_t hrx_module_load_vmfb(hrx_device_t device,
                                          const void* vmfb_data,
                                          size_t vmfb_size,
                                          hrx_module_t* module);

HRX_API void hrx_module_retain(hrx_module_t module);
HRX_API void hrx_module_release(hrx_module_t module);

// Resolves an exported function by fully-qualified name. The returned function
// retains |module|, so it remains valid until hrx_function_release().
HRX_API hrx_status_t hrx_module_lookup_function(hrx_module_t module,
                                                const char* name,
                                                hrx_function_t* function);

HRX_API void hrx_function_retain(hrx_function_t function);
HRX_API void hrx_function_release(hrx_function_t function);

HRX_API hrx_status_t hrx_function_invoke(hrx_module_t module,
                                         hrx_function_t function,
                                         hrx_value_list_t args,
                                         hrx_value_list_t rets);

//===----------------------------------------------------------------------===//
// VM value lists
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_value_list_create(size_t capacity,
                                           hrx_value_list_t* list);

HRX_API void hrx_value_list_retain(hrx_value_list_t list);
HRX_API void hrx_value_list_release(hrx_value_list_t list);

HRX_API hrx_status_t hrx_value_list_size(hrx_value_list_t list, size_t* size);

HRX_API hrx_status_t hrx_value_list_push_i64(hrx_value_list_t list,
                                             int64_t value);

HRX_API hrx_status_t hrx_value_list_get_i64(hrx_value_list_t list, size_t index,
                                            int64_t* value);

HRX_API hrx_status_t hrx_value_list_push_null_ref(hrx_value_list_t list);

HRX_API hrx_status_t hrx_value_list_push_buffer(hrx_value_list_t list,
                                                hrx_buffer_t buffer);

HRX_API hrx_status_t hrx_value_list_push_buffer_view(
    hrx_value_list_t list, hrx_buffer_view_t buffer_view);

HRX_API hrx_status_t hrx_value_list_push_fence(hrx_value_list_t list,
                                               hrx_fence_t fence);

//===----------------------------------------------------------------------===//
// Fences
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_fence_create(size_t capacity, hrx_fence_t* fence);

HRX_API hrx_status_t hrx_fence_create_at(hrx_semaphore_t semaphore,
                                         uint64_t value, hrx_fence_t* fence);

HRX_API void hrx_fence_retain(hrx_fence_t fence);
HRX_API void hrx_fence_release(hrx_fence_t fence);

HRX_API hrx_status_t hrx_fence_insert(hrx_fence_t fence,
                                      hrx_semaphore_t semaphore,
                                      uint64_t value);

HRX_API hrx_status_t hrx_fence_extend(hrx_fence_t into_fence,
                                      hrx_fence_t from_fence);

HRX_API hrx_status_t hrx_fence_signal(hrx_fence_t fence);

HRX_API hrx_status_t hrx_fence_wait(hrx_fence_t fence, uint64_t timeout_ns);

//===----------------------------------------------------------------------===//
// Buffer views
//===----------------------------------------------------------------------===//

typedef uint32_t hrx_element_type_t;
#define HRX_ELEMENT_TYPE_NONE 0x00000000u
#define HRX_ELEMENT_TYPE_OPAQUE_8 0x00000008u
#define HRX_ELEMENT_TYPE_BOOL_8 0x13000008u
#define HRX_ELEMENT_TYPE_INT_16 0x10000010u
#define HRX_ELEMENT_TYPE_UINT_8 0x12000008u
#define HRX_ELEMENT_TYPE_INT_32 0x10000020u
#define HRX_ELEMENT_TYPE_INT_64 0x10000040u
#define HRX_ELEMENT_TYPE_SINT_8 0x11000008u
#define HRX_ELEMENT_TYPE_SINT_16 0x11000010u
#define HRX_ELEMENT_TYPE_SINT_32 0x11000020u
#define HRX_ELEMENT_TYPE_SINT_64 0x11000040u
#define HRX_ELEMENT_TYPE_FLOAT_16 0x21000010u
#define HRX_ELEMENT_TYPE_FLOAT_32 0x21000020u
#define HRX_ELEMENT_TYPE_FLOAT_64 0x21000040u
#define HRX_ELEMENT_TYPE_BFLOAT_16 0x22000010u

typedef uint32_t hrx_encoding_type_t;
#define HRX_ENCODING_TYPE_OPAQUE 0u
#define HRX_ENCODING_TYPE_DENSE_ROW_MAJOR 1u

HRX_API hrx_status_t hrx_buffer_view_create(hrx_buffer_t buffer,
                                            size_t shape_rank,
                                            const int64_t* shape,
                                            hrx_element_type_t element_type,
                                            hrx_encoding_type_t encoding_type,
                                            hrx_buffer_view_t* buffer_view);

HRX_API void hrx_buffer_view_retain(hrx_buffer_view_t buffer_view);

HRX_API void hrx_buffer_view_release(hrx_buffer_view_t buffer_view);

HRX_API hrx_status_t hrx_buffer_view_rank(hrx_buffer_view_t buffer_view,
                                          size_t* rank);

HRX_API hrx_status_t hrx_buffer_view_dim(hrx_buffer_view_t buffer_view,
                                         size_t dim, int64_t* value);

//===----------------------------------------------------------------------===//
// Virtual memory (allocator methods)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_allocator_query_virtual_memory(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, bool* supported,
    size_t* min_page_size, size_t* recommended_page_size);

HRX_API hrx_status_t hrx_allocator_virtual_memory_reserve(
    hrx_allocator_t allocator, hrx_queue_affinity_t affinity, size_t size,
    hrx_buffer_t* virtual_buffer);

HRX_API hrx_status_t hrx_allocator_virtual_memory_release(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer);

HRX_API hrx_status_t hrx_allocator_physical_memory_allocate(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, size_t size,
    hrx_physical_memory_t* physical);

HRX_API hrx_status_t hrx_allocator_physical_memory_free(
    hrx_allocator_t allocator, hrx_physical_memory_t physical);

HRX_API hrx_status_t hrx_allocator_virtual_memory_map(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, hrx_physical_memory_t physical,
    size_t physical_offset, size_t size);

HRX_API hrx_status_t hrx_allocator_virtual_memory_unmap(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size);

// Memory protection bitfield. Values match iree_hal_memory_protection_t.
typedef uint64_t hrx_memory_protection_t;
#define HRX_MEMORY_PROTECTION_NONE 0ull
#define HRX_MEMORY_PROTECTION_READ (1ull << 0)
#define HRX_MEMORY_PROTECTION_WRITE (1ull << 1)
#define HRX_MEMORY_PROTECTION_READ_WRITE \
  (HRX_MEMORY_PROTECTION_READ | HRX_MEMORY_PROTECTION_WRITE)

HRX_API hrx_status_t hrx_allocator_virtual_memory_protect(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, hrx_memory_protection_t protection);

//===----------------------------------------------------------------------===//
// Events (stream synchronization points)
//===----------------------------------------------------------------------===//

typedef enum hrx_event_flags_t {
  HRX_EVENT_FLAG_NONE = 0,
  HRX_EVENT_FLAG_BLOCKING_SYNC = 1 << 0,
  HRX_EVENT_FLAG_DISABLE_TIMING = 1 << 1,
  HRX_EVENT_FLAG_INTERPROCESS = 1 << 2,
} hrx_event_flags_t;

HRX_API hrx_status_t hrx_event_create(hrx_device_t device,
                                      hrx_event_flags_t flags,
                                      hrx_event_t* out_event);
HRX_API void hrx_event_retain(hrx_event_t event);
HRX_API void hrx_event_release(hrx_event_t event);
HRX_API hrx_status_t hrx_event_record(hrx_event_t event, hrx_stream_t stream);
HRX_API hrx_status_t hrx_event_query(hrx_event_t event, bool* complete);
HRX_API hrx_status_t hrx_event_synchronize(hrx_event_t event);
HRX_API hrx_status_t hrx_event_elapsed_time(hrx_event_t start, hrx_event_t stop,
                                            float* ms);
HRX_API hrx_status_t hrx_stream_wait_event(hrx_stream_t stream,
                                           hrx_event_t event);

//===----------------------------------------------------------------------===//
// Device memory/peer info
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_device_memory_info(hrx_device_t device,
                                            size_t* free_bytes,
                                            size_t* total_bytes);
HRX_API hrx_status_t hrx_device_can_access_peer(hrx_device_t device_a,
                                                hrx_device_t device_b,
                                                bool* can_access);

//===----------------------------------------------------------------------===//
// Host memory registration / buffer lookup
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_host_memory_register(hrx_device_t device,
                                              void* host_ptr, size_t size,
                                              uint32_t flags);
HRX_API hrx_status_t hrx_host_memory_unregister(hrx_device_t device,
                                                void* host_ptr);
HRX_API hrx_status_t hrx_buffer_lookup(hrx_device_t device,
                                       const void* device_ptr,
                                       hrx_buffer_t* buffer, size_t* offset);

//===----------------------------------------------------------------------===//
// Async host/device transfers
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_stream_copy_h2d(hrx_stream_t stream,
                                         const void* host_src, hrx_buffer_t dst,
                                         size_t dst_offset, size_t size);
HRX_API hrx_status_t hrx_stream_copy_d2h(hrx_stream_t stream, hrx_buffer_t src,
                                         size_t src_offset, void* host_dst,
                                         size_t size);

//===----------------------------------------------------------------------===//
// Memory pools (stream-ordered memory management)
//===----------------------------------------------------------------------===//

typedef enum hrx_mem_pool_attr_t {
  HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 0,
  HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 1,
  HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
  HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD = 3,
  HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT = 4,
  HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH = 5,
  HRX_MEM_POOL_ATTR_USED_MEM_CURRENT = 6,
  HRX_MEM_POOL_ATTR_USED_MEM_HIGH = 7,
} hrx_mem_pool_attr_t;

typedef struct hrx_mem_pool_props_t {
  uint32_t alloc_handle_type;
  uint32_t location_type;
  int location_id;
} hrx_mem_pool_props_t;

HRX_API hrx_status_t hrx_mem_pool_create(hrx_device_t device,
                                         const hrx_mem_pool_props_t* props,
                                         hrx_mem_pool_t* out_pool);
HRX_API void hrx_mem_pool_retain(hrx_mem_pool_t pool);
HRX_API void hrx_mem_pool_release(hrx_mem_pool_t pool);
HRX_API hrx_status_t hrx_mem_pool_get_attribute(hrx_mem_pool_t pool,
                                                hrx_mem_pool_attr_t attr,
                                                uint64_t* out_value);
HRX_API hrx_status_t hrx_mem_pool_set_attribute(hrx_mem_pool_t pool,
                                                hrx_mem_pool_attr_t attr,
                                                uint64_t value);
HRX_API hrx_status_t hrx_mem_pool_trim(hrx_mem_pool_t pool,
                                       size_t min_bytes_to_keep);

//===----------------------------------------------------------------------===//
// Graphs (CUDA/HIP-style execution graphs)
//===----------------------------------------------------------------------===//

typedef enum hrx_graph_node_type_t {
  HRX_GRAPH_NODE_EMPTY = 0,
  HRX_GRAPH_NODE_KERNEL = 1,
  HRX_GRAPH_NODE_MEMCPY = 2,
  HRX_GRAPH_NODE_MEMSET = 3,
  HRX_GRAPH_NODE_HOST_CALL = 4,
  HRX_GRAPH_NODE_GRAPH = 5,
} hrx_graph_node_type_t;

typedef struct hrx_graph_kernel_node_attrs_t {
  hrx_executable_t executable;
  uint32_t export_ordinal;
  hrx_dispatch_config_t config;
  const void* constants;
  size_t constants_size;
  const hrx_buffer_ref_t* bindings;
  size_t binding_count;
  uint32_t flags;
} hrx_graph_kernel_node_attrs_t;

typedef struct hrx_graph_memcpy_node_attrs_t {
  void* dst;
  const void* src;
  size_t size;
  uint32_t kind;
} hrx_graph_memcpy_node_attrs_t;

typedef struct hrx_graph_memset_node_attrs_t {
  void* dst;
  uint32_t value;
  size_t count;
} hrx_graph_memset_node_attrs_t;

typedef struct hrx_graph_host_call_node_attrs_t {
  hrx_host_call_fn_t fn;
  void* user_data;
} hrx_graph_host_call_node_attrs_t;

HRX_API hrx_status_t hrx_graph_create(hrx_device_t device, uint32_t flags,
                                      hrx_graph_t* out_graph);
HRX_API void hrx_graph_retain(hrx_graph_t graph);
HRX_API void hrx_graph_release(hrx_graph_t graph);

HRX_API hrx_status_t hrx_graph_add_empty_node(hrx_graph_t graph,
                                              const hrx_graph_node_t* deps,
                                              size_t dep_count,
                                              hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_kernel_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_kernel_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_memcpy_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_memcpy_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_memset_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_memset_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_host_call_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_host_call_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_dependencies(
    hrx_graph_t graph, const hrx_graph_node_t* from_nodes,
    const hrx_graph_node_t* to_nodes, size_t count);
HRX_API hrx_status_t hrx_graph_size(hrx_graph_t graph, size_t* out_count);
HRX_API hrx_status_t hrx_graph_get_nodes(hrx_graph_t graph,
                                         hrx_graph_node_t* nodes,
                                         size_t* inout_count);
HRX_API hrx_status_t hrx_graph_instantiate(hrx_graph_t graph, uint32_t flags,
                                           hrx_graph_exec_t* out_exec);
HRX_API void hrx_graph_exec_retain(hrx_graph_exec_t exec);
HRX_API void hrx_graph_exec_release(hrx_graph_exec_t exec);
HRX_API hrx_status_t hrx_graph_exec_launch(hrx_graph_exec_t exec,
                                           hrx_stream_t stream);
HRX_API hrx_status_t hrx_graph_exec_update(hrx_graph_exec_t exec,
                                           hrx_graph_t graph);

typedef enum hrx_capture_mode_t {
  HRX_CAPTURE_MODE_GLOBAL = 0,
  HRX_CAPTURE_MODE_THREAD_LOCAL = 1,
  HRX_CAPTURE_MODE_RELAXED = 2,
} hrx_capture_mode_t;

typedef enum hrx_capture_status_t {
  HRX_CAPTURE_STATUS_NONE = 0,
  HRX_CAPTURE_STATUS_ACTIVE = 1,
  HRX_CAPTURE_STATUS_INVALIDATED = 2,
} hrx_capture_status_t;

HRX_API hrx_status_t hrx_stream_begin_capture(hrx_stream_t stream,
                                              hrx_capture_mode_t mode);
HRX_API hrx_status_t hrx_stream_end_capture(hrx_stream_t stream,
                                            hrx_graph_t* out_graph);
HRX_API hrx_capture_status_t hrx_stream_capture_status(hrx_stream_t stream);
HRX_API bool hrx_stream_is_capturing(hrx_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // HRX_RUNTIME_H_
