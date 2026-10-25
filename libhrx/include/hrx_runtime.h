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
// Values match iree_status_code_t. Verified by static_assert in
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
// All values match their IREE counterparts. Verified by static_assert
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
#define HRX_MEMORY_TYPE_HOST_LOCAL 0x00000042u
#define HRX_MEMORY_TYPE_DEVICE_VISIBLE 0x00000010u
#define HRX_MEMORY_TYPE_DEVICE_LOCAL 0x00000030u
#define HRX_MEMORY_TYPE_DEVICE_UNCACHED 0x00000080u

// Memory access bitfield. Values match iree_hal_memory_access_t.
typedef uint16_t hrx_memory_access_t;
#define HRX_MEMORY_ACCESS_NONE 0x00
#define HRX_MEMORY_ACCESS_READ 0x01
#define HRX_MEMORY_ACCESS_WRITE 0x02
#define HRX_MEMORY_ACCESS_ALL 0x03

// Buffer usage bitfield. Transfer, storage, and host-mapping values match HAL.
typedef uint32_t hrx_buffer_usage_t;
#define HRX_BUFFER_USAGE_NONE 0x00000000u
#define HRX_BUFFER_USAGE_TRANSFER_SOURCE 0x00000001u
#define HRX_BUFFER_USAGE_TRANSFER_TARGET 0x00000002u
#define HRX_BUFFER_USAGE_TRANSFER 0x00000003u
#define HRX_BUFFER_USAGE_STORAGE_READ 0x00000400u
#define HRX_BUFFER_USAGE_STORAGE_WRITE 0x00000800u
#define HRX_BUFFER_USAGE_STORAGE 0x00000C00u
#define HRX_BUFFER_USAGE_MAPPING_SCOPED 0x01000000u
#define HRX_BUFFER_USAGE_MAPPING_PERSISTENT 0x02000000u
// Owned backing prepared for explicit virtual-address mappings throughout the
// device's sealed group. The ordinary buffer address remains usable. Source
// construction fails if that complete access set is unsupported.
// Callers retain the backing until every alias is detached and all accesses
// have completed. Host-pointer import cannot provide this capability.
#define HRX_BUFFER_USAGE_MAPPING_SOURCE 0x04000000u
#define HRX_BUFFER_USAGE_DEFAULT 0x00000C03u

// Operation flags for hrx_buffer_map. READ/WRITE are permissions; DISCARD
// and MAY_ALIAS apply only to the requested mapping.
typedef uint16_t hrx_map_flags_t;
#define HRX_MAP_READ HRX_MEMORY_ACCESS_READ
#define HRX_MAP_WRITE HRX_MEMORY_ACCESS_WRITE
// Prior contents of the mapped range may be discarded. Implies WRITE.
#define HRX_MAP_DISCARD 0x04
// Mapped accesses may alias other concurrent accesses; supplies no ordering.
#define HRX_MAP_MAY_ALIAS 0x08

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

// Borrowed string storage valid only for the duration documented by the
// containing API.
typedef struct hrx_string_view_t {
  // Borrowed string data; not necessarily NUL terminated.
  const char* data;
  // String length in bytes.
  size_t size;
} hrx_string_view_t;

// Borrowed byte storage valid only for the duration documented by the
// containing API.
typedef struct hrx_const_byte_span_t {
  // Borrowed byte data, or NULL when |data_length| is zero.
  const void* data;
  // Byte length of |data|.
  size_t data_length;
} hrx_const_byte_span_t;

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
// Device events
//===----------------------------------------------------------------------===//

#define HRX_DEVICE_EVENT_ABI_VERSION_0 0u
#define HRX_DEVICE_ASAN_REPORT_ABI_VERSION_0 0u
#define HRX_DEVICE_PRINTF_EVENT_ABI_VERSION_0 0u

typedef uint32_t hrx_device_event_type_t;
enum hrx_device_event_type_bits_t {
  HRX_DEVICE_EVENT_TYPE_NONE = 0u,
  HRX_DEVICE_EVENT_TYPE_DRIVER_FAILURE = 1u,
  HRX_DEVICE_EVENT_TYPE_ASAN_REPORT = 2u,
  HRX_DEVICE_EVENT_TYPE_UBSAN_REPORT = 3u,
  HRX_DEVICE_EVENT_TYPE_PRINTF = 4u,
  HRX_DEVICE_EVENT_TYPE_HOST_CALL = 5u,
  HRX_DEVICE_EVENT_TYPE_USER = 0x8000u,
};

typedef uint32_t hrx_device_event_severity_t;
enum hrx_device_event_severity_bits_t {
  HRX_DEVICE_EVENT_SEVERITY_TRACE = 0u,
  HRX_DEVICE_EVENT_SEVERITY_INFO = 1u,
  HRX_DEVICE_EVENT_SEVERITY_WARNING = 2u,
  HRX_DEVICE_EVENT_SEVERITY_ERROR = 3u,
  HRX_DEVICE_EVENT_SEVERITY_FATAL = 4u,
};

typedef uint64_t hrx_device_event_flags_t;
enum hrx_device_event_flag_bits_t {
  HRX_DEVICE_EVENT_FLAG_NONE = 0u,
};

// Source attribution for a device event.
typedef struct hrx_device_event_source_t {
  // Stable device identifier string when available.
  hrx_string_view_t device_id;
  // Driver name such as "legacy-amdgpu", "task", "vulkan", or "webgpu".
  hrx_string_view_t driver_id;
  // Physical device ordinal, or UINT32_MAX when not applicable.
  uint32_t physical_device_ordinal;
  // Queue ordinal, or UINT32_MAX when not applicable.
  uint32_t queue_ordinal;
  // Backend-assigned executable identifier, or 0 when not applicable.
  uint64_t executable_id;
  // Executable export ordinal, or UINT32_MAX when not applicable.
  uint32_t export_ordinal;
} hrx_device_event_source_t;

// Device event envelope passed to hrx_device_event_sink_t callbacks.
//
// Events and referenced payload storage are borrowed and valid only for the
// duration of the sink callback. Sinks retaining event data must copy it.
typedef struct hrx_device_event_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this event envelope.
  uint32_t abi_version;
  // Concrete event family.
  hrx_device_event_type_t type;
  // Producer-assigned severity.
  hrx_device_event_severity_t severity;
  // Event-family flags.
  hrx_device_event_flags_t flags;
  // Monotonic producer sequence when available, or 0.
  uint64_t sequence;
  // Host timestamp in nanoseconds when available, or 0.
  uint64_t host_time_ns;
  // Source attribution for the event producer.
  hrx_device_event_source_t source;
  // Event-family payload selected by |type|.
  hrx_const_byte_span_t payload;
  // Optional backend-native payload for advanced tools.
  hrx_const_byte_span_t implementation_payload;
} hrx_device_event_t;

// Device printf stream classification.
typedef uint32_t hrx_device_printf_stream_t;
enum hrx_device_printf_stream_bits_t {
  HRX_DEVICE_PRINTF_STREAM_DEFAULT = 0u,
  HRX_DEVICE_PRINTF_STREAM_STDOUT = 1u,
  HRX_DEVICE_PRINTF_STREAM_STDERR = 2u,
};

typedef uint32_t hrx_device_printf_flags_t;
enum hrx_device_printf_flag_bits_t {
  HRX_DEVICE_PRINTF_FLAG_NONE = 0u,
};

// Device printf payload.
typedef struct hrx_device_printf_event_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this printf payload.
  uint32_t abi_version;
  // Stream hint for the output.
  hrx_device_printf_stream_t stream;
  // Printf event flags.
  hrx_device_printf_flags_t flags;
  // Format string identifier, or 0 when |text| is already formatted.
  uint64_t format_id;
  // Borrowed already-formatted text when available.
  hrx_string_view_t text;
  // Borrowed encoded arguments when available.
  hrx_const_byte_span_t arguments;
} hrx_device_printf_event_t;

typedef uint32_t hrx_device_asan_access_kind_t;
enum hrx_device_asan_access_kind_bits_t {
  HRX_DEVICE_ASAN_ACCESS_KIND_UNKNOWN = 0u,
  HRX_DEVICE_ASAN_ACCESS_KIND_READ = 1u,
  HRX_DEVICE_ASAN_ACCESS_KIND_WRITE = 2u,
  HRX_DEVICE_ASAN_ACCESS_KIND_ATOMIC = 3u,
};

typedef uint32_t hrx_device_asan_report_flags_t;
enum hrx_device_asan_report_flag_bits_t {
  HRX_DEVICE_ASAN_REPORT_FLAG_NONE = 0u,
};

// Address-sanitizer report payload.
typedef struct hrx_device_asan_report_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this report payload.
  uint32_t abi_version;
  // Instrumented access kind that triggered the report.
  hrx_device_asan_access_kind_t access_kind;
  // ASAN report flags.
  hrx_device_asan_report_flags_t flags;
  // Application address that failed the ASAN check.
  uint64_t fault_address;
  // Access size in bytes.
  uint64_t access_length;
  // Compiler-assigned instrumentation site identifier.
  uint64_t site_id;
  // Shadow address consulted by the check, or 0 when unavailable.
  uint64_t shadow_address;
  // Shadow value observed by the check, or 0 when unavailable.
  uint64_t shadow_value;
  // Workgroup id that produced the report.
  uint32_t workgroup_id[3];
  // Workitem id that produced the report.
  uint32_t workitem_id[3];
  // Device-visible dispatch packet pointer, or 0 when unavailable.
  uint64_t source_dispatch_ptr;
} hrx_device_asan_report_t;

typedef void (*hrx_device_event_sink_fn_t)(void* user_data,
                                           const hrx_device_event_t* event);

// Value-type event sink copied into HRX devices at creation time.
typedef struct hrx_device_event_sink_t {
  // Callback receiving one complete event.
  hrx_device_event_sink_fn_t fn;
  // Opaque application-owned callback data.
  void* user_data;
} hrx_device_event_sink_t;

static inline void hrx_device_event_sink_discard_callback(
    void* user_data, const hrx_device_event_t* event) {
  (void)user_data;
  (void)event;
}

static inline hrx_device_event_sink_t hrx_device_event_sink_discard(void) {
  hrx_device_event_sink_t sink;
  sink.fn = hrx_device_event_sink_discard_callback;
  sink.user_data = NULL;
  return sink;
}

// Sets the process-wide sink used by subsequently created HRX devices.
//
// The sink must be set before any accelerator is initialized. Device-event
// callbacks may run on driver-owned service, callback, or completion threads.
// Sink implementations must not call back into the originating device or
// submit/wait/destroy work from the callback; slow processing should copy the
// event into application-owned storage and return.
HRX_API hrx_status_t
hrx_runtime_set_device_event_sink(hrx_device_event_sink_t sink);

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
// Use when out-of-band work (e.g. an external queue submission) will
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

// Returns the whole-buffer native device address. This does not map the buffer
// or alter an active scoped mapping. HRX allocates and imports CPU storage with
// persistent mapping permission to expose its address. A buffer wrapped from an
// external HAL caller must already permit persistent mapping on CPU storage.
// The caller retains the buffer while using the address. On failure the output
// is NULL and the native export error is returned.
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
// Callers pass the executable artifact target family and key. The runtime
// selects a compatible advertised device target using family-owned rules.
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_executable_load_data(hrx_device_t device,
                                              const void* executable_data,
                                              size_t executable_data_size,
                                              const char* target_family,
                                              const char* target_key,
                                              hrx_executable_t* executable);

HRX_API hrx_status_t hrx_executable_load_file(hrx_device_t device,
                                              const char* path,
                                              const char* target_family,
                                              const char* target_key,
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
// Virtual buffers and mapping
//===----------------------------------------------------------------------===//

// Access site selected within a device's sealed group.
typedef enum hrx_memory_site_t {
  HRX_MEMORY_SITE_HOST = 0,
  HRX_MEMORY_SITE_QUEUE = 1,
  HRX_MEMORY_SITE_PROGRAM = 2,
} hrx_memory_site_t;

// Borrowed process-local scope token. Its device group must remain live.
// A zero-initialized token selects every admitted reservation projection.
typedef struct hrx_memory_scope_t {
  // Opaque group identity; never dereferenced by callers.
  const void* domain;
  // Opaque coordinate within the group.
  uint32_t id;
} hrx_memory_scope_t;

// Resolves a scope once for reuse in mapping descriptors. QUEUE and PROGRAM
// select a provisioned transfer or dispatch queue using |affinity|. HOST
// ignores affinity and selects the group's public host site.
HRX_API hrx_status_t hrx_device_query_memory_scope(
    hrx_device_t device, hrx_queue_affinity_t affinity, hrx_memory_site_t site,
    hrx_memory_scope_t* out_scope);

// Complete simultaneous access to one address reservation. All families of
// every listed device participate. Devices must belong to the same sealed
// group; the arrays are borrowed only for query/construction.
typedef struct hrx_virtual_buffer_scope_t {
  // Nonzero number of distinct participating devices.
  size_t device_count;
  // Devices whose first entry supplies the stable buffer's device identity.
  const hrx_device_t* devices;
  // Required public host mapping access, or NONE for execution-only addresses.
  hrx_memory_access_t host_access;
} hrx_virtual_buffer_scope_t;

// Caller-declared prior state. No installed-source table is maintained by HRX.
typedef enum hrx_memory_mapping_operation_t {
  // Replaces the interval, including holes; a NULL source clears it.
  HRX_MEMORY_MAPPING_OPERATION_REPLACE = 0,
  // Binds a non-NULL source into an interval the caller knows is unbound.
  HRX_MEMORY_MAPPING_OPERATION_BIND = 1,
  // Removes a complete binding; requires NULL source and NONE access.
  HRX_MEMORY_MAPPING_OPERATION_UNBIND = 2,
  // Changes permissions, supplying the currently installed source and offset.
  HRX_MEMORY_MAPPING_OPERATION_PROTECT = 3,
} hrx_memory_mapping_operation_t;

typedef struct hrx_virtual_buffer_capabilities_t {
  // Largest address interval in bytes; zero means the scope is unsupported.
  size_t max_byte_length;
  // Required reservation length multiple and minimum base alignment.
  size_t reservation_granularity;
  // Smallest independently replaceable destination offset/length unit.
  size_t mapping_granularity;
  // Largest supported power-of-two minimum base alignment.
  size_t max_alignment;
  // Supported operations, with bit N corresponding to mapping operation N.
  uint32_t operations;
} hrx_virtual_buffer_capabilities_t;

// Cold query: acquires no addresses or physical payload. Unsupported scopes
// return OK with zero capabilities; invalid scope and native failures
// propagate.
HRX_API hrx_status_t hrx_virtual_buffer_query_capabilities(
    hrx_virtual_buffer_scope_t scope,
    hrx_virtual_buffer_capabilities_t* out_capabilities);

typedef struct hrx_virtual_buffer_options_t {
  // Nonzero virtual buffer length in bytes.
  size_t byte_length;
  // Minimum power-of-two base alignment; zero selects the native minimum.
  size_t min_alignment;
  // Minimum power-of-two mapping unit; zero selects the native minimum.
  size_t mapping_granularity;
} hrx_virtual_buffer_options_t;

// Reserves addresses without physical payload and returns one owned ordinary
// buffer reference. On failure the output is NULL. Buffer references and views
// keep addresses live; installed backing remains independently caller-owned.
HRX_API hrx_status_t hrx_virtual_buffer_create(
    hrx_virtual_buffer_scope_t scope,
    const hrx_virtual_buffer_options_t* options, hrx_buffer_t* out_buffer);

typedef struct hrx_virtual_buffer_info_t {
  // Visible byte zero in the native target's mapping coordinates, in bytes.
  size_t mapping_byte_offset;
  // Achieved independently replaceable mapping unit, in bytes.
  size_t mapping_granularity;
  // Supported operations, with bit N corresponding to mapping operation N.
  uint32_t operations;
  // Number of native page-table projections owned by the root buffer.
  uint32_t projection_count;
} hrx_virtual_buffer_info_t;

// Queries achieved geometry after any asynchronous allocation has committed.
HRX_API hrx_status_t hrx_virtual_buffer_query_info(
    hrx_buffer_t buffer, hrx_virtual_buffer_info_t* out_info);

// Resolves one concrete scope to a native page-table projection. Equal IDs on
// the same native root share mapping state. An all-sites scope is not concrete.
HRX_API hrx_status_t hrx_virtual_buffer_resolve_projection(
    hrx_buffer_t buffer, hrx_memory_scope_t scope, uint32_t* out_projection);

// Removes all mappings at the selected scope without releasing addresses. The
// buffer must cover the complete native target. The caller has completed all
// affected execution and accepted mapping work and serializes mutation. This
// cold call may wait for native removal, including after partial mapping
// failure. It preserves any earlier failure; only success certifies backing
// reuse, subject to other aliases and users. A zero scope selects all admitted
// sites.
HRX_API hrx_status_t hrx_virtual_buffer_detach(hrx_buffer_t buffer,
                                               hrx_memory_scope_t scope);

typedef uint32_t hrx_memory_mapping_source_flags_t;
// Each binding must cover the entire native allocation at source offset zero.
// Suballocating such storage does not make its pieces independently mappable.
#define HRX_MEMORY_MAPPING_SOURCE_FLAG_WHOLE_ALLOCATION (1u << 0)

typedef struct hrx_memory_mapping_requirements_t {
  // Required native source offset/length granule; zero means no prepared
  // source.
  size_t minimum_granularity;
  // Preferred physical allocation multiple, at least the minimum granularity.
  size_t recommended_granularity;
  // Additional native restrictions, including whole-allocation-only mappings.
  hrx_memory_mapping_source_flags_t flags;
} hrx_memory_mapping_requirements_t;

// Queries the ordinary MAPPING_SOURCE allocation class without allocating
// payload. Use its granularity with allocator or stream buffer allocation.
// Native source construction/query failures propagate, including unsupported
// memory classes.
HRX_API hrx_status_t hrx_allocator_query_mapping_requirements(
    hrx_allocator_t allocator, hrx_memory_type_t memory_type,
    hrx_memory_mapping_requirements_t* out_requirements);

typedef struct hrx_memory_mapping_t {
  // Borrowed virtual target buffer or view.
  hrx_buffer_t target;
  // Prepared scope token; zero selects all admitted projections.
  hrx_memory_scope_t scope;
  // Byte offset within the stable reservation view.
  size_t target_offset;
  // Nonzero mapping length satisfying both source and destination granularity.
  size_t byte_length;
  // Ordinary prepared backing, or NULL for removal.
  hrx_buffer_t source;
  // Byte offset within source; zero when source is NULL.
  size_t source_offset;
  // Required permissions, including NONE for revocation or an inaccessible
  // bind.
  hrx_memory_access_t access;
  // Caller-declared operation and prior binding state.
  hrx_memory_mapping_operation_t operation;
} hrx_memory_mapping_t;

typedef uint64_t hrx_queue_map_flags_t;
#define HRX_QUEUE_MAP_FLAG_NONE 0ull
#define HRX_QUEUE_MAP_FLAG_BORROW_BUFFERS (1ull << 0)

// Submits the entire descriptor vector with the supplied dependencies. Arrays
// are captured before returning. Native changes complete before signals become
// satisfied; queue submission order alone does not establish ordering. An empty
// batch is a dependency edge. Backing must remain live while installed.
// BORROW_BUFFERS also omits the optional HAL wrapper retention through
// completion. Captured target references keep their address range live; neither
// mode retains backing after completion or extends an ended allocation epoch.
HRX_API hrx_status_t hrx_queue_map(
    hrx_device_t device, hrx_queue_affinity_t affinity,
    const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, size_t mapping_count,
    const hrx_memory_mapping_t* mappings, hrx_queue_map_flags_t flags);

//===----------------------------------------------------------------------===//
// Legacy scalar virtual memory (allocator methods)
//===----------------------------------------------------------------------===//

// Deprecated compatibility surface for the ROCr-backed HIP implementation.
// New callers use virtual buffers, ordinary MAPPING_SOURCE
// buffers, and hrx_queue_map. Scalar physical handles are initially unmapped
// and belong to the allocator that created them. They cannot be used as
// queue_map sources.

// Queries scalar support and optional granularity outputs in bytes. Unsupported
// allocators return OK, false, and zero granularity; native query errors
// propagate.
HRX_API hrx_status_t hrx_allocator_query_virtual_memory(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, bool* supported,
    size_t* min_page_size, size_t* recommended_page_size);

// Reserves addresses without physical backing. The caller owns the returned
// buffer and must release it through virtual_memory_release, not
// buffer_release.
HRX_API hrx_status_t hrx_allocator_virtual_memory_reserve(
    hrx_allocator_t allocator, hrx_queue_affinity_t affinity, size_t size,
    hrx_buffer_t* virtual_buffer);

// Consumes the reservation only on success. Failure leaves it owned by the
// caller so mappings can be removed and release retried. All uses must complete
// and every mapping and additional buffer reference must be released first.
HRX_API hrx_status_t hrx_allocator_virtual_memory_release(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer);

// Creates an initially unmapped physical handle owned by the caller. Native
// granularity and whole-allocation mapping restrictions still apply.
HRX_API hrx_status_t hrx_allocator_physical_memory_allocate(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, size_t size,
    hrx_physical_memory_t* physical);

// Consumes the handle only on success. The caller must first remove all
// mappings and complete all uses; failure retains ownership for retry.
HRX_API hrx_status_t hrx_allocator_physical_memory_free(
    hrx_allocator_t allocator, hrx_physical_memory_t physical);

// Installs a physical mapping without granting access. Both handles remain
// caller-owned, and physical storage must remain live until the mapping is
// gone.
HRX_API hrx_status_t hrx_allocator_virtual_memory_map(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, hrx_physical_memory_t physical,
    size_t physical_offset, size_t size);

// Removes a mapped interval after all of its users have completed.
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

typedef enum hrx_virtual_memory_access_scope_t {
  HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL = 0,
  HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST = 1,
  HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE = 2,
} hrx_virtual_memory_access_scope_t;

// Changes access for the allocator's device scope after conflicting uses
// finish.
HRX_API hrx_status_t hrx_allocator_virtual_memory_protect(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, hrx_memory_protection_t protection);

// Changes access for the specified host/device scope. The allocator selects the
// device receiving access, which may differ from the reservation's allocator.
HRX_API hrx_status_t hrx_allocator_virtual_memory_protect_scoped(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size,
    hrx_virtual_memory_access_scope_t access_scope,
    hrx_memory_protection_t protection);

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
// Returns a borrowed buffer and byte offset for a registered address. The
// caller must keep the registration live during lookup and retain the returned
// buffer explicitly if its metadata must survive unregistering the address.
HRX_API hrx_status_t hrx_buffer_lookup(hrx_device_t device,
                                       const void* device_ptr,
                                       hrx_buffer_t* buffer, size_t* offset);

//===----------------------------------------------------------------------===//
// Async host/device transfers
//===----------------------------------------------------------------------===//

// Enqueues a host-to-device transfer after prior work in |stream|. |host_src|
// remains live and unmodified until the stream reaches the transfer's timeline
// point. A NULL |stream| performs the transfer synchronously.
HRX_API hrx_status_t hrx_stream_copy_h2d(hrx_stream_t stream,
                                         const void* host_src, hrx_buffer_t dst,
                                         size_t dst_offset, size_t size);

// Enqueues a device-to-host transfer after prior work in |stream|. |host_dst|
// remains live and inaccessible until the stream reaches the transfer's
// timeline point. A NULL |stream| performs the transfer synchronously.
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
  // Retained-byte floor contributed to the device's shared backing cache.
  // Compatible pools share storage, so their floors combine by maximum.
  HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD = 3,
  // Bytes borrowed by this pool, including padding and fragmentation. Empty
  // slabs return to the device cache and cease to be charged to this pool;
  // shared idle storage is not counted again for every potential borrower.
  HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT = 4,
  // Peak bytes borrowed by this pool. Writing zero resets the watermark.
  HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH = 5,
  HRX_MEM_POOL_ATTR_USED_MEM_CURRENT = 6,
  HRX_MEM_POOL_ATTR_USED_MEM_HIGH = 7,
} hrx_mem_pool_attr_t;

typedef struct hrx_mem_pool_props_t {
  // Platform allocation-handle types requested when the pool was created.
  uint32_t alloc_handle_type;
  // Platform location type for the pool's backing storage.
  uint32_t location_type;
  // Platform device ordinal for the pool's backing storage.
  int location_id;
  // Maximum live reservation bytes, or zero when the pool is unbounded.
  size_t max_size;
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
// Returns unused child backing and trims the shared device cache, preserving
// |min_bytes_to_keep| and every other pool's current retention request. This
// replaces this pool's floor until an attribute update or automatic release
// applies its configured threshold again. It does not allocate missing bytes.
HRX_API hrx_status_t hrx_mem_pool_trim(hrx_mem_pool_t pool,
                                       size_t min_bytes_to_keep);

// Returns unused child backing and applies the configured release threshold to
// the device's shared cache, preserving other pools' retention requests. This
// is intended for stream-ordered free completion paths. Reclamation runs on
// the memory owner without waiting here; statistics converge after queued
// whole-slab returns and the ordered cache sweep complete.
HRX_API hrx_status_t hrx_mem_pool_release_unused(hrx_mem_pool_t pool);

// Records a logical allocation backed by |pool|. Logical usage is distinct
// from physical TLSF reservations, which can remain retained for later
// stream-ordered reuse.
HRX_API void hrx_mem_pool_record_logical_allocation(hrx_mem_pool_t pool,
                                                    size_t size);

// Records retirement of a logical allocation backed by |pool|. The physical
// reservation remains owned by its buffer until that buffer is released or
// reused.
HRX_API void hrx_mem_pool_record_logical_free(hrx_mem_pool_t pool, size_t size);

// Allocates a buffer with at least 256-byte alignment and the device access
// permissions captured by |pool|. The returned buffer owns the allocation and
// remains valid independently of the public |pool| handle.
HRX_API hrx_status_t hrx_mem_pool_allocate_buffer(hrx_mem_pool_t pool,
                                                  size_t size,
                                                  hrx_buffer_t* buffer);

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

// Raw device-pointer buffer copy. Recording resolves the pointers through the
// device allocation table and incurs pointer accounting overhead. Prefer
// hrx_graph_add_copy_buffer_node() when buffer handles are available.
typedef struct hrx_graph_copy_ptr_node_attrs_t {
  void* dst;
  const void* src;
  size_t size;
  uint32_t kind;
} hrx_graph_copy_ptr_node_attrs_t;

// Raw device-pointer buffer fill. Recording resolves the pointer through the
// device allocation table and incurs pointer accounting overhead. Prefer
// hrx_graph_add_fill_buffer_node() when a buffer handle is available.
typedef struct hrx_graph_fill_ptr_node_attrs_t {
  void* dst;
  uint32_t value;
  size_t count;
} hrx_graph_fill_ptr_node_attrs_t;

// Native handle-based buffer copy. Source and destination lengths must be
// equal and nonzero. Recording uses the buffer handles directly without raw
// pointer accounting overhead. The ranges are retained by the instantiated
// graph through the underlying command buffer resources.
typedef struct hrx_graph_copy_buffer_node_attrs_t {
  hrx_buffer_ref_t src;
  hrx_buffer_ref_t dst;
} hrx_graph_copy_buffer_node_attrs_t;

// Native handle-based buffer fill. |pattern_size| must be 1, 2, or 4 and the
// destination offset and length must be integral multiples of it. Pattern
// bytes are copied into the graph node during recording. Recording uses the
// buffer handle directly without raw pointer accounting overhead.
typedef struct hrx_graph_fill_buffer_node_attrs_t {
  hrx_buffer_ref_t dst;
  uint32_t pattern;
  size_t pattern_size;
} hrx_graph_fill_buffer_node_attrs_t;

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
HRX_API hrx_status_t hrx_graph_add_copy_ptr_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_copy_ptr_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_fill_ptr_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_fill_ptr_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_copy_buffer_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_copy_buffer_node_attrs_t* attrs,
    hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_fill_buffer_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_fill_buffer_node_attrs_t* attrs,
    hrx_graph_node_t* out_node);
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
