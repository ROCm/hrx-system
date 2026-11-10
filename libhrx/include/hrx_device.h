// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_DEVICE_H_
#define HRX_DEVICE_H_

#include "hrx_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_runtime_s* hrx_runtime_t;
typedef struct hrx_device_group_s* hrx_device_group_t;
typedef struct hrx_device_s* hrx_device_t;

typedef enum hrx_accelerator_type_t {
  HRX_ACCELERATOR_GPU = 0,
  HRX_ACCELERATOR_CPU = 1,
  HRX_ACCELERATOR_NPU = 2,
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
  // Driver name such as "amd", "task", "vulkan", or "webgpu".
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

// Host services shared by all groups created from one runtime. Construction
// creates no devices, task workers, or default memory pools. Services are
// prepared on demand during device creation, never on queue submission.
typedef struct hrx_runtime_options_t {
  // Copied event receiver. NULL fn discards events. Callback state must outlive
  // the runtime and every retaining group; callbacks may run on driver threads.
  hrx_device_event_sink_t event_sink;
  // Configuration of the shared task executor, created by the first task
  // device.
  struct {
    // Number of worker threads; zero selects four workers.
    size_t worker_count;
    // Optional platform logical CPU IDs, one per worker. Copied at
    // construction. NULL leaves placement to the operating system.
    const uint32_t* cpu_ids;
    // Stack size per worker in bytes; zero selects 256 KiB.
    size_t worker_stack_size;
  } task;
} hrx_runtime_options_t;

HRX_API hrx_status_t hrx_runtime_create(const hrx_runtime_options_t* options,
                                        hrx_host_allocator_t host_allocator,
                                        hrx_runtime_t* out_runtime);
HRX_API void hrx_runtime_retain(hrx_runtime_t runtime);
// Final release synchronously destroys remaining runtime-owned services.
HRX_API void hrx_runtime_release(hrx_runtime_t runtime);

// One passively discovered device in a HAL URI scheme. Strings are copied into
// the same allocation as the containing info array.
typedef struct hrx_device_info_t {
  // NUL-terminated URI accepted by hrx_device_group_create.
  const char* uri;
  // NUL-terminated display name for selection interfaces.
  const char* name;
} hrx_device_info_t;

// Lists devices in |driver_name| (such as "task", "amdgpu", or "xdna") without
// creating HAL devices or task executors. Native discovery connections may be
// retained by the process. An empty inventory succeeds with zero entries.
//
// The returned array and its strings occupy one allocation, independent of the
// runtime lifetime. Free it with hrx_host_allocator_free(host_allocator,
// infos). Failure leaves |out_count| zero and |out_infos| NULL.
HRX_API hrx_status_t hrx_runtime_query_available_devices(
    hrx_runtime_t runtime, const char* driver_name,
    hrx_host_allocator_t host_allocator, size_t* out_count,
    hrx_device_info_t** out_infos);

// Completion observation policy for device-owned workers. Explicit host waits
// retain their own behavior. Backends choose the observation mechanism.
typedef uint32_t hrx_wait_flags_t;
#define HRX_WAIT_FLAG_NONE 0u
#define HRX_WAIT_FLAG_YIELD (1u << 0)
#define HRX_WAIT_FLAG_ACTIVE (1u << 1)

typedef struct hrx_device_create_params_t {
  // Borrowed, NUL-terminated HAL device URI, such as task:// or rdna://0.
  // Driver options and generic feedback/asan/tsan options use URI parameters.
  const char* uri;
  // Zero disables optional active completion observation. YIELD permits bounded
  // observation; ACTIVE prioritizes completion latency over CPU usage.
  hrx_wait_flags_t completion_wait_flags;
} hrx_device_create_params_t;

// Creates exactly the requested devices and seals their immutable topology.
// All inputs are borrowed for this call. Failure leaves |out_group| NULL and
// releases every partially created device. The group retains |runtime|.
HRX_API hrx_status_t hrx_device_group_create(
    hrx_runtime_t runtime, size_t device_count,
    const hrx_device_create_params_t* devices, hrx_device_group_t* out_group);
HRX_API void hrx_device_group_retain(hrx_device_group_t group);
// Final release synchronously destroys the group's devices. Applications must
// finish outstanding work and stop using borrowed handles before final release.
HRX_API void hrx_device_group_release(hrx_device_group_t group);
HRX_API size_t hrx_device_group_device_count(hrx_device_group_t group);
// Returns a borrowed device, or NULL for an out-of-range index. Retaining a
// device keeps its group and runtime alive; releasing the group does not
// invalidate an independently retained device or its retaining resources.
HRX_API hrx_device_t hrx_device_group_get_device(hrx_device_group_t group,
                                                 size_t index);

// Returns the borrowed sealed owner, or NULL for NULL. The device or any
// resource retaining it keeps the returned group alive.
HRX_API hrx_device_group_t hrx_device_get_device_group(hrx_device_t device);

// Immutable normalized topology. Node ordinals are stable for the group
// lifetime; UINT32_MAX denotes an absent parent/device, not an uncertain
// capability.
typedef enum hrx_topology_node_kind_t {
  HRX_TOPOLOGY_NODE_KIND_HOST_NUMA = 1,
  HRX_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE = 2,
  HRX_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE = 3,
  HRX_TOPOLOGY_NODE_KIND_MEMORY_DOMAIN = 4,
  HRX_TOPOLOGY_NODE_KIND_QUEUE_FAMILY = 5,
  HRX_TOPOLOGY_NODE_KIND_FABRIC_ENDPOINT = 6,
} hrx_topology_node_kind_t;

typedef struct hrx_topology_node_t {
  // Stable ordinal within this group's topology.
  uint32_t ordinal;
  // Parent node, or UINT32_MAX for a root.
  uint32_t parent_ordinal;
  // Group device index, or UINT32_MAX when not tied to a logical device.
  uint32_t device_ordinal;
  // Concrete object represented by this node.
  hrx_topology_node_kind_t kind;
  // Kind-specific coordinate: OS NUMA node, physical device, memory domain,
  // or queue family ordinal.
  uint32_t local_ordinal;
  // Physical-device subset within the associated logical device; zero when
  // not applicable. This is topology identity, not queue selection policy.
  uint64_t physical_device_affinity;
} hrx_topology_node_t;

HRX_API size_t hrx_device_group_topology_node_count(hrx_device_group_t group);
HRX_API hrx_status_t hrx_device_group_query_topology_node(
    hrx_device_group_t group, size_t ordinal, hrx_topology_node_t* out_node);

//===----------------------------------------------------------------------===//
// Device operations (generic, works for any accelerator type)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_device_get_property(hrx_device_t device,
                                             hrx_device_property_t prop,
                                             void* value, size_t value_size);

HRX_API hrx_status_t hrx_device_get_type(hrx_device_t device,
                                         hrx_accelerator_type_t* type);

HRX_API void hrx_device_retain(hrx_device_t device);
HRX_API void hrx_device_release(hrx_device_t device);

#ifdef __cplusplus
}
#endif

#endif  // HRX_DEVICE_H_
