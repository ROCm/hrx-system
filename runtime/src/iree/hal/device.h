// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_H_
#define IREE_HAL_DEVICE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/time.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/channel.h"
#include "iree/hal/channel_provider.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/device_event.h"
#include "iree/hal/device_spec.h"
#include "iree/hal/event.h"
#include "iree/hal/fence.h"
#include "iree/hal/file.h"
#include "iree/hal/memory/asan.h"
#include "iree/hal/pool.h"
#include "iree/hal/profile_options.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// An opaque driver-specific handle to identify different devices.
typedef uintptr_t iree_hal_device_id_t;

#define IREE_HAL_DEVICE_ID_DEFAULT 0ull

// Describes features supported by a device.
// These flags indicate the availability of features that may be enabled at the
// request of the calling application. Note that certain features may disable
// runtime optimizations or require compilation flags to ensure the required
// metadata is present in executables.
typedef uint64_t iree_hal_device_feature_t;
enum iree_hal_device_feature_bits_t {
  IREE_HAL_DEVICE_FEATURE_NONE = 0u,

  // Device supports executable debugging.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_LOAD_FLAG_ENABLE_DEBUGGING and will have usable
  // debugging related methods. Note that if the input executables do not have
  // embedded debugging information they still may not be able to perform
  // disassembly or fine-grained breakpoint insertion.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING = 1u << 0,

  // Device supports executable coverage information.
  // When present instrumented executables may produce coverage buffers during
  // dispatch. Input executables must have partial embedded debug information
  // to allow mapping back to source offsets.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE = 1u << 1,

  // Device supports executable and command queue profiling.
  // When present executables and queue operations may produce profiling data.
  // Input executables may require partial embedded debug information to map
  // profiling results back to source offsets.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING = 1u << 2,
};

// Runtime services requested when creating a HAL device.
typedef uint64_t iree_hal_device_runtime_feature_flags_t;
enum iree_hal_device_runtime_feature_flag_bits_e {
  // No optional runtime services are requested.
  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE = 0ull,
  // Device-originated feedback events may be emitted by executables.
  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK = 1ull << 0,
  // Address sanitizer support may be required by executables.
  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN = 1ull << 1,
  // Thread/race sanitizer support may be required by executables.
  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN = 1ull << 2,
};

#define IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAGS_KNOWN                                          \
  ((iree_hal_device_runtime_feature_flags_t)(IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK | \
                                             IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN |     \
                                             IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN))

// Describes an enumerated HAL device.
typedef struct iree_hal_device_info_t {
  // Opaque handle used by drivers. Not valid across driver instances.
  iree_hal_device_id_t device_id;
  // Stable driver-specific path used to reference the device.
  iree_string_view_t path;
  // Human-readable name of the device as returned by the API.
  iree_string_view_t name;
} iree_hal_device_info_t;

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_async_notification_t iree_async_notification_t;
typedef struct iree_hal_slab_provider_t iree_hal_slab_provider_t;

// Common prefix for driver-specific device creation parameter extensions.
//
// Drivers walk the chain by interpreting each extension through this prefix.
// Extension type values are scoped to the driver consuming the chain.
typedef struct iree_hal_device_create_params_extension_t {
  // Driver-specific extension type identifier.
  uint32_t type;

  // Next driver-specific device creation parameter extension, or NULL.
  const void* next;
} iree_hal_device_create_params_extension_t;

// Parameters for device creation that apply across all HAL drivers.
//
// Callers stack-allocate and initialize with
// iree_hal_device_create_params_default(), then customize fields as needed
// before passing to device creation functions. All creation paths require a
// valid pointer — callers must always provide one.
//
// The |next| pointer enables a Vulkan-style extension chain: drivers may define
// their own params structs that chain off this base struct. Each extension must
// begin with iree_hal_device_create_params_extension_t. Unrecognized extensions
// are silently skipped for forward compatibility.
typedef struct iree_hal_device_create_params_t {
  IREE_API_UNSTABLE

  // Extension chain pointer for driver-specific parameters, or NULL.
  const void* next;

  // Proactor pool for async I/O. Drivers select a proactor from this pool
  // based on their NUMA affinity during device creation. The device retains
  // the pool to ensure proactor threads outlive the device.
  // Callers must always provide a valid pool.
  iree_async_proactor_pool_t* proactor_pool;

  // Programmatic sink receiving device-originated events. Defaults to discard.
  // The sink is copied into the device and |event_sink.user_data| must outlive
  // the device.
  iree_hal_device_event_sink_t event_sink;

  // Runtime services requested for executables loaded into the device.
  iree_hal_device_runtime_feature_flags_t runtime_features;

} iree_hal_device_create_params_t;

// Returns default device creation parameters with a discard event sink.
static inline iree_hal_device_create_params_t
iree_hal_device_create_params_default(void) {
  iree_hal_device_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.event_sink = iree_hal_device_event_sink_discard();
  return params;
}

// Verifies that |params| contains all device-creation inputs required by every
// HAL backend.
IREE_API_EXPORT iree_status_t iree_hal_device_create_params_verify(
    const iree_hal_device_create_params_t* params);

// Bitfield selecting external capture behavior.
typedef uint64_t iree_hal_device_external_capture_flags_t;
enum iree_hal_device_external_capture_flag_bits_t {
  IREE_HAL_DEVICE_EXTERNAL_CAPTURE_FLAG_NONE = 0u,
};

// Controls an external profiler/tool capture range.
//
// External capture controls tools that produce non-IREE artifacts such as
// RenderDoc .rdc captures, Metal .gputrace documents, vendor profiler UI
// sessions, or future CUPTI/PIX/RGP-style captures. It is separate from
// HAL-native profiling: success does not imply an iree_hal_profile_sink_t
// session, and native profiling success must not depend on external tools.
typedef struct iree_hal_device_external_capture_options_t {
  // External capture provider/tool id, such as "renderdoc" or "metal".
  iree_string_view_t provider;

  // Optional provider-specific output path or path template.
  iree_string_view_t file_path;

  // Optional human-readable range label for providers with named captures.
  iree_string_view_t label;

  // External capture behavior flags.
  iree_hal_device_external_capture_flags_t flags;
} iree_hal_device_external_capture_options_t;

// Bitfield specifying flags controlling an async allocation operation.
typedef uint64_t iree_hal_alloca_flags_t;
enum iree_hal_alloca_flag_bits_t {
  IREE_HAL_ALLOCA_FLAG_NONE = 0,

  // Buffer lifetime is indeterminate indicating that the compiler or
  // application allocating the buffer is unable to determine when it is safe to
  // deallocate the buffer. Explicit deallocation requests are ignored and the
  // buffer deallocation will happen synchronously when the last remaining
  // reference to the buffer is released.
  IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME = 1ull << 0,

  // Allows the queue to satisfy the allocation from recycled pool memory whose
  // death frontier is not dominated by the requester's current queue frontier
  // by inserting an internal dependency on that death frontier before the
  // buffer's bytes are used. Without this flag, pools must return only
  // immediately-usable reservations to queue_alloca.
  IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER = 1ull << 1,
};

// Bitfield specifying flags controlling an async deallocation operation.
typedef uint64_t iree_hal_dealloca_flags_t;
enum iree_hal_dealloca_flag_bits_t {
  IREE_HAL_DEALLOCA_FLAG_NONE = 0,

  // The provided device and queue affinity will be overridden with the origin
  // of the allocation as defined by the placement, if available.
  // If the buffer has no origin device (imported heap buffers and other rare
  // cases) or was not allocated asynchronously the provided device and queue
  // affinity will be used to insert a queue barrier. Callers must ensure the
  // provided device is compatible with the fences provided.
  IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN = 1ull << 0,
};

// Bitfield specifying flags controlling a file read operation.
typedef uint64_t iree_hal_read_flags_t;
enum iree_hal_read_flag_bits_t {
  IREE_HAL_READ_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling a file write operation.
typedef uint64_t iree_hal_write_flags_t;
enum iree_hal_write_flag_bits_t {
  IREE_HAL_WRITE_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling a host call operation.
typedef uint64_t iree_hal_host_call_flags_t;
enum iree_hal_host_call_flag_bits_e {
  IREE_HAL_HOST_CALL_FLAG_NONE = 0ull,

  // The call will not block the queue it is executing on.
  // The signal semaphores provided to iree_hal_device_queue_host_call will be
  // signaled immediately after the queue has issued the call so that work can
  // progress. The queue will not wait for the call to be made and it's possible
  // for it to happen out of order with respect to subsequent work on the queue.
  // The application itself must ensure that any references captured by the call
  // (user_data or args) are valid until the callback has completed.
  //
  // This is intended primarily for use as an optimization for custom signaling
  // behavior or notifications.
  IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING = 1ull << 0,

  // Hints that the host call is expected to be very short and that the issuing
  // queue may want to spin (possibly with backoff) until the host call has
  // signaled completion.
  IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE = 1ull << 1,

  // Hints that the host call does not require the device to flush/invalidate
  // caches. Use if the call does not consume any device resources that may have
  // been produced but not yet flushed to host memory and does not produce any
  // device resources that will be consumed without invalidation.
  IREE_HAL_HOST_CALL_FLAG_RELAXED = 1ull << 2,
};

// Provides context to a host call about where it was made from as well as any
// additional data requested.
typedef struct iree_hal_host_call_context_t {
  // The device the call was issued on.
  iree_hal_device_t* device;
  // The queue the call was issued on.
  // This is guaranteed to be equal-to or a subset-of the queue affinity
  // provided when the call was enqueued. Implementations are allowed to pick a
  // single queue to call the operation on and block that or block entire groups
  // of queues if there is some internal aliasing that introduces progress
  // issues if only one queue is treated as blocked.
  iree_hal_queue_affinity_t queue_affinity;
  // A list of semaphores that must be signaled once the call has completed.
  // Omitted if IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING was requested.
  //
  // The list lives on the stack and must be copied and each semaphore retained
  // if the call function does not immediately signal them inline. Asynchronous
  // completion would clone the list, retain the semaphores, fire off the async
  // operation, and then upon completion signal the semaphores and release them.
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_host_call_context_t;

// Executes a user-requested host call in queue order.
// If the call succeeds and returns OK the semaphores will be signaled and
// otherwise they will be failed. In non-blocking mode any error returned is
// ignored and no semaphores are available.
//
// To implement asynchronous callbacks the signal_semaphore_list provided in
// |context| should be cloned (list of pointers and retains on semaphores) and
// stored for later signaling. The callback must return IREE_STATUS_DEFERRED to
// indicate the asynchronous operation and when the operation has completed use
// iree_hal_semaphore_list_signal or iree_hal_semaphore_list_fail based on
// result.
typedef iree_status_t(IREE_API_PTR* iree_hal_host_call_fn_t)(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context);

// Bound host call function and user data.
typedef struct iree_hal_host_call_t {
  // Callback function pointer in the host program.
  iree_hal_host_call_fn_t fn;

  // User data passed to the callback function. Unowned.
  // When |resource| is provided it may retain |user_data| transitively.
  void* user_data;

  // Optional resource retaining callback state.
  //
  // The caller must keep its reference live until
  // iree_hal_device_queue_host_call returns. Implementations retain the
  // resource when the call may execute after the queue operation returns and
  // release it after |fn| returns or the queued call is cancelled before
  // invocation.
  //
  // Returning IREE_STATUS_DEFERRED transfers signal semaphore completion to
  // |fn| but does not extend this retain. A deferred continuation that still
  // needs the resource must retain it before returning and release it after
  // completion.
  iree_hal_resource_t* resource;
} iree_hal_host_call_t;

// Backend-native ingredients required to create queue-allocation pools for a
// device memory domain.
//
// |slab_provider| and |notification| are borrowed from the device. Pool
// constructors retain those objects, but the objects themselves,
// |epoch_query|, and |asan| may still read or describe backend-owned state. The
// device must therefore outlive all pools created from this bundle.
//
// |notification| may be shared by multiple pools in the same physical-memory
// domain and can wake callers whose own pool state did not change. Callers
// should always retry reservations after wakeup instead of assuming precise
// notification routing.
typedef struct iree_hal_queue_pool_backend_t {
  // Slab provider for the selected queue memory domain.
  iree_hal_slab_provider_t* slab_provider;

  // Notification shared by pools over the selected queue memory domain.
  iree_async_notification_t* notification;

  // Optional host-side epoch query for zero-sync block reuse.
  iree_hal_pool_epoch_query_t epoch_query;

  // Active ASAN allocation-shaping policy for custom pools over this backend.
  // Disabled when the device does not support or has not enabled ASAN pools.
  iree_hal_asan_pool_options_t asan;
} iree_hal_queue_pool_backend_t;

// Returns a host call bound to the given function pointer and user data.
static inline iree_hal_host_call_t iree_hal_make_host_call(
    iree_hal_host_call_fn_t fn, void* user_data) {
  iree_hal_host_call_t call = {
      /*.fn=*/fn,
      /*.user_data=*/user_data,
      /*.resource=*/NULL,
  };
  return call;
}

// Returns a host call whose callback state is retained by |resource|.
static inline iree_hal_host_call_t iree_hal_make_host_call_with_resource(
    iree_hal_host_call_fn_t fn, void* user_data,
    iree_hal_resource_t* resource) {
  iree_hal_host_call_t call = {
      /*.fn=*/fn,
      /*.user_data=*/user_data,
      /*.resource=*/resource,
  };
  return call;
}

// Bitfield specifying flags controlling an execution operation.
typedef uint64_t iree_hal_execute_flags_t;
enum iree_hal_execute_flag_bits_t {
  IREE_HAL_EXECUTE_FLAG_NONE = 0,
  // Allows the implementation to borrow binding table buffer lifetimes instead
  // of retaining them until the submitted work completes. Callers using this
  // flag must keep all buffers referenced by the binding table live and backed
  // by stable storage until the submission's signal semaphores indicate
  // completion. The binding table entries themselves are still captured during
  // the queue_execute call and need not remain live after it returns.
  IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME = 1ull << 0,
};

// Device's cached view of topology for fast compatibility checks.
//
// The bitmaps provide O(1) "can I interact with device X?" answers for the
// common path. For cost-sensitive decisions (e.g., choosing between two
// compatible devices), the topology pointer gives access to the full edge
// matrix with cost metrics, latency classes, and handle type negotiation.
//
// Populated during device group creation via
// iree_hal_device_assign_topology_info. Devices are not topology-complete until
// they have been assigned to a group.
typedef struct iree_hal_device_topology_info_t {
  // Scheduling word from the device's self-edge (edge[i][i].lo).
  iree_hal_topology_edge_scheduling_word_t self_edge;
  // Index of this device in the topology (0 to device_count-1).
  uint32_t topology_index;

  // Pointer to the immutable topology matrix owned by the device group.
  // NULL for standalone devices not in a topology group.
  // Lifetime: valid for the lifetime of this device (the topology outlives
  // all devices in the group).
  const iree_hal_topology_t* topology;

  // Bitmap of peer devices whose semaphores this device can wait on.
  iree_hal_topology_device_bitmap_t can_wait_from;

  // Bitmap of peer devices that can observe semaphores signaled by this device.
  iree_hal_topology_device_bitmap_t can_signal_to;

  // Bitmap of peer devices this device can import buffers from.
  iree_hal_topology_device_bitmap_t can_import_from;

  // Bitmap of peer devices this device can directly access or P2P-copy with.
  iree_hal_topology_device_bitmap_t can_p2p_with;

  // Frontier identity assigned to this device by its causal domain.
  struct {
    // Shared tracker used to publish/query queue progress for this device.
    iree_async_frontier_tracker_t* tracker;

    // QUEUE-domain base axis for this device with queue_index bits set to 0.
    iree_async_axis_t base_axis;
  } frontier;
} iree_hal_device_topology_info_t;

// Queries the full 128-bit edge between two devices using their topology info.
// Returns an empty edge if either device is not in a topology or they are in
// different topology groups.
static inline iree_hal_topology_edge_t iree_hal_device_topology_query_edge(
    const iree_hal_device_topology_info_t* src_info,
    const iree_hal_device_topology_info_t* dst_info) {
  if (!src_info->topology || src_info->topology != dst_info->topology) {
    return iree_hal_topology_edge_empty();
  }
  return iree_hal_topology_query_edge(
      src_info->topology, src_info->topology_index, dst_info->topology_index);
}

// Top-level device observation groups requested by callers and populated by
// devices during a point-in-time sample.
typedef uint64_t iree_hal_device_observation_flags_t;
typedef enum iree_hal_device_observation_flag_bits_e {
  // No observation groups are requested or populated.
  IREE_HAL_DEVICE_OBSERVATION_FLAG_NONE = 0ull,
  // Device memory availability and total allocation budget fields are requested
  // or populated.
  IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY = 1ull << 0,
  // Device sanitizer state fields are requested or populated.
  IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER = 1ull << 1,
  // All currently defined observation groups.
  IREE_HAL_DEVICE_OBSERVATION_FLAG_ALL =
      IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY |
      IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER,
} iree_hal_device_observation_flag_bits_t;

// Memory fields populated in an observation sample.
typedef uint64_t iree_hal_device_memory_observation_flags_t;
typedef enum iree_hal_device_memory_observation_flag_bits_e {
  // No memory observation fields are populated.
  IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_NONE = 0ull,
  // The total_bytes field is populated.
  IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES = 1ull << 0,
  // The available_bytes field is populated.
  IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_AVAILABLE_BYTES = 1ull << 1,
  // All currently defined memory observation fields.
  IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_ALL =
      IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES |
      IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_AVAILABLE_BYTES,
} iree_hal_device_memory_observation_flag_bits_t;

// Sampled device memory state.
typedef struct iree_hal_device_memory_observation_t {
  // Memory fields populated by the device.
  iree_hal_device_memory_observation_flags_t flags;
  // Total memory bytes represented by this sample.
  //
  // The value is source-defined: it may be immutable physical capacity when
  // sourced from device specs or a live allocation budget when sourced from a
  // backend budget query. Callers that need immutable hardware capacity should
  // use iree_hal_device_spec_t instead.
  iree_device_size_t total_bytes;
  // Memory bytes available for new allocations at sample time.
  iree_device_size_t available_bytes;
} iree_hal_device_memory_observation_t;

// ASAN fields populated in an observation sample.
typedef uint64_t iree_hal_device_asan_observation_flags_t;
typedef enum iree_hal_device_asan_observation_flag_bits_e {
  // No ASAN observation fields are populated.
  IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_NONE = 0ull,
  // The quarantine_size field is populated.
  IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_QUARANTINE_SIZE = 1ull << 0,
  // The quarantine_eviction_count field is populated.
  IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_QUARANTINE_EVICTION_COUNT = 1ull << 1,
  // The shadow_mapped_slab_count field is populated.
  IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_SHADOW_MAPPED_SLAB_COUNT = 1ull << 2,
  // The shadow_committed_size field is populated.
  IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_SHADOW_COMMITTED_SIZE = 1ull << 3,
  // All currently defined ASAN observation fields.
  IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_ALL =
      IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_QUARANTINE_SIZE |
      IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_QUARANTINE_EVICTION_COUNT |
      IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_SHADOW_MAPPED_SLAB_COUNT |
      IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_SHADOW_COMMITTED_SIZE,
} iree_hal_device_asan_observation_flag_bits_t;

// Sampled ASAN state.
typedef struct iree_hal_device_asan_observation_t {
  // ASAN fields populated by the device.
  iree_hal_device_asan_observation_flags_t flags;
  // Current total bytes retained by the ASAN quarantine FIFO.
  iree_device_size_t quarantine_size;
  // Cumulative count of mappings released due to ASAN quarantine pressure.
  uint64_t quarantine_eviction_count;
  // Number of precise physical shadow slabs currently mapped.
  uint64_t shadow_mapped_slab_count;
  // Physical shadow bytes currently committed.
  iree_device_size_t shadow_committed_size;
} iree_hal_device_asan_observation_t;

// Sampled device sanitizer state.
typedef struct iree_hal_device_sanitizer_observation_t {
  // Sampled ASAN state.
  iree_hal_device_asan_observation_t asan;
} iree_hal_device_sanitizer_observation_t;

// Point-in-time device state observation.
//
// Observations contain sampled device state that may change over the lifetime
// of a device. Immutable hardware and driver-interface facts belong in
// iree_hal_device_spec_t instead.
typedef struct iree_hal_device_observation_t {
  // Top-level observation groups requested by the caller.
  iree_hal_device_observation_flags_t requested_flags;
  // Top-level observation groups populated by the device.
  iree_hal_device_observation_flags_t provided_flags;
  // Monotonic host timestamp captured when sampling began.
  iree_time_t sample_time_ns;
  // Sampled memory state.
  iree_hal_device_memory_observation_t memory;
  // Sampled sanitizer state.
  iree_hal_device_sanitizer_observation_t sanitizer;
} iree_hal_device_observation_t;

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_t iree_hal_device_t;

// Retains the given |device| for the caller.
IREE_API_EXPORT void iree_hal_device_retain(iree_hal_device_t* device);

// Releases the given |device| from the caller.
IREE_API_EXPORT void iree_hal_device_release(iree_hal_device_t* device);

// Returns the device identifier.
// This identifier may vary based on the runtime device type; for example, a
// Vulkan device may return `vulkan-v1.1` or `vulkan-v1.2-spec1`.
IREE_API_EXPORT iree_string_view_t
iree_hal_device_id(iree_hal_device_t* device);

// Returns the host allocator used for objects.
IREE_API_EXPORT iree_allocator_t
iree_hal_device_host_allocator(iree_hal_device_t* device);

// Returns a reference to the allocator of the device that can be used for
// allocating buffers.
IREE_API_EXPORT iree_hal_allocator_t* iree_hal_device_allocator(
    iree_hal_device_t* device);

// Replaces the default device memory allocator with a compatible wrapper.
//
// The |new_allocator| will be retained for the lifetime of the device or until
// the allocator is replaced again. The common usage pattern is to wrap the
// default allocator without changing the device memory, virtual-memory,
// external-handle, or topology source facts captured in the immutable spec:
//   // Retain the existing allocator in the new wrapper.
//   wrap_allocator(iree_hal_device_allocator(device), &new_allocator);
//   // Update the device to use the wrapper for allocations.
//   IREE_RETURN_IF_ERROR(
//       iree_hal_device_replace_allocator(device, new_allocator));
//
// Replacements must preserve the allocation capabilities described by
// iree_hal_device_spec(). They may add instrumentation, caching, recording, or
// debugging behavior, but must not make the cached device spec inaccurate. A
// caller that needs different memory-system facts must create a new device.
// Returns an error if the wrapper cannot be installed.
//
// WARNING: this is not thread-safe and must only be performed when the device
// is idle and all buffers that may have been allocated from the existing
// allocator have been released. In general the only safe time to call this is
// immediately after device creation and before any buffers have been allocated.
// Beware: there are no internal checks for this condition!
//
// TODO(benvanik): remove this method and instead allow allocators to be
// composed without the safety caveats. This may take the form of unbound
// allocators that the device can inject the base allocator into. Another
// approach would be to replace the singular allocator with queue-specific pools
// and make the user register those pools explicitly with the implementation
// they desire.
IREE_API_EXPORT iree_status_t iree_hal_device_replace_allocator(
    iree_hal_device_t* device, iree_hal_allocator_t* new_allocator);

// Replaces the current collective channel provider.
// The |new_provider| will be retained for the lifetime of the device or until
// the provider is replaced again.
//
// WARNING: this is not thread-safe and must only be performed when the device
// is idle and all channels that may have been created from the existing
// provider have been released. In general the only safe time to call this is
// immediately after device creation and before any channels have been created.
// Beware: there are no internal checks for this condition!
IREE_API_EXPORT void iree_hal_device_replace_channel_provider(
    iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider);

// Trims pools and caches used by the HAL to the minimum required for live
// allocations. This can be used on low-memory conditions or when
// suspending/parking instances.
IREE_API_EXPORT
iree_status_t iree_hal_device_trim(iree_hal_device_t* device);

// Returns immutable facts for |device|.
//
// The returned pointer is owned by |device| and remains valid until |device| is
// destroyed. Callers that need to keep the spec beyond the device lifetime must
// retain it.
IREE_API_EXPORT const iree_hal_device_spec_t* iree_hal_device_spec(
    iree_hal_device_t* device);

// Initializes |out_observation| for a device state sample.
IREE_API_EXPORT void iree_hal_device_observation_initialize(
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation);

// Marks |total_bytes| as populated in |out_observation|.
IREE_API_EXPORT void iree_hal_device_observation_set_memory_total(
    iree_device_size_t total_bytes,
    iree_hal_device_observation_t* out_observation);

// Marks |available_bytes| as populated in |out_observation|.
IREE_API_EXPORT void iree_hal_device_observation_set_memory_available(
    iree_device_size_t available_bytes,
    iree_hal_device_observation_t* out_observation);

// Populates memory total capacity from known heap capacities in |device_spec|.
//
// Heap capacities marked unknown are skipped. If no known heap capacities are
// present then |out_observation| is left unchanged.
IREE_API_EXPORT iree_status_t
iree_hal_device_observation_populate_memory_total_from_spec(
    const iree_hal_device_spec_t* device_spec,
    iree_hal_device_observation_t* out_observation);

// Samples dynamic state from |device| into |out_observation|.
//
// The returned observation is a point-in-time snapshot: devices populate only
// requested groups and fields they can sample without inventing fallback
// values. Missing groups and fields are reported by leaving the corresponding
// provided/field flags unset.
IREE_API_EXPORT iree_status_t iree_hal_device_sample_observation(
    iree_hal_device_t* device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation);

// Returns a pointer to device's topology info populated during device creation.
// Returns NULL if device is not part of a topology.
// Pointer lifetime matches device lifetime.
IREE_API_EXPORT const iree_hal_device_topology_info_t*
iree_hal_device_topology_info(iree_hal_device_t* device);

// Refines a topology edge from |src_device| to |dst_device|.
//
// Device specs provide the serializable source facts for topology projection.
// This hook is called only for same-runtime-domain pairs where a driver may
// prove additional process-local facts from live backend handles. If the device
// has no such facts, it returns OK without modification.
IREE_API_EXPORT iree_status_t iree_hal_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge);

// Assigns topology information to |device| during device group construction.
// The |topology_info| struct is copied into the device's internal storage. The
// topology pointer within |topology_info| must remain valid for the lifetime of
// the device (ensured by the device group retaining all its devices).
//
// If |topology_info| is NULL, aborts a prior assignment on a device whose
// containing device group was never returned to the caller. NULL is only valid
// during construction failure unwinding and must not be used once a device
// group has escaped or once any work may have been scheduled.
IREE_API_EXPORT iree_status_t iree_hal_device_assign_topology_info(
    iree_hal_device_t* device,
    const iree_hal_device_topology_info_t* topology_info);

// Queries in what ways the given |semaphore| may be used with |device|.
IREE_API_EXPORT iree_hal_semaphore_compatibility_t
iree_hal_device_query_semaphore_compatibility(iree_hal_device_t* device,
                                              iree_hal_semaphore_t* semaphore);

// Queries the slab provider, notification, and epoch-query callback to use when
// constructing custom pools for |queue_affinity|.
//
// Implementations may collapse a multi-bit |queue_affinity| to one physical
// memory domain using the same queue-selection policy they use for submission.
// The returned pointers are borrowed from |device| and remain valid until the
// device is destroyed.
//
// Requires that |device| has been assigned to a device group.
IREE_API_EXPORT iree_status_t iree_hal_device_query_queue_pool_backend(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend);

// Loads a native executable artifact for |target| on |queue_affinity|.
//
// |target| must be an exact borrowed row from iree_hal_device_spec(device).
// The backend resolves |queue_affinity| to physical devices and requires every
// selected physical device to be present in |target->physical_device_affinity|.
// An ANY queue affinity therefore requires a target valid for every physical
// device selected by the backend.
//
// The executable data and constants are borrowed only for the duration of the
// call. Implementations must finish consuming or copy any retained data before
// returning. Loading is a cold path and implementations may parse, verify,
// link, or optimize the native artifact before returning.
IREE_API_EXPORT iree_status_t iree_hal_device_load_executable(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_hal_executable_t** out_executable);

// Reserves and returns a device-local queue-ordered transient buffer.
// The allocation will not be committed until the entire |wait_semaphore_list|
// has been reached. Once the storage is available for use the
// |signal_semaphore_list| will be signaled. The contents of the buffer are
// undefined until signaled even if all waits have been resolved and callers
// must always wait for the signal.
//
// |pool| is a borrowed allocation pool selector. NULL selects the device's
// default pool. Any non-NULL pool must outlive all queue submissions,
// reservations, and materialized buffers that use it.
//
// For optimal performance and minimal memory consumption the returned buffer
// should be deallocated using iree_hal_device_queue_dealloca as soon as
// possible. It's still safe to synchronously release the buffer but the
// lifetime will then be controlled by all potential retainers.
//
// Usage:
//   iree_hal_device_queue_alloca(wait(0), signal(1), &buffer);
//   iree_hal_device_queue_execute(wait(1), signal(2), commands...);
//   iree_hal_device_queue_dealloca(wait(2), signal(3), buffer);
IREE_API_EXPORT iree_status_t iree_hal_device_queue_alloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

// Deallocates a queue-ordered transient buffer.
// The deallocation will not be made until the entire |wait_semaphore_list| has
// been reached. Once the storage is available for reuse the
// |signal_semaphore_list| will be signaled. After all waits have been resolved
// the contents of the buffer are immediately undefined even if the signal has
// not yet occurred. Queue implementations must perform target-visible release
// effects such as decommit, sanitizer poisoning, or guard-page updates after
// all waits are satisfied and before any signals are published. If the buffer
// was not allocated asynchronously a barrier will be inserted to preserve fence
// timelines.
//
// Deallocations will only be queue-ordered if the |buffer| was originally
// allocated with iree_hal_device_queue_alloca. Any synchronous allocations will
// be ignored and deallocated when the |buffer| has been released but a queue
// barrier will be inserted to preserve the timeline.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_dealloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

// Enqueues a single queue-ordered fill operation.
// The |target_buffer| must be visible to the device queue performing the fill.
//
// WARNING: individual fills have a high overhead and batching should be
// performed by the caller instead of calling this multiple times. The
// iree_hal_create_transfer_command_buffer utility makes it easy to create
// batches of transfer operations (fill, update, copy) and is only a few lines
// more code.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_fill(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

// Enqueues a single queue-ordered buffer update operation.
// The provided |source_buffer| will be captured and need not remain live or
// unchanged while the operation is queued. The |target_buffer| must be visible
// to the device queue performing the update.
//
// Some implementations may have limits on the size of the update or may perform
// poorly if the size is larger than an implementation-defined limit. Updates
// should be kept as small and infrequent as possible.
//
// WARNING: individual copies have a high overhead and batching should be
// performed by the caller instead of calling this multiple times. The
// iree_hal_create_transfer_command_buffer utility makes it easy to create
// batches of transfer operations (fill, update, copy) and is only a few lines
// more code.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_update(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

// Enqueues a single queue-ordered copy operation.
// The |source_buffer| and |target_buffer| must both be visible to the device
// queue performing the copy.
//
// WARNING: individual copies have a high overhead and batching should be
// performed by the caller instead of calling this multiple times. The
// iree_hal_create_transfer_command_buffer utility makes it easy to create
// batches of transfer operations (fill, update, copy) and is only a few lines
// more code.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_copy(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

// Enqueues a file read operation that streams a segment of the |source_file|
// defined by the |source_offset| and |length| into the HAL |target_buffer| at
// the specified |target_offset|. The |queue_affinity| should be set to where
// the target buffer will be consumed. The source file must have read permission
// and the target buffer must have transfer-target usage.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_read(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Enqueues a file write operation that streams a segment of the HAL
// |source_buffer| defined by the |source_offset| and |length| into the
// |target_file| at the specified |target_offset|. The |queue_affinity| should
// be set to where the source buffer was produced. The source buffer must have
// transfer-source usage and the target file must have write permission.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_write(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

// Enqueues a host call request.
// The device will issue the host call once all waits are satisfied. Host calls
// receive the signal semaphores provided and can be either synchronous (signal
// inline) or asynchronous (signal at any point in the future). A non-blocking
// mode is provided for unidirectional/post-style calls.
//
// WARNING: re-entrancy is not supported. It is safe to perform semaphore
// queries and signals and synchronously allocate/deallocate buffers and
// resources but queue operations _may_ lead to hangs/crashes. Avoid using any
// iree_hal_device_queue_* API or performing any blocking waits. If queuing is
// required then bounce the call to another thread and have it performed there.
//
// Arguments are passed without modification from the enqueue operation to the
// callback. If the arguments contain pointers those must remain live until the
// host call has executed.
//
// Calls block dependent work by default. Once all waits have been satisfied the
// queue will issue the call to the host with the signals provided and the host
// call is responsible for either completing its work and returning OK to
// automatically signal the semaphores. Note that other independent work in the
// queue is allowed to progress while the host call is in-flight. Calls can be
// implemented asynchronously by cloning and retaining the signal semaphores
// they are provided, returning IREE_STATUS_DEFERRED, and signaling them at any
// point in the future (from an async completion callback, another queue, etc).
//
// The IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING flag can be used to instead have the
// queue issue the call after waits have been satisfied and then immediately
// signal dependencies prior to the host call being executed. This allows post
// style notifications without blocking subsequent device work and can be used
// as a generic signaling mechanism.
//
// Call lifetime in both modes:
// ```
// BLOCKING (call responsible for signaling):
//   [alloc state]->[wait]->[call on host]->[signal]->[free state]
//                            ^             ^
//                            |             |
//                            |             Call must signal before returning
//                            Call receives signal_semaphore_list
//
// NON_BLOCKING (queue signals, call runs detached):
//   [alloc state]->[wait]->[signal]->[call on host]->[free state]
//                            ^       ^
//                            |       |
//                            |       Call receives empty signal_semaphore_list
//                            Queue signals immediately
// ```
//
// NOTE: host calls can be extremely expensive and result in significant
// performance issues. Some implementations are not able to natively support
// host calls and require emulation with poller threads and other techniques
// that add non-trivial latency in device->host->device situations. Avoid host
// calls if at all possible.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_host_call(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags);

// Enqueues a dispatch over a 3D grid of workgroups.
// The request may execute overlapped with any other queue operations. The
// executable specified must be registered for use with the device driver owning
// the queue it is scheduled on.
//
// The provided constant data and binding list will be recorded into the queue
// and need not remain live beyond the call. By default the executable, binding
// buffers, and indirect parameter buffer will be retained by the queue until
// the operation has completed. Callers that already guarantee resource
// lifetimes may pass IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES to allow
// implementations to skip that tracking on hot paths.
//
// All provided |bindings| must be directly specified and not reference binding
// table slots.
//
// See iree_hal_command_buffer_dispatch for more information.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_dispatch(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

// Executes a command buffer on a device queue.
// No commands will execute until the wait fence has been reached and the signal
// fence will be signaled when all commands have completed. If a command buffer
// is omitted this will act as a barrier.
//
// The queue is selected based on the command buffer submitted and the
// |queue_affinity|. As the number of available queues can vary the
// |queue_affinity| is used to hash into the available queues for the required
// categories. For example if 2 queues support transfer commands and the
// affinity is 5 the resulting queue could be index hash(5)=1. The affinity can
// thus be treated as just a way to indicate whether two submissions must be
// placed on to the same queue. Note that the exact hashing function is
// implementation dependent.
//
// An optional binding table must be provided if the command buffer has indirect
// bindings and may otherwise be `iree_hal_buffer_binding_table_empty()`. The
// binding table contents will be captured during the call and need not persist
// after the call returns. By default, buffers referenced by the binding table
// are retained until the submitted work completes. Callers that already
// guarantee buffer lifetimes may pass
// IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME to allow implementations
// to skip that tracking on hot paths.
//
// The submission behavior matches Vulkan's vkQueueSubmit, with each submission
// executing its command buffers in the order they are defined but allowing the
// command buffers to complete out-of-order. See:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit.html
IREE_API_EXPORT iree_status_t iree_hal_device_queue_execute(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags);

// Flags controlling iree_hal_device_queue_timestamp behavior.
typedef uint64_t iree_hal_timestamp_flags_t;
enum iree_hal_timestamp_flag_bits_t {
  IREE_HAL_TIMESTAMP_FLAG_NONE = 0u,
};

// Enqueues a device-side timestamp at the point in the |queue_affinity|
// timeline reached after |wait_semaphore_list| is satisfied, signaling
// |signal_semaphore_list| when that point is reached. A single 64-bit device
// timestamp tick is written by the device into |target_buffer| at
// |target_offset|. The target must be 8-byte aligned and be a transfer target
// the device can write 8 bytes into; to read the tick back on the host the
// buffer must additionally be host-visible (e.g. mappable). Callers must wait
// on |signal_semaphore_list| before reading the tick back from |target_buffer|.
//
// The captured value is a tick in the timestamp domain of the queue the
// implementation selects for |queue_affinity|. A device may span several
// physical devices whose timestamp domains are independent counters, and each
// queue family in iree_hal_device_spec_queues() publishes the timestamp facts
// and physical_device_affinity identifying the domain its queues capture in.
// Two ticks may only be compared or subtracted when both captures resolved to
// the same domain. Two captures issued with the same |queue_affinity| always
// resolve to the same domain: as with iree_hal_device_queue_execute an
// affinity selects one queue and a queue captures in one domain, so reusing an
// affinity keeps captures comparable on a device spanning several physical
// devices. Which family an affinity resolves to is internal to the
// implementation, but a caller pairing its own captures need not know it:
// iree_hal_queue_affinity_t bits and physical-device affinity bits are
// independent namespaces that the device spec does not map between. Differing
// affinities may or may not share a domain, though a device reporting one
// queue family covering one physical device has a single domain in which any
// two of its captures are comparable. Correlation across domains, or with host
// time, is not expressed by this operation.
//
// Convert with the facts of the family the capture resolved to: only the low
// timestamp_valid_bits of a tick are defined and the counter wraps at that
// width, so reduce the difference of two ticks modulo that width before
// dividing it by the family's timestamp_frequency_hz to convert to seconds.
// Subtracting the raw values is only correct where timestamp_valid_bits is 64.
// iree_hal_device_spec_timing() summarizes the same two facts device-scope and
// may aggregate families that differ, so it stands in for the family facts
// only on a device reporting one queue family.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_timestamp(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags);

// Enqueues a barrier waiting for |wait_semaphore_list| and signaling
// |signal_semaphore_list| when reached.
// Equivalent to iree_hal_device_queue_execute with no command buffers.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_barrier(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_execute_flags_t flags);

// Flushes any locally-pending submissions in the queue.
// When submitting many queue operations this can be used to eagerly flush
// earlier submissions while later ones are still being constructed.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_flush(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity);

// Blocks the caller until the semaphores reach or exceed the specified payload
// values or the |timeout| elapses. All semaphores in |semaphore_list| must be
// created from this device (or be imported into it).
//
// |wait_mode| can be used to decide when the wait will proceed; whether *all*
// semaphores in |semaphore_list| must be signaled or whether *any* (one or
// more) can be signaled before an early return.
//
// Returns success if the wait is successful and semaphores have been signaled
// satisfying the |wait_mode|.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the |timeout| elapses without the
// |wait_mode| being satisfied. Note that even on success only a subset of the
// semaphores may have been signaled and each can be queried to see which ones.
//
// Returns IREE_STATUS_ABORTED if one or more semaphores has failed. Callers can
// use iree_hal_semaphore_query on the semaphores to find the ones that have
// failed and get the status.
IREE_API_EXPORT iree_status_t iree_hal_device_wait_semaphores(
    iree_hal_device_t* device, iree_async_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_async_wait_flags_t flags);

// Begins a HAL-native structured profiling session on |device| with |options|.
// A zero data-family set is a valid no-op and starts no session.
//
// A successful nonzero begin creates one active session on the device until
// iree_hal_device_profiling_end is called. Nested begin calls must fail with
// IREE_STATUS_FAILED_PRECONDITION. Unsupported requested data must fail loudly
// instead of returning success with no profile output.
//
// Callers must externally serialize begin/end with queue submission, command
// buffer recording that may later observe the session, and concurrent
// begin/flush/end calls on the same device. Unless the backend explicitly
// documents dynamic profiling toggles, there must be no in-flight queue work
// when begin or end is called. This lets producers keep ordinary queue hot
// paths to cheap explicit profiling checks instead of locks or atomics around
// every operation.
//
// Flush and end must not invoke sink callbacks while holding queue locks,
// semaphore callback locks, task-worker hot-loop locks, or a profiling mutation
// lock that queue completion needs. Sink callbacks may block or allocate.
//
// Profiling can dramatically increase overhead, with some data families adding
// enough host and device cost to invalidate measurements from other mechanisms.
// Use the narrowest data-family set that captures the data being investigated.
IREE_API_EXPORT iree_status_t iree_hal_device_profiling_begin(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options);

// Flushes pending profiling data for the active profiling session.
//
// Flush may be a no-op for producers that do not buffer completed records. It
// may run while work is in flight only when the producer has a safe snapshot
// boundary for the requested profiling data. In-flight spans, timestamp
// packets, counters, or traces must not be emitted as complete records.
IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_flush(iree_hal_device_t* device);

// Ends a profiling session previously started with
// iree_hal_device_profiling_begin.
//
// Callers must satisfy the same external serialization and idle-device
// requirements as begin unless the backend explicitly documents dynamic
// toggling support. Implementations must release session-owned resources even
// if flushing, producer teardown, or sink end-session callbacks fail.
IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_end(iree_hal_device_t* device);

// Begins an external profiler/tool capture range on |device|.
//
// External capture is for provider-specific artifacts and UI sessions outside
// the HAL-native profile sink format. Examples include RenderDoc .rdc captures
// and Metal .gputrace documents. A successful begin means the requested
// provider started its capture; it does not imply any HAL profile chunks will
// be produced.
//
// A device may support at most one active external capture unless the provider
// explicitly documents nested or concurrent capture support.
IREE_API_EXPORT iree_status_t iree_hal_device_external_capture_begin(
    iree_hal_device_t* device,
    const iree_hal_device_external_capture_options_t* options);

// Ends an external profiler/tool capture range previously started with
// iree_hal_device_external_capture_begin.
IREE_API_EXPORT iree_status_t
iree_hal_device_external_capture_end(iree_hal_device_t* device);

//===----------------------------------------------------------------------===//
// iree_hal_device_list_t
//===----------------------------------------------------------------------===//

// A fixed-size list of retained devices.
typedef struct iree_hal_device_list_t {
  iree_allocator_t host_allocator;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_device_t* devices[];
} iree_hal_device_list_t;

// Allocates an empty device list with the given capacity.
IREE_API_EXPORT iree_status_t iree_hal_device_list_allocate(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_device_list_t** out_list);

// Frees a device |list|.
IREE_API_EXPORT void iree_hal_device_list_free(iree_hal_device_list_t* list);

// Pushes a |device| onto the |list| and retains it.
IREE_API_EXPORT iree_status_t iree_hal_device_list_push_back(
    iree_hal_device_list_t* list, iree_hal_device_t* device);

// Returns the device at index |i| in the |list| or NULL if out of range.
// Callers must retain the device if it's possible for the returned pointer to
// live beyond the list.
IREE_API_EXPORT iree_hal_device_t* iree_hal_device_list_at(
    const iree_hal_device_list_t* list, iree_host_size_t i);

//===----------------------------------------------------------------------===//
// iree_hal_device_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_device_t* device);

  iree_string_view_t(IREE_API_PTR* id)(iree_hal_device_t* device);

  iree_allocator_t(IREE_API_PTR* host_allocator)(iree_hal_device_t* device);
  iree_hal_allocator_t*(IREE_API_PTR* device_allocator)(
      iree_hal_device_t* device);
  iree_status_t(IREE_API_PTR* replace_device_allocator)(
      iree_hal_device_t* device, iree_hal_allocator_t* new_allocator);
  void(IREE_API_PTR* replace_channel_provider)(
      iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider);

  iree_status_t(IREE_API_PTR* trim)(iree_hal_device_t* device);

  const iree_hal_device_spec_t*(IREE_API_PTR* device_spec)(
      iree_hal_device_t* device);

  iree_status_t(IREE_API_PTR* sample_observation)(
      iree_hal_device_t* device,
      iree_hal_device_observation_flags_t requested_flags,
      iree_hal_device_observation_t* out_observation);

  const iree_hal_device_topology_info_t*(IREE_API_PTR* topology_info)(
      iree_hal_device_t* device);

  iree_status_t(IREE_API_PTR* refine_topology_edge)(
      iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
      iree_hal_topology_edge_t* edge);

  iree_status_t(IREE_API_PTR* assign_topology_info)(
      iree_hal_device_t* device,
      const iree_hal_device_topology_info_t* topology_info);

  iree_status_t(IREE_API_PTR* create_channel)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);

  iree_status_t(IREE_API_PTR* create_command_buffer)(
      iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_queue_affinity_t queue_affinity,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer);

  iree_status_t(IREE_API_PTR* create_event)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_event_flags_t flags, iree_hal_event_t** out_event);

  iree_status_t(IREE_API_PTR* load_executable)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_executable_target_t* target,
      const iree_hal_executable_load_params_t* params,
      iree_hal_executable_t** out_executable);

  iree_status_t(IREE_API_PTR* import_file)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
      iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file);

  iree_status_t(IREE_API_PTR* create_semaphore)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      uint64_t initial_value, iree_hal_semaphore_flags_t flags,
      iree_hal_semaphore_t** out_semaphore);

  iree_hal_semaphore_compatibility_t(
      IREE_API_PTR* query_semaphore_compatibility)(
      iree_hal_device_t* device, iree_hal_semaphore_t* semaphore);

  iree_status_t(IREE_API_PTR* query_queue_pool_backend)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_queue_pool_backend_t* out_backend);

  iree_status_t(IREE_API_PTR* queue_alloca)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
      iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
      iree_hal_buffer_t** IREE_RESTRICT out_buffer);

  iree_status_t(IREE_API_PTR* queue_dealloca)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_fill)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, const void* pattern,
      iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_update)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      const void* source_buffer, iree_host_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_update_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_copy)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_copy_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_read)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_file_t* source_file, uint64_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_read_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_write)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_file_t* target_file, uint64_t target_offset,
      iree_device_size_t length, iree_hal_write_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_host_call)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_host_call_t call, const uint64_t args[4],
      iree_hal_host_call_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_dispatch)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_executable_t* executable,
      iree_hal_executable_function_t function,
      const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
      const iree_hal_buffer_ref_list_t bindings,
      iree_hal_dispatch_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_execute)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table,
      iree_hal_execute_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_timestamp)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_timestamp_flags_t flags);

  iree_status_t(IREE_API_PTR* queue_flush)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity);

  iree_status_t(IREE_API_PTR* profiling_begin)(
      iree_hal_device_t* device,
      const iree_hal_device_profiling_options_t* options);
  iree_status_t(IREE_API_PTR* profiling_flush)(iree_hal_device_t* device);
  iree_status_t(IREE_API_PTR* profiling_end)(iree_hal_device_t* device);

  iree_status_t(IREE_API_PTR* external_capture_begin)(
      iree_hal_device_t* device,
      const iree_hal_device_external_capture_options_t* options);
  iree_status_t(IREE_API_PTR* external_capture_end)(iree_hal_device_t* device);
} iree_hal_device_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_device_vtable_t);

IREE_API_EXPORT void iree_hal_device_destroy(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_H_
