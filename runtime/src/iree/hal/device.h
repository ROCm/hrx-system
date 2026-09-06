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

// Device creation parameter extension types.
//
// Values are globally assigned by the HAL. Drivers must silently skip extension
// types they do not recognize to preserve forward compatibility.
typedef uint32_t iree_hal_device_create_params_extension_type_t;
typedef enum iree_hal_device_create_params_extension_type_e {
  IREE_HAL_DEVICE_CREATE_PARAMS_EXTENSION_TYPE_NONE = 0u,
  // Provides one opaque hostcall service for each physical device.
  IREE_HAL_DEVICE_CREATE_PARAMS_EXTENSION_TYPE_HOSTCALL_PROVIDER = 1u,
} iree_hal_device_create_params_extension_type_e;

// Common prefix for device creation parameter extensions.
//
// Drivers walk the chain by interpreting each extension through this prefix.
typedef struct iree_hal_device_create_params_extension_t {
  // Globally assigned extension type identifier.
  iree_hal_device_create_params_extension_type_t type;

  // Next device creation parameter extension, or NULL.
  const void* next;
} iree_hal_device_create_params_extension_t;

// Notification primitive types supported by hostcall providers.
typedef uint32_t iree_hal_hostcall_notification_type_t;
typedef enum iree_hal_hostcall_notification_type_e {
  IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_NONE = 0u,
  // Token is a device-visible HSA signal handle.
  IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL = 1u,
} iree_hal_hostcall_notification_type_e;

// Device notification primitive supplied to a hostcall provider.
typedef struct iree_hal_hostcall_notification_t {
  // Concrete notification primitive type.
  iree_hal_hostcall_notification_type_t type;
  // Reserved for future use and must be zero.
  uint32_t reserved;
  // Device-visible token encoded according to |type|.
  uint64_t token;
} iree_hal_hostcall_notification_t;

// Scalar physical-device facts available to opaque hostcall providers.
typedef struct iree_hal_hostcall_provider_device_info_t {
  // Ordinal of the physical device within the logical-device topology.
  uint32_t physical_device_ordinal;
  // Number of execution units reported for the physical device.
  uint32_t execution_unit_count;
  // Maximum resident subgroup count per execution unit.
  uint32_t maximum_resident_subgroup_count;
  // Reserved for future use and must be zero.
  uint32_t reserved;
} iree_hal_hostcall_provider_device_info_t;

// Shared allocation and notification requirements of a hostcall provider.
typedef struct iree_hal_hostcall_provider_requirements_t {
  // Required allocation size in bytes.
  iree_host_size_t allocation_size;
  // Required power-of-two allocation alignment in bytes.
  iree_host_size_t allocation_alignment;
  // Notification primitive the provider requires.
  iree_hal_hostcall_notification_type_t notification_type;
  // Reserved for future use and must be zero.
  uint32_t reserved;
} iree_hal_hostcall_provider_requirements_t;

// Handles an asynchronous terminal error from a hostcall provider.
//
// May be called from a driver thread and must not call back into the
// originating provider or driver APIs. Ownership of |status| transfers to the
// callback and must be propagated, retained, or freed before returning.
typedef void(IREE_API_PTR* iree_hal_hostcall_error_fn_t)(void* user_data,
                                                         iree_status_t status);

// Callback consuming asynchronous terminal hostcall provider errors.
typedef struct iree_hal_hostcall_error_callback_t {
  // Function consuming ownership of a terminal status.
  iree_hal_hostcall_error_fn_t fn;
  // Opaque data passed to |fn|.
  void* user_data;
} iree_hal_hostcall_error_callback_t;

// Queries shared allocation and notification requirements for one physical
// device.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_hostcall_provider_query_requirements_fn_t)(
    void* user_data,
    const iree_hal_hostcall_provider_device_info_t* device_info,
    iree_hal_hostcall_provider_requirements_t* out_requirements);

// Initializes one provider context over a HAL-owned shared allocation.
//
// |shared_memory| and |device_address| name the host and device views of the
// same allocation and remain live until the matching deinitialize call.
// |notification| remains valid for the same lifetime and may only be
// interpreted according to its type. |error_callback| consumes terminal
// provider failures from arbitrary service threads and remains valid until
// deinitialize. Implementations set |out_context| to NULL before invoking the
// callback. If the callback returns an error after publishing a non-NULL
// context, the matching deinitialize callback is invoked exactly once while
// the allocation and notification are still valid.
typedef iree_status_t(IREE_API_PTR* iree_hal_hostcall_provider_initialize_fn_t)(
    void* user_data,
    const iree_hal_hostcall_provider_device_info_t* device_info,
    iree_byte_span_t shared_memory, uint64_t device_address,
    iree_hal_hostcall_notification_t notification,
    iree_hal_hostcall_error_callback_t error_callback, void** out_context);

// Services all provider work currently ready in one physical-device context.
//
// The provider owns protocol outcomes. Structural failures are published
// through the error callback captured during initialization.
typedef void(IREE_API_PTR* iree_hal_hostcall_provider_service_fn_t)(
    void* context);

// Deinitializes one provider context after its service thread has joined and
// before its shared allocation or notification is released.
typedef void(IREE_API_PTR* iree_hal_hostcall_provider_deinitialize_fn_t)(
    void* context);

// Immutable opaque hostcall provider copied during device creation.
typedef struct iree_hal_hostcall_provider_t {
  // Provider-owned data passed to requirement and initialization callbacks.
  void* user_data;
  // Queries shared allocation and notification requirements.
  iree_hal_hostcall_provider_query_requirements_fn_t query_requirements;
  // Initializes one physical-device provider context.
  iree_hal_hostcall_provider_initialize_fn_t initialize;
  // Services ready work on the listener thread.
  iree_hal_hostcall_provider_service_fn_t service;
  // Deinitializes the provider context after listener shutdown.
  iree_hal_hostcall_provider_deinitialize_fn_t deinitialize;
} iree_hal_hostcall_provider_t;

// Device creation extension enabling an opaque hostcall provider.
//
// Drivers supporting the extension instantiate the provider once per physical
// device. The provider value is copied during device creation. Provider
// |user_data| must remain valid until the created device is destroyed.
typedef struct iree_hal_hostcall_provider_extension_t {
  // Common device creation extension prefix.
  iree_hal_device_create_params_extension_t base;
  // Provider instantiated once for each physical device.
  iree_hal_hostcall_provider_t provider;
} iree_hal_hostcall_provider_extension_t;

// Parameters for device creation that apply across all HAL drivers.
//
// Callers stack-allocate and initialize with
// iree_hal_device_create_params_default(), then customize fields as needed
// before passing to device creation functions. All creation paths require a
// valid pointer — callers must always provide one.
//
// The |next| pointer enables a Vulkan-style extension chain. Each extension
// must begin with iree_hal_device_create_params_extension_t. Unrecognized
// extensions are silently skipped for forward compatibility.
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

// Returns the queue family at canonical |family_ordinal| or NULL if invalid.
//
// The returned pointer is borrowed from |device| and remains stable until the
// device is destroyed.
IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_device_queue_family(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t family_ordinal);

// Returns the provisioned hardware queue at the given coordinate or NULL if
// either ordinal is invalid.
//
// This is an infallible lookup over queues provisioned during device creation;
// it never creates or acquires a queue. The returned pointer is borrowed from
// |device| and remains stable until the device is destroyed. The caller may
// retain the queue but must still ensure the device outlives that reference.
IREE_API_EXPORT iree_hal_queue_t* iree_hal_device_queue(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal);

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

// Loads a native executable artifact for |target| on |queue_family|.
//
// |queue_family| must be the exact family identity borrowed from |device| and
// |target| must be an exact borrowed row from iree_hal_device_spec(device).
// |target| must support every physical device serviced by |queue_family|.
// The returned executable may only be used with command buffers and direct
// dispatches targeting |queue_family|.
//
// The executable data and constants are borrowed only for the duration of the
// call. Implementations must finish consuming or copy any retained data before
// returning. Loading is a cold path and implementations may parse, verify,
// link, or optimize the native artifact before returning.
IREE_API_EXPORT iree_status_t iree_hal_device_load_executable(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_hal_executable_t** out_executable);

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
  void(IREE_API_PTR* replace_channel_provider)(
      iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider);

  iree_status_t(IREE_API_PTR* trim)(iree_hal_device_t* device);

  const iree_hal_device_spec_t*(IREE_API_PTR* device_spec)(
      iree_hal_device_t* device);

  const iree_hal_queue_family_t*(IREE_API_PTR* queue_family)(
      iree_hal_device_t* device,
      iree_hal_queue_family_ordinal_t family_ordinal);

  iree_hal_queue_t*(IREE_API_PTR* queue)(
      iree_hal_device_t* device, iree_hal_queue_family_ordinal_t family_ordinal,
      iree_hal_queue_ordinal_t queue_ordinal);

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
      iree_hal_device_t* device,
      iree_hal_queue_family_affinity_t queue_family_affinity,
      iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);

  iree_status_t(IREE_API_PTR* create_command_buffer)(
      iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer);

  iree_status_t(IREE_API_PTR* load_executable)(
      iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
      const iree_hal_executable_target_t* target,
      const iree_hal_executable_load_params_t* params,
      iree_hal_executable_t** out_executable);

  iree_status_t(IREE_API_PTR* import_file)(
      iree_hal_device_t* device,
      iree_hal_queue_family_affinity_t queue_family_affinity,
      iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
      iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file);

  iree_status_t(IREE_API_PTR* create_semaphore)(
      iree_hal_device_t* device,
      iree_hal_queue_family_affinity_t queue_family_affinity,
      uint64_t initial_value, iree_hal_semaphore_flags_t flags,
      iree_hal_semaphore_t** out_semaphore);

  iree_hal_semaphore_compatibility_t(
      IREE_API_PTR* query_semaphore_compatibility)(
      iree_hal_device_t* device, iree_hal_semaphore_t* semaphore);

  iree_status_t(IREE_API_PTR* query_queue_pool_backend)(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_queue_pool_backend_t* out_backend);

  iree_status_t(IREE_API_PTR* profiling_begin)(
      iree_hal_device_t* device,
      const iree_hal_device_profiling_options_t* options);
  iree_status_t(IREE_API_PTR* profiling_flush)(iree_hal_device_t* device);
  iree_status_t(IREE_API_PTR* profiling_end)(iree_hal_device_t* device);
} iree_hal_device_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_device_vtable_t);

IREE_API_EXPORT void iree_hal_device_destroy(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_H_
