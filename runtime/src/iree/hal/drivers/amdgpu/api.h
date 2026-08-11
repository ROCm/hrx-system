// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_API_H_
#define IREE_HAL_DRIVERS_AMDGPU_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Exported for API interop:
#include "iree/hal/drivers/amdgpu/util/error_callback.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/util/libhsa.h"          // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/util/topology.h"        // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_hostcall_provider_t
//===----------------------------------------------------------------------===//

// AMDGPU device creation parameter extension types.
typedef uint32_t iree_hal_amdgpu_device_create_params_extension_type_t;
enum iree_hal_amdgpu_device_create_params_extension_type_e {
  // Provides one opaque hostcall service for each physical device.
  IREE_HAL_AMDGPU_DEVICE_CREATE_PARAMS_EXTENSION_TYPE_HOSTCALL_PROVIDER = 1u,
};

// Scalar physical-device facts available to opaque hostcall providers.
typedef struct iree_hal_amdgpu_hostcall_provider_device_info_t {
  // Ordinal of the physical device within the logical-device topology.
  iree_host_size_t physical_device_ordinal;

  // Number of compute units reported for the physical device.
  uint32_t compute_unit_count;

  // Maximum resident wave count per compute unit.
  uint32_t maximum_waves_per_compute_unit;

  // Native wavefront size reported for the physical device.
  uint32_t wavefront_size;
} iree_hal_amdgpu_hostcall_provider_device_info_t;

// Opaque shared allocation requirements returned by a hostcall provider.
typedef struct iree_hal_amdgpu_hostcall_provider_requirements_t {
  // Required allocation size in bytes.
  iree_host_size_t allocation_size;

  // Required power-of-two allocation alignment in bytes.
  iree_host_size_t allocation_alignment;
} iree_hal_amdgpu_hostcall_provider_requirements_t;

// Queries shared allocation requirements for one physical device.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amdgpu_hostcall_provider_query_requirements_fn_t)(
    void* user_data,
    const iree_hal_amdgpu_hostcall_provider_device_info_t* device_info,
    iree_hal_amdgpu_hostcall_provider_requirements_t* out_requirements);

// Initializes one provider context over an HAL-owned shared allocation.
//
// |shared_memory| remains live until the matching deinitialize call.
// |device_address| is the stable device-visible address of the same allocation.
// |notification_token| is opaque to the provider except when encoding its
// device protocol. Only the HAL may use it as a native synchronization object.
// |error_callback| consumes terminal provider failures and remains valid until
// deinitialize.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amdgpu_hostcall_provider_initialize_fn_t)(
    void* user_data,
    const iree_hal_amdgpu_hostcall_provider_device_info_t* device_info,
    iree_byte_span_t shared_memory, uint64_t device_address,
    uint64_t notification_token,
    iree_hal_amdgpu_error_callback_t error_callback, void** out_context);

// Services all provider work currently ready in one physical-device context.
//
// The provider owns protocol outcomes. Structural failures are published
// through the error callback captured during initialization.
typedef void(IREE_API_PTR* iree_hal_amdgpu_hostcall_provider_service_fn_t)(
    void* context);

// Deinitializes one provider context after its service thread has joined.
typedef void(IREE_API_PTR* iree_hal_amdgpu_hostcall_provider_deinitialize_fn_t)(
    void* context);

// Immutable opaque hostcall provider copied during device creation.
typedef struct iree_hal_amdgpu_hostcall_provider_t {
  // Provider-owned data passed to requirement and initialization callbacks.
  void* user_data;

  // Queries shared allocation requirements for a physical device.
  iree_hal_amdgpu_hostcall_provider_query_requirements_fn_t query_requirements;

  // Initializes one physical-device provider context.
  iree_hal_amdgpu_hostcall_provider_initialize_fn_t initialize;

  // Services ready work on the listener thread.
  iree_hal_amdgpu_hostcall_provider_service_fn_t service;

  // Deinitializes the provider context after listener shutdown.
  iree_hal_amdgpu_hostcall_provider_deinitialize_fn_t deinitialize;
} iree_hal_amdgpu_hostcall_provider_t;

// AMDGPU device creation extension enabling an opaque hostcall provider.
//
// The provider value is copied during device creation. Provider |user_data|
// must remain valid until the created device is destroyed.
typedef struct iree_hal_amdgpu_hostcall_provider_extension_t {
  // Common device creation extension prefix.
  iree_hal_device_create_params_extension_t base;

  // Provider instantiated once for each physical device.
  iree_hal_amdgpu_hostcall_provider_t provider;
} iree_hal_amdgpu_hostcall_provider_extension_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

// Controls where the queue operates.
typedef enum iree_hal_amdgpu_queue_placement_e {
  // Automatically select the best supported placement. Today this selects the
  // host queue path because device-side queue scheduling is not implemented.
  IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY = 0,
  // Queue executes entirely on the host via iree_hal_amdgpu_host_queue_t.
  // This introduces additional latency on all queue operations but can operate
  // on systems without host/device atomics (PCIe atomics, xGMI, etc). It is
  // also useful for debugging.
  IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST,
  // Queue executes entirely on the device. Not implemented; requests for this
  // explicit placement fail during device option verification.
  IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE,
} iree_hal_amdgpu_queue_placement_t;

// Selects the command-buffer encoding and execution path.
typedef enum iree_hal_amdgpu_command_buffer_mode_e {
  // Records and replays command buffers as AMDGPU AQL command-buffer programs.
  IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL = 0,
  // Records dispatch-only reusable command buffers into resident PM4 IBs and
  // submits them through AQL PM4-IB envelopes.
  IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4 = 1,
  // Automatically selects PM4 when the command-buffer request and physical
  // device meet every currently validated PM4 requirement; otherwise uses AQL.
  IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO = 2,
} iree_hal_amdgpu_command_buffer_mode_t;

// Selects how PM4 command buffers publish materialized resident storage.
typedef enum iree_hal_amdgpu_pm4_command_buffer_publication_mode_e {
  // Reserved direct-resident materialization mode. This is not accepted by the
  // driver because CPU writes into executable device-local PM4 storage are not
  // a validated publication path.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_DIRECT = 0,
  // Materializes into host-owned staging builders, then copies each populated
  // resident segment into the HSA allocation with hsa_memory_copy.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_COPY = 1,
  // Materializes into a host-owned staging image, then copies the contiguous
  // resident image into the HSA allocation with hsa_amd_memory_async_copy.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_ASYNC_COPY = 2,
  // Like HOST_ASYNC_COPY, but end() returns after launching the async copy.
  // Queue execution waits on copy completion before the PM4 IB can execute.
  IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_ASYNC_COPY_NONBLOCKING =
      3,
} iree_hal_amdgpu_pm4_command_buffer_publication_mode_t;

// Device-visible virtual shadow reservation size used by ASAN by default.
#define IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SIZE \
  ((iree_device_size_t)32ull << 40)

// HAL-owned application virtual address window size used by ASAN by default.
#define IREE_HAL_AMDGPU_ASAN_DEFAULT_OWNED_APPLICATION_SIZE \
  ((iree_device_size_t)8ull << 40)

// Preferred base address for the ASAN-owned application allocation window.
#define IREE_HAL_AMDGPU_ASAN_PREFERRED_APPLICATION_WINDOW_BASE \
  ((uint64_t)0x0000600000000000ull)

// Maximum log2 application bytes representable by one shadow byte while
// keeping poison magic values distinguishable from partial-granule lengths.
#define IREE_HAL_AMDGPU_ASAN_MAX_SHADOW_SCALE_SHIFT 7u

// Physical shadow slab size used by ASAN by default.
#define IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SLAB_SIZE \
  ((iree_device_size_t)128 * 1024 * 1024)

// Freed ASAN allocation mapping budget kept resident for stale-pointer checks.
#define IREE_HAL_AMDGPU_ASAN_DEFAULT_QUARANTINE_SIZE \
  ((iree_device_size_t)256 * 1024 * 1024)

// Device-visible TSAN shadow entry size in bytes.
#define IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE 8u

// Per-workgroup TSAN shadow header size in bytes.
#define IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_SIZE 8u

// Default log2 local-memory bytes represented by one TSAN shadow entry.
#define IREE_HAL_AMDGPU_TSAN_DEFAULT_MEMORY_GRANULE_SHIFT 2u

// Default local-memory byte capacity represented for each workgroup.
// Zero selects the backend default group segment limit.
#define IREE_HAL_AMDGPU_TSAN_DEFAULT_WORKGROUP_LOCAL_MEMORY_SIZE 0u

// Default number of workgroup ordinals represented by one dispatch shadow.
#define IREE_HAL_AMDGPU_TSAN_DEFAULT_WORKGROUP_CAPACITY 256u

// Default number of queue-local dispatch shadow slots.
#define IREE_HAL_AMDGPU_TSAN_DEFAULT_SHADOW_SLOT_COUNT 16u

// Host-build compatibility for an AMDGPU logical-device option set.
typedef uint32_t iree_hal_amdgpu_logical_device_host_compatibility_t;
enum iree_hal_amdgpu_logical_device_host_compatibility_e {
  // The host build does not rule out this option set before device creation.
  //
  // This does not imply that the requested feature is supported by the
  // selected hardware or runtime. Full support is established by the normal
  // device creation and device-spec query paths.
  IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_COMPATIBLE = 0,
  // Host ThreadSanitizer reserves process virtual address space in a way that
  // is incompatible with AMDGPU ASAN's production sparse address layout.
  IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_INCOMPATIBLE_HOST_TSAN_ASAN =
      1,
};

// Selects how AMDGPU ASAN reports affect the owning logical device.
typedef enum iree_hal_amdgpu_asan_report_policy_e {
  // Emit ASAN reports through the device event sink and keep the logical device
  // usable for subsequent work.
  IREE_HAL_AMDGPU_ASAN_REPORT_POLICY_REPORT_ONLY = 0,
  // Emit ASAN reports through the device event sink and then fail the logical
  // device so queue users observe the violation as device loss.
  IREE_HAL_AMDGPU_ASAN_REPORT_POLICY_FAIL_DEVICE = 1,
} iree_hal_amdgpu_asan_report_policy_t;

// Selects how AMDGPU TSAN reports affect the owning logical device. TSAN
// reports always stop the offending kernel path after the report is emitted.
typedef enum iree_hal_amdgpu_tsan_report_policy_e {
  // Emit TSAN reports through the device event sink and keep the logical
  // device usable for subsequent work.
  IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_REPORT_ONLY = 0,
  // Emit TSAN reports through the device event sink and then fail the logical
  // device so queue users observe the violation as device loss.
  IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_FAIL_DEVICE = 1,
} iree_hal_amdgpu_tsan_report_policy_t;

// Selects how ASAN shadow virtual address space is mapped.
typedef enum iree_hal_amdgpu_asan_shadow_mode_e {
  // Reserve shadow virtual address space and map physical shadow slabs only
  // when allocation/import publication touches them.
  IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_SPARSE = 0,
  // Premap every shadow slab to a shared poisoned physical slab, then replace
  // aliases with precise writable slabs as allocation/import publication
  // touches them.
  IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_PREMAPPED = 1,
} iree_hal_amdgpu_asan_shadow_mode_t;

// Selects the physical memory backing ASAN shadow slabs.
typedef enum iree_hal_amdgpu_asan_shadow_backing_e {
  // Back shadow slabs with device-local VRAM from the representative physical
  // device. This keeps instrumented shadow reads local to the GPU and is the
  // default production policy.
  IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL = 0,
  // Back shadow slabs with pinned host memory mapped for the logical-device
  // topology. Host-local shadow updates must happen before queue submissions
  // whose dispatches read them, or after waited work retires before release
  // poisoning mutates them.
  IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_HOST_LOCAL = 1,
} iree_hal_amdgpu_asan_shadow_backing_t;

// Parameters configuring an iree_hal_amdgpu_logical_device_t.
// Must be initialized with iree_hal_amdgpu_logical_device_options_initialize
// prior to use.
typedef struct iree_hal_amdgpu_logical_device_options_t {
  // Size of a block in each host block pool.
  struct {
    // Small host block pool options.
    struct {
      // Size in bytes of a small host block. Must be a power of two.
      iree_host_size_t block_size;
    } small;
    // Large host block pool options.
    struct {
      // Size in bytes of a large host block. Must be a power of two.
      iree_host_size_t block_size;
    } large;
    // Command-buffer host block pool options.
    struct {
      // Usable byte capacity of a command-buffer recording block. Must be a
      // power of two.
      iree_host_size_t usable_block_size;
    } command_buffer;
  } host_block_pools;

  // Size of a block in each device block pool.
  struct {
    struct {
      // Size in bytes of a small device block. Must be a power of two.
      iree_device_size_t block_size;
      // Initial small block pool block allocation count.
      iree_host_size_t initial_capacity;
    } small;
    struct {
      // Size in bytes of a large device block. Must be a power of two.
      iree_device_size_t block_size;
      // Initial large block pool block allocation count.
      iree_host_size_t initial_capacity;
    } large;
  } device_block_pools;

  // Default queue-allocation pool policy.
  struct {
    // Logical byte length of the default TLSF pool range per physical device.
    iree_device_size_t range_length;

    // Minimum byte alignment for every default-pool reservation.
    iree_device_size_t alignment;

    // Maximum death-frontier entry count stored per free TLSF block.
    uint8_t frontier_capacity;
  } default_pool;

  // Controls where queues are placed. ANY and HOST currently select host
  // queues. DEVICE is reserved for future device-side scheduling and fails
  // loudly until that path is implemented.
  iree_hal_amdgpu_queue_placement_t queue_placement;

  // Selects the command-buffer recording and replay implementation.
  iree_hal_amdgpu_command_buffer_mode_t command_buffer_mode;

  // Selects how PM4 command buffers publish their resident storage.
  iree_hal_amdgpu_pm4_command_buffer_publication_mode_t
      pm4_command_buffer_publication_mode;

  // Per-physical-device host queue policy.
  struct {
    // HSA AQL ring capacity in packets for each host queue. Must be a power of
    // two. Larger rings allow more in-flight packet work before submitters see
    // AQL backpressure.
    uint32_t aql_capacity;
    // Completion/reclaim ring capacity for each host queue. Must be a power of
    // two. This bounds in-flight host-visible completion epochs before replay
    // must park and resume after drain.
    uint32_t notification_capacity;
    // Kernarg ring capacity in 64-byte blocks for each host queue. Must be a
    // power of two and at least 2x |aql_capacity| to cover one tail-padding
    // gap at wrap. Submission admission checks kernarg and AQL capacity
    // together before publishing packets.
    uint32_t kernarg_capacity;
    // Device-visible control upload ring capacity in bytes for each host queue.
    // Zero disables the optional upload ring in AQL mode. PM4 and automatic
    // modes resolve zero to the capacity required for PM4 command buffers.
    // Non-zero values must be powers of two. This carries queue-ordered
    // metadata such as device-side command-buffer fixup inputs without using
    // the file staging pool.
    uint32_t upload_capacity;
  } host_queues;

  // Per-physical-device queue_read/queue_write file staging policy.
  struct {
    // Byte length of each staging slot. Must be a non-zero power of two.
    iree_host_size_t slot_size;
    // Number of staging slots. Must be non-zero and a power of two.
    uint32_t slot_count;
    // True to force fine-grained host memory instead of coarse-grained memory.
    uint64_t force_fine_host_memory : 1;
  } file_staging;

  // Optional device-side feedback channel support.
  struct {
    // True to reserve feedback channel state for the logical device.
    uint64_t enabled : 1;
  } feedback;

  // Optional ASAN device-side checking support.
  struct {
    // True to reserve ASAN shadow state for the logical device.
    uint64_t enabled : 1;

    // Policy applied after a valid ASAN report is emitted.
    iree_hal_amdgpu_asan_report_policy_t report_policy;

    // Shadow mapping policy used for the reserved shadow address space.
    iree_hal_amdgpu_asan_shadow_mode_t shadow_mode;

    // Physical memory placement policy used for shadow slabs.
    iree_hal_amdgpu_asan_shadow_backing_t shadow_backing;

    // Log2 application bytes represented by one shadow byte.
    uint32_t shadow_scale_shift;

    // Device-visible virtual shadow reservation size in bytes.
    iree_device_size_t shadow_size;

    // HAL-owned application virtual address reservation size in bytes.
    iree_device_size_t owned_application_size;

    // Physical shadow slab size in bytes.
    iree_device_size_t shadow_slab_size;

    // Freed allocation mapping budget in bytes kept resident and poisoned.
    iree_device_size_t quarantine_size;
  } asan;

  // Optional TSAN device-side race checking support.
  struct {
    // True to reserve TSAN shadow state for the logical device.
    uint64_t enabled : 1;

    // Policy applied after a valid TSAN report is emitted.
    iree_hal_amdgpu_tsan_report_policy_t report_policy;

    // Log2 local-memory bytes represented by one shadow entry.
    uint32_t memory_granule_shift;

    // Local-memory byte capacity represented for each workgroup. Zero selects
    // the backend default group segment limit.
    uint32_t workgroup_local_memory_size;

    // Maximum workgroup ordinals represented by one dispatch shadow.
    uint32_t workgroup_capacity;

    // Number of queue-local dispatch shadow slots available. Command-buffer
    // recording may insert execution barriers when a TSAN-instrumented span
    // would otherwise exceed this window.
    uint32_t shadow_slot_count;
  } tsan;

  // Preallocates a reasonable number of resources in pools to reduce initial
  // execution latency.
  uint64_t preallocate_pools : 1;

  // Reserved for a future exclusive queue scheduling mode. Unsupported today;
  // enabling it fails option verification.
  uint64_t exclusive_execution : 1;

  // Forces cross-queue wait barriers to use software deferral instead of the
  // device-side strategy selected from the GPU ISA. Useful for testing the
  // conservative host-only fallback path.
  uint64_t force_wait_barrier_defer : 1;

  // Enables PM4 dispatch command-buffer capabilities on unvalidated gfx9-gfx12
  // targets. This is an explicit hardware bring-up override; automatic PM4
  // selection remains limited to validated GPU ISAs when this is unset.
  uint64_t enable_experimental_pm4_command_buffers : 1;

  // Suppresses fine-grained GPU-local memory pools even if the HSA agent
  // reports them. This is a hardware bring-up and compatibility testing
  // override for validating the coarse-grained device-local memory path used on
  // GPUs that do not expose host-coherent VRAM.
  uint64_t suppress_device_fine_memory : 1;

  // Reserved for future HSA active-wait tuning. Must be zero today because no
  // wait path consumes it yet.
  iree_duration_t wait_active_for_ns;
} iree_hal_amdgpu_logical_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_amdgpu_logical_device_options_initialize(
    iree_hal_amdgpu_logical_device_options_t* out_options);

// Queries whether |options| are compatible with this host build configuration.
//
// This is not full option validation and does not query HSA. It exists so tests
// and tools can skip known-impossible host/sanitizer combinations before
// attempting device creation.
IREE_API_EXPORT iree_hal_amdgpu_logical_device_host_compatibility_t
iree_hal_amdgpu_logical_device_options_query_host_compatibility(
    const iree_hal_amdgpu_logical_device_options_t* options);

// Parses |params| and updates |options|. No AMDGPU logical-device string
// parameters are currently supported; nonempty lists fail loudly instead of
// being ignored.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_options_parse(
    iree_hal_amdgpu_logical_device_options_t* options,
    iree_string_pair_list_t params);

// Creates a AMDGPU HAL device with the given |options| and |topology|.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |options|, |libhsa|, and |topology| will be cloned into the device and need
// not live beyond the call.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Queries immutable metadata for one executable function.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_executable_query_function_attributes(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    uint32_t* out_maximum_workgroup_invocations,
    uint32_t* out_workgroup_local_memory_size,
    uint32_t* out_private_memory_size, uint32_t* out_register_count,
    uint32_t* out_maximum_dynamic_workgroup_local_memory_size,
    bool* out_requires_uniform_workgroups);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_amdgpu_driver_t.
// Must be initialized with iree_hal_amdgpu_driver_options_initialize prior to
// use.
typedef struct iree_hal_amdgpu_driver_options_t {
  // Search paths (directories or files) for finding the HSA runtime shared
  // library. Driver creation clones these strings; callers only need to keep
  // them live until iree_hal_amdgpu_driver_create returns.
  iree_string_view_list_t libhsa_search_paths;

  // Default device options when none are provided during device creation.
  iree_hal_amdgpu_logical_device_options_t default_device_options;
} iree_hal_amdgpu_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_amdgpu_driver_options_initialize(
    iree_hal_amdgpu_driver_options_t* out_options);

// Parses |params| and updates |options|. No AMDGPU driver string parameters are
// currently supported; nonempty lists fail loudly instead of being ignored.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_options_parse(
    iree_hal_amdgpu_driver_options_t* options, iree_string_pair_list_t params);

// Creates a AMDGPU HAL driver with the given |options|, from which AMDGPU
// devices can be enumerated and created with specific parameters.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_API_H_
