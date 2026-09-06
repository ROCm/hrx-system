// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/kernarg_layout.h"
#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"
#include "iree/hal/drivers/amdgpu/profile_metadata.h"
#include "iree/hal/drivers/amdgpu/queue_scope.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

typedef struct iree_hal_amdgpu_asan_state_t iree_hal_amdgpu_asan_state_t;
typedef struct iree_hal_amdgpu_feedback_state_t
    iree_hal_amdgpu_feedback_state_t;
typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;
typedef struct iree_hal_amdgpu_tsan_state_t iree_hal_amdgpu_tsan_state_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_t
//===----------------------------------------------------------------------===//

// Immutable launch limits precomputed for one export on one physical device.
typedef struct iree_hal_amdgpu_executable_dispatch_limits_t {
  // Maximum total workgroup size accepted by both the function and device.
  uint32_t maximum_workgroup_invocations;
  // Maximum XYZ workgroup sizes accepted by the physical device.
  uint16_t maximum_workgroup_size[3];
  // Maximum dynamic workgroup-local memory accepted by the function.
  uint32_t maximum_dynamic_workgroup_local_memory_size;
} iree_hal_amdgpu_executable_dispatch_limits_t;

// Validates a caller-provided workgroup size against |dispatch_limits|.
iree_status_t
iree_hal_amdgpu_executable_dispatch_limits_validate_workgroup_size(
    const iree_hal_amdgpu_executable_dispatch_limits_t* dispatch_limits,
    const uint32_t workgroup_size[3]);

// Host-resident dispatch metadata precomputed for one executable export on one
// physical device.
//
// Descriptors are immutable after executable creation and remain valid for the
// lifetime of the executable. They intentionally duplicate the device-visible
// kernel argument table in ordinary host memory so queue submission does not
// read per-dispatch metadata from memory allocated for GPU visibility.
typedef struct iree_hal_amdgpu_executable_dispatch_descriptor_t {
  // Device-specific kernel arguments with a valid kernel_object for dispatch.
  iree_hal_amdgpu_device_kernel_args_t kernel_args;
  // Native kernarg layout for normal metadata-described dispatches.
  const iree_hal_amdgpu_kernarg_layout_t* kernarg_layout;
  // Custom direct-argument kernarg layout derived from |kernel_args|.
  iree_hal_amdgpu_device_dispatch_kernarg_layout_t custom_kernarg_layout;
  // Queue kernarg-ring block count for normal metadata-described dispatches.
  uint32_t kernarg_block_count;
  // Queue kernarg-ring block count for custom direct-argument dispatches.
  uint32_t custom_kernarg_block_count;
  // Maximum static workgroup count accepted for each dimension.
  uint32_t maximum_workgroup_count[3];
  // Function and physical-device dispatch limits.
  iree_hal_amdgpu_executable_dispatch_limits_t limits;
  // Cluster-count limits for this descriptor's physical device.
  iree_hal_amdgpu_dispatch_dimension_limits_t workgroup_cluster_count_limits;
  // Physical device ordinal owning |workgroup_cluster_count_limits|.
  iree_host_size_t physical_device_ordinal;
  // PM4 launch state for the default executable workgroup size.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t pm4_launch_state;
  // PM4 setup packet dwords for |pm4_launch_state|.
  uint32_t pm4_setup_dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT];
  // Number of valid dwords in |pm4_setup_dwords|.
  uint32_t pm4_setup_dword_count;
  // AMDHSA descriptor fixed group segment byte size validated for PM4 launch.
  uint32_t pm4_group_segment_fixed_size;
  // True when this export has no reflected parameter metadata and may only be
  // launched with a caller-provided native kernarg buffer.
  bool custom_direct_only;
  // True when the PM4 metadata fields are valid for this dispatch.
  bool pm4_launch_state_valid;
} iree_hal_amdgpu_executable_dispatch_descriptor_t;

// Creates an AMDGPU executable from a binary in memory. Each executable may
// contain multiple entry points and be composed of several modules presented to
// the HAL as a single instance. See iree_hal_executable_load_params_t for more
// information about the lifetime of the resources referenced within.
//
// |libhsa| and |topology| are captured by-reference and must remain valid for
// the lifetime of the executable.
//
// |queue_family| selects the physical device onto which the executable is
// loaded. It and |target| must be exact borrowed rows from |device|'s immutable
// state, and |target| must cover the family's physical device.
//
// |executable_id| is a non-zero logical-device-local identifier assigned to
// this executable before it is visible to profiling or device-originated
// diagnostics.
//
// |asan_state| is captured by-reference for this load and used to publish ASAN
// config globals when enabled. It may be NULL when ASAN is unavailable.
//
// |feedback_state| is captured by-reference for this load and used to publish
// feedback config globals when enabled. It may be NULL when feedback is
// unavailable.
//
// |tsan_state| is captured by-reference for this load and used to publish TSAN
// config globals when enabled. It may be NULL when TSAN is unavailable.
//
// |queue_scopes| captures immutable queue identities for the owning logical
// queue family. Executables with queue-scoped globals may load one HSA
// executable variant per physical queue ordinal and publish per-queue config
// without mutating state on each dispatch.
//
// |physical_device_list| contains |physical_device_count| devices in topology
// ordinal order. It is used only during creation to validate metadata against
// every selected device's immutable capabilities.
//
// Exact code-object image bytes and loader load ranges are retained in profile
// metadata for offline trace/disassembly workflows. Executable trace profiling
// may begin after executable preparation, so this cold-path metadata is always
// durable instead of being gated on an active profiling session.
iree_status_t iree_hal_amdgpu_executable_create(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    uint64_t executable_id, iree_hal_amdgpu_feedback_state_t* feedback_state,
    iree_hal_amdgpu_asan_state_t* asan_state,
    iree_hal_amdgpu_tsan_state_t* tsan_state,
    iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_device_list,
    iree_host_size_t queue_scope_count,
    const iree_hal_amdgpu_queue_scope_t* queue_scopes,
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the logical-device-local executable id assigned at creation.
uint64_t iree_hal_amdgpu_executable_id(iree_hal_executable_t* executable);

// Returns metadata about an exported kernel function in host memory.
// The returned pointers will remain valid for the lifetime of the executable.
// The returned kernel_object field is undefined in the returned args as there
// is no host representation and objects are per agent. To get an agent-specific
// kernel_object use iree_hal_amdgpu_executable_lookup_kernel_args_for_device.
iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_host(
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args);

// Returns metadata about an exported kernel function in device memory.
// Kernel arguments are specific to the physical device specified by
// |device_ordinal| in the topology and cannot be used on any other device. The
// lookup fails if the executable's queue family does not service
// |device_ordinal|. The returned pointers will remain valid for the lifetime of
// the executable.
iree_status_t iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args);

// Returns host-resident dispatch metadata for an exported kernel function on a
// physical device.
//
// The returned descriptor is specific to |device_ordinal| because the kernel
// object embedded in the dispatch packet is per device. The lookup fails if the
// executable's queue family does not service |device_ordinal|. The pointer
// remains valid for the lifetime of the executable.
iree_status_t iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor);

// Returns host-resident dispatch metadata for an exported kernel function on a
// queue.
//
// Queue-scoped executable variants require this lookup so the selected kernel
// object and executable globals match the queue that will receive the dispatch.
// Non-queue-scoped executables collapse this to the existing per-device lookup.
iree_status_t iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue(
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor);

// Returns true when dispatches must select executable metadata by queue.
bool iree_hal_amdgpu_executable_requires_queue_scope(
    iree_hal_executable_t* executable);

// Returns PM4 compute launch state for an exported kernel function on a
// physical device.
//
// The returned state is precomputed at executable load from the HSA kernel
// object and immutable executable metadata. It is suitable for recording into
// a PM4 command-buffer block; dynamic kernarg/userdata values are intentionally
// not embedded. The pointer remains valid for the lifetime of the executable.
iree_status_t
iree_hal_amdgpu_executable_lookup_pm4_dispatch_launch_state_for_device(
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    iree_host_size_t device_ordinal,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t** out_state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_H_
