// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SPEC_BUILDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SPEC_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/drivers/amdgpu/target/identity.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// AMDGPU physical-device parameter flags.
typedef uint32_t iree_hal_amdgpu_device_spec_physical_device_flags_t;
typedef enum iree_hal_amdgpu_device_spec_physical_device_flag_bits_e {
  // No optional physical-device parameters are present.
  IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_NONE = 0u,
  // Physical device UUID bytes are present.
  IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID = 1u << 0,
  // PCI address fields are present.
  IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS = 1u << 1,
} iree_hal_amdgpu_device_spec_physical_device_flag_bits_t;

// Immutable AMDGPU physical-device facts used to create a device spec.
typedef struct iree_hal_amdgpu_device_spec_physical_device_params_t {
  // Parsed target identity for this physical device.
  iree_hal_amdgpu_target_identity_t identity;
  // Stable physical device UUID bytes.
  iree_hal_uuid_t uuid;
  // PCI address.
  iree_hal_pci_address_t pci;
  // Device-side timestamp tick rate in hz, from
  // HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY. Must be nonzero.
  uint64_t timestamp_frequency_hz;
  // Host NUMA node nearest this physical device.
  iree_hal_numa_node_t numa;
  // Physical device ordinal within the AMDGPU topology.
  uint32_t physical_ordinal;
  // Host queue count initialized for this physical device.
  uint32_t queue_count;
  // Compute unit count visible on this physical device.
  uint32_t compute_unit_count;
  // Native wavefront size in lanes.
  uint32_t wavefront_size;
  // Maximum resident wave count per compute unit.
  uint32_t maximum_waves_per_compute_unit;
  // Maximum workgroup local-memory byte length.
  uint32_t maximum_workgroup_local_memory_size;
  // Immutable vendor packet capabilities available on this physical device.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Optional physical-device parameter flags.
  iree_hal_amdgpu_device_spec_physical_device_flags_t flags;
} iree_hal_amdgpu_device_spec_physical_device_params_t;

// AMDGPU device spec parameter flags.
typedef uint32_t iree_hal_amdgpu_device_spec_param_flags_t;
typedef enum iree_hal_amdgpu_device_spec_param_flag_bits_e {
  // No optional device spec parameters are present.
  IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_NONE = 0u,
  // DMA-BUF import/export is supported.
  IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_DMABUF = 1u << 0,
} iree_hal_amdgpu_device_spec_param_flag_bits_t;

// Parameters for creating an AMDGPU HAL device spec.
typedef struct iree_hal_amdgpu_device_spec_params_t {
  // Stable logical device identifier.
  iree_string_view_t logical_device_id;
  // Human-readable logical device name.
  iree_string_view_t display_name;
  // Number of physical devices in |physical_devices|.
  iree_host_size_t physical_device_count;
  // Physical devices covered by the logical device.
  const iree_hal_amdgpu_device_spec_physical_device_params_t* physical_devices;
  // Number of queue-affinity lanes reserved for each physical device. Must be
  // at least each physical device's exposed queue count.
  uint32_t queue_affinity_stride;
  // Total logical device-local memory capacity in bytes, if known.
  uint64_t device_memory_capacity_bytes;
  // Device allocator used to query stable allocation classes.
  iree_hal_allocator_t* device_allocator;
  // Sanitizer configuration selected for this logical device.
  iree_hal_device_sanitizer_spec_t sanitizer;
  // Optional AMDGPU device spec parameter flags.
  iree_hal_amdgpu_device_spec_param_flags_t flags;
} iree_hal_amdgpu_device_spec_params_t;

// Creates an immutable spec for an AMDGPU HAL device.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_device_spec_create(
    const iree_hal_amdgpu_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SPEC_BUILDER_H_
