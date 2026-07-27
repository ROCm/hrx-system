// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_DEVICE_SPEC_BUILDER_H_
#define IREE_HAL_DRIVERS_HIP_DEVICE_SPEC_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_HIP_MAX_GCN_ARCH_NAME_LENGTH 256

// HIP architecture facts used while populating common HAL specs.
typedef struct iree_hal_hip_device_architecture_facts_t {
  // NUL-terminated HIP GCN architecture name.
  char gcn_arch_name[IREE_HAL_HIP_MAX_GCN_ARCH_NAME_LENGTH];
} iree_hal_hip_device_architecture_facts_t;

// HIP launch-limit facts used while populating common HAL specs.
typedef struct iree_hal_hip_device_launch_facts_t {
  // Maximum total threads in one HIP block.
  uint32_t maximum_workgroup_invocations;
  // Maximum HIP block size per dimension.
  uint32_t maximum_workgroup_size[3];
  // Maximum HIP grid size per dimension.
  uint32_t maximum_workgroup_count[3];
  // Maximum resident blocks per compute unit.
  uint32_t maximum_workgroups_per_execution_unit;
  // Maximum resident threads per compute unit.
  uint32_t maximum_invocations_per_execution_unit;
  // Maximum 32-bit registers per HIP block.
  uint32_t maximum_workgroup_register_count;
  // Maximum local memory per compute unit in bytes.
  uint64_t maximum_local_memory_size;
  // Maximum local memory per HIP block in bytes.
  uint64_t maximum_workgroup_local_memory_size;
} iree_hal_hip_device_launch_facts_t;

// HIP clock facts used while populating common HAL specs.
typedef struct iree_hal_hip_device_clock_facts_t {
  // Device-side clock instruction frequency in hertz.
  uint64_t clock_instruction_frequency_hz;
} iree_hal_hip_device_clock_facts_t;

// HIP physical-device facts used while populating common HAL specs.
typedef struct iree_hal_hip_device_facts_t {
  // Architecture facts.
  iree_hal_hip_device_architecture_facts_t architecture;
  // Launch-limit facts.
  iree_hal_hip_device_launch_facts_t launch;
  // Clock facts.
  iree_hal_hip_device_clock_facts_t clocks;
  // Number of compute units visible to the logical device.
  uint32_t execution_unit_count;
  // Native wavefront width in lanes.
  uint32_t subgroup_size;
} iree_hal_hip_device_facts_t;

// HIP physical-device parameter flags.
typedef uint32_t iree_hal_hip_device_spec_physical_device_flags_t;
typedef enum iree_hal_hip_device_spec_physical_device_flag_bits_e {
  // No optional physical-device parameters are present.
  IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_NONE = 0u,
  // Physical device UUID bytes are present.
  IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID = 1u << 0,
  // PCI address fields are present.
  IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS = 1u << 1,
} iree_hal_hip_device_spec_physical_device_flag_bits_t;

// Immutable HIP physical-device facts used to create a device spec.
typedef struct iree_hal_hip_device_spec_physical_device_params_t {
  // Human-readable physical device name.
  iree_string_view_t display_name;
  // Backend-native physical device path.
  iree_string_view_t backend_path;
  // Stable physical device UUID bytes.
  iree_hal_uuid_t uuid;
  // PCI address.
  iree_hal_pci_address_t pci;
  // Physical device ordinal within the HIP topology.
  uint32_t physical_ordinal;
  // HIP facts used to populate common HAL spec fields.
  iree_hal_hip_device_facts_t facts;
  // Optional physical-device parameter flags.
  iree_hal_hip_device_spec_physical_device_flags_t flags;
} iree_hal_hip_device_spec_physical_device_params_t;

// Parameters for creating a HIP HAL device spec.
typedef struct iree_hal_hip_device_spec_params_t {
  // Stable logical device identifier.
  iree_string_view_t logical_device_id;
  // Human-readable logical device name.
  iree_string_view_t display_name;
  // Number of HIP queues exposed on each physical device.
  iree_host_size_t queue_count;
  // Number of physical devices in |physical_devices|.
  iree_host_size_t physical_device_count;
  // Physical devices covered by the logical device.
  const iree_hal_hip_device_spec_physical_device_params_t* physical_devices;
  // Device allocator used to query stable allocation classes.
  iree_hal_allocator_t* device_allocator;
} iree_hal_hip_device_spec_params_t;

// Creates an immutable spec for a HIP HAL device.
IREE_API_EXPORT iree_status_t iree_hal_hip_device_spec_create(
    const iree_hal_hip_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_DEVICE_SPEC_BUILDER_H_
