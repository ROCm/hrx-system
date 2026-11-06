// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_QUEUE_EXECUTION_RESOURCES_H_
#define IREE_HAL_DRIVER_AMDGPU_QUEUE_EXECUTION_RESOURCES_H_

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"
#include "iree/hal/driver/amdgpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable mapping between HAL execution resources and a native HSA CU mask.
typedef struct iree_hal_amdgpu_queue_execution_resource_topology_t {
  // Number of family-local raw execution units addressable by the native mask.
  uint32_t execution_unit_count;
  // Number of adjacent native mask bits forming one selectable HAL resource.
  uint32_t execution_units_per_resource;
  // Number of hardware partitions interleaved across native mask bits.
  uint32_t partition_count;
} iree_hal_amdgpu_queue_execution_resource_topology_t;

// Verifies that |topology| describes a representable native queue domain.
iree_status_t iree_hal_amdgpu_queue_execution_resource_topology_verify(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology);

// Derives and validates the queue execution-resource topology for an agent.
//
// HSA queue masks use architecture-specific granularity and interleave mask
// bits across hardware partitions. Every independently selectable HAL resource
// covers one representable native mask group. A zero mask for one partition is
// interpreted as unconfined by the native stack, so each partition becomes a
// HAL resource constraint group requiring at least one selected resource.
// |out_topology| is unchanged on failure.
iree_status_t iree_hal_amdgpu_queue_execution_resource_topology_initialize(
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    uint32_t execution_unit_count, uint32_t partition_count,
    iree_hal_amdgpu_queue_execution_resource_topology_t* out_topology);

// Returns the number of constraint groups exposed by |topology|.
iree_host_size_t iree_hal_amdgpu_queue_execution_resource_group_count(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology);

// Returns the number of independently selectable resources in |topology|.
iree_host_size_t iree_hal_amdgpu_queue_execution_resource_count(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology);

// Returns the native CU-mask bit count required by HSA for |topology|.
uint32_t iree_hal_amdgpu_queue_execution_resource_mask_bit_count(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology);

// Populates exactly queue_execution_resource_group_count constraint groups.
void iree_hal_amdgpu_queue_execution_resource_populate_groups(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    iree_hal_queue_execution_resource_group_spec_t* out_groups);

// Populates exactly queue_execution_resource_count resource records.
void iree_hal_amdgpu_queue_execution_resource_populate_resources(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    iree_hal_queue_execution_resource_spec_t* out_resources);

// Lowers |resources| to the exact native CU mask for |topology|.
//
// An empty resource list selects every advertised resource. Nonempty lists
// must contain valid resource ordinals and retain at least one resource in
// every hardware partition. |out_mask| must contain exactly
// queue_execution_resource_mask_bit_count bits and is unchanged on failure.
iree_status_t iree_hal_amdgpu_queue_execution_resource_write_mask(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    iree_hal_queue_execution_resource_list_t resources,
    uint32_t out_mask_bit_count, uint32_t* out_mask);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_QUEUE_EXECUTION_RESOURCES_H_
