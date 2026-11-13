// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_FAMILY_SPEC_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_FAMILY_SPEC_H_

#include "iree/hal/device_spec.h"
#include "iree/hal/driver/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold native-family facts borrowed by a public spec until its builder copies
// them. This owner acquires no queue, backing, or execution resources.
typedef struct iree_hal_amd_gpu_queue_family_spec_storage_t {
  // Sorted native priority levels accepted by the selected carrier.
  iree_hal_queue_priority_t priorities[3];
  // Indivisible compute group when native resource selection is unavailable.
  iree_hal_queue_execution_resource_group_spec_t resource_group;
  // Complete indivisible compute resource; transfer queues expose no CUs.
  iree_hal_queue_execution_resource_spec_t execution_resource;
  // Owned selectable resource groups followed by their resource descriptions.
  iree_hal_queue_execution_resource_group_spec_t* resource_groups;
} iree_hal_amd_gpu_queue_family_spec_storage_t;

// Translates an admitted native family's implemented execution capabilities.
// The caller selects the native carrier before this query. Storage must be
// deinitialized after either success or failure.
iree_status_t iree_hal_amd_gpu_queue_family_spec_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    amdf_queue_publication_modes_t publication_mode, uint32_t queue_count,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_queue_family_spec_storage_t* out_storage,
    iree_hal_queue_family_spec_t* out_spec);

void iree_hal_amd_gpu_queue_family_spec_deinitialize(
    iree_hal_amd_gpu_queue_family_spec_storage_t* storage,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_FAMILY_SPEC_H_
