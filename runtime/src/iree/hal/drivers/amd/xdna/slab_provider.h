// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_SLAB_PROVIDER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_SLAB_PROVIDER_H_

#include "iree/hal/drivers/amd/xdna/memory.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cold native construction contract copied into one slab provider.
typedef struct iree_hal_amd_xdna_slab_provider_options_t {
  // System memory scope used for every native allocation.
  amdf_memory_scope_t* scope;
  // Qualified CREATE and HOST_MAP profile for the complete access set.
  amdf_memory_profile_t profile;
  // Number of caller-ordered native device accesses.
  uint32_t access_count;
  // Borrowed during creation and copied into provider-owned storage.
  const amdf_memory_device_access_t* accesses;
  // HAL memory properties promised by every materialized buffer.
  iree_hal_memory_type_t memory_type;
  // HAL usages implemented by every materialized buffer.
  iree_hal_buffer_usage_t supported_usage;
  // Smallest independently cache-maintained range across all access pairs.
  iree_device_size_t maintenance_alignment;
} iree_hal_amd_xdna_slab_provider_options_t;

// Creates a provider that obtains one fully attached libamdf allocation per
// slab. The provider copies |options.accesses| and borrows |device|, |context|,
// the memory scope, and every native device named by the copied accesses. The
// sealed HAL device group must outlive the provider and all derived objects.
iree_status_t iree_hal_amd_xdna_slab_provider_create(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    const iree_hal_amd_xdna_slab_provider_options_t* options,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_SLAB_PROVIDER_H_
