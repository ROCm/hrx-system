// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/device/grid_sync.h"

#if !defined(IREE_AMDGPU_TARGET_DEVICE)

#include <inttypes.h>

iree_status_t iree_hal_amdgpu_grid_sync_info_initialize(
    iree_hal_amdgpu_grid_sync_strategy_t strategy,
    const uint32_t workgroup_count[3], const uint16_t workgroup_size[3],
    iree_amdgpu_grid_sync_info_t* out_info) {
  uint64_t maximum_workgroup_count = 0;
  switch (strategy) {
    case IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_MEMORY:
      maximum_workgroup_count = UINT16_MAX;
      break;
    case IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_GWS:
      maximum_workgroup_count = UINT32_MAX;
      break;
    case IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_NONE:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cooperative grid synchronization strategy is unavailable");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unrecognized grid sync strategy %d", strategy);
  }

  iree_device_size_t total_workgroup_count = 1;
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(!iree_device_size_checked_mul(total_workgroup_count,
                                                    workgroup_count[i],
                                                    &total_workgroup_count))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cooperative dispatch workgroup count product overflows uint64_t");
    }
  }
  if (IREE_UNLIKELY(total_workgroup_count > maximum_workgroup_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "cooperative dispatch has %" PRIdsz
        " workgroups but the grid sync strategy supports at most %" PRIu64,
        total_workgroup_count, maximum_workgroup_count);
  }

  iree_device_size_t total_workitem_count = total_workgroup_count;
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(!iree_device_size_checked_mul(
            total_workitem_count, workgroup_size[i], &total_workitem_count))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cooperative dispatch workitem count product overflows uint64_t");
    }
  }

  iree_amdgpu_grid_sync_info_t info = {0};
  if (total_workgroup_count != 0) {
    info.grid_count = 1;
    info.total_workitem_count = total_workitem_count;
    info.workgroup_count = (uint32_t)total_workgroup_count;
  }
  *out_info = info;
  return iree_ok_status();
}

void iree_hal_amdgpu_device_grid_sync_gws_initialize_emplace(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        initialize_kernel_args,
    uint32_t workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr, uint16_t* out_setup) {
  iree_hal_amdgpu_device_grid_sync_gws_initialize_kernargs_t*
      IREE_AMDGPU_RESTRICT kernargs =
          (iree_hal_amdgpu_device_grid_sync_gws_initialize_kernargs_t*)
              kernarg_ptr;
  kernargs->workgroup_count_minus_one = workgroup_count - 1;

  const uint32_t initialize_workgroup_count[3] = {1, 1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      initialize_kernel_args, initialize_workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, dispatch_packet, kernarg_ptr,
      out_setup);
}

#endif  // !IREE_AMDGPU_TARGET_DEVICE

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Hardware GWS is used only on the GFX9 and GFX10 architectures where ROCm's
// cooperative-groups device library uses the same instruction. The target
// feature lets LLVM select the instruction without applying GWS requirements
// to every builtin in the linked code object.
//
// Newer architectures use a memory-backed grid barrier. Their code objects
// retain this inert kernel because builtin kernel descriptors are resolved as
// one eager table, but the host must never submit it.
#if defined(__GFX9__) || defined(__GFX10__)
#define IREE_HAL_AMDGPU_DEVICE_GRID_SYNC_GWS_ATTRIBUTE \
  __attribute__((target("gws")))
#else
#define IREE_HAL_AMDGPU_DEVICE_GRID_SYNC_GWS_ATTRIBUTE
#endif  // __GFX9__ || __GFX10__

IREE_AMDGPU_ATTRIBUTE_KERNEL IREE_HAL_AMDGPU_DEVICE_GRID_SYNC_GWS_ATTRIBUTE void
iree_hal_amdgpu_device_grid_sync_gws_initialize(
    uint32_t workgroup_count_minus_one) {
#if defined(__GFX9__) || defined(__GFX10__)
  __builtin_amdgcn_ds_gws_init(workgroup_count_minus_one, 0);
#else
  (void)workgroup_count_minus_one;
#endif  // __GFX9__ || __GFX10__
}

#undef IREE_HAL_AMDGPU_DEVICE_GRID_SYNC_GWS_ATTRIBUTE

#endif  // IREE_AMDGPU_TARGET_DEVICE
