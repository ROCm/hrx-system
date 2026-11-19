// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_MAPPING_CONTEXT_H_
#define AMDF_SRC_GPU_UMD_DXG_MAPPING_CONTEXT_H_

#include "libamdf/src/gpu/umd/dxg/memory_reservation.h"
#include "libamdf/src/platform/dxg/kmt.h"

#ifdef __cplusplus
extern "C" {
#endif

// Private synchronization-only KMT context shared by the logical device's
// host-completed mapping batches. No command work enters its paging timeline.
typedef struct amdf_gpu_dxg_mapping_context_t {
  // Native context created on the first multi-span update.
  uint32_t handle;
  // Monitored fence and latest accepted value, under the publication lock.
  amdf_gpu_dxg_mapping_point_t latest;
  // Native setup definitively rejected synchronization-only paging on this
  // device.
  bool unsupported;
} amdf_gpu_dxg_mapping_context_t;

// Expands physical allocation boundaries into the exact native MAP_PROTECT
// and UNMAP records shared by execution queues and host-completed batches.
// The reservation allocator owns the returned array.
amdf_status_t amdf_gpu_dxg_mapping_operations_create(
    const amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t update_count,
    const amdf_gpu_umd_mapping_update_t* updates, uint64_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION** out_operations);

// Submits the complete vector at predecessor_value, signaling its successor.
amdf_status_t amdf_gpu_dxg_mapping_operations_submit(
    const amdf_dxg_kmt_t* kmt, uint32_t device_handle, uint32_t context_handle,
    uint32_t fence_handle, uint64_t predecessor_value, uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations);

// Requires exclusive device publication ownership. Preparation failure leaves
// the reservation untouched. A successfully accepted update publishes the
// point the caller must join after dropping publication ownership.
amdf_status_t amdf_gpu_dxg_mapping_context_submit(
    amdf_gpu_dxg_mapping_context_t* context, const amdf_dxg_kmt_t* kmt,
    uint32_t device_handle, uint32_t physical_adapter_index,
    uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations,
    bool* mutation_started, amdf_gpu_dxg_mapping_point_t* out_point);

// Checks final native retirement before removing the context and fence.
// A failed wait preserves both handles for a later teardown attempt.
amdf_status_t amdf_gpu_dxg_mapping_context_deinitialize(
    amdf_gpu_dxg_mapping_context_t* context, const amdf_dxg_kmt_t* kmt,
    const amdf_gpu_dxg_memory_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_DXG_MAPPING_CONTEXT_H_
