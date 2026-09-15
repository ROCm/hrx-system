// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_ADAPTER_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_ADAPTER_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/loader.h"
#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parsed private state for one physical GPU adapter.
typedef struct amdf_gpu_wddm_wkmi_adapter_t {
  // Borrowed API table owned by the loader that outlives this adapter.
  const amdf_wkmi_bridge_api_t* api;
  // Opaque parsed adapter state owned by the loaded bridge.
  amdf_wkmi_bridge_gpu_adapter_t* native;
} amdf_gpu_wddm_wkmi_adapter_t;

// Parses one physical GPU adapter through |loader|. Failure leaves both output
// structures unchanged and creates no native adapter ownership.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_initialize(
    const amdf_gpu_wddm_wkmi_loader_t* loader, D3DKMT_HANDLE adapter,
    uint32_t physical_adapter_index, amdf_allocator_t host_allocator,
    amdf_gpu_wddm_wkmi_adapter_t* out_adapter,
    amdf_wkmi_bridge_gpu_properties_t* out_properties);

// Releases parsed state after every dependent bridge object is gone.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_deinitialize(
    amdf_gpu_wddm_wkmi_adapter_t* adapter);

// Queries the native allocation layout for one aggregate allocation.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_query_allocation_layout(
    const amdf_gpu_wddm_wkmi_adapter_t* adapter, uint64_t byte_length,
    uint32_t* out_allocation_count,
    uint64_t* out_maximum_allocation_byte_length);

// Creates grouped native allocations using WKMI-private driver records.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_create_allocations(
    const amdf_gpu_wddm_wkmi_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, D3DKMT_HANDLE* out_allocation_handles,
    D3DKMT_HANDLE* out_resource, uint32_t* out_allocation_count);

// Creates one native GPU kernel queue through pinned WKMI.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_create_kernel_queue(
    const amdf_gpu_wddm_wkmi_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
    amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info);

// Publishes one already-materialized native GPU command stream.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_submit_kernel_queue(
    const amdf_gpu_wddm_wkmi_adapter_t* adapter,
    amdf_wkmi_bridge_gpu_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t progress_value);

// Releases one native GPU kernel queue.
amdf_status_t amdf_gpu_wddm_wkmi_adapter_destroy_kernel_queue(
    const amdf_gpu_wddm_wkmi_adapter_t* adapter,
    amdf_wkmi_bridge_gpu_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_ADAPTER_H_
