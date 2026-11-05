// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ADAPTER_H_
#define AMDF_SRC_GPU_UMD_DXG_ADAPTER_H_

#include <stdbool.h>
#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/gpu/umd/dxg/provider_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parsed private state for one physical GPU adapter.
typedef struct amdf_gpu_dxg_adapter_t {
  // Borrowed API table selected statically for this adapter.
  const amdf_dxg_provider_api_t* api;
  // Opaque parsed adapter state owned by the provider.
  amdf_dxg_provider_gpu_adapter_t* native;
} amdf_gpu_dxg_adapter_t;

// Parses one physical GPU adapter through |api|. Failure leaves both output
// structures unchanged and creates no native adapter ownership.
amdf_status_t amdf_gpu_dxg_adapter_initialize(
    const amdf_dxg_provider_api_t* api,
    const amdf_dxg_provider_gpu_adapter_open_info_t* open_info,
    amdf_allocator_t host_allocator, amdf_gpu_dxg_adapter_t* out_adapter,
    amdf_dxg_provider_gpu_properties_t* out_properties);

// Releases parsed state after every dependent provider object is gone.
amdf_status_t amdf_gpu_dxg_adapter_deinitialize(
    amdf_gpu_dxg_adapter_t* adapter);

// Queries the selected queue scheduler's native calibration tuple without
// retaining state. Failure preserves out_sample.
amdf_status_t amdf_gpu_dxg_adapter_query_clock_sample(
    const amdf_gpu_dxg_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index,
    amdf_dxg_provider_gpu_queue_command_type_t command_type,
    amdf_dxg_provider_gpu_clock_sample_t* out_sample);

// Queries the native allocation layout for one aggregate allocation.
amdf_status_t amdf_gpu_dxg_adapter_query_allocation_layout(
    const amdf_gpu_dxg_adapter_t* adapter, uint64_t byte_length,
    uint32_t* out_allocation_count,
    uint64_t* out_maximum_allocation_byte_length);

// Creates grouped native allocations using provider-private driver records.
amdf_status_t amdf_gpu_dxg_adapter_create_allocations(
    const amdf_gpu_dxg_adapter_t* adapter,
    const amdf_dxg_provider_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource, uint32_t* out_allocation_count);

// Prepares a typed shared buffer in the caller's initially zero native slots.
// Acquired handles remain with the caller even when a later step fails.
amdf_status_t amdf_gpu_dxg_adapter_prepare_buffer_import(
    const amdf_gpu_dxg_adapter_t* adapter,
    const amdf_dxg_provider_gpu_buffer_import_info_t* import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length, uint64_t* out_buffer_byte_length);

// Creates one native GPU kernel queue through the selected provider.
amdf_status_t amdf_gpu_dxg_adapter_create_kernel_queue(
    const amdf_gpu_dxg_adapter_t* adapter,
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_provider_gpu_kernel_queue_t** out_queue,
    amdf_dxg_provider_gpu_kernel_queue_info_t* out_info);

// Publishes one already-materialized native GPU command stream.
amdf_status_t amdf_gpu_dxg_adapter_submit_kernel_queue(
    const amdf_gpu_dxg_adapter_t* adapter,
    amdf_dxg_provider_gpu_kernel_queue_t* queue,
    uint64_t command_buffer_address, uint64_t command_buffer_byte_length,
    uint64_t progress_value, bool* out_accepted);

// Releases one native GPU kernel queue.
amdf_status_t amdf_gpu_dxg_adapter_destroy_kernel_queue(
    const amdf_gpu_dxg_adapter_t* adapter,
    amdf_dxg_provider_gpu_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_ADAPTER_H_
