// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_IPC_H_
#define LIBHRX_SRC_BINDING_HIP_IPC_H_

#include "binding/hip/api.h"
#include "common/ipc_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encodes the stock ROCm memory-handle wire representation. The output is
// zeroed before validation and remains zero when encoding fails.
iree_status_t iree_hip_ipc_memory_handle_encode(
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_host_size_t device_ordinal, int32_t creator_process_id,
    hipIpcMemHandle_t* out_handle);

// Decodes and validates the stock ROCm memory-handle wire representation. The
// outputs are zeroed before validation and remain zero when decoding fails.
iree_status_t iree_hip_ipc_memory_handle_decode(
    hipIpcMemHandle_t handle, int32_t current_process_id,
    iree_host_size_t visible_device_count,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor,
    iree_device_size_t* out_view_size);

// Maps memory export failures after public argument validation. Unsupported
// builds remain distinguishable from stock invalid-allocation failures.
hipError_t iree_hip_ipc_memory_export_status_to_result(iree_status_t status);

// Exports the HIP wire representation of a plain hipMalloc allocation.
iree_status_t iree_hip_ipc_memory_export(iree_hal_streaming_context_t* context,
                                         void* device_ptr,
                                         hipIpcMemHandle_t* out_handle);

// Imports a HIP memory handle into |context|.
iree_status_t iree_hip_ipc_memory_import(iree_hal_streaming_context_t* context,
                                         hipIpcMemHandle_t handle,
                                         void** out_device_ptr);

// Closes one process-wide open reference identified by |device_ptr|,
// preferring a reference opened by |context|.
iree_status_t iree_hip_ipc_memory_close(iree_hal_streaming_context_t* context,
                                        void* device_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_IPC_H_
