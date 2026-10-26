// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_DIRECT_TRANSFER_H_
#define IREE_HAL_STREAMING_DIRECT_TRANSFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_buffer_ref_t iree_hal_streaming_buffer_ref_t;

// Performs a blocking host-to-device transfer on |context|'s provisioned
// transfer queue.
iree_status_t iree_hal_streaming_direct_transfer_h2d(
    iree_hal_streaming_context_t* context, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length);

// Performs a blocking device-to-host transfer on |context|'s provisioned
// transfer queue.
iree_status_t iree_hal_streaming_direct_transfer_d2h(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, void* target, iree_device_size_t length);

// Downloads validated source rows to unregistered host memory in one native
// transfer batch. The caller reserves and completes one point on |stream|'s
// timeline without routing CPU work to a callback worker. Prepared |sources|
// and the host destination remain borrowed until this blocking call returns.
iree_status_t iree_hal_streaming_direct_transfer_d2h_rows(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream,
    void* destination, iree_device_size_t destination_pitch,
    iree_device_size_t width, iree_host_size_t row_count,
    const iree_hal_streaming_buffer_ref_t* sources);

// Performs a blocking device-to-device transfer on |context|'s provisioned
// transfer queue.
iree_status_t iree_hal_streaming_direct_transfer_d2d(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_DIRECT_TRANSFER_H_
