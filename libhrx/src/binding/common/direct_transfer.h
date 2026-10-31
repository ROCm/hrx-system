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

// Submits uncaptured transfers after the stream's preceding work, preserving
// its execution queue lease and publishing one new timeline point. Descriptors
// are borrowed for this call; the HAL captures accepted operations. Default
// queue barriers publish system-visible results before stream completion.
// Success is asynchronous. A post-accept flush failure drains the accepted
// transfer before returning ownership of borrowed host storage.
iree_status_t iree_hal_streaming_direct_transfer_enqueue(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations);

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

// Runs a blocking transfer batch after this stream's preceding work on the
// context's transfer queue. Descriptors and host bytes remain borrowed until
// return. The caller executes host participation without a callback worker.
iree_status_t iree_hal_streaming_direct_transfer_and_wait(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations);

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
