// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_STREAM_VALUE_H_
#define IREE_EXPERIMENTAL_STREAMING_STREAM_VALUE_H_

#include "command_recording.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_hal_streaming_value_flush_timer_t
    iree_hal_streaming_value_flush_timer_t;

// Validates one fully resolved operation, including its target range.
iree_status_t iree_hal_streaming_value_operation_validate(
    const hrx_atomic_operation_t* operation);

// Returns true when |family_spec| can dedicate an exact queue to a memory wait.
// Waits must consume no dispatch resources so kernels that satisfy their
// predicates can continue to execute.
bool iree_hal_streaming_queue_family_supports_value_waits(
    const iree_hal_queue_family_spec_t* family_spec);

// Enqueues |operations| as one ordered stream transaction. All resources and
// command storage are prepared before the transaction is submitted. Batches
// containing a wait use an independently progressing context-owned queue that
// is recycled only after every accepted lane submission reaches its own
// terminal completion record.
// Synchronization: the caller must hold ordinary capture admission while
// deciding capture disposition and calling this function. Flushes pending
// stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_value_operations(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const hrx_atomic_operation_t* operations);

// Enqueues a device-side value wait at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_wait_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_wait_params_t params);

// Enqueues a device-side atomic store at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_store_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_store_params_t params);

// Enqueues a device-side atomic update at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_update_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_rmw_params_t params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_STREAM_VALUE_H_
