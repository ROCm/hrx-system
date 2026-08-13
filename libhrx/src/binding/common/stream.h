// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_STREAM_H_
#define IREE_EXPERIMENTAL_STREAMING_STREAM_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

// Orders future work on |stream| after work already enqueued on
// |source_stream|. The dependency is submitted to the device queue and does
// not wait for either stream on the calling thread.
iree_status_t iree_hal_streaming_stream_wait_stream(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* source_stream);

// Enqueues a HAL host call at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_host_call(
    iree_hal_streaming_stream_t* stream, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_STREAM_H_
