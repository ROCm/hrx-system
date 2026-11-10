// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_STREAM_H_
#define LIBHRX_SRC_LIBHRX_STREAM_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"
#include "queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_stream_s {
  // Public references to this stream and its owned recording/timeline state.
  iree_atomic_ref_count_t ref_count;
  // Recording categories admitted by the immutable queue family.
  iree_hal_command_category_t command_categories;
  // Retained exact execution queue. The semaphore keeps its device alive.
  hrx_queue_t queue;
  // Cached native queue borrowed from |queue| for submission.
  iree_hal_queue_t* hal_queue;
  // Owned timeline, which also retains the queue's parent device.
  hrx_semaphore_t semaphore;
  // Last reserved point, including externally published work.
  uint64_t timepoint;
  // Owned one-shot recording, or NULL between batches.
  iree_hal_command_buffer_t* pending_cb;
  // Whether |pending_cb| contains work that must be flushed.
  bool has_pending_work;
  // Creation flags supplied by the caller.
  uint32_t flags;
} hrx_stream_s;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_STREAM_H_
