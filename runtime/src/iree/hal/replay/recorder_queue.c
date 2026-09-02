// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_queue.h"

static void iree_hal_replay_recorder_queue_destroy(
    iree_hal_queue_t* base_queue) {
  // The proxy is embedded in its wrapper device allocation and has no
  // independently owned state.
  (void)base_queue;
}

static const iree_hal_queue_vtable_t iree_hal_replay_recorder_queue_vtable = {
    .destroy = iree_hal_replay_recorder_queue_destroy,
};

void iree_hal_replay_recorder_queue_initialize(
    const iree_hal_queue_family_t* queue_family, iree_hal_queue_t* base_queue,
    iree_hal_replay_recorder_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(base_queue);
  IREE_ASSERT_ARGUMENT(out_queue);
  iree_hal_queue_initialize(
      queue_family, &iree_hal_replay_recorder_queue_vtable, &out_queue->base);
  out_queue->base_queue = base_queue;
}
