// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_QUEUE_H_
#define IREE_HAL_REPLAY_RECORDER_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Exact provisioned queue exposed by a recording wrapper device.
typedef struct iree_hal_replay_recorder_queue_t {
  // HAL queue resource. Must be at offset zero.
  iree_hal_queue_t base;

  // Underlying provisioned queue receiving forwarded calls. Borrowed from the
  // wrapped device, which outlives this proxy.
  iree_hal_queue_t* base_queue;
} iree_hal_replay_recorder_queue_t;

// Initializes an embedded recording queue proxy with one owning reference.
void iree_hal_replay_recorder_queue_initialize(
    const iree_hal_queue_family_t* queue_family, iree_hal_queue_t* base_queue,
    iree_hal_replay_recorder_queue_t* out_queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_QUEUE_H_
