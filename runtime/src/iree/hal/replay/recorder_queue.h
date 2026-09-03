// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_QUEUE_H_
#define IREE_HAL_REPLAY_RECORDER_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/queue.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Exact provisioned queue exposed by a recording wrapper device.
typedef struct iree_hal_replay_recorder_queue_t {
  // HAL queue resource. Must be at offset zero.
  iree_hal_queue_t base;

  // Host allocator used for temporary recording storage.
  iree_allocator_t host_allocator;

  // Shared recorder receiving captured operations. Borrowed from the wrapper
  // device, which outlives this embedded proxy.
  iree_hal_replay_recorder_t* recorder;

  // Underlying provisioned queue receiving forwarded calls. Borrowed from the
  // wrapped device, which outlives this proxy.
  iree_hal_queue_t* base_queue;

  // Session-local parent device object id.
  iree_hal_replay_object_id_t device_id;

  // Session-local object id assigned to this exact queue.
  iree_hal_replay_object_id_t queue_id;
} iree_hal_replay_recorder_queue_t;

// Initializes an embedded recording queue proxy with one owning reference.
void iree_hal_replay_recorder_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t queue_id, iree_hal_queue_t* base_queue,
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_queue_t* out_queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_QUEUE_H_
