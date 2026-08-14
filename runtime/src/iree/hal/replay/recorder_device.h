// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_DEVICE_H_
#define IREE_HAL_REPLAY_RECORDER_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/device_group.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque shared recorder for one HAL replay capture stream.
typedef struct iree_hal_replay_recorder_t iree_hal_replay_recorder_t;

// Creates a device group whose devices record operations to |recorder|.
//
// |base_group| is retained by the wrappers so the underlying devices keep their
// original topology assignment. The returned group contains replacement
// wrapper devices in the same topology order, with the source topology copied
// into the returned group. All wrappers share |recorder| so multi-device
// captures are emitted to one ordered stream.
IREE_API_EXPORT iree_status_t iree_hal_replay_wrap_device_group(
    iree_hal_replay_recorder_t* recorder, iree_hal_device_group_t* base_group,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_DEVICE_H_
