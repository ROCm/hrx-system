// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_TEST_UTIL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Produces one valid TSAN data-race failure under fail-device policy and passes
// it through the logical-device sticky failure path. The caller owns the
// returned status.
iree_status_t iree_hal_amdgpu_feedback_state_test_make_tsan_fail_device_status(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_TEST_UTIL_H_
