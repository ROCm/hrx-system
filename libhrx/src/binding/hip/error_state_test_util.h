// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_ERROR_STATE_TEST_UTIL_H_
#define HRX_BINDING_HIP_ERROR_STATE_TEST_UTIL_H_

#include <stdint.h>

#include "binding/hip/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Callback invoked immediately before public result publication. The callback
// may replace |*result| before the production path observes process state.
typedef void (*iree_hip_error_state_test_publish_hook_t)(hipError_t* result,
                                                         void* user_data);

// Configures a test interceptor at the public result publication boundary.
// Tests configure and clear the hook only while no publication can race the
// update. This symbol remains private to the HIP DSO.
void iree_hip_error_state_test_set_publish_hook(
    iree_hip_error_state_test_publish_hook_t hook, void* user_data);

// Callback invoked immediately before a process-fatal compare-exchange.
typedef void (*iree_hip_error_state_test_compare_exchange_hook_t)(
    void* user_data);

// Configures a test observer for the fatal compare-exchange boundary. Tests
// configure and clear the hook only while no publication can race the update.
void iree_hip_error_state_test_set_compare_exchange_hook(
    iree_hip_error_state_test_compare_exchange_hook_t hook, void* user_data);

// Resets the count of attempted process-fatal compare-exchanges.
void iree_hip_error_state_test_reset_compare_exchange_count(void);

// Returns the count of attempted process-fatal compare-exchanges.
uint32_t iree_hip_error_state_test_compare_exchange_count(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_ERROR_STATE_TEST_UTIL_H_
