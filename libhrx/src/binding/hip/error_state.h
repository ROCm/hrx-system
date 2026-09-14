// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_ERROR_STATE_H_
#define HRX_BINDING_HIP_ERROR_STATE_H_

#include "binding/hip/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Publishes |result| as the calling thread's last error. Illegal-address
// failures are also latched process-wide so later HIP entry points and other
// threads observe the fatal device state. Returns the effective result, which
// is the process-wide fatal error once one has been latched.
hipError_t iree_hip_error_state_publish(hipError_t result);

// Returns the process-wide fatal device result, or hipSuccess when none has
// been observed. This does not modify the calling thread's last-error state.
hipError_t iree_hip_error_state_fatal_result(void);

// Returns the process-fatal result when one is latched, otherwise returns the
// calling thread's last error and clears non-fatal thread state.
hipError_t iree_hip_error_state_get_and_clear(void);

// Returns the process-fatal result when one is latched, otherwise returns the
// calling thread's last error without clearing it.
hipError_t iree_hip_error_state_peek(void);

// Clears process and calling-thread state when the embedded runtime is fully
// deinitialized and no device work remains.
void iree_hip_error_state_reset(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_ERROR_STATE_H_
