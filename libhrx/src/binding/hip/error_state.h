// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_ERROR_STATE_H_
#define HRX_BINDING_HIP_ERROR_STATE_H_

#include "binding/hip/api.h"
#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Publishes the result of a public HIP call. Every result becomes the calling
// thread's command result. Errors other than hipErrorNotReady also become the
// thread's ordinary last error. Illegal-address failures are latched
// process-wide so later HIP entry points and other threads observe the fatal
// device state. Returns the effective result, which is the process-wide fatal
// error once one has been latched.
hipError_t iree_hip_error_state_publish(hipError_t result);

// Returns the process-wide fatal device result, or hipSuccess when none has
// been observed. This does not modify the calling thread's last-error state.
hipError_t iree_hip_error_state_fatal_result(void);

// Returns the process-wide fatal result and optionally the runtime-lifetime
// generation from one atomic snapshot. Full runtime teardown advances the
// generation so other per-thread HIP state can lazily discard stale values.
hipError_t iree_hip_error_state_snapshot(uint32_t* out_generation);

// Starts an ordinary public HIP call. Returns and records the process-fatal
// result from one generation-consistent snapshot, or hipSuccess when the call
// may proceed. Unlike publication, this never latches a new fatal result.
hipError_t iree_hip_error_state_begin(void);

// Returns the process-fatal result when one is latched, otherwise returns and
// clears the calling thread's ordinary last error.
hipError_t iree_hip_error_state_get_and_clear_last_error(void);

// Returns the process-fatal result when one is latched, otherwise returns and
// clears the calling thread's most recent command result.
hipError_t iree_hip_error_state_get_and_clear_command_error(void);

// Returns the process-fatal result when one is latched, otherwise returns the
// calling thread's last error without clearing it.
hipError_t iree_hip_error_state_peek_last_error(void);

// Clears process and calling-thread state when the embedded runtime is fully
// deinitialized and no device work remains.
void iree_hip_error_state_reset(void);

// Completes a public HIP API call through the shared error-state boundary.
#define HIP_RETURN_ERROR(error)                   \
  do {                                            \
    return iree_hip_error_state_publish((error)); \
  } while (0)

// Rejects an ordinary public HIP API call before it can observe or mutate
// runtime state when a process-fatal device error is active. Error inspection
// and runtime teardown entry points intentionally omit this boundary.
#define HIP_API_BEGIN_OR_RETURN(return_value)                      \
  do {                                                             \
    const hipError_t _fatal_result = iree_hip_error_state_begin(); \
    if (IREE_UNLIKELY(_fatal_result != hipSuccess)) {              \
      return (return_value);                                       \
    }                                                              \
  } while (0)
#define HIP_API_BEGIN() HIP_API_BEGIN_OR_RETURN(_fatal_result)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_ERROR_STATE_H_
