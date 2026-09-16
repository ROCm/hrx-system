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

// Identifies the process error-state generation in which one public HIP call
// began. Tokens are call-local so nested public calls cannot overwrite an
// outer call's publication identity.
typedef struct iree_hip_error_state_token_t {
  // Runtime-lifetime generation observed at the public call boundary.
  uint32_t generation;
  // Process-fatal result observed at the public call boundary.
  hipError_t fatal_result;
} iree_hip_error_state_token_t;

// Publishes the result of a public HIP call. Every result becomes the calling
// thread's command result. Errors other than hipErrorNotReady also become the
// thread's ordinary last error. Illegal-address failures are latched
// process-wide so later HIP entry points and other threads observe the fatal
// device state only when |token| still identifies the active runtime
// generation. A result from an older generation is returned without changing
// the active generation's process or thread state. Returns the effective
// result, which is the process-wide fatal error once one has been latched.
hipError_t iree_hip_error_state_publish(iree_hip_error_state_token_t token,
                                        hipError_t result);

// Returns the process-wide fatal device result, or hipSuccess when none has
// been observed. This does not modify the calling thread's last-error state.
hipError_t iree_hip_error_state_fatal_result(void);

// Returns the process-wide fatal result and optionally the runtime-lifetime
// generation from one atomic snapshot. Full runtime teardown advances the
// generation so other per-thread HIP state can lazily discard stale values.
hipError_t iree_hip_error_state_snapshot(uint32_t* out_generation);

// Captures a call-local token without changing thread error state. Teardown and
// error-inspection entry points use this because they remain callable while a
// process-fatal result is active.
iree_hip_error_state_token_t iree_hip_error_state_capture(void);

// Starts an ordinary public HIP call. Returns a generation-consistent token
// and records its process-fatal result when one is active. Unlike publication,
// this never latches a new fatal result.
iree_hip_error_state_token_t iree_hip_error_state_begin(void);

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
#define HIP_RETURN_ERROR(error)                                           \
  do {                                                                    \
    return iree_hip_error_state_publish(_hip_error_state_token, (error)); \
  } while (0)

// Captures publication identity for teardown and error-inspection entry points
// without rejecting a process-fatal runtime generation.
#define HIP_API_CAPTURE_ERROR_STATE()                         \
  const iree_hip_error_state_token_t _hip_error_state_token = \
      iree_hip_error_state_capture()

// Rejects an ordinary public HIP API call before it can observe or mutate
// runtime state when a process-fatal device error is active. Error inspection
// and runtime teardown entry points intentionally omit this boundary.
#define HIP_API_BEGIN_OR_RETURN(return_value)                               \
  const iree_hip_error_state_token_t _hip_error_state_token =               \
      iree_hip_error_state_begin();                                         \
  do {                                                                      \
    if (IREE_UNLIKELY(_hip_error_state_token.fatal_result != hipSuccess)) { \
      return (return_value);                                                \
    }                                                                       \
  } while (0)
#define HIP_API_BEGIN() \
  HIP_API_BEGIN_OR_RETURN(_hip_error_state_token.fatal_result)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_ERROR_STATE_H_
