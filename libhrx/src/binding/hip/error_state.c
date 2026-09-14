// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/error_state.h"

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

typedef struct iree_hip_thread_error_state_t {
  // Most recent error published by a HIP call on this thread.
  hipError_t last_error;
  // True when last_error reflects the process-wide fatal device state.
  bool sticky;
} iree_hip_thread_error_state_t;

static IREE_THREAD_LOCAL iree_hip_thread_error_state_t
    iree_hip_thread_error_state = {hipSuccess, false};

// Stores the first fatal device error observed by any thread. The latch is
// checked at ordinary HIP initialization gates so a failure observed through
// one stream or event becomes visible process-wide.
static iree_atomic_int32_t iree_hip_fatal_error =
    IREE_ATOMIC_VAR_INIT(hipSuccess);

static void iree_hip_error_state_set_thread(hipError_t error, bool sticky) {
  if (iree_hip_thread_error_state.sticky && !sticky) return;
  iree_hip_thread_error_state.last_error = error;
  iree_hip_thread_error_state.sticky = sticky;
}

hipError_t iree_hip_error_state_fatal_result(void) {
  return (hipError_t)iree_atomic_load(&iree_hip_fatal_error,
                                      iree_memory_order_acquire);
}

hipError_t iree_hip_error_state_publish(hipError_t result) {
  if (result == hipErrorIllegalAddress) {
    int32_t expected = hipSuccess;
    iree_atomic_compare_exchange_strong(
        &iree_hip_fatal_error, &expected, (int32_t)result,
        iree_memory_order_acq_rel, iree_memory_order_acquire);
  }

  const hipError_t fatal_result = iree_hip_error_state_fatal_result();
  if (fatal_result != hipSuccess) result = fatal_result;
  if (result != hipSuccess) {
    iree_hip_error_state_set_thread(result, fatal_result != hipSuccess);
  }
  return result;
}

hipError_t iree_hip_error_state_get_and_clear(void) {
  const hipError_t fatal_result = iree_hip_error_state_fatal_result();
  if (fatal_result != hipSuccess) {
    iree_hip_error_state_set_thread(fatal_result, /*sticky=*/true);
    return fatal_result;
  }
  const hipError_t result = iree_hip_thread_error_state.last_error;
  if (!iree_hip_thread_error_state.sticky) {
    iree_hip_thread_error_state.last_error = hipSuccess;
  }
  return result;
}

hipError_t iree_hip_error_state_peek(void) {
  const hipError_t fatal_result = iree_hip_error_state_fatal_result();
  if (fatal_result != hipSuccess) {
    iree_hip_error_state_set_thread(fatal_result, /*sticky=*/true);
  }
  return iree_hip_thread_error_state.last_error;
}

void iree_hip_error_state_reset(void) {
  iree_atomic_store(&iree_hip_fatal_error, hipSuccess,
                    iree_memory_order_release);
  iree_hip_thread_error_state.last_error = hipSuccess;
  iree_hip_thread_error_state.sticky = false;
}
