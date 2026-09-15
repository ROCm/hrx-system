// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/error_state.h"

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

typedef struct iree_hip_thread_error_state_t {
  // Most recent ordinary error published by a HIP call on this thread.
  hipError_t last_error;
  // Result returned by the most recent HIP call on this thread.
  hipError_t last_command_error;
  // Process error-state generation represented by the two result slots.
  uint32_t generation;
} iree_hip_thread_error_state_t;

static IREE_THREAD_LOCAL iree_hip_thread_error_state_t
    iree_hip_thread_error_state = {hipSuccess, hipSuccess, 0};

// Packs the reset generation in the high word and the fatal HIP result in the
// low word. Updating them together prevents a reset racing an error publication
// from exposing a new generation with an old fatal result, or vice versa.
static iree_atomic_int64_t iree_hip_process_error_state =
    IREE_ATOMIC_VAR_INIT(0);

static uint32_t iree_hip_error_state_generation(int64_t state) {
  return (uint32_t)((uint64_t)state >> 32);
}

static hipError_t iree_hip_error_state_result(int64_t state) {
  return (hipError_t)(uint32_t)state;
}

static int64_t iree_hip_error_state_pack(uint32_t generation,
                                         hipError_t result) {
  return (int64_t)(((uint64_t)generation << 32) | (uint32_t)result);
}

static void iree_hip_error_state_sync_thread(int64_t process_state) {
  const uint32_t generation = iree_hip_error_state_generation(process_state);
  if (iree_hip_thread_error_state.generation == generation) return;
  iree_hip_thread_error_state.last_error = hipSuccess;
  iree_hip_thread_error_state.last_command_error = hipSuccess;
  iree_hip_thread_error_state.generation = generation;
}

hipError_t iree_hip_error_state_fatal_result(void) {
  return iree_hip_error_state_snapshot(/*out_generation=*/NULL);
}

hipError_t iree_hip_error_state_snapshot(uint32_t* out_generation) {
  const int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                                 iree_memory_order_acquire);
  if (out_generation) {
    *out_generation = iree_hip_error_state_generation(process_state);
  }
  return iree_hip_error_state_result(process_state);
}

hipError_t iree_hip_error_state_begin(void) {
  const int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                                 iree_memory_order_acquire);
  iree_hip_error_state_sync_thread(process_state);
  const hipError_t fatal_result = iree_hip_error_state_result(process_state);
  if (fatal_result != hipSuccess) {
    iree_hip_thread_error_state.last_command_error = fatal_result;
    iree_hip_thread_error_state.last_error = fatal_result;
  }
  return fatal_result;
}

hipError_t iree_hip_error_state_publish(hipError_t result) {
  int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                           iree_memory_order_acquire);
  if (result == hipErrorIllegalAddress) {
    while (iree_hip_error_state_result(process_state) == hipSuccess) {
      const int64_t desired_state = iree_hip_error_state_pack(
          iree_hip_error_state_generation(process_state), result);
      if (iree_atomic_compare_exchange_strong(
              &iree_hip_process_error_state, &process_state, desired_state,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        process_state = desired_state;
        break;
      }
    }
  }

  iree_hip_error_state_sync_thread(process_state);
  const hipError_t fatal_result = iree_hip_error_state_result(process_state);
  if (fatal_result != hipSuccess) result = fatal_result;
  iree_hip_thread_error_state.last_command_error = result;
  if (result != hipSuccess && result != hipErrorNotReady) {
    iree_hip_thread_error_state.last_error = result;
  }
  return result;
}

hipError_t iree_hip_error_state_get_and_clear_last_error(void) {
  const int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                                 iree_memory_order_acquire);
  iree_hip_error_state_sync_thread(process_state);
  const hipError_t fatal_result = iree_hip_error_state_result(process_state);
  if (fatal_result != hipSuccess) {
    return fatal_result;
  }
  const hipError_t result = iree_hip_thread_error_state.last_error;
  iree_hip_thread_error_state.last_error = hipSuccess;
  return result;
}

hipError_t iree_hip_error_state_get_and_clear_command_error(void) {
  const int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                                 iree_memory_order_acquire);
  iree_hip_error_state_sync_thread(process_state);
  const hipError_t fatal_result = iree_hip_error_state_result(process_state);
  if (fatal_result != hipSuccess) return fatal_result;
  const hipError_t result = iree_hip_thread_error_state.last_command_error;
  iree_hip_thread_error_state.last_command_error = hipSuccess;
  return result;
}

hipError_t iree_hip_error_state_peek_last_error(void) {
  const int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                                 iree_memory_order_acquire);
  iree_hip_error_state_sync_thread(process_state);
  const hipError_t fatal_result = iree_hip_error_state_result(process_state);
  return fatal_result != hipSuccess ? fatal_result
                                    : iree_hip_thread_error_state.last_error;
}

void iree_hip_error_state_reset(void) {
  int64_t process_state = iree_atomic_load(&iree_hip_process_error_state,
                                           iree_memory_order_acquire);
  int64_t desired_state;
  do {
    desired_state = iree_hip_error_state_pack(
        iree_hip_error_state_generation(process_state) + 1, hipSuccess);
  } while (!iree_atomic_compare_exchange_strong(
      &iree_hip_process_error_state, &process_state, desired_state,
      iree_memory_order_acq_rel, iree_memory_order_acquire));
  iree_hip_error_state_sync_thread(desired_state);
}
