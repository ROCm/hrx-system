// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_WAIT_H_
#define AMDF_SRC_WAIT_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

// One caller budget shared by host contention, active polling, and native wait.
typedef struct amdf_wait_deadline_t {
  // Absolute monotonic deadline in nanoseconds; UINT64_MAX means infinite.
  uint64_t timeout;
  // Absolute end of active polling, no later than timeout.
  uint64_t poll;
} amdf_wait_deadline_t;

// Relative durations still available within one absolute wait deadline.
typedef struct amdf_wait_budget_t {
  // Remaining total duration in nanoseconds; UINT64_MAX means infinite.
  uint64_t timeout;
  // Remaining active polling duration in nanoseconds.
  uint64_t poll;
} amdf_wait_budget_t;

// Captures the clock exactly once when the public wait begins.
amdf_status_t amdf_wait_deadline_initialize(uint64_t timeout_nanoseconds,
                                            uint64_t poll_nanoseconds,
                                            amdf_wait_deadline_t* out_deadline);

// Returns the remaining caller budgets without restarting either clock.
amdf_status_t amdf_wait_deadline_query_remaining(
    const amdf_wait_deadline_t* deadline, amdf_wait_budget_t* out_remaining);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_WAIT_H_
