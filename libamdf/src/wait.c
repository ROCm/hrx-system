// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/wait.h"

#include "libamdf/src/platform/wait.h"

static uint64_t amdf_wait_deadline_add(uint64_t now, uint64_t duration) {
  if (duration == AMDF_TIMEOUT_INFINITE) return UINT64_MAX;
  // Saturation must preserve the distinction between finite and infinite.
  return duration >= UINT64_MAX - now ? UINT64_MAX - 1 : now + duration;
}

amdf_status_t amdf_wait_deadline_initialize(
    uint64_t timeout_nanoseconds, uint64_t poll_nanoseconds,
    amdf_wait_deadline_t* out_deadline) {
  uint64_t now;
  const amdf_status_t status = amdf_platform_wait_query_time(&now);
  if (!amdf_status_is_ok(status)) return status;
  out_deadline->timeout = amdf_wait_deadline_add(now, timeout_nanoseconds);
  out_deadline->poll = amdf_wait_deadline_add(
      now, poll_nanoseconds < timeout_nanoseconds ? poll_nanoseconds
                                                  : timeout_nanoseconds);
  return AMDF_STATUS_OK;
}

static uint64_t amdf_wait_deadline_remaining(uint64_t deadline, uint64_t now) {
  return deadline == UINT64_MAX ? AMDF_TIMEOUT_INFINITE
                                : (deadline > now ? deadline - now : 0);
}

amdf_status_t amdf_wait_deadline_query_remaining(
    const amdf_wait_deadline_t* deadline, amdf_wait_budget_t* out_remaining) {
  uint64_t now;
  const amdf_status_t status = amdf_platform_wait_query_time(&now);
  if (!amdf_status_is_ok(status)) return status;
  out_remaining->timeout = amdf_wait_deadline_remaining(deadline->timeout, now);
  out_remaining->poll = amdf_wait_deadline_remaining(deadline->poll, now);
  return AMDF_STATUS_OK;
}
