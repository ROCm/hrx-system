// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_ADMISSION_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_ADMISSION_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bounded backfill policy over a FIFO maintained by the caller. This value
// selects queue positions only: it neither owns requests nor estimates fit.
// Native reservation decides fit before the caller reports an acquisition.
typedef struct loom_serve_text_admission_t {
  // Maximum successful overtakes while the same oldest request remains.
  // Zero preserves strict FIFO; deployments default to their execution rows.
  iree_host_size_t budget;
  // Successful overtakes since the oldest request last changed.
  iree_host_size_t overtakes;
  // Selected FIFO position, held stable through one preparation/reservation.
  iree_host_size_t cursor;
} loom_serve_text_admission_t;

// Reports capacity denial for cursor. Returns true with the next position if
// the budget permits another probe in this pass. Otherwise restarts at zero
// for the next capacity/arrival observation and returns false. Denied probes
// do not spend budget. Count is the current nonempty FIFO size.
bool loom_serve_text_admission_blocked(loom_serve_text_admission_t* admission,
                                       iree_host_size_t count);

// Accounts a successful selected admission, before removing its queue entry.
void loom_serve_text_admission_acquired(loom_serve_text_admission_t* admission);

// Accounts removal after admission, rejection or cancellation. Removing the
// oldest resets its budget. Removing a tail never replenishes that budget.
// Removal at/before cursor restarts selection at the oldest surviving entry.
void loom_serve_text_admission_remove(loom_serve_text_admission_t* admission,
                                      iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_ADMISSION_H_
