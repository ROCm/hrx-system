// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/child_tracker.h"

void amdf_child_tracker_initialize(amdf_child_tracker_t* tracker) {
  amdf_atomic_uint32_initialize(&tracker->count, 0);
}

uint32_t amdf_child_tracker_count(const amdf_child_tracker_t* tracker) {
  return amdf_atomic_uint32_load_acquire(&tracker->count);
}

amdf_status_t amdf_child_tracker_register(amdf_child_tracker_t* tracker) {
  uint32_t old_count = amdf_atomic_uint32_load_relaxed(&tracker->count);
  do {
    if (old_count == UINT32_MAX) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
  } while (!amdf_atomic_uint32_compare_exchange_acq_rel(
      &tracker->count, &old_count, old_count + 1));
  return AMDF_STATUS_OK;
}

void amdf_child_tracker_unregister(amdf_child_tracker_t* tracker) {
  uint32_t old_count = amdf_atomic_uint32_load_relaxed(&tracker->count);
  for (;;) {
    if (old_count == 0) {
      amdf_assert(false && "unbalanced child registration");
      return;
    }
    if (amdf_atomic_uint32_compare_exchange_acq_rel(&tracker->count, &old_count,
                                                    old_count - 1)) {
      return;
    }
  }
}
