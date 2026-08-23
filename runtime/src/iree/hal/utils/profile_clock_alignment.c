// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/profile_clock_alignment.h"

#include <string.h>

void iree_hal_profile_clock_alignment_reset(
    iree_hal_profile_clock_alignment_t* alignment) {
  IREE_ASSERT_ARGUMENT(alignment);
  memset(alignment, 0, sizeof(*alignment));
  alignment->minimum_clock_tick = UINT64_MAX;
  alignment->minimum_event_tick = UINT64_MAX;
}

void iree_hal_profile_clock_alignment_record_event_range(
    iree_hal_profile_clock_alignment_t* alignment, uint64_t start_tick,
    uint64_t end_tick) {
  IREE_ASSERT_ARGUMENT(alignment);
  if (IREE_UNLIKELY(end_tick < start_tick)) {
    alignment->has_invalid_alignment = true;
    return;
  }
  if (alignment->has_event_ticks) {
    alignment->minimum_event_tick =
        iree_min(alignment->minimum_event_tick, start_tick);
    alignment->maximum_event_tick =
        iree_max(alignment->maximum_event_tick, end_tick);
  } else {
    alignment->minimum_event_tick = start_tick;
    alignment->maximum_event_tick = end_tick;
    alignment->has_event_ticks = true;
  }
}

bool iree_hal_profile_clock_alignment_record_clock_tick(
    iree_hal_profile_clock_alignment_t* alignment, uint64_t clock_tick) {
  IREE_ASSERT_ARGUMENT(alignment);
  if (alignment->has_clock_ticks) {
    alignment->minimum_clock_tick =
        iree_min(alignment->minimum_clock_tick, clock_tick);
    alignment->maximum_clock_tick =
        iree_max(alignment->maximum_clock_tick, clock_tick);
  } else {
    alignment->minimum_clock_tick = clock_tick;
    alignment->maximum_clock_tick = clock_tick;
    alignment->has_clock_ticks = true;
  }
  if (alignment->has_event_ticks &&
      (alignment->minimum_event_tick < alignment->minimum_clock_tick ||
       alignment->maximum_event_tick > alignment->maximum_clock_tick)) {
    alignment->has_invalid_alignment = true;
  }
  return alignment->has_invalid_alignment;
}
