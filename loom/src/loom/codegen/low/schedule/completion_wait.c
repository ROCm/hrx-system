// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_wait.h"

bool loom_low_schedule_class_query_completion_wait(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_class_t* schedule_class,
    uint16_t* out_wait_cycles) {
  *out_wait_cycles = 0;
  if (descriptor_set == NULL || schedule_class == NULL) return false;
  bool has_wait_counter_hazard = false;
  for (uint16_t i = 0; i < schedule_class->hazard_count; ++i) {
    const loom_low_hazard_t* hazard =
        &descriptor_set->hazards[schedule_class->hazard_start + i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_WAIT_COUNTER) continue;
    has_wait_counter_hazard = true;
    if (hazard->distance > *out_wait_cycles) {
      *out_wait_cycles = hazard->distance;
    }
  }
  return has_wait_counter_hazard;
}
