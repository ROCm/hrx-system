// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor schedule-row projection into schedule table records.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_DESCRIPTOR_ROWS_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_DESCRIPTOR_ROWS_H_

#include "loom/codegen/low/schedule/context.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_low_schedule_note_descriptor_rows_for_node(
    loom_low_schedule_build_state_t* state, uint32_t node_index);

void loom_low_schedule_compact_resource_summaries(
    loom_low_schedule_build_state_t* state);

void loom_low_schedule_compact_model_summaries(
    loom_low_schedule_build_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_DESCRIPTOR_ROWS_H_
