// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-pressure repair by fixed-value live-range detachment.
//
// Fixed values model ABI or target locations that must occupy a particular
// physical slot while live. When such a value stays live long enough to force a
// spill, this utility can insert one low.copy immediately after the fixed
// source is materialized and rewrite later users to the copy result. The fixed
// physical location then dies at the copy while the ordinary virtual copy
// remains allocatable by normal rules.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_live_range_split_trigger_e {
  // Unknown or uninitialized split trigger.
  LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_UNKNOWN = 0,
  // A predicted spill plan was avoided by detaching a fixed value.
  LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN = 1,
} loom_low_allocation_live_range_split_trigger_t;

typedef struct loom_low_allocation_live_range_split_result_t {
  // Fixed SSA value whose physical live range was detached.
  loom_value_id_t source_value_id;
  // low.copy result carrying the ordinary virtual live range.
  loom_value_id_t split_value_id;
  // Original allocation assignment associated with |source_value_id|, or
  // UINT32_MAX when the fixed value had no completed assignment.
  uint32_t source_assignment_index;
  // Number of low.copy packets inserted.
  uint32_t copy_packet_count;
  // Number of operand uses rewritten to |split_value_id|.
  uint32_t rewritten_operand_count;
} loom_low_allocation_live_range_split_result_t;

// Attempts to repair one predicted spill plan by detaching an overlapping
// fixed value from its fixed physical location.
//
// Returns OK with a zero result when no fixed value is a safe split candidate.
// When a value is rewritten, callers must rebuild allocation before consulting
// the old allocation table again.
iree_status_t loom_low_allocation_split_fixed_value_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* out_result);

// Emits a structured remark describing a successful live-range split result.
iree_status_t loom_low_allocation_live_range_split_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_live_range_split_trigger_t trigger,
    const loom_low_allocation_live_range_split_result_t* result,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_SPLITTING_H_
