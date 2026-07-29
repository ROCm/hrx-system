// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_LOW_MIX_H_
#define LOOM_TARGET_REPORTING_LOW_MIX_H_

#include "loom/codegen/low/frame.h"
#include "loom/target/reporting/report.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Exact dynamic execution facts derived for one low emission frame.
typedef struct loom_target_compile_report_low_dynamic_context_t {
  // Arena used for value facts and CFG scratch while building dynamic facts.
  iree_arena_allocator_t arena;
  // Value facts used to prove exact nested loop trip counts.
  loom_value_fact_table_t fact_table;
  // Exact execution multiplier per scheduled low block.
  uint64_t* block_multipliers;
  // True when |arena| was initialized and must be deinitialized.
  bool initialized;
  // True when every loop/backedge needed for dynamic counts was modeled.
  bool exact;
} loom_target_compile_report_low_dynamic_context_t;

// Builds exact dynamic execution facts for |frame| when available.
iree_status_t loom_target_compile_report_low_dynamic_context_initialize(
    const loom_low_emission_frame_t* frame,
    loom_target_compile_report_low_dynamic_context_t* out_context);

// Releases scratch storage owned by |context|.
void loom_target_compile_report_low_dynamic_context_deinitialize(
    loom_target_compile_report_low_dynamic_context_t* context);

// Accumulates the static instruction mix of |node| into |mix|.
void loom_target_compile_report_accumulate_low_node_static_mix(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node,
    loom_target_compile_report_static_instruction_mix_t* mix);

// Accumulates |source| into |target|.
void loom_target_compile_report_accumulate_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source);

// Accumulates |source| scaled by |scale|, returning false on overflow.
bool loom_target_compile_report_accumulate_scaled_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source,
    uint64_t scale);

// Returns the exact execution multiplier for |node| when one is known.
bool loom_target_compile_report_low_node_execution_multiplier(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const uint64_t* block_multipliers, const loom_low_schedule_node_t* node,
    uint64_t* out_multiplier);

// Records the static instruction mix of |frame|.
void loom_target_compile_report_record_low_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame);

// Records the exact dynamic instruction mix of |frame| when available.
iree_status_t loom_target_compile_report_record_low_dynamic_mix(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_LOW_MIX_H_
