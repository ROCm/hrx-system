// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage-alias ownership used by low scheduler pressure simulation.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_PRESSURE_ALIAS_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_PRESSURE_ALIAS_H_

#include "loom/codegen/low/schedule/context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_schedule_pressure_alias_record_t
    loom_low_schedule_pressure_alias_record_t;
typedef struct loom_low_schedule_pressure_state_t
    loom_low_schedule_pressure_state_t;

// Mutable storage-alias state for one pressure simulation.
typedef struct loom_low_schedule_pressure_alias_state_t {
  // Relation-index list heads keyed by source value ordinal.
  uint32_t* source_heads;
  // Source value ordinals touched in source_heads.
  loom_value_ordinal_t* source_ordinals;
  // Active alias units claimed from each source value ordinal.
  uint32_t* source_unit_counts;
  // Mutable alias records indexed by compact storage-relation index.
  loom_low_schedule_pressure_alias_record_t* records;
  // Number of populated entries in source_ordinals.
  iree_host_size_t source_count;
  // Current nonzero traversal epoch.
  uint32_t epoch;
} loom_low_schedule_pressure_alias_state_t;

// Initializes relation-indexed alias storage. The zero-relation path performs
// no allocation.
iree_status_t loom_low_schedule_pressure_alias_initialize(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_alias_state_t* out_alias_state);

// Resets the touched alias state before another block simulation.
void loom_low_schedule_pressure_alias_reset(
    loom_low_schedule_pressure_alias_state_t* alias_state);

// Adds aliases for one result while reconstructing authored source order and
// returns the units already owned by live sources.
uint32_t loom_low_schedule_pressure_alias_append_source_baseline_result(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_block_t* block, loom_value_ordinal_t result_ordinal);

// Transfers source-owned units to live alias results when the source dies.
uint32_t loom_low_schedule_pressure_alias_transfer_from_source(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal);

// Transfers result-owned units to a newly live source during reverse baseline
// reconstruction.
uint32_t loom_low_schedule_pressure_alias_transfer_to_source(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal);

// Releases every active alias relation owned by a dead result.
void loom_low_schedule_pressure_alias_deactivate_result(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t result_ordinal);

// Accounts for source capacity released by a candidate's dying result.
void loom_low_schedule_pressure_alias_note_candidate_result_releases(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t result_ordinal);

// Returns units transferred to live results when a candidate kills a source.
uint32_t loom_low_schedule_pressure_alias_candidate_transfer_from_source(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal);

// Returns units a candidate result can alias after candidate operand deaths.
uint32_t loom_low_schedule_pressure_alias_candidate_result_units(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_value_ordinal_t result_ordinal);

// Activates aliases for a newly scheduled result and returns aliased units.
uint32_t loom_low_schedule_pressure_alias_append_scheduled_result(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_value_ordinal_t result_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_PRESSURE_ALIAS_H_
