// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Register capacity observed at retained call boundaries.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_CALL_PRESSURE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_CALL_PRESSURE_H_

#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/schedule/pressure.h"

#ifdef __cplusplus
extern "C" {
#endif

// One capacity reduced by a retained call's clobber contract. Shared register
// classes have one representative row; packing resources retain separate rows.
typedef struct loom_low_schedule_call_pressure_limit_t {
  // Register-class or register-packing-resource table ID.
  uint16_t source_id;
  // Shared pressure-set ID for a register-class limit, otherwise zero.
  uint16_t pressure_set_id;
  // Pressure units that survive the call clobber phase.
  uint32_t preserved_unit_capacity;
  // Interpretation of |source_id|.
  loom_low_schedule_pressure_source_kind_t source_kind;
  // Whether exceeding this capacity cannot be repaired by spilling.
  bool requires_hard_capacity;
} loom_low_schedule_call_pressure_limit_t;

// Builds the capacities reduced by one call from the state's resolved target,
// allocation constraints, and ordinary pressure limits. No IR or provider
// lookup is performed. |limits| provides space for the descriptor set's
// register classes plus register-packing resources. Returns the populated
// prefix length; only reduced capacities have rows.
iree_host_size_t loom_low_schedule_call_pressure_build_limits(
    const loom_low_schedule_build_state_t* state,
    const loom_low_call_contract_t* contract,
    loom_low_call_clobber_list_t common_clobbers,
    loom_low_schedule_call_pressure_limit_t* limits);

// Validates retained call contracts and builds sparse boundary capacities.
iree_status_t loom_low_schedule_call_pressure_initialize(
    loom_low_schedule_build_state_t* state);

// Scores the retained capacities against the call-crossing footprint: current
// live units minus dying arguments, excluding results produced after the call.
// Shared classes consume their aggregate pressure set; packing resources round
// the surviving member units. Accumulates debt without replacing an earlier
// pressure-cliff observation. No IR or call lookup is performed here.
void loom_low_schedule_call_pressure_score_limits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_call_pressure_limit_t* limits, uint32_t limit_count,
    loom_low_schedule_candidate_score_t* score);

// Adds transient pressure debt when live values exceed the storage preserved by
// a retained call candidate.
void loom_low_schedule_call_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_CALL_PRESSURE_H_
