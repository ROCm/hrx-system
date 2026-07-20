// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact target-low residency validation and recorded placement repair.

#ifndef LOOM_CODEGEN_LOW_RESIDENCY_CONTRACT_H_
#define LOOM_CODEGEN_LOW_RESIDENCY_CONTRACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_rematerialization.h"
#include "loom/ir/ir.h"
#include "loom/target/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

// One finite source-planner alternative retained through target lowering.
typedef struct loom_low_residency_contract_candidate_t {
  // Stable source-planner identity retained through configured unrolling.
  uint32_t candidate_id;
  // Projected dynamic materialization cost used for deterministic repair.
  uint32_t recompute_cost;
  // Root target-low value that the accepted placement kept materialized.
  loom_value_id_t value_id;
  // Exact low operand uses covered by this materialization boundary.
  const loom_use_t* uses;
  // Number of entries in |uses|.
  uint32_t use_count;
  // Target-low producer slice implementing the source materialization in
  // dependency order, or NULL for a one-packet legacy candidate.
  loom_op_t* const* materialization_ops;
  // Number of entries in |materialization_ops|.
  uint32_t materialization_op_count;
  // Values captured by, but not defined within, |materialization_ops|.
  const loom_value_id_t* materialization_inputs;
  // Number of entries in |materialization_inputs|.
  uint32_t materialization_input_count;
  // True after this alternative was considered for repair.
  bool attempted;
  // True when every surviving recorded use no longer retains |value_id|.
  bool restored;
  // Target-low packets cloned while attempting this alternative.
  uint32_t cloned_packet_count;
  // Exact target-low operands rewritten while attempting this alternative.
  uint32_t rewritten_operand_count;
  // True when selecting this alternative protected its source baseline.
  bool preserves_baseline;
} loom_low_residency_contract_candidate_t;

// Function-wide exact residency contract consumed by emission-frame repair.
typedef struct loom_low_residency_contract_t {
  // Current exact tier floor, initially the maximum authored requirement.
  uint32_t required_tier;
  // Authored numeric tier floor, or zero when absent.
  uint32_t minimum_required_tier;
  // Projected source baseline used to trigger exact recovery, or zero.
  uint32_t projected_baseline_tier;
  // True when structured source placement emitted an exact requirement.
  bool has_requirement;
  // True when |minimum_required_tier| is an authored numeric floor.
  bool has_minimum_requirement;
  // True when candidates recover the authored placement baseline.
  bool preserves_baseline;
  // True after every baseline-protecting candidate restored the exact
  // authored placement and that allocation established the baseline tier.
  bool preserved_baseline_resolved;
  // Arena-owned recorded materialization alternatives.
  loom_low_residency_contract_candidate_t* candidates;
  // Number of entries in |candidates|.
  iree_host_size_t candidate_count;
} loom_low_residency_contract_t;

// Candidate subset eligible for one exact repair phase.
typedef enum loom_low_residency_repair_scope_e {
  // Every finite placement alternative in the function-wide contract.
  LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES = 0,
  // Only alternatives whose source decision protected the authored baseline.
  LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE = 1,
} loom_low_residency_repair_scope_t;

// Collects and removes low.residency markers from one low function.
//
// The requirement must be the first non-ABI operation in the entry block. A
// contract-free function therefore takes only a bounded preamble probe; the
// complete function walk is paid only when a requirement is present.
// Candidate identity results are replaced by their root source values after
// their exact leaf uses are retained in |arena|. The resulting function is
// ordinary schedulable target-low IR. |arena| must outlive every subsequent
// validation and repair using |out_contract|.
iree_status_t loom_low_residency_contract_consume(
    loom_module_t* module, loom_op_t* low_func_op,
    iree_arena_allocator_t* arena, loom_low_residency_contract_t* out_contract);

// Evaluates exact allocated register extents against |contract|.
iree_status_t loom_low_residency_contract_evaluate(
    const loom_low_residency_contract_t* contract,
    const loom_target_residency_model_t* residency_model,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, uint32_t* out_tier, bool* out_satisfied);

// Attempts one not-yet-considered recorded rematerialization alternative.
//
// Alternatives that no longer have eligible uses are marked attempted and
// skipped in the same call. A nonzero result requires rebuilding scheduling
// and allocation before evaluating the contract again.
iree_status_t loom_low_residency_contract_try_repair(
    loom_module_t* module, const loom_low_allocation_table_t* allocation,
    loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result);

// Attempts every remaining candidate in one finite terminal repair batch.
// |out_repaired_candidate_count| counts candidates that rewrote IR.
iree_status_t loom_low_residency_contract_try_repair_remaining(
    loom_module_t* module, const loom_low_allocation_table_t* allocation,
    loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result,
    uint32_t* out_repaired_candidate_count);

// Returns true after every finite candidate in |scope| has been considered.
bool loom_low_residency_contract_candidates_exhausted(
    const loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope);

// Returns true when every finite candidate in |scope| restored its uses.
bool loom_low_residency_contract_candidates_restored(
    const loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope);

// Replaces an optimistic projected preserve floor with the exact fully
// restored baseline while retaining any authored numeric minimum.
void loom_low_residency_contract_resolve_preserved_baseline(
    loom_low_residency_contract_t* contract, uint32_t exact_baseline_tier);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_RESIDENCY_CONTRACT_H_
