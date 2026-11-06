// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Protective ownership for source-relative borrows in a scheduled snapshot.
#ifndef LOOM_CODEGEN_LOW_SCHEDULE_BORROW_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_BORROW_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_low_borrow_protection_t;
enum loom_low_borrow_protection_e {
  LOOM_LOW_BORROW_NONE = 0,
  // The original owner covers every use in this scheduled basic block.
  LOOM_LOW_BORROW_COVERED,
  // The owner may retire at or before a pointer use.
  LOOM_LOW_BORROW_OWNER_RETIRED,
  // The pointer chain crosses a control-flow boundary.
  LOOM_LOW_BORROW_CONTROL_FLOW,
  // A storage or escaping boundary may keep the pointer beyond this block.
  LOOM_LOW_BORROW_CAPTURED,
  // Required storage sharing does not establish independent payload lifetimes.
  LOOM_LOW_BORROW_SHARED_STORAGE,
};

typedef struct loom_low_borrow_decision_t {
  // Borrow operation's node in the retained schedule.
  uint32_t node_index;
  // Last observed use of the pointer component in the borrow block, or the
  // borrow for an unused ptr.
  uint32_t last_use_node;
  // Boundary that prevents elision, or UINT32_MAX for a covered pointer.
  uint32_t reason_node;
  // Protection selected for the pointer result.
  loom_low_borrow_protection_t protection;
} loom_low_borrow_decision_t;

typedef struct loom_low_borrow_plan_t {
  // Decisions indexed by borrow result ordinal; other values contain NONE.
  const loom_low_borrow_protection_t* values;
  // Sparse decisions in source node order, retained for reporting.
  const loom_low_borrow_decision_t* decisions;
  // Number of borrow operations in this immutable snapshot.
  uint32_t count;
} loom_low_borrow_plan_t;

struct loom_low_schedule_table_t;

// Builds once after scheduling. The plan borrows this immutable schedule and
// lives in |arena|; any rewriting or rescheduling requires a new plan. Exact
// storage identities come from the function model, not a second alias walk.
// Copy components are flattened once before use queries. Elision proves a
// block-local covering owner; uncertain control/storage paths retain
// protection.
iree_status_t loom_low_schedule_borrow_plan_build(
    const struct loom_low_schedule_table_t* schedule,
    const loom_value_ordinal_t* storage_origins,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    loom_low_borrow_plan_t* out_plan);

// Reports protection from the final accepted schedule. Local proofs identify
// the final pointer-copy use; uncertain paths identify the boundary beyond
// which the owner could not be proved to cover the pointer's lifetime.
iree_status_t loom_low_schedule_borrow_diagnostics_emit(
    const struct loom_low_schedule_table_t* schedule,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_CODEGEN_LOW_SCHEDULE_BORROW_H_
