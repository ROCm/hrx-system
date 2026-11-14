// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation materialization for target-low functions.
//
// Allocation tables keep physical placement and spill decisions outside the
// SSA program. This layer turns selected table decisions back into explicit
// low IR when later emission needs real storage traffic. Generic targets
// receive structural low.spill/low.reload operations; a target planner may
// instead select and emit its final traffic recipe from the same immutable
// materialization plan.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MATERIALIZATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;
typedef struct loom_low_spill_materialization_plan_t
    loom_low_spill_materialization_plan_t;

typedef uint32_t loom_low_allocation_materialized_spill_flags_t;

enum {
  // The spilled value was a block argument when spill traffic was materialized.
  LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT =
      (1u << 0),
};

typedef struct loom_low_allocation_materialized_spill_t {
  // SSA value represented by the materialized spill storage.
  loom_value_id_t value_id;
  // Original register class in the emission frame's target descriptor set.
  // Retained across allocation snapshot rebuilds after spill insertion.
  uint16_t descriptor_reg_class_id;
  // Materialized spill record flags.
  loom_low_allocation_materialized_spill_flags_t flags;
  // Allocation assignment index associated with |value_id|.
  uint32_t assignment_index;
  // Logical spill slot ordinal assigned before physical reservation reuse.
  uint32_t slot_index;
  // Spill storage space used for the materialized storage.
  loom_low_spill_slot_space_t slot_space;
  // Slot size in bytes.
  uint64_t byte_size;
  // Required slot alignment in bytes.
  uint64_t byte_alignment;
  // Number of spill stores materialized for this value.
  uint64_t store_count;
  // Byte traffic from spill stores materialized for this value.
  uint64_t store_bytes;
  // Number of spill reloads materialized for this value.
  uint64_t reload_count;
  // Byte traffic from spill reloads materialized for this value.
  uint64_t reload_bytes;
} loom_low_allocation_materialized_spill_t;

typedef struct loom_low_allocation_materialized_spill_vec_t {
  // Materialized spill records owned by the caller-provided arena.
  const loom_low_allocation_materialized_spill_t* records;
  // Number of records in |records|.
  iree_host_size_t record_count;
  // Next record chunk in insertion order.
  struct loom_low_allocation_materialized_spill_vec_t* next;
} loom_low_allocation_materialized_spill_vec_t;

typedef struct loom_low_allocation_materialized_spill_list_t {
  // First record chunk in insertion order.
  loom_low_allocation_materialized_spill_vec_t* head;
  // Last record chunk in insertion order.
  loom_low_allocation_materialized_spill_vec_t* tail;
  // Total number of records across all chunks.
  iree_host_size_t record_count;
} loom_low_allocation_materialized_spill_list_t;

typedef struct loom_low_allocation_materialized_value_list_t {
  // Materialized values owned by the caller-provided arena.
  const loom_value_id_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_low_allocation_materialized_value_list_t;

typedef uint8_t loom_low_spill_materialized_value_flags_t;

enum {
  // The value is defined outside the surrounding program point's semantic
  // execution domain and must acquire storage at its own definition.
  LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_EXPANDED_EXECUTION = 1u << 0,
  // The value is private setup for one recipe-local consumer and may use
  // per-user placement.
  LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_PRIVATE_TO_CONSUMER = 1u << 1,
};

typedef struct loom_low_spill_materialized_value_t {
  // Register value emitted while realizing one planned traffic action.
  loom_value_id_t value_id;
  // Placement facts established by the retained traffic recipe.
  loom_low_spill_materialized_value_flags_t flags;
} loom_low_spill_materialized_value_t;

// Concrete result of applying one planned spill traffic action.
typedef struct loom_low_spill_materialized_action_t {
  // Exported reload result, or LOOM_VALUE_ID_INVALID for a store action.
  loom_value_id_t value_id;
  // Arena-owned register values created by this action's traffic recipe.
  const loom_low_spill_materialized_value_t* emitted_values;
  // Number of entries in |emitted_values|.
  iree_host_size_t emitted_value_count;
} loom_low_spill_materialized_action_t;

// Emits the already-selected traffic recipe for |instruction_index| at
// |builder|'s current insertion point. Store actions receive a valid
// |source_value_id|; reload actions return one exported value. The callback
// performs no IR discovery or recipe selection.
typedef iree_status_t (*loom_low_spill_materialization_emit_traffic_fn_t)(
    void* user_data, iree_host_size_t instruction_index,
    loom_builder_t* builder, loom_value_id_t storage_value_id,
    loom_value_id_t source_value_id, loom_location_id_t location,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialized_action_t* out_action);

typedef struct loom_low_spill_materialization_traffic_plan_t {
  // Target-owned immutable plan passed to |emit|.
  void* user_data;
  // Traffic emitter for every retained execution-order instruction.
  loom_low_spill_materialization_emit_traffic_fn_t emit;
} loom_low_spill_materialization_traffic_plan_t;

// Selects target traffic recipes from one immutable generic materialization
// plan before any storage or traffic IR is created. The returned plan and its
// payload are owned by |arena| and consumed during the same materialization.
// Program rejection emits through |emitter| and sets |out_accepted| to false;
// statuses carry infrastructure failures. No IR is mutated on rejection.
typedef iree_status_t (*loom_low_spill_materialization_plan_traffic_fn_t)(
    void* user_data, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    bool* out_accepted,
    loom_low_spill_materialization_traffic_plan_t* out_plan);

typedef struct loom_low_allocation_materialization_options_t {
  // True when |supported_storage_spaces| names the complete target lowering
  // capability for newly generated spill storage.
  bool has_supported_storage_spaces;
  // Storage spaces this materialization may create for spill slots.
  loom_low_storage_space_set_t supported_storage_spaces;
  // True when an explicit diagnostic mode requests BACKEND/009 feedback for
  // successful spill insertion. Compile reports use
  // |record_materialized_spills| instead.
  bool emit_spill_diagnostics;
  // True when materialized spill records should be retained in |out_result|.
  bool record_materialized_spills;
  // Borrowed module-value membership for retained private producer placement.
  // Materialization clears values whose users it changes, preserving placement
  // for other producers across the next schedule rebuild. May be empty.
  iree_bitmap_t per_user_placement_values;
  // Optional target traffic planner. When absent, materialization emits the
  // generic low.spill/low.reload representation.
  loom_low_spill_materialization_plan_traffic_fn_t plan_traffic;
  // Caller-owned data passed to |plan_traffic|.
  void* plan_traffic_user_data;
  // Structured diagnostic emitter for materialized allocation feedback.
  iree_diagnostic_emitter_t emitter;
} loom_low_allocation_materialization_options_t;

typedef struct loom_low_allocation_materialization_result_t {
  // Number of error diagnostics emitted before mutating the low function.
  uint32_t error_count;
  // Register values required by the materialized traffic representation.
  // Includes generic reload results and values consumed by generic stores, or
  // the helper values published by a directly emitted target recipe.
  loom_low_allocation_materialized_value_list_t required_register_values;
  // Single-consumer reload results and recipe helpers private to their
  // consumer setup chain.
  loom_low_allocation_materialized_value_list_t per_user_placement_values;
  // Fresh register values defined outside their surrounding semantic execution
  // domain by a target traffic recipe.
  loom_low_allocation_materialized_value_list_t expanded_execution_values;
  // Number of distinct low.storage.reserve ops created after lifetime reuse.
  uint32_t storage_count;
  // Number of spill stores materialized.
  uint32_t spill_count;
  // Number of spill reloads materialized.
  uint32_t reload_count;
  // Byte size of the distinct reservations, excluding inter-reservation
  // padding.
  uint64_t storage_bytes;
  // Strongest byte alignment required by the created spill storage.
  uint64_t storage_minimum_alignment;
  // Byte traffic from spill stores materialized.
  uint64_t spill_bytes;
  // Byte traffic from spill reloads materialized.
  uint64_t reload_bytes;
  // Materialized spill records owned by the caller-provided arena.
  const loom_low_allocation_materialized_spill_t* materialized_spills;
  // Number of records in |materialized_spills|.
  iree_host_size_t materialized_spill_count;
} loom_low_allocation_materialization_result_t;

// Materializes the selected |spill_plans| from |table| into storage
// reservations and generic or target-planned traffic. The allocation snapshot
// remains immutable; callers may omit plans whose values were reconstructed
// without storage. Existing entry-block storage reservations are moved to its
// prefix after ABI live-in/resource imports without changing their relative
// order or byte offsets. New reservations append to that prefix, dominating
// all generated traffic even when earlier repairs inserted executable entry
// copies. Stores are inserted at the defining point of each spilled op-result
// value, function-region non-entry block argument stores are inserted on
// incoming edges before the branch, and reloads are inserted immediately before
// each remaining original operand use. Materialized function-region non-entry
// block arguments are removed from the block signature and from all predecessor
// Low direct-branch payloads. Nested-region block arguments are register ABI
// values and never appear in a spill plan.
iree_status_t loom_low_allocation_materialize_spills(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MATERIALIZATION_H_
