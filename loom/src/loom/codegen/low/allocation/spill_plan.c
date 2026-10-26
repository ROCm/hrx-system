// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_plan.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/spill_storage.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/util/adaptive_sort.h"

#define LOOM_LOW_ALLOCATION_SPILL_PLAN_MAX_NATURAL_ALIGNMENT 16u

static uint32_t loom_low_allocation_spill_plan_natural_chunk_units(
    uint32_t unit_count) {
  if (unit_count >= 4) {
    return 4;
  }
  if (unit_count >= 2) {
    return 2;
  }
  return 1;
}

iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment) {
  uint64_t bit_size = (uint64_t)assignment->unit_count * alloc_unit_bits;
  uint64_t byte_size = (bit_size + 7u) / 8u;
  if (byte_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "spill slot byte size exceeds uint32_t");
  }
  uint32_t unit_byte_size = ((uint32_t)alloc_unit_bits + 7u) / 8u;
  const uint32_t chunk_units =
      loom_low_allocation_spill_plan_natural_chunk_units(
          assignment->unit_count);
  uint64_t natural_alignment = (uint64_t)unit_byte_size * chunk_units;
  if (natural_alignment >
      LOOM_LOW_ALLOCATION_SPILL_PLAN_MAX_NATURAL_ALIGNMENT) {
    natural_alignment = LOOM_LOW_ALLOCATION_SPILL_PLAN_MAX_NATURAL_ALIGNMENT;
  }
  const uint32_t byte_alignment =
      iree_math_round_up_to_pow2_u32((uint32_t)natural_alignment);
  *out_byte_size = (uint32_t)byte_size;
  *out_byte_alignment = byte_alignment;
  return iree_ok_status();
}

bool loom_low_allocation_spill_plan_slice_reload_byte_size(
    const loom_module_t* module,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t spill_byte_size, const loom_op_t* use_op,
    uint32_t* out_reload_byte_size) {
  *out_reload_byte_size = 0;
  if (!loom_low_slice_isa(use_op) ||
      spill_byte_size % assignment->unit_count != 0) {
    return false;
  }
  const uint32_t unit_byte_size = spill_byte_size / assignment->unit_count;
  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_slice_result(use_op));
  *out_reload_byte_size =
      unit_byte_size * loom_low_register_type_unit_count(result_type);
  return true;
}

bool loom_low_allocation_spill_plan_use_full_slice_reload(
    uint32_t slice_count, uint64_t narrow_reload_bytes,
    uint32_t spill_byte_size) {
  return slice_count > 1 &&
         (narrow_reload_bytes > spill_byte_size ||
          slice_count >=
              LOOM_LOW_ALLOCATION_DENSE_SLICE_RELOAD_MIN_SLICE_COUNT);
}

typedef struct loom_low_spill_materialization_logical_use_t {
  // Operand occurrence represented by this logical use.
  loom_low_spill_materialization_target_t target;
  // Branch payload cell index, or UINT32_MAX for other occurrences.
  uint32_t branch_payload_index;
} loom_low_spill_materialization_logical_use_t;

typedef struct loom_low_spill_materialization_generated_use_t {
  // Action whose source operand uses the selected original value.
  uint32_t action_index;
  // Next use record index plus one, or zero at the end of the list.
  uint32_t next_index_plus_one;
} loom_low_spill_materialization_generated_use_t;

typedef struct loom_low_spill_materialization_generated_use_list_t {
  // First generated-use record index plus one, or zero for an empty list.
  uint32_t head_index_plus_one;
  // Last generated-use record index plus one, or zero for an empty list.
  uint32_t tail_index_plus_one;
} loom_low_spill_materialization_generated_use_list_t;

typedef struct loom_low_spill_materialization_branch_payload_t {
  // Current logical value forwarded by this original payload position.
  loom_low_spill_materialization_value_ref_t current_value;
  // Original destination block argument identifying this payload position.
  loom_value_id_t block_argument_value_id;
  // True until an earlier action removes this original payload position.
  bool present;
} loom_low_spill_materialization_branch_payload_t;

typedef struct loom_low_spill_materialization_slice_group_t {
  // Earliest slice operation in source program order.
  loom_op_t* first_slice_op;
  // Total bytes required by independent narrow reloads.
  uint64_t narrow_reload_bytes;
  // First target slot reserved for this group.
  uint32_t target_start;
  // Next target slot to initialize while populating this group.
  uint32_t target_cursor;
  // Number of slice uses in this group.
  uint32_t slice_count;
  // Dense source-body block index shared by the slice uses.
  uint16_t block_index;
  // True when one full reload replaces the independent narrow reloads.
  bool use_full_reload;
} loom_low_spill_materialization_slice_group_t;

typedef struct loom_low_spill_materialization_plan_builder_t {
  // Immutable allocation snapshot supplying assignments and CFG adjacency.
  const loom_low_allocation_table_t* table;
  // Selected spill plans in materialization order.
  const loom_low_allocation_spill_plan_t* spill_plans;
  // Number of records in |spill_plans| and |records|.
  uint32_t spill_plan_count;
  // Function body represented by the allocation snapshot.
  const loom_region_t* body;
  // Temporary arena released after the retained plan is copied.
  iree_arena_allocator_t scratch_arena;
  // Action vector under construction.
  loom_low_spill_materialization_action_t* actions;
  // Number of initialized records in |actions|.
  uint32_t action_count;
  // Allocated record capacity in |actions|.
  iree_host_size_t action_capacity;
  // Shared-slice operand target vector under construction.
  loom_low_spill_materialization_target_t* targets;
  // Number of initialized records in |targets|.
  uint32_t target_count;
  // Allocated record capacity in |targets|.
  iree_host_size_t target_capacity;
  // Exact per-spill records under construction.
  loom_low_spill_materialization_record_t* records;
  // Earliest traffic-producing plan sharing each reservation, or NULL when
  // active plans have independent slots.
  uint32_t* slot_representatives;
  // Generated-use list heads indexed by selected spill-plan ordinal.
  loom_low_spill_materialization_generated_use_list_t* generated_use_lists;
  // Generated action operands absent from the immutable source use lists.
  loom_low_spill_materialization_generated_use_t* generated_uses;
  // Number of initialized records in |generated_uses|.
  uint32_t generated_use_count;
  // Module-owned value scratch encoding plan states and replacement actions.
  loom_value_u32_scratch_t* value_index;
  // First branch-payload cell indexed by dense source-body block.
  uint32_t* branch_payload_starts;
  // Logical state for every original top-level low.br payload.
  loom_low_spill_materialization_branch_payload_t* branch_payloads;
  // Number of records in |branch_payloads|.
  uint32_t branch_payload_count;
  // Reused group index plus one indexed by source-body block.
  uint32_t* slice_group_indices_plus_one_by_block;
  // Reused slice groups bounded by selected-value uses and body blocks.
  loom_low_spill_materialization_slice_group_t* slice_groups;
} loom_low_spill_materialization_plan_builder_t;

static loom_type_t loom_low_spill_materialization_action_result_type(
    const loom_module_t* module,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_spill_materialization_plan_t* plan, uint32_t action_index) {
  IREE_ASSERT_LT(action_index, plan->action_count);
  const loom_low_spill_materialization_action_t* action =
      &plan->actions[action_index];
  const loom_low_allocation_spill_plan_t* spill_plan =
      &spill_plans[action->spill_plan_index];
  switch (action->kind) {
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD:
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD:
      return loom_module_value_type(module, spill_plan->value_id);
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD:
      return loom_module_value_type(
          module,
          loom_low_slice_result(action->value.narrow_slice_reload.slice_op));
    default:
      IREE_ASSERT_UNREACHABLE(
          "spill materialization action does not produce a value");
      IREE_BUILTIN_UNREACHABLE();
  }
}

bool loom_low_spill_materialization_plan_query_traffic(
    const loom_module_t* module,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_spill_materialization_plan_t* plan,
    iree_host_size_t action_index,
    loom_low_spill_materialization_traffic_t* out_traffic) {
  IREE_ASSERT_LT(action_index, plan->action_count);
  const loom_low_spill_materialization_action_t* action =
      &plan->actions[action_index];
  const loom_low_allocation_spill_plan_t* spill_plan =
      &spill_plans[action->spill_plan_index];
  *out_traffic = (loom_low_spill_materialization_traffic_t){
      .value_type = loom_type_none(),
  };
  switch (action->kind) {
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE:
      out_traffic->kind = LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_STORE;
      out_traffic->value_type =
          loom_module_value_type(module, spill_plan->value_id);
      return true;
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE: {
      out_traffic->kind = LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_STORE;
      const loom_low_spill_materialization_value_ref_t source =
          action->value.edge_store.source;
      out_traffic->value_type =
          source.action_index == UINT32_MAX
              ? loom_module_value_type(module, source.value_id)
              : loom_low_spill_materialization_action_result_type(
                    module, spill_plans, plan, source.action_index);
      return true;
    }
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD:
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD:
      out_traffic->kind = LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_RELOAD;
      out_traffic->value_type =
          loom_module_value_type(module, spill_plan->value_id);
      return true;
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD:
      out_traffic->kind = LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_RELOAD;
      out_traffic->value_type =
          loom_low_spill_materialization_action_result_type(
              module, spill_plans, plan, (uint32_t)action_index);
      out_traffic->byte_offset = action->value.narrow_slice_reload.byte_offset;
      return true;
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT:
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT:
      return false;
  }
  IREE_ASSERT_UNREACHABLE("unknown spill materialization action kind");
  IREE_BUILTIN_UNREACHABLE();
}

typedef struct loom_low_spill_materialization_instruction_sort_t {
  // Retained instruction projection copied after sorting.
  loom_low_spill_materialization_instruction_t instruction;
  // Insertion class and order for instructions sharing one anchor.
  uint32_t root_order;
  // Order before the root instruction, or UINT32_MAX for the root itself.
  uint32_t local_order;
} loom_low_spill_materialization_instruction_sort_t;

#define LOOM_LOW_SPILL_MATERIALIZATION_ORDINARY_ROOT_BIT (UINT32_C(1) << 31)

static uint32_t loom_low_spill_materialization_instruction_point(
    const loom_liveness_analysis_t* liveness,
    const loom_low_spill_materialization_instruction_t* instruction) {
  const loom_liveness_operation_point_t* operation =
      loom_liveness_operation_at(liveness, instruction->operation_index);
  if (loom_low_spill_materialization_instruction_is_after(instruction)) {
    return operation->end_point;
  }
  IREE_ASSERT_LT(operation->start_point, UINT32_MAX);
  return operation->start_point + 1u;
}

static bool loom_low_spill_materialization_instruction_less(
    const loom_liveness_analysis_t* liveness,
    const loom_low_spill_materialization_instruction_sort_t* lhs,
    const loom_low_spill_materialization_instruction_sort_t* rhs) {
  const uint32_t lhs_point = loom_low_spill_materialization_instruction_point(
      liveness, &lhs->instruction);
  const uint32_t rhs_point = loom_low_spill_materialization_instruction_point(
      liveness, &rhs->instruction);
  if (lhs_point != rhs_point) {
    return lhs_point < rhs_point;
  }
  const bool lhs_after =
      loom_low_spill_materialization_instruction_is_after(&lhs->instruction);
  const bool rhs_after =
      loom_low_spill_materialization_instruction_is_after(&rhs->instruction);
  if (lhs_after != rhs_after) {
    return !lhs_after;
  }
  if (lhs->root_order != rhs->root_order) {
    return lhs->root_order < rhs->root_order;
  }
  return lhs->local_order < rhs->local_order;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT_OUTLINED_HEAP(
    loom_low_spill_materialization_sort_instructions,
    loom_low_spill_materialization_instruction_sort_t,
    const loom_liveness_analysis_t*,
    loom_low_spill_materialization_instruction_less)

static loom_low_spill_materialization_value_ref_t
loom_low_spill_materialization_original_value(loom_value_id_t value_id) {
  return (loom_low_spill_materialization_value_ref_t){
      .value_id = value_id,
      .action_index = UINT32_MAX,
  };
}

static loom_low_spill_materialization_value_ref_t
loom_low_spill_materialization_action_value(uint32_t action_index) {
  return (loom_low_spill_materialization_value_ref_t){
      .value_id = LOOM_VALUE_ID_INVALID,
      .action_index = action_index,
  };
}

#define LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT (UINT32_C(1) << 31)

static bool loom_low_spill_materialization_try_spill_plan_index(
    const loom_low_spill_materialization_plan_builder_t* builder,
    loom_value_id_t value_id, uint32_t* out_spill_plan_index) {
  const uint32_t value_index =
      loom_value_u32_scratch_load(builder->value_index, value_id);
  if (value_index == LOOM_VALUE_ORDINAL_INVALID ||
      iree_any_bit_set(value_index,
                       LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT)) {
    return false;
  }
  *out_spill_plan_index = value_index;
  return true;
}

static bool loom_low_spill_materialization_try_replacement(
    const loom_low_spill_materialization_plan_builder_t* builder,
    loom_value_id_t value_id, uint32_t* out_action_index) {
  const uint32_t value_index =
      loom_value_u32_scratch_load(builder->value_index, value_id);
  if (value_index == LOOM_VALUE_ORDINAL_INVALID ||
      !iree_any_bit_set(value_index,
                        LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT)) {
    return false;
  }
  *out_action_index =
      value_index & ~LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT;
  return true;
}

static loom_low_spill_materialization_value_ref_t
loom_low_spill_materialization_resolve_value(
    const loom_low_spill_materialization_plan_builder_t* builder,
    loom_low_spill_materialization_value_ref_t value) {
  if (value.action_index != UINT32_MAX) {
    return value;
  }
  uint32_t replacement_action_index = 0;
  if (loom_low_spill_materialization_try_replacement(
          builder, value.value_id, &replacement_action_index)) {
    return loom_low_spill_materialization_action_value(
        replacement_action_index);
  }
  return value;
}

static uint32_t loom_low_spill_materialization_append_action(
    loom_low_spill_materialization_plan_builder_t* builder,
    loom_low_spill_materialization_action_t action) {
  IREE_ASSERT_LT(builder->action_count, builder->action_capacity);
  const uint32_t action_index = builder->action_count++;
  builder->actions[action_index] = action;
  return action_index;
}

static void loom_low_spill_materialization_record_action_use(
    loom_low_spill_materialization_plan_builder_t* builder,
    loom_low_spill_materialization_value_ref_t source, uint32_t action_index) {
  source = loom_low_spill_materialization_resolve_value(builder, source);
  if (source.action_index != UINT32_MAX) {
    return;
  }
  uint32_t spill_plan_index = 0;
  if (!loom_low_spill_materialization_try_spill_plan_index(
          builder, source.value_id, &spill_plan_index)) {
    return;
  }
  const uint32_t use_index = builder->generated_use_count++;
  builder->generated_uses[use_index] =
      (loom_low_spill_materialization_generated_use_t){
          .action_index = action_index,
      };
  loom_low_spill_materialization_generated_use_list_t* list =
      &builder->generated_use_lists[spill_plan_index];
  if (list->tail_index_plus_one == 0) {
    list->head_index_plus_one = use_index + 1;
  } else {
    builder->generated_uses[list->tail_index_plus_one - 1].next_index_plus_one =
        use_index + 1;
  }
  list->tail_index_plus_one = use_index + 1;
}

static iree_status_t loom_low_spill_materialization_initialize_branches(
    loom_low_spill_materialization_plan_builder_t* builder) {
  const iree_host_size_t block_count = builder->body->block_count;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->scratch_arena, block_count + 1,
                                sizeof(*builder->branch_payload_starts),
                                (void**)&builder->branch_payload_starts));
  uint64_t payload_count = 0;
  for (iree_host_size_t i = 0; i < block_count; ++i) {
    builder->branch_payload_starts[i] = (uint32_t)payload_count;
    const loom_op_t* terminator =
        loom_block_const_last_op(builder->body->blocks[i]);
    if (terminator && loom_low_br_isa(terminator)) {
      payload_count += terminator->operand_count;
    }
    if (payload_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "spill branch payload count overflow");
    }
  }
  builder->branch_payload_starts[block_count] = (uint32_t)payload_count;
  builder->branch_payload_count = (uint32_t)payload_count;
  if (payload_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &builder->scratch_arena, (iree_host_size_t)payload_count,
      sizeof(*builder->branch_payloads), (void**)&builder->branch_payloads));
  for (iree_host_size_t i = 0; i < block_count; ++i) {
    const loom_op_t* terminator =
        loom_block_const_last_op(builder->body->blocks[i]);
    if (!terminator || !loom_low_br_isa(terminator)) {
      continue;
    }
    loom_block_t* destination = loom_low_br_dest(terminator);
    const loom_value_id_t* operands = loom_op_const_operands(terminator);
    for (uint16_t j = 0; j < terminator->operand_count; ++j) {
      IREE_ASSERT_LT(j, destination->arg_count);
      const uint32_t payload_index = builder->branch_payload_starts[i] + j;
      builder->branch_payloads[payload_index] =
          (loom_low_spill_materialization_branch_payload_t){
              .current_value =
                  loom_low_spill_materialization_original_value(operands[j]),
              .block_argument_value_id = destination->arg_ids[j],
              .present = true,
          };
    }
  }
  return iree_ok_status();
}

static bool loom_low_spill_materialization_try_branch_payload_index(
    const loom_low_spill_materialization_plan_builder_t* builder,
    loom_use_t use, uint32_t* out_payload_index) {
  loom_op_t* user_op = loom_use_user_op(use);
  loom_block_t* source_block = user_op->parent_block;
  if (!loom_low_br_isa(user_op) ||
      source_block->parent_region != builder->body ||
      loom_block_const_last_op(source_block) != user_op) {
    return false;
  }
  const uint16_t block_index = loom_block_region_index(source_block);
  const uint16_t operand_index = loom_use_operand_index(use);
  const uint32_t payload_start = builder->branch_payload_starts[block_index];
  const uint32_t payload_end = builder->branch_payload_starts[block_index + 1];
  IREE_ASSERT_LT(payload_start + operand_index, payload_end);
  *out_payload_index = payload_start + operand_index;
  return true;
}

static iree_status_t loom_low_spill_materialization_initialize_values(
    loom_low_spill_materialization_plan_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &builder->scratch_arena, builder->spill_plan_count,
      sizeof(*builder->generated_use_lists),
      (void**)&builder->generated_use_lists));
  memset(builder->generated_use_lists, 0,
         (iree_host_size_t)builder->spill_plan_count *
             sizeof(*builder->generated_use_lists));
  uint64_t initial_use_count = 0;
  uint64_t predecessor_count = 0;
  uint32_t maximum_initial_use_count = 0;
  for (uint32_t i = 0; i < builder->spill_plan_count; ++i) {
    const loom_value_id_t value_id = builder->spill_plans[i].value_id;
    loom_module_value_ordinal_scratch_set(builder->table->module, value_id, i);
    const loom_value_t* value =
        loom_module_value(builder->table->module, value_id);
    initial_use_count += value->use_count;
    maximum_initial_use_count =
        iree_max(maximum_initial_use_count, value->use_count);
    if (loom_value_is_block_arg(value)) {
      const loom_block_t* block = loom_value_def_block(value);
      if (block != loom_region_const_entry_block(builder->body)) {
        predecessor_count +=
            loom_cfg_graph_predecessors(&builder->table->cfg_graph,
                                        loom_block_region_index(block))
                .count;
      }
    }
  }

  // Let U be the original selected-value uses, P the selected spill plans,
  // and B the predecessor incidences of selected non-entry block arguments.
  // An original use adds at most one reload action. A spill plan adds at most
  // one definition store and one block-argument removal. Each predecessor adds
  // at most one edge store, one branch-argument removal, and one generated use
  // of the edge-store source that may require a later reload. Therefore
  // U + 2P + 3B is an upper bound for the complete ordered action sequence.
  const uint64_t action_capacity = initial_use_count +
                                   (uint64_t)builder->spill_plan_count * 2u +
                                   predecessor_count * 3u;
  if (action_capacity > LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "spill materialization action count overflow");
  }
  builder->action_capacity = (iree_host_size_t)action_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &builder->scratch_arena, builder->action_capacity,
      sizeof(*builder->actions), (void**)&builder->actions));
  builder->target_capacity = (iree_host_size_t)initial_use_count;
  if (builder->target_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &builder->scratch_arena, builder->target_capacity,
        sizeof(*builder->targets), (void**)&builder->targets));
  }
  if (predecessor_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &builder->scratch_arena, (iree_host_size_t)predecessor_count,
        sizeof(*builder->generated_uses), (void**)&builder->generated_uses));
  }
  const iree_host_size_t slice_group_capacity = iree_min(
      (iree_host_size_t)maximum_initial_use_count, builder->body->block_count);
  if (slice_group_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &builder->scratch_arena, slice_group_capacity,
        sizeof(*builder->slice_groups), (void**)&builder->slice_groups));
  }
  return iree_ok_status();
}

static bool loom_low_spill_materialization_try_source_use(
    const loom_low_spill_materialization_plan_builder_t* builder,
    loom_value_id_t value_id, bool remove_argument_payload,
    loom_use_t source_use,
    loom_low_spill_materialization_logical_use_t* out_use) {
  *out_use = (loom_low_spill_materialization_logical_use_t){
      .target =
          {
              .kind = LOOM_LOW_SPILL_MATERIALIZATION_TARGET_OPERATION_OPERAND,
              .operand_index = loom_use_operand_index(source_use),
              .owner.operation = loom_use_user_op(source_use),
          },
      .branch_payload_index = UINT32_MAX,
  };
  uint32_t payload_index = 0;
  if (!loom_low_spill_materialization_try_branch_payload_index(
          builder, source_use, &payload_index)) {
    return true;
  }

  const loom_low_spill_materialization_branch_payload_t* payload =
      &builder->branch_payloads[payload_index];
  if (!payload->present) {
    return false;
  }
  const loom_low_spill_materialization_value_ref_t current_value =
      loom_low_spill_materialization_resolve_value(builder,
                                                   payload->current_value);
  if (current_value.action_index != UINT32_MAX ||
      current_value.value_id != value_id ||
      (remove_argument_payload &&
       payload->block_argument_value_id == value_id)) {
    return false;
  }
  out_use->target = (loom_low_spill_materialization_target_t){
      .kind = LOOM_LOW_SPILL_MATERIALIZATION_TARGET_BRANCH_ARGUMENT,
      .block_argument_value_id = payload->block_argument_value_id,
      .owner.predecessor_block = loom_use_user_op(source_use)->parent_block,
  };
  out_use->branch_payload_index = payload_index;
  return true;
}

static bool loom_low_spill_materialization_try_slice_use(
    const loom_low_spill_materialization_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_allocation_spill_plan_t* spill_plan, loom_use_t source_use,
    loom_op_t** out_slice_op, uint16_t* out_block_index,
    uint32_t* out_reload_byte_size) {
  loom_op_t* slice_op = loom_use_user_op(source_use);
  if (!loom_low_allocation_spill_plan_slice_reload_byte_size(
          builder->table->module, assignment, spill_plan->byte_size, slice_op,
          out_reload_byte_size) ||
      !loom_region_try_block_index(builder->body, slice_op->parent_block,
                                   out_block_index)) {
    return false;
  }
  *out_slice_op = slice_op;
  return true;
}

static void loom_low_spill_materialization_prepare_slice_groups(
    loom_low_spill_materialization_plan_builder_t* builder,
    uint32_t spill_plan_index, uint32_t* out_group_count) {
  *out_group_count = 0;
  const loom_low_allocation_spill_plan_t* spill_plan =
      &builder->spill_plans[spill_plan_index];
  const loom_low_allocation_assignment_t* assignment =
      &builder->table->assignments[spill_plan->assignment_index];
  if (assignment->unit_count <= 1 ||
      spill_plan->byte_size % assignment->unit_count != 0) {
    return;
  }

  const loom_value_id_t value_id = spill_plan->value_id;
  const loom_value_t* value =
      loom_module_value(builder->table->module, value_id);
  const loom_use_t* source_uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* slice_op = NULL;
    uint32_t reload_byte_size = 0;
    uint16_t block_index = 0;
    if (!loom_low_spill_materialization_try_slice_use(
            builder, assignment, spill_plan, source_uses[i], &slice_op,
            &block_index, &reload_byte_size)) {
      continue;
    }
    uint32_t group_index =
        builder->slice_group_indices_plus_one_by_block[block_index];
    if (group_index == 0) {
      group_index = (*out_group_count)++;
      builder->slice_group_indices_plus_one_by_block[block_index] =
          group_index + 1;
      builder->slice_groups[group_index] =
          (loom_low_spill_materialization_slice_group_t){
              .block_index = block_index,
              .first_slice_op = slice_op,
          };
    } else {
      --group_index;
      if (slice_op->block_ordinal <
          builder->slice_groups[group_index].first_slice_op->block_ordinal) {
        builder->slice_groups[group_index].first_slice_op = slice_op;
      }
    }
    ++builder->slice_groups[group_index].slice_count;
    builder->slice_groups[group_index].narrow_reload_bytes += reload_byte_size;
  }

  for (uint32_t i = 0; i < *out_group_count; ++i) {
    loom_low_spill_materialization_slice_group_t* group =
        &builder->slice_groups[i];
    group->use_full_reload =
        loom_low_allocation_spill_plan_use_full_slice_reload(
            group->slice_count, group->narrow_reload_bytes,
            spill_plan->byte_size);
  }
}

static void loom_low_spill_materialization_append_operand_reload(
    loom_low_spill_materialization_plan_builder_t* builder,
    uint32_t spill_plan_index, loom_low_spill_materialization_logical_use_t use,
    loom_low_spill_materialization_record_t* record) {
  const loom_low_allocation_spill_plan_t* spill_plan =
      &builder->spill_plans[spill_plan_index];
  const uint32_t action_index = loom_low_spill_materialization_append_action(
      builder, (loom_low_spill_materialization_action_t){
                   .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD,
                   .spill_plan_index = spill_plan_index,
                   .value.operand_reload = {.target = use.target},
               });
  if (use.branch_payload_index != UINT32_MAX) {
    builder->branch_payloads[use.branch_payload_index].current_value =
        loom_low_spill_materialization_action_value(action_index);
  }
  ++record->reload_count;
  record->reload_bytes += spill_plan->byte_size;
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
static void loom_low_spill_materialization_append_reloads(
    loom_low_spill_materialization_plan_builder_t* builder,
    uint32_t spill_plan_index, bool remove_argument_payload,
    uint32_t group_count, loom_low_spill_materialization_record_t* record) {
  const loom_low_allocation_spill_plan_t* spill_plan =
      &builder->spill_plans[spill_plan_index];
  const loom_low_allocation_assignment_t* assignment =
      &builder->table->assignments[spill_plan->assignment_index];
  const bool can_slice = assignment->unit_count > 1 &&
                         spill_plan->byte_size % assignment->unit_count == 0;
  loom_low_spill_materialization_slice_group_t* groups = builder->slice_groups;

  for (uint32_t i = 0; i < group_count; ++i) {
    loom_low_spill_materialization_slice_group_t* group = &groups[i];
    if (group->use_full_reload) {
      IREE_ASSERT_LE(builder->target_count, builder->target_capacity);
      IREE_ASSERT_LE((iree_host_size_t)group->slice_count,
                     builder->target_capacity - builder->target_count);
      group->target_start = builder->target_count;
      group->target_cursor = group->target_start;
      builder->target_count += group->slice_count;
      loom_low_spill_materialization_append_action(
          builder,
          (loom_low_spill_materialization_action_t){
              .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD,
              .spill_plan_index = spill_plan_index,
              .value.shared_slice_reload =
                  {
                      .first_slice_op = group->first_slice_op,
                      .target_start = group->target_start,
                      .target_count = group->slice_count,
                  },
          });
      ++record->reload_count;
      record->reload_bytes += spill_plan->byte_size;
    }
  }

  const uint32_t unit_byte_size =
      can_slice ? spill_plan->byte_size / assignment->unit_count : 0;
  const loom_value_id_t value_id = spill_plan->value_id;
  const loom_value_t* value =
      loom_module_value(builder->table->module, value_id);
  const loom_use_t* source_uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_low_spill_materialization_logical_use_t use;
    if (!loom_low_spill_materialization_try_source_use(
            builder, value_id, remove_argument_payload, source_uses[i], &use)) {
      continue;
    }
    loom_op_t* slice_op = NULL;
    uint16_t block_index = 0;
    uint32_t reload_byte_size = 0;
    uint32_t group_index = UINT32_MAX;
    if (can_slice && loom_low_spill_materialization_try_slice_use(
                         builder, assignment, spill_plan, source_uses[i],
                         &slice_op, &block_index, &reload_byte_size)) {
      const uint32_t group_index_plus_one =
          builder->slice_group_indices_plus_one_by_block[block_index];
      IREE_ASSERT_NE(group_index_plus_one, 0u);
      group_index = group_index_plus_one - 1;
    }
    if (group_index != UINT32_MAX && groups[group_index].use_full_reload) {
      loom_low_spill_materialization_slice_group_t* group =
          &groups[group_index];
      IREE_ASSERT_LT(group->target_cursor,
                     group->target_start + group->slice_count);
      builder->targets[group->target_cursor++] = use.target;
      continue;
    }

    if (group_index != UINT32_MAX) {
      const int64_t byte_offset =
          loom_low_slice_offset(slice_op) * (int64_t)unit_byte_size;
      const uint32_t action_index =
          loom_low_spill_materialization_append_action(
              builder,
              (loom_low_spill_materialization_action_t){
                  .kind =
                      LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD,
                  .spill_plan_index = spill_plan_index,
                  .value.narrow_slice_reload =
                      {
                          .slice_op = slice_op,
                          .byte_offset = byte_offset,
                      },
              });
      const loom_value_id_t slice_result = loom_low_slice_result(slice_op);
      IREE_ASSERT_LT(action_index,
                     LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT);
      loom_value_u32_scratch_store(
          builder->value_index, slice_result,
          LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT | action_index);
      ++record->reload_count;
      record->reload_bytes +=
          (uint64_t)unit_byte_size *
          loom_low_register_type_unit_count(
              loom_module_value_type(builder->table->module, slice_result));
      continue;
    }

    loom_low_spill_materialization_append_operand_reload(
        builder, spill_plan_index, use, record);
  }

  uint32_t generated_use_index_plus_one =
      builder->generated_use_lists[spill_plan_index].head_index_plus_one;
  while (generated_use_index_plus_one != 0) {
    const loom_low_spill_materialization_generated_use_t* generated_use =
        &builder->generated_uses[generated_use_index_plus_one - 1];
    generated_use_index_plus_one = generated_use->next_index_plus_one;
    loom_low_spill_materialization_append_action(
        builder,
        (loom_low_spill_materialization_action_t){
            .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD,
            .spill_plan_index = spill_plan_index,
            .value.operand_reload =
                {
                    .target =
                        {
                            .kind =
                                LOOM_LOW_SPILL_MATERIALIZATION_TARGET_ACTION_SOURCE,
                            .owner.action_index = generated_use->action_index,
                        },
                },
        });
    ++record->reload_count;
    record->reload_bytes += spill_plan->byte_size;
  }

  for (uint32_t i = 0; i < group_count; ++i) {
    const loom_low_spill_materialization_slice_group_t* group = &groups[i];
    if (group->use_full_reload) {
      IREE_ASSERT_EQ(group->target_cursor,
                     group->target_start + group->slice_count);
    }
    builder->slice_group_indices_plus_one_by_block[group->block_index] = 0;
  }
}

static void loom_low_spill_materialization_plan_one(
    loom_low_spill_materialization_plan_builder_t* builder,
    uint32_t spill_plan_index) {
  const loom_low_allocation_spill_plan_t* spill_plan =
      &builder->spill_plans[spill_plan_index];
  loom_low_spill_materialization_record_t* record =
      &builder->records[spill_plan_index];

  const loom_value_t* value =
      loom_module_value(builder->table->module, spill_plan->value_id);
  const bool is_block_argument = loom_value_is_block_arg(value);
  record->value_is_block_argument = is_block_argument;
  loom_block_t* argument_block =
      is_block_argument ? loom_value_def_block(value) : NULL;
  const bool remove_argument_payload =
      is_block_argument &&
      argument_block != loom_region_const_entry_block(builder->body);

  const loom_low_spill_materialization_value_ref_t current_value =
      loom_low_spill_materialization_resolve_value(
          builder,
          loom_low_spill_materialization_original_value(spill_plan->value_id));
  if (current_value.action_index != UINT32_MAX) {
    return;
  }

  uint32_t slice_group_count = 0;
  loom_low_spill_materialization_prepare_slice_groups(builder, spill_plan_index,
                                                      &slice_group_count);

  // Reserve the store before streaming reloads to preserve action order. A
  // value with no surviving use emits no reload and releases the reservation.
  const uint32_t definition_store_action_index = builder->action_count;
  if (!remove_argument_payload) {
    loom_low_spill_materialization_append_action(
        builder,
        (loom_low_spill_materialization_action_t){
            .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE,
            .spill_plan_index = spill_plan_index,
        });
  }

  loom_low_spill_materialization_append_reloads(builder, spill_plan_index,
                                                remove_argument_payload,
                                                slice_group_count, record);
  if (!remove_argument_payload) {
    if (record->reload_count == 0) {
      IREE_ASSERT_EQ(builder->action_count, definition_store_action_index + 1);
      builder->action_count = definition_store_action_index;
    } else {
      ++record->store_count;
      record->store_bytes += spill_plan->byte_size;
    }
  }

  if (remove_argument_payload) {
    const uint16_t original_argument_index = loom_value_def_index(value);
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(&builder->table->cfg_graph,
                                    loom_block_region_index(argument_block));
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      const uint16_t predecessor_index = predecessors.values[i];
      loom_block_t* predecessor_block =
          (loom_block_t*)builder->table->cfg_graph.blocks[predecessor_index]
              .block;
      const uint32_t payload_index =
          builder->branch_payload_starts[predecessor_index] +
          original_argument_index;
      IREE_ASSERT_LT(payload_index, builder->branch_payload_count);
      loom_low_spill_materialization_branch_payload_t* payload =
          &builder->branch_payloads[payload_index];
      IREE_ASSERT(payload->present);
      const loom_low_spill_materialization_value_ref_t payload_value =
          loom_low_spill_materialization_resolve_value(builder,
                                                       payload->current_value);
      if (record->reload_count != 0 &&
          (payload_value.action_index != UINT32_MAX ||
           payload_value.value_id != spill_plan->value_id)) {
        const uint32_t store_action_index =
            loom_low_spill_materialization_append_action(
                builder,
                (loom_low_spill_materialization_action_t){
                    .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE,
                    .spill_plan_index = spill_plan_index,
                    .value.edge_store =
                        {
                            .source = payload_value,
                            .predecessor_block = predecessor_block,
                        },
                });
        loom_low_spill_materialization_record_action_use(builder, payload_value,
                                                         store_action_index);
        ++record->store_count;
        record->store_bytes += spill_plan->byte_size;
      }

      loom_low_spill_materialization_append_action(
          builder,
          (loom_low_spill_materialization_action_t){
              .kind =
                  LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT,
              .spill_plan_index = spill_plan_index,
              .value.remove_branch_argument =
                  {
                      .predecessor_block = predecessor_block,
                      .block_argument_value_id = spill_plan->value_id,
                  },
          });
      payload->present = false;
    }
    IREE_ASSERT(record->reload_count == 0 || record->store_count != 0,
                "a live non-entry block argument has an incoming payload");
    loom_low_spill_materialization_append_action(
        builder,
        (loom_low_spill_materialization_action_t){
            .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT,
            .spill_plan_index = spill_plan_index,
            .value.remove_block_argument =
                {
                    .value_id = spill_plan->value_id,
                },
        });
  }
}

static void loom_low_spill_materialization_plan_all(
    loom_low_spill_materialization_plan_builder_t* builder) {
  const loom_low_placement_table_t* placement = &builder->table->placement;
  if (placement->storage_value_order_count == 0) {
    for (uint32_t i = 0; i < builder->spill_plan_count; ++i) {
      loom_low_spill_materialization_plan_one(builder, i);
    }
    return;
  }

  // Placement retains structural SSA storage topology in users-before-sources
  // order. Visit it in reverse so a source plan can replace slice results
  // before their now-stale plans observe the immutable source IR.
  const loom_liveness_analysis_t* liveness = &builder->table->liveness;
  IREE_ASSERT_EQ(placement->storage_value_order_count, liveness->value_count);
  for (loom_value_ordinal_t cursor = placement->storage_value_order_count;
       cursor > 0; --cursor) {
    const loom_value_ordinal_t value_ordinal =
        placement->storage_value_order[cursor - 1];
    IREE_ASSERT_LT(value_ordinal, liveness->value_count);
    uint32_t spill_plan_index = 0;
    if (loom_low_spill_materialization_try_spill_plan_index(
            builder, liveness->value_ids[value_ordinal], &spill_plan_index)) {
      loom_low_spill_materialization_plan_one(builder, spill_plan_index);
    }
  }
}

static uint32_t loom_low_spill_materialization_operation_index(
    const loom_low_spill_materialization_plan_builder_t* builder,
    const loom_op_t* op) {
  const uint32_t operation_index =
      loom_liveness_operation_index_for_op(&builder->table->liveness, op);
  IREE_ASSERT_NE(operation_index, UINT32_MAX);
  return operation_index;
}

static const loom_op_t* loom_low_spill_materialization_target_source_op(
    const loom_low_spill_materialization_target_t* target) {
  switch (target->kind) {
    case LOOM_LOW_SPILL_MATERIALIZATION_TARGET_OPERATION_OPERAND:
      return target->owner.operation;
    case LOOM_LOW_SPILL_MATERIALIZATION_TARGET_BRANCH_ARGUMENT:
      return loom_block_const_last_op(target->owner.predecessor_block);
    case LOOM_LOW_SPILL_MATERIALIZATION_TARGET_ACTION_SOURCE:
      return NULL;
  }
  IREE_ASSERT_UNREACHABLE("unknown spill materialization target kind");
  return NULL;
}

static bool loom_low_spill_materialization_definition_is_entry_traffic(
    const loom_low_spill_materialization_plan_builder_t* builder,
    const loom_value_t* value) {
  const loom_block_t* entry_block =
      loom_region_const_entry_block(builder->body);
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value) == entry_block;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op->parent_block == entry_block &&
         loom_low_function_entry_preamble_op(defining_op);
}

static iree_status_t loom_low_spill_materialization_project_instructions(
    loom_low_spill_materialization_plan_builder_t* builder,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialization_instruction_t** out_instructions,
    uint32_t* out_instruction_count) {
  *out_instructions = NULL;
  *out_instruction_count = 0;
  if (builder->action_count == 0) {
    return iree_ok_status();
  }

  loom_low_spill_materialization_instruction_sort_t* action_anchors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &builder->scratch_arena, builder->action_count, sizeof(*action_anchors),
      (void**)&action_anchors));
  for (uint32_t i = 0; i < builder->action_count; ++i) {
    action_anchors[i].instruction.operation_index = UINT32_MAX;
  }
  for (uint32_t i = 0; i < builder->action_count; ++i) {
    const loom_low_spill_materialization_action_t* action =
        &builder->actions[i];
    uint32_t operation_index = UINT32_MAX;
    uint32_t root_order = LOOM_LOW_SPILL_MATERIALIZATION_ORDINARY_ROOT_BIT | i;
    uint32_t local_order = UINT32_MAX;
    bool executes_after = false;
    switch (action->kind) {
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE: {
        const loom_low_allocation_spill_plan_t* spill_plan =
            &builder->spill_plans[action->spill_plan_index];
        const loom_value_t* value =
            loom_module_value(builder->table->module, spill_plan->value_id);
        if (loom_low_spill_materialization_definition_is_entry_traffic(builder,
                                                                       value)) {
          root_order = i;
          if (builder->table->entry_traffic_end != NULL) {
            operation_index = loom_low_spill_materialization_operation_index(
                builder, builder->table->entry_traffic_end);
            executes_after = true;
          } else {
            const loom_liveness_analysis_t* liveness =
                &builder->table->liveness;
            IREE_ASSERT_NE(liveness->block_count, 0u);
            IREE_ASSERT_NE(liveness->blocks[0].operation_count, 0u);
            operation_index = liveness->blocks[0].operation_start;
          }
        } else {
          IREE_ASSERT(!loom_value_is_block_arg(value));
          operation_index = loom_low_spill_materialization_operation_index(
              builder, loom_value_def_op(value));
          executes_after = true;
          // Repeated fixed-anchor insertions execute in reverse
          // transformation order.
          root_order = LOOM_LOW_SPILL_MATERIALIZATION_ORDINARY_ROOT_BIT |
                       (INT32_MAX - i);
        }
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE:
        operation_index = loom_low_spill_materialization_operation_index(
            builder, loom_block_const_last_op(
                         action->value.edge_store.predecessor_block));
        break;
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD: {
        const loom_low_spill_materialization_target_t* target =
            &action->value.operand_reload.target;
        if (target->kind ==
            LOOM_LOW_SPILL_MATERIALIZATION_TARGET_ACTION_SOURCE) {
          const uint32_t target_action_index = target->owner.action_index;
          IREE_ASSERT_LT(target_action_index, i);
          const loom_low_spill_materialization_instruction_sort_t*
              target_anchor = &action_anchors[target_action_index];
          IREE_ASSERT_NE(target_anchor->instruction.operation_index,
                         UINT32_MAX);
          operation_index = target_anchor->instruction.operation_index;
          executes_after = loom_low_spill_materialization_instruction_is_after(
              &target_anchor->instruction);
          root_order = target_anchor->root_order;
          local_order = i;
        } else {
          operation_index = loom_low_spill_materialization_operation_index(
              builder, loom_low_spill_materialization_target_source_op(target));
        }
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD:
        operation_index = loom_low_spill_materialization_operation_index(
            builder, action->value.narrow_slice_reload.slice_op);
        break;
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD: {
        const uint32_t target_start =
            action->value.shared_slice_reload.target_start;
        const uint32_t target_count =
            action->value.shared_slice_reload.target_count;
        IREE_ASSERT_NE(target_count, 0u);
        IREE_ASSERT_LE((iree_host_size_t)target_start + target_count,
                       builder->target_count);
        uint32_t earliest_point = UINT32_MAX;
        for (uint32_t j = 0; j < target_count; ++j) {
          const loom_op_t* target_op =
              loom_low_spill_materialization_target_source_op(
                  &builder->targets[target_start + j]);
          IREE_ASSERT_ARGUMENT(target_op);
          const uint32_t target_operation_index =
              loom_low_spill_materialization_operation_index(builder,
                                                             target_op);
          const loom_liveness_operation_point_t* target_operation =
              loom_liveness_operation_at(&builder->table->liveness,
                                         target_operation_index);
          if (target_operation->start_point < earliest_point) {
            earliest_point = target_operation->start_point;
            operation_index = target_operation_index;
          }
        }
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT:
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT:
        continue;
    }
    IREE_ASSERT_NE(operation_index, UINT32_MAX);
    action_anchors[i] = (loom_low_spill_materialization_instruction_sort_t){
        .instruction =
            {
                .operation_index = operation_index,
                .action_index_and_after =
                    i |
                    (executes_after
                         ? LOOM_LOW_SPILL_MATERIALIZATION_INSTRUCTION_AFTER_BIT
                         : 0),
            },
        .root_order = root_order,
        .local_order = local_order,
    };
  }

  uint32_t instruction_count = 0;
  for (uint32_t i = 0; i < builder->action_count; ++i) {
    if (action_anchors[i].instruction.operation_index != UINT32_MAX) {
      action_anchors[instruction_count++] = action_anchors[i];
    }
  }
  uint64_t expected_instruction_count = 0;
  for (uint32_t i = 0; i < builder->spill_plan_count; ++i) {
    expected_instruction_count += builder->records[i].store_count;
    expected_instruction_count += builder->records[i].reload_count;
  }
  IREE_ASSERT_EQ(instruction_count, expected_instruction_count);
  loom_low_spill_materialization_sort_instructions(
      &builder->table->liveness, action_anchors, instruction_count);

  loom_low_spill_materialization_instruction_t* instructions = NULL;
  if (instruction_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, instruction_count,
                                                   sizeof(*instructions),
                                                   (void**)&instructions));
  }
  for (uint32_t i = 0; i < instruction_count; ++i) {
    instructions[i] = action_anchors[i].instruction;
  }
  *out_instructions = instructions;
  *out_instruction_count = instruction_count;
  return iree_ok_status();
}

static iree_status_t loom_low_spill_materialization_plan_copy(
    loom_low_spill_materialization_plan_builder_t* builder,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialization_plan_t* out_plan) {
  loom_low_spill_materialization_instruction_t* instructions = NULL;
  uint32_t instruction_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_spill_materialization_project_instructions(
      builder, arena, &instructions, &instruction_count));
  loom_low_spill_materialization_action_t* actions = NULL;
  if (builder->action_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, builder->action_count, sizeof(*actions), (void**)&actions));
    memcpy(actions, builder->actions,
           (iree_host_size_t)builder->action_count * sizeof(*actions));
  }
  loom_low_spill_materialization_target_t* targets = NULL;
  if (builder->target_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, builder->target_count, sizeof(*targets), (void**)&targets));
    memcpy(targets, builder->targets,
           (iree_host_size_t)builder->target_count * sizeof(*targets));
  }
  *out_plan = (loom_low_spill_materialization_plan_t){
      .actions = actions,
      .action_count = builder->action_count,
      .instructions = instructions,
      .instruction_count = instruction_count,
      .targets = targets,
      .target_count = builder->target_count,
      .records = builder->records,
      .record_count = builder->spill_plan_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_spill_materialization_plan_storage(
    loom_low_spill_materialization_plan_builder_t* builder,
    iree_arena_allocator_t* arena) {
  iree_host_size_t included_plan_count = 0;
  for (uint32_t i = 0; i < builder->spill_plan_count; ++i) {
    included_plan_count +=
        loom_low_spill_materialization_record_has_traffic(&builder->records[i]);
  }
  if (included_plan_count < 2 || builder->table->liveness.block_count == 1) {
    return iree_ok_status();
  }

  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(builder->spill_plan_count);
  iree_bitmap_t included_plans = {
      .bit_count = builder->spill_plan_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &builder->scratch_arena, word_count, sizeof(*included_plans.words),
      (void**)&included_plans.words));
  memset(included_plans.words, 0, word_count * sizeof(*included_plans.words));
  for (uint32_t i = 0; i < builder->spill_plan_count; ++i) {
    if (loom_low_spill_materialization_record_has_traffic(
            &builder->records[i])) {
      iree_bitmap_set(included_plans, i);
    }
  }

  const uint32_t* scratch_representatives = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_storage_plan(
      builder->table, builder->spill_plans, included_plans,
      &builder->scratch_arena, &scratch_representatives));
  if (scratch_representatives != NULL) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, builder->spill_plan_count,
                                  sizeof(*builder->slot_representatives),
                                  (void**)&builder->slot_representatives));
    memcpy(builder->slot_representatives, scratch_representatives,
           (iree_host_size_t)builder->spill_plan_count *
               sizeof(*builder->slot_representatives));
  }
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE iree_status_t loom_low_spill_materialization_plan_build(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count, iree_arena_allocator_t* arena,
    loom_low_spill_materialization_plan_t* out_plan) {
  *out_plan = (loom_low_spill_materialization_plan_t){0};
  if (spill_plan_count == 0) {
    return iree_ok_status();
  }
  if (spill_plan_count >= LOOM_LOW_SPILL_MATERIALIZATION_REPLACEMENT_BIT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "spill materialization plan count overflow");
  }

  loom_low_spill_materialization_plan_builder_t builder = {
      .table = table,
      .spill_plans = spill_plans,
      .spill_plan_count = (uint32_t)spill_plan_count,
      .body = loom_low_function_const_body(table->function_op),
  };
  loom_low_spill_materialization_plan_t plan = {0};
  loom_module_value_ordinal_scratch_acquire(table->module);
  builder.value_index = &table->module->scratch.values;
  iree_arena_initialize(arena->block_pool, &builder.scratch_arena);
  iree_status_t status = iree_arena_allocate_array(arena, spill_plan_count,
                                                   sizeof(*builder.records),
                                                   (void**)&builder.records);
  if (iree_status_is_ok(status)) {
    memset(builder.records, 0, spill_plan_count * sizeof(*builder.records));
    status = loom_low_spill_materialization_initialize_branches(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_spill_materialization_initialize_values(&builder);
  }
  if (iree_status_is_ok(status) && builder.body->block_count != 0) {
    status = iree_arena_allocate_array(
        &builder.scratch_arena, builder.body->block_count,
        sizeof(*builder.slice_group_indices_plus_one_by_block),
        (void**)&builder.slice_group_indices_plus_one_by_block);
    if (iree_status_is_ok(status)) {
      memset(builder.slice_group_indices_plus_one_by_block, 0,
             builder.body->block_count *
                 sizeof(*builder.slice_group_indices_plus_one_by_block));
    }
  }
  if (iree_status_is_ok(status)) {
    loom_low_spill_materialization_plan_all(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_spill_materialization_plan_copy(&builder, arena, &plan);
  }
  for (uint32_t i = 0; i < builder.spill_plan_count; ++i) {
    loom_module_value_ordinal_scratch_clear(table->module,
                                            spill_plans[i].value_id);
  }
  for (uint32_t i = 0; i < builder.action_count; ++i) {
    const loom_low_spill_materialization_action_t* action = &builder.actions[i];
    if (action->kind ==
        LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD) {
      loom_module_value_ordinal_scratch_clear(
          table->module,
          loom_low_slice_result(action->value.narrow_slice_reload.slice_op));
    }
  }
  loom_module_value_ordinal_scratch_release(table->module);
  if (iree_status_is_ok(status)) {
    iree_arena_reset(&builder.scratch_arena);
    status = loom_low_spill_materialization_plan_storage(&builder, arena);
  }
  if (iree_status_is_ok(status)) {
    plan.slot_representatives = builder.slot_representatives;
    *out_plan = plan;
  }
  iree_arena_deinitialize(&builder.scratch_arena);
  return status;
}

static bool loom_low_allocation_spill_plan_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  return block && loom_low_br_isa(user_op) &&
         loom_low_br_dest(user_op) == block &&
         loom_use_operand_index(use) == arg_index;
}

typedef struct loom_low_allocation_spill_plan_slice_group_t {
  // Sum of narrow reload bytes for this value's slice uses in the block.
  uint64_t narrow_reload_bytes;
  // Number of slice uses; zero marks an untouched row.
  uint32_t slice_count;
  // Next touched block, or UINT16_MAX at the end of the current query's list.
  uint16_t next_block_index;
} loom_low_allocation_spill_plan_slice_group_t;

iree_status_t loom_low_allocation_spill_plan_traffic_workspace_initialize(
    const loom_region_t* body, iree_arena_allocator_t* arena,
    loom_low_allocation_spill_plan_traffic_workspace_t* out_workspace) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, body->block_count, sizeof(*out_workspace->slice_groups_by_block),
      (void**)&out_workspace->slice_groups_by_block));
  memset(out_workspace->slice_groups_by_block, 0,
         body->block_count * sizeof(*out_workspace->slice_groups_by_block));
  return iree_ok_status();
}

static void loom_low_allocation_spill_plan_value_reload_traffic(
    const loom_module_t* module,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t spill_byte_size, const loom_value_t* value,
    const loom_region_t* body,
    loom_low_allocation_spill_plan_traffic_workspace_t* workspace,
    uint32_t* out_reload_count, uint64_t* out_reload_bytes) {
  uint16_t first_touched_block_index = UINT16_MAX;

  const loom_block_t* block = NULL;
  uint16_t arg_index = 0;
  if (loom_value_is_block_arg(value)) {
    block = loom_value_def_block(value);
    arg_index = loom_value_def_index(value);
  }

  uint32_t reload_count = 0;
  uint64_t reload_bytes = 0;
  const bool can_group_slices = value->use_count > 1;
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_low_allocation_spill_plan_use_is_removed_block_arg_edge(
            uses[i], block, arg_index)) {
      continue;
    }

    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    uint32_t reload_byte_size = 0;
    const bool is_slice_reload =
        loom_low_allocation_spill_plan_slice_reload_byte_size(
            module, assignment, spill_byte_size, user_op, &reload_byte_size);

    uint16_t use_block_index = 0;
    if (!is_slice_reload || !can_group_slices ||
        !loom_region_try_block_index(body, user_op->parent_block,
                                     &use_block_index)) {
      ++reload_count;
      reload_bytes += is_slice_reload ? reload_byte_size : spill_byte_size;
      continue;
    }

    loom_low_allocation_spill_plan_slice_group_t* group =
        &workspace->slice_groups_by_block[use_block_index];
    if (group->slice_count == 0) {
      group->next_block_index = first_touched_block_index;
      first_touched_block_index = use_block_index;
    }
    ++group->slice_count;
    group->narrow_reload_bytes += reload_byte_size;
  }

  while (first_touched_block_index != UINT16_MAX) {
    loom_low_allocation_spill_plan_slice_group_t* group =
        &workspace->slice_groups_by_block[first_touched_block_index];
    if (loom_low_allocation_spill_plan_use_full_slice_reload(
            group->slice_count, group->narrow_reload_bytes, spill_byte_size)) {
      ++reload_count;
      reload_bytes += spill_byte_size;
    } else {
      reload_count += group->slice_count;
      reload_bytes += group->narrow_reload_bytes;
    }
    first_touched_block_index = group->next_block_index;
    *group = (loom_low_allocation_spill_plan_slice_group_t){0};
  }
  *out_reload_count = reload_count;
  *out_reload_bytes = reload_bytes;
}

static iree_status_t loom_low_allocation_spill_plan_value_store_count(
    const loom_cfg_graph_t* cfg_graph, loom_value_id_t value_id,
    const loom_value_t* value, uint32_t reload_count,
    uint32_t* out_store_count) {
  *out_store_count = 0;
  if (reload_count == 0) {
    return iree_ok_status();
  }
  if (!loom_value_is_block_arg(value)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  const loom_block_t* block = loom_value_def_block(value);
  const uint16_t arg_index = loom_value_def_index(value);
  if (block == loom_region_const_entry_block(cfg_graph->region)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  uint32_t store_count = 0;
  const loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(cfg_graph,
                                       loom_block_region_index(block));
  for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        loom_cfg_graph_edge(cfg_graph, predecessor_edges.values[i]);
    const loom_op_t* terminator = edge->terminator;
    const loom_value_id_t payload =
        loom_op_const_operands(terminator)[arg_index];
    if (payload == value_id) {
      continue;
    }
    ++store_count;
  }
  if (store_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "spilled non-entry block argument has reloads but no incoming value "
        "to store");
  }
  *out_store_count = store_count;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_spill_plan_traffic_for_layout(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment, uint32_t byte_size,
    loom_low_allocation_spill_plan_traffic_workspace_t* workspace,
    loom_low_allocation_spill_plan_traffic_t* out_traffic) {
  *out_traffic = (loom_low_allocation_spill_plan_traffic_t){0};
  const loom_value_id_t value_id = assignment->value_id;
  const loom_value_t* value = loom_module_value(module, value_id);
  uint32_t reload_count = 0;
  uint64_t reload_bytes = 0;
  loom_low_allocation_spill_plan_value_reload_traffic(
      module, assignment, byte_size, value, cfg_graph->region, workspace,
      &reload_count, &reload_bytes);
  uint32_t store_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_value_store_count(
      cfg_graph, value_id, value, reload_count, &store_count));
  *out_traffic = (loom_low_allocation_spill_plan_traffic_t){
      .store_count = store_count,
      .store_bytes = (uint64_t)store_count * byte_size,
      .reload_count = reload_count,
      .reload_bytes = reload_bytes,
  };
  return iree_ok_status();
}

iree_status_t loom_low_allocation_spill_plan_traffic(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits,
    loom_low_allocation_spill_plan_traffic_workspace_t* workspace,
    loom_low_allocation_spill_plan_traffic_t* out_traffic) {
  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_layout(
      assignment, alloc_unit_bits, &byte_size, &byte_alignment));
  return loom_low_allocation_spill_plan_traffic_for_layout(
      module, cfg_graph, assignment, byte_size, workspace, out_traffic);
}

iree_status_t loom_low_allocation_spill_plan_append(
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, uint16_t alloc_unit_bits,
    loom_low_spill_slot_space_t spill_slot_space,
    const loom_low_allocation_spill_plan_traffic_t* traffic,
    loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t* inout_spill_plan_count) {
  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_layout(
      assignment, alloc_unit_bits, &byte_size, &byte_alignment));

  spill_plans[(*inout_spill_plan_count)++] = (loom_low_allocation_spill_plan_t){
      .value_id = assignment->value_id,
      .assignment_index = assignment_index,
      .slot_index = assignment->location_base,
      .slot_space = spill_slot_space,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
      .store_count = traffic->store_count,
      .reload_count = traffic->reload_count,
  };
  return iree_ok_status();
}

void loom_low_allocation_spill_remark_record(
    loom_low_allocation_remark_t* remarks, iree_host_size_t* inout_remark_count,
    uint32_t assignment_index, uint32_t budget_units, uint32_t required_units) {
  remarks[(*inout_remark_count)++] = (loom_low_allocation_remark_t){
      .kind = LOOM_LOW_ALLOCATION_REMARK_SPILL,
      .assignment_index = assignment_index,
      .budget_units = budget_units,
      .required_units = required_units,
  };
}
