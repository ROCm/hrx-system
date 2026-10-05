// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/rematerialization.h"

#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/adaptive_sort.h"

void loom_low_rematerialization_invalidate_placement(
    loom_low_rematerialization_state_t* state) {
  iree_bitmap_reset_all(state->per_user_values);
}

static iree_status_t loom_low_rematerialization_reserve_values(
    iree_host_size_t required_bit_count, iree_arena_allocator_t* arena,
    iree_bitmap_t* values) {
  const iree_host_size_t required_word_count =
      iree_bitmap_calculate_words(required_bit_count);
  iree_host_size_t word_capacity =
      iree_bitmap_calculate_words(values->bit_count);
  if (required_word_count > word_capacity) {
    const iree_host_size_t old_word_count = word_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, old_word_count, required_word_count, sizeof(*values->words),
        &word_capacity, (void**)&values->words));
    memset(values->words + old_word_count, 0,
           (word_capacity - old_word_count) * sizeof(*values->words));
    values->bit_count = word_capacity * IREE_BITMAP_BITS_PER_WORD;
  }
  return iree_ok_status();
}

static loom_low_allocation_rematerialization_result_t
loom_low_allocation_rematerialization_result_empty(void) {
  return (loom_low_allocation_rematerialization_result_t){
      .value.value_id = LOOM_VALUE_ID_INVALID,
  };
}

static loom_low_value_rematerialization_result_t
loom_low_value_rematerialization_result_empty(void) {
  return (loom_low_value_rematerialization_result_t){
      .value_id = LOOM_VALUE_ID_INVALID,
  };
}

static bool loom_low_allocation_failure_is_rematerializable_pressure(
    const loom_low_allocation_failure_t* failure) {
  return loom_low_allocation_failure_is_present(failure) &&
         iree_string_view_equal(failure->failure_code,
                                IREE_SV("unspillable-register-exhausted"));
}

static bool loom_low_descriptor_packet_kind_may_rematerialize(
    loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ||
         kind == LOOM_LOW_DESCRIPTOR_PACKET_CONST;
}

typedef struct loom_low_rematerialization_recipe_t {
  // Descriptor packets from the selected value toward its ownership root.
  loom_op_t** packets;
  // Number of packets in the ownership chain.
  iree_host_size_t packet_count;
} loom_low_rematerialization_recipe_t;

typedef enum loom_low_rematerialization_action_e {
  LOOM_LOW_REMATERIALIZATION_ACTION_NONE = 0,
  LOOM_LOW_REMATERIALIZATION_ACTION_RETAIN_PLACEMENT,
  LOOM_LOW_REMATERIALIZATION_ACTION_CLONE,
} loom_low_rematerialization_action_t;

// One prepared value repair. Its recipe and captured uses remain valid until
// application rewrites IR; placement retention leaves the snapshot unchanged.
typedef struct loom_low_rematerialization_plan_t {
  // Descriptor-backed ownership chain to recreate for each user.
  loom_low_rematerialization_recipe_t recipe;
  // Existing uses captured once before deciding whether mutation is required.
  const loom_use_t* uses;
  // Number of captured operand uses.
  uint32_t use_count;
  // Repair selected by recipe eligibility and source/scheduled placement.
  loom_low_rematerialization_action_t action;
} loom_low_rematerialization_plan_t;

static bool loom_low_rematerialization_packet_is_eligible(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, loom_op_t** out_defining_op) {
  *out_defining_op = NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, value_id)) {
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_low_register_type_resolver_t register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          target->descriptor_set);
  if (loom_low_register_type_resolver_has_class_flags(
          &register_type_resolver, value_type,
          LOOM_LOW_REG_CLASS_FLAG_REFERENCE)) {
    return false;
  }

  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->result_count != 1 || defining_op->region_count != 0 ||
      defining_op->successor_count != 0 ||
      iree_any_bit_set(loom_op_effective_traits(module, defining_op),
                       LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }

  const uint16_t result_index = loom_value_def_index(value);
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(target->descriptor_set, defining_op,
                                        &packet);
  if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
      !loom_low_descriptor_result_can_rematerialize(
          target->descriptor_set, packet.descriptor, result_index)) {
    return false;
  }
  *out_defining_op = defining_op;
  return true;
}

static loom_value_id_t loom_low_rematerialization_tied_source(
    const loom_op_t* op) {
  const loom_tied_result_t tied = loom_op_tied_results(op)[0];
  return loom_op_const_operands(op)[tied.operand_index];
}

static bool loom_low_rematerialization_inputs_remain_available(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t tied_source_value_id) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] != tied_source_value_id &&
        loom_consumption_find_consuming_use(
            module, loom_module_value(module, operands[i]), NULL)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_rematerialization_plan_recipe(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, iree_arena_allocator_t* arena,
    loom_low_rematerialization_recipe_t* out_recipe) {
  *out_recipe = (loom_low_rematerialization_recipe_t){0};
  loom_op_t** packets = NULL;
  iree_host_size_t packet_count = 0;
  iree_host_size_t packet_capacity = 0;
  loom_value_id_t current_value_id = value_id;
  for (;;) {
    loom_op_t* defining_op = NULL;
    if (!loom_low_rematerialization_packet_is_eligible(
            module, target, current_value_id, &defining_op)) {
      return iree_ok_status();
    }
    if (packet_count == packet_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, packet_count, packet_count + 1, sizeof(*packets),
          &packet_capacity, (void**)&packets));
    }
    packets[packet_count++] = defining_op;
    loom_value_id_t tied_source_value_id = LOOM_VALUE_ID_INVALID;
    if (defining_op->tied_result_count != 0) {
      tied_source_value_id =
          loom_low_rematerialization_tied_source(defining_op);
    }
    if (!loom_low_rematerialization_inputs_remain_available(
            module, defining_op, tied_source_value_id)) {
      return iree_ok_status();
    }
    if (tied_source_value_id == LOOM_VALUE_ID_INVALID) {
      break;
    }
    current_value_id = tied_source_value_id;
  }
  *out_recipe = (loom_low_rematerialization_recipe_t){
      .packets = packets,
      .packet_count = packet_count,
  };
  return iree_ok_status();
}

static bool loom_low_rematerialization_use_is_eligible(
    loom_value_id_t value_id, const loom_op_t* defining_op, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  if (user_op == NULL || user_op == defining_op ||
      iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) ||
      user_op->parent_block == NULL ||
      operand_index >= user_op->operand_count) {
    return false;
  }
  if (loom_op_operands(user_op)[operand_index] != value_id) {
    return false;
  }
  if (user_op->parent_block == defining_op->parent_block &&
      user_op->block_ordinal <= defining_op->block_ordinal) {
    return false;
  }
  return true;
}

static bool loom_low_rematerialization_use_shortens_live_range(
    const loom_op_t* defining_op, loom_use_t use) {
  const loom_op_t* user_op = loom_use_user_op(use);
  // An already-adjacent producer needs placement retention, not another clone.
  return user_op != NULL &&
         (user_op->parent_block != defining_op->parent_block ||
          user_op->prev_op != defining_op);
}

static iree_status_t loom_low_rematerialization_clone_for_use(
    loom_rewriter_t* rewriter,
    const loom_low_rematerialization_recipe_t* recipe, loom_use_t use,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_value_id_t* out_cloned_value_id) {
  *out_cloned_value_id = LOOM_VALUE_ID_INVALID;

  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(rewriter->module, rewriter->module, arena,
                               &(loom_ir_remap_options_t){
                                   .allow_unmapped_values = true,
                               },
                               &remap));

  loom_op_t* user_op = loom_use_user_op(use);
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, user_op);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = recipe->packet_count;
       i > 0 && iree_status_is_ok(status); --i) {
    loom_op_t* source_op = recipe->packets[i - 1];
    loom_op_t* cloned_op = NULL;
    status =
        loom_ir_clone_op(&rewriter->builder, source_op, &remap, &cloned_op);
    if (!iree_status_is_ok(status)) {
      break;
    }

    const loom_value_id_t source_value_id = loom_op_const_results(source_op)[0];
    const loom_value_id_t cloned_value_id = loom_op_results(cloned_op)[0];
    status = loom_rewriter_clear_value_name(rewriter, cloned_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_try_set_derived_value_name(
          rewriter, source_value_id, cloned_value_id, IREE_SV("remat"));
    }
    if (iree_status_is_ok(status)) {
      iree_bitmap_set(state->per_user_values, cloned_value_id);
      if (state->required_register_values != NULL &&
          source_value_id < state->required_register_values->bit_count &&
          iree_bitmap_test(*state->required_register_values, source_value_id)) {
        iree_bitmap_set(*state->required_register_values, cloned_value_id);
      }
      *out_cloned_value_id = cloned_value_id;
    }
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  return iree_ok_status();
}

static iree_status_t loom_low_rematerialization_prepare_value(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, const loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_rematerialization_plan_t* out_plan) {
  *out_plan = (loom_low_rematerialization_plan_t){0};
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (value_id < state->per_user_values.bit_count &&
      iree_bitmap_test(state->per_user_values, value_id)) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) ||
      loom_consumption_find_consuming_use(module, value, NULL) ||
      loom_value_has_attribute_uses(value) || value->use_count == 0 ||
      loom_module_value_has_type_uses(module, value_id)) {
    return iree_ok_status();
  }

  loom_low_rematerialization_recipe_t recipe = {0};
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_plan_recipe(
      module, target, value_id, arena, &recipe));
  if (recipe.packet_count == 0) {
    return iree_ok_status();
  }
  loom_op_t* defining_op = recipe.packets[0];

  const uint32_t use_count = value->use_count;
  loom_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, use_count,
                                                 sizeof(*uses), (void**)&uses));
  memcpy(uses, loom_value_uses(value), use_count * sizeof(*uses));
  bool shortens_live_range = false;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (!loom_low_rematerialization_use_is_eligible(value_id, defining_op,
                                                    uses[i])) {
      return iree_ok_status();
    }
    shortens_live_range = shortens_live_range ||
                          loom_low_rematerialization_use_shortens_live_range(
                              defining_op, uses[i]);
  }
  loom_low_rematerialization_action_t action =
      LOOM_LOW_REMATERIALIZATION_ACTION_CLONE;
  if (!shortens_live_range) {
    // All eligible uses name the immediately following operation. Source
    // adjacency does not bound the lifetime in a reordered allocation: retain
    // this private producer exactly as a per-user clone when it was separated
    // from its consumer. Input-free materializations remain pressure-scheduled.
    if (schedule == NULL || defining_op->operand_count == 0) {
      return iree_ok_status();
    }
    const loom_low_schedule_node_t* producer =
        loom_low_schedule_node_for_op(schedule, defining_op);
    // Allocation includes nested regions; scheduling covers direct body ops.
    if (producer == NULL) {
      return iree_ok_status();
    }
    const loom_low_schedule_node_t* consumer =
        loom_low_schedule_node_for_op(schedule, loom_use_user_op(uses[0]));
    if (consumer->scheduled_ordinal <= producer->scheduled_ordinal + 1u) {
      return iree_ok_status();
    }
    action = LOOM_LOW_REMATERIALIZATION_ACTION_RETAIN_PLACEMENT;
  }
  *out_plan = (loom_low_rematerialization_plan_t){
      .recipe = recipe,
      .uses = uses,
      .use_count = use_count,
      .action = action,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_rematerialization_apply_value(
    loom_module_t* module, loom_value_id_t value_id,
    const loom_low_rematerialization_plan_t* plan,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result) {
  if (plan->action == LOOM_LOW_REMATERIALIZATION_ACTION_NONE) {
    return iree_ok_status();
  }
  if (plan->action == LOOM_LOW_REMATERIALIZATION_ACTION_RETAIN_PLACEMENT) {
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_values(
        module->values.count, state->arena, &state->per_user_values));
    iree_bitmap_set(state->per_user_values, value_id);
    *out_result = (loom_low_value_rematerialization_result_t){
        .value_id = value_id,
        .retained_placement_count = 1,
    };
    return iree_ok_status();
  }

  const loom_low_rematerialization_recipe_t* recipe = &plan->recipe;
  loom_op_t* defining_op = recipe->packets[0];
  const loom_use_t* uses = plan->uses;
  const uint32_t use_count = plan->use_count;

  // Verified SSA makes every packet input and external type/attribute capture
  // available at its definition, which dominates each existing operand use.
  // Recipe planning rejects consumed external inputs, so cloning cannot cross
  // an ownership transfer.
  // Clone tied ownership chains from their root so each tied packet consumes a
  // private predecessor instead of consuming one source several times. Each
  // eligible packet has one result and no regions. Reserve membership for the
  // worst-case per-use clones before mutation changes the module value count.
  if (recipe->packet_count > UINT32_MAX / use_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "rematerialization clone count exceeds uint32 capacity");
  }
  const uint32_t maximum_clone_count =
      (uint32_t)(recipe->packet_count * use_count);
  if (maximum_clone_count > LOOM_VALUE_ID_INVALID - module->values.count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "rematerialization value count exceeds value ID capacity");
  }
  const iree_host_size_t required_value_count =
      module->values.count + maximum_clone_count;
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_values(
      required_value_count, state->arena, &state->per_user_values));
  if (state->required_register_values != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_values(
        required_value_count, state->arena, state->required_register_values));
  }
  // Index users by their first rewritten captured use. That exact operand
  // retains the clone ID for other occurrences in the same instruction.
  const iree_host_size_t user_capacity =
      iree_host_size_next_power_of_two((iree_host_size_t)use_count * 2);
  const iree_host_size_t user_mask = user_capacity - 1;
  uint32_t inline_user_uses[16];
  uint32_t* user_uses = inline_user_uses;
  if (user_capacity > IREE_ARRAYSIZE(inline_user_uses)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, user_capacity, sizeof(*user_uses), (void**)&user_uses));
  }
  memset(user_uses, 0, user_capacity * sizeof(*user_uses));
  loom_rewriter_t rewriter = {0};
  loom_low_value_rematerialization_result_t result = {
      .value_id = value_id,
  };
  loom_rewriter_initialize(&rewriter, module, arena);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < use_count && iree_status_is_ok(status); ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    const uint32_t user_hash =
        loom_structural_hash_finalize(loom_structural_hash_mix_u64(
            loom_structural_hash_initialize(), (uint64_t)(uintptr_t)user_op));
    iree_host_size_t user_slot = user_hash & user_mask;
    while (user_uses[user_slot] != 0 &&
           loom_use_user_op(uses[user_uses[user_slot] - 1]) != user_op) {
      user_slot = (user_slot + 1) & user_mask;
    }
    const bool first_user_use = user_uses[user_slot] == 0;
    loom_value_id_t cloned_value_id = LOOM_VALUE_ID_INVALID;
    if (first_user_use) {
      status = loom_low_rematerialization_clone_for_use(
          &rewriter, recipe, uses[i], state, arena, &cloned_value_id);
    } else {
      const loom_use_t first_use = uses[user_uses[user_slot] - 1];
      cloned_value_id =
          loom_op_operands(user_op)[loom_use_operand_index(first_use)];
    }
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operand(
          &rewriter, user_op, loom_use_operand_index(uses[i]), cloned_value_id);
    }
    if (iree_status_is_ok(status)) {
      if (first_user_use) {
        user_uses[user_slot] = i + 1;
        result.cloned_packet_count += (uint32_t)recipe->packet_count;
      }
      ++result.rewritten_operand_count;
    }
  }
  if (iree_status_is_ok(status)) {
    // Cloning the chain may give external inputs additional users in other
    // blocks. Those inputs no longer retain a per-user placement guarantee.
    for (iree_host_size_t packet_index = 0; packet_index < recipe->packet_count;
         ++packet_index) {
      const loom_op_t* packet = recipe->packets[packet_index];
      const loom_value_id_t* operands = loom_op_const_operands(packet);
      for (uint16_t operand_index = 0; operand_index < packet->operand_count;
           ++operand_index) {
        const loom_value_id_t operand = operands[operand_index];
        bool is_recipe_result = false;
        for (iree_host_size_t i = 0; i < recipe->packet_count; ++i) {
          is_recipe_result |=
              operand == loom_op_const_results(recipe->packets[i])[0];
        }
        if (!is_recipe_result && operand < state->per_user_values.bit_count) {
          iree_bitmap_reset(state->per_user_values, operand);
        }
      }
    }
    // Erase the selected packet and every ownership predecessor made dead by
    // its removal. Stop at the first predecessor retained by another use.
    IREE_ASSERT(loom_op_results_unused(module, defining_op));
    for (iree_host_size_t i = 0;
         i < recipe->packet_count && iree_status_is_ok(status); ++i) {
      loom_op_t* packet = recipe->packets[i];
      if (!loom_op_results_unused(module, packet)) {
        break;
      }
      status = loom_rewriter_erase(&rewriter, packet);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_result = result;
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_rematerialize_value_uses(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, const loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result) {
  *out_result = loom_low_value_rematerialization_result_empty();
  loom_low_rematerialization_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_prepare_value(
      module, target, value_id, schedule, state, arena, &plan));
  return loom_low_rematerialization_apply_value(module, value_id, &plan, state,
                                                arena, out_result);
}

// Returns true when any storage unit in |assignment| is live at |point|.
//
// Whole-assignment bounds reject most candidates cheaply. Sparse storage
// segments exclude values that only appear live because mutually exclusive CFG
// paths share the linear program-point space, and refined per-unit bounds
// retain the allocation model's subrange precision.
static bool loom_low_allocation_assignment_is_live_at_point(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment, uint32_t point) {
  if (point < assignment->start_point || point >= assignment->end_point) {
    return false;
  }
  if (assignment->liveness_segments.count != 0 &&
      !loom_liveness_segment_range_contains(
          table->storage_segments, assignment->liveness_segments, point)) {
    return false;
  }
  for (uint32_t i = 0; i < assignment->unit_count; ++i) {
    const uint32_t start_point =
        loom_low_allocation_live_range_assignment_unit_start_point(
            table->unit_start_points, table->unit_point_count, assignment, i);
    const uint32_t end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            table->unit_end_points, table->unit_point_count, assignment, i);
    if (point >= start_point && point < end_point) {
      return true;
    }
  }
  return false;
}

typedef enum loom_low_allocation_rematerialization_frontier_e {
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_PRESSURE_CLASS = 0,
  LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_OVERLAPPING_STORAGE = 1,
} loom_low_allocation_rematerialization_frontier_t;

static bool loom_low_allocation_assignment_overlaps_failure_storage(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_low_allocation_failure_t* failure = &table->failure;
  const loom_low_descriptor_set_t* descriptor_set =
      table->target.descriptor_set;
  if (failure->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      failure->descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }

  const loom_low_reg_class_t* failed_reg_class =
      &descriptor_set->reg_classes[failure->descriptor_reg_class_id];
  loom_low_allocation_assignment_t candidate = {
      .value_id = failure->value_id,
      .descriptor_reg_class_id = failure->descriptor_reg_class_id,
      .unit_count = failure->required_unit_count,
      .location_kind = failure->location_kind,
      .location_count = failure->required_unit_count,
  };
  if (!loom_low_reg_class_uses_explicit_physical_registers(failed_reg_class)) {
    if (failure->location_base == UINT32_MAX || failure->location_count == 0) {
      return false;
    }
    candidate.location_base = failure->location_base;
    candidate.location_count = failure->location_count;
    return loom_low_allocation_storage_assignment_ranges_overlap(
        descriptor_set, &candidate, assignment);
  }

  for (uint32_t physical_register_id = 0;
       physical_register_id < descriptor_set->physical_register_count;
       ++physical_register_id) {
    if (!loom_low_allocation_storage_explicit_physical_register_view(
            descriptor_set, failure->descriptor_reg_class_id,
            physical_register_id, failure->required_unit_count,
            /*out_first_candidate_ordinal=*/NULL,
            /*out_pressure_extent=*/NULL)) {
      continue;
    }
    candidate.location_base = physical_register_id;
    if (loom_low_allocation_storage_assignment_ranges_overlap(
            descriptor_set, &candidate, assignment)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_rematerialization_frontier_contains(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    loom_low_allocation_rematerialization_frontier_t frontier) {
  const bool has_failed_class = assignment->descriptor_reg_class_id ==
                                table->failure.descriptor_reg_class_id;
  if (frontier ==
      LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_PRESSURE_CLASS) {
    return has_failed_class;
  }
  return !has_failed_class &&
         loom_low_allocation_assignment_overlaps_failure_storage(table,
                                                                 assignment);
}

// Attempts live values in one failed pressure frontier. Allocation reports one
// concrete collision, but that assignment is not necessarily the live value
// dominating the failure frontier. Candidates are ordered by remaining live
// unit-points so a successful rewrite removes the largest conservative
// pressure area first. Placement-only repairs preserve this snapshot and can
// accumulate before rebuilding; an IR rewrite ends its use immediately.
static iree_status_t loom_low_allocation_try_rematerialize_live_frontier(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_schedule_table_t* schedule,
    loom_low_allocation_rematerialization_frontier_t frontier,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_batch,
    loom_low_allocation_rematerialization_result_t* out_result) {
  const loom_low_allocation_failure_t* failure = &table->failure;
  uint64_t previous_pressure_area = UINT64_MAX;
  iree_host_size_t previous_assignment_index = IREE_HOST_SIZE_MAX;
  for (;;) {
    uint64_t best_pressure_area = 0;
    iree_host_size_t best_assignment_index = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
      const loom_low_allocation_assignment_t* assignment =
          &table->assignments[i];
      if (!loom_low_allocation_rematerialization_frontier_contains(
              table, assignment, frontier) ||
          !loom_low_allocation_assignment_is_live_at_point(
              table, assignment, failure->start_point)) {
        continue;
      }
      const uint64_t remaining_points =
          assignment->end_point - failure->start_point;
      uint64_t pressure_area = 0;
      if (!iree_checked_mul_u64(remaining_points, assignment->unit_count,
                                &pressure_area)) {
        pressure_area = UINT64_MAX;
      }
      if (previous_assignment_index != IREE_HOST_SIZE_MAX &&
          (pressure_area > previous_pressure_area ||
           (pressure_area == previous_pressure_area &&
            i <= previous_assignment_index))) {
        continue;
      }
      if (best_assignment_index == IREE_HOST_SIZE_MAX ||
          pressure_area > best_pressure_area ||
          (pressure_area == best_pressure_area && i < best_assignment_index)) {
        best_pressure_area = pressure_area;
        best_assignment_index = i;
      }
    }
    if (best_assignment_index == IREE_HOST_SIZE_MAX) {
      return iree_ok_status();
    }

    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[best_assignment_index];
    out_result->value = loom_low_value_rematerialization_result_empty();
    loom_low_rematerialization_plan_t plan = {0};
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_prepare_value(
        module, &table->target, assignment->value_id, schedule, state, arena,
        &plan));
    // Retained placement already warrants a fresh schedule. Try that order
    // before paying for a clone that its allocation may no longer require.
    if (plan.action == LOOM_LOW_REMATERIALIZATION_ACTION_CLONE &&
        out_batch->retained_placement_count != 0) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_apply_value(
        module, assignment->value_id, &plan, state, arena, &out_result->value));
    out_batch->cloned_packet_count += out_result->value.cloned_packet_count;
    out_batch->rewritten_operand_count +=
        out_result->value.rewritten_operand_count;
    out_batch->retained_placement_count +=
        out_result->value.retained_placement_count;
    if (out_result->value.rewritten_operand_count != 0) {
      out_result->descriptor_reg_class_id = assignment->descriptor_reg_class_id;
      return iree_ok_status();
    }
    previous_pressure_area = best_pressure_area;
    previous_assignment_index = best_assignment_index;
  }
}

static iree_status_t loom_low_allocation_rematerialize_failure_value(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_batch,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  const loom_low_allocation_failure_t* failure = &table->failure;
  IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_live_frontier(
      module, table, schedule,
      LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_PRESSURE_CLASS, state,
      arena, out_batch, out_result));
  if (out_batch->rewritten_operand_count != 0 ||
      out_batch->retained_placement_count != 0) {
    return iree_ok_status();
  }

  // Explicit physical classes can overlap several other register classes and
  // locations. Exhaust the whole allocatable storage domain instead of only
  // the first concrete collision selected for diagnostics.
  IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_live_frontier(
      module, table, schedule,
      LOOM_LOW_ALLOCATION_REMATERIALIZATION_FRONTIER_OVERLAPPING_STORAGE, state,
      arena, out_batch, out_result));
  if (out_batch->rewritten_operand_count != 0 ||
      out_batch->retained_placement_count != 0) {
    return iree_ok_status();
  }

  // The failed value may not have an assignment in the partial table and is
  // therefore the only pressure candidate not covered by the live frontier.
  IREE_RETURN_IF_ERROR(loom_low_rematerialize_value_uses(
      module, &table->target, failure->value_id, schedule, state, arena,
      &out_result->value));
  out_batch->cloned_packet_count += out_result->value.cloned_packet_count;
  out_batch->rewritten_operand_count +=
      out_result->value.rewritten_operand_count;
  out_batch->retained_placement_count +=
      out_result->value.retained_placement_count;
  if (out_result->value.rewritten_operand_count != 0 ||
      out_result->value.retained_placement_count != 0) {
    out_result->descriptor_reg_class_id = failure->descriptor_reg_class_id;
    return iree_ok_status();
  }
  return iree_ok_status();
}

typedef struct loom_low_rematerialization_candidate_t {
  // Original value and pressure class, retained by the allocation snapshot.
  const loom_liveness_interval_t* interval;
  // Producer position within its block before the first rewrite.
  uint64_t block_ordinal;
  // Retained CFG DFS preorder; strict dominators have smaller preorders.
  uint16_t block_preorder;
} loom_low_rematerialization_candidate_t;

static bool loom_low_rematerialization_candidate_less(
    const loom_low_rematerialization_candidate_t* lhs,
    const loom_low_rematerialization_candidate_t* rhs) {
  if (lhs->block_preorder != rhs->block_preorder) {
    return lhs->block_preorder > rhs->block_preorder;
  }
  return lhs->block_ordinal > rhs->block_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_rematerialization_candidate_sort,
                          loom_low_rematerialization_candidate_t,
                          loom_low_rematerialization_candidate_less)

static iree_status_t loom_low_rematerialization_plan_cross_block_batch(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_rematerialization_candidate_t** out_candidates,
    iree_host_size_t* out_count) {
  *out_candidates = NULL;
  *out_count = 0;
  const loom_liveness_analysis_t* liveness = &table->liveness;
  if (liveness->block_count < 2) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptors = table->target.descriptor_set;
  uint8_t* pressure_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptors->reg_class_count, sizeof(*pressure_classes),
      (void**)&pressure_classes));
  memset(pressure_classes, 0, descriptors->reg_class_count);
  uint32_t pressure_class_count = 0;
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    if (summary->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    const uint16_t class_id = summary->value_class.register_class_id;
    const loom_low_reg_class_t* reg_class = &descriptors->reg_classes[class_id];
    const uint32_t capacity =
        loom_liveness_value_class_equal(summary->value_class,
                                        table->failure.value_class)
            ? table->failure.budget_units
            : reg_class->allocatable_count;
    if (iree_any_bit_set(reg_class->flags,
                         LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE) &&
        capacity != 0 && summary->peak_live_units > capacity) {
      pressure_classes[class_id] = 1;
      ++pressure_class_count;
    }
  }
  if (pressure_class_count == 0) {
    return iree_ok_status();
  }

  loom_low_rematerialization_candidate_t* candidates = NULL;
  iree_host_size_t count = 0;
  iree_host_size_t capacity = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    // Segments are block-local. This retained fact replaces inspecting every
    // use again to discover whether the value crosses a CFG boundary.
    if (liveness->value_segment_ranges[ordinal].count < 2) {
      continue;
    }
    const loom_liveness_interval_t* interval =
        &liveness->intervals[liveness->value_interval_indices[ordinal]];
    if (interval->value_class.type_kind != LOOM_TYPE_REGISTER ||
        !pressure_classes[interval->value_class.register_class_id]) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, interval->value_id);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    const loom_op_t* op = loom_value_def_op(value);
    if (op->parent_block->parent_region != liveness->region ||
        op->result_count != 1) {
      continue;
    }
    const loom_cfg_block_info_t* block =
        &table->cfg_graph.blocks[op->parent_block->region_index];
    if (!block->reachable) {
      continue;
    }
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(descriptors, op, &packet);
    if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
        !loom_low_descriptor_result_can_rematerialize(descriptors,
                                                      packet.descriptor, 0)) {
      continue;
    }
    if (count == capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, count, count + 1,
                                                 sizeof(*candidates), &capacity,
                                                 (void**)&candidates));
    }
    candidates[count++] = (loom_low_rematerialization_candidate_t){
        .interval = interval,
        .block_ordinal = op->block_ordinal,
        .block_preorder = block->preorder,
    };
  }
  loom_low_rematerialization_candidate_sort(candidates, count);
  *out_candidates = candidates;
  *out_count = count;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result) {
  *out_result = (loom_low_rematerialization_batch_result_t){0};
  if (!loom_low_allocation_failure_is_rematerializable_pressure(
          &table->failure)) {
    return iree_ok_status();
  }
  loom_low_rematerialization_candidate_t* candidates = NULL;
  iree_host_size_t candidate_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_plan_cross_block_batch(
      module, table, arena, &candidates, &candidate_count));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < candidate_count && iree_status_is_ok(status);
       ++i) {
    const loom_low_rematerialization_candidate_t* candidate = &candidates[i];
    loom_low_allocation_rematerialization_result_t result = {
        .descriptor_reg_class_id =
            candidate->interval->value_class.register_class_id,
    };
    status = loom_low_rematerialize_value_uses(
        module, &table->target, candidate->interval->value_id,
        /*schedule=*/NULL, state, arena, &result.value);
    if (iree_status_is_ok(status)) {
      out_result->cloned_packet_count += result.value.cloned_packet_count;
      out_result->rewritten_operand_count +=
          result.value.rewritten_operand_count;
      status = loom_low_allocation_rematerialization_emit_decision(
          table,
          LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE,
          &result, emitter);
    }
  }
  if (iree_status_is_ok(status) && out_result->rewritten_operand_count == 0) {
    loom_low_allocation_rematerialization_result_t result = {0};
    status = loom_low_allocation_rematerialize_failure_value(
        module, table, schedule, state, arena, out_result, &result);
    if (iree_status_is_ok(status)) {
      status = loom_low_allocation_rematerialization_emit_decision(
          table,
          LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE,
          &result, emitter);
    }
  }
  return status;
}

iree_status_t loom_low_allocation_rematerialize_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = loom_low_allocation_rematerialization_result_empty();
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    IREE_RETURN_IF_ERROR(loom_low_rematerialize_value_uses(
        module, &table->target, spill_plan->value_id, /*schedule=*/NULL, state,
        arena, &out_result->value));
    if (out_result->value.rewritten_operand_count != 0) {
      out_result->descriptor_reg_class_id =
          table->assignments[spill_plan->assignment_index]
              .descriptor_reg_class_id;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_rematerialization_trigger_name(
    loom_low_allocation_rematerialization_trigger_t trigger) {
  switch (trigger) {
    case LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE:
      return IREE_SV("allocation-failure");
    case LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_SPILL_PLAN:
      return IREE_SV("spill-plan");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_low_allocation_rematerialization_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_rematerialization_trigger_t trigger,
    const loom_low_allocation_rematerialization_result_t* result,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(result);
  if (emitter.fn == NULL || result->value.rewritten_operand_count == 0) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                       result->value.value_id)),
      loom_param_string(loom_low_diagnostic_reg_class_name(
          table->target.descriptor_set, result->descriptor_reg_class_id)),
      loom_param_string(
          loom_low_allocation_rematerialization_trigger_name(trigger)),
      loom_param_u32(result->value.cloned_packet_count),
      loom_param_u32(result->value.rewritten_operand_count),
      loom_param_string(IREE_SV("descriptor-rematerializable")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = table->function_op,
      .error = LOOM_ERR_BACKEND_045,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
