// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_rematerialization.h"

#include <string.h>

#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

static bool loom_low_allocation_failure_is_rematerializable_pressure(
    const loom_low_allocation_failure_t* failure) {
  return loom_low_allocation_failure_is_present(failure) &&
         iree_string_view_equal(failure->failure_code,
                                IREE_SV("unspillable-register-exhausted"));
}

static bool loom_low_allocation_value_has_type_uses(const loom_module_t* module,
                                                    loom_value_id_t value_id) {
  if (value_id >= module->values.count ||
      value_id >= module->type_uses.value_capacity) {
    return false;
  }
  return module->type_uses.value_heads[value_id].first_incoming_use_id !=
         LOOM_TYPE_USE_ID_INVALID;
}

static bool loom_low_descriptor_packet_kind_may_rematerialize(
    loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ||
         kind == LOOM_LOW_DESCRIPTOR_PACKET_CONST;
}

static bool loom_low_allocation_rematerialization_use_is_eligible(
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

static iree_status_t loom_low_allocation_rematerialization_clone_for_use(
    loom_rewriter_t* rewriter, const loom_op_t* defining_op,
    uint16_t result_index, loom_value_id_t source_value_id, loom_use_t use,
    iree_arena_allocator_t* arena, loom_value_id_t* out_cloned_value_id) {
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
  loom_op_t* cloned_op = NULL;
  iree_status_t status =
      loom_ir_clone_op(&rewriter->builder, defining_op, &remap, &cloned_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  if (result_index >= cloned_op->result_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "rematerialized packet result index is outside cloned packet results");
  }
  const loom_value_id_t cloned_value_id =
      loom_op_results(cloned_op)[result_index];
  IREE_RETURN_IF_ERROR(
      loom_rewriter_clear_value_name(rewriter, cloned_value_id));
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      rewriter, source_value_id, cloned_value_id, IREE_SV("remat")));
  *out_cloned_value_id = cloned_value_id;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_try_rematerialize_value(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* result) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_is_consumed(value) ||
      loom_module_value_has_predicate_attribute_uses(module, value_id) ||
      value->use_count == 0 ||
      loom_low_allocation_value_has_type_uses(module, value_id)) {
    return iree_ok_status();
  }

  const uint16_t result_index = loom_value_def_index(value);
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->result_count != 1 || defining_op->region_count != 0 ||
      defining_op->successor_count != 0 ||
      defining_op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_descriptor_packet(module, target, defining_op, &packet));
  if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
      !loom_low_descriptor_result_can_rematerialize(
          target->descriptor_set, packet.descriptor, result_index)) {
    return iree_ok_status();
  }

  const uint32_t use_count = value->use_count;
  loom_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, use_count,
                                                 sizeof(*uses), (void**)&uses));
  memcpy(uses, loom_value_uses(value), use_count * sizeof(*uses));
  for (uint32_t i = 0; i < use_count; ++i) {
    if (!loom_low_allocation_rematerialization_use_is_eligible(
            value_id, defining_op, uses[i])) {
      return iree_ok_status();
    }
  }

  loom_rewriter_t rewriter = {0};
  iree_status_t status = loom_rewriter_initialize(&rewriter, module, arena);
  if (!iree_status_is_ok(status)) return status;
  for (uint32_t i = 0; i < use_count && iree_status_is_ok(status); ++i) {
    loom_value_id_t cloned_value_id = LOOM_VALUE_ID_INVALID;
    status = loom_low_allocation_rematerialization_clone_for_use(
        &rewriter, defining_op, result_index, value_id, uses[i], arena,
        &cloned_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operand(&rewriter, loom_use_user_op(uses[i]),
                                         loom_use_operand_index(uses[i]),
                                         cloned_value_id);
    }
    if (iree_status_is_ok(status)) {
      ++result->cloned_packet_count;
      ++result->rewritten_operand_count;
    }
  }
  bool erased = false;
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase_if_dead(&rewriter, defining_op, &erased);
  }
  if (iree_status_is_ok(status) && !erased) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "rematerialized packet producer remained live after rewriting all "
        "operand uses");
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = (loom_low_allocation_rematerialization_result_t){0};
  if (!loom_low_allocation_failure_is_rematerializable_pressure(
          &table->failure)) {
    return iree_ok_status();
  }

  const loom_low_allocation_failure_t* failure = &table->failure;
  if (failure->blocking_kind ==
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT &&
      failure->conflict_value_id != failure->value_id) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, failure->conflict_value_id, arena, out_result));
    if (out_result->rewritten_operand_count != 0) return iree_ok_status();
  }

  return loom_low_allocation_try_rematerialize_value(
      module, &table->target, failure->value_id, arena, out_result);
}

iree_status_t loom_low_allocation_rematerialize_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  *out_result = (loom_low_allocation_rematerialization_result_t){0};
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    IREE_RETURN_IF_ERROR(loom_low_allocation_try_rematerialize_value(
        module, &table->target, spill_plan->value_id, arena, out_result));
    if (out_result->rewritten_operand_count != 0) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}
