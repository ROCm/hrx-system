// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/operand_ties.h"

#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"

bool loom_low_operand_ties_need_materialization(
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op) {
  loom_low_descriptor_packet_t packet_storage;
  loom_low_descriptor_packet_initialize(descriptor_set, op, &packet_storage);
  const loom_low_descriptor_packet_t* packet = &packet_storage;
  if (packet->kind != LOOM_LOW_DESCRIPTOR_PACKET_OP ||
      !iree_any_bit_set(packet->descriptor->flags,
                        LOOM_LOW_DESCRIPTOR_FLAG_TIED_RESULTS) ||
      packet->op->tied_result_count == packet->op->result_count) {
    return false;
  }
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  for (uint16_t i = 0; i < descriptor_layout->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor_layout->constraint_start + i];
    if ((constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED ||
         constraint->kind == LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) &&
        !loom_op_find_tied_result(packet->op, constraint->lhs_operand_index)) {
      return true;
    }
  }
  return false;
}

// Physical input/result aliasing does not consume the authored input's IR
// ownership. A copy gives the update its own storage lineage; ordinary copy
// coalescing removes transport when the source is no longer observable.
iree_status_t loom_low_operand_ties_materialize(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_low_memory_access_map_t* memory_accesses, loom_op_t* op,
    loom_op_t** out_replacement) {
  *out_replacement = op;
  if (!loom_low_operand_ties_need_materialization(descriptor_set, op)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[loom_low_op_descriptor(op)];
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  loom_tied_result_t* ties = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*ties), (void**)&ties));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    ties[i] = (loom_tied_result_t){
        .result_index = i,
        .operand_index = LOOM_LOW_ID_NONE,
    };
  }
  const loom_tied_result_t* original_ties = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    ties[original_ties[i].result_index] = original_ties[i];
  }
  loom_value_id_t* operands = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->operand_count,
                                sizeof(*operands), (void**)&operands));
  memcpy(operands, loom_op_const_operands(op),
         op->operand_count * sizeof(*operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*result_types), (void**)&result_types));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_types[i] =
        loom_module_value_type(rewriter->module, loom_op_results(op)[i]);
  }

  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, op);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       iree_status_is_ok(status) && i < descriptor_layout->constraint_count;
       ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor_layout->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED &&
        constraint->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    loom_tied_result_t* tie = &ties[constraint->lhs_operand_index];
    if (tie->operand_index != LOOM_LOW_ID_NONE) {
      continue;
    }
    const uint16_t operand_index =
        loom_low_descriptor_operand(descriptor_set, descriptor,
                                    constraint->rhs_operand_index)
            ->source_value_index;
    tie->operand_index = operand_index;
    const loom_value_id_t source = operands[operand_index];
    loom_op_t* copy = NULL;
    status = loom_low_copy_build(
        &rewriter->builder, source, /*detached=*/false,
        loom_module_value_type(rewriter->module, source), op->location, &copy);
    if (iree_status_is_ok(status)) {
      operands[operand_index] = loom_op_results(copy)[0];
      status = loom_rewriter_try_set_derived_value_name(
          rewriter, source, operands[operand_index], IREE_SV("copy"));
    }
  }
  uint16_t tie_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (ties[i].operand_index != LOOM_LOW_ID_NONE) {
      ties[tie_count++] = ties[i];
    }
  }
  loom_op_t* replacement = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_build_prepared_descriptor_op(
        &rewriter->builder, descriptor_set, descriptor, op->instance_flags,
        operands, op->operand_count, loom_low_op_attrs_attr(op), result_types,
        op->result_count, ties, tie_count, op->location, &replacement);
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(
      loom_low_memory_access_map_replace(memory_accesses, op, replacement));
  const loom_value_id_t* replacements = loom_op_results(replacement);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, replacement->result_count, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, replacement->result_count));
  *out_replacement = replacement;
  return iree_ok_status();
}
