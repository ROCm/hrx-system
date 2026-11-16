// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/narrowing.h"

#include "loom/ir/module.h"
#include "loom/ir/type_dependencies.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

static loom_scalar_narrowing_operand_t loom_scalar_narrowing_select_operand(
    const loom_rewriter_t* rewriter, loom_value_id_t input) {
  loom_op_t* producer = loom_rewriter_value_defining_op(rewriter, input);
  if (producer && (loom_scalar_constant_isa(producer) ||
                   loom_index_constant_isa(producer))) {
    // Keep the literal conversion defined even when the low word is negative.
    int64_t constant =
        (uint32_t)loom_attr_as_i64(loom_scalar_constant_isa(producer)
                                       ? loom_scalar_constant_value(producer)
                                       : loom_index_constant_value(producer));
    if (constant >= INT64_C(0x80000000)) {
      constant -= INT64_C(0x100000000);
    }
    return (loom_scalar_narrowing_operand_t){
        .kind = LOOM_SCALAR_NARROWING_OPERAND_CONSTANT,
        .constant = constant,
    };
  }
  if (producer &&
      (loom_scalar_extsi_isa(producer) || loom_scalar_extui_isa(producer))) {
    const loom_value_id_t source = loom_op_const_operands(producer)[0];
    if (loom_type_equal(loom_module_value_type(rewriter->module, source),
                        loom_type_scalar(LOOM_SCALAR_TYPE_I32))) {
      return (loom_scalar_narrowing_operand_t){
          .kind = LOOM_SCALAR_NARROWING_OPERAND_REUSE,
          .value = source,
      };
    }
  }
  if (producer && loom_index_cast_isa(producer)) {
    const loom_value_id_t source = loom_index_cast_input(producer);
    if (loom_type_equal(loom_module_value_type(rewriter->module, source),
                        loom_type_scalar(LOOM_SCALAR_TYPE_I32))) {
      return (loom_scalar_narrowing_operand_t){
          .kind = LOOM_SCALAR_NARROWING_OPERAND_REUSE,
          .value = source,
      };
    }
    if (loom_type_equal(loom_module_value_type(rewriter->module, source),
                        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)) &&
        rewriter->fact_table &&
        loom_index_target_carrier_bitwidth(&rewriter->fact_table->context,
                                           LOOM_SCALAR_TYPE_OFFSET) == 32) {
      return (loom_scalar_narrowing_operand_t){
          .kind = LOOM_SCALAR_NARROWING_OPERAND_ADDRESS,
          .value = source,
      };
    }
  }
  const loom_scalar_type_t input_type =
      loom_type_element_type(loom_module_value_type(rewriter->module, input));
  if (input_type == LOOM_SCALAR_TYPE_INDEX ||
      input_type == LOOM_SCALAR_TYPE_OFFSET) {
    return (loom_scalar_narrowing_operand_t){
        .kind = LOOM_SCALAR_NARROWING_OPERAND_ADDRESS,
        .value = input,
    };
  }
  return (loom_scalar_narrowing_operand_t){
      .kind = LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE,
      .value = input,
  };
}

static bool loom_scalar_narrowing_has_only_attribute_owner(
    const loom_rewriter_t* rewriter, loom_value_id_t input, loom_op_t* owner) {
  if (!loom_value_has_attribute_uses(
          loom_module_value(rewriter->module, input))) {
    return true;
  }
  if (!owner) {
    return false;
  }
  // The retained membership index supplies at most two owners: the permitted
  // local assumption followed by end-of-range. Other observers reject cloning.
  loom_type_use_iterator_t iterator;
  loom_attribute_users_begin(&rewriter->module->type_uses, input, &iterator);
  if (loom_attribute_users_next(&iterator).op != owner) {
    return false;
  }
  return !loom_attribute_users_next(&iterator).op;
}

bool loom_scalar_narrowing_select(loom_rewriter_t* rewriter,
                                  loom_op_t* producer,
                                  loom_op_t* attribute_owner,
                                  loom_scalar_narrowing_plan_t* out_plan) {
  if (producer->instance_flags) {
    return false;
  }
  loom_op_kind_t kind = producer->kind;
  switch (producer->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_SCALAR_ANDI:
    case LOOM_OP_SCALAR_ORI:
    case LOOM_OP_SCALAR_XORI:
    case LOOM_OP_INDEX_MADD:
      break;
    case LOOM_OP_INDEX_ADD:
      kind = LOOM_OP_SCALAR_ADDI;
      break;
    case LOOM_OP_INDEX_SUB:
      kind = LOOM_OP_SCALAR_SUBI;
      break;
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_SCALE:
      kind = LOOM_OP_SCALAR_MULI;
      break;
    case LOOM_OP_INDEX_ANDI:
      kind = LOOM_OP_SCALAR_ANDI;
      break;
    case LOOM_OP_INDEX_ORI:
      kind = LOOM_OP_SCALAR_ORI;
      break;
    case LOOM_OP_INDEX_XORI:
      kind = LOOM_OP_SCALAR_XORI;
      break;
    case LOOM_OP_INDEX_SHLI:
    case LOOM_OP_SCALAR_SHLI: {
      if (!loom_value_facts_fit_unsigned_bit_count(
              loom_rewriter_value_facts(rewriter,
                                        loom_op_const_operands(producer)[1]),
              5)) {
        return false;
      }
      kind = LOOM_OP_SCALAR_SHLI;
      break;
    }
    default:
      return false;
  }
  const loom_value_id_t input = loom_op_const_results(producer)[0];
  const loom_scalar_type_t input_type =
      loom_type_element_type(loom_module_value_type(rewriter->module, input));
  if ((input_type == LOOM_SCALAR_TYPE_INDEX ||
       input_type == LOOM_SCALAR_TYPE_OFFSET) &&
      (!rewriter->fact_table ||
       loom_index_target_carrier_bitwidth(&rewriter->fact_table->context,
                                          input_type) < 32)) {
    return false;
  }
  const loom_value_t* value = loom_module_value(rewriter->module, input);
  if (!loom_value_has_single_use(value) ||
      loom_module_value_has_type_uses(rewriter->module, input)) {
    return false;
  }
  loom_scalar_narrowing_plan_t plan = {
      .kind = kind,
      .operands =
          {
              loom_scalar_narrowing_select_operand(
                  rewriter, loom_op_const_operands(producer)[0]),
              loom_scalar_narrowing_select_operand(
                  rewriter, loom_op_const_operands(producer)[1]),
          },
  };
  if (kind == LOOM_OP_INDEX_MADD) {
    plan.operands[2] = loom_scalar_narrowing_select_operand(
        rewriter, loom_op_const_operands(producer)[2]);
  }
  if ((plan.operands[0].kind == LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE &&
       plan.operands[1].kind == LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE &&
       (kind != LOOM_OP_INDEX_MADD ||
        plan.operands[2].kind == LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE)) ||
      !loom_scalar_narrowing_has_only_attribute_owner(rewriter, input,
                                                      attribute_owner)) {
    return false;
  }
  *out_plan = plan;
  return true;
}

static iree_status_t loom_scalar_narrowing_build_operand(
    loom_rewriter_t* rewriter, const loom_scalar_narrowing_operand_t* operand,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  switch (operand->kind) {
    case LOOM_SCALAR_NARROWING_OPERAND_REUSE:
      *out_value = operand->value;
      return iree_ok_status();
    case LOOM_SCALAR_NARROWING_OPERAND_CONSTANT: {
      IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
          &rewriter->builder, loom_attr_i64(operand->constant), i32, location,
          &op));
      break;
    }
    case LOOM_SCALAR_NARROWING_OPERAND_ADDRESS: {
      IREE_RETURN_IF_ERROR(loom_index_cast_build(
          &rewriter->builder, operand->value,
          loom_module_value_type(rewriter->module, operand->value), i32,
          location, &op));
      break;
    }
    case LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE: {
      IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
          &rewriter->builder, operand->value,
          loom_type_scalar(LOOM_SCALAR_TYPE_I64), i32, location, &op));
      break;
    }
  }
  *out_value = loom_op_const_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_scalar_narrowing_build_arithmetic(
    loom_rewriter_t* rewriter, loom_op_kind_t kind, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location, loom_op_t** out_op) {
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  switch (kind) {
    case LOOM_OP_SCALAR_ADDI:
      return loom_scalar_addi_build(&rewriter->builder, 0, lhs, rhs, i32,
                                    location, out_op);
    case LOOM_OP_SCALAR_SUBI:
      return loom_scalar_subi_build(&rewriter->builder, 0, lhs, rhs, i32,
                                    location, out_op);
    case LOOM_OP_SCALAR_MULI:
      return loom_scalar_muli_build(&rewriter->builder, 0, lhs, rhs, i32,
                                    location, out_op);
    case LOOM_OP_SCALAR_ANDI:
      return loom_scalar_andi_build(&rewriter->builder, lhs, rhs, i32, location,
                                    out_op);
    case LOOM_OP_SCALAR_ORI:
      return loom_scalar_ori_build(&rewriter->builder, lhs, rhs, i32, location,
                                   out_op);
    case LOOM_OP_SCALAR_XORI:
      return loom_scalar_xori_build(&rewriter->builder, lhs, rhs, i32, location,
                                    out_op);
    case LOOM_OP_SCALAR_SHLI:
      return loom_scalar_shli_build(&rewriter->builder, 0, lhs, rhs, i32,
                                    location, out_op);
    default:
      IREE_ASSERT_UNREACHABLE("selected low-word arithmetic");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_scalar_narrowing_build(
    loom_rewriter_t* rewriter, const loom_scalar_narrowing_plan_t* plan,
    loom_location_id_t location, loom_op_t** out_op) {
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build_operand(
      rewriter, &plan->operands[0], location, &lhs));
  IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build_operand(
      rewriter, &plan->operands[1], location, &rhs));
  if (plan->kind == LOOM_OP_INDEX_MADD) {
    loom_value_id_t addend = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build_operand(
        rewriter, &plan->operands[2], location, &addend));
    loom_op_t* product = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
        &rewriter->builder, 0, lhs, rhs, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
        location, &product));
    return loom_scalar_addi_build(
        &rewriter->builder, 0, loom_scalar_muli_result(product), addend,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, out_op);
  }
  return loom_scalar_narrowing_build_arithmetic(rewriter, plan->kind, lhs, rhs,
                                                location, out_op);
}

iree_status_t loom_scalar_narrowing_truncate(loom_rewriter_t* rewriter,
                                             loom_op_t* truncation,
                                             loom_op_t* producer,
                                             bool* out_changed) {
  *out_changed = false;
  const loom_value_id_t input = loom_scalar_trunci_input(truncation);
  const loom_value_id_t result = loom_scalar_trunci_result(truncation);
  if (!loom_type_equal(loom_module_value_type(rewriter->module, input),
                       loom_type_scalar(LOOM_SCALAR_TYPE_I64)) ||
      !loom_type_equal(loom_module_value_type(rewriter->module, result),
                       loom_type_scalar(LOOM_SCALAR_TYPE_I32))) {
    return iree_ok_status();
  }
  if (loom_index_cast_isa(producer)) {
    const loom_value_id_t source = loom_index_cast_input(producer);
    const loom_type_t source_type =
        loom_module_value_type(rewriter->module, source);
    if (!loom_type_equal(source_type,
                         loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET))) {
      return iree_ok_status();
    }
    loom_builder_set_before(&rewriter->builder, truncation);
    loom_op_t* replacement = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_cast_build(&rewriter->builder, source, source_type,
                              loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                              truncation->location, &replacement));
    const loom_value_id_t value = loom_op_const_results(replacement)[0];
    IREE_RETURN_IF_ERROR(
        loom_rewriter_copy_value_name(rewriter, result, value));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, truncation, &value, 1));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_scalar_narrowing_plan_t plan;
  if (!loom_scalar_narrowing_select(rewriter, producer, NULL, &plan)) {
    return iree_ok_status();
  }
  const bool left_residual =
      plan.operands[0].kind == LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE;
  const bool right_residual =
      plan.operands[1].kind == LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE;
  const loom_value_t* value = loom_module_value(rewriter->module, result);
  if (left_residual != right_residual &&
      truncation->parent_block == producer->parent_block &&
      !loom_value_has_attribute_uses(value) &&
      !loom_module_value_has_type_uses(rewriter->module, result)) {
    const loom_scalar_narrowing_operand_t* residual =
        &plan.operands[left_residual ? 0 : 1];
    const loom_scalar_narrowing_operand_t* folded =
        &plan.operands[left_residual ? 1 : 0];
    loom_builder_set_before(&rewriter->builder, producer);
    loom_value_id_t folded_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build_operand(
        rewriter, folded, truncation->location, &folded_value));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_operand(rewriter, truncation, 0, residual->value));
    // Both inputs dominate the owned producer. Keeping the reused truncation
    // beside it avoids accumulating insertions after a stationary late anchor.
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, truncation, producer));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, producer));
    loom_builder_set_after(&rewriter->builder, truncation);
    loom_op_t* replacement = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build_arithmetic(
        rewriter, plan.kind, left_residual ? result : folded_value,
        left_residual ? folded_value : result, truncation->location,
        &replacement));
    const loom_value_id_t replacement_value =
        loom_op_const_results(replacement)[0];
    // The retained truncation now means the operand, not the complete result.
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_value_name(rewriter, result, replacement_value));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
        rewriter, result, replacement_value, replacement));
  } else {
    loom_builder_set_before(&rewriter->builder, truncation);
    loom_op_t* replacement = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build(
        rewriter, &plan, truncation->location, &replacement));
    const loom_value_id_t replacement_value =
        loom_op_const_results(replacement)[0];
    IREE_RETURN_IF_ERROR(
        loom_rewriter_copy_value_name(rewriter, result, replacement_value));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, truncation, &replacement_value, 1));
    // Its only observer is gone. Retire it before new operand truncations are
    // processed so their own producers do not still appear shared.
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, producer));
  }
  *out_changed = true;
  return iree_ok_status();
}
