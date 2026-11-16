// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/index/combine_patterns.h"

#include "loom/ir/module.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/scalar/narrowing.h"

static loom_op_t* loom_index_combine_defining_op(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(rewriter->module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

// Only the complete unsigned word domain is implicit in the replacement cast.
// Stronger bounds, divisibility, and relations remain explicit assumptions so
// later consumers retain their source contracts after facts are recomputed.
static bool loom_index_combine_is_unsigned_word_assumption(
    const loom_op_t* op) {
  const loom_attribute_t predicates = loom_scalar_assume_predicates(op);
  if (predicates.count != 1) {
    return false;
  }
  const loom_predicate_t* predicate = predicates.predicate_list;
  return predicate->kind == LOOM_PREDICATE_RANGE &&
         predicate->arg_tags[0] == LOOM_PRED_ARG_VALUE &&
         predicate->args[0] == loom_op_const_operands(op)[0] &&
         predicate->arg_tags[1] == LOOM_PRED_ARG_CONST &&
         predicate->args[1] == 0 &&
         predicate->arg_tags[2] == LOOM_PRED_ARG_CONST &&
         predicate->args[2] == UINT32_MAX;
}

static iree_status_t loom_index_narrow_offset_arithmetic_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  *out_changed = false;
  const loom_value_id_t input = loom_index_cast_input(op);
  const loom_value_id_t result = loom_index_cast_result(op);
  if (!loom_type_equal(loom_module_value_type(rewriter->module, input),
                       loom_type_scalar(LOOM_SCALAR_TYPE_I64)) ||
      !loom_type_equal(loom_module_value_type(rewriter->module, result),
                       loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)) ||
      !rewriter->fact_table ||
      loom_index_target_carrier_bitwidth(&rewriter->fact_table->context,
                                         LOOM_SCALAR_TYPE_OFFSET) != 32 ||
      !loom_value_facts_fit_unsigned_bit_count(
          loom_rewriter_value_facts(rewriter, input), 32)) {
    return iree_ok_status();
  }

  loom_op_t* producer = loom_index_combine_defining_op(rewriter, input);
  loom_op_t* assumption = NULL;
  if (producer && loom_scalar_assume_isa(producer) &&
      producer->operand_count == 1 && producer->result_count == 1) {
    const loom_value_t* value = loom_module_value(rewriter->module, input);
    if (!loom_value_has_single_use(value) ||
        loom_value_has_attribute_uses(value) ||
        loom_module_value_has_type_uses(rewriter->module, input) ||
        !loom_index_combine_is_unsigned_word_assumption(producer)) {
      return iree_ok_status();
    }
    assumption = producer;
    producer = loom_index_combine_defining_op(
        rewriter, loom_op_const_operands(assumption)[0]);
  }
  loom_scalar_narrowing_plan_t plan;
  if (!producer ||
      !loom_scalar_narrowing_select(rewriter, producer, assumption, &plan)) {
    return iree_ok_status();
  }

  // The original numeric value is proven unsigned32. Its low word followed by
  // the unsigned i32-to-offset conversion therefore preserves that value,
  // including bit 31. Select actual arithmetic before introducing conversions.
  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* arithmetic = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build(
      rewriter, &plan, producer->location, &arithmetic));
  const loom_value_id_t arithmetic_result =
      loom_op_const_results(arithmetic)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
      rewriter, loom_op_const_results(producer)[0], arithmetic_result));
  loom_op_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &rewriter->builder, arithmetic_result,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), op->location, &replacement));
  const loom_value_id_t replacement_result =
      loom_index_cast_result(replacement);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_copy_value_name(rewriter, result, replacement_result));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, &replacement_result, 1));
  if (assumption) {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, assumption));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, producer));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_index_narrow_payload_arithmetic_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  *out_changed = false;
  const loom_value_id_t input = loom_index_cast_input(op);
  const loom_value_id_t result = loom_index_cast_result(op);
  const loom_scalar_type_t input_type =
      loom_type_element_type(loom_module_value_type(rewriter->module, input));
  if ((input_type != LOOM_SCALAR_TYPE_INDEX &&
       input_type != LOOM_SCALAR_TYPE_OFFSET) ||
      !loom_type_equal(loom_module_value_type(rewriter->module, result),
                       loom_type_scalar(LOOM_SCALAR_TYPE_I32))) {
    return iree_ok_status();
  }
  loom_op_t* producer = loom_index_combine_defining_op(rewriter, input);
  loom_scalar_narrowing_plan_t plan;
  if (!producer ||
      !loom_scalar_narrowing_select(rewriter, producer, NULL, &plan)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* arithmetic = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_narrowing_build(
      rewriter, &plan, producer->location, &arithmetic));
  const loom_value_id_t replacement = loom_op_const_results(arithmetic)[0];
  IREE_RETURN_IF_ERROR(
      loom_rewriter_copy_value_name(rewriter, result, replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
      rewriter, loom_op_const_results(producer)[0], replacement));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  // Retire the owned producer before the worklist visits its new operand
  // casts. Their producers can then narrow without appearing shared.
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, producer));
  *out_changed = true;
  return iree_ok_status();
}

// A selected producer has one ordinary use, which already identifies its
// operand slot in this assumption. Use that retained index to remap every
// predicate in one pass instead of copying the list once per narrowed input.
static iree_status_t loom_index_narrow_assumed_predicates(
    loom_rewriter_t* rewriter, loom_op_t* assumption,
    const loom_value_id_t* replacements) {
  const loom_attribute_t original = loom_index_assume_predicates(assumption);
  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_copy_predicate_list_attr_storage(
      &rewriter->builder, original.predicate_list, original.count,
      IREE_SV("narrowed index predicates"), &predicates));
  for (uint16_t i = 0; i < original.count; ++i) {
    loom_predicate_t* predicate = &predicates[i];
    for (uint8_t j = 0; j < predicate->arg_count; ++j) {
      if (predicate->arg_tags[j] != LOOM_PRED_ARG_VALUE ||
          predicate->args[j] < 0) {
        continue;
      }
      const loom_use_t* use = loom_value_single_use(loom_module_value(
          rewriter->module, (loom_value_id_t)predicate->args[j]));
      if (use && loom_use_user_op(*use) == assumption) {
        predicate->args[j] = replacements[loom_use_operand_index(*use)];
      }
    }
  }
  return loom_rewriter_set_attr(
      rewriter, assumption, 0,
      loom_attr_predicate_list(predicates, original.count));
}

static iree_status_t loom_index_narrow_assumed_arithmetic_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  *out_changed = false;
  if (!rewriter->fact_table ||
      loom_index_target_carrier_bitwidth(&rewriter->fact_table->context,
                                         LOOM_SCALAR_TYPE_INDEX) != 32) {
    return iree_ok_status();
  }
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t* original_inputs = NULL;
  loom_value_id_t* replacements = NULL;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < op->operand_count && iree_status_is_ok(status);
       ++i) {
    const loom_value_id_t input = loom_op_const_operands(op)[i];
    const loom_value_id_t result = loom_op_const_results(op)[i];
    if (!loom_type_equal(loom_module_value_type(rewriter->module, input),
                         index_type) ||
        loom_value_facts_fit_signed_bit_count(
            loom_rewriter_value_facts(rewriter, input), 32) ||
        !loom_value_facts_fit_signed_bit_count(
            loom_rewriter_value_facts(rewriter, result), 32)) {
      continue;
    }
    loom_op_t* producer = loom_index_combine_defining_op(rewriter, input);
    loom_scalar_narrowing_plan_t plan;
    if (!producer ||
        !loom_scalar_narrowing_select(rewriter, producer, op, &plan)) {
      continue;
    }
    if (!replacements) {
      status = iree_arena_allocate_array(
          rewriter->arena, 2 * (iree_host_size_t)op->operand_count,
          sizeof(*replacements), (void**)&original_inputs);
      if (iree_status_is_ok(status)) {
        replacements = original_inputs + op->operand_count;
        for (uint16_t j = 0; j < op->operand_count; ++j) {
          original_inputs[j] = replacements[j] = loom_op_const_operands(op)[j];
        }
      }
    }
    // The signed32 result contract makes the low word sufficient, including
    // negative values. Restore index before the unchanged assumption so its
    // stronger bounds and relations survive later fact recomputation.
    loom_op_t* arithmetic = NULL;
    if (iree_status_is_ok(status)) {
      loom_builder_set_before(&rewriter->builder, op);
      status = loom_scalar_narrowing_build(rewriter, &plan, producer->location,
                                           &arithmetic);
    }
    if (iree_status_is_ok(status)) {
      loom_op_t* restored = NULL;
      status = loom_index_cast_build(&rewriter->builder,
                                     loom_op_const_results(arithmetic)[0],
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                     index_type, producer->location, &restored);
      if (iree_status_is_ok(status)) {
        replacements[i] = loom_index_cast_result(restored);
        status =
            loom_rewriter_copy_value_name(rewriter, input, replacements[i]);
      }
    }
  }
  if (iree_status_is_ok(status) && replacements) {
    status = loom_index_narrow_assumed_predicates(rewriter, op, replacements);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operands(rewriter, op, replacements);
    }
    for (uint16_t i = 0; i < op->operand_count && iree_status_is_ok(status);
         ++i) {
      if (original_inputs[i] != replacements[i]) {
        loom_op_t* producer =
            loom_index_combine_defining_op(rewriter, original_inputs[i]);
        status = loom_rewriter_replace_all_uses_and_erase(rewriter, producer,
                                                          &replacements[i], 1);
      }
    }
    *out_changed = iree_status_is_ok(status);
  }
  return status;
}

static const loom_rewrite_pattern_t kIndexSourceCombinePatterns[] = {
    {
        .root_kind = LOOM_OP_INDEX_CAST,
        .match_and_rewrite = loom_index_narrow_offset_arithmetic_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_CAST,
        .match_and_rewrite = loom_index_narrow_payload_arithmetic_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_ASSUME,
        .match_and_rewrite = loom_index_narrow_assumed_arithmetic_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_index_source_combine_pattern_provider = {
        .name = IREE_SVL("index"),
        .patterns = kIndexSourceCombinePatterns,
        .pattern_count = IREE_ARRAYSIZE(kIndexSourceCombinePatterns),
};
