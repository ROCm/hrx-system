// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/value_refinement.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/rewrite/rewriter.h"
#include "loom/rewrite/type_propagation.h"

iree_status_t loom_value_refinement_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  const loom_type_t source_type =
      loom_module_value_type(module, loom_op_operands(op)[0]);
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  if (!loom_type_rank_equals(source_type, result_type)) {
    return iree_ok_status();
  }
  for (uint8_t axis = 0; axis < loom_type_rank(source_type); ++axis) {
    if (loom_type_dim_is_dynamic_at(source_type, axis) ||
        loom_type_dim_is_dynamic_at(result_type, axis)) {
      continue;
    }
    const int64_t source_size = loom_type_dim_static_size_at(source_type, axis);
    const int64_t result_size = loom_type_dim_static_size_at(result_type, axis);
    if (source_size == result_size) {
      continue;
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("source static dimension")),
        loom_param_i64(source_size),
        loom_param_string(IREE_SV("result static dimension")),
        loom_param_i64(result_size),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_SHAPE_001,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  return iree_ok_status();
}

iree_status_t loom_value_refinement_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_value_refinement_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  (void)module;
  const loom_value_id_t source = loom_op_operands(op)[0];
  const loom_value_id_t result = loom_op_results(op)[0];
  const loom_type_t source_type =
      loom_type_transfer_value_type(context, source);
  const loom_type_t result_type =
      loom_type_transfer_value_type(context, result);
  IREE_RETURN_IF_ERROR(loom_type_transfer_seed_static_structure_from_type(
      context, source, result_type));
  return loom_type_transfer_seed_static_structure_from_type(context, result,
                                                            source_type);
}

iree_status_t loom_value_refinement_canonicalize(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  const loom_value_id_t source = loom_op_operands(op)[0];
  const loom_value_id_t result = loom_op_results(op)[0];
  if (!loom_type_equal(loom_module_value_type(rewriter->module, source),
                       loom_module_value_type(rewriter->module, result))) {
    return iree_ok_status();
  }
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &source, 1);
}
