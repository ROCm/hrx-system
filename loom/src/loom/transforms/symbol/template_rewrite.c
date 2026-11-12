// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_rewrite.h"

#include "loom/ir/module.h"
#include "loom/ir/type_refs.h"
#include "loom/ops/group/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/tensor/ops.h"
#include "loom/ops/tile/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

typedef iree_status_t (*loom_template_refine_fn_t)(loom_builder_t*,
                                                   loom_value_id_t, loom_type_t,
                                                   loom_location_id_t,
                                                   loom_op_t**);

static const loom_template_refine_fn_t loom_template_input_refiners[] = {
    [LOOM_TYPE_GROUP] = loom_group_refine_build,
    [LOOM_TYPE_TENSOR] = loom_tensor_refine_build,
    [LOOM_TYPE_TILE] = loom_tile_refine_build,
    [LOOM_TYPE_VECTOR] = loom_vector_refine_build,
    [LOOM_TYPE_VIEW] = loom_view_refine_build,
};

iree_status_t loom_template_rewrite_assumed_inputs(
    loom_rewriter_t* rewriter, const loom_op_t* application,
    const loom_value_id_t* source_values, const loom_value_id_t* assumed_values,
    uint16_t value_count, loom_value_id_t* operands) {
  loom_ir_remap_t remap;
  bool initialized = false;
  for (uint16_t i = 0; i < application->operand_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(rewriter->module, operands[i]);
    if (!loom_type_has_dimensions(type) ||
        !loom_type_may_reference_values(type)) {
      continue;
    }
    if (!initialized) {
      const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_initialize(rewriter->module, rewriter->module,
                                   rewriter->arena, &options, &remap));
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
          &remap, source_values, assumed_values, value_count));
      initialized = true;
    }
    loom_type_t refined_type;
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(&remap, type, &refined_type));
    if (loom_type_equal(type, refined_type)) {
      continue;
    }
    loom_op_t* refined;
    IREE_RETURN_IF_ERROR(loom_template_input_refiners[loom_type_kind(type)](
        &rewriter->builder, operands[i], refined_type, application->location,
        &refined));
    operands[i] = loom_op_results(refined)[0];
  }
  return iree_ok_status();
}

// Selection proved these input contracts. Preserve ordinary exact call typing
// with explicit carrier refinements; already-exact applications allocate none.
static iree_status_t loom_template_rewrite_inputs(
    loom_rewriter_t* rewriter, const loom_op_t* application,
    loom_symbol_ref_t callee, const loom_value_id_t** inout_operands) {
  const loom_module_t* module = rewriter->module;
  const loom_func_like_t function = loom_func_like_cast(
      module, module->symbols.entries[callee.symbol_id].defining_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_type_value_remap_t signature_remap = {
      .source_values = arguments,
      .target_values = *inout_operands,
      .count = argument_count,
      .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  loom_value_id_t* adapted = NULL;
  loom_ir_remap_t remap;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t expected = loom_module_value_type(module, arguments[i]);
    if (!loom_type_has_dimensions(expected)) {
      continue;
    }
    const loom_type_t actual =
        loom_module_value_type(module, (*inout_operands)[i]);
    if (loom_type_equal_after_value_remap(module, expected, actual,
                                          &signature_remap)) {
      continue;
    }
    if (!adapted) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          rewriter->arena, argument_count, sizeof(*adapted), (void**)&adapted));
      memcpy(adapted, *inout_operands, argument_count * sizeof(*adapted));
      IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
          module, rewriter->module, rewriter->arena, NULL, &remap));
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
          &remap, arguments, *inout_operands, argument_count));
    }
    loom_type_t refined_type;
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(&remap, expected, &refined_type));
    loom_op_t* refine = NULL;
    IREE_RETURN_IF_ERROR(loom_template_input_refiners[loom_type_kind(expected)](
        &rewriter->builder, (*inout_operands)[i], refined_type,
        application->location, &refine));
    adapted[i] = loom_op_results(refine)[0];
  }
  if (adapted) {
    *inout_operands = adapted;
  }
  return iree_ok_status();
}

iree_status_t loom_template_rewrite_apply_as_exact_call(
    loom_rewriter_t* rewriter, loom_op_t* apply_op, loom_symbol_ref_t callee,
    const loom_value_id_t* operands, loom_op_t** out_call_op) {
  IREE_ASSERT(loom_template_apply_isa(apply_op));
  loom_builder_set_before(&rewriter->builder, apply_op);
  IREE_RETURN_IF_ERROR(
      loom_template_rewrite_inputs(rewriter, apply_op, callee, &operands));
  const loom_value_slice_t results = loom_template_apply_results(apply_op);
  loom_type_t* result_types = NULL;
  if (results.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, results.count, sizeof(*result_types),
        (void**)&result_types));
    for (uint16_t i = 0; i < results.count; ++i) {
      result_types[i] =
          loom_module_value_type(rewriter->module, results.values[i]);
    }
  }

  loom_template_call_build_flags_t build_flags = 0;
  const loom_callable_effect_contract_t effect_contract =
      loom_template_apply_effect_contract(apply_op);
  if (effect_contract != 0) {
    build_flags |= LOOM_TEMPLATE_CALL_BUILD_FLAG_HAS_EFFECT_CONTRACT;
  }
  const uint8_t temperature = loom_template_apply_temperature(apply_op);
  if (temperature != 0) {
    build_flags |= LOOM_TEMPLATE_CALL_BUILD_FLAG_HAS_TEMPERATURE;
  }

  loom_builder_set_before(&rewriter->builder, apply_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_template_call_build(
      &rewriter->builder, build_flags, effect_contract, temperature, callee,
      operands, apply_op->operand_count, result_types, results.count,
      loom_op_tied_results(apply_op), apply_op->tied_result_count,
      apply_op->location, &call_op));
  const loom_value_slice_t call_results = loom_template_call_results(call_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, apply_op, call_results.values, call_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, apply_op, call_results.values, call_results.count));
  if (out_call_op) {
    *out_call_op = call_op;
  }
  return iree_ok_status();
}
