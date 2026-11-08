// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/bindings.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/ir/module.h"

void loom_low_lower_plan_tied_result_types(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(source_op);
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->tied_result_count; ++i) {
    const loom_tied_result_t tied_result = tied_results[i];
    if (!tied_result.has_type_change) {
      loom_low_lower_inherit_value_type(context,
                                        operands[tied_result.operand_index],
                                        results[tied_result.result_index]);
    }
  }
}

void loom_low_lower_plan_value_type_id(loom_low_lower_context_t* context,
                                       loom_value_id_t source_value_id,
                                       loom_type_id_t type_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  if (iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                       LOOM_LOW_LOWER_VALUE_INHERITED_TYPE)) {
    const loom_value_ordinal_t source =
        context->lowering->value_bindings[ordinal].type_source;
    IREE_ASSERT_EQ(context->lowering->value_bindings[source].type, type_id,
                   "same-type tied result must retain its operand carrier");
    return;
  }
  context->lowering->value_bindings[ordinal].type = type_id;
  context->lowering->source_plan.value_flags[ordinal] &=
      ~LOOM_LOW_LOWER_VALUE_INHERITED_TYPE;
}

iree_status_t loom_low_lower_plan_value_type(loom_low_lower_context_t* context,
                                             loom_value_id_t source_value_id,
                                             loom_type_t type) {
  loom_type_id_t type_id;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type_id(context->module, type, &type_id));
  loom_low_lower_plan_value_type_id(context, source_value_id, type_id);
  return iree_ok_status();
}

void loom_low_lower_inherit_value_type(loom_low_lower_context_t* context,
                                       loom_value_id_t source_value_id,
                                       loom_value_id_t result_value_id) {
  loom_value_ordinal_t source =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  const loom_value_ordinal_t result =
      loom_low_lowering_frame_value_ordinal(context->lowering, result_value_id);
  if (iree_any_bit_set(context->lowering->source_plan.value_flags[source],
                       LOOM_LOW_LOWER_VALUE_INHERITED_TYPE)) {
    source = context->lowering->value_bindings[source].type_source;
  }
  context->lowering->value_bindings[result].type_source = source;
  context->lowering->source_plan.value_flags[result] |=
      LOOM_LOW_LOWER_VALUE_INHERITED_TYPE;
}

bool loom_low_lower_value_binding_type_is_planned(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  return context->lowering->value_bindings[ordinal].type !=
             LOOM_TYPE_ID_INVALID ||
         iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                          LOOM_LOW_LOWER_VALUE_INHERITED_TYPE);
}

loom_type_t loom_low_lower_value_binding_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  const loom_low_lower_value_flags_t flags =
      context->lowering->source_plan.value_flags[ordinal];
  if (!iree_any_bit_set(flags, LOOM_LOW_LOWER_VALUE_MATERIALIZED) &&
      iree_any_bit_set(flags, LOOM_LOW_LOWER_VALUE_INHERITED_TYPE)) {
    ordinal = context->lowering->value_bindings[ordinal].type_source;
  }
  const loom_low_lower_value_binding_t binding =
      context->lowering->value_bindings[ordinal];
  if (iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                       LOOM_LOW_LOWER_VALUE_MATERIALIZED)) {
    return loom_module_value_type(context->module, binding.value);
  }
  IREE_ASSERT_NE(binding.type, LOOM_TYPE_ID_INVALID,
                 "native carrier must be published by the selected producer");
  return loom_type_table_get(&context->module->types, binding.type);
}

loom_value_id_t loom_low_lower_lookup_value(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  IREE_ASSERT(
      iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                       LOOM_LOW_LOWER_VALUE_MATERIALIZED),
      "SSA lookup must follow producer materialization");
  const loom_value_id_t low_value =
      context->lowering->value_bindings[ordinal].value;
  IREE_ASSERT_NE(low_value, LOOM_LOW_LOWER_VALUE_ID_ELIDED,
                 "source-to-low requested elided source value");
  return low_value;
}

bool loom_low_lower_source_value_has_low_mapping(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  return iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                          LOOM_LOW_LOWER_VALUE_MATERIALIZED) &&
         context->lowering->value_bindings[ordinal].value !=
             LOOM_LOW_LOWER_VALUE_ID_ELIDED;
}

iree_status_t loom_low_lower_bind_value(loom_low_lower_context_t* context,
                                        loom_value_id_t source_value_id,
                                        loom_value_id_t low_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  loom_low_lower_value_binding_t* binding =
      &context->lowering->value_bindings[ordinal];
  if (iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                       LOOM_LOW_LOWER_VALUE_MATERIALIZED)) {
    IREE_ASSERT_EQ(binding->value, low_value_id);
  } else if (binding->type != LOOM_TYPE_ID_INVALID) {
    IREE_ASSERT(loom_type_equal(
                    loom_low_lower_value_binding_type(context, source_value_id),
                    loom_module_value_type(context->module, low_value_id)),
                "materialization must preserve the selected producer carrier");
  }
  binding->value = low_value_id;
  context->lowering->source_plan.value_flags[ordinal] |=
      LOOM_LOW_LOWER_VALUE_MATERIALIZED;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

iree_status_t loom_low_lower_bind_argument_transport(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id, loom_type_t abi_type) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  IREE_ASSERT(!iree_any_bit_set(
      context->lowering->source_plan.value_flags[ordinal],
      LOOM_LOW_LOWER_VALUE_MATERIALIZED));
  IREE_ASSERT(loom_type_equal(
      abi_type, loom_module_value_type(context->module, low_value_id)));
  context->lowering->value_bindings[ordinal].value = low_value_id;
  context->lowering->source_plan.value_flags[ordinal] |=
      LOOM_LOW_LOWER_VALUE_MATERIALIZED;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

iree_status_t loom_low_lower_replace_value_binding(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  IREE_ASSERT(
      loom_low_lower_source_value_has_low_mapping(context, source_value_id));
  context->lowering->value_bindings[ordinal].value = low_value_id;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

loom_value_id_t loom_low_lower_swap_value_binding(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  IREE_ASSERT(
      loom_low_lower_source_value_has_low_mapping(context, source_value_id));
  loom_low_lower_value_binding_t* binding =
      &context->lowering->value_bindings[ordinal];
  const loom_value_id_t previous = binding->value;
  IREE_ASSERT(
      loom_type_equal(loom_module_value_type(context->module, previous),
                      loom_module_value_type(context->module, low_value_id)));
  binding->value = low_value_id;
  return previous;
}

iree_status_t loom_low_lower_bind_value_alias(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value_id,
                                              loom_value_id_t result_value_id) {
  loom_value_id_t low_value_id =
      loom_low_lower_lookup_value(context, source_value_id);
  return loom_low_lower_bind_value(context, result_value_id, low_value_id);
}

void loom_low_lower_elide_value(loom_low_lower_context_t* context,
                                loom_value_id_t source_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  context->lowering->value_bindings[ordinal].value =
      LOOM_LOW_LOWER_VALUE_ID_ELIDED;
  context->lowering->source_plan.value_flags[ordinal] |=
      LOOM_LOW_LOWER_VALUE_MATERIALIZED;
}
