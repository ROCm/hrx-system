// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the command dialect.

#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/view/reference.h"

iree_status_t loom_command_parameter_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_command_parameter_result(op));
  return loom_view_reference_make_unplaced_view(
      context, module, loom_command_parameter_result(op), operand_facts[0],
      result_type, &result_facts[0]);
}

iree_status_t loom_command_parameter_base_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  loom_value_fact_buffer_reference_t buffer = {
      .maximum_byte_extent = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = loom_command_parameter_base_result(op),
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  loom_value_fact_view_reference_t view = {0};
  if (loom_value_facts_query_view_reference(context, operand_facts[0], &view)) {
    buffer.minimum_alignment = view.root_minimum_alignment;
    buffer.memory_space = view.memory_space;
    buffer.root_value_id = loom_value_fact_view_reference_resolve_root_value(
        view, loom_command_parameter_base_source(op));
    buffer.alias_scope_id = view.alias_scope_id;
    buffer.nullability = view.nullability;
    buffer.origin = view.origin;
  }
  return loom_value_facts_make_buffer_reference(context, buffer,
                                                &result_facts[0]);
}
