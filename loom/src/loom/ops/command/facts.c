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
  return loom_view_reference_make_buffer_view(
      context, module, loom_command_parameter_source(op), operand_facts[0],
      loom_value_facts_unknown(), result_type, &result_facts[0]);
}
