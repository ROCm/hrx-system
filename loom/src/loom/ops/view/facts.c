// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the view dialect.

#include "loom/ir/module.h"
#include "loom/ops/view/ops.h"
#include "loom/ops/view/reference.h"

iree_status_t loom_view_subview_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_view_subview_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_view_subview_result(op));
  return loom_view_reference_make_subview(
      context, module, loom_view_subview_source(op), operand_facts[0],
      loom_view_subview_static_offsets(op), loom_view_subview_offsets(op),
      source_type, result_type, &result_facts[0]);
}

iree_status_t loom_view_refine_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_view_refine_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_view_refine_result(op));
  return loom_view_reference_make_refine(
      context, module, loom_view_refine_source(op), operand_facts[0],
      source_type, result_type, &result_facts[0]);
}
