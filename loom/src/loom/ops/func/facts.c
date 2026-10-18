// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/ops.h"
#include "loom/util/fact_table.h"

iree_status_t loom_func_call_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  if (context->resolve_call_results.fn) {
    return context->resolve_call_results.fn(
        context->resolve_call_results.user_data, context, module, op,
        operand_facts, result_facts);
  }

  // Preserve externally seeded call summaries when no owning resolver is
  // active. func.call used to be opaque to local inference, and opaque results
  // consume these seeds directly in the generic fact-table traversal.
  loom_value_fact_table_t* table = context->table;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = results[i];
    if (result / 64 >= table->seeds.word_count ||
        !(table->seeds.selected_bits[result / 64] &
          (UINT64_C(1) << (result % 64)))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        table, table->seeds.table, module,
        loom_module_value_type(module, result),
        loom_value_fact_table_lookup(table->seeds.table, result),
        &result_facts[i]));
  }
  return iree_ok_status();
}
