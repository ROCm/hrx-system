// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

static const loom_op_t* loom_global_load_definition(const loom_module_t* module,
                                                    const loom_op_t* op) {
  loom_symbol_ref_t ref = loom_global_load_global(op);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL)) {
    return NULL;
  }
  return symbol->defining_op;
}

iree_status_t loom_global_load_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  if (!rewriter->fact_table || !rewriter->materialize_constant) {
    return iree_ok_status();
  }

  const loom_op_t* definition_op =
      loom_global_load_definition(rewriter->module, op);
  if (!definition_op || !loom_global_constant_isa(definition_op)) {
    return iree_ok_status();
  }

  loom_value_slice_t results = loom_global_load_result(op);
  if (results.count == 0) return iree_ok_status();
  for (uint16_t i = 0; i < results.count; ++i) {
    if (results.values[i] == LOOM_VALUE_ID_INVALID) return iree_ok_status();
    loom_value_facts_t facts =
        loom_rewriter_value_facts(rewriter, results.values[i]);
    if (!loom_value_facts_is_exact(facts)) return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t* replacement_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_value_id_scratch(
      rewriter->fact_table, results.count, &replacement_ids));
  for (uint16_t i = 0; i < results.count; ++i) {
    loom_value_facts_t facts =
        loom_rewriter_value_facts(rewriter, results.values[i]);
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, results.values[i]);
    IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
        rewriter, facts, result_type, op->location, &replacement_ids[i]));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacement_ids, results.count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacement_ids,
                                                  results.count);
}
