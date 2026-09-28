// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

iree_status_t loom_config_get_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  const loom_value_id_t result = loom_config_get_result(op);
  const loom_type_t type = loom_module_value_type(rewriter->module, result);
  if (!loom_type_is_encoding(type) || !rewriter->fact_table) {
    return iree_ok_status();
  }
  // Scalar configs use the ordinary constant folder. Encoding constants carry
  // their exact specification in the encoding fact domain instead.
  loom_value_fact_encoding_summary_t summary;
  if (!loom_value_facts_query_encoding_summary(
          &rewriter->fact_table->context,
          loom_rewriter_value_facts(rewriter, result), &summary) ||
      summary.static_spec_encoding_id == 0) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_define_build(
      &rewriter->builder, summary.static_spec_encoding_id, NULL, 0, type,
      op->location, &constant));
  const loom_value_id_t replacement = loom_encoding_define_result(constant);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}
