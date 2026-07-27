// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scf/ops.h"
#include "loom/rewrite/type_propagation.h"

iree_status_t loom_scf_region_branch_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  loom_region_branch_t branch = loom_region_branch_cast(module, op);
  if (!loom_region_branch_isa(branch) || op->result_count == 0) {
    return iree_ok_status();
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_op_t* terminator =
        loom_region_branch_region_terminator(module, branch, region_index);
    if (!terminator || terminator->operand_count != op->result_count) {
      return iree_ok_status();
    }

    const loom_value_id_t* yielded_values = loom_op_const_operands(terminator);
    for (uint16_t result_index = 0; result_index < op->result_count;
         ++result_index) {
      loom_value_id_t result = results[result_index];
      loom_value_id_t yielded = yielded_values[result_index];
      loom_type_t result_type = loom_type_transfer_value_type(context, result);
      loom_type_t yielded_type =
          loom_type_transfer_value_type(context, yielded);
      IREE_RETURN_IF_ERROR(loom_type_transfer_seed_static_structure_from_type(
          context, result, yielded_type));
      IREE_RETURN_IF_ERROR(loom_type_transfer_seed_static_structure_from_type(
          context, yielded, result_type));
    }
  }
  return iree_ok_status();
}
