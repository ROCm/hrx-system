// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/block_arguments.h"

#include "loom/ir/module.h"

bool loom_cfg_block_arguments_can_replace(
    const loom_module_t* module, const loom_dominance_info_t* dominance,
    const loom_block_t* block, loom_value_slice_t replacements,
    const loom_op_t* before_op) {
  if (replacements.count != block->arg_count) {
    return false;
  }
  const loom_type_value_remap_t remap = {
      .source_values = block->arg_ids,
      .target_values = replacements.values,
      .count = block->arg_count,
      .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t replacement = replacements.values[i];
    if (!loom_value_is_available_before_op(dominance, replacement, before_op) ||
        !loom_value_type_is_available_before_op(dominance, replacement,
                                                before_op) ||
        !loom_type_equal_after_value_remap(
            module, loom_module_value_type(module, loom_block_arg_id(block, i)),
            loom_module_value_type(module, replacement), &remap)) {
      return false;
    }
  }
  return true;
}
