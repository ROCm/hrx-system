// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/block_arguments.h"

#include "loom/ir/module.h"
#include "loom/ir/type_refs.h"

bool loom_cfg_block_arguments_can_bind(const loom_module_t* module,
                                       const loom_dominance_info_t* dominance,
                                       const loom_block_t* block,
                                       loom_value_slice_t replacements,
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
        !loom_type_equal_after_value_remap(
            module, loom_module_value_type(module, loom_block_arg_id(block, i)),
            loom_module_value_type(module, replacement), &remap)) {
      return false;
    }
  }
  return true;
}

bool loom_cfg_block_argument_can_forward(const loom_module_t* module,
                                         loom_value_id_t argument,
                                         loom_value_id_t source) {
  const loom_value_t* source_value = loom_module_value(module, source);
  const loom_type_t* payload =
      loom_type_register_value_type(source_value->type);
  if (loom_type_is_reference(payload ? *payload : source_value->type)) {
    // Managed CFG arguments transfer lifetime state. Equal payloads do not
    // establish interchangeable owners or borrow protection. Descriptor-only
    // use summaries omit type-directed call, return, and edge transfers.
    return false;
  }
  if (loom_value_ownership_use_count(source_value) != 0 ||
      loom_value_ownership_use_count(loom_module_value(module, argument)) !=
          0) {
    return false;
  }
  if (!loom_value_is_block_arg(source_value)) {
    const loom_op_t* definition = loom_value_def_op(source_value);
    if (definition && loom_traits_have_storage_relation(definition->traits) &&
        (loom_traits_are_fact_identity(definition->traits) ||
         loom_traits_are_value_alias(definition->traits))) {
      // Nonwriting aliases share an owner's observations that are not present
      // in this value's use list. Tied results instead start new ownership.
      return false;
    }
  }
  return true;
}
