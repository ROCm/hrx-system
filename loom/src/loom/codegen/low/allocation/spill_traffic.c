// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_traffic.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

loom_low_allocation_spill_register_requirement_t
loom_low_allocation_spill_register_requirement_for_value(
    const loom_module_t* module, const loom_region_t* function_region,
    loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) &&
      loom_value_def_block(value)->parent_region != function_region) {
    return LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NESTED_REGION_ARGUMENT;
  }
  const loom_op_t* defining_op = loom_def_op(value->def);
  if (defining_op && loom_low_reload_isa(defining_op)) {
    return LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC;
  }
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op && loom_low_spill_isa(user_op) &&
        loom_use_operand_index(uses[i]) == 0) {
      return LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC;
    }
  }
  return LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE;
}
