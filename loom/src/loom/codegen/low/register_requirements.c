// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/register_requirements.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"

static loom_low_register_requirement_t loom_low_register_requirement_classify(
    const loom_module_t* module, const loom_region_t* function_region,
    loom_value_id_t value_id) {
  if (!loom_type_is_register(loom_module_value_type(module, value_id))) {
    return LOOM_LOW_REGISTER_REQUIREMENT_NONE;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    if (loom_value_def_block(value)->parent_region != function_region) {
      return LOOM_LOW_REGISTER_REQUIREMENT_NESTED_REGION_ARGUMENT;
    }
  } else {
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (defining_op && loom_low_storage_load_isa(defining_op)) {
      return LOOM_LOW_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC;
    }
  }
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op && loom_low_storage_store_isa(user_op) &&
        loom_use_operand_index(uses[i]) == 0) {
      return LOOM_LOW_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC;
    }
  }
  return value->use_count == 0 ? LOOM_LOW_REGISTER_REQUIREMENT_NO_OPERAND_USES
                               : LOOM_LOW_REGISTER_REQUIREMENT_NONE;
}

iree_status_t loom_low_register_requirements_build(
    const loom_local_value_domain_t* value_domain,
    const loom_region_t* function_region, iree_arena_allocator_t* arena,
    loom_low_register_requirement_t** out_requirements) {
  *out_requirements = NULL;
  for (loom_value_ordinal_t ordinal = 0; ordinal < value_domain->value_count;
       ++ordinal) {
    const loom_low_register_requirement_t requirement =
        loom_low_register_requirement_classify(
            value_domain->module, function_region,
            value_domain->value_ids[ordinal]);
    if (requirement == LOOM_LOW_REGISTER_REQUIREMENT_NONE) {
      continue;
    }
    if (*out_requirements == NULL) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, value_domain->value_count, sizeof(**out_requirements),
          (void**)out_requirements));
      memset(*out_requirements, 0,
             value_domain->value_count * sizeof(**out_requirements));
    }
    (*out_requirements)[ordinal] = requirement;
  }
  return iree_ok_status();
}
