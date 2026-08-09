// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/ops.h"
#include "loom/target/condition.h"

static iree_status_t loom_target_subgroup_size_condition_validate(
    loom_attribute_t condition) {
  const int64_t subgroup_size = loom_target_subgroup_size_attr_size(condition);
  if (subgroup_size > 0 && subgroup_size <= UINT32_MAX) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "target.subgroup.size requires a nonzero unsigned 32-bit size; got "
      "%" PRId64,
      subgroup_size);
}

static loom_target_condition_outcome_t
loom_target_subgroup_size_condition_evaluate(const loom_target_facts_t* facts,
                                             loom_attribute_t condition) {
  const uint32_t actual_size = facts->storage.bundle.snapshot->subgroup_size;
  if (actual_size == 0) return LOOM_TARGET_CONDITION_UNKNOWN;
  const uint32_t required_size =
      (uint32_t)loom_target_subgroup_size_attr_size(condition);
  return actual_size == required_size ? LOOM_TARGET_CONDITION_MATCH
                                      : LOOM_TARGET_CONDITION_REJECT;
}

const loom_target_condition_descriptor_t loom_target_subgroup_size_condition = {
    .validate = loom_target_subgroup_size_condition_validate,
    .evaluate = loom_target_subgroup_size_condition_evaluate,
};
