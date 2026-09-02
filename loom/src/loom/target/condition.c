// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/condition.h"

#include "loom/ir/context.h"

const loom_target_condition_descriptor_t* loom_target_condition_resolve(
    const loom_context_t* context, loom_attribute_t condition) {
  const loom_parameterized_attr_kind_t family_kind =
      loom_attr_as_parameterized_kind(condition);
  const loom_parameterized_attr_descriptor_t* family =
      loom_context_resolve_parameterized_attr(context, family_kind);
  return family->target_condition;
}

iree_string_view_t loom_target_condition_validate(
    const loom_target_condition_descriptor_t* descriptor,
    loom_attribute_t condition) {
  return descriptor->validate != NULL ? descriptor->validate(condition)
                                      : iree_string_view_empty();
}
