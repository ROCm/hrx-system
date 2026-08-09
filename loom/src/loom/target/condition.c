// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/condition.h"

#include "loom/ir/context.h"
#include "loom/util/bstring.h"

iree_status_t loom_target_condition_resolve(
    const loom_context_t* context, loom_attribute_t condition,
    const loom_target_condition_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (condition.kind != LOOM_ATTR_PARAMETERIZED) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target condition must be a parameterized "
                            "attribute");
  }

  const loom_parameterized_attr_kind_t family_kind =
      loom_attr_as_parameterized_kind(condition);
  const loom_parameterized_attr_descriptor_t* family =
      loom_context_resolve_parameterized_attr(context, family_kind);
  if (family == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target condition family kind 0x%04X is not "
                            "registered",
                            family_kind);
  }
  if (family->target_condition == NULL) {
    const iree_string_view_t family_name = loom_bstring_view(family->name);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family '%.*s' is not a target condition",
        (int)family_name.size, family_name.data);
  }

  const loom_target_condition_descriptor_t* descriptor =
      family->target_condition;
  if (descriptor->validate != NULL) {
    IREE_RETURN_IF_ERROR(descriptor->validate(condition));
  }
  *out_descriptor = descriptor;
  return iree_ok_status();
}
