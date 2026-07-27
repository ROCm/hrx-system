// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/context/context.h"

#include "loom/ops/op_registry.h"
#include "loom/ops/test/registry.h"

iree_status_t loom_tooling_context_register_tool_dialects(
    loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  return loom_test_dialect_register(context);
}

iree_status_t
loom_tooling_context_register_tool_dialects_with_target_environment(
    const loom_target_environment_t* target_environment,
    loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_tooling_context_register_tool_dialects(context));
  if (target_environment == NULL) {
    return iree_ok_status();
  }
  return loom_target_environment_register_context(target_environment, context);
}
