// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/sanitizer/ops.h"
#include "loom/rewrite/type_propagation.h"

iree_status_t loom_sanitizer_assert_layout_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op) {
  (void)module;
  loom_value_id_t source = loom_sanitizer_assert_layout_view(op);
  loom_value_id_t result = loom_sanitizer_assert_layout_result(op);
  loom_type_t source_type = loom_type_transfer_value_type(context, source);

  return loom_type_transfer_seed_static_structure_from_type(context, result,
                                                            source_type);
}
