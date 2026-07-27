// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_vector.h"

#include "loom/analysis/contract_vector.h"
#include "loom/target/arch/x86/packed_dot_contract_projection.h"

bool loom_x86_packed_dot_match_request_from_vector_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_x86_packed_dot_match_request_t* out_request) {
  if (out_request == NULL) {
    return false;
  }
  *out_request = (loom_x86_packed_dot_match_request_t){0};
  if (module == NULL || op == NULL) {
    return false;
  }

  loom_contract_request_t contract_request = {0};
  if (!loom_contract_request_from_vector_dot_op(
          module, op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
          &contract_request, NULL)) {
    return false;
  }
  return loom_x86_packed_dot_match_request_from_contract(&contract_request, 0,
                                                         out_request, NULL);
}
