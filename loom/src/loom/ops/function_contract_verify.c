// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"

iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  // Targetless functions are valid generic program representations. A compile
  // invocation may bind an exact target later, so source verification cannot
  // require an authored target attribute.
  (void)module;
  (void)op;
  (void)emitter;
  return iree_ok_status();
}
