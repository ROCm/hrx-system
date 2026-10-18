// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/latency.h"

// Each loaded copy embeds the real static runtime, including its local state.
__declspec(dllexport) bool iree_latency_test_begin(
    iree_latency_scope_t* out_scope) {
  return iree_latency_scope_begin(iree_latency_scope_options_default(),
                                  out_scope);
}

__declspec(dllexport) iree_status_code_t
iree_latency_test_end(iree_latency_scope_t* scope) {
  return iree_status_consume_code(iree_latency_scope_end(scope));
}
