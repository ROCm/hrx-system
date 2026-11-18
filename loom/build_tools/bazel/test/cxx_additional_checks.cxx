// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <model_provider.h>

// Both translation units consume the same dependency's exported interface.
LOOM_CHECK_CASE(imported_negative_bias) {
  loom::check::expect_equal(model_add_bias(-7), 0);
}
