// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <model_provider.h>

#ifdef MODEL_LOCAL_BIAS
#error "Private definitions must not propagate to consumers."
#endif

static_assert(sizeof(MODEL_TEXT) == 18);

LOOM_CHECK_CASE(imported_bias) {
  loom::check::expect_equal(model_add_bias(35), 42);
}
