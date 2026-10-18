// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

#include "function_link_helpers.h"

[[loom::symbol("library.adjust")]] unsigned adjust(unsigned value);

static unsigned local_adjust(unsigned value) { return value + 11u; }

unsigned apply_adjustment(unsigned value) {
  return adjust(value) + shared_inline(value) + shared_template<7u>(value) +
         local_adjust(value);
}

LOOM_CHECK_CASE(function_link) {
  loom::check::expect_equal(apply_adjustment(2u), 56u);
}
