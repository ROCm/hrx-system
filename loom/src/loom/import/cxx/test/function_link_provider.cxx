// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "function_link_helpers.h"

static unsigned local_adjust(unsigned value) { return value + 5u; }

[[loom::symbol("library.adjust")]] unsigned adjust(unsigned value) {
  return shared_inline(value) + shared_template<7u>(value) +
         local_adjust(value);
}
