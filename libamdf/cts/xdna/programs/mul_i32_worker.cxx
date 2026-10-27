// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/index.h>
#include <loomcxx/view.h>

namespace loomt = loom::type;
using input_record = loomt::view<loomt::shape<16>, const int>;
using output_record = loomt::view<loomt::shape<16>, int>;

[[loom::symbol("multiply_record"), loom::force_inline]] void multiply_record(
    input_record lhs, input_record rhs, output_record output) {
  for (unsigned i = 0; i < 16; ++i) {
    const auto index = loom::index::cast(i);
    loom::view::store(
        loom::view::load(lhs, index) * loom::view::load(rhs, index), output,
        index);
  }
}
