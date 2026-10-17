// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/index.h>

static int index_sum(int begin, int end) {
  const auto lower = loom::index::cast(begin);
  const auto upper = loom::index::cast(end);
  auto total = loom::type::index{0};
  for (auto position = lower; position < upper;
       position = position + loom::type::index{1}) {
    total = total + position;
  }
  return loom::index::cast<int>(total);
}

static unsigned offset_sum(unsigned begin, unsigned end) {
  const auto lower = loom::offset::cast(begin);
  const auto upper = loom::offset::cast(end);
  auto total = loom::type::offset{0};
  for (auto position = lower; position < upper;
       position = loom::type::offset{1} + position) {
    total = total + position;
  }
  return loom::offset::cast<unsigned>(total);
}

LOOM_CHECK_CASE(address_counted_loops_functions) {
  const auto index_negative = index_sum(-3, 2);
  const auto index_equal = index_sum(11, 11);
  const auto index_reversed = index_sum(7, -2);
  const auto offset_positive = offset_sum(2, 6);
  const auto offset_equal = offset_sum(11, 11);
  const auto offset_reversed = offset_sum(7, 2);
  loom::check::expect_equal(index_negative, -5);
  loom::check::expect_equal(index_equal, 0);
  loom::check::expect_equal(index_reversed, 0);
  loom::check::expect_equal(offset_positive, 14u);
  loom::check::expect_equal(offset_equal, 0u);
  loom::check::expect_equal(offset_reversed, 0u);
}
