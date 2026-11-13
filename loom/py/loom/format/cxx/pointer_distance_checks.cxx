// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

LOOM_CHECK_CASE(signed_endpoints) {
  const auto forward = endpoint_distance(0u, 8u);
  const auto reverse = endpoint_distance(8u, 0u);
  const auto interior = endpoint_distance(5u, 2u);
  const auto equal = endpoint_distance(3u, 3u);
  loom::check::expect_equal(forward, static_cast<decltype(forward)>(8ll));
  loom::check::expect_equal(reverse, static_cast<decltype(reverse)>(-8ll));
  loom::check::expect_equal(interior, static_cast<decltype(interior)>(-3ll));
  loom::check::expect_equal(equal, static_cast<decltype(equal)>(0ll));
}

LOOM_CHECK_CASE(iterator_roots) {
  const auto walk = walked_distance(8u);
  const auto stationary = walked_distance(0u);
  const auto list = list_distance(7);
  const auto empty = empty_distance();
  loom::check::expect_equal(walk, static_cast<decltype(walk)>(-8ll));
  loom::check::expect_equal(stationary, static_cast<decltype(stationary)>(0ll));
  loom::check::expect_equal(list, static_cast<decltype(list)>(-3ll));
  loom::check::expect_equal(empty, static_cast<decltype(empty)>(0ll));
}
