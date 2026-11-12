// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

LOOM_CHECK_CASE(explicit_staging_shape) {
  const auto first = staging(11u);
  const auto second = staging(100u);
  const auto sliced = slice_rows(100u);
  loom::check::expect_equal(first, 29u);
  loom::check::expect_equal(second, 207u);
  loom::check::expect_equal(sliced, 215u);
}

LOOM_CHECK_CASE(native_and_semantic_carriers) {
  const auto native = native_carrier(11u);
  const auto volume = overflow_rank(100u);
  loom::check::expect_equal(native, 29u);
  loom::check::expect_equal(volume, 207u);
}

LOOM_CHECK_CASE(dependent_vector_loop_transport) {
  const auto empty = loop_transport(0u, 11u);
  const auto swapped = loop_transport(1u, 11u);
  const auto restored = loop_transport(2u, 11u);
  loom::check::expect_equal(empty, 22u);
  loom::check::expect_equal(swapped, 28u);
  loom::check::expect_equal(restored, 22u);
}
