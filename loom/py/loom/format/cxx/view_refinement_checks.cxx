// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

LOOM_CHECK_CASE(refinement_retains_loop_carried_layout) {
  const auto dense = retained_layout(4u, 0u);
  const auto strided = retained_layout(6u, 0u);
  const auto swapped = retained_layout(6u, 1u);
  const auto restored = retained_layout(6u, 2u);
  loom::check::expect_equal(dense, 21u);
  loom::check::expect_equal(strided, 25u);
  loom::check::expect_equal(swapped, 111u);
  loom::check::expect_equal(restored, 25u);
}

LOOM_CHECK_CASE(refinement_retains_borrowed_storage) {
  const auto stored = retained_storage(173u);
  const auto encoded = explicit_encoding(919u);
  const auto volume = rank_three(53u);
  loom::check::expect_equal(stored, 173u);
  loom::check::expect_equal(encoded, 919u);
  loom::check::expect_equal(volume, 53u);
}
