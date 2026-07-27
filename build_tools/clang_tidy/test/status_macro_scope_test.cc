// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define IREE_EXPECT_OK(rexpr) ((void)(rexpr))
#define IREE_ASSERT_OK(rexpr) ((void)(rexpr))
#define IREE_EXPECT_STATUS_IS(expected_code, expr) ((void)(expr))

void iree_clang_tidy_style_test_status_macros_are_allowed_in_tests() {
  IREE_ASSERT_OK(0);
  IREE_EXPECT_OK(0);
  IREE_EXPECT_STATUS_IS(0, 0);
}
