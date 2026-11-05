// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/execution.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ExecutionTest, NullLifetimeOperations) {
  loom_serve_execution_retain(nullptr);
  loom_serve_execution_release(nullptr);
  IREE_EXPECT_OK(loom_serve_execution_drain_all(0, nullptr));
  loom_serve_execution_t* partial[] = {nullptr, nullptr};
  IREE_EXPECT_OK(
      loom_serve_execution_drain_all(IREE_ARRAYSIZE(partial), partial));
}

}  // namespace
