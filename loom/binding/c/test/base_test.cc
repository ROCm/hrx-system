// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/base.h"

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

TEST(BaseTest, SystemAllocatorAllocatesAndFrees) {
  loomc_allocator_t allocator = loomc_allocator_system();
  EXPECT_TRUE(loomc_allocator_is_valid(allocator));

  void* ptr = nullptr;
  LOOMC_EXPECT_OK(loomc_allocator_malloc(allocator, 16, &ptr));
  ASSERT_NE(ptr, nullptr);
  loomc_allocator_free(allocator, ptr);
}

TEST(BaseTest, ZeroAllocatorIsInvalid) {
  loomc_allocator_t allocator = {};
  EXPECT_FALSE(loomc_allocator_is_valid(allocator));

  void* ptr = nullptr;
  loomc_status_t status = loomc_allocator_malloc(allocator, 16, &ptr);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(ptr, nullptr);
}

}  // namespace
