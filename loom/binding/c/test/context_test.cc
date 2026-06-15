// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/context.h"

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

TEST(ContextTest, CreatesRetainsAndReleases) {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_ASSERT_OK(status);
  ASSERT_NE(context, nullptr);

  loomc_context_retain(context);
  loomc_context_release(context);
  loomc_context_release(context);
}

TEST(ContextTest, RejectsUnknownOptions) {
  loomc_context_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
  };
  loomc_context_t* context = reinterpret_cast<loomc_context_t*>(0x1);
  loomc_status_t status =
      loomc_context_create(&options, loomc_allocator_system(), &context);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(context, nullptr);
}

}  // namespace
