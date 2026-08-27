// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/workspace.h"

#include <limits>

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

TEST(WorkspaceTest, CreateTrimRetainRelease) {
  loomc_workspace_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.block_size=*/4096,
  };
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(&options, loomc_allocator_system(), &workspace);
  LOOMC_ASSERT_OK(status);
  ASSERT_NE(workspace, nullptr);

  loomc_workspace_statistics_t statistics;
  loomc_workspace_query_statistics(workspace, &statistics);
  EXPECT_EQ(statistics.total_block_size, 4096u);
  EXPECT_LT(statistics.usable_block_size, statistics.total_block_size);
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, 0u);

  loomc_workspace_retain(workspace);
  loomc_workspace_trim(workspace);
  loomc_workspace_release(workspace);
  loomc_workspace_release(workspace);
}

TEST(WorkspaceTest, QueryNullWorkspace) {
  loomc_workspace_statistics_t statistics = {
      /*.total_block_size=*/1,
  };
  loomc_workspace_query_statistics(nullptr, &statistics);
  EXPECT_EQ(statistics.total_block_size, 0u);
  loomc_workspace_query_statistics(nullptr, nullptr);
}

TEST(WorkspaceTest, RejectsInvalidExtensions) {
  int extension = 0;
  loomc_workspace_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&extension,
  };
  loomc_workspace_t* workspace = reinterpret_cast<loomc_workspace_t*>(0x1);
  loomc_status_t status =
      loomc_workspace_create(&options, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(workspace, nullptr);
}

TEST(WorkspaceTest, RejectsInvalidBlockSizes) {
  loomc_workspace_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.block_size=*/1,
  };
  loomc_workspace_t* workspace = reinterpret_cast<loomc_workspace_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_workspace_create(&options, loomc_allocator_system(), &workspace));
  EXPECT_EQ(workspace, nullptr);

  options.block_size = std::numeric_limits<loomc_host_size_t>::max();
  workspace = reinterpret_cast<loomc_workspace_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_OUT_OF_RANGE,
      loomc_workspace_create(&options, loomc_allocator_system(), &workspace));
  EXPECT_EQ(workspace, nullptr);
}

}  // namespace
