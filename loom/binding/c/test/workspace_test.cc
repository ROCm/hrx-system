// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/workspace.h"

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

  loomc_workspace_retain(workspace);
  loomc_workspace_trim(workspace);
  loomc_workspace_release(workspace);
  loomc_workspace_release(workspace);
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

}  // namespace
