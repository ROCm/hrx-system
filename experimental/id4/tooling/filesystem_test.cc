// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/filesystem.h"

#include <fstream>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

TEST(FilesystemTest, EnsuresDirectoryAndFormatsChildPath) {
  iree::testing::TempFilePath directory("id4_tooling_filesystem");
  IREE_ASSERT_OK(id4_tooling_ensure_directory(directory.path_view(),
                                              iree_allocator_system()));
  EXPECT_TRUE(directory.Exists());

  iree_string_view_t child_path = iree_string_view_empty();
  IREE_ASSERT_OK(
      id4_tooling_format_child_path(directory.path_view(), IREE_SV("plan.json"),
                                    iree_allocator_system(), &child_path));
  std::string expected = directory.path() + "/plan.json";
  EXPECT_EQ(std::string(child_path.data, child_path.size), expected);
  id4_tooling_free_path(&child_path, iree_allocator_system());
}

TEST(FilesystemTest, RejectsExistingNonDirectory) {
  iree::testing::TempFilePath file_path("id4_tooling_filesystem_file");
  std::ofstream file(file_path.path());
  file << "not a directory";
  file.close();

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        id4_tooling_ensure_directory(file_path.path_view(),
                                                     iree_allocator_system()));
}

}  // namespace
