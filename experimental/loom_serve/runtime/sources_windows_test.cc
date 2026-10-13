// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "experimental/loom_serve/runtime/sources.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SourcesWindowsTest, FullyQualifiedPaths) {
  for (auto path :
       {IREE_SV("C:/models/kernel.loombc"), IREE_SV(R"(d:\models\kernel.loom)"),
        IREE_SV(R"(\\host\share\kernel.loom)"),
        IREE_SV(R"(\\?\C:\models\kernel.loom)")}) {
    EXPECT_TRUE(loom_serve_path_is_absolute(path));
    char* resolved = nullptr;
    IREE_ASSERT_OK(loom_serve_path_resolve(IREE_SV("ignored"), path,
                                           iree_allocator_system(), &resolved));
    EXPECT_EQ(std::string(resolved), std::string(path.data, path.size));
    iree_allocator_free(iree_allocator_system(), resolved);
  }
}

TEST(SourcesWindowsTest, DriveRelativeAndRootRelativeAreNotFullyQualified) {
  for (auto path :
       {IREE_SV("C:kernel.loom"), IREE_SV("/models"), IREE_SV(R"(\models)")}) {
    EXPECT_FALSE(loom_serve_path_is_absolute(path));
    char* resolved = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_path_resolve(IREE_SV("C:/base"), path,
                                iree_allocator_system(), &resolved));
    EXPECT_EQ(resolved, nullptr);
  }
}

}  // namespace
