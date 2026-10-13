// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/sources.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SourcesTest, RelativePathsRequireABase) {
  EXPECT_FALSE(loom_serve_path_is_absolute(IREE_SV("")));
  EXPECT_FALSE(loom_serve_path_is_absolute(IREE_SV("model/prepare.loombc")));
  char* path = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_path_resolve({}, IREE_SV("prepare.loombc"),
                              iree_allocator_system(), &path));
  EXPECT_EQ(path, nullptr);
  IREE_ASSERT_OK(loom_serve_path_resolve(IREE_SV("models/local"),
                                         IREE_SV("../shared/kernel.loom"),
                                         iree_allocator_system(), &path));
  EXPECT_STREQ(path, "models/local/../shared/kernel.loom");
  iree_allocator_free(iree_allocator_system(), path);
}

TEST(SourcesTest, AbsolutePathsIgnoreBaseAndRemainAuthored) {
  // A share path is absolute in both supported local filesystem namespaces.
  const auto locator = IREE_SV("//files/models/model.loombc");
  EXPECT_TRUE(loom_serve_path_is_absolute(locator));
  for (const auto base : {iree_string_view_empty(), IREE_SV("ignored")}) {
    char* path = nullptr;
    IREE_ASSERT_OK(
        loom_serve_path_resolve(base, locator, iree_allocator_system(), &path));
    EXPECT_STREQ(path, "//files/models/model.loombc");
    EXPECT_NE(path, locator.data);
    iree_allocator_free(iree_allocator_system(), path);
  }
}

}  // namespace
