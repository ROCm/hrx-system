// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/sources.h"
#include "iree/testing/gtest.h"

namespace {

TEST(SourcesPosixTest, RootedAndForeignNames) {
  EXPECT_TRUE(loom_serve_path_is_absolute(IREE_SV("/")));
  EXPECT_TRUE(loom_serve_path_is_absolute(IREE_SV("/models/kernel.loombc")));
  EXPECT_FALSE(loom_serve_path_is_absolute(IREE_SV("C:/models/kernel.loombc")));
  EXPECT_FALSE(loom_serve_path_is_absolute(IREE_SV(R"(\models\kernel.loom)")));
}

}  // namespace
