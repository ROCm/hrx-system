// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/module_format.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(ModuleFormatTest, ParsesAcceptedSpellings) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_AUTO;

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("auto"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_AUTO);

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("text"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_TEXT);

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("bc"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_BYTECODE);

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("bytecode"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_BYTECODE);
}

TEST(ModuleFormatTest, RejectsAutoWhenDisallowed) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_TEXT;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_format_parse(IREE_SV("auto"), false, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_TEXT);
}

}  // namespace
}  // namespace loom
