// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/tool/process.h"

#include <string_view>

#include "iree/testing/gtest.h"

namespace {

TEST(ToolOutputTest, NormalizesCrLfNewlines) {
  char text[] = "first\r\nsecond\r\r\nthird\nfourth\rfifth\r\rtext";
  loom_tool_output_t output = {text, sizeof(text) - 1};

  loom_tool_output_normalize_newlines(&output);

  EXPECT_EQ(std::string_view(output.data, output.length),
            "first\nsecond\nthird\nfourth\rfifth\r\rtext");
  EXPECT_EQ(output.data[output.length], '\0');
}

TEST(ToolOutputTest, AcceptsEmptyOutput) {
  loom_tool_output_t output = {0};

  loom_tool_output_normalize_newlines(&output);

  EXPECT_EQ(output.data, nullptr);
  EXPECT_EQ(output.length, 0u);
}

}  // namespace
