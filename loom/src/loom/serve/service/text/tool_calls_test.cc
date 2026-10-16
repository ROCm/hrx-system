// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/tool_calls.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string String(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class ToolCallsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls_);
    iree_string_builder_initialize(iree_allocator_system(), &output_);
  }
  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    loom_serve_text_tool_calls_deinitialize(&calls_);
  }

  // Owned records, independent of the caller's input strings.
  loom_serve_text_tool_calls_t calls_;
  // Encoded HTTP function records.
  iree_string_builder_t output_;
};

TEST_F(ToolCallsTest, EmptyOutputNeedsNoRecordStorage) {
  loom_serve_text_tool_calls_reset(&calls_);
  IREE_ASSERT_OK(
      loom_serve_text_tool_calls_write_json(&calls_, false, &output_));
  EXPECT_EQ(String(iree_string_builder_view(&output_)), "[]");
  EXPECT_EQ(calls_.count, 0u);
  EXPECT_EQ(calls_.ranges, nullptr);
  EXPECT_EQ(calls_.storage.buffer, nullptr);
}

TEST_F(ToolCallsTest, OwnedValuesAndIdsSurviveInputReuseAndMove) {
  std::string name = "read";
  std::string arguments = R"({"path":"λ\n\"a\""})";
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
      &calls_, 73, iree_make_string_view(name.data(), name.size()),
      iree_make_string_view(arguments.data(), arguments.size())));
  name.assign("changed");
  arguments.assign("{}");
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(&calls_, 73, IREE_SV("list"),
                                                   IREE_SV("{}")));
  auto moved = calls_;
  loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls_);
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
      &calls_, 91, IREE_SV("replace"), IREE_SV("{}")));
  const auto first = loom_serve_text_tool_calls_at(&moved, 0);
  EXPECT_EQ(String(first.id), "call_73_0");
  EXPECT_EQ(String(first.name), "read");
  EXPECT_EQ(String(first.arguments), R"({"path":"λ\n\"a\""})");
  EXPECT_EQ(String(loom_serve_text_tool_calls_at(&moved, 1).id), "call_73_1");
  IREE_ASSERT_OK(
      loom_serve_text_tool_calls_write_json(&moved, false, &output_));
  EXPECT_EQ(
      String(iree_string_builder_view(&output_)),
      R"([{"id":"call_73_0","type":"function","function":{"name":"read","arguments":"{\"path\":\"λ\\n\\\"a\\\"\"}"}},{"id":"call_73_1","type":"function","function":{"name":"list","arguments":"{}"}}])");
  iree_string_builder_reset(&output_);
  IREE_ASSERT_OK(loom_serve_text_tool_calls_write_json(&moved, true, &output_));
  EXPECT_EQ(
      String(iree_string_builder_view(&output_)),
      R"([{"index":0,"id":"call_73_0","type":"function","function":{"name":"read","arguments":"{\"path\":\"λ\\n\\\"a\\\"\"}"}},{"index":1,"id":"call_73_1","type":"function","function":{"name":"list","arguments":"{}"}}])");
  loom_serve_text_tool_calls_deinitialize(&moved);
}

TEST_F(ToolCallsTest, RangesSurviveGrowthAndScratchReuse) {
  for (size_t i = 0; i < 9; ++i) {
    const std::string name = "tool_" + std::to_string(i);
    const std::string arguments =
        "{\"text\":\"" + std::string(i * 64, 'x') + "\"}";
    IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
        &calls_, UINT64_MAX, iree_make_string_view(name.data(), name.size()),
        iree_make_string_view(arguments.data(), arguments.size())));
  }
  EXPECT_EQ(calls_.count, 9u);
  for (size_t i = 0; i < calls_.count; ++i) {
    const auto call = loom_serve_text_tool_calls_at(&calls_, i);
    EXPECT_EQ(String(call.id),
              "call_18446744073709551615_" + std::to_string(i));
    EXPECT_EQ(String(call.name), "tool_" + std::to_string(i));
    EXPECT_EQ(String(call.arguments),
              "{\"text\":\"" + std::string(i * 64, 'x') + "\"}");
  }
  const auto* ranges = calls_.ranges;
  const char* storage = calls_.storage.buffer;
  loom_serve_text_tool_calls_reset(&calls_);
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(&calls_, 1, IREE_SV("next"),
                                                   IREE_SV("{}")));
  EXPECT_EQ(calls_.ranges, ranges);
  EXPECT_EQ(calls_.storage.buffer, storage);
  EXPECT_EQ(calls_.count, 1u);
  EXPECT_EQ(String(loom_serve_text_tool_calls_at(&calls_, 0).id), "call_1_0");
}

}  // namespace
