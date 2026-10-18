// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/output.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class OutputTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    loom_serve_realtime_conversation_initialize(7, iree_allocator_system(),
                                                &conversation_);
    IREE_ASSERT_OK(loom_serve_realtime_conversation_begin(
        &conversation_, std::get<1>(GetParam()), &group_));
    for (int i = 0; i < 2; ++i) {
      loom_serve_realtime_item_t* item = nullptr;
      IREE_ASSERT_OK(loom_serve_realtime_conversation_add(
          &conversation_, group_, LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL,
          LOOM_SERVE_REALTIME_ASSISTANT, false, &item));
      IREE_ASSERT_OK(loom_serve_realtime_conversation_function(
          &conversation_, item, iree_string_view_empty(), IREE_SV("lookup"),
          IREE_SV("{\"value\":\"")));
    }
    view_ = {
        .id = IREE_SV("resp_7_1"),
        .conversation_id = std::get<1>(GetParam()) ? 0u : 7u,
        .first = group_->first,
        .last = group_->last,
        .metadata = IREE_SV("{\"purpose\":\"lookup\"}"),
        .audio = true,
        .state = LOOM_SERVE_REALTIME_OUTPUT_COMPLETED,
        .input_tokens = 10,
        .cached_tokens = 2,
        .output_tokens = 20,
    };
  }
  void TearDown() override {
    loom_serve_realtime_conversation_deinitialize(&conversation_);
  }
  iree_host_size_t Measure(loom_serve_realtime_output_state_t state) {
    auto view = view_;
    view.state = state;
    loom_output_stream_t output;
    loom_output_stream_null(&output);
    IREE_EXPECT_OK(loom_serve_realtime_wire_prefix(7, UINT64_MAX,
                                                   "response.done", &output));
    IREE_EXPECT_OK(loom_serve_realtime_output_write(&view, &output));
    return output.offset;
  }
  void Append(loom_serve_realtime_item_t* item, const std::string& value) {
    IREE_ASSERT_OK(loom_serve_realtime_conversation_append_text(
        &conversation_, item,
        iree_make_string_view(value.data(), value.size())));
  }

  // Actual canonical storage and read-only admission view.
  loom_serve_realtime_conversation_t conversation_ = {};
  // One retained response with two complete native call payloads.
  loom_serve_realtime_group_t* group_ = nullptr;
  // Real publication serializer input; arguments are filled before admission.
  loom_serve_realtime_output_t view_ = {};
};

TEST_P(OutputTest, AggregateEscapedBytesEstablishTheExecutableBoundary) {
  // Both call objects are independently small enough; the final aggregate is
  // the binding limit. Arguments remain valid JSON, including escaped slashes.
  const std::string unit = std::get<0>(GetParam()) ? "\\\\" : "a";
  const size_t encoded_width = std::get<0>(GetParam()) ? 4 : 1;
  Append(group_->last, std::string(128 * 1024, 'a') + "\"}");
  const size_t base =
      std::max({Measure(LOOM_SERVE_REALTIME_OUTPUT_COMPLETED),
                Measure(LOOM_SERVE_REALTIME_OUTPUT_CLIENT_CANCELLED),
                Measure(LOOM_SERVE_REALTIME_OUTPUT_TURN_DETECTED)});
  // Closing the argument string/object adds three wire bytes: an escaped quote
  // and the closing brace.
  const size_t closing_width = 3;
  const size_t room = 512 * 1024 - base - closing_width;
  std::string payload;
  for (size_t i = 0; i < room / encoded_width; ++i) {
    payload += unit;
  }
  payload.append(room % encoded_width, 'a');
  payload += "\"}";
  Append(group_->first, payload);
  EXPECT_EQ(std::max({Measure(LOOM_SERVE_REALTIME_OUTPUT_COMPLETED),
                      Measure(LOOM_SERVE_REALTIME_OUTPUT_CLIENT_CANCELLED),
                      Measure(LOOM_SERVE_REALTIME_OUTPUT_TURN_DETECTED)}),
            512 * 1024);
  IREE_ASSERT_OK(loom_serve_realtime_output_admit(7, &view_));
  // Insert one valid character before the argument's closing quote. The real
  // serializer now exceeds admission by precisely one byte.
  auto* text = &group_->first->text;
  text->buffer[text->size - 2] = 'b';
  --text->size;
  --conversation_.bytes;
  Append(group_->first, "\"}");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_realtime_output_admit(7, &view_));
  EXPECT_EQ(group_->completed_calls, 0);
}

INSTANTIATE_TEST_SUITE_P(Arguments, OutputTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
