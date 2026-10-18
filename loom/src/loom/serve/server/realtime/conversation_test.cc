// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/conversation.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ConversationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_realtime_conversation_initialize(7, iree_allocator_system(),
                                                &conversation_);
  }
  void TearDown() override {
    loom_serve_realtime_conversation_deinitialize(&conversation_);
  }

  iree_status_t Message(iree_string_view_t id, iree_string_view_t text,
                        loom_serve_realtime_item_t** out_item) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_reserve(
        &conversation_, id, LOOM_SERVE_REALTIME_USER, false, out_item));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_append_text(
        &conversation_, *out_item, text));
    (*out_item)->status = LOOM_SERVE_REALTIME_ITEM_COMPLETED;
    return loom_serve_realtime_conversation_insert(
        &conversation_, conversation_.last, *out_item);
  }

  std::string Request(loom_serve_realtime_group_t* exclude = nullptr) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    IREE_EXPECT_OK(loom_serve_realtime_conversation_request(
        &conversation_, exclude, IREE_SV("model"), iree_string_view_empty(), 32,
        42, &output));
    std::string value(output.buffer, output.size);
    iree_string_builder_deinitialize(&output);
    return value;
  }

  // Actual identity and canonical-history owner, without native model state.
  loom_serve_realtime_conversation_t conversation_;
};

TEST_F(ConversationTest, ReservationOwnsIdentityBeforeCommit) {
  loom_serve_realtime_item_t* microphone = nullptr;
  IREE_ASSERT_OK(loom_serve_realtime_conversation_reserve(
      &conversation_, iree_string_view_empty(), LOOM_SERVE_REALTIME_USER, true,
      &microphone));
  const std::string id(microphone->id.buffer, microphone->id.size);
  EXPECT_EQ(loom_serve_realtime_conversation_find(
                &conversation_, iree_make_cstring_view(id.c_str())),
            nullptr);
  loom_serve_realtime_item_t* collision = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_realtime_conversation_reserve(
                            &conversation_, iree_make_cstring_view(id.c_str()),
                            LOOM_SERVE_REALTIME_USER, false, &collision));
  EXPECT_EQ(collision, nullptr);
  IREE_ASSERT_OK(loom_serve_realtime_conversation_insert(
      &conversation_, conversation_.last, microphone));
  EXPECT_EQ(loom_serve_realtime_conversation_find(
                &conversation_, iree_make_cstring_view(id.c_str())),
            microphone);
  loom_serve_realtime_conversation_remove(&conversation_, microphone);
  IREE_ASSERT_OK(
      Message(iree_make_cstring_view(id.c_str()), IREE_SV("a"), &collision));
}

TEST_F(ConversationTest, GeneratedIdentitySkipsAcceptedClientNames) {
  loom_serve_realtime_item_t* client = nullptr;
  IREE_ASSERT_OK(Message(IREE_SV("item_7_1"), IREE_SV("a"), &client));
  loom_serve_realtime_item_t* generated = nullptr;
  IREE_ASSERT_OK(Message(iree_string_view_empty(), IREE_SV("b"), &generated));
  EXPECT_EQ(std::string(generated->id.buffer, generated->id.size), "item_7_2");
  loom_serve_realtime_conversation_remove(&conversation_, generated);
  IREE_ASSERT_OK(Message(iree_string_view_empty(), IREE_SV("x"), &generated));
  EXPECT_EQ(std::string(generated->id.buffer, generated->id.size), "item_7_3");
}

TEST_F(ConversationTest, PendingResponseKeepsItsCanonicalInputFrontier) {
  loom_serve_realtime_item_t* before = nullptr;
  IREE_ASSERT_OK(Message(IREE_SV("before"), IREE_SV("a"), &before));
  loom_serve_realtime_group_t* response = nullptr;
  IREE_ASSERT_OK(
      loom_serve_realtime_conversation_begin(&conversation_, &response));
  loom_serve_realtime_item_t* after = nullptr;
  IREE_ASSERT_OK(Message(IREE_SV("after"), IREE_SV("x"), &after));
  EXPECT_EQ(
      Request(response),
      R"({"model":"model","stream":true,"max_tokens":32,"seed":42,"messages":[{"role":"user","content":"a"}]})");
  loom_serve_realtime_item_t* output = nullptr;
  IREE_ASSERT_OK(loom_serve_realtime_conversation_add(
      &conversation_, response, LOOM_SERVE_REALTIME_ASSISTANT, true, &output));
  IREE_ASSERT_OK(loom_serve_realtime_conversation_append_text(
      &conversation_, output, IREE_SV("bb")));
  output->status = LOOM_SERVE_REALTIME_ITEM_COMPLETED;
  EXPECT_EQ(loom_serve_realtime_conversation_previous(response), before);
  EXPECT_EQ(loom_serve_realtime_conversation_previous(after->group), output);
  loom_serve_realtime_conversation_release(&conversation_, response);
  EXPECT_EQ(
      Request(),
      R"({"model":"model","stream":true,"max_tokens":32,"seed":42,"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"bb"},{"role":"user","content":"x"}]})");
}

TEST_F(ConversationTest, EmptyResponseRetainsCapacityUntilRelease) {
  loom_serve_realtime_group_t* response = nullptr;
  IREE_ASSERT_OK(
      loom_serve_realtime_conversation_begin(&conversation_, &response));
  for (iree_host_size_t i = 0; i < 127; ++i) {
    loom_serve_realtime_item_t* item = nullptr;
    IREE_ASSERT_OK(loom_serve_realtime_conversation_reserve(
        &conversation_, iree_string_view_empty(), LOOM_SERVE_REALTIME_USER,
        true, &item));
  }
  loom_serve_realtime_item_t* item = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_realtime_conversation_reserve(
                            &conversation_, iree_string_view_empty(),
                            LOOM_SERVE_REALTIME_USER, true, &item));
  IREE_ASSERT_OK(loom_serve_realtime_conversation_add(
      &conversation_, response, LOOM_SERVE_REALTIME_ASSISTANT, true, &item));
  loom_serve_realtime_conversation_remove(&conversation_, item);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_realtime_conversation_reserve(
                            &conversation_, iree_string_view_empty(),
                            LOOM_SERVE_REALTIME_USER, true, &item));
  loom_serve_realtime_conversation_release(&conversation_, response);
  IREE_ASSERT_OK(loom_serve_realtime_conversation_reserve(
      &conversation_, iree_string_view_empty(), LOOM_SERVE_REALTIME_USER, true,
      &item));
}

TEST_F(ConversationTest, CancelledEmptyFrontiersLeaveNoHistoryItems) {
  loom_serve_realtime_group_t* first = nullptr;
  loom_serve_realtime_group_t* second = nullptr;
  IREE_ASSERT_OK(
      loom_serve_realtime_conversation_begin(&conversation_, &first));
  IREE_ASSERT_OK(
      loom_serve_realtime_conversation_begin(&conversation_, &second));
  loom_serve_realtime_item_t* after = nullptr;
  IREE_ASSERT_OK(Message(IREE_SV("after"), IREE_SV("a"), &after));
  loom_serve_realtime_conversation_release(&conversation_, first);
  EXPECT_EQ(loom_serve_realtime_conversation_previous(after->group), nullptr);
  EXPECT_EQ(
      Request(second),
      R"({"model":"model","stream":true,"max_tokens":32,"seed":42,"messages":[]})");
  loom_serve_realtime_conversation_release(&conversation_, second);
  EXPECT_EQ(
      Request(),
      R"({"model":"model","stream":true,"max_tokens":32,"seed":42,"messages":[{"role":"user","content":"a"}]})");
}

}  // namespace
