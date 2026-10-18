// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/checkpoint_control.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/service/text/telemetry.h"

namespace {

TEST(TextCheckpointRequestTest, ValidatesOperationAndIdentifierFamily) {
  const loom_serve_text_checkpoint_request_t valid[] = {
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, IREE_SV("a-Z_0.9"),
       IREE_SV("conversation")},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND, IREE_SV("retained"), {}},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE, IREE_SV("retained"), {}},
  };
  for (const auto& request : valid) {
    IREE_EXPECT_OK(loom_serve_text_checkpoint_request_validate(&request));
  }
  const loom_serve_text_checkpoint_request_t invalid[] = {
      {static_cast<loom_serve_text_checkpoint_action_t>(3),
       IREE_SV("retained"),
       {}},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, {}, IREE_SV("conversation")},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, IREE_SV("bad/name"),
       IREE_SV("conversation")},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, IREE_SV("retained"), {}},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN, IREE_SV("retained"),
       IREE_SV("bad session")},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND, IREE_SV("retained"),
       IREE_SV("conversation")},
      {LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE, IREE_SV("retained"),
       IREE_SV("conversation")},
  };
  for (const auto& request : invalid) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_text_checkpoint_request_validate(&request));
  }
  std::string name(64, 'a');
  loom_serve_text_checkpoint_request_t request = {
      LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
      iree_make_cstring_view(name.c_str()),
      {}};
  IREE_EXPECT_OK(loom_serve_text_checkpoint_request_validate(&request));
  name.push_back('a');
  request.name = iree_make_cstring_view(name.c_str());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_checkpoint_request_validate(&request));
}

class TextCheckpointControlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_text_telemetry_initialize(nullptr, allocator_, &telemetry_));
    iree_string_builder_initialize(allocator_, &scratch_);
    const loom_serve_text_checkpoint_request_t request = {
        .action = LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
        .name = IREE_SV("retained"),
        .session = IREE_SV("conversation"),
    };
    IREE_ASSERT_OK(loom_serve_text_checkpoint_request_validate(&request));
    IREE_ASSERT_OK(loom_serve_text_checkpoint_control_allocate(
        &request, allocator_, &control_, &consumer_));
    receipt_ = loom_serve_text_telemetry_begin(&telemetry_);
  }

  void TearDown() override {
    if (consumer_.self) {
      consumer_.abort(consumer_.self, LOOM_SERVE_TELEMETRY_CANCELLED,
                      iree_status_from_code(IREE_STATUS_CANCELLED), &telemetry_,
                      &receipt_);
    }
    loom_serve_text_checkpoint_control_close(control_);
    iree_string_builder_deinitialize(&scratch_);
    loom_serve_telemetry_stream_free(telemetry_.stream);
  }

  // Allocation owner independent of model and telemetry lifetimes.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Caller ownership retained until close.
  loom_serve_text_checkpoint_control_t* control_ = nullptr;
  // Service receipt retained until explicit retirement.
  loom_serve_text_checkpoint_consumer_t consumer_ = {};
  // Real accounting owner, with no device/model dependency in the handle.
  loom_serve_text_telemetry_t telemetry_ = {};
  // One accepted control's identity.
  loom_serve_text_receipt_t receipt_ = {};
  // Borrowed response scratch at terminal delivery.
  iree_string_builder_t scratch_ = {};
};

TEST_F(TextCheckpointControlTest, CompletionHasIndependentResultStorage) {
  EXPECT_FALSE(loom_serve_text_checkpoint_control_is_complete(control_));
  EXPECT_EQ(loom_serve_text_checkpoint_control_result(control_), nullptr);
  IREE_ASSERT_OK(loom_serve_text_checkpoint_control_status(control_));
  loom_serve_text_checkpoint_result_t result = {
      .position = 23,
      .host_bytes = 96,
  };
  IREE_EXPECT_OK(consumer_.end(
      consumer_.self, loom_serve_text_checkpoint_control_request(control_),
      &result, iree_ok_status(), &telemetry_, &receipt_, &scratch_));
  consumer_ = {};
  result = {};
  EXPECT_TRUE(loom_serve_text_checkpoint_control_is_complete(control_));
  const auto* copied = loom_serve_text_checkpoint_control_result(control_);
  ASSERT_NE(copied, nullptr);
  EXPECT_EQ(copied->position, 23u);
  EXPECT_EQ(copied->host_bytes, 96u);
  EXPECT_EQ(telemetry_.counters.completed, 1u);
  loom_serve_telemetry_stream_free(telemetry_.stream);
  telemetry_.stream = nullptr;
  IREE_ASSERT_OK(loom_serve_text_checkpoint_control_status(control_));
  EXPECT_EQ(copied->position, 23u);
}

TEST_F(TextCheckpointControlTest, CloseRetainsExecutionReceiptUntilAbort) {
  EXPECT_TRUE(consumer_.is_open(consumer_.self));
  loom_serve_text_checkpoint_control_close(control_);
  control_ = nullptr;
  EXPECT_FALSE(consumer_.is_open(consumer_.self));
  consumer_.abort(consumer_.self, LOOM_SERVE_TELEMETRY_CANCELLED,
                  iree_status_from_code(IREE_STATUS_CANCELLED), &telemetry_,
                  &receipt_);
  consumer_ = {};
  EXPECT_EQ(telemetry_.counters.cancelled, 1u);
}

TEST_F(TextCheckpointControlTest, RejectionOwnsStatusBeyondDelivery) {
  auto failure =
      iree_make_status(IREE_STATUS_NOT_FOUND, "session is not retained");
  IREE_EXPECT_OK(consumer_.end(
      consumer_.self, loom_serve_text_checkpoint_control_request(control_),
      nullptr, failure, &telemetry_, &receipt_, &scratch_));
  consumer_ = {};
  iree_status_free(failure);
  EXPECT_TRUE(loom_serve_text_checkpoint_control_is_complete(control_));
  EXPECT_EQ(loom_serve_text_checkpoint_control_result(control_), nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_text_checkpoint_control_status(control_));
  EXPECT_EQ(telemetry_.counters.rejected, 1u);
  EXPECT_EQ(telemetry_.counters.failed, 0u);
}

TEST_F(TextCheckpointControlTest, FailedRetirementPublishesFailure) {
  auto failure = iree_make_status(IREE_STATUS_ABORTED, "device work failed");
  consumer_.abort(consumer_.self, LOOM_SERVE_TELEMETRY_FAILED, failure,
                  &telemetry_, &receipt_);
  consumer_ = {};
  iree_status_free(failure);
  EXPECT_TRUE(loom_serve_text_checkpoint_control_is_complete(control_));
  EXPECT_EQ(loom_serve_text_checkpoint_control_result(control_), nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_text_checkpoint_control_status(control_));
  EXPECT_EQ(telemetry_.counters.failed, 1u);
}

}  // namespace
