// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/call.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FiniteCallTest
    : public ::testing::TestWithParam<loom_service_finite_output_format_t> {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &body_);
    iree_string_builder_initialize(iree_allocator_system(), &response_);
    IREE_ASSERT_OK(
        iree_string_builder_append_cstring(&body_, "{\"state\":42}"));
  }

  void TearDown() override {
    loom_service_finite_call_close(call_);
    if (consumer_.self) {
      bool delivered = false;
      IREE_EXPECT_OK(consumer_.vtable->end(
          consumer_.self, nullptr, iree_status_from_code(IREE_STATUS_CANCELLED),
          false, &response_, &delivered));
    }
    iree_string_builder_deinitialize(&body_);
    iree_string_builder_deinitialize(&response_);
  }

  void End(const iree_status_t status, bool rejected) {
    bool delivered = false;
    IREE_EXPECT_OK(consumer_.vtable->end(consumer_.self, nullptr, status,
                                         rejected, &response_, &delivered));
    consumer_ = {};
    EXPECT_TRUE(delivered);
  }

  // Caller-owned input before acceptance, initialized empty afterward.
  iree_string_builder_t body_;
  // Encoding storage, moved to JSON receipts on successful completion.
  iree_string_builder_t response_;
  // Independent caller ownership of the receipt.
  loom_service_finite_call_t* call_ = nullptr;
  // Producer ownership, released only after its accepted borrowers retire.
  loom_service_finite_consumer_t consumer_ = {};
};

TEST_P(FiniteCallTest, AllocationFailurePreservesInput) {
  const auto original = iree_string_builder_view(&body_);
  const iree_allocator_t failing = {
      nullptr, [](void*, iree_allocator_command_t, const void*, void**) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_service_finite_call_allocate(
                            &body_, GetParam(), failing, &call_, &consumer_));
  EXPECT_EQ(call_, nullptr);
  EXPECT_EQ(consumer_.self, nullptr);
  EXPECT_EQ(iree_string_builder_view(&body_).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body_), original.size);
}

TEST_P(FiniteCallTest, UnknownOutputPreservesInput) {
  const auto original = iree_string_builder_view(&body_);
  const auto unknown = static_cast<loom_service_finite_output_format_t>(
      2 + static_cast<int>(GetParam()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_service_finite_call_allocate(
          &body_, unknown, iree_allocator_system(), &call_, &consumer_));
  EXPECT_EQ(call_, nullptr);
  EXPECT_EQ(consumer_.self, nullptr);
  EXPECT_EQ(iree_string_builder_view(&body_).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body_), original.size);
}

TEST_P(FiniteCallTest, FixedAndSizingBuildersCannotTransferBorrowedStorage) {
  char storage[8] = "{}";
  iree_string_builder_t borrowed;
  iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                              &borrowed);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_service_finite_call_allocate(
          &borrowed, GetParam(), iree_allocator_system(), &call_, &consumer_));
  EXPECT_EQ(borrowed.buffer, storage);
  iree_string_builder_deinitialize(&borrowed);
  iree_string_builder_initialize(iree_allocator_null(), &borrowed);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_service_finite_call_allocate(
          &borrowed, GetParam(), iree_allocator_system(), &call_, &consumer_));
  iree_string_builder_deinitialize(&borrowed);
}

TEST_P(FiniteCallTest, CallerCloseDoesNotReleaseBorrowedInput) {
  const auto original = iree_string_builder_view(&body_);
  IREE_ASSERT_OK(loom_service_finite_call_allocate(
      &body_, GetParam(), iree_allocator_system(), &call_, &consumer_));
  EXPECT_EQ(loom_service_finite_call_body(call_).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body_), 0u);
  auto* borrowed = call_;
  loom_service_finite_call_close(call_);
  call_ = nullptr;
  EXPECT_FALSE(consumer_.vtable->is_open(consumer_.self));
  EXPECT_TRUE(iree_string_view_equal(loom_service_finite_call_body(borrowed),
                                     IREE_SV("{\"state\":42}")));
}

TEST_P(FiniteCallTest, RejectedReceiptRetainsStatusAndNotifiesCompletion) {
  IREE_ASSERT_OK(loom_service_finite_call_allocate(
      &body_, GetParam(), iree_allocator_system(), &call_, &consumer_));
  uint32_t notifications = 0;
  loom_service_finite_call_set_observer(
      call_,
      {[](void* self) { ++*static_cast<uint32_t*>(self); }, &notifications});
  EXPECT_EQ(notifications, 0u);
  EXPECT_FALSE(loom_service_finite_call_is_complete(call_));
  auto status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad input");
  End(status, true);
  iree_status_free(status);
  EXPECT_EQ(notifications, 1u);
  EXPECT_TRUE(loom_service_finite_call_is_complete(call_));
  EXPECT_EQ(loom_service_finite_call_output(call_), nullptr);
  EXPECT_TRUE(
      iree_string_view_is_empty(loom_service_finite_call_response(call_)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(call_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(call_));
  loom_service_finite_call_set_observer(call_, {});
}

INSTANTIATE_TEST_SUITE_P(OutputFormats, FiniteCallTest,
                         ::testing::Values(LOOM_SERVICE_FINITE_OUTPUT_DEVICE,
                                           LOOM_SERVICE_FINITE_OUTPUT_JSON));

TEST(FiniteCallJsonTest, MovesSuccessfulResponseWithoutCopying) {
  iree_string_builder_t input;
  iree_string_builder_initialize(iree_allocator_system(), &input);
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&input, "{\"state\":42}"));
  loom_service_finite_call_t* call = nullptr;
  loom_service_finite_consumer_t consumer;
  auto status = loom_service_finite_call_allocate(
      &input, LOOM_SERVICE_FINITE_OUTPUT_JSON, iree_allocator_system(), &call,
      &consumer);
  iree_string_builder_deinitialize(&input);
  IREE_ASSERT_OK(status);
  EXPECT_NE(consumer.vtable->encode, nullptr);
  EXPECT_EQ(consumer.vtable->publish, nullptr);
  EXPECT_TRUE(
      iree_string_view_is_empty(loom_service_finite_call_response(call)));

  iree_string_builder_t response;
  iree_string_builder_initialize(iree_allocator_system(), &response);
  IREE_EXPECT_OK(
      iree_string_builder_append_cstring(&response, "{\"answer\":42}"));
  const auto original = iree_string_builder_view(&response);
  bool delivered = false;
  IREE_EXPECT_OK(consumer.vtable->end(consumer.self, nullptr, iree_ok_status(),
                                      false, &response, &delivered));
  EXPECT_TRUE(delivered);
  EXPECT_EQ(iree_string_builder_size(&response), 0u);
  EXPECT_EQ(iree_string_builder_buffer(&response), nullptr);
  iree_string_builder_deinitialize(&response);
  EXPECT_TRUE(loom_service_finite_call_is_complete(call));
  EXPECT_EQ(loom_service_finite_call_output(call), nullptr);
  const auto retained = loom_service_finite_call_response(call);
  EXPECT_EQ(retained.data, original.data);
  EXPECT_TRUE(iree_string_view_equal(retained, IREE_SV("{\"answer\":42}")));
  loom_service_finite_call_close(call);
}

TEST(FiniteCallJsonTest, FailureDoesNotPublishPartialEncoding) {
  iree_string_builder_t input;
  iree_string_builder_initialize(iree_allocator_system(), &input);
  loom_service_finite_call_t* call = nullptr;
  loom_service_finite_consumer_t consumer;
  IREE_ASSERT_OK(loom_service_finite_call_allocate(
      &input, LOOM_SERVICE_FINITE_OUTPUT_JSON, iree_allocator_system(), &call,
      &consumer));
  iree_string_builder_deinitialize(&input);
  iree_string_builder_t response;
  iree_string_builder_initialize(iree_allocator_system(), &response);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(&response, "{\"answer\":"));
  bool delivered = false;
  IREE_EXPECT_OK(consumer.vtable->end(
      consumer.self, nullptr, iree_status_from_code(IREE_STATUS_DATA_LOSS),
      false, &response, &delivered));
  EXPECT_TRUE(delivered);
  EXPECT_TRUE(loom_service_finite_call_is_complete(call));
  EXPECT_TRUE(
      iree_string_view_is_empty(loom_service_finite_call_response(call)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        loom_service_finite_call_status(call));
  iree_string_builder_deinitialize(&response);
  loom_service_finite_call_close(call);
}

}  // namespace
