// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/call.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FiniteCallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &body_);
    iree_string_builder_initialize(iree_allocator_system(), &response_);
    IREE_ASSERT_OK(
        iree_string_builder_append_cstring(&body_, "{\"state\":42}"));
  }

  void TearDown() override {
    loom_serve_finite_call_close(call_);
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
  // Reusable encoding storage, unused by native delivery.
  iree_string_builder_t response_;
  // Independent caller ownership of the receipt.
  loom_serve_finite_call_t* call_ = nullptr;
  // Producer ownership, released only after its accepted borrowers retire.
  loom_serve_finite_consumer_t consumer_ = {};
};

TEST_F(FiniteCallTest, AllocationFailurePreservesInput) {
  const auto original = iree_string_builder_view(&body_);
  const iree_allocator_t failing = {
      nullptr, [](void*, iree_allocator_command_t, const void*, void**) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_finite_call_allocate(&body_, failing, &call_, &consumer_));
  EXPECT_EQ(call_, nullptr);
  EXPECT_EQ(consumer_.self, nullptr);
  EXPECT_EQ(iree_string_builder_view(&body_).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body_), original.size);
}

TEST_F(FiniteCallTest, FixedAndSizingBuildersCannotTransferBorrowedStorage) {
  char storage[8] = "{}";
  iree_string_builder_t borrowed;
  iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                              &borrowed);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_finite_call_allocate(&borrowed, iree_allocator_system(),
                                      &call_, &consumer_));
  EXPECT_EQ(borrowed.buffer, storage);
  iree_string_builder_deinitialize(&borrowed);
  iree_string_builder_initialize(iree_allocator_null(), &borrowed);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_finite_call_allocate(&borrowed, iree_allocator_system(),
                                      &call_, &consumer_));
  iree_string_builder_deinitialize(&borrowed);
}

TEST_F(FiniteCallTest, CallerCloseDoesNotReleaseBorrowedInput) {
  const auto original = iree_string_builder_view(&body_);
  IREE_ASSERT_OK(loom_serve_finite_call_allocate(
      &body_, iree_allocator_system(), &call_, &consumer_));
  EXPECT_EQ(loom_serve_finite_call_body(call_).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body_), 0u);
  auto* borrowed = call_;
  loom_serve_finite_call_close(call_);
  call_ = nullptr;
  EXPECT_FALSE(consumer_.vtable->is_open(consumer_.self));
  EXPECT_TRUE(iree_string_view_equal(loom_serve_finite_call_body(borrowed),
                                     IREE_SV("{\"state\":42}")));
}

TEST_F(FiniteCallTest, RejectedReceiptRetainsStatusAndNotifiesCompletion) {
  IREE_ASSERT_OK(loom_serve_finite_call_allocate(
      &body_, iree_allocator_system(), &call_, &consumer_));
  uint32_t notifications = 0;
  loom_serve_finite_call_set_observer(
      call_,
      {[](void* self) { ++*static_cast<uint32_t*>(self); }, &notifications});
  EXPECT_EQ(notifications, 0u);
  EXPECT_FALSE(loom_serve_finite_call_is_complete(call_));
  auto status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad input");
  End(status, true);
  iree_status_free(status);
  EXPECT_EQ(notifications, 1u);
  EXPECT_TRUE(loom_serve_finite_call_is_complete(call_));
  EXPECT_EQ(loom_serve_finite_call_output(call_), nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_finite_call_status(call_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_finite_call_status(call_));
  loom_serve_finite_call_set_observer(call_, {});
}

}  // namespace
