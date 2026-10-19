// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/capture.h"

#include <array>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CaptureTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_text_capture_release(capture_);
    iree_vm_buffer_release(plan_);
  }

  iree_status_t CreatePlan(size_t length = sizeof(words_)) {
    iree_vm_buffer_release(std::exchange(plan_, nullptr));
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_create(length, 8, iree_allocator_system(), &plan_));
    iree_byte_span_t bytes;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_map_write(plan_, 0, length, &bytes));
    for (size_t i = 0; i < length / 8; ++i) {
      iree_unaligned_store_le_u64(bytes.data + i * 8, words_[i]);
    }
    return iree_ok_status();
  }

  // Two spans gather discontiguous, unaligned opaque bytes in source order.
  std::array<uint64_t, 9> words_ = {0, 19, 3, 0, 1, 7, 1, 11, 5};
  // Source-owned metadata, reusable immediately after successful validation.
  iree_vm_buffer_t* plan_ = nullptr;
  // Independent checked metadata, with no device storage needed by these tests.
  loom_serve_text_capture_t* capture_ = nullptr;
};

TEST_F(CaptureTest, SourceReusePreservesConcatenatedSpanMetadata) {
  IREE_ASSERT_OK(CreatePlan());
  IREE_ASSERT_OK(loom_serve_text_capture_create(
      plan_, 2, 32, iree_allocator_system(), &capture_));
  iree_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_write(plan_, 0, sizeof(words_), &bytes));
  memset(bytes.data, 0xFF, bytes.data_length);
  iree_vm_buffer_release(std::exchange(plan_, nullptr));
  const auto first = loom_serve_text_capture_span(capture_, 0);
  const auto second = loom_serve_text_capture_span(capture_, 1);
  EXPECT_EQ(first.offset, 0u);
  EXPECT_EQ(first.length, 10u);
  EXPECT_EQ(second.offset, 10u);
  EXPECT_EQ(second.length, 5u);
}

TEST_F(CaptureTest, RejectsMalformedRangesBeforePublication) {
  const auto valid = words_;
  for (const auto& [word, value] :
       std::array<std::pair<size_t, uint64_t>, 9>{{{0, 1},
                                                   {0, UINT64_MAX},
                                                   {1, 33},
                                                   {2, 0},
                                                   {2, 14},
                                                   {3, 2},
                                                   {6, 0},
                                                   {7, UINT64_MAX},
                                                   {8, UINT64_MAX}}}) {
    SCOPED_TRACE(word);
    words_ = valid;
    words_[word] = value;
    IREE_ASSERT_OK(CreatePlan());
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_text_capture_create(plan_, 2, 32, iree_allocator_system(),
                                       &capture_));
    EXPECT_EQ(capture_, nullptr);
  }
  words_ = valid;
  for (size_t length : {size_t{0}, size_t{24}, sizeof(words_) - 1}) {
    IREE_ASSERT_OK(CreatePlan(length));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_text_capture_create(plan_, 2, 32, iree_allocator_system(),
                                       &capture_));
    EXPECT_EQ(capture_, nullptr);
  }
}

TEST_F(CaptureTest, AllocationFailureDoesNotPublishMetadata) {
  IREE_ASSERT_OK(CreatePlan());
  const iree_allocator_t failing_allocator = {
      nullptr, [](void* user_data, iree_allocator_command_t command,
                  const void* parameters, void** inout_pointer) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_text_capture_create(
                            plan_, 2, 32, failing_allocator, &capture_));
  EXPECT_EQ(capture_, nullptr);
}

}  // namespace
