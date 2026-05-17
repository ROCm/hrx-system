// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_transfer_tracker.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BulkTransferTrackerTest, CompleteInOrder) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/10, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  EXPECT_FALSE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/0, /*data_length=*/4));
  EXPECT_FALSE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/4, /*data_length=*/4));
  EXPECT_FALSE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/8, /*data_length=*/2));
  EXPECT_TRUE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, CompleteOutOfOrder) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/10, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/8, /*data_length=*/2));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/0, /*data_length=*/4));
  EXPECT_FALSE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/4, /*data_length=*/4));
  EXPECT_TRUE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, DuplicateRejected) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/8, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/0, /*data_length=*/4));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_hal_remote_bulk_transfer_tracker_record_chunk(
                            &tracker, /*chunk_offset=*/0, /*data_length=*/4));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, GapRemainsIncomplete) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/12, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/0, /*data_length=*/4));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &tracker, /*chunk_offset=*/8, /*data_length=*/4));
  EXPECT_FALSE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, RejectsMisalignedOffset) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/8, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_remote_bulk_transfer_tracker_record_chunk(
                            &tracker, /*chunk_offset=*/2, /*data_length=*/4));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, RejectsUnexpectedChunkLength) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/10, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_remote_bulk_transfer_tracker_record_chunk(
                            &tracker, /*chunk_offset=*/0, /*data_length=*/3));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_remote_bulk_transfer_tracker_record_chunk(
                            &tracker, /*chunk_offset=*/4, /*data_length=*/2));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, RejectsOutOfRangeChunk) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/8, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_remote_bulk_transfer_tracker_record_chunk(
                            &tracker, /*chunk_offset=*/8, /*data_length=*/4));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, ZeroLengthTransferCompletesImmediately) {
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/0, /*chunk_length=*/4, iree_allocator_system(),
      &tracker));

  EXPECT_TRUE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_remote_bulk_transfer_tracker_record_chunk(
                            &tracker, /*chunk_offset=*/0, /*data_length=*/0));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

TEST(BulkTransferTrackerTest, LargeTransferUsesAllocatedBitmap) {
  constexpr uint64_t kChunkLength = 4;
  constexpr uint64_t kChunkCount = 300;
  iree_hal_remote_bulk_transfer_tracker_t tracker;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_initialize(
      /*total_length=*/kChunkLength * kChunkCount,
      /*chunk_length=*/kChunkLength, iree_allocator_system(), &tracker));

  EXPECT_NE(tracker.allocated_words, nullptr);
  for (uint64_t i = kChunkCount; i > 0; --i) {
    IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_tracker_record_chunk(
        &tracker, /*chunk_offset=*/(i - 1) * kChunkLength,
        /*data_length=*/kChunkLength));
  }
  EXPECT_TRUE(iree_hal_remote_bulk_transfer_tracker_is_complete(&tracker));

  iree_hal_remote_bulk_transfer_tracker_deinitialize(&tracker);
}

}  // namespace
