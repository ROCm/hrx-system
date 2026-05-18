// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_upload_receiver.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class BulkUploadReceiverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_initialize(
        /*server=*/nullptr, /*session_slot=*/nullptr, /*session_id=*/0,
        kTransferId, kTotalLength, kChunkLength, iree_allocator_system(),
        &transfer_));
  }

  void TearDown() override {
    iree_hal_remote_server_bulk_upload_transfer_deinitialize(&transfer_);
  }

  static constexpr uint64_t kTransferId = 42;
  static constexpr uint64_t kTotalLength = 64;
  static constexpr iree_host_size_t kChunkLength = 16;

  iree_hal_remote_server_bulk_upload_transfer_t transfer_;
};

TEST_F(BulkUploadReceiverTest, StartRejectsFlags) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_remote_server_bulk_upload_transfer_mark_start(
                            &transfer_, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK));
}

TEST_F(BulkUploadReceiverTest, DataRequiresStart) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0,
          kChunkLength, IREE_NET_BULK_FRAME_FLAG_NONE));
}

TEST_F(BulkUploadReceiverTest, DataRejectsOutOfRange) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/60,
          /*chunk_length=*/8, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK));
}

TEST_F(BulkUploadReceiverTest, DataRejectsFinalFlagMismatch) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0,
          kChunkLength, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/48,
          kChunkLength, IREE_NET_BULK_FRAME_FLAG_NONE));
}

TEST_F(BulkUploadReceiverTest, DataRejectsMisalignedOffset) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/1,
          kChunkLength, IREE_NET_BULK_FRAME_FLAG_NONE));
}

TEST_F(BulkUploadReceiverTest, DataRejectsDuplicateChunk) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0,
          kChunkLength, IREE_NET_BULK_FRAME_FLAG_NONE));
}

TEST_F(BulkUploadReceiverTest, CompleteRequiresAllData) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
          &transfer_, kTransferId));
}

TEST_F(BulkUploadReceiverTest, CompleteSucceedsAfterAllData) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/16, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/32, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/48, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK));

  IREE_EXPECT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
      &transfer_, kTransferId));
}

}  // namespace
