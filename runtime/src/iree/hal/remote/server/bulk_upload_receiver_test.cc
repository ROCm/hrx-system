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

static void DeinitializeUploadTransfer(void* user_data,
                                       iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_upload_transfer_deinitialize(
      iree_hal_remote_server_bulk_upload_transfer_storage(transfer));
}

static iree_status_t AllocateUploadScheduler(
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler) {
  iree_hal_remote_bulk_transfer_scheduler_options_t options =
      iree_hal_remote_bulk_transfer_scheduler_options_default();
  options.capacity = 1;
  options.user_storage_size =
      sizeof(iree_hal_remote_server_bulk_upload_transfer_t);
  options.user_storage_alignment =
      iree_alignof(iree_hal_remote_server_bulk_upload_transfer_t);
  iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {};
  callbacks.deinitialize = DeinitializeUploadTransfer;
  return iree_hal_remote_bulk_transfer_scheduler_allocate(
      &options, callbacks, iree_allocator_system(), out_scheduler);
}

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

TEST_F(BulkUploadReceiverTest, CompleteSucceedsWithOutOfOrderData) {
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_start(
      &transfer_, IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/16, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/48, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/0, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      &transfer_, kTransferId, kTotalLength, /*chunk_offset=*/32, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_NONE));

  IREE_EXPECT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
      &transfer_, kTransferId));
}

TEST(BulkUploadReceiverSchedulerTest, DuplicateCommandAttachIsRejected) {
  constexpr uint64_t kTransferKind = 1;
  constexpr uint64_t kTransferId = 42;
  constexpr uint64_t kTotalLength = 64;
  constexpr iree_host_size_t kChunkLength = 16;

  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = nullptr;
  IREE_ASSERT_OK(AllocateUploadScheduler(&scheduler));

  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler, kTransferId, kTotalLength, kTransferKind, &table_transfer));
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_initialize(
      /*server=*/nullptr, /*session_slot=*/nullptr, /*session_id=*/0,
      kTransferId, kTotalLength, kChunkLength, iree_allocator_system(),
      transfer));
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY;

  iree_hal_semaphore_t* ready_semaphore = nullptr;
  iree_hal_remote_server_bulk_upload_ready_t* ready_context = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_hal_remote_server_bulk_upload_attach_command_locked(
          /*session_slot=*/nullptr, table_transfer, /*local_device=*/nullptr,
          iree_hal_semaphore_list_empty(), /*target_buffer=*/nullptr,
          /*target_offset=*/0, &ready_semaphore, &ready_context,
          /*response_envelope=*/nullptr));

  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED;
  iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
}

}  // namespace
