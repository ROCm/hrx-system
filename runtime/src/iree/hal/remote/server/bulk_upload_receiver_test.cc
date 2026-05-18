// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_upload_receiver.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
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

static iree_status_t RegisterLocalTaskDriver() {
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }
  return status;
}

static void CountLeaseRelease(void* user_data,
                              iree_async_buffer_index_t buffer_index) {
  (void)buffer_index;
  int* release_count = static_cast<int*>(user_data);
  ++*release_count;
}

static iree_async_buffer_lease_t MakeCountingLease(int* release_count) {
  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  lease.release.fn = CountLeaseRelease;
  lease.release.user_data = release_count;
  return lease;
}

static iree_status_t RecordCredit(void* user_data, uint32_t credit_delta) {
  uint32_t* total_credit_delta = static_cast<uint32_t*>(user_data);
  *total_credit_delta += credit_delta;
  return iree_ok_status();
}

static void UnexpectedStagingCallback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t signal_value, iree_status_t status) {
  (void)user_data;
  (void)slot;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, status);
  iree_hal_remote_server_bulk_staging_slot_release(slot, signal_value);
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

class BulkUploadReceiverSessionTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kTransferKind = 1;
  static constexpr uint64_t kTransferId = 42;
  static constexpr uint64_t kTotalLength = 64;
  static constexpr iree_host_size_t kChunkLength = 64;

  void SetUp() override {
    IREE_ASSERT_OK(RegisterLocalTaskDriver());
    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(),
        &driver_));
    IREE_ASSERT_OK(CreateDevice(&device_));

    iree_atomic_ref_count_init(&server_.ref_count);
    server_.host_allocator = iree_allocator_system();
    session_.server = &server_;
    session_.session_id = 1;
    iree_slim_mutex_initialize(&session_.bulk_transfer_mutex);

    IREE_ASSERT_OK(AllocateUploadScheduler(&session_.bulk_transfer_scheduler));
    IREE_ASSERT_OK(AllocateReceiveWindow());
    IREE_ASSERT_OK(AllocateStagingPool(/*slot_count=*/1));
  }

  void TearDown() override {
    iree_hal_remote_bulk_transfer_scheduler_free(
        session_.bulk_transfer_scheduler);
    session_.bulk_transfer_scheduler = nullptr;
    iree_hal_remote_server_bulk_staging_pool_release(
        session_.bulk_staging_pool);
    session_.bulk_staging_pool = nullptr;
    iree_net_bulk_receive_window_free(session_.bulk_receive_window);
    session_.bulk_receive_window = nullptr;
    iree_slim_mutex_deinitialize(&session_.bulk_transfer_mutex);
    iree_hal_device_release(device_);
    iree_hal_driver_release(driver_);
  }

  iree_status_t CreateDevice(iree_hal_device_t** out_device) {
    *out_device = nullptr;
    iree_async_proactor_pool_t* proactor_pool = nullptr;
    iree_status_t status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
    iree_hal_device_t* device = nullptr;
    if (iree_status_is_ok(status)) {
      iree_hal_device_create_params_t create_params =
          iree_hal_device_create_params_default();
      create_params.proactor_pool = proactor_pool;
      status = iree_hal_driver_create_default_device(
          driver_, &create_params, iree_allocator_system(), &device);
    }
    iree_async_proactor_pool_release(proactor_pool);
    if (iree_status_is_ok(status)) {
      *out_device = device;
    } else {
      iree_hal_device_release(device);
    }
    return status;
  }

  iree_status_t AllocateReceiveWindow() {
    iree_net_bulk_receive_window_options_t options =
        iree_net_bulk_receive_window_options_default();
    options.chunk_pool.capacity = 1;
    iree_net_bulk_receive_window_callbacks_t callbacks = {};
    callbacks.send_credit = RecordCredit;
    callbacks.user_data = &total_credit_delta_;
    iree_status_t status = iree_net_bulk_receive_window_allocate(
        &options, callbacks, iree_allocator_system(),
        &session_.bulk_receive_window);
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_receive_window_flush_credit(
          session_.bulk_receive_window);
    }
    return status;
  }

  iree_status_t AllocateStagingPool(iree_host_size_t slot_count) {
    iree_hal_remote_server_bulk_staging_pool_options_t options =
        iree_hal_remote_server_bulk_staging_pool_options_default();
    options.slot_count = slot_count;
    options.slot_length = kChunkLength;
    options.user_storage_size =
        sizeof(iree_hal_remote_server_bulk_upload_staging_callback_t);
    options.user_storage_alignment =
        iree_alignof(iree_hal_remote_server_bulk_upload_staging_callback_t);
    return iree_hal_remote_server_bulk_staging_pool_allocate(
        &options, iree_allocator_system(), &session_.bulk_staging_pool);
  }

  iree_status_t StartUpload(iree_net_bulk_transfer_t** out_table_transfer) {
    *out_table_transfer = nullptr;
    iree_status_t status = iree_hal_remote_server_bulk_upload_on_start_locked(
        &session_, kTransferKind, kTransferId, kTotalLength,
        IREE_NET_BULK_FRAME_FLAG_NONE, kChunkLength);
    if (iree_status_is_ok(status)) {
      *out_table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
          &session_, kTransferKind, kTransferId);
    }
    return status;
  }

  iree_status_t AttachReadyCommand(iree_net_bulk_transfer_t* table_transfer) {
    iree_hal_semaphore_t* ready_semaphore = nullptr;
    iree_status_t status = iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/1,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &ready_semaphore);
    iree_hal_remote_server_bulk_upload_ready_t* ready_context = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_attach_command_locked(
          &session_, table_transfer, device_, iree_hal_semaphore_list_empty(),
          /*target_buffer=*/nullptr, /*target_offset=*/0, &ready_semaphore,
          &ready_context, /*response_envelope=*/nullptr);
    }
    iree_hal_semaphore_release(ready_semaphore);
    iree_hal_remote_server_bulk_upload_ready_release(ready_context);
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_bulk_upload_transfer_t* transfer =
          iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
      transfer->flags |=
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_COMPLETE;
    }
    return status;
  }

  iree_status_t RecordData(iree_net_bulk_frame_flags_t flags,
                           iree_async_buffer_lease_t* lease) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(payload_); ++i) {
      payload_[i] = static_cast<uint8_t>(i);
    }
    return iree_hal_remote_server_bulk_upload_on_data_locked(
        &session_, kTransferKind, kTransferId, /*chunk_offset=*/0,
        /*sequence=*/1, flags,
        iree_make_const_byte_span(payload_, sizeof(payload_)), lease,
        UnexpectedStagingCallback);
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_remote_server_t server_ = {};
  iree_hal_remote_server_session_t session_ = {};
  uint32_t total_credit_delta_ = 0;
  int lease_release_count_ = 0;
  uint8_t payload_[kChunkLength] = {};
};

TEST_F(BulkUploadReceiverSessionTest,
       StagingSlotExhaustionKeepsUploadChunkRetained) {
  iree_hal_remote_server_bulk_staging_slot_t* held_slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_acquire(
      session_.bulk_staging_pool, device_, &held_slot));

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(StartUpload(&table_transfer));
  ASSERT_NE(table_transfer, nullptr);
  IREE_ASSERT_OK(AttachReadyCommand(table_transfer));
  iree_async_buffer_lease_t lease = MakeCountingLease(&lease_release_count_);
  IREE_ASSERT_OK(RecordData(IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, &lease));
  EXPECT_EQ(iree_net_bulk_receive_window_count(session_.bulk_receive_window),
            1u);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(
                session_.bulk_staging_pool),
            1u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count_, 0);
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  iree_hal_remote_server_bulk_staging_slot_release(held_slot,
                                                   /*last_signal_value=*/0);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(
                session_.bulk_staging_pool),
            0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(session_.bulk_receive_window),
            1u);

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
      &session_, kTransferKind, kTransferId);
  ASSERT_NE(table_transfer, nullptr);
  iree_hal_remote_server_bulk_upload_fail_locked(
      &session_, table_transfer,
      iree_make_status(IREE_STATUS_CANCELLED, "test cleanup"));
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);
  EXPECT_EQ(iree_net_bulk_receive_window_count(session_.bulk_receive_window),
            0u);
  EXPECT_EQ(lease_release_count_, 1);
}

TEST_F(BulkUploadReceiverSessionTest, PeerAbortReleasesRetainedChunks) {
  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(StartUpload(&table_transfer));
  ASSERT_NE(table_transfer, nullptr);
  iree_async_buffer_lease_t lease = MakeCountingLease(&lease_release_count_);
  IREE_ASSERT_OK(RecordData(IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, &lease));
  EXPECT_EQ(iree_net_bulk_receive_window_count(session_.bulk_receive_window),
            1u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count_, 0);

  iree_hal_remote_server_bulk_upload_fail_locked(
      &session_, table_transfer,
      iree_make_status(IREE_STATUS_ABORTED,
                       "remote client aborted bulk transfer"));
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                session_.bulk_transfer_scheduler),
            0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(session_.bulk_receive_window),
            0u);
  EXPECT_EQ(lease_release_count_, 1);
}

}  // namespace
