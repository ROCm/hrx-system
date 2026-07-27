// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_download_receiver.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/client/bulk_test_util.h"
#include "iree/hal/remote/client/file.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/bulk/transfer_table.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

using iree::hal::remote::client::testing::BulkChannelCallbacks;
using iree::hal::remote::client::testing::MockEndpoint;
using iree::hal::remote::client::testing::ParseBulkHeader;

struct CompletionRecorder {
  // Status codes observed by the transfer completion callback.
  std::vector<iree_status_code_t>* status_codes;
};

static void RecordCompletion(void* user_data, iree_status_t status) {
  CompletionRecorder* recorder = static_cast<CompletionRecorder*>(user_data);
  recorder->status_codes->push_back(iree_status_code(status));
  iree_status_free(status);
}

static void DestroyTestDevice(iree_hal_device_t* device) { (void)device; }

static const iree_hal_device_vtable_t kTestDeviceVTable = {
    /*.destroy=*/DestroyTestDevice,
};

static void RecordBufferRelease(void* user_data, uint32_t buffer_index) {
  int* release_count = static_cast<int*>(user_data);
  ++*release_count;
  (void)buffer_index;
}

static iree_async_buffer_lease_t MakeTestLease(std::vector<uint8_t>* storage,
                                               int* release_count) {
  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  lease.span = iree_async_span_from_ptr(storage->data(), storage->size());
  lease.release.fn = RecordBufferRelease;
  lease.release.user_data = release_count;
  return lease;
}

class ClientBulkDownloadReceiverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&device_, 0, sizeof(device_));
    iree_hal_resource_initialize(&kTestDeviceVTable, &device_.resource);
    device_.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_session_initialize(
        device_.host_allocator, &device_.bulk_session));
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_initialize_bulk_transfers(&device_));

    iree_async_buffer_pool_t* header_pool = NULL;
    IREE_ASSERT_OK(iree_hal_remote_create_queue_header_pool(
        /*buffer_count=*/16, /*buffer_size=*/128, iree_allocator_system(),
        &header_pool));
    IREE_ASSERT_OK(iree_net_bulk_channel_create(
        endpoint_.as_endpoint(), /*options=*/NULL, header_pool,
        bulk_callbacks_.MakeCallbacks(), iree_allocator_system(),
        &bulk_channel_));
    IREE_ASSERT_OK(iree_net_bulk_channel_activate(bulk_channel_));
    ASSERT_EQ(iree_hal_remote_client_bulk_session_exchange_channel(
                  &device_.bulk_session, bulk_channel_),
              nullptr);
  }

  void TearDown() override {
    endpoint_.CompleteAllSends();
    if (bulk_channel_) {
      iree_hal_remote_client_bulk_session_exchange_channel(
          &device_.bulk_session, NULL);
      iree_net_bulk_channel_detach(bulk_channel_);
      iree_net_bulk_channel_release(bulk_channel_);
      bulk_channel_ = NULL;
    }
    iree_hal_remote_client_device_deinitialize_bulk_transfers(&device_);
    iree_hal_remote_client_bulk_session_deinitialize(&device_.bulk_session);
    iree_async_proactor_release(proactor_);
    proactor_ = NULL;
  }

  uint64_t BeginBufferMapRead(iree_byte_span_t target_bytes) {
    completion_recorder_.status_codes = &completion_status_codes_;
    iree_hal_remote_client_bulk_completion_callback_t callback = {
        RecordCompletion, &completion_recorder_};
    uint64_t transfer_id = 0;
    IREE_EXPECT_OK(iree_hal_remote_client_bulk_begin_buffer_map_read(
        &device_, target_bytes, callback, &transfer_id));
    return transfer_id;
  }

  void ReceiveData(uint64_t transfer_id, uint64_t chunk_offset,
                   uint32_t sequence, iree_net_bulk_frame_flags_t flags,
                   iree_const_byte_span_t chunk_data) {
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    bool handled = false;
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_download_receiver_on_data(
        &device_, bulk_channel_, transfer_id, chunk_offset, sequence, flags,
        chunk_data, &lease, &handled));
    ASSERT_TRUE(handled);
  }

  void CompleteDownload(uint64_t transfer_id) {
    bool handled = false;
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_download_receiver_on_complete(
        &device_, bulk_channel_, transfer_id, &handled));
    ASSERT_TRUE(handled);
  }

  iree_status_t CreatePlatformProactor() {
    iree_async_proactor_options_t proactor_options =
        iree_async_proactor_options_default();
    iree_status_t status = iree_async_proactor_create_platform(
        proactor_options, iree_allocator_system(), &proactor_);
    if (iree_status_is_ok(status)) device_.proactor = proactor_;
    return status;
  }

  iree_status_t ImportAsyncTempFile(const iree::testing::TempFilePath& path,
                                    iree_hal_file_t** out_file) {
    *out_file = NULL;
    iree_io_file_handle_t* handle = NULL;
    iree_status_t status = iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
        path.path_view(), iree_allocator_system(), &handle);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_client_file_import(
          /*device=*/NULL, /*queue_affinity=*/0,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
          IREE_HAL_EXTERNAL_FILE_FLAG_NONE, proactor_, iree_allocator_system(),
          out_file);
    }
    iree_io_file_handle_release(handle);
    return status;
  }

  template <typename Predicate>
  void PollProactorUntil(Predicate predicate) {
    while (!predicate()) {
      iree_host_size_t completion_count = 0;
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), &completion_count));
    }
  }

  // Minimal remote client device state owning the bulk session under test.
  iree_hal_remote_client_device_t device_;

  // Message endpoint backing the in-process bulk channel.
  MockEndpoint endpoint_;

  // Bulk channel callback recorder.
  BulkChannelCallbacks bulk_callbacks_;

  // Published bulk channel used by synchronous and async receiver paths.
  iree_net_bulk_channel_t* bulk_channel_ = NULL;

  // Platform proactor created by tests that exercise async file writes.
  iree_async_proactor_t* proactor_ = NULL;

  // Completion callback context for buffer-map transfers.
  CompletionRecorder completion_recorder_ = {};

  // Status codes observed by buffer-map transfer completions.
  std::vector<iree_status_code_t> completion_status_codes_;
};

TEST_F(ClientBulkDownloadReceiverTest, HostBufferReceivesChunksAndCompletes) {
  constexpr iree_host_size_t kTotalLength =
      IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH + 2;
  std::vector<uint8_t> target(kTotalLength, 0);
  uint64_t transfer_id =
      BeginBufferMapRead(iree_make_byte_span(target.data(), target.size()));

  const uint8_t tail[] = {3, 4};
  ReceiveData(transfer_id,
              /*chunk_offset=*/IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH,
              /*sequence=*/1, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
              iree_make_const_byte_span(tail, sizeof(tail)));
  std::vector<uint8_t> head(IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH, 1);
  head.back() = 2;
  ReceiveData(transfer_id, /*chunk_offset=*/0, /*sequence=*/2,
              IREE_NET_BULK_FRAME_FLAG_NONE,
              iree_make_const_byte_span(head.data(), head.size()));
  CompleteDownload(transfer_id);

  EXPECT_EQ(target.front(), 1u);
  EXPECT_EQ(target[IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH - 1], 2u);
  EXPECT_EQ(target[IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH], 3u);
  EXPECT_EQ(target[IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH + 1], 4u);
  ASSERT_EQ(completion_status_codes_.size(), 1u);
  EXPECT_EQ(completion_status_codes_[0], IREE_STATUS_OK);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);

  ASSERT_EQ(endpoint_.sends.size(), 3u);
  iree_net_bulk_frame_header_t credit0 = ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit0));
  EXPECT_EQ(credit0.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  iree_net_bulk_frame_header_t credit1 = ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit1));
  EXPECT_EQ(credit1.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  iree_net_bulk_frame_header_t complete = ParseBulkHeader(endpoint_.sends[2]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(complete));
  EXPECT_EQ(complete.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(complete.transfer_id, transfer_id);
}

TEST_F(ClientBulkDownloadReceiverTest, RejectsOutOfRangeData) {
  std::vector<uint8_t> target(4, 0);
  uint64_t transfer_id =
      BeginBufferMapRead(iree_make_byte_span(target.data(), target.size()));

  const uint8_t data[] = {1, 2};
  bool handled = false;
  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_remote_client_bulk_download_receiver_on_data(
          &device_, bulk_channel_, transfer_id, /*chunk_offset=*/3,
          /*sequence=*/1, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
          iree_make_const_byte_span(data, sizeof(data)), &lease, &handled));

  EXPECT_TRUE(handled);
  ASSERT_EQ(completion_status_codes_.size(), 1u);
  EXPECT_EQ(completion_status_codes_[0], IREE_STATUS_OUT_OF_RANGE);
  EXPECT_TRUE(endpoint_.sends.empty());
}

TEST_F(ClientBulkDownloadReceiverTest, CompleteBeforeDataFailsTransfer) {
  std::vector<uint8_t> target(IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH + 1, 0);
  uint64_t transfer_id =
      BeginBufferMapRead(iree_make_byte_span(target.data(), target.size()));

  std::vector<uint8_t> data(IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH, 1);
  ReceiveData(transfer_id, /*chunk_offset=*/0, /*sequence=*/1,
              IREE_NET_BULK_FRAME_FLAG_NONE,
              iree_make_const_byte_span(data.data(), data.size()));

  bool handled = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_client_bulk_download_receiver_on_complete(
          &device_, bulk_channel_, transfer_id, &handled));

  EXPECT_TRUE(handled);
  ASSERT_EQ(completion_status_codes_.size(), 1u);
  EXPECT_EQ(completion_status_codes_[0], IREE_STATUS_INVALID_ARGUMENT);
  ASSERT_EQ(endpoint_.sends.size(), 1u);
  iree_net_bulk_frame_header_t credit = ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit));
  EXPECT_EQ(credit.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
}

TEST_F(ClientBulkDownloadReceiverTest, CompleteReturnsAckSendFailure) {
  std::vector<uint8_t> target(1, 0);
  uint64_t transfer_id =
      BeginBufferMapRead(iree_make_byte_span(target.data(), target.size()));

  const uint8_t data[] = {1};
  ReceiveData(transfer_id, /*chunk_offset=*/0, /*sequence=*/1,
              IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
              iree_make_const_byte_span(data, sizeof(data)));
  endpoint_.next_send_error = IREE_STATUS_UNAVAILABLE;

  bool handled = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_remote_client_bulk_download_receiver_on_complete(
          &device_, bulk_channel_, transfer_id, &handled));

  EXPECT_TRUE(handled);
  ASSERT_EQ(completion_status_codes_.size(), 1u);
  EXPECT_EQ(completion_status_codes_[0], IREE_STATUS_UNAVAILABLE);
  ASSERT_EQ(endpoint_.sends.size(), 1u);
  iree_net_bulk_frame_header_t credit = ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit));
  EXPECT_EQ(credit.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
}

TEST_F(ClientBulkDownloadReceiverTest, ServerAbortFailsAndReleasesTransfer) {
  std::vector<uint8_t> target(1, 0);
  uint64_t transfer_id =
      BeginBufferMapRead(iree_make_byte_span(target.data(), target.size()));

  bool handled = false;
  IREE_EXPECT_OK(iree_hal_remote_client_bulk_download_receiver_on_abort(
      &device_, transfer_id, &handled));

  EXPECT_TRUE(handled);
  ASSERT_EQ(completion_status_codes_.size(), 1u);
  EXPECT_EQ(completion_status_codes_[0], IREE_STATUS_ABORTED);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
}

TEST_F(ClientBulkDownloadReceiverTest,
       TerminalFailureWakesMapAndRejectsNewTransfers) {
  std::vector<uint8_t> target(4, 0);
  BeginBufferMapRead(iree_make_byte_span(target.data(), target.size()));

  iree_hal_remote_client_bulk_fail_transfers(
      &device_,
      iree_make_status(IREE_STATUS_ABORTED, "injected terminal failure"));

  ASSERT_EQ(completion_status_codes_.size(), 1u);
  EXPECT_EQ(completion_status_codes_[0], IREE_STATUS_ABORTED);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);

  uint64_t transfer_id = 0;
  CompletionRecorder recorder = {&completion_status_codes_};
  iree_hal_remote_client_bulk_completion_callback_t callback = {
      RecordCompletion, &recorder};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_remote_client_bulk_begin_buffer_map_read(
          &device_, iree_make_byte_span(target.data(), target.size()), callback,
          &transfer_id));
  EXPECT_EQ(transfer_id, 0u);
  EXPECT_EQ(completion_status_codes_.size(), 1u);
}

TEST_F(ClientBulkDownloadReceiverTest,
       AsyncFileWriteCompletesAfterServerComplete) {
  iree_status_t status = CreatePlatformProactor();
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_bulk_download");
  const uint8_t initial_contents[] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(initial_contents, sizeof(initial_contents)),
      iree_allocator_system()));

  iree_hal_file_t* file = NULL;
  status = ImportAsyncTempFile(path, &file);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_hal_remote_client_file_view_t file_view;
  IREE_ASSERT_OK(iree_hal_remote_client_file_resolve(file, &file_view));
  file_view.length = sizeof(initial_contents);
  uint64_t transfer_id = 0;
  IREE_ASSERT_OK(iree_hal_remote_client_bulk_begin_file_write(
      &device_, file, &file_view, /*target_offset=*/0,
      /*length=*/sizeof(initial_contents), &transfer_id));

  std::vector<uint8_t> payload = {1, 2, 3, 4};
  int release_count = 0;
  iree_async_buffer_lease_t lease = MakeTestLease(&payload, &release_count);
  bool handled = false;
  IREE_ASSERT_OK(iree_hal_remote_client_bulk_download_receiver_on_data(
      &device_, bulk_channel_, transfer_id, /*chunk_offset=*/0, /*sequence=*/1,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
      iree_make_const_byte_span(payload.data(), payload.size()), &lease,
      &handled));
  ASSERT_TRUE(handled);
  EXPECT_TRUE(endpoint_.sends.empty());

  CompleteDownload(transfer_id);
  EXPECT_TRUE(endpoint_.sends.empty());

  PollProactorUntil([&]() { return endpoint_.sends.size() >= 2; });

  EXPECT_EQ(release_count, 1);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
  ASSERT_EQ(endpoint_.sends.size(), 2u);
  iree_net_bulk_frame_header_t credit = ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit));
  EXPECT_EQ(credit.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  iree_net_bulk_frame_header_t complete = ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(complete));
  EXPECT_EQ(complete.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(complete.transfer_id, transfer_id);

  iree_io_file_contents_t* contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &contents));
  ASSERT_EQ(contents->const_buffer.data_length, payload.size());
  EXPECT_EQ(memcmp(contents->const_buffer.data, payload.data(), payload.size()),
            0);
  iree_io_file_contents_free(contents);

  iree_hal_file_release(file);
}

TEST_F(ClientBulkDownloadReceiverTest,
       TerminalFailureRetiresAdmittedAsyncWriteWithoutSending) {
  iree_status_t status = CreatePlatformProactor();
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_bulk_terminal");
  const uint8_t initial_contents[] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(initial_contents, sizeof(initial_contents)),
      iree_allocator_system()));

  iree_hal_file_t* file = NULL;
  status = ImportAsyncTempFile(path, &file);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_hal_remote_client_file_view_t file_view;
  IREE_ASSERT_OK(iree_hal_remote_client_file_resolve(file, &file_view));
  file_view.length = sizeof(initial_contents);
  uint64_t transfer_id = 0;
  IREE_ASSERT_OK(iree_hal_remote_client_bulk_begin_file_write(
      &device_, file, &file_view, /*target_offset=*/0,
      /*length=*/sizeof(initial_contents), &transfer_id));

  std::vector<uint8_t> payload = {1, 2, 3, 4};
  int release_count = 0;
  iree_async_buffer_lease_t lease = MakeTestLease(&payload, &release_count);
  bool handled = false;
  IREE_ASSERT_OK(iree_hal_remote_client_bulk_download_receiver_on_data(
      &device_, bulk_channel_, transfer_id, /*chunk_offset=*/0, /*sequence=*/1,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
      iree_make_const_byte_span(payload.data(), payload.size()), &lease,
      &handled));
  ASSERT_TRUE(handled);

  iree_hal_remote_client_bulk_fail_transfers(
      &device_,
      iree_make_status(IREE_STATUS_ABORTED, "injected terminal failure"));
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);
  iree_hal_remote_client_bulk_cancel_transfer(&device_, transfer_id);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);

  PollProactorUntil([&]() {
    return iree_net_bulk_transfer_table_count(device_.bulk_session.transfers) ==
           0;
  });

  EXPECT_EQ(release_count, 1);
  EXPECT_TRUE(endpoint_.sends.empty());
  iree_hal_file_release(file);
}

}  // namespace
