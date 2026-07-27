// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_upload_sender.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/client/bulk_test_util.h"
#include "iree/hal/remote/client/file.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/hal/remote/util/recv_pool.h"
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

static void DestroyTestDevice(iree_hal_device_t* device) { (void)device; }

static const iree_hal_device_vtable_t kTestDeviceVTable = {
    /*.destroy=*/DestroyTestDevice,
};

class ClientBulkUploadSenderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&device_, 0, sizeof(device_));
    iree_hal_resource_initialize(&kTestDeviceVTable, &device_.resource);
    device_.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_session_initialize(
        device_.host_allocator, &device_.bulk_session));
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_initialize_bulk_transfers(&device_));

    bulk_callbacks_.send_complete_hook = DispatchSendComplete;
    bulk_callbacks_.send_complete_user_data = this;

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
    if (recv_pool_) {
      device_.recv_pool = NULL;
      iree_hal_remote_recv_pool_release(recv_pool_);
      recv_pool_ = NULL;
    }
    iree_hal_remote_client_bulk_session_deinitialize(&device_.bulk_session);
    if (proactor_) {
      device_.proactor = NULL;
      iree_async_proactor_release(proactor_);
      proactor_ = NULL;
    }
  }

  static iree_status_t DispatchSendComplete(void* user_data,
                                            uint64_t operation_user_data,
                                            iree_status_t status) {
    ClientBulkUploadSenderTest* test =
        static_cast<ClientBulkUploadSenderTest*>(user_data);
    return iree_hal_remote_client_bulk_upload_sender_send_complete(
        &test->device_, test->bulk_channel_, operation_user_data, status);
  }

  void GrantRemoteChunkCredit(uint32_t credit_delta) {
    IREE_ASSERT_OK(endpoint_.InjectCredit(credit_delta));
  }

  void TrySendAllUploads() {
    iree_slim_mutex_lock(&device_.bulk_session.transfer_mutex);
    iree_status_t status =
        iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
            &device_, bulk_channel_);
    iree_slim_mutex_unlock(&device_.bulk_session.transfer_mutex);
    IREE_ASSERT_OK(status);
  }

  uint64_t BeginBufferUnmapWrite(iree_const_byte_span_t source_bytes) {
    uint64_t transfer_id = 0;
    IREE_EXPECT_OK(iree_hal_remote_client_bulk_begin_buffer_unmap_write(
        &device_, source_bytes, &transfer_id));
    return transfer_id;
  }

  uint64_t BeginFileRead(iree_hal_file_t* file,
                         const iree_hal_remote_client_file_view_t* file_view,
                         iree_device_size_t length) {
    uint64_t transfer_id = 0;
    IREE_EXPECT_OK(iree_hal_remote_client_bulk_begin_file_read(
        &device_, file, file_view, /*source_offset=*/0, length, &transfer_id));
    return transfer_id;
  }

  void Upload(uint64_t transfer_id) {
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_upload_sender_upload(
        &device_, transfer_id));
  }

  iree_const_byte_span_t SendPayload(iree_host_size_t send_index) {
    const std::vector<uint8_t>& data = endpoint_.sends[send_index].data;
    EXPECT_GE(data.size(), (iree_host_size_t)IREE_NET_BULK_FRAME_HEADER_SIZE);
    return iree_make_const_byte_span(
        data.data() + IREE_NET_BULK_FRAME_HEADER_SIZE,
        data.size() - IREE_NET_BULK_FRAME_HEADER_SIZE);
  }

  iree_status_t CreatePlatformProactorWithRecvPool() {
    iree_async_proactor_options_t proactor_options =
        iree_async_proactor_options_default();
    iree_status_t status = iree_async_proactor_create_platform(
        proactor_options, iree_allocator_system(), &proactor_);
    if (iree_status_is_ok(status)) {
      device_.proactor = proactor_;
    }

    iree_async_slab_t* slab = NULL;
    iree_async_region_t* region = NULL;
    iree_async_buffer_pool_t* buffer_pool = NULL;
    if (iree_status_is_ok(status)) {
      iree_async_slab_options_t slab_options;
      memset(&slab_options, 0, sizeof(slab_options));
      slab_options.buffer_size = IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH;
      slab_options.buffer_count = 4;
      status =
          iree_async_slab_create(slab_options, iree_allocator_system(), &slab);
    }
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_register_slab(
          proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region);
    }
    if (iree_status_is_ok(status)) {
      status = iree_async_buffer_pool_create(region, iree_allocator_system(),
                                             &buffer_pool);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_remote_recv_pool_wrap(proactor_, slab, region, buffer_pool,
                                         iree_allocator_system(), &recv_pool_);
    }
    if (iree_status_is_ok(status)) {
      device_.recv_pool = recv_pool_;
      buffer_pool = NULL;
    }

    iree_async_buffer_pool_release(buffer_pool);
    iree_async_region_release(region);
    iree_async_slab_release(slab);
    return status;
  }

  iree_status_t ImportAsyncTempFile(const iree::testing::TempFilePath& path,
                                    iree_hal_file_t** out_file) {
    *out_file = NULL;
    iree_io_file_handle_t* handle = NULL;
    iree_status_t status = iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
            IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_ASYNC,
        path.path_view(), iree_allocator_system(), &handle);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_client_file_import(
          /*device=*/NULL, /*queue_affinity=*/0, IREE_HAL_MEMORY_ACCESS_READ,
          handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, proactor_,
          iree_allocator_system(), out_file);
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

  // Bulk channel callbacks routed into the upload sender.
  BulkChannelCallbacks bulk_callbacks_;

  // Published bulk channel used by upload send and completion paths.
  iree_net_bulk_channel_t* bulk_channel_ = NULL;

  // Platform proactor created by tests that exercise async file reads.
  iree_async_proactor_t* proactor_ = NULL;

  // Receive buffer pool used as async file read staging storage.
  iree_hal_remote_recv_pool_t* recv_pool_ = NULL;
};

TEST_F(ClientBulkUploadSenderTest,
       HostBufferSendsStartDataCompleteAndReleasesAfterCompletions) {
  constexpr iree_host_size_t kTotalLength =
      IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH + 2;
  std::vector<uint8_t> source(kTotalLength, 1);
  source[IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH - 1] = 2;
  source[IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH] = 3;
  source[IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH + 1] = 4;

  GrantRemoteChunkCredit(/*credit_delta=*/2);
  uint64_t transfer_id = BeginBufferUnmapWrite(
      iree_make_const_byte_span(source.data(), source.size()));
  Upload(transfer_id);

  ASSERT_EQ(endpoint_.sends.size(), 4u);
  iree_net_bulk_frame_header_t start = ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(start));
  EXPECT_EQ(start.type, IREE_NET_BULK_FRAME_TYPE_START);
  EXPECT_EQ(start.transfer_id, transfer_id);
  EXPECT_EQ(start.total_size, kTotalLength);

  iree_net_bulk_frame_header_t data0 = ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(data0));
  EXPECT_EQ(data0.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(data0.transfer_id, transfer_id);
  EXPECT_EQ(data0.chunk_offset, 0u);
  EXPECT_EQ(data0.sequence, 0u);
  EXPECT_EQ(data0.flags, IREE_NET_BULK_FRAME_FLAG_NONE);
  EXPECT_EQ(data0.chunk_length, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
  iree_const_byte_span_t data0_payload = SendPayload(1);
  ASSERT_EQ(data0_payload.data_length, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
  EXPECT_EQ(memcmp(data0_payload.data, source.data(),
                   IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH),
            0);

  iree_net_bulk_frame_header_t data1 = ParseBulkHeader(endpoint_.sends[2]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(data1));
  EXPECT_EQ(data1.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(data1.transfer_id, transfer_id);
  EXPECT_EQ(data1.chunk_offset, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
  EXPECT_EQ(data1.sequence, 1u);
  EXPECT_EQ(data1.flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(data1.chunk_length, 2u);
  iree_const_byte_span_t data1_payload = SendPayload(2);
  ASSERT_EQ(data1_payload.data_length, 2u);
  EXPECT_EQ(memcmp(data1_payload.data,
                   source.data() + IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH, 2),
            0);

  iree_net_bulk_frame_header_t complete = ParseBulkHeader(endpoint_.sends[3]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(complete));
  EXPECT_EQ(complete.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(complete.transfer_id, transfer_id);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);

  endpoint_.CompleteAllSends();

  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
  EXPECT_EQ(bulk_callbacks_.send_complete_result_status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK, IREE_STATUS_OK,
                                             IREE_STATUS_OK, IREE_STATUS_OK}));
}

TEST_F(ClientBulkUploadSenderTest, RemoteCreditBackpressureResumesUpload) {
  std::vector<uint8_t> source(IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH + 1, 5);

  GrantRemoteChunkCredit(/*credit_delta=*/1);
  uint64_t transfer_id = BeginBufferUnmapWrite(
      iree_make_const_byte_span(source.data(), source.size()));
  Upload(transfer_id);

  ASSERT_EQ(endpoint_.sends.size(), 2u);
  EXPECT_EQ(ParseBulkHeader(endpoint_.sends[0]).type,
            IREE_NET_BULK_FRAME_TYPE_START);
  iree_net_bulk_frame_header_t first_data = ParseBulkHeader(endpoint_.sends[1]);
  EXPECT_EQ(first_data.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(first_data.flags, IREE_NET_BULK_FRAME_FLAG_NONE);
  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(bulk_channel_), 0u);

  endpoint_.CompleteAllSends();
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);

  GrantRemoteChunkCredit(/*credit_delta=*/1);
  TrySendAllUploads();

  ASSERT_EQ(endpoint_.sends.size(), 4u);
  iree_net_bulk_frame_header_t final_data = ParseBulkHeader(endpoint_.sends[2]);
  EXPECT_EQ(final_data.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(final_data.chunk_offset, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
  EXPECT_EQ(final_data.flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(final_data.chunk_length, 1u);
  EXPECT_EQ(ParseBulkHeader(endpoint_.sends[3]).type,
            IREE_NET_BULK_FRAME_TYPE_COMPLETE);

  endpoint_.CompleteAllSends();
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
}

TEST_F(ClientBulkUploadSenderTest,
       SendCompletionFailureReleasesTransferAndPropagatesStatus) {
  const uint8_t source[] = {1, 2, 3, 4};

  GrantRemoteChunkCredit(/*credit_delta=*/1);
  uint64_t transfer_id =
      BeginBufferUnmapWrite(iree_make_const_byte_span(source, sizeof(source)));
  Upload(transfer_id);

  ASSERT_EQ(endpoint_.sends.size(), 3u);
  endpoint_.CompleteSend(0, iree_status_from_code(IREE_STATUS_UNAVAILABLE));

  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
  ASSERT_EQ(bulk_callbacks_.send_complete_result_status_codes.size(), 1u);
  EXPECT_EQ(bulk_callbacks_.send_complete_result_status_codes[0],
            IREE_STATUS_UNAVAILABLE);

  endpoint_.CompleteAllSends();
}

TEST_F(ClientBulkUploadSenderTest,
       TerminalFailureRetainsPayloadUntilSendCompletions) {
  const uint8_t source[] = {1, 2, 3, 4};
  GrantRemoteChunkCredit(/*credit_delta=*/1);
  uint64_t transfer_id =
      BeginBufferUnmapWrite(iree_make_const_byte_span(source, sizeof(source)));
  Upload(transfer_id);
  ASSERT_EQ(endpoint_.sends.size(), 3u);

  iree_hal_remote_client_bulk_fail_transfers(
      &device_,
      iree_make_status(IREE_STATUS_ABORTED, "injected terminal failure"));

  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);
  iree_hal_remote_client_bulk_cancel_transfer(&device_, transfer_id);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);
  endpoint_.CompleteAllSends();
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
  EXPECT_EQ(endpoint_.sends.size(), 3u);

  uint64_t rejected_transfer_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_remote_client_bulk_begin_buffer_unmap_write(
          &device_, iree_make_const_byte_span(source, sizeof(source)),
          &rejected_transfer_id));
  EXPECT_EQ(rejected_transfer_id, 0u);
}

TEST_F(ClientBulkUploadSenderTest, AsyncFileReadSendsDataAfterReadCompletion) {
  iree_status_t status = CreatePlatformProactorWithRecvPool();
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "Platform async file read staging unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_bulk_upload");
  const uint8_t contents[] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(contents, sizeof(contents)),
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
  file_view.length = sizeof(contents);
  GrantRemoteChunkCredit(/*credit_delta=*/1);
  uint64_t transfer_id = BeginFileRead(file, &file_view, sizeof(contents));
  Upload(transfer_id);

  ASSERT_EQ(endpoint_.sends.size(), 1u);
  EXPECT_EQ(ParseBulkHeader(endpoint_.sends[0]).type,
            IREE_NET_BULK_FRAME_TYPE_START);
  endpoint_.CompleteSend(0, iree_ok_status());

  PollProactorUntil([&]() { return endpoint_.sends.size() >= 2; });

  iree_net_bulk_frame_header_t data = ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(data));
  EXPECT_EQ(data.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(data.transfer_id, transfer_id);
  EXPECT_EQ(data.chunk_offset, 0u);
  EXPECT_EQ(data.sequence, 0u);
  EXPECT_EQ(data.flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  iree_const_byte_span_t data_payload = SendPayload(1);
  ASSERT_EQ(data_payload.data_length, sizeof(contents));
  EXPECT_EQ(memcmp(data_payload.data, contents, sizeof(contents)), 0);

  endpoint_.CompleteSend(1, iree_ok_status());

  ASSERT_EQ(endpoint_.sends.size(), 3u);
  iree_net_bulk_frame_header_t complete = ParseBulkHeader(endpoint_.sends[2]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(complete));
  EXPECT_EQ(complete.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(complete.transfer_id, transfer_id);
  endpoint_.CompleteSend(2, iree_ok_status());

  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);

  iree_hal_file_release(file);
}

TEST_F(ClientBulkUploadSenderTest,
       TerminalFailureRetiresAsyncReadWithoutSendingData) {
  iree_status_t status = CreatePlatformProactorWithRecvPool();
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "Platform async file read staging unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_bulk_terminal");
  const uint8_t contents[] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(contents, sizeof(contents)),
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
  file_view.length = sizeof(contents);
  GrantRemoteChunkCredit(/*credit_delta=*/1);
  uint64_t transfer_id = BeginFileRead(file, &file_view, sizeof(contents));
  Upload(transfer_id);
  ASSERT_EQ(endpoint_.sends.size(), 1u);

  iree_hal_remote_client_bulk_fail_transfers(
      &device_,
      iree_make_status(IREE_STATUS_ABORTED, "injected terminal failure"));
  iree_hal_remote_client_bulk_cancel_transfer(&device_, transfer_id);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);
  endpoint_.CompleteSend(0, iree_ok_status());

  PollProactorUntil([&]() {
    return iree_net_bulk_transfer_table_count(device_.bulk_session.transfers) ==
           0;
  });
  EXPECT_EQ(endpoint_.sends.size(), 1u);

  iree_hal_file_release(file);
}

TEST_F(ClientBulkUploadSenderTest, AsyncFileReadShortReadSendsAbort) {
  iree_status_t status = CreatePlatformProactorWithRecvPool();
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "Platform async file read staging unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_bulk_upload");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_const_byte_span_empty(), iree_allocator_system()));

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
  file_view.length = 4;
  GrantRemoteChunkCredit(/*credit_delta=*/1);
  uint64_t transfer_id = BeginFileRead(file, &file_view, /*length=*/4);
  Upload(transfer_id);

  ASSERT_EQ(endpoint_.sends.size(), 1u);
  EXPECT_EQ(ParseBulkHeader(endpoint_.sends[0]).type,
            IREE_NET_BULK_FRAME_TYPE_START);

  PollProactorUntil([&]() { return endpoint_.sends.size() >= 2; });

  iree_net_bulk_frame_header_t abort = ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(abort));
  EXPECT_EQ(abort.type, IREE_NET_BULK_FRAME_TYPE_ABORT);
  EXPECT_EQ(abort.transfer_id, transfer_id);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);

  endpoint_.CompleteAllSends();
  iree_hal_file_release(file);
}

}  // namespace
