// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/client/bulk_test_util.h"
#include "iree/hal/remote/client/bulk_upload_sender.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/bulk/transfer_table.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_CLIENT_UPLOAD_BENCHMARK_SMALL_BATCH_COUNT 64
#define IREE_HAL_REMOTE_CLIENT_UPLOAD_BENCHMARK_LARGE_BATCH_COUNT 8

namespace {

using iree::hal::remote::client::testing::BulkChannelCallbacks;
using iree::hal::remote::client::testing::MockEndpoint;

static void DestroyBenchmarkDevice(iree_hal_device_t* device) { (void)device; }

static const iree_hal_device_vtable_t kBenchmarkDeviceVTable = {
    /*.destroy=*/DestroyBenchmarkDevice,
};

class BulkUploadBenchmarkContext {
 public:
  iree_status_t Initialize(iree_host_size_t send_reserve_count) {
    memset(&device_, 0, sizeof(device_));
    iree_hal_resource_initialize(&kBenchmarkDeviceVTable, &device_.resource);
    device_.host_allocator = iree_allocator_system();

    iree_status_t status = iree_hal_remote_client_bulk_session_initialize(
        device_.host_allocator, &device_.bulk_session);
    if (iree_status_is_ok(status)) {
      session_initialized_ = true;
      status =
          iree_hal_remote_client_device_initialize_bulk_transfers(&device_);
    }
    if (iree_status_is_ok(status)) {
      transfer_table_initialized_ = true;
    }

    if (iree_status_is_ok(status)) {
      bulk_callbacks_.send_complete_hook = DispatchSendComplete;
      bulk_callbacks_.send_complete_user_data = this;
      endpoint_.capture_send_payload = false;
      endpoint_.send_budget = {
          /*.bytes=*/IREE_HOST_SIZE_MAX,
          /*.slots=*/UINT32_MAX,
      };
      endpoint_.sends.reserve(send_reserve_count);
    }

    if (iree_status_is_ok(status) && send_reserve_count > UINT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "benchmark send reserve count too large for remote credit: %" PRIhsz,
          send_reserve_count);
    }

    iree_net_bulk_channel_options_t channel_options =
        iree_net_bulk_channel_options_default();
    channel_options.send_context_capacity = send_reserve_count;
    channel_options.remote_chunk_credit_capacity = (uint32_t)send_reserve_count;

    iree_async_buffer_pool_t* header_pool = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_create_queue_header_pool(
          send_reserve_count, /*buffer_size=*/128, iree_allocator_system(),
          &header_pool);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_create(
          endpoint_.as_endpoint(), &channel_options, header_pool,
          bulk_callbacks_.MakeCallbacks(), iree_allocator_system(),
          &bulk_channel_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(bulk_channel_);
    }
    if (iree_status_is_ok(status)) {
      iree_net_bulk_channel_t* old_channel =
          iree_hal_remote_client_bulk_session_exchange_channel(
              &device_.bulk_session, bulk_channel_);
      iree_net_bulk_channel_release(old_channel);
    }
    return status;
  }

  ~BulkUploadBenchmarkContext() {
    CompleteAllSendsAndReset();
    if (bulk_channel_) {
      iree_hal_remote_client_bulk_session_exchange_channel(
          &device_.bulk_session, NULL);
      iree_net_bulk_channel_detach(bulk_channel_);
      iree_net_bulk_channel_release(bulk_channel_);
      bulk_channel_ = NULL;
    }
    if (transfer_table_initialized_) {
      iree_hal_remote_client_device_deinitialize_bulk_transfers(&device_);
    }
    if (session_initialized_) {
      iree_hal_remote_client_bulk_session_deinitialize(&device_.bulk_session);
    }
  }

  iree_status_t GrantChunkCredit(uint32_t credit_delta) {
    return endpoint_.InjectCredit(credit_delta);
  }

  iree_status_t BeginHostBufferUpload(iree_const_byte_span_t source_bytes) {
    uint64_t transfer_id = 0;
    iree_status_t status = iree_hal_remote_client_bulk_begin_buffer_unmap_write(
        &device_, source_bytes, &transfer_id);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_client_bulk_upload_sender_upload(&device_,
                                                                transfer_id);
    }
    return status;
  }

  iree_status_t TrySendAllUploads() {
    iree_slim_mutex_lock(&device_.bulk_session.transfer_mutex);
    iree_status_t status =
        iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
            &device_, bulk_channel_);
    iree_slim_mutex_unlock(&device_.bulk_session.transfer_mutex);
    return status;
  }

  iree_status_t VerifyCapturedSendCount(iree_host_size_t expected_count) const {
    if (endpoint_.sends.size() != expected_count) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "benchmark expected %" PRIhsz
                              " captured sends but saw %" PRIhsz,
                              expected_count, endpoint_.sends.size());
    }
    return iree_ok_status();
  }

  iree_status_t VerifyNoActiveTransfers() const {
    iree_host_size_t active_transfer_count =
        iree_net_bulk_transfer_table_count(device_.bulk_session.transfers);
    if (active_transfer_count != 0) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "benchmark leaked %" PRIhsz
                              " active upload transfers",
                              active_transfer_count);
    }
    return iree_ok_status();
  }

  void CompleteAllSendsAndReset() {
    endpoint_.CompleteAllSends();
    endpoint_.sends.clear();
  }

 private:
  static iree_status_t DispatchSendComplete(void* user_data,
                                            uint64_t operation_user_data,
                                            iree_status_t status) {
    BulkUploadBenchmarkContext* context =
        static_cast<BulkUploadBenchmarkContext*>(user_data);
    return iree_hal_remote_client_bulk_upload_sender_send_complete(
        &context->device_, context->bulk_channel_, operation_user_data, status);
  }

  // Minimal remote client device state owning the bulk session under test.
  iree_hal_remote_client_device_t device_ = {};

  // Message endpoint backing the in-process bulk channel.
  MockEndpoint endpoint_;

  // Bulk channel callbacks routed into the upload sender.
  BulkChannelCallbacks bulk_callbacks_;

  // Published bulk channel used by upload send and completion paths.
  iree_net_bulk_channel_t* bulk_channel_ = NULL;

  // True once |device_.bulk_session| has been initialized.
  bool session_initialized_ = false;

  // True once |device_.bulk_session.transfers| has been initialized.
  bool transfer_table_initialized_ = false;
};

static iree_host_size_t ChunkCountForLength(iree_host_size_t payload_length) {
  return iree_host_size_ceil_div(
      payload_length, (iree_host_size_t)IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
}

static iree_status_t RunReadyHostBufferUploadBenchmark(
    iree_benchmark_state_t* benchmark_state, iree_host_size_t payload_length,
    iree_host_size_t batch_count) {
  const iree_host_size_t chunk_count = ChunkCountForLength(payload_length);
  const iree_host_size_t sends_per_transfer = chunk_count + 2;
  BulkUploadBenchmarkContext context;
  iree_status_t status = context.Initialize(batch_count * sends_per_transfer);
  std::vector<uint8_t> payload(payload_length, 0xA5);

  int64_t total_transfer_count = 0;
  int64_t total_byte_count = 0;
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, batch_count)) {
    iree_benchmark_pause_timing(benchmark_state);
    status = context.GrantChunkCredit((uint32_t)(batch_count * chunk_count));
    iree_benchmark_resume_timing(benchmark_state);

    for (iree_host_size_t i = 0; i < batch_count && iree_status_is_ok(status);
         ++i) {
      status = context.BeginHostBufferUpload(
          iree_make_const_byte_span(payload.data(), payload.size()));
    }

    iree_benchmark_pause_timing(benchmark_state);
    if (iree_status_is_ok(status)) {
      status =
          context.VerifyCapturedSendCount(batch_count * sends_per_transfer);
    }
    context.CompleteAllSendsAndReset();
    if (iree_status_is_ok(status)) {
      status = context.VerifyNoActiveTransfers();
    }
    if (iree_status_is_ok(status)) {
      total_transfer_count += (int64_t)batch_count;
      total_byte_count += (int64_t)(batch_count * payload_length);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_benchmark_set_items_processed(benchmark_state, total_transfer_count);
  iree_benchmark_set_bytes_processed(benchmark_state, total_byte_count);
  return status;
}

IREE_BENCHMARK_FN(BM_UploadHostBuffer1KiB) {
  (void)benchmark_def;
  return RunReadyHostBufferUploadBenchmark(
      benchmark_state, 1024,
      IREE_HAL_REMOTE_CLIENT_UPLOAD_BENCHMARK_SMALL_BATCH_COUNT);
}

IREE_BENCHMARK_REGISTER(BM_UploadHostBuffer1KiB);

IREE_BENCHMARK_FN(BM_UploadHostBuffer32KiB) {
  (void)benchmark_def;
  return RunReadyHostBufferUploadBenchmark(
      benchmark_state, 32 * 1024,
      IREE_HAL_REMOTE_CLIENT_UPLOAD_BENCHMARK_SMALL_BATCH_COUNT);
}

IREE_BENCHMARK_REGISTER(BM_UploadHostBuffer32KiB);

IREE_BENCHMARK_FN(BM_UploadHostBuffer1MiB) {
  (void)benchmark_def;
  return RunReadyHostBufferUploadBenchmark(
      benchmark_state, 1024 * 1024,
      IREE_HAL_REMOTE_CLIENT_UPLOAD_BENCHMARK_LARGE_BATCH_COUNT);
}

IREE_BENCHMARK_REGISTER(BM_UploadHostBuffer1MiB);

IREE_BENCHMARK_FN(BM_UploadHostBufferCreditBackpressured) {
  (void)benchmark_def;
  static constexpr iree_host_size_t kBatchCount =
      IREE_HAL_REMOTE_CLIENT_UPLOAD_BENCHMARK_SMALL_BATCH_COUNT;
  static constexpr iree_host_size_t kPayloadLength = 1024;
  static constexpr iree_host_size_t kChunkCount = 1;
  static constexpr iree_host_size_t kSendsPerTransfer = kChunkCount + 2;

  BulkUploadBenchmarkContext context;
  iree_status_t status = context.Initialize(kBatchCount * kSendsPerTransfer);
  std::vector<uint8_t> payload(kPayloadLength, 0xA5);

  int64_t total_transfer_count = 0;
  int64_t total_byte_count = 0;
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kBatchCount)) {
    iree_benchmark_pause_timing(benchmark_state);
    for (iree_host_size_t i = 0; i < kBatchCount && iree_status_is_ok(status);
         ++i) {
      status = context.BeginHostBufferUpload(
          iree_make_const_byte_span(payload.data(), payload.size()));
    }
    if (iree_status_is_ok(status)) {
      status = context.VerifyCapturedSendCount(kBatchCount);
    }
    if (iree_status_is_ok(status)) {
      status = context.GrantChunkCredit((uint32_t)(kBatchCount * kChunkCount));
    }
    iree_benchmark_resume_timing(benchmark_state);

    if (iree_status_is_ok(status)) {
      status = context.TrySendAllUploads();
    }

    iree_benchmark_pause_timing(benchmark_state);
    if (iree_status_is_ok(status)) {
      status = context.VerifyCapturedSendCount(kBatchCount * kSendsPerTransfer);
    }
    context.CompleteAllSendsAndReset();
    if (iree_status_is_ok(status)) {
      status = context.VerifyNoActiveTransfers();
    }
    if (iree_status_is_ok(status)) {
      total_transfer_count += (int64_t)kBatchCount;
      total_byte_count += (int64_t)(kBatchCount * kPayloadLength);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_benchmark_set_items_processed(benchmark_state, total_transfer_count);
  iree_benchmark_set_bytes_processed(benchmark_state, total_byte_count);
  return status;
}

IREE_BENCHMARK_REGISTER(BM_UploadHostBufferCreditBackpressured);

}  // namespace
