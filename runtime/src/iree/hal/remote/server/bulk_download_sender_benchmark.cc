// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/server/bulk_download_sender.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_BULK_DOWNLOAD_BENCHMARK_BATCH_COUNT 64
#define IREE_HAL_REMOTE_BULK_DOWNLOAD_STAGING_LENGTH (32 * 1024)

namespace {

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::TestBufferPool;

static constexpr uint64_t kTransferKind = 2;

static iree_status_t RegisterLocalTaskDriver() {
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }
  return status;
}

static iree_status_t CreateLocalTaskDevice(iree_hal_device_t** out_device) {
  *out_device = nullptr;
  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = RegisterLocalTaskDriver();
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(), &driver);
  }

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        driver, &create_params, iree_allocator_system(), &device);
  }

  iree_async_proactor_pool_release(proactor_pool);
  iree_hal_driver_release(driver);
  if (iree_status_is_ok(status)) {
    *out_device = device;
  } else {
    iree_hal_device_release(device);
  }
  return status;
}

static void UnusedReadyCallback(void* user_data,
                                iree_async_semaphore_timepoint_t* timepoint,
                                iree_status_t status) {
  (void)user_data;
  (void)timepoint;
  iree_status_free(status);
}

class BulkDownloadBenchmarkContext {
 public:
  iree_status_t Initialize(iree_host_size_t active_transfer_capacity,
                           iree_host_size_t staging_slot_count,
                           iree_host_size_t staging_slot_length,
                           bool capture_send_payload) {
    iree_status_t status = CreateLocalTaskDevice(&device_);
    if (iree_status_is_ok(status)) {
      status = AllocateSourceBuffer(staging_slot_length);
    }

    iree_atomic_ref_count_init(&server_.ref_count);
    server_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&server_.session_mutex);
    server_mutex_initialized_ = true;

    session_.server = &server_;
    session_.session_id = 1;
    session_.session = reinterpret_cast<iree_net_session_t*>(this);

    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_bulk_session_options_t bulk_options =
          iree_hal_remote_server_bulk_session_options_default();
      bulk_options.active_transfer_capacity = active_transfer_capacity;
      bulk_options.staging_slot_count = staging_slot_count;
      bulk_options.staging_slot_length = staging_slot_length;
      status = iree_hal_remote_server_bulk_session_create(
          &session_, &bulk_options, iree_allocator_system(),
          &session_.bulk_session);
    }

    if (iree_status_is_ok(status)) {
      carrier_ = MockCarrier::Create();
      carrier_->capture_send_payload = capture_send_payload;
      carrier_->sends.reserve(active_transfer_capacity);
      endpoint_.carrier = carrier_.get();
      TestBufferPool buffer_pool;
      status =
          buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024);
      if (iree_status_is_ok(status)) {
        status = iree_net_bulk_channel_create(
            endpoint_.as_endpoint(), nullptr, buffer_pool.release(),
            iree_hal_remote_server_bulk_session_channel_callbacks(&session_),
            iree_allocator_system(), &bulk_channel_);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(bulk_channel_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_session_attach_channel(
          &session_, bulk_channel_);
    }
    return status;
  }

  ~BulkDownloadBenchmarkContext() {
    ReleaseHeldStagingSlot();
    ClearScheduler();
    CompleteAllCapturedSends();
    if (session_.bulk_session) {
      iree_hal_remote_server_bulk_session_free(session_.bulk_session);
      session_.bulk_session = nullptr;
    }
    iree_net_bulk_channel_release(bulk_channel_);
    iree_hal_buffer_release(source_buffer_);
    iree_hal_device_release(device_);
    if (server_mutex_initialized_) {
      iree_slim_mutex_deinitialize(&server_.session_mutex);
    }
  }

  iree_status_t PreparePayload(iree_host_size_t payload_length) {
    payload_storage_.resize(payload_length);
    memset(payload_storage_.data(), 0xA5, payload_storage_.size());
    return iree_ok_status();
  }

  iree_status_t GrantChunkCredit(uint32_t credit_delta) {
    return endpoint_.InjectCredit(credit_delta);
  }

  iree_status_t InsertReadyDataTransfer(uint64_t transfer_id,
                                        uint64_t total_length,
                                        iree_host_size_t chunk_length) {
    iree_net_bulk_transfer_t* table_transfer = nullptr;
    bool transfer_inserted = false;
    iree_status_t status = iree_hal_remote_bulk_transfer_scheduler_insert_peer(
        iree_hal_remote_server_bulk_session_scheduler(&session_), transfer_id,
        total_length, kTransferKind, &table_transfer);
    transfer_inserted = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_bulk_download_transfer_t* transfer =
          iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
      status = iree_hal_remote_server_bulk_download_transfer_initialize(
          &server_, &session_, session_.session_id, transfer_id, device_,
          source_buffer_, /*source_offset=*/0, IREE_HAL_WRITE_FLAG_NONE,
          UnusedReadyCallback, iree_allocator_system(), transfer);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_bulk_download_transfer_t* transfer =
          iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
      transfer->flags |=
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_START_SENT |
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY;
      transfer->staging_contents =
          iree_make_byte_span(payload_storage_.data(), chunk_length);
      transfer->staging_offset = 0;
      transfer->staging_length = chunk_length;
      transfer->next_staging_offset = chunk_length;
    } else if (transfer_inserted) {
      iree_hal_remote_bulk_transfer_scheduler_release(
          iree_hal_remote_server_bulk_session_scheduler(&session_),
          table_transfer);
    }
    return status;
  }

  iree_status_t SendReadyDataTransfers(uint64_t first_transfer_id,
                                       iree_host_size_t transfer_count) {
    iree_status_t status = iree_ok_status();
    iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
    for (iree_host_size_t i = 0;
         i < transfer_count && iree_status_is_ok(status); ++i) {
      iree_net_bulk_transfer_t* table_transfer =
          iree_hal_remote_bulk_transfer_scheduler_lookup(
              iree_hal_remote_server_bulk_session_scheduler(&session_),
              first_transfer_id + i);
      if (table_transfer) {
        iree_hal_remote_server_bulk_download_try_send_locked(
            &session_, bulk_channel_, table_transfer);
      } else {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "benchmark download transfer missing");
      }
    }
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_));
    return status;
  }

  void SendAllReadyDataTransfers() {
    iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
    iree_hal_remote_server_bulk_download_try_send_all_locked(
        &session_, bulk_channel_, kTransferKind);
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_));
  }

  iree_status_t AcquireHeldStagingSlot() {
    return iree_hal_remote_server_bulk_staging_pool_acquire(
        iree_hal_remote_server_bulk_session_staging_pool(&session_), device_,
        &held_staging_slot_);
  }

  void ReleaseHeldStagingSlot() {
    iree_hal_remote_server_bulk_staging_slot_release(held_staging_slot_,
                                                     /*last_signal_value=*/0);
    held_staging_slot_ = nullptr;
  }

  iree_status_t SubmitWithSaturatedStagingPool(uint64_t transfer_id,
                                               iree_device_size_t length) {
    iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
    iree_status_t status = iree_hal_remote_server_bulk_download_submit_locked(
        &session_, bulk_channel_, kTransferKind, session_.session_id, device_,
        iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
        transfer_id, source_buffer_, /*source_offset=*/0, length,
        IREE_HAL_WRITE_FLAG_NONE, UnusedReadyCallback);
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_));

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_free(status);
      return iree_ok_status();
    }
    if (iree_status_is_ok(status)) {
      ClearScheduler();
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "benchmark expected staging slot exhaustion");
    }
    return status;
  }

  iree_host_size_t captured_send_count() const {
    return carrier_ ? carrier_->sends.size() : 0;
  }

  void CompleteAllCapturedSends() {
    if (!carrier_) return;
    for (iree_host_size_t i = 0; i < carrier_->sends.size(); ++i) {
      if (!carrier_->sends[i].completed) {
        carrier_->CompleteSend(i, iree_ok_status());
      }
    }
  }

  void ResetCapturedSends() {
    if (carrier_) carrier_->sends.clear();
  }

  void ClearScheduler() {
    if (!session_.bulk_session ||
        !iree_hal_remote_server_bulk_session_scheduler(&session_)) {
      return;
    }
    iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
    if (iree_hal_remote_server_bulk_session_scheduler(&session_)) {
      iree_hal_remote_bulk_transfer_scheduler_clear(
          iree_hal_remote_server_bulk_session_scheduler(&session_));
    }
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_));
  }

 private:
  iree_status_t AllocateSourceBuffer(iree_device_size_t length) {
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
    iree_hal_buffer_params_t params = {0};
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    return iree_hal_allocator_allocate_buffer(allocator, params, length,
                                              &source_buffer_);
  }

  // Stack server shell retained by transfer state.
  iree_hal_remote_server_t server_ = {};

  // Stack session shell containing the scheduler under test.
  iree_hal_remote_server_session_t session_ = {};

  // Local-task HAL device used for source buffers and staging slots.
  iree_hal_device_t* device_ = nullptr;

  // Source buffer retained by transfers that exercise the submit path.
  iree_hal_buffer_t* source_buffer_ = nullptr;

  // Capturing carrier backing |bulk_channel_|.
  std::unique_ptr<MockCarrier> carrier_;

  // Capturing endpoint backing |bulk_channel_|.
  MockEndpoint endpoint_;

  // Bulk channel used by sender operations.
  iree_net_bulk_channel_t* bulk_channel_ = nullptr;

  // Staging slot held to model delayed local queue_write completion.
  iree_hal_remote_server_bulk_staging_slot_t* held_staging_slot_ = nullptr;

  // Stable DATA bytes referenced by zero-copy bulk channel sends.
  std::vector<uint8_t> payload_storage_;

  // True once |server_.session_mutex| has been initialized.
  bool server_mutex_initialized_ = false;
};

static iree_status_t RunReadyDataBenchmark(
    iree_benchmark_state_t* benchmark_state, iree_host_size_t chunk_length) {
  static constexpr iree_host_size_t kBatchCount =
      IREE_HAL_REMOTE_BULK_DOWNLOAD_BENCHMARK_BATCH_COUNT;
  BulkDownloadBenchmarkContext context;
  iree_status_t status = context.Initialize(
      /*active_transfer_capacity=*/kBatchCount, /*staging_slot_count=*/1,
      IREE_HAL_REMOTE_BULK_DOWNLOAD_STAGING_LENGTH,
      /*capture_send_payload=*/false);
  if (iree_status_is_ok(status)) {
    status = context.PreparePayload(chunk_length);
  }

  uint64_t transfer_id = 1;
  int64_t total_transfers = 0;
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kBatchCount)) {
    iree_benchmark_pause_timing(benchmark_state);
    status = context.GrantChunkCredit((uint32_t)kBatchCount);
    for (iree_host_size_t i = 0; i < kBatchCount && iree_status_is_ok(status);
         ++i) {
      status = context.InsertReadyDataTransfer(transfer_id + i, chunk_length,
                                               chunk_length);
    }
    iree_benchmark_resume_timing(benchmark_state);

    if (iree_status_is_ok(status)) {
      status = context.SendReadyDataTransfers(transfer_id, kBatchCount);
    }

    iree_benchmark_pause_timing(benchmark_state);
    if (iree_status_is_ok(status) &&
        context.captured_send_count() != kBatchCount) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "benchmark expected DATA sends");
    }
    if (iree_status_is_ok(status)) {
      total_transfers += (int64_t)kBatchCount;
    }
    context.ClearScheduler();
    context.CompleteAllCapturedSends();
    context.ResetCapturedSends();
    iree_benchmark_resume_timing(benchmark_state);

    transfer_id += kBatchCount;
  }
  iree_benchmark_set_items_processed(benchmark_state, total_transfers);
  return status;
}

IREE_BENCHMARK_FN(BM_SendReadyData1KiB) {
  (void)benchmark_def;
  return RunReadyDataBenchmark(benchmark_state, 1024);
}

IREE_BENCHMARK_REGISTER(BM_SendReadyData1KiB);

IREE_BENCHMARK_FN(BM_SendReadyData32KiB) {
  (void)benchmark_def;
  return RunReadyDataBenchmark(benchmark_state, 32 * 1024);
}

IREE_BENCHMARK_REGISTER(BM_SendReadyData32KiB);

IREE_BENCHMARK_FN(BM_SendReadyData1MiB) {
  (void)benchmark_def;
  return RunReadyDataBenchmark(benchmark_state, 1024 * 1024);
}

IREE_BENCHMARK_REGISTER(BM_SendReadyData1MiB);

IREE_BENCHMARK_FN(BM_SendAllReadyDataDelayedCompletions) {
  (void)benchmark_def;
  static constexpr iree_host_size_t kBatchCount =
      IREE_HAL_REMOTE_BULK_DOWNLOAD_BENCHMARK_BATCH_COUNT;
  static constexpr iree_host_size_t kChunkLength = 1024;
  BulkDownloadBenchmarkContext context;
  iree_status_t status = context.Initialize(
      /*active_transfer_capacity=*/kBatchCount, /*staging_slot_count=*/1,
      IREE_HAL_REMOTE_BULK_DOWNLOAD_STAGING_LENGTH,
      /*capture_send_payload=*/false);
  if (iree_status_is_ok(status)) {
    status = context.PreparePayload(kChunkLength);
  }

  uint64_t transfer_id = 1;
  int64_t total_transfers = 0;
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kBatchCount)) {
    iree_benchmark_pause_timing(benchmark_state);
    status = context.GrantChunkCredit((uint32_t)kBatchCount);
    for (iree_host_size_t i = 0; i < kBatchCount && iree_status_is_ok(status);
         ++i) {
      status = context.InsertReadyDataTransfer(transfer_id + i, kChunkLength,
                                               kChunkLength);
    }
    iree_benchmark_resume_timing(benchmark_state);

    if (iree_status_is_ok(status)) {
      context.SendAllReadyDataTransfers();
    }

    iree_benchmark_pause_timing(benchmark_state);
    if (iree_status_is_ok(status) &&
        context.captured_send_count() != kBatchCount) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "benchmark expected delayed DATA sends");
    }
    if (iree_status_is_ok(status)) {
      total_transfers += (int64_t)kBatchCount;
    }
    context.ClearScheduler();
    context.CompleteAllCapturedSends();
    context.ResetCapturedSends();
    iree_benchmark_resume_timing(benchmark_state);

    transfer_id += kBatchCount;
  }
  iree_benchmark_set_items_processed(benchmark_state, total_transfers);
  return status;
}

IREE_BENCHMARK_REGISTER(BM_SendAllReadyDataDelayedCompletions);

IREE_BENCHMARK_FN(BM_SubmitSaturatedStagingSlot) {
  (void)benchmark_def;
  static constexpr iree_host_size_t kBatchCount =
      IREE_HAL_REMOTE_BULK_DOWNLOAD_BENCHMARK_BATCH_COUNT;
  static constexpr iree_device_size_t kTransferLength =
      IREE_HAL_REMOTE_BULK_DOWNLOAD_STAGING_LENGTH;
  BulkDownloadBenchmarkContext context;
  iree_status_t status = context.Initialize(
      /*active_transfer_capacity=*/kBatchCount, /*staging_slot_count=*/1,
      IREE_HAL_REMOTE_BULK_DOWNLOAD_STAGING_LENGTH,
      /*capture_send_payload=*/false);
  if (iree_status_is_ok(status)) {
    status = context.AcquireHeldStagingSlot();
  }

  uint64_t transfer_id = 1;
  int64_t total_submissions = 0;
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kBatchCount)) {
    for (iree_host_size_t i = 0; i < kBatchCount && iree_status_is_ok(status);
         ++i) {
      status = context.SubmitWithSaturatedStagingPool(transfer_id++,
                                                      kTransferLength);
    }
    if (iree_status_is_ok(status)) {
      total_submissions += (int64_t)kBatchCount;
    }
  }
  iree_benchmark_set_items_processed(benchmark_state, total_submissions);
  return status;
}

IREE_BENCHMARK_REGISTER(BM_SubmitSaturatedStagingSlot);

}  // namespace
