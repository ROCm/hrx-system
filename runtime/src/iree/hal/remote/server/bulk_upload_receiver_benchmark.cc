// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/bulk_upload_receiver.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH (32 * 1024)
#define IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT 64
#define IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_TOTAL_LENGTH \
  (IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH *    \
   IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT)
#define IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_LENGTH 1024
#define IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_COUNT 64
#define IREE_HAL_REMOTE_BULK_UPLOAD_BURST_TOTAL_LENGTH \
  (IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_LENGTH *    \
   IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_COUNT)

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::TestBufferPool;

static iree_status_t iree_hal_remote_benchmark_register_local_task_driver(
    void) {
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }
  return status;
}

static iree_status_t iree_hal_remote_benchmark_create_local_task_device(
    iree_hal_device_t** out_device) {
  *out_device = NULL;
  iree_hal_driver_t* driver = NULL;
  iree_status_t status = iree_hal_remote_benchmark_register_local_task_driver();
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(), &driver);
  }

  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  iree_hal_device_t* device = NULL;
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

static void iree_hal_remote_benchmark_release_lease(
    void* user_data, iree_async_buffer_index_t buffer_index) {
  (void)user_data;
  (void)buffer_index;
}

static void iree_hal_remote_benchmark_unexpected_staging_callback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t signal_value, iree_status_t status) {
  (void)user_data;
  iree_status_free(status);
  iree_hal_remote_server_bulk_staging_slot_release(slot, signal_value);
}

static iree_status_t iree_hal_remote_benchmark_attach_ready_upload(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_hal_device_t* device) {
  iree_hal_semaphore_t* ready_semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/1,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &ready_semaphore);
  iree_hal_remote_server_bulk_upload_ready_t* ready_context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_upload_attach_command_locked(
        session_slot, table_transfer, device, iree_hal_semaphore_list_empty(),
        /*target_buffer=*/NULL, /*target_offset=*/0, &ready_semaphore,
        &ready_context, /*response_envelope=*/NULL);
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

static void iree_hal_remote_benchmark_complete_captured_sends(
    MockCarrier* carrier) {
  if (!carrier) return;
  for (iree_host_size_t i = 0; i < carrier->sends.size(); ++i) {
    if (!carrier->sends[i].completed) {
      carrier->CompleteSend(i, iree_ok_status());
    }
  }
}

static iree_status_t BM_RecordCompleteTransfer(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  (void)benchmark_def;
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  while (
      iree_status_is_ok(status) &&
      iree_benchmark_keep_running(
          benchmark_state, IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT)) {
    iree_hal_remote_server_bulk_upload_transfer_t transfer;
    bool transfer_initialized = false;
    status = iree_hal_remote_server_bulk_upload_transfer_initialize(
        /*server=*/NULL, /*session_slot=*/NULL, /*session_id=*/0,
        /*transfer_id=*/1, IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_TOTAL_LENGTH,
        IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH, host_allocator,
        &transfer);
    transfer_initialized = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_transfer_mark_start(
          &transfer, IREE_NET_BULK_FRAME_FLAG_NONE);
    }
    for (iree_host_size_t i = 0;
         i < IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT &&
         iree_status_is_ok(status);
         ++i) {
      const bool final_chunk =
          i + 1 == IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT;
      status = iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer, /*transfer_id=*/1,
          IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_TOTAL_LENGTH,
          (uint64_t)i * IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH,
          IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH,
          final_chunk ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                      : IREE_NET_BULK_FRAME_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
          &transfer, /*transfer_id=*/1);
    }
    if (transfer_initialized) {
      iree_hal_remote_server_bulk_upload_transfer_deinitialize(&transfer);
    }
  }
  return status;
}

IREE_BENCHMARK_REGISTER(BM_RecordCompleteTransfer);

static iree_status_t BM_RecordBackpressuredDataBurst(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  (void)benchmark_def;
  iree_hal_device_t* device = NULL;
  iree_status_t status =
      iree_hal_remote_benchmark_create_local_task_device(&device);

  iree_hal_remote_server_t server;
  memset(&server, 0, sizeof(server));
  iree_atomic_ref_count_init(&server.ref_count);
  server.host_allocator = iree_allocator_system();
  iree_slim_mutex_initialize(&server.session_mutex);

  iree_hal_remote_server_session_t session_slot;
  memset(&session_slot, 0, sizeof(session_slot));
  session_slot.server = &server;
  session_slot.session_id = 1;
  session_slot.session = (iree_net_session_t*)&session_slot;

  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_session_options_t bulk_options =
        iree_hal_remote_server_bulk_session_options_default();
    bulk_options.active_transfer_capacity = 1;
    bulk_options.staging_slot_count = 1;
    bulk_options.staging_slot_length =
        IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_LENGTH;
    bulk_options.receive_chunk_capacity =
        IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_COUNT;
    status = iree_hal_remote_server_bulk_session_allocate(
        &session_slot, &bulk_options, iree_allocator_system(),
        &session_slot.bulk_session);
  }

  std::unique_ptr<MockCarrier> carrier;
  MockEndpoint endpoint;
  iree_net_bulk_channel_t* bulk_channel = NULL;
  if (iree_status_is_ok(status)) {
    carrier = MockCarrier::Create();
    endpoint.carrier = carrier.get();
    TestBufferPool buffer_pool;
    status = buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024);
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_create(
          endpoint.as_endpoint(), NULL, buffer_pool.release(),
          iree_hal_remote_server_bulk_session_channel_callbacks(&session_slot),
          iree_allocator_system(), &bulk_channel);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_channel_activate(bulk_channel);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_session_attach_channel(&session_slot,
                                                                bulk_channel);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_server_bulk_session_flush_receive_window(&session_slot);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_benchmark_complete_captured_sends(carrier.get());
  }

  iree_hal_remote_server_bulk_staging_slot_t* held_slot = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_staging_pool_acquire(
        iree_hal_remote_server_bulk_session_staging_pool(&session_slot), device,
        &held_slot);
  }

  uint8_t payload[IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_LENGTH];
  memset(payload, 0xA5, sizeof(payload));
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(
             benchmark_state, IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_COUNT)) {
    iree_slim_mutex_lock(
        iree_hal_remote_server_bulk_session_mutex(&session_slot));

    const uint64_t transfer_id = 1;
    iree_net_bulk_transfer_t* table_transfer = NULL;
    status = iree_hal_remote_server_bulk_upload_on_start_locked(
        &session_slot, /*transfer_kind=*/1, transfer_id,
        IREE_HAL_REMOTE_BULK_UPLOAD_BURST_TOTAL_LENGTH,
        IREE_NET_BULK_FRAME_FLAG_NONE,
        IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_LENGTH);
    if (iree_status_is_ok(status)) {
      table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
          &session_slot, /*transfer_kind=*/1, transfer_id);
      if (table_transfer) {
        status = iree_hal_remote_benchmark_attach_ready_upload(
            &session_slot, table_transfer, device);
      } else {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "benchmark upload transfer missing");
      }
    }

    for (iree_host_size_t i = 0;
         i < IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_COUNT &&
         iree_status_is_ok(status);
         ++i) {
      iree_async_buffer_lease_t lease;
      memset(&lease, 0, sizeof(lease));
      lease.release.fn = iree_hal_remote_benchmark_release_lease;
      const bool final_chunk =
          i + 1 == IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_COUNT;
      status = iree_hal_remote_server_bulk_upload_on_data_locked(
          &session_slot, /*transfer_kind=*/1, transfer_id,
          i * IREE_HAL_REMOTE_BULK_UPLOAD_BURST_CHUNK_LENGTH,
          /*sequence=*/(uint32_t)(i + 1),
          final_chunk ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                      : IREE_NET_BULK_FRAME_FLAG_NONE,
          iree_make_const_byte_span(payload, sizeof(payload)), &lease,
          iree_hal_remote_benchmark_unexpected_staging_callback);
      iree_async_buffer_lease_release(&lease);
    }

    if (table_transfer) {
      iree_hal_remote_server_bulk_upload_fail_locked(
          &session_slot, table_transfer,
          iree_make_status(IREE_STATUS_CANCELLED, "benchmark cleanup"));
    }
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_slot));

    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_session_flush_receive_window(
          &session_slot);
      iree_benchmark_pause_timing(benchmark_state);
      iree_hal_remote_benchmark_complete_captured_sends(carrier.get());
      iree_benchmark_resume_timing(benchmark_state);
    }
  }

  iree_hal_remote_server_bulk_staging_slot_release(held_slot,
                                                   /*last_signal_value=*/0);
  iree_hal_remote_benchmark_complete_captured_sends(carrier.get());
  iree_hal_remote_server_bulk_session_free(session_slot.bulk_session);
  session_slot.bulk_session = NULL;
  iree_net_bulk_channel_release(bulk_channel);
  iree_slim_mutex_deinitialize(&server.session_mutex);
  iree_hal_device_release(device);
  return status;
}

IREE_BENCHMARK_REGISTER(BM_RecordBackpressuredDataBurst);
