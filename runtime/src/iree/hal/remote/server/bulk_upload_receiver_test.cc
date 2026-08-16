// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_upload_receiver.h"

#include <memory>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::TestBufferPool;

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

typedef struct failing_queue_copy_device_t {
  // HAL resource header for the test wrapper device.
  iree_hal_resource_t resource;

  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;

  // Underlying device used for files and semaphores.
  iree_hal_device_t* base_device;
} failing_queue_copy_device_t;

extern const iree_hal_device_vtable_t failing_queue_copy_device_vtable;

static failing_queue_copy_device_t* failing_queue_copy_device_cast(
    iree_hal_device_t* base_device) {
  IREE_HAL_ASSERT_TYPE(base_device, &failing_queue_copy_device_vtable);
  return (failing_queue_copy_device_t*)base_device;
}

static void failing_queue_copy_device_destroy(iree_hal_device_t* base_device) {
  failing_queue_copy_device_t* device =
      failing_queue_copy_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  iree_hal_device_release(device->base_device);
  iree_allocator_free(host_allocator, device);
}

static iree_string_view_t failing_queue_copy_device_id(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_cstring_view("failing-queue-copy");
}

static iree_allocator_t failing_queue_copy_device_host_allocator(
    iree_hal_device_t* base_device) {
  failing_queue_copy_device_t* device =
      failing_queue_copy_device_cast(base_device);
  return device->host_allocator;
}

static iree_status_t failing_queue_copy_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  failing_queue_copy_device_t* device =
      failing_queue_copy_device_cast(base_device);
  return iree_hal_file_import(device->base_device, queue_affinity, access,
                              handle, flags, out_file);
}

static iree_status_t failing_queue_copy_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  failing_queue_copy_device_t* device =
      failing_queue_copy_device_cast(base_device);
  return iree_hal_semaphore_create(device->base_device, queue_affinity,
                                   initial_value, flags, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
failing_queue_copy_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  failing_queue_copy_device_t* device =
      failing_queue_copy_device_cast(base_device);
  return iree_hal_device_query_semaphore_compatibility(device->base_device,
                                                       semaphore);
}

static iree_status_t failing_queue_copy_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)source_offset;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)flags;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "injected queue_copy failure");
}

const iree_hal_device_vtable_t failing_queue_copy_device_vtable = {
    /*.destroy=*/failing_queue_copy_device_destroy,
    /*.id=*/failing_queue_copy_device_id,
    /*.host_allocator=*/failing_queue_copy_device_host_allocator,
    /*.device_allocator=*/nullptr,
    /*.replace_device_allocator=*/nullptr,
    /*.replace_channel_provider=*/nullptr,
    /*.trim=*/nullptr,
    /*.device_spec=*/nullptr,
    /*.sample_observation=*/nullptr,
    /*.topology_info=*/nullptr,
    /*.refine_topology_edge=*/nullptr,
    /*.assign_topology_info=*/nullptr,
    /*.create_channel=*/nullptr,
    /*.create_command_buffer=*/nullptr,
    /*.load_executable=*/nullptr,
    /*.import_file=*/failing_queue_copy_device_import_file,
    /*.create_semaphore=*/failing_queue_copy_device_create_semaphore,
    /*.query_semaphore_compatibility=*/
    failing_queue_copy_device_query_semaphore_compatibility,
    /*.query_queue_pool_backend=*/nullptr,
    /*.queue_alloca=*/nullptr,
    /*.queue_dealloca=*/nullptr,
    /*.queue_fill=*/nullptr,
    /*.queue_update=*/nullptr,
    /*.queue_copy=*/failing_queue_copy_device_queue_copy,
    /*.queue_read=*/nullptr,
};

static iree_status_t CreateFailingQueueCopyDevice(
    iree_hal_device_t* base_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  *out_device = nullptr;
  failing_queue_copy_device_t* device = nullptr;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*device), (void**)&device);
  if (iree_status_is_ok(status)) {
    memset(device, 0, sizeof(*device));
    iree_hal_resource_initialize(&failing_queue_copy_device_vtable,
                                 &device->resource);
    device->host_allocator = host_allocator;
    device->base_device = base_device;
    iree_hal_device_retain(device->base_device);
    *out_device = (iree_hal_device_t*)device;
  }
  return status;
}

static void UnexpectedStagingCallback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t signal_value, iree_status_t status) {
  (void)user_data;
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
    IREE_ASSERT_OK(CreateTargetBuffer(&target_buffer_));

    iree_atomic_ref_count_init(&server_.ref_count);
    server_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&server_.session_mutex);
    session_.server = &server_;
    session_.session_id = 1;
    session_.session = reinterpret_cast<iree_net_session_t*>(this);
    iree_hal_remote_server_bulk_session_options_t bulk_options =
        iree_hal_remote_server_bulk_session_options_default();
    bulk_options.active_transfer_capacity = 1;
    bulk_options.staging_slot_count = 1;
    bulk_options.staging_slot_length = kChunkLength;
    bulk_options.receive_chunk_capacity = 1;
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_create(
        &session_, &bulk_options, iree_allocator_system(),
        &session_.bulk_session));

    carrier_ = MockCarrier::Create();
    endpoint_.carrier = carrier_.get();
    TestBufferPool buffer_pool;
    IREE_ASSERT_OK(
        buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024));
    IREE_ASSERT_OK(iree_net_bulk_channel_create(
        endpoint_.as_endpoint(), nullptr, buffer_pool.release(),
        iree_hal_remote_server_bulk_session_channel_callbacks(&session_),
        iree_allocator_system(), &bulk_channel_));
    IREE_ASSERT_OK(iree_net_bulk_channel_activate(bulk_channel_));
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_attach_channel(
        &session_, bulk_channel_));
    IREE_ASSERT_OK(
        iree_hal_remote_server_bulk_session_flush_receive_window(&session_));
    CompleteCapturedSends();
  }

  void TearDown() override {
    CompleteCapturedSends();
    iree_hal_remote_server_bulk_session_free(session_.bulk_session);
    session_.bulk_session = nullptr;
    iree_net_bulk_channel_release(bulk_channel_);
    bulk_channel_ = nullptr;
    iree_slim_mutex_deinitialize(&server_.session_mutex);
    iree_hal_buffer_release(target_buffer_);
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

  iree_status_t CreateTargetBuffer(iree_hal_buffer_t** out_buffer) {
    *out_buffer = nullptr;
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
    iree_hal_buffer_params_t params = {0};
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    return iree_hal_allocator_allocate_buffer(allocator, params, kTotalLength,
                                              out_buffer);
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
    return AttachReadyCommand(table_transfer, device_,
                              iree_hal_semaphore_list_empty());
  }

  iree_status_t AttachReadyCommand(iree_net_bulk_transfer_t* table_transfer,
                                   iree_hal_device_t* local_device,
                                   iree_hal_semaphore_list_t signal_list) {
    return AttachReadyCommand(table_transfer, local_device, signal_list,
                              /*response_envelope=*/nullptr);
  }

  iree_status_t AttachReadyControlResponse(
      iree_net_bulk_transfer_t* table_transfer,
      const iree_hal_remote_control_envelope_t* response_envelope) {
    return AttachReadyCommand(table_transfer, device_,
                              iree_hal_semaphore_list_empty(),
                              response_envelope);
  }

  iree_status_t AttachReadyCommand(
      iree_net_bulk_transfer_t* table_transfer, iree_hal_device_t* local_device,
      iree_hal_semaphore_list_t signal_list,
      const iree_hal_remote_control_envelope_t* response_envelope) {
    iree_hal_semaphore_t* ready_semaphore = nullptr;
    iree_status_t status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/1,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &ready_semaphore);
    iree_hal_remote_server_bulk_upload_ready_t* ready_context = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_attach_command_locked(
          &session_, table_transfer, local_device, signal_list, target_buffer_,
          /*target_offset=*/0, &ready_semaphore, &ready_context,
          response_envelope);
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

  void CompleteCapturedSends() {
    if (!carrier_) return;
    for (iree_host_size_t i = 0; i < carrier_->sends.size(); ++i) {
      if (!carrier_->sends[i].completed) {
        carrier_->CompleteSend(i, iree_ok_status());
      }
    }
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_buffer_t* target_buffer_ = nullptr;
  iree_hal_remote_server_t server_ = {};
  iree_hal_remote_server_session_t session_ = {};
  std::unique_ptr<MockCarrier> carrier_;
  MockEndpoint endpoint_;
  iree_net_bulk_channel_t* bulk_channel_ = nullptr;
  int lease_release_count_ = 0;
  uint8_t payload_[kChunkLength] = {};
};

TEST_F(BulkUploadReceiverSessionTest,
       StagingSlotExhaustionKeepsUploadChunkRetained) {
  iree_hal_remote_server_bulk_staging_slot_t* held_slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_acquire(
      iree_hal_remote_server_bulk_session_staging_pool(&session_), device_,
      &held_slot));

  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(StartUpload(&table_transfer));
  ASSERT_NE(table_transfer, nullptr);
  IREE_ASSERT_OK(AttachReadyCommand(table_transfer));
  iree_async_buffer_lease_t lease = MakeCountingLease(&lease_release_count_);
  IREE_ASSERT_OK(RecordData(IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, &lease));
  EXPECT_EQ(iree_net_bulk_receive_window_count(
                iree_hal_remote_server_bulk_session_receive_window(&session_)),
            1u);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(
                iree_hal_remote_server_bulk_session_staging_pool(&session_)),
            1u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count_, 0);
  iree_slim_mutex_unlock(iree_hal_remote_server_bulk_session_mutex(&session_));

  iree_hal_remote_server_bulk_staging_slot_release(held_slot,
                                                   /*last_signal_value=*/0);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(
                iree_hal_remote_server_bulk_session_staging_pool(&session_)),
            0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(
                iree_hal_remote_server_bulk_session_receive_window(&session_)),
            1u);

  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
  table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
      &session_, kTransferKind, kTransferId);
  ASSERT_NE(table_transfer, nullptr);
  iree_hal_remote_server_bulk_upload_fail_locked(
      &session_, table_transfer,
      iree_make_status(IREE_STATUS_CANCELLED, "test cleanup"));
  iree_slim_mutex_unlock(iree_hal_remote_server_bulk_session_mutex(&session_));
  EXPECT_EQ(iree_net_bulk_receive_window_count(
                iree_hal_remote_server_bulk_session_receive_window(&session_)),
            0u);
  EXPECT_EQ(lease_release_count_, 1);
}

TEST_F(BulkUploadReceiverSessionTest,
       ControlResponseSendFailureReleasesTransfer) {
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(StartUpload(&table_transfer));
  ASSERT_NE(table_transfer, nullptr);
  iree_hal_remote_control_envelope_t response_envelope = {};
  response_envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP;
  response_envelope.request_id = 7;
  IREE_ASSERT_OK(
      AttachReadyControlResponse(table_transfer, &response_envelope));

  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_record_data(
      transfer, kTransferId, kTotalLength, /*chunk_offset=*/0, kChunkLength,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
      transfer, kTransferId));

  session_.session = nullptr;
  iree_hal_remote_server_bulk_upload_try_finish_locked(&session_,
                                                       table_transfer);
  iree_slim_mutex_unlock(iree_hal_remote_server_bulk_session_mutex(&session_));

  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                iree_hal_remote_server_bulk_session_scheduler(&session_)),
            0u);
}

TEST_F(BulkUploadReceiverSessionTest, QueueCopyFailureFailsSignalSemaphore) {
  iree_hal_device_t* failing_device = nullptr;
  IREE_ASSERT_OK(CreateFailingQueueCopyDevice(device_, iree_allocator_system(),
                                              &failing_device));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      failing_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &signal_semaphore));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      &signal_value,
  };

  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(StartUpload(&table_transfer));
  ASSERT_NE(table_transfer, nullptr);
  IREE_ASSERT_OK(
      AttachReadyCommand(table_transfer, failing_device, signal_list));
  iree_async_buffer_lease_t lease = MakeCountingLease(&lease_release_count_);
  IREE_ASSERT_OK(RecordData(IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, &lease));
  iree_slim_mutex_unlock(iree_hal_remote_server_bulk_session_mutex(&session_));

  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                iree_hal_remote_server_bulk_session_scheduler(&session_)),
            0u);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(
                iree_hal_remote_server_bulk_session_staging_pool(&session_)),
            0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(
                iree_hal_remote_server_bulk_session_receive_window(&session_)),
            0u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count_, 1);

  uint64_t current_value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_semaphore_query(signal_semaphore, &current_value));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_device_release(failing_device);
}

TEST_F(BulkUploadReceiverSessionTest, PeerAbortReleasesRetainedChunks) {
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(StartUpload(&table_transfer));
  ASSERT_NE(table_transfer, nullptr);
  iree_async_buffer_lease_t lease = MakeCountingLease(&lease_release_count_);
  IREE_ASSERT_OK(RecordData(IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, &lease));
  EXPECT_EQ(iree_net_bulk_receive_window_count(
                iree_hal_remote_server_bulk_session_receive_window(&session_)),
            1u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count_, 0);

  iree_hal_remote_server_bulk_upload_fail_locked(
      &session_, table_transfer,
      iree_make_status(IREE_STATUS_ABORTED,
                       "remote client aborted bulk transfer"));
  iree_slim_mutex_unlock(iree_hal_remote_server_bulk_session_mutex(&session_));

  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                iree_hal_remote_server_bulk_session_scheduler(&session_)),
            0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(
                iree_hal_remote_server_bulk_session_receive_window(&session_)),
            0u);
  EXPECT_EQ(lease_release_count_, 1);
}

}  // namespace
