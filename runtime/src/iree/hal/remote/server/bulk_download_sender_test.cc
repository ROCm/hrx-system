// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_download_sender.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/net/carrier.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TestBufferPool {
 public:
  TestBufferPool(iree_host_size_t buffer_count, iree_host_size_t buffer_size) {
    iree_host_size_t total_size = buffer_count * buffer_size;
    void* memory = malloc(total_size);
    memset(memory, 0, total_size);

    region_ =
        static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
    memset(region_, 0, sizeof(*region_));
    iree_atomic_ref_count_init(&region_->ref_count);
    region_->destroy_fn = DestroyRegion;
    region_->base_ptr = memory;
    region_->length = total_size;
    region_->buffer_size = buffer_size;
    region_->buffer_count = static_cast<uint32_t>(buffer_count);

    iree_status_t status = iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &pool_);
    IREE_CHECK_OK(status);
    iree_async_region_release(region_);
  }

  ~TestBufferPool() {
    if (pool_) iree_async_buffer_pool_free(pool_);
  }

  iree_async_buffer_pool_t* release() {
    iree_async_buffer_pool_t* pool = pool_;
    pool_ = nullptr;
    return pool;
  }

 private:
  static void DestroyRegion(iree_async_region_t* region) {
    free(region->base_ptr);
    free(region);
  }

  // Owned async region backing |pool_|.
  iree_async_region_t* region_ = nullptr;

  // Owned buffer pool handed to the bulk channel.
  iree_async_buffer_pool_t* pool_ = nullptr;
};

struct CapturedSend {
  // Concatenated scatter-gather send bytes captured at the carrier boundary.
  std::vector<uint8_t> data;

  // Carrier completion user data for the captured send.
  uint64_t user_data = 0;

  // Whether the mock carrier has fired completion for this send.
  bool completed = false;
};

struct MockCarrier {
  // Base carrier structure.
  iree_net_carrier_t base;

  // Captured sends in submission order.
  std::vector<CapturedSend> sends;

  // Optional status for the next send submission.
  iree_status_code_t next_send_error = IREE_STATUS_OK;

  static void Destroy(iree_net_carrier_t* carrier) {}

  static void SetRecvHandler(iree_net_carrier_t* carrier,
                             iree_net_carrier_recv_handler_t handler) {
    carrier->recv_handler = handler;
  }

  static iree_status_t Activate(iree_net_carrier_t* carrier) {
    iree_net_carrier_set_state(carrier, IREE_NET_CARRIER_STATE_ACTIVE);
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      iree_net_carrier_t* carrier,
      iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
    iree_net_carrier_set_state(carrier, IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) callback(user_data);
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(
      iree_net_carrier_t* carrier) {
    return {1024 * 1024, 64};
  }

  static iree_status_t Send(iree_net_carrier_t* carrier,
                            const iree_net_send_params_t* params) {
    MockCarrier* mock = reinterpret_cast<MockCarrier*>(carrier);
    if (mock->next_send_error != IREE_STATUS_OK) {
      iree_status_code_t error = mock->next_send_error;
      mock->next_send_error = IREE_STATUS_OK;
      return iree_status_from_code(error);
    }

    CapturedSend captured;
    captured.user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      iree_async_span_t span = params->data.values[i];
      uint8_t* pointer = iree_async_span_ptr(span);
      captured.data.insert(captured.data.end(), pointer, pointer + span.length);
    }
    mock->sends.push_back(std::move(captured));
    return iree_ok_status();
  }

  static iree_status_t CarrierBeginSend(
      iree_net_carrier_t* carrier, iree_host_size_t size, void** out_ptr,
      iree_net_carrier_send_handle_t* out_handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t CarrierCommitSend(
      iree_net_carrier_t* carrier, iree_net_carrier_send_handle_t handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void CarrierAbortSend(iree_net_carrier_t* carrier,
                               iree_net_carrier_send_handle_t handle) {}

  static iree_status_t Shutdown(iree_net_carrier_t* carrier) {
    return iree_ok_status();
  }

  static iree_status_t DirectWrite(
      iree_net_carrier_t* carrier,
      const iree_net_direct_write_params_t* params) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t DirectRead(iree_net_carrier_t* carrier,
                                  const iree_net_direct_read_params_t* params) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t RegisterBuffer(iree_net_carrier_t* carrier,
                                      iree_async_region_t* region,
                                      iree_net_remote_handle_t* out_handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void UnregisterBuffer(iree_net_carrier_t* carrier,
                               iree_net_remote_handle_t handle) {}

  static const iree_net_carrier_vtable_t kVtable;

  static std::unique_ptr<MockCarrier> Create() {
    auto mock = std::make_unique<MockCarrier>();
    iree_net_carrier_callback_t send_callback;
    send_callback.fn = iree_net_frame_sender_dispatch_carrier_completion;
    send_callback.user_data = nullptr;
    iree_net_carrier_initialize(&kVtable, IREE_NET_CARRIER_CAPABILITY_RELIABLE,
                                0, 8, send_callback, iree_allocator_system(),
                                &mock->base);
    return mock;
  }
};

const iree_net_carrier_vtable_t MockCarrier::kVtable = {
    MockCarrier::Destroy,          MockCarrier::SetRecvHandler,
    MockCarrier::Activate,         MockCarrier::Deactivate,
    MockCarrier::QuerySendBudget,  MockCarrier::Send,
    MockCarrier::CarrierBeginSend, MockCarrier::CarrierCommitSend,
    MockCarrier::CarrierAbortSend, MockCarrier::Shutdown,
    MockCarrier::DirectWrite,      MockCarrier::DirectRead,
    MockCarrier::RegisterBuffer,   MockCarrier::UnregisterBuffer,
};

struct MockEndpoint {
  // Installed endpoint callbacks.
  iree_net_message_endpoint_callbacks_t callbacks = {};

  // Carrier used for send forwarding.
  MockCarrier* carrier = nullptr;

  // Whether the endpoint has been activated.
  bool activated = false;

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<MockEndpoint*>(self)->callbacks = callbacks;
  }

  static iree_status_t Activate(void* self) {
    static_cast<MockEndpoint*>(self)->activated = true;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    static_cast<MockEndpoint*>(self)->activated = false;
    if (callback) callback(user_data);
    return iree_ok_status();
  }

  static iree_status_t Send(
      void* self, const iree_net_message_endpoint_send_params_t* params) {
    MockEndpoint* mock = static_cast<MockEndpoint*>(self);
    iree_net_send_params_t carrier_params = {};
    carrier_params.data = params->data;
    carrier_params.flags = IREE_NET_SEND_FLAG_NONE;
    carrier_params.user_data = params->user_data;
    return iree_net_carrier_send(&mock->carrier->base, &carrier_params);
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    (void)self;
    return {1024 * 1024, 64};
  }

  static iree_status_t BeginSend(void* self, iree_host_size_t size,
                                 void** out_ptr,
                                 iree_net_carrier_send_handle_t* out_handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t CommitSend(void* self,
                                  iree_net_carrier_send_handle_t handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void AbortSend(void* self, iree_net_carrier_send_handle_t handle) {}

  static const iree_net_message_endpoint_vtable_t vtable;

  iree_net_message_endpoint_t as_endpoint() { return {this, &vtable}; }

  iree_status_t InjectCredit(uint32_t credit_delta) {
    std::vector<uint8_t> frame(IREE_NET_BULK_FRAME_HEADER_SIZE);
    iree_net_bulk_frame_header_t header;
    iree_net_bulk_frame_header_initialize(
        IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
        /*transfer_id=*/0, credit_delta, /*chunk_offset=*/0,
        /*chunk_length=*/0, /*sequence=*/0, &header);
    memcpy(frame.data(), &header, sizeof(header));

    iree_const_byte_span_t message =
        iree_make_const_byte_span(frame.data(), frame.size());
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(frame.data(), frame.size());
    return callbacks.on_message(callbacks.user_data, message, &lease);
  }
};

const iree_net_message_endpoint_vtable_t MockEndpoint::vtable = {
    MockEndpoint::SetCallbacks,    MockEndpoint::Activate,
    MockEndpoint::Deactivate,      MockEndpoint::Send,
    MockEndpoint::QuerySendBudget, MockEndpoint::BeginSend,
    MockEndpoint::CommitSend,      MockEndpoint::AbortSend,
};

struct BulkChannelCallbacks {
  // Send completion user data values observed by the bulk channel.
  std::vector<uint64_t> send_completions;

  // Send completion status codes observed by the bulk channel.
  std::vector<iree_status_code_t> send_completion_errors;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
    return iree_ok_status();
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t chunk_offset, uint32_t sequence,
                              iree_net_bulk_frame_flags_t flags,
                              iree_const_byte_span_t chunk_data,
                              iree_async_buffer_lease_t* lease) {
    return iree_ok_status();
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    return iree_ok_status();
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t abort_data,
                               iree_async_buffer_lease_t* lease) {
    return iree_ok_status();
  }

  static void OnTransportError(void* user_data, iree_status_t status) {
    iree_status_free(status);
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    BulkChannelCallbacks* callbacks =
        static_cast<BulkChannelCallbacks*>(user_data);
    callbacks->send_completions.push_back(operation_user_data);
    callbacks->send_completion_errors.push_back(iree_status_code(status));
    iree_status_free(status);
  }

  iree_net_bulk_channel_callbacks_t MakeCallbacks() {
    iree_net_bulk_channel_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_start = OnStart;
    callbacks.on_data = OnData;
    callbacks.on_complete = OnComplete;
    callbacks.on_abort = OnAbort;
    callbacks.on_transport_error = OnTransportError;
    callbacks.on_send_complete = OnSendComplete;
    callbacks.user_data = this;
    return callbacks;
  }
};

static iree_net_bulk_frame_header_t ParseBulkHeader(
    const std::vector<uint8_t>& message) {
  iree_net_bulk_frame_header_t header;
  memcpy(&header, message.data(), sizeof(header));
  return header;
}

static void DeinitializeDownloadTransfer(void* user_data,
                                         iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_download_transfer_deinitialize(
      iree_hal_remote_server_bulk_download_transfer_storage(transfer));
}

static iree_status_t AllocateDownloadScheduler(
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler) {
  iree_hal_remote_bulk_transfer_scheduler_options_t options =
      iree_hal_remote_bulk_transfer_scheduler_options_default();
  options.capacity = 4;
  options.user_storage_size =
      sizeof(iree_hal_remote_server_bulk_download_transfer_t);
  options.user_storage_alignment =
      iree_alignof(iree_hal_remote_server_bulk_download_transfer_t);
  iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {};
  callbacks.deinitialize = DeinitializeDownloadTransfer;
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

static void UnusedReadyCallback(void* user_data,
                                iree_async_semaphore_timepoint_t* timepoint,
                                iree_status_t status) {
  (void)user_data;
  (void)timepoint;
  iree_status_free(status);
}

class BulkDownloadSenderTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kTransferKind = 2;
  static constexpr uint64_t kTransferId = 42;

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
    IREE_ASSERT_OK(
        AllocateDownloadScheduler(&session_.bulk_transfer_scheduler));

    carrier_ = MockCarrier::Create();
    endpoint_.carrier = carrier_.get();
    IREE_ASSERT_OK(iree_net_bulk_channel_create(
        endpoint_.as_endpoint(), nullptr,
        TestBufferPool(/*buffer_count=*/16, /*buffer_size=*/1024).release(),
        channel_callbacks_.MakeCallbacks(), iree_allocator_system(),
        &bulk_channel_));
    IREE_ASSERT_OK(iree_net_bulk_channel_activate(bulk_channel_));
    session_.bulk_channel = bulk_channel_;
  }

  void TearDown() override {
    session_.bulk_channel = nullptr;
    iree_net_bulk_channel_release(bulk_channel_);
    bulk_channel_ = nullptr;
    iree_hal_remote_bulk_transfer_scheduler_free(
        session_.bulk_transfer_scheduler);
    session_.bulk_transfer_scheduler = nullptr;
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

  iree_status_t InsertTransfer(uint64_t total_length,
                               iree_net_bulk_transfer_t** out_table_transfer) {
    *out_table_transfer = nullptr;
    iree_net_bulk_transfer_t* table_transfer = nullptr;
    iree_status_t status = iree_hal_remote_bulk_transfer_scheduler_insert_peer(
        session_.bulk_transfer_scheduler, kTransferId, total_length,
        kTransferKind, &table_transfer);
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_bulk_download_transfer_t* transfer =
          iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
      status = iree_hal_remote_server_bulk_download_transfer_initialize(
          &server_, &session_, session_.session_id, kTransferId, device_,
          /*source_buffer=*/nullptr, /*source_offset=*/0,
          IREE_HAL_WRITE_FLAG_NONE, UnusedReadyCallback,
          iree_allocator_system(), transfer);
    }
    if (iree_status_is_ok(status)) {
      *out_table_transfer = table_transfer;
    }
    return status;
  }

  void PrepareStagedData(iree_net_bulk_transfer_t* table_transfer,
                         uint64_t staging_offset,
                         iree_host_size_t staging_length,
                         uint64_t next_staging_offset) {
    for (iree_host_size_t i = 0; i < staging_length; ++i) {
      staged_data_[i] = static_cast<uint8_t>(i + 1);
    }
    iree_hal_remote_server_bulk_download_transfer_t* transfer =
        iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_START_SENT |
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY;
    transfer->staging_contents =
        iree_make_byte_span(staged_data_, sizeof(staged_data_));
    transfer->staging_offset = staging_offset;
    transfer->staging_length = staging_length;
    transfer->next_staging_offset = next_staging_offset;
  }

  iree_status_t CloneSignalList(
      iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value,
      iree_hal_remote_server_bulk_download_transfer_t* transfer) {
    iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
    iree_hal_semaphore_list_t signal_list = {
        IREE_ARRAYSIZE(signal_semaphores),
        signal_semaphores,
        &signal_value,
    };
    return iree_hal_semaphore_list_clone(&signal_list, iree_allocator_system(),
                                         &transfer->signal_semaphore_list);
  }

  // Local-task driver used for semaphore creation.
  iree_hal_driver_t* driver_ = nullptr;

  // Local-task device used for semaphore creation.
  iree_hal_device_t* device_ = nullptr;

  // Stack server shell retained by transfer state.
  iree_hal_remote_server_t server_ = {};

  // Stack session shell containing the scheduler under test.
  iree_hal_remote_server_session_t session_ = {};

  // Capturing carrier backing |bulk_channel_|.
  std::unique_ptr<MockCarrier> carrier_;

  // Capturing endpoint backing |bulk_channel_|.
  MockEndpoint endpoint_;

  // Bulk channel callback capture state.
  BulkChannelCallbacks channel_callbacks_;

  // Bulk channel used by sender operations.
  iree_net_bulk_channel_t* bulk_channel_ = nullptr;

  // Stable staged DATA bytes for zero-copy channel sends.
  uint8_t staged_data_[16] = {};
};

TEST_F(BulkDownloadSenderTest, CreditExhaustionKeepsDataReady) {
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(InsertTransfer(/*total_length=*/4, &table_transfer));
  PrepareStagedData(table_transfer, /*staging_offset=*/0,
                    /*staging_length=*/4, /*next_staging_offset=*/4);

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_try_send_locked(&session_, bulk_channel_,
                                                       table_transfer);
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  EXPECT_TRUE(carrier_->sends.empty());
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  EXPECT_TRUE(iree_any_bit_set(
      transfer->flags,
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY));
  EXPECT_EQ(transfer->pending_operation_count, 0u);
}

TEST_F(BulkDownloadSenderTest, CreditAllowsFinalDataSend) {
  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(InsertTransfer(/*total_length=*/4, &table_transfer));
  PrepareStagedData(table_transfer, /*staging_offset=*/0,
                    /*staging_length=*/4, /*next_staging_offset=*/4);

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_try_send_locked(&session_, bulk_channel_,
                                                       table_transfer);
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  ASSERT_EQ(carrier_->sends.size(), 1u);
  iree_net_bulk_frame_header_t header =
      ParseBulkHeader(carrier_->sends[0].data);
  EXPECT_EQ(header.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(header.transfer_id, kTransferId);
  EXPECT_EQ(header.chunk_offset, 0u);
  EXPECT_EQ(header.sequence, 0u);
  EXPECT_EQ(header.flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(header.chunk_length, 4u);
  EXPECT_EQ(
      memcmp(carrier_->sends[0].data.data() + IREE_NET_BULK_FRAME_HEADER_SIZE,
             staged_data_, 4),
      0);
}

TEST_F(BulkDownloadSenderTest, CreditAllowsNonFinalDataSend) {
  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(InsertTransfer(/*total_length=*/4, &table_transfer));
  PrepareStagedData(table_transfer, /*staging_offset=*/0,
                    /*staging_length=*/2, /*next_staging_offset=*/2);

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_try_send_locked(&session_, bulk_channel_,
                                                       table_transfer);
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  ASSERT_EQ(carrier_->sends.size(), 1u);
  iree_net_bulk_frame_header_t header =
      ParseBulkHeader(carrier_->sends[0].data);
  EXPECT_EQ(header.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(header.flags, IREE_NET_BULK_FRAME_FLAG_NONE);
  EXPECT_EQ(header.chunk_length, 2u);
}

TEST_F(BulkDownloadSenderTest, PermanentSendFailureReleasesTransfer) {
  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(InsertTransfer(/*total_length=*/4, &table_transfer));
  PrepareStagedData(table_transfer, /*staging_offset=*/0,
                    /*staging_length=*/4, /*next_staging_offset=*/4);
  carrier_->next_send_error = IREE_STATUS_INTERNAL;

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_try_send_locked(&session_, bulk_channel_,
                                                       table_transfer);
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                session_.bulk_transfer_scheduler),
            0u);
}

TEST_F(BulkDownloadSenderTest, PeerCompleteSignalsSignalSemaphore) {
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(InsertTransfer(/*total_length=*/0, &table_transfer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &signal_semaphore));
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  IREE_ASSERT_OK(
      CloneSignalList(signal_semaphore, /*signal_value=*/1, transfer));

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  IREE_EXPECT_OK(iree_hal_remote_server_bulk_download_on_complete_locked(
      &session_, table_transfer, kTransferId));
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  uint64_t current_value = 0;
  IREE_EXPECT_OK(iree_hal_semaphore_query(signal_semaphore, &current_value));
  EXPECT_EQ(current_value, 1u);
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                session_.bulk_transfer_scheduler),
            0u);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_F(BulkDownloadSenderTest, PeerAbortFailsSignalSemaphore) {
  iree_net_bulk_transfer_t* table_transfer = nullptr;
  IREE_ASSERT_OK(InsertTransfer(/*total_length=*/0, &table_transfer));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &signal_semaphore));
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  IREE_ASSERT_OK(
      CloneSignalList(signal_semaphore, /*signal_value=*/1, transfer));

  iree_slim_mutex_lock(&session_.bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_fail_locked(
      &session_, table_transfer,
      iree_make_status(IREE_STATUS_ABORTED,
                       "remote client aborted bulk transfer"));
  iree_slim_mutex_unlock(&session_.bulk_transfer_mutex);

  uint64_t current_value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_query(signal_semaphore, &current_value));
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(
                session_.bulk_transfer_scheduler),
            0u);
  iree_hal_semaphore_release(signal_semaphore);
}

}  // namespace
