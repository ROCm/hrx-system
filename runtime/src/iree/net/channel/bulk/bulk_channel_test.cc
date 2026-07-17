// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/bulk_channel.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Test buffer pool
//===----------------------------------------------------------------------===//

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

    iree_status_t status =
        iree_async_buffer_pool_create(region_, iree_allocator_system(), &pool_);
    IREE_CHECK_OK(status);
    iree_async_region_release(region_);
  }

  ~TestBufferPool() { iree_async_buffer_pool_release(pool_); }

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

  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* pool_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Mock carrier for send path
//===----------------------------------------------------------------------===//

struct CapturedSend {
  // Concatenated scatter-gather send bytes captured at the carrier boundary.
  std::vector<uint8_t> data;

  // Carrier completion user data for the captured send.
  uint64_t user_data;

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

  // Whether send submissions complete synchronously inside Send.
  bool auto_complete = true;

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

    if (mock->auto_complete) {
      mock->CompleteSend(mock->sends.size() - 1, iree_ok_status());
    }
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

  void CompleteSend(size_t send_index, iree_status_t status) {
    ASSERT_LT(send_index, sends.size());
    ASSERT_FALSE(sends[send_index].completed);
    sends[send_index].completed = true;
    base.callback.fn(base.callback.user_data, IREE_NET_CARRIER_COMPLETION_SEND,
                     sends[send_index].user_data, status,
                     sends[send_index].data.size(), nullptr);
  }

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

//===----------------------------------------------------------------------===//
// Mock message endpoint
//===----------------------------------------------------------------------===//

struct MockEndpoint {
  // Installed endpoint callbacks.
  iree_net_message_endpoint_callbacks_t callbacks = {};

  // Carrier used for send forwarding.
  MockCarrier* carrier = nullptr;

  // Whether the endpoint has been activated.
  bool activated = false;

  // Number of endpoint deactivation requests.
  iree_host_size_t deactivate_count = 0;

  // Optional status for the next activation.
  iree_status_code_t next_activate_error = IREE_STATUS_OK;

  // Send budget reported by QuerySendBudget.
  iree_net_carrier_send_budget_t send_budget = {1024 * 1024, 64};

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<MockEndpoint*>(self)->callbacks = callbacks;
  }

  static iree_status_t Activate(void* self) {
    MockEndpoint* mock = static_cast<MockEndpoint*>(self);
    if (mock->next_activate_error != IREE_STATUS_OK) {
      iree_status_code_t error = mock->next_activate_error;
      mock->next_activate_error = IREE_STATUS_OK;
      return iree_status_from_code(error);
    }
    mock->activated = true;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    MockEndpoint* mock = static_cast<MockEndpoint*>(self);
    mock->activated = false;
    ++mock->deactivate_count;
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
    return static_cast<MockEndpoint*>(self)->send_budget;
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

  iree_status_t InjectMessage(const std::vector<uint8_t>& data) {
    iree_const_byte_span_t message =
        iree_make_const_byte_span(data.data(), data.size());
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(const_cast<uint8_t*>(data.data()),
                                          data.size());
    return callbacks.on_message(callbacks.user_data, message, &lease);
  }

  void InjectError(iree_status_t status) {
    if (callbacks.on_error) {
      callbacks.on_error(callbacks.user_data, status);
    }
  }

  void InjectSendReady() {
    if (callbacks.on_send_ready) {
      callbacks.on_send_ready(callbacks.user_data);
    }
  }
};

const iree_net_message_endpoint_vtable_t MockEndpoint::vtable = {
    MockEndpoint::SetCallbacks,    MockEndpoint::Activate,
    MockEndpoint::Deactivate,      MockEndpoint::Send,
    MockEndpoint::QuerySendBudget, MockEndpoint::BeginSend,
    MockEndpoint::CommitSend,      MockEndpoint::AbortSend,
};

//===----------------------------------------------------------------------===//
// Test context for capturing callbacks
//===----------------------------------------------------------------------===//

struct StartRecord {
  // Transfer ID announced by the START frame.
  uint64_t transfer_id = 0;

  // Total transfer size announced by the START frame.
  uint64_t total_size = 0;

  // Frame flags announced by the START frame.
  iree_net_bulk_frame_flags_t flags = 0;
};

struct DataRecord {
  // Transfer ID for the DATA chunk.
  uint64_t transfer_id = 0;

  // Byte offset of the DATA chunk in the transfer.
  uint64_t chunk_offset = 0;

  // DATA chunk sequence number.
  uint32_t sequence = 0;

  // Frame flags attached to the DATA chunk.
  iree_net_bulk_frame_flags_t flags = 0;

  // DATA chunk payload bytes.
  std::vector<uint8_t> payload;
};

struct AbortRecord {
  // Transfer ID aborted by the ABORT frame.
  uint64_t transfer_id = 0;

  // ABORT payload bytes.
  std::vector<uint8_t> payload;
};

struct TestContext {
  // START callback records.
  std::vector<StartRecord> starts;

  // DATA callback records.
  std::vector<DataRecord> data_chunks;

  // COMPLETE callback transfer IDs.
  std::vector<uint64_t> completions;

  // ABORT callback records.
  std::vector<AbortRecord> aborts;

  // Transport error status codes.
  std::vector<iree_status_code_t> transport_errors;

  // Send completion operation user data values.
  std::vector<uint64_t> send_completions;

  // Send completion status codes.
  std::vector<iree_status_code_t> send_completion_errors;

  // Send readiness notification count.
  iree_host_size_t send_ready_count = 0;

  // Credit replenishment deltas.
  std::vector<uint32_t> credit_deltas;

  // Available credit counts after each replenishment.
  std::vector<uint32_t> available_credit_counts;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
    auto* context = static_cast<TestContext*>(user_data);
    context->starts.push_back({transfer_id, total_size, flags});
    return iree_ok_status();
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t chunk_offset, uint32_t sequence,
                              iree_net_bulk_frame_flags_t flags,
                              iree_const_byte_span_t chunk_data,
                              iree_async_buffer_lease_t* lease) {
    auto* context = static_cast<TestContext*>(user_data);
    DataRecord record;
    record.transfer_id = transfer_id;
    record.chunk_offset = chunk_offset;
    record.sequence = sequence;
    record.flags = flags;
    record.payload.assign(chunk_data.data,
                          chunk_data.data + chunk_data.data_length);
    context->data_chunks.push_back(std::move(record));
    return iree_ok_status();
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    auto* context = static_cast<TestContext*>(user_data);
    context->completions.push_back(transfer_id);
    return iree_ok_status();
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t abort_data,
                               iree_async_buffer_lease_t* lease) {
    auto* context = static_cast<TestContext*>(user_data);
    AbortRecord record;
    record.transfer_id = transfer_id;
    record.payload.assign(abort_data.data,
                          abort_data.data + abort_data.data_length);
    context->aborts.push_back(std::move(record));
    return iree_ok_status();
  }

  static void OnTransportError(void* user_data, iree_status_t status) {
    auto* context = static_cast<TestContext*>(user_data);
    context->transport_errors.push_back(iree_status_code(status));
    iree_status_ignore(status);
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    auto* context = static_cast<TestContext*>(user_data);
    context->send_completions.push_back(operation_user_data);
    context->send_completion_errors.push_back(iree_status_code(status));
    iree_status_ignore(status);
  }

  static void OnSendReady(void* user_data) {
    auto* context = static_cast<TestContext*>(user_data);
    ++context->send_ready_count;
  }

  static void OnCredit(void* user_data, uint32_t credit_delta,
                       uint32_t available_credit_count) {
    auto* context = static_cast<TestContext*>(user_data);
    context->credit_deltas.push_back(credit_delta);
    context->available_credit_counts.push_back(available_credit_count);
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
    callbacks.on_send_ready = OnSendReady;
    callbacks.on_credit = OnCredit;
    callbacks.user_data = this;
    return callbacks;
  }
};

//===----------------------------------------------------------------------===//
// Message construction helpers
//===----------------------------------------------------------------------===//

static std::vector<uint8_t> BuildBulkFrame(
    iree_net_bulk_frame_type_t type, iree_net_bulk_frame_flags_t flags,
    uint64_t transfer_id, uint64_t total_size, uint64_t chunk_offset,
    uint32_t sequence, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> message(IREE_NET_BULK_FRAME_HEADER_SIZE +
                               payload.size());
  iree_net_bulk_frame_header_t header;
  iree_net_bulk_frame_header_initialize(
      type, flags, transfer_id, total_size, chunk_offset,
      static_cast<uint32_t>(payload.size()), sequence, &header);
  memcpy(message.data(), &header, sizeof(header));
  if (!payload.empty()) {
    memcpy(message.data() + IREE_NET_BULK_FRAME_HEADER_SIZE, payload.data(),
           payload.size());
  }
  return message;
}

static iree_net_bulk_frame_header_t ParseBulkHeader(
    const std::vector<uint8_t>& message) {
  iree_net_bulk_frame_header_t header;
  memcpy(&header, message.data(), sizeof(header));
  return header;
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class BulkChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    carrier_ = MockCarrier::Create();
    endpoint_.carrier = carrier_.get();
  }

  void TearDown() override {
    iree_net_bulk_channel_detach(channel_);
    if (channel_ && endpoint_.activated) {
      IREE_ASSERT_OK(iree_net_message_endpoint_deactivate(
          endpoint_.as_endpoint(), /*callback=*/nullptr,
          /*user_data=*/nullptr));
    }
    iree_net_bulk_channel_release(channel_);
    channel_ = nullptr;
  }

  iree_async_buffer_pool_t* CreatePool() {
    auto pool = std::make_unique<TestBufferPool>(/*buffer_count=*/16,
                                                 /*buffer_size=*/1024);
    return pool->release();
  }

  void CreateAndActivate() {
    IREE_ASSERT_OK(iree_net_bulk_channel_create(
        endpoint_.as_endpoint(), nullptr, CreatePool(),
        context_.MakeCallbacks(), iree_allocator_system(), &channel_));
    IREE_ASSERT_OK(iree_net_bulk_channel_activate(channel_));
  }

  std::unique_ptr<MockCarrier> carrier_;

  // Endpoint under test.
  MockEndpoint endpoint_;

  // Callback capture context.
  TestContext context_;

  // Channel under test.
  iree_net_bulk_channel_t* channel_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(BulkChannelTest, CreateAndRelease) {
  iree_net_bulk_channel_t* channel = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_channel_create(
      endpoint_.as_endpoint(), nullptr, CreatePool(), context_.MakeCallbacks(),
      iree_allocator_system(), &channel));
  EXPECT_EQ(iree_net_bulk_channel_state(channel),
            IREE_NET_BULK_CHANNEL_STATE_CREATED);
  iree_net_bulk_channel_release(channel);
}

TEST_F(BulkChannelTest, ActivateTransitionsToOperational) {
  CreateAndActivate();
  EXPECT_EQ(iree_net_bulk_channel_state(channel_),
            IREE_NET_BULK_CHANNEL_STATE_OPERATIONAL);
  EXPECT_TRUE(endpoint_.activated);
}

TEST_F(BulkChannelTest, DetachDoesNotDeactivateBorrowedEndpoint) {
  CreateAndActivate();

  iree_net_bulk_channel_detach(channel_);

  EXPECT_EQ(iree_net_bulk_channel_state(channel_),
            IREE_NET_BULK_CHANNEL_STATE_ERROR);
  EXPECT_TRUE(endpoint_.activated);
  EXPECT_EQ(endpoint_.deactivate_count, 0u);
  EXPECT_EQ(endpoint_.callbacks.on_message, nullptr);
}

TEST_F(BulkChannelTest, MissingCallbacksRejects) {
  iree_net_bulk_channel_t* channel = nullptr;
  iree_net_bulk_channel_callbacks_t callbacks = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_channel_create(
                            endpoint_.as_endpoint(), nullptr, CreatePool(),
                            callbacks, iree_allocator_system(), &channel));
}

//===----------------------------------------------------------------------===//
// Receive path
//===----------------------------------------------------------------------===//

TEST_F(BulkChannelTest, ReceiveStartDataComplete) {
  CreateAndActivate();

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_START, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/42, /*total_size=*/4096, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_DATA, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
      /*transfer_id=*/42, /*total_size=*/0, /*chunk_offset=*/128,
      /*sequence=*/7, {0x01, 0x02, 0x03})));
  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_COMPLETE, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/42, /*total_size=*/0, /*chunk_offset=*/0,
      /*sequence=*/0, {})));

  ASSERT_EQ(context_.starts.size(), 1u);
  EXPECT_EQ(context_.starts[0].transfer_id, 42u);
  EXPECT_EQ(context_.starts[0].total_size, 4096u);

  ASSERT_EQ(context_.data_chunks.size(), 1u);
  EXPECT_EQ(context_.data_chunks[0].transfer_id, 42u);
  EXPECT_EQ(context_.data_chunks[0].chunk_offset, 128u);
  EXPECT_EQ(context_.data_chunks[0].sequence, 7u);
  EXPECT_EQ(context_.data_chunks[0].flags,
            IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(context_.data_chunks[0].payload,
            (std::vector<uint8_t>{0x01, 0x02, 0x03}));

  ASSERT_EQ(context_.completions.size(), 1u);
  EXPECT_EQ(context_.completions[0], 42u);
}

TEST_F(BulkChannelTest, ReceiveDataOutOfOrder) {
  CreateAndActivate();

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_START, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/99, /*total_size=*/4, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_DATA, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/99, /*total_size=*/0, /*chunk_offset=*/2,
      /*sequence=*/2, {0x03, 0x04})));
  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_DATA, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/99, /*total_size=*/0, /*chunk_offset=*/0,
      /*sequence=*/1, {0x01, 0x02})));

  ASSERT_EQ(context_.data_chunks.size(), 2u);
  EXPECT_EQ(context_.data_chunks[0].chunk_offset, 2u);
  EXPECT_EQ(context_.data_chunks[0].sequence, 2u);
  EXPECT_EQ(context_.data_chunks[1].chunk_offset, 0u);
  EXPECT_EQ(context_.data_chunks[1].sequence, 1u);
}

TEST_F(BulkChannelTest, ReceiveAbortWithPayload) {
  CreateAndActivate();

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_ABORT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/7, /*total_size=*/0, /*chunk_offset=*/0,
      /*sequence=*/0, {'b', 'a', 'd'})));

  ASSERT_EQ(context_.aborts.size(), 1u);
  EXPECT_EQ(context_.aborts[0].transfer_id, 7u);
  EXPECT_EQ(context_.aborts[0].payload, (std::vector<uint8_t>{'b', 'a', 'd'}));
}

TEST_F(BulkChannelTest, ReceiveCreditReplenishesRemoteChunkCredit) {
  CreateAndActivate();

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/3, /*chunk_offset=*/0,
      /*sequence=*/0, {})));

  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 3u);
  EXPECT_EQ(context_.credit_deltas, (std::vector<uint32_t>{3}));
  EXPECT_EQ(context_.available_credit_counts, (std::vector<uint32_t>{3}));
}

TEST_F(BulkChannelTest, ReceiveCreditIgnoresDuplicateLimit) {
  CreateAndActivate();

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/3, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/3, /*chunk_offset=*/0,
      /*sequence=*/0, {})));

  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 3u);
  EXPECT_EQ(context_.credit_deltas, (std::vector<uint32_t>{3}));
  EXPECT_EQ(context_.available_credit_counts, (std::vector<uint32_t>{3}));
}

TEST_F(BulkChannelTest, ReceiveCreditRejectsWindowOverflow) {
  iree_net_bulk_channel_options_t options =
      iree_net_bulk_channel_options_default();
  options.remote_chunk_credit_capacity = 2;
  IREE_ASSERT_OK(iree_net_bulk_channel_create(
      endpoint_.as_endpoint(), &options, CreatePool(), context_.MakeCallbacks(),
      iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_bulk_channel_activate(channel_));

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/2, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      endpoint_.InjectMessage(BuildBulkFrame(
          IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
          /*transfer_id=*/0, /*total_size=*/3,
          /*chunk_offset=*/0, /*sequence=*/0, {})));
  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 2u);
}

TEST_F(BulkChannelTest, ReceiveBadMagic) {
  CreateAndActivate();
  auto message = BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_START,
                                IREE_NET_BULK_FRAME_FLAG_NONE, 1, 0, 0, 0, {});
  message[0] = 0xFF;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(BulkChannelTest, ReceiveChunkLengthMismatch) {
  CreateAndActivate();
  auto message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_DATA,
                     IREE_NET_BULK_FRAME_FLAG_NONE, 1, 0, 0, 0, {0x01, 0x02});
  auto* header =
      reinterpret_cast<iree_net_bulk_frame_header_t*>(message.data());
  header->chunk_length = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(BulkChannelTest, ReceiveInvalidLifecycleMetadata) {
  CreateAndActivate();

  auto start_message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_START,
                     IREE_NET_BULK_FRAME_FLAG_NONE, 1, 1024, 0, 0, {});
  auto* start_header =
      reinterpret_cast<iree_net_bulk_frame_header_t*>(start_message.data());
  start_header->chunk_offset = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(start_message));

  auto data_message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_DATA,
                     IREE_NET_BULK_FRAME_FLAG_NONE, 1, 0, 0, 0, {0x01});
  auto* data_header =
      reinterpret_cast<iree_net_bulk_frame_header_t*>(data_message.data());
  data_header->total_size = 1024;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(data_message));

  auto complete_message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_COMPLETE,
                     IREE_NET_BULK_FRAME_FLAG_NONE, 1, 0, 0, 0, {});
  auto* complete_header =
      reinterpret_cast<iree_net_bulk_frame_header_t*>(complete_message.data());
  complete_header->sequence = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(complete_message));

  auto abort_message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_ABORT,
                     IREE_NET_BULK_FRAME_FLAG_NONE, 1, 0, 0, 0, {});
  auto* abort_header =
      reinterpret_cast<iree_net_bulk_frame_header_t*>(abort_message.data());
  abort_header->chunk_offset = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(abort_message));

  auto credit_message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_CREDIT,
                     IREE_NET_BULK_FRAME_FLAG_NONE, 0, 1, 0, 0, {});
  auto* credit_header =
      reinterpret_cast<iree_net_bulk_frame_header_t*>(credit_message.data());
  credit_header->transfer_id = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(credit_message));
}

TEST_F(BulkChannelTest, ReceiveCodecFlagRejected) {
  CreateAndActivate();
  auto message =
      BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_DATA,
                     IREE_NET_BULK_FRAME_FLAG_COMPRESSED, 1, 0, 0, 0, {0x01});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        endpoint_.InjectMessage(message));
}

TEST_F(BulkChannelTest, ReceiveUnknownFlagRejected) {
  CreateAndActivate();
  auto message = BuildBulkFrame(IREE_NET_BULK_FRAME_TYPE_DATA,
                                static_cast<iree_net_bulk_frame_flags_t>(0x80),
                                1, 0, 0, 0, {0x01});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(BulkChannelTest, ReceiveUnknownFrameType) {
  CreateAndActivate();
  auto message = BuildBulkFrame(static_cast<iree_net_bulk_frame_type_t>(0xFF),
                                IREE_NET_BULK_FRAME_FLAG_NONE, 1, 0, 0, 0, {});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(BulkChannelTest, TransportErrorTransitionsToError) {
  CreateAndActivate();
  endpoint_.InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport down"));
  EXPECT_EQ(iree_net_bulk_channel_state(channel_),
            IREE_NET_BULK_CHANNEL_STATE_ERROR);
  ASSERT_EQ(context_.transport_errors.size(), 1u);
  EXPECT_EQ(context_.transport_errors[0], IREE_STATUS_UNAVAILABLE);
}

TEST_F(BulkChannelTest, SendReadyCallbackFiresInOperationalState) {
  CreateAndActivate();
  endpoint_.InjectSendReady();
  EXPECT_EQ(context_.send_ready_count, 1u);
}

TEST_F(BulkChannelTest, SendReadyIgnoredInErrorState) {
  CreateAndActivate();
  endpoint_.InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport down"));
  endpoint_.InjectSendReady();
  EXPECT_EQ(context_.send_ready_count, 0u);
}

//===----------------------------------------------------------------------===//
// Send path
//===----------------------------------------------------------------------===//

TEST_F(BulkChannelTest, SendStartDataCompleteAbort) {
  CreateAndActivate();

  uint8_t chunk_bytes[] = {0x10, 0x11, 0x12, 0x13};
  iree_async_span_t chunk_span =
      iree_async_span_from_ptr(chunk_bytes, sizeof(chunk_bytes));
  iree_async_span_list_t chunk_payload =
      iree_async_span_list_make(&chunk_span, 1);

  uint8_t abort_bytes[] = {'n', 'o'};
  iree_async_span_t abort_span =
      iree_async_span_from_ptr(abort_bytes, sizeof(abort_bytes));
  iree_async_span_list_t abort_payload =
      iree_async_span_list_make(&abort_span, 1);

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/1, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_start(
      channel_, /*transfer_id=*/3, /*total_size=*/1024,
      IREE_NET_BULK_FRAME_FLAG_NONE, /*operation_user_data=*/10));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, /*transfer_id=*/3, /*chunk_offset=*/64, /*sequence=*/2,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, chunk_payload,
      /*operation_user_data=*/11));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_complete(
      channel_, /*transfer_id=*/3, /*operation_user_data=*/12));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_abort(
      channel_, /*transfer_id=*/4, abort_payload, /*operation_user_data=*/13));

  ASSERT_EQ(carrier_->sends.size(), 4u);
  ASSERT_EQ(context_.send_completions.size(), 4u);
  EXPECT_EQ(context_.send_completions, (std::vector<uint64_t>{10, 11, 12, 13}));

  iree_net_bulk_frame_header_t start_header =
      ParseBulkHeader(carrier_->sends[0].data);
  EXPECT_EQ(start_header.type, IREE_NET_BULK_FRAME_TYPE_START);
  EXPECT_EQ(start_header.transfer_id, 3u);
  EXPECT_EQ(start_header.total_size, 1024u);
  EXPECT_EQ(start_header.chunk_length, 0u);

  iree_net_bulk_frame_header_t data_header =
      ParseBulkHeader(carrier_->sends[1].data);
  EXPECT_EQ(data_header.type, IREE_NET_BULK_FRAME_TYPE_DATA);
  EXPECT_EQ(data_header.transfer_id, 3u);
  EXPECT_EQ(data_header.chunk_offset, 64u);
  EXPECT_EQ(data_header.sequence, 2u);
  EXPECT_EQ(data_header.flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(data_header.chunk_length, sizeof(chunk_bytes));
  EXPECT_EQ(
      memcmp(carrier_->sends[1].data.data() + IREE_NET_BULK_FRAME_HEADER_SIZE,
             chunk_bytes, sizeof(chunk_bytes)),
      0);

  iree_net_bulk_frame_header_t complete_header =
      ParseBulkHeader(carrier_->sends[2].data);
  EXPECT_EQ(complete_header.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(complete_header.transfer_id, 3u);

  iree_net_bulk_frame_header_t abort_header =
      ParseBulkHeader(carrier_->sends[3].data);
  EXPECT_EQ(abort_header.type, IREE_NET_BULK_FRAME_TYPE_ABORT);
  EXPECT_EQ(abort_header.transfer_id, 4u);
  EXPECT_EQ(abort_header.chunk_length, sizeof(abort_bytes));
}

TEST_F(BulkChannelTest, SendDataRequiresRemoteChunkCredit) {
  CreateAndActivate();

  uint8_t chunk_bytes[] = {0x10};
  iree_async_span_t chunk_span =
      iree_async_span_from_ptr(chunk_bytes, sizeof(chunk_bytes));
  iree_async_span_list_t chunk_payload =
      iree_async_span_list_make(&chunk_span, 1);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_channel_send_data(
                            channel_, /*transfer_id=*/3, /*chunk_offset=*/0,
                            /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
                            chunk_payload, /*operation_user_data=*/1));
  EXPECT_TRUE(carrier_->sends.empty());

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/1, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, /*transfer_id=*/3, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, chunk_payload, /*operation_user_data=*/2));
  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 0u);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_channel_send_data(
                            channel_, /*transfer_id=*/3, /*chunk_offset=*/1,
                            /*sequence=*/1, IREE_NET_BULK_FRAME_FLAG_NONE,
                            chunk_payload, /*operation_user_data=*/3));

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/2, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 1u);
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, /*transfer_id=*/3, /*chunk_offset=*/1, /*sequence=*/1,
      IREE_NET_BULK_FRAME_FLAG_NONE, chunk_payload, /*operation_user_data=*/4));
  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 0u);
}

TEST_F(BulkChannelTest, SendDataRefundsCreditOnSubmitFailure) {
  CreateAndActivate();

  uint8_t chunk_bytes[] = {0x10};
  iree_async_span_t chunk_span =
      iree_async_span_from_ptr(chunk_bytes, sizeof(chunk_bytes));
  iree_async_span_list_t chunk_payload =
      iree_async_span_list_make(&chunk_span, 1);

  IREE_ASSERT_OK(endpoint_.InjectMessage(BuildBulkFrame(
      IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/1, /*chunk_offset=*/0,
      /*sequence=*/0, {})));
  carrier_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_channel_send_data(
                            channel_, /*transfer_id=*/3, /*chunk_offset=*/0,
                            /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
                            chunk_payload, /*operation_user_data=*/1));
  EXPECT_EQ(iree_net_bulk_channel_remote_chunk_credit_count(channel_), 1u);
  EXPECT_TRUE(context_.send_completions.empty());
}

TEST_F(BulkChannelTest, SendCredit) {
  CreateAndActivate();

  IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(channel_, /*credit_delta=*/7,
                                                   /*operation_user_data=*/20));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(channel_, /*credit_delta=*/1,
                                                   /*operation_user_data=*/21));
  IREE_ASSERT_OK(iree_net_bulk_channel_refresh_credit(
      channel_, /*operation_user_data=*/22));

  ASSERT_EQ(carrier_->sends.size(), 3u);
  ASSERT_EQ(context_.send_completions.size(), 3u);
  EXPECT_EQ(context_.send_completions, (std::vector<uint64_t>{20, 21, 22}));

  iree_net_bulk_frame_header_t credit_header =
      ParseBulkHeader(carrier_->sends[0].data);
  EXPECT_EQ(credit_header.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  EXPECT_EQ(credit_header.transfer_id, 0u);
  EXPECT_EQ(credit_header.total_size, 7u);
  EXPECT_EQ(credit_header.chunk_offset, 0u);
  EXPECT_EQ(credit_header.chunk_length, 0u);
  EXPECT_EQ(credit_header.sequence, 0u);

  iree_net_bulk_frame_header_t next_credit_header =
      ParseBulkHeader(carrier_->sends[1].data);
  EXPECT_EQ(next_credit_header.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  EXPECT_EQ(next_credit_header.total_size, 8u);

  iree_net_bulk_frame_header_t refresh_credit_header =
      ParseBulkHeader(carrier_->sends[2].data);
  EXPECT_EQ(refresh_credit_header.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  EXPECT_EQ(refresh_credit_header.total_size, 8u);
}

TEST_F(BulkChannelTest, SendBeforeActivateFails) {
  iree_net_bulk_channel_t* channel = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_channel_create(
      endpoint_.as_endpoint(), nullptr, CreatePool(), context_.MakeCallbacks(),
      iree_allocator_system(), &channel));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_bulk_channel_send_start(
                            channel, 1, 0, IREE_NET_BULK_FRAME_FLAG_NONE, 0));

  iree_net_bulk_channel_release(channel);
}

TEST_F(BulkChannelTest, SendAfterErrorFails) {
  CreateAndActivate();
  endpoint_.InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport down"));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_bulk_channel_send_start(
                            channel_, 1, 0, IREE_NET_BULK_FRAME_FLAG_NONE, 0));
}

TEST_F(BulkChannelTest, SendCodecFlagRejected) {
  CreateAndActivate();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_net_bulk_channel_send_start(channel_, 1, 0,
                                       IREE_NET_BULK_FRAME_FLAG_COMPRESSED, 0));
}

TEST_F(BulkChannelTest, SendContextCapacityBackpressures) {
  carrier_->auto_complete = false;
  iree_net_bulk_channel_options_t options =
      iree_net_bulk_channel_options_default();
  options.send_context_capacity = 1;

  IREE_ASSERT_OK(iree_net_bulk_channel_create(
      endpoint_.as_endpoint(), &options, CreatePool(), context_.MakeCallbacks(),
      iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_bulk_channel_activate(channel_));

  IREE_ASSERT_OK(iree_net_bulk_channel_send_start(
      channel_, 1, 0, IREE_NET_BULK_FRAME_FLAG_NONE, 100));
  EXPECT_TRUE(iree_net_bulk_channel_has_pending_sends(channel_));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_bulk_channel_send_start(channel_, 2, 0,
                                       IREE_NET_BULK_FRAME_FLAG_NONE, 101));

  carrier_->CompleteSend(0, iree_ok_status());
  EXPECT_FALSE(iree_net_bulk_channel_has_pending_sends(channel_));
  ASSERT_EQ(context_.send_completions.size(), 1u);
  EXPECT_EQ(context_.send_completions[0], 100u);
}

TEST_F(BulkChannelTest, SendCarrierBackpressureDoesNotComplete) {
  CreateAndActivate();

  carrier_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_channel_send_start(
                            channel_, 1, 0, IREE_NET_BULK_FRAME_FLAG_NONE, 99));
  EXPECT_TRUE(context_.send_completions.empty());
}

TEST_F(BulkChannelTest, QuerySendBudget) {
  CreateAndActivate();
  endpoint_.send_budget.bytes = 4096;
  endpoint_.send_budget.slots = 3;

  iree_net_carrier_send_budget_t budget =
      iree_net_bulk_channel_query_send_budget(channel_);
  EXPECT_EQ(budget.bytes, 4096u);
  EXPECT_EQ(budget.slots, 3u);
}

}  // namespace
}  // namespace net
}  // namespace iree
