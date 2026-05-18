// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_BULK_TEST_UTIL_H_
#define IREE_HAL_REMOTE_SERVER_BULK_TEST_UTIL_H_

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"

namespace iree::hal::remote::server::testing {

class TestBufferPool {
 public:
  TestBufferPool() = default;
  TestBufferPool(const TestBufferPool&) = delete;
  TestBufferPool& operator=(const TestBufferPool&) = delete;

  iree_status_t Initialize(iree_host_size_t buffer_count,
                           iree_host_size_t buffer_size) {
    if (pool_) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "test buffer pool is already initialized");
    }
    if (buffer_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "test buffer pool count too large");
    }

    iree_host_size_t total_size = 0;
    if (!iree_host_size_checked_mul(buffer_count, buffer_size, &total_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "test buffer pool size overflow");
    }

    void* memory = malloc(total_size);
    if (!memory) return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    memset(memory, 0, total_size);

    iree_async_region_t* region =
        static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
    if (!region) {
      free(memory);
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    memset(region, 0, sizeof(*region));
    iree_atomic_ref_count_init(&region->ref_count);
    region->destroy_fn = DestroyRegion;
    region->base_ptr = memory;
    region->length = total_size;
    region->buffer_size = buffer_size;
    region->buffer_count = static_cast<uint32_t>(buffer_count);

    iree_status_t status = iree_async_buffer_pool_allocate(
        region, iree_allocator_system(), &pool_);
    iree_async_region_release(region);
    return status;
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

  // Owned buffer pool handed to the bulk channel.
  iree_async_buffer_pool_t* pool_ = nullptr;
};

struct CapturedSend {
  // Concatenated scatter-gather send bytes captured at the carrier boundary.
  std::vector<uint8_t> data;

  // Total scatter-gather send byte length.
  iree_host_size_t total_length = 0;

  // Carrier completion user data for the captured send.
  uint64_t carrier_operation_user_data = 0;

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

  // Whether send payload bytes are captured into |CapturedSend::data|.
  bool capture_send_payload = true;

  void CompleteSend(iree_host_size_t send_index, iree_status_t status) {
    CapturedSend& captured = sends[send_index];
    captured.completed = true;
    base.callback.fn(base.callback.user_data,
                     captured.carrier_operation_user_data, status,
                     captured.total_length, nullptr);
  }

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
    captured.carrier_operation_user_data = params->user_data;
    iree_host_size_t total_length = 0;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      if (!iree_host_size_checked_add(
              total_length, params->data.values[i].length, &total_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "mock send payload size overflow");
      }
    }
    captured.total_length = total_length;
    iree_host_size_t capture_length =
        mock->capture_send_payload
            ? total_length
            : iree_min(total_length, IREE_NET_BULK_FRAME_HEADER_SIZE);
    captured.data.reserve(capture_length);
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      if (captured.data.size() == capture_length) break;
      iree_async_span_t span = params->data.values[i];
      uint8_t* pointer = iree_async_span_ptr(span);
      iree_host_size_t span_capture_length =
          iree_min(span.length, capture_length - captured.data.size());
      captured.data.insert(captured.data.end(), pointer,
                           pointer + span_capture_length);
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

  static const iree_net_carrier_vtable_t* VTable() {
    static const iree_net_carrier_vtable_t vtable = {
        MockCarrier::Destroy,          MockCarrier::SetRecvHandler,
        MockCarrier::Activate,         MockCarrier::Deactivate,
        MockCarrier::QuerySendBudget,  MockCarrier::Send,
        MockCarrier::CarrierBeginSend, MockCarrier::CarrierCommitSend,
        MockCarrier::CarrierAbortSend, MockCarrier::Shutdown,
        MockCarrier::DirectWrite,      MockCarrier::DirectRead,
        MockCarrier::RegisterBuffer,   MockCarrier::UnregisterBuffer,
    };
    return &vtable;
  }

  static std::unique_ptr<MockCarrier> Create() {
    auto mock = std::make_unique<MockCarrier>();
    iree_net_carrier_callback_t send_callback;
    send_callback.fn = iree_net_frame_sender_dispatch_carrier_completion;
    send_callback.user_data = nullptr;
    iree_net_carrier_initialize(VTable(), IREE_NET_CARRIER_CAPABILITY_RELIABLE,
                                0, 8, send_callback, iree_allocator_system(),
                                &mock->base);
    return mock;
  }
};

struct MockEndpoint {
  // Installed endpoint callbacks.
  iree_net_message_endpoint_callbacks_t callbacks = {};

  // Carrier used for send forwarding.
  MockCarrier* carrier = nullptr;

  // Cumulative peer DATA chunk receive credit limit injected so far.
  uint64_t remote_credit_limit = 0;

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

  static const iree_net_message_endpoint_vtable_t* VTable() {
    static const iree_net_message_endpoint_vtable_t vtable = {
        MockEndpoint::SetCallbacks,    MockEndpoint::Activate,
        MockEndpoint::Deactivate,      MockEndpoint::Send,
        MockEndpoint::QuerySendBudget, MockEndpoint::BeginSend,
        MockEndpoint::CommitSend,      MockEndpoint::AbortSend,
    };
    return &vtable;
  }

  iree_net_message_endpoint_t as_endpoint() { return {this, VTable()}; }

  iree_status_t InjectCredit(uint32_t credit_delta) {
    if (remote_credit_limit > UINT64_MAX - (uint64_t)credit_delta) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock bulk credit limit overflow");
    }
    remote_credit_limit += credit_delta;
    std::vector<uint8_t> frame(IREE_NET_BULK_FRAME_HEADER_SIZE);
    iree_net_bulk_frame_header_t header;
    iree_net_bulk_frame_header_initialize(
        IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
        /*transfer_id=*/0, remote_credit_limit, /*chunk_offset=*/0,
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

static inline iree_net_bulk_frame_header_t ParseBulkHeader(
    const std::vector<uint8_t>& message) {
  iree_net_bulk_frame_header_t header;
  memcpy(&header, message.data(), sizeof(header));
  return header;
}

}  // namespace iree::hal::remote::server::testing

#endif  // IREE_HAL_REMOTE_SERVER_BULK_TEST_UTIL_H_
