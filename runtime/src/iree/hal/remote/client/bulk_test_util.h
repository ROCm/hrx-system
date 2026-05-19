// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_TEST_UTIL_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_TEST_UTIL_H_

#include <cstring>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"

namespace iree::hal::remote::client::testing {

using BulkSendCompleteHook = iree_status_t (*)(void* user_data,
                                               uint64_t operation_user_data,
                                               iree_status_t status);

struct CapturedBulkSend {
  // Concatenated scatter-gather bytes submitted through the endpoint.
  std::vector<uint8_t> data;

  // Total byte length reported to frame-sender completion.
  iree_host_size_t total_length = 0;

  // Frame-sender context pointer forwarded as endpoint user data.
  uint64_t endpoint_user_data = 0;

  // True after the test completes the send.
  bool completed = false;
};

struct MockEndpoint {
  // Endpoint callbacks installed by the bulk channel.
  iree_net_message_endpoint_callbacks_t callbacks = {};

  // Sends captured at the endpoint boundary.
  std::vector<CapturedBulkSend> sends;

  // Status returned by the next endpoint send.
  iree_status_code_t next_send_error = IREE_STATUS_OK;

  // Whether send payload bytes are captured into |CapturedBulkSend::data|.
  bool capture_send_payload = true;

  // Endpoint send budget reported to the bulk channel.
  iree_net_carrier_send_budget_t send_budget = {1024 * 1024, 64};

  // Cumulative peer DATA receive credit limit advertised to the channel.
  uint64_t remote_credit_limit = 0;

  // True after endpoint activation.
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
    MockEndpoint* endpoint = static_cast<MockEndpoint*>(self);
    if (endpoint->next_send_error != IREE_STATUS_OK) {
      iree_status_code_t send_error = endpoint->next_send_error;
      endpoint->next_send_error = IREE_STATUS_OK;
      return iree_status_from_code(send_error);
    }

    CapturedBulkSend captured;
    captured.endpoint_user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      if (!iree_host_size_checked_add(captured.total_length,
                                      params->data.values[i].length,
                                      &captured.total_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "captured send size overflow");
      }
    }
    iree_host_size_t capture_capacity =
        endpoint->capture_send_payload
            ? captured.total_length
            : iree_min(captured.total_length,
                       (iree_host_size_t)IREE_NET_BULK_FRAME_HEADER_SIZE);
    captured.data.reserve(capture_capacity);
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      if (!endpoint->capture_send_payload &&
          captured.data.size() >= IREE_NET_BULK_FRAME_HEADER_SIZE) {
        break;
      }
      iree_async_span_t span = params->data.values[i];
      const uint8_t* span_data = iree_async_span_ptr(span);
      iree_host_size_t capture_length = span.length;
      if (!endpoint->capture_send_payload) {
        capture_length = iree_min(span.length, IREE_NET_BULK_FRAME_HEADER_SIZE -
                                                   captured.data.size());
      }
      captured.data.insert(captured.data.end(), span_data,
                           span_data + capture_length);
    }
    endpoint->sends.push_back(std::move(captured));
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    MockEndpoint* endpoint = static_cast<MockEndpoint*>(self);
    return endpoint->send_budget;
  }

  static iree_status_t BeginSend(void* self, iree_host_size_t size,
                                 void** out_ptr,
                                 iree_net_carrier_send_handle_t* out_handle) {
    (void)self;
    (void)size;
    (void)out_ptr;
    (void)out_handle;
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t CommitSend(void* self,
                                  iree_net_carrier_send_handle_t handle) {
    (void)self;
    (void)handle;
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void AbortSend(void* self, iree_net_carrier_send_handle_t handle) {
    (void)self;
    (void)handle;
  }

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

  iree_status_t InjectMessage(const std::vector<uint8_t>& message) {
    if (!callbacks.on_message) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "mock endpoint has no message callback");
    }
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(const_cast<uint8_t*>(message.data()),
                                          message.size());
    return callbacks.on_message(
        callbacks.user_data,
        iree_make_const_byte_span(message.data(), message.size()), &lease);
  }

  iree_status_t InjectCredit(uint32_t credit_delta) {
    if (remote_credit_limit > UINT64_MAX - (uint64_t)credit_delta) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock bulk credit limit overflow");
    }
    remote_credit_limit += credit_delta;
    std::vector<uint8_t> message(IREE_NET_BULK_FRAME_HEADER_SIZE);
    iree_net_bulk_frame_header_t header;
    iree_net_bulk_frame_header_initialize(
        IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
        /*transfer_id=*/0, remote_credit_limit, /*chunk_offset=*/0,
        /*chunk_length=*/0, /*sequence=*/0, &header);
    memcpy(message.data(), &header, sizeof(header));
    return InjectMessage(message);
  }

  void CompleteSend(iree_host_size_t send_index, iree_status_t status) {
    CapturedBulkSend& captured = sends[send_index];
    captured.completed = true;
    iree_net_frame_sender_dispatch_carrier_completion(
        /*callback_user_data=*/NULL, IREE_NET_CARRIER_COMPLETION_SEND,
        captured.endpoint_user_data, status, captured.total_length,
        /*recv_lease=*/NULL);
  }

  void CompleteAllSends() {
    for (iree_host_size_t i = 0; i < sends.size(); ++i) {
      if (!sends[i].completed) CompleteSend(i, iree_ok_status());
    }
  }
};

struct BulkChannelCallbacks {
  // Send-completion operation user data values observed by the channel.
  std::vector<uint64_t> send_completions;

  // Send-completion status codes observed by the channel.
  std::vector<iree_status_code_t> send_completion_status_codes;

  // Status codes returned by the optional send-completion hook.
  std::vector<iree_status_code_t> send_complete_result_status_codes;

  // Optional hook that consumes the send-completion status.
  BulkSendCompleteHook send_complete_hook = NULL;

  // User data passed to |send_complete_hook|.
  void* send_complete_user_data = NULL;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
    (void)user_data;
    (void)transfer_id;
    (void)total_size;
    (void)flags;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected START");
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t chunk_offset, uint32_t sequence,
                              iree_net_bulk_frame_flags_t flags,
                              iree_const_byte_span_t chunk_data,
                              iree_async_buffer_lease_t* lease) {
    (void)user_data;
    (void)transfer_id;
    (void)chunk_offset;
    (void)sequence;
    (void)flags;
    (void)chunk_data;
    (void)lease;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected DATA");
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    (void)user_data;
    (void)transfer_id;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected COMPLETE");
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t abort_data,
                               iree_async_buffer_lease_t* lease) {
    (void)user_data;
    (void)transfer_id;
    (void)abort_data;
    (void)lease;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected ABORT");
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    BulkChannelCallbacks* callbacks =
        static_cast<BulkChannelCallbacks*>(user_data);
    callbacks->send_completions.push_back(operation_user_data);
    callbacks->send_completion_status_codes.push_back(iree_status_code(status));
    if (callbacks->send_complete_hook) {
      iree_status_t hook_status = callbacks->send_complete_hook(
          callbacks->send_complete_user_data, operation_user_data, status);
      callbacks->send_complete_result_status_codes.push_back(
          iree_status_code(hook_status));
      iree_status_free(hook_status);
    } else {
      iree_status_free(status);
    }
  }

  iree_net_bulk_channel_callbacks_t MakeCallbacks() {
    iree_net_bulk_channel_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_start = OnStart;
    callbacks.on_data = OnData;
    callbacks.on_complete = OnComplete;
    callbacks.on_abort = OnAbort;
    callbacks.on_send_complete = OnSendComplete;
    callbacks.user_data = this;
    return callbacks;
  }
};

static inline iree_net_bulk_frame_header_t ParseBulkHeader(
    const CapturedBulkSend& send) {
  iree_net_bulk_frame_header_t header;
  memset(&header, 0, sizeof(header));
  memcpy(&header, send.data.data(), sizeof(header));
  return header;
}

}  // namespace iree::hal::remote::client::testing

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_TEST_UTIL_H_
