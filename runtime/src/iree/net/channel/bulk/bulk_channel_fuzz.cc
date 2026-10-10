// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/net/channel/bulk/bulk_channel.h"

namespace {

struct FuzzEndpoint {
  iree_net_message_endpoint_callbacks_t callbacks = {};

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<FuzzEndpoint*>(self)->callbacks = callbacks;
  }

  static iree_status_t Activate(void* self) {
    (void)self;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    (void)self;
    if (callback) {
      callback(user_data);
    }
    return iree_ok_status();
  }

  static iree_status_t Send(
      void* self, const iree_net_message_endpoint_send_params_t* params) {
    (void)self;
    uint8_t prefix_storage[IREE_NET_BULK_MESSAGE_HEADER_SIZE];
    IREE_RETURN_IF_ERROR(params->generated_prefix.write(
        params->generated_prefix.user_data,
        iree_make_byte_span(prefix_storage, sizeof(prefix_storage))));
    iree_host_size_t total_length = params->generated_prefix.length;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      total_length += params->data.values[i].length;
    }
    params->completion_callback.fn(params->completion_callback.user_data,
                                   iree_ok_status(), total_length);
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    (void)self;
    return {};
  }

  iree_net_message_endpoint_t endpoint() {
    static const iree_net_message_endpoint_vtable_t vtable = {
        .set_callbacks = SetCallbacks,
        .activate = Activate,
        .deactivate = Deactivate,
        .send = Send,
        .query_send_budget = QuerySendBudget,
    };
    return {
        .self = this,
        .vtable = &vtable,
    };
  }
};

static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                             uint64_t total_length) {
  (void)user_data;
  (void)transfer_id;
  (void)total_length;
  return iree_ok_status();
}

static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                            uint64_t offset, iree_const_byte_span_t payload,
                            iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)transfer_id;
  (void)offset;
  (void)payload;
  (void)lease;
  return iree_ok_status();
}

static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
  (void)user_data;
  (void)transfer_id;
  return iree_ok_status();
}

static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                             iree_const_byte_span_t detail,
                             iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)transfer_id;
  (void)detail;
  (void)lease;
  return iree_ok_status();
}

static iree_status_t OnCredit(void* user_data, uint64_t credit_delta,
                              uint64_t available_credit_count) {
  (void)user_data;
  (void)credit_delta;
  (void)available_credit_count;
  return iree_ok_status();
}

static void OnError(void* user_data, iree_status_t status) {
  (void)user_data;
  iree_status_free(status);
}

static void OnSendComplete(void* user_data, iree_status_t status,
                           iree_host_size_t bytes_transferred) {
  (void)user_data;
  (void)bytes_transferred;
  iree_status_free(status);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FuzzEndpoint endpoint;
  iree_net_bulk_channel_t* channel = nullptr;
  iree_status_t status =
      iree_net_bulk_channel_allocate(endpoint.endpoint(),
                                     {
                                         .on_start = OnStart,
                                         .on_data = OnData,
                                         .on_complete = OnComplete,
                                         .on_abort = OnAbort,
                                         .on_credit = OnCredit,
                                         .on_error = OnError,
                                         .user_data = nullptr,
                                     },
                                     iree_allocator_system(), &channel);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return 0;
  }

  iree_net_bulk_channel_attach(channel);
  status = iree_net_bulk_channel_send_credit(channel, UINT32_MAX,
                                             {
                                                 .fn = OnSendComplete,
                                                 .user_data = nullptr,
                                             });
  if (iree_status_is_ok(status)) {
    status = endpoint.callbacks.on_message(
        endpoint.callbacks.user_data, iree_make_const_byte_span(data, size),
        /*lease=*/nullptr);
  }
  iree_status_free(status);
  iree_net_bulk_channel_free(channel);
  return 0;
}
