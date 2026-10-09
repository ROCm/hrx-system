// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/net/channel/control/control_channel.h"

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
    (void)params;
    return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
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

static iree_status_t OnData(void* user_data,
                            iree_net_control_data_flags_t flags,
                            iree_const_byte_span_t payload,
                            iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)flags;
  (void)payload;
  (void)lease;
  return iree_ok_status();
}

static void OnGoaway(void* user_data, uint32_t reason_code) {
  (void)user_data;
  (void)reason_code;
}

static void OnError(void* user_data, iree_status_t status) {
  (void)user_data;
  iree_status_free(status);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FuzzEndpoint endpoint;
  iree_net_control_channel_t* channel = nullptr;
  iree_status_t status =
      iree_net_control_channel_allocate(endpoint.endpoint(),
                                        {
                                            .on_data = OnData,
                                            .on_goaway = OnGoaway,
                                            .on_error = OnError,
                                            .user_data = nullptr,
                                        },
                                        iree_allocator_system(), &channel);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return 0;
  }

  iree_net_control_channel_attach(channel);
  status = endpoint.callbacks.on_message(endpoint.callbacks.user_data,
                                         iree_make_const_byte_span(data, size),
                                         /*lease=*/nullptr);
  iree_status_free(status);
  iree_net_control_channel_free(channel);
  return 0;
}
