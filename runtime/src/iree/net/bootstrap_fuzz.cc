// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include "iree/net/bootstrap.h"

#define FUZZ_ASSERT(condition) \
  do {                         \
    if (!(condition)) {        \
      __builtin_trap();        \
    }                          \
  } while (0)

static bool SpanIsWithin(iree_const_byte_span_t outer,
                         iree_const_byte_span_t inner) {
  uintptr_t outer_begin = reinterpret_cast<uintptr_t>(outer.data);
  uintptr_t outer_end = outer_begin + outer.data_length;
  uintptr_t inner_begin = reinterpret_cast<uintptr_t>(inner.data);
  uintptr_t inner_end = inner_begin + inner.data_length;
  return outer_end >= outer_begin && inner_end >= inner_begin &&
         inner_begin >= outer_begin && inner_end <= outer_end;
}

static void CheckTopologyView(iree_const_byte_span_t payload,
                              iree_net_bootstrap_type_t message_type,
                              uint16_t encoded_axis_count,
                              uint64_t encoded_application_data_length,
                              const iree_net_bootstrap_axis_list_t& axes,
                              iree_const_byte_span_t application_data) {
  FUZZ_ASSERT(axes.count == encoded_axis_count);
  FUZZ_ASSERT(axes.encoded_entries.data_length ==
              axes.count * sizeof(iree_net_bootstrap_axis_entry_t));
  FUZZ_ASSERT(SpanIsWithin(payload, axes.encoded_entries));
  FUZZ_ASSERT(application_data.data_length == encoded_application_data_length);
  FUZZ_ASSERT(SpanIsWithin(payload, application_data));

  iree_net_bootstrap_topology_layout_t layout;
  iree_status_t status = iree_net_bootstrap_topology_layout_calculate(
      message_type, axes.count, application_data.data_length, &layout);
  FUZZ_ASSERT(iree_status_is_ok(status));
  FUZZ_ASSERT(layout.payload_length == payload.data_length);
  for (uint32_t i = 0; i < axes.count; ++i) {
    (void)iree_net_bootstrap_axis_list_get(&axes, i);
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_const_byte_span_t payload = iree_make_const_byte_span(data, size);
  iree_net_bootstrap_message_view_t message;
  iree_status_t status = iree_net_bootstrap_message_parse(payload, &message);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return 0;
  }

  switch (message.type) {
    case IREE_NET_BOOTSTRAP_TYPE_HELLO:
      FUZZ_ASSERT(message.value.hello.fixed.header.type == message.type);
      CheckTopologyView(
          payload, message.type, message.value.hello.fixed.axis_count,
          message.value.hello.fixed.application_data_length,
          message.value.hello.axes, message.value.hello.application_data);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK:
      FUZZ_ASSERT(message.value.hello_ack.fixed.header.type == message.type);
      CheckTopologyView(payload, message.type,
                        message.value.hello_ack.fixed.axis_count,
                        message.value.hello_ack.fixed.application_data_length,
                        message.value.hello_ack.axes,
                        message.value.hello_ack.application_data);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_REJECT:
      FUZZ_ASSERT(message.value.reject.fixed.header.type == message.type);
      FUZZ_ASSERT(SpanIsWithin(payload, iree_make_const_byte_span(
                                            message.value.reject.reason.data,
                                            message.value.reject.reason.size)));
      break;
    default:
      FUZZ_ASSERT(false);
      break;
  }

  return 0;
}
