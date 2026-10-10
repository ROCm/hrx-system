// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/bootstrap.h"

static void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
  iree_status_free(status);
}

static iree_net_bootstrap_peer_info_t ReconstructPeer(
    const iree_net_bootstrap_peer_info_view_t& view,
    std::vector<iree_async_frontier_entry_t>* axes) {
  axes->resize(view.axes.count);
  for (uint32_t i = 0; i < view.axes.count; ++i) {
    (*axes)[i] = iree_net_bootstrap_axis_list_get(&view.axes, i);
  }
  return {
      .capabilities = view.capabilities,
      .application_endpoint_count = view.application_endpoint_count,
      .axes = axes->data(),
      .axis_count = static_cast<uint32_t>(axes->size()),
      .application_data = view.application_data,
      .machine_index = view.machine_index,
      .session_epoch = view.session_epoch,
  };
}

static void CheckCanonicalWire(const uint8_t* data, size_t size) {
  iree_net_bootstrap_message_view_t parsed;
  iree_status_t parse_status = iree_net_bootstrap_message_parse(
      iree_make_const_byte_span(data, size), &parsed);
  if (!iree_status_is_ok(parse_status)) {
    iree_status_free(parse_status);
    return;
  }
  iree_status_free(parse_status);

  std::vector<iree_async_frontier_entry_t> axes;
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = parsed.type;
  switch (parsed.type) {
    case IREE_NET_BOOTSTRAP_TYPE_HELLO:
      message.value.hello = ReconstructPeer(parsed.value.hello, &axes);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK:
      message.value.hello_ack = ReconstructPeer(parsed.value.hello_ack, &axes);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_REJECT:
      message.value.reject.status_code = parsed.value.reject.status_code;
      message.value.reject.reason = parsed.value.reject.reason;
      break;
    default:
      std::abort();
  }

  iree_host_size_t wire_size = 0;
  CheckStatus(iree_net_bootstrap_message_calculate_size(&message, &wire_size));
  if (wire_size != size) {
    std::abort();
  }
  std::vector<uint8_t> wire(wire_size);
  CheckStatus(iree_net_bootstrap_message_serialize(
      &message, iree_make_byte_span(wire.data(), wire.size())));
  if (std::memcmp(wire.data(), data, size) != 0) {
    std::abort();
  }

  iree_net_bootstrap_message_view_t reparsed;
  CheckStatus(iree_net_bootstrap_message_parse(
      iree_make_const_byte_span(wire.data(), wire.size()), &reparsed));
  if (reparsed.type != parsed.type) {
    std::abort();
  }
}

static uint8_t ReadByte(const uint8_t* data, size_t size, size_t index) {
  return index < size ? data[index] : 0;
}

static void CheckSynthesizedWire(const uint8_t* data, size_t size) {
  const uint8_t selector = ReadByte(data, size, 0);
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = static_cast<iree_net_bootstrap_type_t>(selector % 3 + 1);

  std::vector<iree_async_frontier_entry_t> axes;
  if (message.type == IREE_NET_BOOTSTRAP_TYPE_REJECT) {
    message.value.reject.status_code = static_cast<iree_status_code_t>(
        ReadByte(data, size, 1) % IREE_STATUS_CODE_MASK + 1);
    message.value.reject.reason = iree_make_cstring_view(
        iree_status_code_string(message.value.reject.status_code));
  } else {
    const uint8_t machine_index = ReadByte(data, size, 1);
    const uint8_t session_epoch = ReadByte(data, size, 2);
    const uint32_t axis_count = ReadByte(data, size, 3) % 9;
    axes.resize(axis_count);
    for (uint32_t i = 0; i < axis_count; ++i) {
      const iree_async_causal_domain_t domain =
          static_cast<iree_async_causal_domain_t>(ReadByte(data, size, 4 + i) %
                                                  4);
      axes[i].axis = iree_async_axis_make(
          session_epoch, machine_index, domain,
          ((uint64_t)i << 8) | ReadByte(data, size, 13 + i));
      axes[i].epoch = ((uint64_t)ReadByte(data, size, 22 + i) << 32) | i;
    }
    const iree_net_bootstrap_peer_info_t peer = {
        .capabilities =
            (iree_net_bootstrap_capabilities_t)(ReadByte(data, size, 31) &
                                                IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED),
        .application_endpoint_count = ReadByte(data, size, 32),
        .axes = axes.data(),
        .axis_count = axis_count,
        .application_data = iree_make_const_byte_span(data, size),
        .machine_index = machine_index,
        .session_epoch = session_epoch,
    };
    if (message.type == IREE_NET_BOOTSTRAP_TYPE_HELLO) {
      message.value.hello = peer;
    } else {
      message.value.hello_ack = peer;
    }
  }

  iree_host_size_t wire_size = 0;
  CheckStatus(iree_net_bootstrap_message_calculate_size(&message, &wire_size));
  std::vector<uint8_t> wire(wire_size);
  CheckStatus(iree_net_bootstrap_message_serialize(
      &message, iree_make_byte_span(wire.data(), wire.size())));
  CheckCanonicalWire(wire.data(), wire.size());
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  CheckCanonicalWire(data, size);
  CheckSynthesizedWire(data, size);
  return 0;
}
