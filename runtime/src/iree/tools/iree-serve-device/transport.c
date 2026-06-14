// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-serve-device/transport.h"

#include "iree/hal/remote/protocol/common.h"
#include "iree/net/carrier/shm/factory.h"
#include "iree/net/carrier/tcp/factory.h"

iree_status_t iree_serve_device_parse_bind_uri(
    iree_string_view_t bind_uri, iree_serve_device_bind_t* out_bind) {
  iree_string_view_t remainder = bind_uri;
  if (iree_string_view_consume_prefix(&remainder, IREE_SV("tcp://"))) {
    out_bind->transport_name = IREE_SV("tcp");
    out_bind->bind_address = remainder;
    return iree_ok_status();
  }
  if (iree_string_view_consume_prefix(&remainder, IREE_SV("shm://"))) {
    out_bind->transport_name = IREE_SV("shm");
    out_bind->bind_address = remainder;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "bind URI must have a transport prefix (tcp://, shm://), got: '%.*s'",
      (int)bind_uri.size, bind_uri.data);
}

iree_status_t iree_serve_device_create_transport_factory(
    iree_string_view_t transport_name, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (iree_string_view_equal(transport_name, IREE_SV("tcp"))) {
    iree_net_tcp_carrier_options_t tcp_options =
        iree_net_tcp_carrier_options_default();
    // HAL remote requires control, queue, and bulk endpoints per connection.
    tcp_options.max_endpoint_count = IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT;
    iree_status_t status =
        iree_net_tcp_factory_create(tcp_options, host_allocator, out_factory);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (iree_string_view_equal(transport_name, IREE_SV("shm"))) {
    iree_status_t status = iree_net_shm_factory_create(
        iree_net_shm_carrier_options_default(), host_allocator, out_factory);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported transport: %.*s",
                          (int)transport_name.size, transport_name.data);
}
