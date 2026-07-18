// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-serve-device/transport.h"

#include "iree/hal/remote/protocol/common.h"
#include "iree/net/carrier/shm/factory.h"
#include "iree/net/carrier/tcp/factory.h"

#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
#include "iree/net/carrier/rdma/factory.h"
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT

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
#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
  if (iree_string_view_consume_prefix(&remainder, IREE_SV("rdma://"))) {
    out_bind->transport_name = IREE_SV("rdma");
    out_bind->bind_address = remainder;
    return iree_ok_status();
  }
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT
#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
  const char* expected_prefixes = "tcp://, shm://, rdma://";
#else
  const char* expected_prefixes = "tcp://, shm://";
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "bind URI must have a transport prefix (%s), got: '%.*s'",
      expected_prefixes, (int)bind_uri.size, bind_uri.data);
}

iree_serve_device_bind_visibility_t iree_serve_device_classify_bind_visibility(
    iree_string_view_t transport_name, iree_string_view_t bound_address) {
  if (iree_string_view_equal(transport_name, IREE_SV("shm"))) {
    return IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY;
  }

  const bool is_network_transport =
      iree_string_view_equal(transport_name, IREE_SV("tcp")) ||
      iree_string_view_equal(transport_name, IREE_SV("rdma"));
  if (is_network_transport &&
      (iree_string_view_starts_with(bound_address, IREE_SV("0.0.0.0:")) ||
       iree_string_view_starts_with(bound_address, IREE_SV("[::]:")))) {
    return IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK_WILDCARD;
  }

  if (iree_string_view_equal(transport_name, IREE_SV("tcp")) &&
      (iree_string_view_starts_with(bound_address, IREE_SV("127.")) ||
       iree_string_view_starts_with(bound_address, IREE_SV("[::1]:")))) {
    return IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY;
  }

  return IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK;
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
#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
  if (iree_string_view_equal(transport_name, IREE_SV("rdma"))) {
    iree_net_rdma_factory_options_t rdma_options =
        iree_net_rdma_factory_options_default();
    // HAL remote requires control, queue, and bulk endpoints per connection.
    rdma_options.max_endpoint_count = IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT;
    iree_status_t status =
        iree_net_rdma_factory_create(rdma_options, host_allocator, out_factory);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported transport: %.*s",
                          (int)transport_name.size, transport_name.data);
}
