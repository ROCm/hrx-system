// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/tcp/factory.h"
#include "iree/net/cts/transport_backend.h"

namespace iree::net::cts {
namespace {

iree_status_t CreateFactory(iree_allocator_t host_allocator,
                            iree_net_transport_factory_t** out_factory) {
  return iree_net_tcp_factory_create(
      /*options=*/nullptr, host_allocator, out_factory);
}

iree_status_t MakeBindAddress(std::string* out_address) {
  *out_address = "127.0.0.1:0";
  return iree_ok_status();
}

}  // namespace

const TransportBackend& GetTransportBackend() {
  static const TransportBackend backend = {
      .name = "tcp",
      .required_capabilities = IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
                               IREE_NET_TRANSPORT_CAPABILITY_ORDERED,
      .create_factory = CreateFactory,
      .make_bind_address = MakeBindAddress,
  };
  return backend;
}

}  // namespace iree::net::cts
