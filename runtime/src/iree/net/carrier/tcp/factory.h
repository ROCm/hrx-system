// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP transport factory backed by platform asynchronous sockets.

#ifndef IREE_NET_CARRIER_TCP_FACTORY_H_
#define IREE_NET_CARRIER_TCP_FACTORY_H_

#include "iree/base/api.h"
#include "iree/net/carrier/tcp/connection.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default size of each connection-owned receive buffer.
#define IREE_NET_TCP_DEFAULT_RECEIVE_BUFFER_SIZE (64 * 1024)

// Default number of receive buffers owned by each connection.
#define IREE_NET_TCP_DEFAULT_RECEIVE_BUFFER_COUNT 16u

// Maximum portable provided receive buffer ring capacity.
#define IREE_NET_TCP_MAX_RECEIVE_BUFFER_COUNT (1u << 15)

// Default kernel listen backlog hint.
#define IREE_NET_TCP_DEFAULT_LISTEN_BACKLOG 128u

// Options controlling TCP connection establishment and bounded resources.
typedef struct iree_net_tcp_factory_options_t {
  // Framing, endpoint, and send admission limits for every connection.
  iree_net_tcp_connection_options_t connection_options;

  // Byte length of each connection-owned registered receive buffer in POOLED
  // mode. PLACEMENT mode allocates no receive pool.
  iree_host_size_t receive_buffer_size;

  // Power-of-two receive buffer count allocated for each connection. Must not
  // exceed IREE_NET_TCP_MAX_RECEIVE_BUFFER_COUNT.
  iree_host_size_t receive_buffer_count;

  // Kernel listen backlog hint. Zero requests the platform default.
  iree_host_size_t listen_backlog;
} iree_net_tcp_factory_options_t;

// Returns default TCP transport factory options.
static inline iree_net_tcp_factory_options_t
iree_net_tcp_factory_options_default(void) {
  iree_net_tcp_factory_options_t options;
  options.connection_options = iree_net_tcp_connection_options_default();
  options.receive_buffer_size = IREE_NET_TCP_DEFAULT_RECEIVE_BUFFER_SIZE;
  options.receive_buffer_count = IREE_NET_TCP_DEFAULT_RECEIVE_BUFFER_COUNT;
  options.listen_backlog = IREE_NET_TCP_DEFAULT_LISTEN_BACKLOG;
  return options;
}

// Creates a TCP transport factory.
//
// Addresses use numeric IPv4 `host:port` or IPv6 `[host]:port` syntax.
// Hostname resolution is deliberately outside the transport factory so that
// connect submission never performs blocking resolver work.
//
// POOLED mode creates a dedicated registered receive pool for each connection;
// PLACEMENT mode receives into the shared framing cursor's current target. The
// generic receive pool arguments passed to connect and create_listener are not
// used by this transport.
//
// Stopping a listener closes its native listening socket before delivering the
// stopped callback. Its bound address remains queryable until the listener is
// freed. Connections already published by the listener retain their independent
// lifetimes.
//
// Capabilities: RELIABLE | ORDERED.
IREE_API_EXPORT iree_status_t
iree_net_tcp_factory_create(const iree_net_tcp_factory_options_t* options,
                            iree_allocator_t host_allocator,
                            iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_TCP_FACTORY_H_
