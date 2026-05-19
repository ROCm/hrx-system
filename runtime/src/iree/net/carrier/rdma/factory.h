// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA transport factory using rdma_cm.
//
// The factory owns the cold connection-management path: it creates/listens on
// rdma_cm IDs, waits for address and route resolution through the IREE
// proactor, creates the RDMA carrier once the CM ID has selected a verbs
// device, exchanges the carrier private-data payload, and then returns a
// standard iree_net_connection_t.
//
// The first implementation exposes one RC QP as one message endpoint. Native
// multi-endpoint QP allocation can layer on the same handshake machinery
// without changing the carrier hot path.

#ifndef IREE_NET_CARRIER_RDMA_FACTORY_H_
#define IREE_NET_CARRIER_RDMA_FACTORY_H_

#include "iree/base/api.h"
#include "iree/net/carrier/rdma/carrier.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default timeout for rdma_cm address and route resolution.
#define IREE_NET_RDMA_FACTORY_DEFAULT_RESOLVE_TIMEOUT_MS 2000

// Default listener backlog.
#define IREE_NET_RDMA_FACTORY_DEFAULT_LISTEN_BACKLOG 128

// Options for RDMA transport-factory creation.
typedef struct iree_net_rdma_factory_options_t {
  // Device/library selection used to create the factory's parent context.
  iree_net_rdma_context_options_t context_options;

  // Carrier queue and inline-data options applied to each accepted QP.
  iree_net_rdma_carrier_options_t carrier_options;

  // Timeout passed to rdma_resolve_addr and rdma_resolve_route.
  int resolve_timeout_ms;

  // Backlog passed to rdma_listen.
  int listen_backlog;

  // Maximum endpoint slots per connection. Only 1 is currently supported.
  uint32_t max_endpoint_count;
} iree_net_rdma_factory_options_t;

// Returns default RDMA factory options.
static inline iree_net_rdma_factory_options_t
iree_net_rdma_factory_options_default(void) {
  iree_net_rdma_factory_options_t options;
  memset(&options, 0, sizeof(options));
  options.context_options = iree_net_rdma_context_options_default();
  options.carrier_options = iree_net_rdma_carrier_options_default();
  options.resolve_timeout_ms = IREE_NET_RDMA_FACTORY_DEFAULT_RESOLVE_TIMEOUT_MS;
  options.listen_backlog = IREE_NET_RDMA_FACTORY_DEFAULT_LISTEN_BACKLOG;
  options.max_endpoint_count = 1;
  return options;
}

// Allocates an RDMA transport factory.
//
// The factory creates a parent RDMA context at construction time to load
// rdma-core symbols and establish the requested device selection. Individual
// connections create child contexts over the exact verbs context selected by
// rdma_cm, then validate it matches the parent selection before allocating QPs
// and memory registrations.
IREE_API_EXPORT iree_status_t iree_net_rdma_factory_create(
    iree_net_rdma_factory_options_t options, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_FACTORY_H_
