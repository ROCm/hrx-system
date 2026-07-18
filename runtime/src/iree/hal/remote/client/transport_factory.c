// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/transport_factory.h"

#include "iree/hal/remote/protocol/common.h"
#include "iree/net/transport_factory.h"

#if defined(IREE_HAVE_NET_TCP_TRANSPORT)
#include "iree/net/carrier/tcp/factory.h"
#endif  // IREE_HAVE_NET_TCP_TRANSPORT

#if defined(IREE_HAVE_NET_SHM_TRANSPORT)
#include "iree/net/carrier/shm/factory.h"
#endif  // IREE_HAVE_NET_SHM_TRANSPORT

#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
#include "iree/net/carrier/rdma/factory.h"
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT

static const iree_string_view_t IREE_HAL_REMOTE_DRIVER_PREFIX =
    iree_string_view_literal("remote-");

IREE_API_EXPORT iree_status_t iree_hal_remote_client_transport_factory_create(
    iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  IREE_ASSERT_ARGUMENT(out_factory);
  *out_factory = NULL;

  if (!iree_string_view_starts_with(driver_name,
                                    IREE_HAL_REMOTE_DRIVER_PREFIX)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "driver '%.*s' is not a remote HAL client",
                            (int)driver_name.size, driver_name.data);
  }
  const iree_string_view_t transport_name = iree_string_view_substr(
      driver_name, IREE_HAL_REMOTE_DRIVER_PREFIX.size,
      driver_name.size - IREE_HAL_REMOTE_DRIVER_PREFIX.size);
  if (iree_string_view_is_empty(transport_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "driver name '%.*s' has no transport suffix after 'remote-'",
        (int)driver_name.size, driver_name.data);
  }

#if defined(IREE_HAVE_NET_TCP_TRANSPORT)
  if (iree_string_view_equal(transport_name, IREE_SV("tcp"))) {
    iree_net_tcp_carrier_options_t options =
        iree_net_tcp_carrier_options_default();
    options.max_endpoint_count = IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT;
    return iree_net_tcp_factory_create(options, host_allocator, out_factory);
  }
#endif  // IREE_HAVE_NET_TCP_TRANSPORT

#if defined(IREE_HAVE_NET_SHM_TRANSPORT)
  if (iree_string_view_equal(transport_name, IREE_SV("shm"))) {
    return iree_net_shm_factory_create(iree_net_shm_carrier_options_default(),
                                       host_allocator, out_factory);
  }
#endif  // IREE_HAVE_NET_SHM_TRANSPORT

#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
  if (iree_string_view_equal(transport_name, IREE_SV("rdma"))) {
    iree_net_rdma_factory_options_t options =
        iree_net_rdma_factory_options_default();
    options.max_endpoint_count = IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT;
    return iree_net_rdma_factory_create(options, host_allocator, out_factory);
  }
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT

  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "transport '%.*s' is not compiled into this remote HAL client; enable "
      "it through //runtime/config/net:transports for Bazel or "
      "IREE_NET_TRANSPORT_* for CMake",
      (int)transport_name.size, transport_name.data);
}
