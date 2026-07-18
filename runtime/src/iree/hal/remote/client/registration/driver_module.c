// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/transport_factory.h"
#include "iree/net/transport_factory.h"

// Remote HAL drivers organized by transport name.
// Each transport uses the same URI scheme (host:port or path) but different
// underlying transport mechanisms. This allows:
//   --device=remote-tcp://server:5000   (TCP sockets)
//   --device=remote-shm:///dev/shm/iree (Shared memory for testing)
//   --device=remote-rdma://addr         (RDMA CM)
//
// The transport name after "remote-" determines which factory is created.
// Available transports depend on what has been compiled in via the
// IREE_HAVE_NET_*_TRANSPORT defines, controlled by //runtime/config/net
// transport settings.

static const iree_hal_driver_info_t iree_hal_remote_driver_infos[] = {
#if defined(IREE_HAVE_NET_TCP_TRANSPORT)
    {
        .driver_name = IREE_SVL("remote-tcp"),
        .full_name = IREE_SVL("Remote HAL Client (TCP)"),
    },
#endif  // IREE_HAVE_NET_TCP_TRANSPORT
#if defined(IREE_HAVE_NET_SHM_TRANSPORT)
    {
        .driver_name = IREE_SVL("remote-shm"),
        .full_name = IREE_SVL("Remote HAL Client (Shared Memory)"),
    },
#endif  // IREE_HAVE_NET_SHM_TRANSPORT
#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
    {
        .driver_name = IREE_SVL("remote-rdma"),
        .full_name = IREE_SVL("Remote HAL Client (RDMA)"),
    },
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT
};

static iree_status_t iree_hal_remote_client_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  *out_driver_info_count = IREE_ARRAYSIZE(iree_hal_remote_driver_infos);
  *out_driver_infos = iree_hal_remote_driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  // Create the transport factory for this transport type.
  iree_net_transport_factory_t* factory = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_transport_factory_create(
      driver_name, host_allocator, &factory));

  // Set up driver options with the factory. The recv_pool is resolved from
  // create_params->proactor_pool during create_device_by_path.
  iree_hal_remote_client_driver_options_t options;
  iree_hal_remote_client_driver_options_initialize(&options);
  options.transport_factory = factory;
  options.default_device_options.transport_factory = factory;

  iree_status_t status = iree_hal_remote_client_driver_create(
      driver_name, &options, host_allocator, out_driver);
  iree_net_transport_factory_release(factory);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_remote_client_driver_factory_enumerate,
      .try_create = iree_hal_remote_client_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
