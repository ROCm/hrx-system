// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/p2p_topology.h"

#include <string.h>

void iree_hal_streaming_p2p_link_initialize(
    iree_host_size_t source_device, iree_host_size_t destination_device,
    iree_string_view_t source_architecture,
    iree_string_view_t destination_architecture,
    iree_hal_topology_edge_t topology_edge,
    iree_hal_streaming_p2p_link_t* out_link) {
  IREE_ASSERT_ARGUMENT(out_link);
  memset(out_link, 0, sizeof(*out_link));

  out_link->source_device = source_device;
  out_link->destination_device = destination_device;
  out_link->link_type = iree_hal_topology_edge_link_type(topology_edge.hi);
  out_link->hop_count = iree_hal_topology_edge_path_hop_count(topology_edge.hi);

  if (source_device == destination_device) {
    out_link->access_supported = true;
    out_link->native_atomic_supported = true;
    out_link->array_access_supported = true;
    return;
  }

  const iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(topology_edge.lo);
  out_link->access_supported =
      iree_any_bit_set(capabilities, IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY);
  out_link->native_atomic_supported = iree_any_bit_set(
      capabilities, IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
                        IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM);
  out_link->array_access_supported =
      out_link->access_supported &&
      iree_string_view_equal(source_architecture, destination_architecture);
}
