// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_P2P_TOPOLOGY_H_
#define LIBHRX_SRC_BINDING_COMMON_P2P_TOPOLOGY_H_

#include "iree/base/api.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable peer capability information for one directed device pair.
typedef struct iree_hal_streaming_p2p_link_t {
  // Source device ordinal.
  iree_host_size_t source_device;
  // Destination device ordinal.
  iree_host_size_t destination_device;
  // True when the source device can directly access destination allocations.
  bool access_supported;
  // True when the physical path supports peer atomic operations.
  bool native_atomic_supported;
  // True when opaque arrays can be accessed across the device pair.
  bool array_access_supported;
  // Physical interconnect technology for the first path hop.
  iree_hal_topology_link_type_t link_type;
  // Physical path length reported by the backend.
  uint32_t hop_count;
} iree_hal_streaming_p2p_link_t;

// Initializes immutable peer capabilities from a refined HAL topology edge.
void iree_hal_streaming_p2p_link_initialize(
    iree_host_size_t source_device, iree_host_size_t destination_device,
    iree_string_view_t source_architecture,
    iree_string_view_t destination_architecture,
    iree_hal_topology_edge_t topology_edge,
    iree_hal_streaming_p2p_link_t* out_link);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_COMMON_P2P_TOPOLOGY_H_
