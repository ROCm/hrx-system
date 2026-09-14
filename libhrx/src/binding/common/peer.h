// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_PEER_H_
#define LIBHRX_SRC_BINDING_COMMON_PEER_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Peer properties projected from one directed HAL topology edge.
typedef struct iree_hal_streaming_peer_properties_t {
  // True when the source device can access destination device allocations.
  bool access_supported;
  // True when the physical path supports at least one native atomic width.
  bool native_atomic_supported;
  // True when peer access extends to opaque arrays.
  bool array_access_supported;
  // First-hop physical interconnect technology.
  iree_hal_topology_link_type_t link_type;
  // Number of physical links in the path.
  uint32_t hop_count;
} iree_hal_streaming_peer_properties_t;

// Projects binding-level peer properties from a refined HAL topology edge.
// |access_supported| is supplied by the lower HRX device contract so direct
// access and grant requirements have one interpretation throughout HRX.
void iree_hal_streaming_peer_properties_initialize(
    bool access_supported, bool architectures_match,
    iree_hal_topology_edge_t topology_edge,
    iree_hal_streaming_peer_properties_t* out_properties);

// Queries immutable peer properties for a directed source-to-destination pair.
// Device ordinals must identify different devices in the active registry.
iree_status_t iree_hal_streaming_device_query_peer_properties(
    iree_host_size_t source_device, iree_host_size_t destination_device,
    iree_hal_streaming_peer_properties_t* out_properties);

// Queries whether |device_ordinal| is capable of accessing allocations owned by
// |peer_device_ordinal|. Equal ordinals are valid and report false because a
// device accessing its own memory is not peer access.
iree_status_t iree_hal_streaming_device_can_access_peer(
    iree_host_size_t device_ordinal, iree_host_size_t peer_device_ordinal,
    bool* out_can_access);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_COMMON_PEER_H_
