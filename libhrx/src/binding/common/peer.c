// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/peer.h"

#include <string.h>

#include "common/internal.h"

void iree_hal_streaming_peer_properties_initialize(
    bool access_supported, bool architectures_match,
    iree_hal_topology_edge_t topology_edge,
    iree_hal_streaming_peer_properties_t* out_properties) {
  IREE_ASSERT_ARGUMENT(out_properties);
  memset(out_properties, 0, sizeof(*out_properties));

  const iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(topology_edge.lo);
  out_properties->access_supported = access_supported;
  out_properties->native_atomic_supported = iree_any_bit_set(
      capabilities, IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32 |
                        IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64);
  out_properties->array_access_supported =
      access_supported && architectures_match;
  out_properties->link_type =
      iree_hal_topology_edge_link_type(topology_edge.hi);
  out_properties->hop_count =
      iree_hal_topology_edge_path_hop_count(topology_edge.hi);
}

iree_status_t iree_hal_streaming_device_query_peer_properties(
    iree_host_size_t source_device, iree_host_size_t destination_device,
    iree_hal_streaming_peer_properties_t* out_properties) {
  IREE_ASSERT_ARGUMENT(out_properties);
  memset(out_properties, 0, sizeof(*out_properties));

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }
  if (source_device >= device_registry->device_count ||
      destination_device >= device_registry->device_count ||
      source_device == destination_device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "peer device ordinals must identify two distinct "
                            "devices in range [0, %" PRIhsz ")",
                            device_registry->device_count);
  }

  const iree_hal_streaming_device_t* source =
      &device_registry->devices[source_device];
  const iree_hal_streaming_device_t* destination =
      &device_registry->devices[destination_device];
  const iree_hal_device_topology_info_t* source_topology_info =
      iree_hal_device_topology_info(source->hal_device);
  const iree_hal_device_topology_info_t* destination_topology_info =
      iree_hal_device_topology_info(destination->hal_device);
  if (!source_topology_info->topology ||
      source_topology_info->topology != destination_topology_info->topology) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "peer devices must belong to the same initialized HAL topology");
  }

  bool access_supported = false;
  IREE_RETURN_IF_ERROR(HRX_CALL(hrx_device_can_access_peer(
      source->hrx_device, destination->hrx_device, &access_supported)));
  const iree_hal_topology_edge_t topology_edge =
      iree_hal_device_topology_query_edge(source_topology_info,
                                          destination_topology_info);
  iree_hal_streaming_peer_properties_initialize(
      access_supported,
      strcmp(source->gcn_arch_name, destination->gcn_arch_name) == 0,
      topology_edge, out_properties);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_can_access_peer(
    iree_host_size_t device_ordinal, iree_host_size_t peer_device_ordinal,
    bool* out_can_access) {
  IREE_ASSERT_ARGUMENT(out_can_access);
  *out_can_access = false;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }
  if (device_ordinal >= device_registry->device_count ||
      peer_device_ordinal >= device_registry->device_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinals out of range [0, %" PRIhsz ")",
                            device_registry->device_count);
  }
  if (device_ordinal == peer_device_ordinal) return iree_ok_status();

  iree_hal_streaming_peer_properties_t properties;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_device_query_peer_properties(
      device_ordinal, peer_device_ordinal, &properties));
  *out_can_access = properties.access_supported;
  return iree_ok_status();
}
