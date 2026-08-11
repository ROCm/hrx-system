// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/p2p_topology.h"

#include "iree/hal/topology_builder.h"
#include "iree/testing/gtest.h"

namespace {

TEST(P2PTopologyTest, ProjectsRefinedPeerCapabilities) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();
  edge.lo = iree_hal_topology_edge_set_capability_flags(
      edge.lo, IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                   IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE);
  edge.hi = iree_hal_topology_edge_set_link_type(
      edge.hi, IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI);
  edge.hi = iree_hal_topology_edge_set_path_hop_count(edge.hi, 2);

  iree_hal_streaming_p2p_link_t link;
  iree_hal_streaming_p2p_link_initialize(0, 1, iree_make_cstring_view("gfx942"),
                                         iree_make_cstring_view("gfx942"), edge,
                                         &link);

  EXPECT_EQ(link.source_device, 0u);
  EXPECT_EQ(link.destination_device, 1u);
  EXPECT_TRUE(link.access_supported);
  EXPECT_TRUE(link.native_atomic_supported);
  EXPECT_TRUE(link.array_access_supported);
  EXPECT_EQ(link.link_type, IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI);
  EXPECT_EQ(link.hop_count, 2u);
}

TEST(P2PTopologyTest, RequiresMatchingArchitectureForArrayAccess) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();
  edge.lo = iree_hal_topology_edge_set_capability_flags(
      edge.lo, IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY);

  iree_hal_streaming_p2p_link_t link;
  iree_hal_streaming_p2p_link_initialize(0, 1, iree_make_cstring_view("gfx942"),
                                         iree_make_cstring_view("gfx950"), edge,
                                         &link);

  EXPECT_TRUE(link.access_supported);
  EXPECT_FALSE(link.native_atomic_supported);
  EXPECT_FALSE(link.array_access_supported);
}

}  // namespace
