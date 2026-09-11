// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/peer.h"

#include "iree/hal/topology_builder.h"
#include "iree/testing/gtest.h"

namespace {

TEST(PeerTest, ProjectsTopologyProperties) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_empty();
  edge.lo = iree_hal_topology_edge_set_capability_flags(
      edge.lo, IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64);
  edge.hi = iree_hal_topology_edge_set_link_type(
      edge.hi, IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI);
  edge.hi = iree_hal_topology_edge_set_path_hop_count(edge.hi, 2);

  iree_hal_streaming_peer_properties_t properties;
  iree_hal_streaming_peer_properties_initialize(
      /*access_supported=*/true, /*architectures_match=*/true, edge,
      &properties);

  EXPECT_TRUE(properties.access_supported);
  EXPECT_TRUE(properties.native_atomic_supported);
  EXPECT_TRUE(properties.array_access_supported);
  EXPECT_EQ(properties.link_type, IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI);
  EXPECT_EQ(properties.hop_count, 2u);
}

TEST(PeerTest, RejectsArrayAccessAcrossArchitectures) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_empty();
  edge.lo = iree_hal_topology_edge_set_capability_flags(
      edge.lo, IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32);

  iree_hal_streaming_peer_properties_t properties;
  iree_hal_streaming_peer_properties_initialize(
      /*access_supported=*/true, /*architectures_match=*/false, edge,
      &properties);

  EXPECT_TRUE(properties.access_supported);
  EXPECT_TRUE(properties.native_atomic_supported);
  EXPECT_FALSE(properties.array_access_supported);
}

TEST(PeerTest, RequiresPeerAccessForArrays) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_empty();

  iree_hal_streaming_peer_properties_t properties;
  iree_hal_streaming_peer_properties_initialize(
      /*access_supported=*/false, /*architectures_match=*/true, edge,
      &properties);

  EXPECT_FALSE(properties.access_supported);
  EXPECT_FALSE(properties.native_atomic_supported);
  EXPECT_FALSE(properties.array_access_supported);
  EXPECT_EQ(properties.link_type, IREE_HAL_TOPOLOGY_LINK_TYPE_UNKNOWN);
  EXPECT_EQ(properties.hop_count, 0u);
}

}  // namespace
