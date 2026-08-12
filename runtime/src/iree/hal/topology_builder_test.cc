// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/topology_builder.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

using ::iree::testing::status::IsOk;
using ::iree::testing::status::StatusIs;
using ::testing::Eq;
using ::testing::Ne;

static iree_hal_device_spec_t* CreateTestDeviceSpec(
    const char* driver_id, uint32_t logical_ordinal, uint32_t physical_ordinal,
    uint32_t numa_node,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  iree_hal_physical_device_spec_t physical_device = {
      /*.identity=*/
      {
          /*.display_name=*/iree_make_cstring_view("test physical device"),
          /*.backend_path=*/iree_make_cstring_view("test://physical"),
          /*.vendor_id=*/0,
          /*.device_id=*/0,
          /*.revision_id=*/0,
          /*.uuid=*/{{0}},
          /*.pci=*/{0, 0, 0, 0},
          /*.numa=*/{numa_node},
          /*.flags=*/IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE,
      },
      /*.physical_ordinal=*/physical_ordinal,
      /*.partition_ordinal=*/0,
      /*.partition_count=*/1,
      /*.physical_device_affinity=*/physical_device_affinity,
  };
  iree_hal_device_identity_spec_t identity = {
      /*.logical_device_id=*/iree_make_cstring_view(driver_id),
      /*.display_name=*/iree_make_cstring_view("test logical device"),
      /*.driver_id=*/iree_make_cstring_view(driver_id),
      /*.driver_version=*/iree_make_cstring_view("test"),
      /*.backend_id=*/iree_make_cstring_view("test"),
      /*.device_path=*/iree_make_cstring_view("test://logical"),
      /*.vendor_name=*/iree_make_cstring_view("test"),
      /*.vendor_id=*/0,
      /*.device_id=*/0,
      /*.revision_id=*/0,
      /*.logical_ordinal=*/logical_ordinal,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/&identity,
  };
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_CHECK_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                            &device_spec));
  return device_spec;
}

//===----------------------------------------------------------------------===//
// Builder initialization tests
//===----------------------------------------------------------------------===//

// Tests basic builder initialization.
TEST(TopologyBuilder, Initialize) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 4);

  // Verify device count was set.
  EXPECT_EQ(builder.device_count, 4);

  // Verify self-edges were initialized with optimal scheduling settings.
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t idx = i * 4 + i;
    EXPECT_TRUE(builder.edges_set[idx]);
    EXPECT_EQ(iree_hal_topology_edge_wait_mode(builder.device_edges[idx].lo),
              IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
    EXPECT_EQ(iree_hal_topology_edge_signal_mode(builder.device_edges[idx].lo),
              IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
    EXPECT_EQ(iree_hal_topology_edge_link_class(builder.device_edges[idx].lo),
              IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
  }

  // Verify self-edges require no external semaphore handles and retain native
  // buffer handles in the cold word.
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t idx = i * 4 + i;
    EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(
                  builder.device_edges[idx].hi),
              IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
    EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(
                  builder.device_edges[idx].hi),
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  }
}

//===----------------------------------------------------------------------===//
// Edge setting tests
//===----------------------------------------------------------------------===//

// Tests setting edges in the builder.
TEST(TopologyBuilder, SetEdges) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Set self-edges (already initialized, but we can re-set them).
  iree_hal_topology_edge_t self_edge = iree_hal_topology_edge_make_self();
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 0, self_edge));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 1, self_edge));

  // Set cross-device edges with custom scheduling and interop settings.
  iree_hal_topology_edge_t cross_edge =
      iree_hal_topology_edge_make_host_staged();
  cross_edge.lo = iree_hal_topology_edge_set_wait_mode(
      cross_edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  cross_edge.lo = iree_hal_topology_edge_set_signal_mode(
      cross_edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  cross_edge.lo = iree_hal_topology_edge_set_link_class(
      cross_edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);

  // Set interop types for the cross-device edge.
  cross_edge.hi = iree_hal_topology_edge_set_semaphore_import_timepoint_types(
      cross_edge.hi, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_ASYNC_PRIMITIVE);
  cross_edge.hi = iree_hal_topology_edge_set_buffer_import_types(
      cross_edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  cross_edge.hi = iree_hal_topology_edge_set_link_type(
      cross_edge.hi, IREE_HAL_TOPOLOGY_LINK_TYPE_PCIE);
  cross_edge.hi = iree_hal_topology_edge_set_path_hop_count(cross_edge.hi, 2);

  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 0, 1, cross_edge));
  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 1, 0, cross_edge));

  // Build and get topology.
  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  // Check the scheduling word in the topology.
  EXPECT_EQ(topology->device_count, 2);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(topology->device_edges[0].lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(topology->device_edges[1].lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Check the cold word survived the finalize copy.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(
                topology->device_edges[1].hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_ASYNC_PRIMITIVE);
  EXPECT_EQ(
      iree_hal_topology_edge_buffer_import_types(topology->device_edges[1].hi),
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_link_type(topology->device_edges[1].hi),
            IREE_HAL_TOPOLOGY_LINK_TYPE_PCIE);
  EXPECT_EQ(iree_hal_topology_edge_path_hop_count(topology->device_edges[1].hi),
            2);
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

// Tests that finalized topologies expose normalized logical-device nodes.
TEST(TopologyBuilder, LogicalDeviceNodes) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  iree_hal_topology_edge_t cross_edge =
      iree_hal_topology_edge_make_host_staged();
  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 0, 1, cross_edge));
  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 1, 0, cross_edge));

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  EXPECT_EQ(iree_hal_topology_node_count(topology), 2);
  const iree_hal_topology_node_t* node0 =
      iree_hal_topology_node_at(topology, 0);
  ASSERT_NE(node0, nullptr);
  EXPECT_EQ(node0->ordinal, 0u);
  EXPECT_EQ(node0->parent_ordinal, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
  EXPECT_EQ(node0->device_ordinal, 0u);
  EXPECT_EQ(node0->kind, IREE_HAL_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE);
  const iree_hal_topology_node_t* node1 =
      iree_hal_topology_node_at(topology, 1);
  ASSERT_NE(node1, nullptr);
  EXPECT_EQ(node1->ordinal, 1u);
  EXPECT_EQ(node1->device_ordinal, 1u);
  EXPECT_EQ(iree_hal_topology_node_at(topology, 2), nullptr);
  EXPECT_EQ(iree_hal_topology_link_count(topology), 0);
  EXPECT_EQ(iree_hal_topology_link_at(topology, 0), nullptr);

  iree_hal_topology_destroy(topology, iree_allocator_system());
}

// Tests that spec-aware finalize exposes physical placement nodes and links.
TEST(TopologyBuilder, DeviceSpecNodesAndLinks) {
  iree_hal_device_spec_t* spec_a =
      CreateTestDeviceSpec("mock", 7, 11, 2, 1ull << 0);
  iree_hal_device_spec_t* spec_b =
      CreateTestDeviceSpec("mock", 8, 12, 3, 1ull << 1);
  const iree_hal_device_spec_t* specs[2] = {spec_a, spec_b};

  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, edge));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, edge));

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize_with_device_specs(
      &builder, specs, iree_allocator_system(), &topology));

  EXPECT_EQ(iree_hal_topology_device_numa_node(topology, 0), 2u);
  EXPECT_EQ(iree_hal_topology_device_numa_node(topology, 1), 3u);
  ASSERT_EQ(iree_hal_topology_node_count(topology), 6u);
  ASSERT_EQ(iree_hal_topology_link_count(topology), 4u);

  const iree_hal_topology_node_t* numa0 =
      iree_hal_topology_node_at(topology, 0);
  ASSERT_NE(numa0, nullptr);
  EXPECT_EQ(numa0->kind, IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA);
  EXPECT_EQ(numa0->local_ordinal, 2u);
  EXPECT_EQ(numa0->device_ordinal, IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID);

  const iree_hal_topology_node_t* logical0 =
      iree_hal_topology_node_at(topology, 2);
  ASSERT_NE(logical0, nullptr);
  EXPECT_EQ(logical0->kind, IREE_HAL_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE);
  EXPECT_EQ(logical0->device_ordinal, 0u);
  EXPECT_EQ(logical0->local_ordinal, 7u);

  const iree_hal_topology_node_t* physical0 =
      iree_hal_topology_node_at(topology, 4);
  ASSERT_NE(physical0, nullptr);
  EXPECT_EQ(physical0->kind, IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE);
  EXPECT_EQ(physical0->parent_ordinal, logical0->ordinal);
  EXPECT_EQ(physical0->device_ordinal, 0u);
  EXPECT_EQ(physical0->local_ordinal, 11u);
  EXPECT_EQ(physical0->physical_device_affinity, 1ull << 0);

  const iree_hal_topology_link_t* contains =
      iree_hal_topology_link_at(topology, 0);
  ASSERT_NE(contains, nullptr);
  EXPECT_EQ(contains->source_node_ordinal, logical0->ordinal);
  EXPECT_EQ(contains->target_node_ordinal, physical0->ordinal);
  EXPECT_EQ(contains->kind, IREE_HAL_TOPOLOGY_LINK_KIND_CONTAINS);

  const iree_hal_topology_link_t* numa_link =
      iree_hal_topology_link_at(topology, 1);
  ASSERT_NE(numa_link, nullptr);
  EXPECT_EQ(numa_link->source_node_ordinal, numa0->ordinal);
  EXPECT_EQ(numa_link->target_node_ordinal, physical0->ordinal);
  EXPECT_EQ(numa_link->kind, IREE_HAL_TOPOLOGY_LINK_KIND_INTERCONNECT);
  EXPECT_TRUE(numa_link->flags & IREE_HAL_TOPOLOGY_LINK_FLAG_BIDIRECTIONAL);

  iree_hal_topology_destroy(topology, iree_allocator_system());
  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

// Tests that both words of an edge are preserved through set_edge and finalize.
TEST(TopologyBuilder, FullEdgePreservation) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Build a fully-populated cross-device edge.
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();
  edge.lo = iree_hal_topology_edge_set_wait_mode(
      edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  edge.lo = iree_hal_topology_edge_set_signal_mode(
      edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  edge.lo = iree_hal_topology_edge_set_capability_flags(
      edge.lo, IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                   IREE_HAL_TOPOLOGY_CAPABILITY_REMOTE_DMA);
  edge.lo = iree_hal_topology_edge_set_wait_cost(edge.lo, 3);
  edge.lo = iree_hal_topology_edge_set_copy_cost(edge.lo, 7);
  edge.lo = iree_hal_topology_edge_set_link_class(
      edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC);

  edge.hi = iree_hal_topology_edge_set_semaphore_import_timepoint_types(
      edge.hi, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT);
  edge.hi = iree_hal_topology_edge_set_semaphore_export_timepoint_types(
      edge.hi, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT);
  edge.hi = iree_hal_topology_edge_set_buffer_import_types(
      edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                   IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  edge.hi = iree_hal_topology_edge_set_buffer_export_types(
      edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                   IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  edge.hi = iree_hal_topology_edge_set_link_type(
      edge.hi, IREE_HAL_TOPOLOGY_LINK_TYPE_INFINIBAND);
  edge.hi = iree_hal_topology_edge_set_path_hop_count(edge.hi, 4);

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, edge));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, edge));

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  // Verify the entire scheduling word survived.
  iree_hal_topology_edge_t result = topology->device_edges[1];
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(result.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(result.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(result.lo),
            IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                IREE_HAL_TOPOLOGY_CAPABILITY_REMOTE_DMA);
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(result.lo), 3);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(result.lo), 7);
  EXPECT_EQ(iree_hal_topology_edge_link_class(result.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC);

  // Verify the entire cold word survived.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(result.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_timepoint_types(result.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(result.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(result.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  EXPECT_EQ(iree_hal_topology_edge_link_type(result.hi),
            IREE_HAL_TOPOLOGY_LINK_TYPE_INFINIBAND);
  EXPECT_EQ(iree_hal_topology_edge_path_hop_count(result.hi), 4);
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

//===----------------------------------------------------------------------===//
// Validation tests
//===----------------------------------------------------------------------===//

// Tests that validation fails when not all edges are set.
TEST(TopologyBuilder, ValidationFailsIncomplete) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Clear one of the edges_set flags to simulate missing edge.
  // Self-edges [0,0] and [1,1] are set by initialize.
  // Don't set cross-edges [0,1] and [1,0].
  builder.edges_set[0 * 2 + 1] = false;  // mark [0,1] as not set
  builder.edges_set[1 * 2 + 0] = false;  // mark [1,0] as not set

  iree_hal_topology_t* topology = NULL;
  iree_status_t status = iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

// Tests that validation fails for asymmetric link classes.
TEST(TopologyBuilder, ValidationFailsAsymmetricLinks) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Set asymmetric cross-device edges (different link classes).
  iree_hal_topology_edge_t edge1 = iree_hal_topology_edge_make_host_staged();
  edge1.lo = iree_hal_topology_edge_set_link_class(
      edge1.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);

  iree_hal_topology_edge_t edge2 = iree_hal_topology_edge_make_host_staged();
  edge2.lo = iree_hal_topology_edge_set_link_class(
      edge2.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, edge1));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, edge2));

  iree_hal_topology_t* topology = NULL;
  iree_status_t status = iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

// Tests that self-edges must have NATIVE wait mode.
TEST(TopologyBuilder, ValidationFailsNonNativeSelfEdge) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 1);

  // Try to set a self-edge with non-NATIVE wait mode.
  iree_hal_topology_edge_t bad_self = iree_hal_topology_edge_make_self();
  bad_self.lo = iree_hal_topology_edge_set_wait_mode(
      bad_self.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  iree_status_t status =
      iree_hal_topology_builder_set_edge(&builder, 0, 0, bad_self);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// NUMA node tests
//===----------------------------------------------------------------------===//

// Tests NUMA node assignment.
TEST(TopologyBuilder, NumaNodes) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 4);

  // Assign NUMA nodes.
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 0, 0));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 1, 0));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 2, 1));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 3, 1));

  // Set cross-device edges (self-edges already set by initialize).
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t j = 0; j < 4; ++j) {
      if (i != j) {
        iree_hal_topology_edge_t cross =
            iree_hal_topology_edge_make_host_staged();
        // Same NUMA node = lower cost (costs are 4-bit, max 15).
        uint8_t cost =
            (builder.device_numa_nodes[i] == builder.device_numa_nodes[j]) ? 5
                                                                           : 12;
        cross.lo = iree_hal_topology_edge_set_copy_cost(cross.lo, cost);
        IREE_ASSERT_OK(
            iree_hal_topology_builder_set_edge(&builder, i, j, cross));
      }
    }
  }

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  // Check NUMA nodes.
  EXPECT_EQ(iree_hal_topology_device_numa_node(topology, 0), 0);
  EXPECT_EQ(iree_hal_topology_device_numa_node(topology, 1), 0);
  EXPECT_EQ(iree_hal_topology_device_numa_node(topology, 2), 1);
  EXPECT_EQ(iree_hal_topology_device_numa_node(topology, 3), 1);

  // Check that intra-NUMA transfers have lower cost.
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(topology->device_edges[1].lo),
            5);  // same NUMA
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(topology->device_edges[2].lo),
            12);  // different NUMA
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

//===----------------------------------------------------------------------===//
// Topology query tests
//===----------------------------------------------------------------------===//

// Tests querying edges from a finalized topology.
TEST(TopologyBuilder, QueryEdges) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 3);

  // Set cross-device edges with varying configurations.
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      if (i != j) {
        iree_hal_topology_edge_t edge =
            iree_hal_topology_edge_make_host_staged();
        // Give each edge a unique copy cost for identification.
        edge.lo =
            iree_hal_topology_edge_set_copy_cost(edge.lo, (uint8_t)(i * 3 + j));
        // Set link class consistently for symmetry validation.
        edge.lo = iree_hal_topology_edge_set_link_class(
            edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);
        IREE_ASSERT_OK(
            iree_hal_topology_builder_set_edge(&builder, i, j, edge));
      }
    }
  }

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  // Query each edge and verify the copy cost encodes the (i,j) pair.
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      iree_hal_topology_edge_t queried =
          iree_hal_topology_query_edge(topology, i, j);
      if (i == j) {
        // Self-edges have zero copy cost.
        EXPECT_EQ(iree_hal_topology_edge_copy_cost(queried.lo), 0);
      } else {
        EXPECT_EQ(iree_hal_topology_edge_copy_cost(queried.lo), i * 3 + j);
      }
    }
  }
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

}  // namespace
}  // namespace iree::hal
