// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

namespace {

class ExecutionPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &source_arena_);
    iree_arena_initialize(&blocks_, &retained_arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&retained_arena_);
    iree_arena_deinitialize(&source_arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  // Blocks shared by two independently retired compilation owners.
  iree_arena_block_pool_t blocks_ = {};
  // Original source selection and compilation.
  iree_arena_allocator_t source_arena_ = {};
  // Independent projected compilation.
  iree_arena_allocator_t retained_arena_ = {};
};

TEST_F(ExecutionPlanTest, SourceRetirementPreservesRoutesAndPhysicalOwnership) {
  const auto* family = loom_xdna_npu2_array_family();
  const size_t tile_count = family->column_count * family->row_count;
  std::vector<loom_aie2p_native_tile_t> tiles(tile_count);
  for (size_t i = 0; i < tile_count; ++i) {
    tiles[i].coordinate = {static_cast<uint16_t>(i / family->row_count),
                           static_cast<uint16_t>(i % family->row_count)};
    tiles[i].facts = loom_xdna_array_tile_facts(family, tiles[i].coordinate);
  }
  loom_pass_t pass = {.arena = &source_arena_};
  loom_aie2p_native_context_t context = {
      .pass = &pass, .family = family, .tiles = tiles.data()};
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      family, 16, &source_arena_, &context.routing));
  std::vector<loom_aie2p_native_control_source_t> sources(tile_count * 2);
  std::vector<uint32_t> destinations(tile_count * 2, UINT32_MAX);
  loom_aie2p_native_control_connection_t connection = {
      .source = {{0, 2}, LOOM_XDNA_STREAM_PORT_CORE, 0},
      .destination = {{0, 1}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0},
      .packet = 3,
  };
  sources[4].destination = 3;
  destinations[3] = 4;
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &context.routing, 0, &sources[4].tree, connection.source.coordinate,
      connection.source.port, 0, connection.destination.coordinate,
      connection.destination.port, 0, true));
  std::vector<loom_aie2p_native_route_t> routes(context.routing.route_count);
  context.routes = routes.data();
  context.control_sources = sources.data();
  context.control_destination_sources = destinations.data();
  loom_aie2p_native_route_label(&context, 0, connection.packet, 4);
  bool valid = false;
  IREE_ASSERT_OK(loom_aie2p_native_switches_build(&context, nullptr, &valid));
  ASSERT_TRUE(valid);

  constexpr auto none = IREE_HOST_SIZE_MAX;
  const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, none},
  };
  const iree_host_size_t owners[] = {1};
  loom_pipeline_lifetimes_t lifetimes = {
      .nodes = nodes, .count = IREE_ARRAYSIZE(nodes), .owners = owners};
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&source_arena_, &lifetimes));
  loom_pipeline_entry_scope_t* scope = nullptr;
  IREE_ASSERT_OK(loom_pipeline_entry_scope_create(
      &loom_aie2p_execution_plan_type, &lifetimes, 1, &source_arena_, &scope));
  std::vector<loom_aie2p_execution_tile_t> occupancy(tile_count);
  occupancy[2] = {.memory_length = 2560,
                  .lock_count = 3,
                  .descriptor_counts = {2, 0},
                  .mm2s_count = 1,
                  .s2mm_count = 1};
  const loom_aie2p_execution_plan_t selected = {
      .tiles = occupancy.data(),
      .routing = context.routing,
      .routes = routes.data(),
      .control_sources = sources.data(),
      .control_destination_sources = destinations.data(),
      .switches = context.switches,
      .connections = &connection,
      .connection_count = 1,
  };
  auto* original = reinterpret_cast<loom_aie2p_execution_plan_t*>(scope);
  IREE_ASSERT_OK(loom_aie2p_execution_plan_copy_resources(
      &selected, &source_arena_, original));
  const loom_pipeline_entry_scope_t* cloned = nullptr;
  IREE_ASSERT_OK(
      loom_pipeline_entry_scope_clone(scope, &retained_arena_, &cloned));
  const auto* retained =
      reinterpret_cast<const loom_aie2p_execution_plan_t*>(cloned);
  EXPECT_EQ(cloned->type, &loom_aie2p_execution_plan_type);
  EXPECT_NE(retained->routing.routes, original->routing.routes);
  EXPECT_NE(retained->switches->banks[0].words,
            original->switches->banks[0].words);
  iree_arena_reset(&source_arena_);
  tiles.clear();
  routes.clear();
  sources.clear();
  destinations.clear();
  occupancy.clear();

  EXPECT_EQ(retained->base.lifetimes.owners[0], 1u);
  EXPECT_EQ(retained->tiles[2].memory_length, 2560u);
  EXPECT_EQ(retained->tiles[2].lock_count, 3u);
  EXPECT_EQ(retained->tiles[2].descriptor_counts[0], 2u);
  EXPECT_EQ(retained->tiles[2].mm2s_count, 1u);
  EXPECT_EQ(retained->tiles[2].s2mm_count, 1u);
  EXPECT_EQ(retained->control_sources[4].destination, 3u);
  EXPECT_EQ(retained->control_destination_sources[3], 4u);
  for (size_t i = 0; i < retained->routing.route_count; ++i) {
    EXPECT_EQ(retained->routes[i].edge, &retained->routing.routes[i]);
    EXPECT_EQ(retained->routes[i].control_source, 4u);
  }
  loom_aie2p_native_connectivity_t* connectivity = nullptr;
  IREE_ASSERT_OK(loom_aie2p_native_connectivity_allocate(
      family, retained->switches, retained->routing.routes,
      retained->routing.route_count, &retained_arena_, &connectivity));
  ASSERT_EQ(retained->connection_count, 1u);
  const auto& intended = retained->connections[0];
  EXPECT_EQ(loom_aie2p_native_connectivity_check(connectivity, intended.source,
                                                 intended.packet,
                                                 &intended.destination, 1, true)
                .failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);

  loom_aie2p_array_route_builder_t extended;
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_clone(
      &retained->routing, retained->routing.route_count + 8, &retained_arena_,
      &extended));
  auto sender = retained->control_sources[4].tree;
  EXPECT_TRUE(loom_aie2p_array_route_stream(
      &extended, 1, &sender, intended.source.coordinate, intended.source.port,
      0, {0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0, true));
  EXPECT_GT(extended.route_count, retained->routing.route_count);
}

}  // namespace
