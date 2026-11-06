// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/route.h"

#include <tuple>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

namespace {

class Aie2pRouteTest : public ::testing::TestWithParam<std::tuple<int, bool>> {
};

TEST_P(Aie2pRouteTest, SharedSourceUsesOneLaneAndRetainsPacketPrefixes) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  loom_aie2p_array_route_builder_t routing;
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(family, 3, 32,
                                                           &arena, &routing));
  const auto [source_row, packet] = GetParam();
  const loom_xdna_tile_coordinate_t source = {
      0, static_cast<uint16_t>(source_row)};
  const uint16_t rows[] = {3, 5, 4};
  uint16_t frontier = source.row;
  for (uint32_t branch = 0; branch < 3; ++branch) {
    const size_t begin = routing.route_count;
    ASSERT_TRUE(loom_aie2p_array_route_workers(&routing, branch, 0, source, 0,
                                               {0, rows[branch]}, 0, packet));
    // Packets need a filter at every hop for each class. Circuits only need
    // the newly allocated suffix and the new endpoint connection.
    const uint16_t first_row = packet ? source.row : frontier;
    const size_t hops = rows[branch] > first_row ? rows[branch] - first_row : 0;
    ASSERT_EQ(routing.route_count - begin, hops + 1);
    for (size_t hop = 0; hop < hops; ++hop) {
      const auto& edge = routing.routes[begin + hop];
      EXPECT_EQ(edge.channel_index, branch);
      EXPECT_EQ(edge.coordinate.column, 0);
      EXPECT_EQ(edge.coordinate.row, first_row + hop);
      EXPECT_EQ(edge.destination_port, LOOM_XDNA_STREAM_PORT_NORTH);
      EXPECT_EQ(edge.destination_channel, 0);
    }
    const auto& endpoint = routing.routes[routing.route_count - 1];
    EXPECT_EQ(endpoint.coordinate.row, rows[branch]);
    EXPECT_EQ(endpoint.source_port, LOOM_XDNA_STREAM_PORT_SOUTH);
    EXPECT_EQ(endpoint.source_channel, 0);
    EXPECT_EQ(endpoint.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
    EXPECT_EQ(endpoint.destination_channel, 0);
    frontier = rows[branch] > frontier ? rows[branch] : frontier;
  }
  for (uint16_t row = source.row; row < 5; ++row) {
    EXPECT_EQ(routing.link_channels.northbound[row * family->column_count], 1u);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

INSTANTIATE_TEST_SUITE_P(SourceAndMode, Aie2pRouteTest,
                         ::testing::Combine(::testing::Values(1, 2),
                                            ::testing::Bool()));

TEST(Aie2pRouteTest, MemoryMulticastBranchesInTheReceivingComputeRow) {
  const auto* family = loom_xdna_npu2_array_family();
  const auto* memory =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_MEMORY);
  const auto* compute =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  for (uint16_t column = 0; column < family->column_count; ++column) {
    for (uint16_t row = compute->first_row;
         row < compute->first_row + compute->row_count; ++row) {
      for (uint8_t source_dma = 0;
           source_dma < memory->dma.channel_count_per_direction; ++source_dma) {
        for (uint8_t destination_dma = 0;
             destination_dma < compute->dma.channel_count_per_direction;
             ++destination_dma) {
          for (bool packet : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "source=" << column << "," << +memory->first_row
                         << " dma=" << +source_dma << " row=" << row << " dma="
                         << +destination_dma << " packet=" << packet);
            iree_arena_block_pool_t blocks;
            iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                             &blocks);
            iree_arena_allocator_t arena;
            iree_arena_initialize(&blocks, &arena);
            loom_aie2p_array_route_builder_t routing;
            IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
                family, family->column_count, 128, &arena, &routing));
            for (uint32_t destination = 0; destination < family->column_count;
                 ++destination) {
              const size_t begin = routing.route_count;
              ASSERT_TRUE(loom_aie2p_array_route_workers(
                  &routing, destination, 0, {column, memory->first_row},
                  source_dma, {static_cast<uint16_t>(destination), row},
                  destination_dma, packet));
              for (size_t i = begin; i < routing.route_count; ++i) {
                const auto& edge = routing.routes[i];
                EXPECT_EQ(edge.channel_index, destination);
                if (edge.destination_port == LOOM_XDNA_STREAM_PORT_NORTH) {
                  EXPECT_EQ(edge.coordinate.column, column);
                  EXPECT_LT(edge.coordinate.row, row);
                } else if (edge.destination_port ==
                               LOOM_XDNA_STREAM_PORT_EAST ||
                           edge.destination_port ==
                               LOOM_XDNA_STREAM_PORT_WEST) {
                  EXPECT_EQ(edge.coordinate.row, row);
                } else {
                  EXPECT_EQ(edge.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
                  EXPECT_EQ(edge.coordinate.column, destination);
                  EXPECT_EQ(edge.coordinate.row, row);
                  EXPECT_EQ(edge.destination_channel, destination_dma);
                }
              }
              const auto& endpoint = routing.routes[routing.route_count - 1];
              EXPECT_EQ(endpoint.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
            }
            // Circuit branches reuse one vertical trunk and one horizontal
            // row spanning every destination; only endpoint taps are added.
            if (!packet) {
              EXPECT_EQ(routing.route_count, row - memory->first_row +
                                                 family->column_count - 1 +
                                                 family->column_count);
            }
            iree_arena_deinitialize(&arena);
            iree_arena_block_pool_deinitialize(&blocks);
          }
        }
      }
    }
  }
}

}  // namespace
