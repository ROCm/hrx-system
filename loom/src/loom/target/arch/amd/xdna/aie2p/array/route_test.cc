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

TEST(Aie2pRouteTest, EveryMemoryAndCorePairUsesLegalConnectedSwitches) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  const uint16_t worker_rows = family->row_count - 1;
  const uint16_t worker_count = family->column_count * worker_rows;
  for (bool packet : {false, true}) {
    for (uint16_t source_index = 0; source_index < worker_count;
         ++source_index) {
      const loom_xdna_tile_coordinate_t source = {
          static_cast<uint16_t>(source_index / worker_rows),
          static_cast<uint16_t>(1 + source_index % worker_rows)};
      for (uint16_t destination_index = 0; destination_index < worker_count;
           ++destination_index) {
        const loom_xdna_tile_coordinate_t destination = {
            static_cast<uint16_t>(destination_index / worker_rows),
            static_cast<uint16_t>(1 + destination_index % worker_rows)};
        SCOPED_TRACE(::testing::Message()
                     << source.column << "," << source.row << " -> "
                     << destination.column << "," << destination.row
                     << " packet=" << packet);
        iree_arena_reset(&arena);
        loom_aie2p_array_route_builder_t routing;
        IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
            family, 1, family->column_count + family->row_count, &arena,
            &routing));
        ASSERT_TRUE(loom_aie2p_array_route_workers(&routing, 0, 0, source, 0,
                                                   destination, 0, packet));
        // The SRAM detour remains within the existing array-wide bound; no
        // extra scratch, route capacity, or arbitrary graph search is needed.
        ASSERT_LE(routing.route_count,
                  family->column_count + family->row_count - 1);
        auto coordinate = source;
        auto arrival = LOOM_XDNA_STREAM_PORT_DMA;
        uint8_t arrival_channel = 0;
        for (size_t i = 0; i < routing.route_count; ++i) {
          const auto& edge = routing.routes[i];
          EXPECT_EQ(edge.coordinate.column, coordinate.column);
          EXPECT_EQ(edge.coordinate.row, coordinate.row);
          EXPECT_EQ(edge.source_port, arrival);
          EXPECT_EQ(edge.source_channel, arrival_channel);
          const auto* tile = loom_xdna_array_tile_facts(family, coordinate);
          const auto* input = loom_xdna_array_stream_port_range(
              family, tile->kind, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
              edge.source_port);
          const auto* output = loom_xdna_array_stream_port_range(
              family, tile->kind, LOOM_XDNA_STREAM_DIRECTION_MASTER,
              edge.destination_port);
          EXPECT_NE(loom_xdna_array_stream_destination_channels(
                        family, input, edge.source_channel, output) &
                        (UINT32_C(1) << edge.destination_channel),
                    0u);
          arrival_channel = edge.destination_channel;
          switch (edge.destination_port) {
            case LOOM_XDNA_STREAM_PORT_NORTH:
              ++coordinate.row;
              arrival = LOOM_XDNA_STREAM_PORT_SOUTH;
              break;
            case LOOM_XDNA_STREAM_PORT_SOUTH:
              --coordinate.row;
              arrival = LOOM_XDNA_STREAM_PORT_NORTH;
              break;
            case LOOM_XDNA_STREAM_PORT_EAST:
              ++coordinate.column;
              arrival = LOOM_XDNA_STREAM_PORT_WEST;
              break;
            case LOOM_XDNA_STREAM_PORT_WEST:
              --coordinate.column;
              arrival = LOOM_XDNA_STREAM_PORT_EAST;
              break;
            default:
              EXPECT_EQ(edge.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
              EXPECT_EQ(edge.destination_channel, 0);
              EXPECT_EQ(i + 1, routing.route_count);
              EXPECT_EQ(coordinate.column, destination.column);
              EXPECT_EQ(coordinate.row, destination.row);
              break;
          }
          EXPECT_LT(coordinate.column, family->column_count);
          EXPECT_LT(coordinate.row, family->row_count);
        }
      }
    }
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, SramMulticastSharesNorthAndHorizontalPrefixes) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  for (bool packet : {false, true}) {
    iree_arena_reset(&arena);
    loom_aie2p_array_route_builder_t routing;
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(family, 3, 32,
                                                             &arena, &routing));
    const uint16_t columns[] = {7, 2, 5};
    for (uint32_t branch = 0; branch < 3; ++branch) {
      const size_t begin = routing.route_count;
      ASSERT_TRUE(loom_aie2p_array_route_workers(
          &routing, branch, 0, {0, 1}, 0, {columns[branch], 3}, 0, packet));
      EXPECT_EQ(routing.route_count - begin,
                packet || branch == 0 ? columns[branch] + 3u : 1u);
      const auto& endpoint = routing.routes[routing.route_count - 1];
      EXPECT_EQ(endpoint.coordinate.column, columns[branch]);
      EXPECT_EQ(endpoint.coordinate.row, 3);
      EXPECT_EQ(endpoint.source_port, LOOM_XDNA_STREAM_PORT_WEST);
      EXPECT_EQ(endpoint.source_channel, 0);
      EXPECT_EQ(endpoint.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
      EXPECT_EQ(endpoint.destination_channel, 0);
    }
    EXPECT_EQ(routing.link_channels.northbound[family->column_count], 1u);
    EXPECT_EQ(routing.link_channels.northbound[2 * family->column_count], 1u);
    for (uint16_t column = 0; column < 7; ++column) {
      EXPECT_EQ(
          routing.link_channels.eastbound[3 * family->column_count + column],
          1u);
    }
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, SramRoutesRejectExhaustedHorizontalLinks) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  for (bool packet : {false, true}) {
    iree_arena_reset(&arena);
    loom_aie2p_array_route_builder_t routing;
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        loom_xdna_npu2_array_family(), 5, 64, &arena, &routing));
    for (uint8_t stream = 0; stream < 4; ++stream) {
      ASSERT_TRUE(loom_aie2p_array_route_workers(
          &routing, stream, stream, {0, 1}, stream,
          {static_cast<uint16_t>(stream + 1), 2}, 0, packet));
    }
    EXPECT_FALSE(loom_aie2p_array_route_workers(&routing, 4, 4, {0, 1}, 4,
                                                {5, 2}, 0, packet));
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

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
