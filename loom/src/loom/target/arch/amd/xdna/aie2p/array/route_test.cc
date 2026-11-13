// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/route.h"

#include <tuple>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

namespace {

class Aie2pRouteTest : public ::testing::TestWithParam<std::tuple<int, bool>> {
};

TEST(Aie2pRouteTest, PayloadAndCompletionSourcesOwnIndependentPrefixes) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  loom_aie2p_array_route_builder_t routing;
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      loom_xdna_npu2_array_family(), 32, &arena, &routing));
  loom_aie2p_array_route_tree_t payload = {};
  loom_aie2p_array_route_tree_t completion = {};
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &routing, 1000, &payload, {0, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0, {0, 3},
      LOOM_XDNA_STREAM_PORT_DMA, 0, false));
  const size_t first = routing.route_count;
  // Labels can coincide without merging independent physical sources.
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &routing, 1000, &completion, {0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL,
      0, {0, 3}, LOOM_XDNA_STREAM_PORT_CORE, 0, true));
  ASSERT_NE(payload.root, completion.root);
  const auto& origin = routing.routes[first];
  EXPECT_EQ(origin.coordinate.column, 0);
  EXPECT_EQ(origin.coordinate.row, 0);
  EXPECT_EQ(origin.source_port, LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
  EXPECT_EQ(origin.source_channel, 0);
  EXPECT_EQ(routing.route_count - first, 4u);
  const auto& terminal = routing.routes[routing.route_count - 1];
  EXPECT_EQ(terminal.coordinate.row, 3);
  EXPECT_EQ(terminal.destination_port, LOOM_XDNA_STREAM_PORT_CORE);
  EXPECT_NE(terminal.source_channel, routing.routes[first - 1].source_channel);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST_P(Aie2pRouteTest, SharedSourceUsesOneLaneAndRetainsPacketPrefixes) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t trees[8] = {};
  IREE_ASSERT_OK(
      loom_aie2p_array_route_builder_initialize(family, 32, &arena, &routing));
  const auto [source_row, packet] = GetParam();
  const loom_xdna_tile_coordinate_t source = {
      0, static_cast<uint16_t>(source_row)};
  const uint16_t rows[] = {3, 5, 4};
  uint16_t frontier = source.row;
  for (uint32_t branch = 0; branch < 3; ++branch) {
    const size_t begin = routing.route_count;
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &routing, branch, &trees[0], source, 0, {0, rows[branch]}, 0, packet));
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
      EXPECT_FALSE(edge.dma_destination);
    }
    const auto& endpoint = routing.routes[routing.route_count - 1];
    EXPECT_EQ(endpoint.coordinate.row, rows[branch]);
    EXPECT_EQ(endpoint.source_port, LOOM_XDNA_STREAM_PORT_SOUTH);
    EXPECT_EQ(endpoint.source_channel, 0);
    EXPECT_EQ(endpoint.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
    EXPECT_EQ(endpoint.destination_channel, 0);
    EXPECT_TRUE(endpoint.dma_destination);
    frontier = rows[branch] > frontier ? rows[branch] : frontier;
  }
  for (uint16_t row = source.row; row < 5; ++row) {
    EXPECT_EQ(routing.link_channels.northbound[row * family->column_count], 1u);
  }
  std::vector<size_t> counts(family->column_count * family->row_count);
  for (size_t i = 0; i < routing.route_count; ++i) {
    const auto coordinate = routing.routes[i].coordinate;
    ++counts[coordinate.column * family->row_count + coordinate.row];
  }
  for (size_t i = 0; i < counts.size(); ++i) {
    EXPECT_EQ(routing.tile_route_counts[i], counts[i]);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST_P(Aie2pRouteTest, CrossColumnEgressPrefersTheShimRow) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t trees[8] = {};
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      loom_xdna_npu2_array_family(), 16, &arena, &routing));
  const auto [source_row, packet] = GetParam();
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &routing, 0, &trees[0], {0, static_cast<uint16_t>(source_row)},
      LOOM_XDNA_STREAM_PORT_DMA, 0, {1, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0,
      packet));
  ASSERT_EQ(routing.route_count, source_row + 3);
  for (size_t i = 0; i < routing.route_count; ++i) {
    const auto& edge = routing.routes[i];
    if (edge.destination_port == LOOM_XDNA_STREAM_PORT_EAST) {
      EXPECT_EQ(edge.coordinate.row, 0);
    }
  }
  EXPECT_EQ(routing.routes[source_row].destination_port,
            LOOM_XDNA_STREAM_PORT_EAST);
  EXPECT_TRUE(routing.routes[routing.route_count - 2].dma_destination);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

INSTANTIATE_TEST_SUITE_P(SourceAndMode, Aie2pRouteTest,
                         ::testing::Combine(::testing::Values(1, 2),
                                            ::testing::Bool()));

TEST(Aie2pRouteTest, RelocatedEgressAvoidsSaturatedSourceColumn) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  for (uint16_t column = 0; column < family->column_count; ++column) {
    SCOPED_TRACE(column);
    iree_arena_reset(&arena);
    loom_aie2p_array_route_builder_t routing;
    loom_aie2p_array_route_tree_t controls[4] = {};
    loom_aie2p_array_route_tree_t payloads[4] = {};
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(family, 64, &arena,
                                                             &routing));
    for (uint16_t worker = 0; worker < 4; ++worker) {
      const loom_xdna_tile_coordinate_t core = {
          column, static_cast<uint16_t>(worker + 2)};
      const loom_xdna_tile_coordinate_t shim = {
          worker < 2 ? column : static_cast<uint16_t>(column ? column - 1 : 1),
          0};
      ASSERT_TRUE(loom_aie2p_array_route_stream(
          &routing, worker * 2, &controls[worker], core,
          LOOM_XDNA_STREAM_PORT_CORE, 0, shim,
          LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0, true));
      ASSERT_TRUE(loom_aie2p_array_route_stream(
          &routing, worker * 2 + 1, &payloads[worker], core,
          LOOM_XDNA_STREAM_PORT_DMA, 0, shim, LOOM_XDNA_STREAM_PORT_DMA,
          worker % 2, false));
    }
    // The first pair already consumes all four southbound links: one control
    // and one payload per core. The other pair must descend in its DMA column.
    EXPECT_EQ(routing.link_channels.southbound[column], 15u);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, CompletionRepliesShareTheirSourceColumnPrefix) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t payloads[6] = {};
  loom_aie2p_array_route_tree_t replies = {};
  IREE_ASSERT_OK(
      loom_aie2p_array_route_builder_initialize(family, 64, &arena, &routing));
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(payloads); ++i) {
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &routing, i, &payloads[i], {1, 1}, i,
        {1, static_cast<uint16_t>(2 + i / 2)}, i % 2, false));
  }
  // The receiving column's SRAM-to-core links are full of payloads. Replies
  // can still reach its cores by sharing an upward prefix in the shim column.
  EXPECT_EQ(routing.link_channels.northbound[family->column_count + 1], 63u);
  const uint16_t destinations[] = {3, 5, 4};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(destinations); ++i) {
    const size_t begin = routing.route_count;
    ASSERT_TRUE(loom_aie2p_array_route_stream(
        &routing, 6 + i, &replies, {0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL,
        0, {1, destinations[i]}, LOOM_XDNA_STREAM_PORT_CORE, 0, true));
    ASSERT_EQ(routing.route_count - begin, destinations[i] + 2);
    for (size_t row = 0; row < destinations[i]; ++row) {
      const auto& edge = routing.routes[begin + row];
      EXPECT_EQ(edge.coordinate.column, 0);
      EXPECT_EQ(edge.coordinate.row, row);
      EXPECT_EQ(edge.destination_port, LOOM_XDNA_STREAM_PORT_NORTH);
      EXPECT_EQ(edge.destination_channel, 0);
    }
    const auto& turn = routing.routes[begin + destinations[i]];
    EXPECT_EQ(turn.coordinate.row, destinations[i]);
    EXPECT_EQ(turn.destination_port, LOOM_XDNA_STREAM_PORT_EAST);
    const auto& end = routing.routes[routing.route_count - 1];
    EXPECT_EQ(end.coordinate.column, 1);
    EXPECT_EQ(end.coordinate.row, destinations[i]);
    EXPECT_EQ(end.destination_port, LOOM_XDNA_STREAM_PORT_CORE);
  }
  for (size_t row = 0; row < 5; ++row) {
    EXPECT_EQ(routing.link_channels.northbound[row * family->column_count], 1u);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, FailedSuffixDiscardsTentativePacketPrefix) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t trees[5] = {};
  IREE_ASSERT_OK(
      loom_aie2p_array_route_builder_initialize(family, 64, &arena, &routing));
  for (uint16_t worker = 0; worker < 4; ++worker) {
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &routing, worker, &trees[worker],
        {0, static_cast<uint16_t>(worker + 2)}, 0, {0, 1}, worker, false));
  }
  // Reusing this horizontal prefix is legal, but extending it to SRAM would
  // need a fifth southbound lane. Rejection must not leave its new filter.
  ASSERT_TRUE(loom_aie2p_array_route_workers(&routing, 4, &trees[4], {1, 3}, 0,
                                             {0, 3}, 0, true));
  const size_t first = routing.route_count;
  const size_t source_tile = family->row_count + 3;
  const auto prefix_count = routing.tile_route_counts[source_tile];
  const auto node_count = routing.prefixes.node_count;
  EXPECT_FALSE(loom_aie2p_array_route_workers(&routing, 5, &trees[4], {1, 3}, 0,
                                              {0, 1}, 4, true));
  EXPECT_EQ(routing.route_count, first);
  EXPECT_EQ(routing.tile_route_counts[source_tile], prefix_count);
  EXPECT_EQ(routing.prefixes.node_count, node_count);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

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
        loom_aie2p_array_route_tree_t trees[8] = {};
        IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
            family, family->column_count + family->row_count, &arena,
            &routing));
        ASSERT_TRUE(loom_aie2p_array_route_workers(
            &routing, 0, &trees[0], source, 0, destination, 0, packet));
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
    loom_aie2p_array_route_tree_t trees[8] = {};
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(family, 32, &arena,
                                                             &routing));
    const uint16_t columns[] = {7, 2, 5};
    for (uint32_t branch = 0; branch < 3; ++branch) {
      const size_t begin = routing.route_count;
      ASSERT_TRUE(
          loom_aie2p_array_route_workers(&routing, branch, &trees[0], {0, 1}, 0,
                                         {columns[branch], 3}, 0, packet));
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
    loom_aie2p_array_route_tree_t trees[8] = {};
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        loom_xdna_npu2_array_family(), 64, &arena, &routing));
    for (uint8_t stream = 0; stream < 4; ++stream) {
      ASSERT_TRUE(loom_aie2p_array_route_workers(
          &routing, stream, &trees[stream], {0, 1}, stream,
          {static_cast<uint16_t>(stream + 1), 2}, 0, packet));
    }
    EXPECT_FALSE(loom_aie2p_array_route_workers(&routing, 4, &trees[4], {0, 1},
                                                4, {5, 2}, 0, packet));
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
            loom_aie2p_array_route_tree_t trees[8] = {};
            IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
                family, 128, &arena, &routing));
            for (uint32_t destination = 0; destination < family->column_count;
                 ++destination) {
              const size_t begin = routing.route_count;
              ASSERT_TRUE(loom_aie2p_array_route_workers(
                  &routing, destination, &trees[0], {column, memory->first_row},
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

TEST(Aie2pRouteTest, ShimDmaSharesPayloadAndQueryRoutesWithCoreCreditReturn) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  const auto* shim_facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  const auto* core_facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  for (uint16_t column = 0; column < family->column_count; ++column) {
    const loom_xdna_tile_coordinate_t shim = {column, 0};
    for (uint16_t destination = 0; destination < family->column_count;
         ++destination) {
      for (uint16_t row = 1; row < family->row_count; ++row) {
        const loom_xdna_tile_coordinate_t payload = {destination, row};
        const loom_xdna_tile_coordinate_t core = {
            destination, static_cast<uint16_t>(row < core_facts->first_row
                                                   ? core_facts->first_row
                                                   : row)};
        for (uint8_t engine = 0;
             engine < shim_facts->dma.channel_count_per_direction; ++engine) {
          SCOPED_TRACE(::testing::Message()
                       << column << ":" << unsigned(engine) << " -> "
                       << destination << "," << row);
          iree_arena_reset(&arena);
          loom_aie2p_array_route_builder_t routing;
          loom_aie2p_array_route_tree_t trees[8] = {};
          IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
              family, 3 * (family->column_count + family->row_count + 2),
              &arena, &routing));
          ASSERT_TRUE(loom_aie2p_array_route_stream(
              &routing, 0, &trees[0], shim, LOOM_XDNA_STREAM_PORT_DMA, engine,
              payload, LOOM_XDNA_STREAM_PORT_DMA, 0, true));
          const auto& source_mux = routing.routes[0];
          EXPECT_EQ(source_mux.switch_kind,
                    LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX);
          EXPECT_EQ(source_mux.source_port, LOOM_XDNA_STREAM_PORT_DMA);
          EXPECT_EQ(source_mux.source_channel, engine);
          EXPECT_EQ(source_mux.destination_port, LOOM_XDNA_STREAM_PORT_NORTH);
          EXPECT_EQ(source_mux.destination_channel,
                    shim_facts->dma.memory_to_stream_port_base +
                        engine * shim_facts->dma.memory_to_stream_port_stride);
          const size_t query_begin = routing.route_count;
          ASSERT_TRUE(loom_aie2p_array_route_stream(
              &routing, 1, &trees[0], shim, LOOM_XDNA_STREAM_PORT_DMA, engine,
              core, LOOM_XDNA_STREAM_PORT_CORE, 0, true));
          // A second packet class retains the full switch path but reuses the
          // physical mux and the same source lane.
          EXPECT_EQ(routing.routes[query_begin].source_port,
                    LOOM_XDNA_STREAM_PORT_SOUTH);
          EXPECT_EQ(routing.routes[query_begin].source_channel,
                    source_mux.destination_channel);
          EXPECT_EQ(routing.routes[routing.route_count - 1].destination_port,
                    LOOM_XDNA_STREAM_PORT_CORE);
          ASSERT_TRUE(loom_aie2p_array_route_stream(
              &routing, 2, &trees[2], core, LOOM_XDNA_STREAM_PORT_CORE, 0, shim,
              LOOM_XDNA_STREAM_PORT_DMA, engine, true));
          const auto& credit_mux = routing.routes[routing.route_count - 1];
          const auto& credit_switch = routing.routes[routing.route_count - 2];
          EXPECT_EQ(credit_mux.switch_kind,
                    LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX);
          EXPECT_EQ(credit_mux.source_port, LOOM_XDNA_STREAM_PORT_NORTH);
          EXPECT_EQ(credit_mux.destination_port, LOOM_XDNA_STREAM_PORT_DMA);
          EXPECT_EQ(credit_mux.destination_channel, engine);
          EXPECT_EQ(credit_switch.destination_port,
                    LOOM_XDNA_STREAM_PORT_SOUTH);
          EXPECT_EQ(credit_switch.destination_channel,
                    credit_mux.source_channel);
          EXPECT_EQ(credit_mux.source_channel,
                    shim_facts->dma.stream_to_memory_port_base +
                        engine * shim_facts->dma.stream_to_memory_port_stride);
          for (size_t i = 0; i < routing.route_count; ++i) {
            const auto& edge = routing.routes[i];
            if (edge.switch_kind == LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX) {
              continue;
            }
            const auto* tile =
                loom_xdna_array_tile_facts(family, edge.coordinate);
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
          }
        }
      }
    }
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, ReceiverGrantSharesOnlyItsPayloadSuffix) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t trees[8] = {};
  IREE_ASSERT_OK(
      loom_aie2p_array_route_builder_initialize(family, 64, &arena, &routing));
  // Four real compute DMA senders occupy every lane of the final two links.
  const loom_xdna_tile_coordinate_t sources[] = {
      {0, 3}, {0, 3}, {0, 4}, {0, 5}};
  const uint8_t engines[] = {0, 1, 0, 0};
  size_t carrier_count = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &routing, i, &trees[i], sources[i], engines[i], {0, 1}, i, false));
    if (!i) {
      carrier_count = routing.route_count;
    }
  }
  loom_aie2p_array_route_sharing_t sharing;
  IREE_ASSERT_OK(loom_aie2p_array_route_sharing_initialize(
      &routing, 0, carrier_count, {0, 1}, &arena, &sharing));
  const loom_xdna_tile_coordinate_t control_sources[] = {{0, 4}, {1, 5}};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(control_sources); ++i) {
    const uint32_t source = 4 + i;
    const size_t first = routing.route_count;
    ASSERT_TRUE(loom_aie2p_array_route_control(&routing, source, &trees[source],
                                               control_sources[i], {0, 1},
                                               &sharing));
    EXPECT_TRUE(sharing.used);
    // Cross-column control enters the carrier column on its compute row,
    // preserving the path before it joins the granted payload suffix.
    if (control_sources[i].column != sharing.destination.column) {
      EXPECT_EQ(routing.routes[first].coordinate.row, control_sources[i].row);
      EXPECT_EQ(routing.routes[first].destination_port,
                LOOM_XDNA_STREAM_PORT_WEST);
    }
    // Both controls converge on the carrier and retain exactly its suffix,
    // so one physical input/packet cannot fork and duplicate the control write.
    for (size_t i = first; i < routing.route_count; ++i) {
      const auto& edge = routing.routes[i];
      if (edge.coordinate.row == 3 || edge.coordinate.row == 2) {
        EXPECT_EQ(edge.destination_port, LOOM_XDNA_STREAM_PORT_SOUTH);
        EXPECT_EQ(edge.destination_channel, 0);
      }
      if (edge.coordinate.row == 2 || edge.coordinate.row == 1) {
        EXPECT_EQ(edge.source_port, LOOM_XDNA_STREAM_PORT_NORTH);
        EXPECT_EQ(edge.source_channel, 0);
      }
    }
    EXPECT_EQ(routing.routes[routing.route_count - 1].destination_port,
              LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
  }
  // An existing source prefix cannot extend the shared stream to another
  // destination outside the granted packet domain.
  EXPECT_FALSE(loom_aie2p_array_route_stream(
      &routing, 6, &trees[4], {0, 4}, LOOM_XDNA_STREAM_PORT_CORE, 0, {0, 2},
      LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0, true));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, AvailableControlLaneKeepsPayloadUnshared) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t trees[8] = {};
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      loom_xdna_npu2_array_family(), 16, &arena, &routing));
  ASSERT_TRUE(loom_aie2p_array_route_workers(&routing, 0, &trees[0], {0, 2}, 0,
                                             {0, 1}, 0, false));
  loom_aie2p_array_route_sharing_t sharing;
  IREE_ASSERT_OK(loom_aie2p_array_route_sharing_initialize(
      &routing, 0, routing.route_count, {0, 1}, &arena, &sharing));
  const size_t first = routing.route_count;
  ASSERT_TRUE(loom_aie2p_array_route_control(&routing, 1, &trees[1], {0, 2},
                                             {0, 1}, &sharing));
  EXPECT_FALSE(sharing.used);
  EXPECT_EQ(routing.routes[first].destination_channel, 1);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, GrantedControlContinuesBeyondPayloadReceiver) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  for (bool shim_first : {false, true}) {
    iree_arena_reset(&arena);
    loom_aie2p_array_route_builder_t routing;
    loom_aie2p_array_route_tree_t trees[8] = {};
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(family, 64, &arena,
                                                             &routing));
    size_t carrier_count = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      ASSERT_TRUE(loom_aie2p_array_route_workers(
          &routing, i, &trees[i], {0, static_cast<uint16_t>(i + 2)}, 0, {0, 1},
          i, false));
      if (!i) {
        carrier_count = routing.route_count;
      }
    }
    loom_aie2p_array_route_sharing_t sharing;
    IREE_ASSERT_OK(loom_aie2p_array_route_sharing_initialize(
        &routing, 0, carrier_count, {0, 1}, &arena, &sharing));
    for (uint32_t i = 0; i < 2; ++i) {
      const uint16_t row = (i == 0) == shim_first ? 0 : 1;
      const size_t first = routing.route_count;
      ASSERT_TRUE(loom_aie2p_array_route_control(&routing, 4 + i, &trees[4],
                                                 {0, 2}, {0, row}, &sharing));
      ASSERT_EQ(routing.route_count - first, 3u - row);
      const auto& shared = routing.routes[first];
      EXPECT_EQ(shared.coordinate.row, 2);
      EXPECT_EQ(shared.source_port, LOOM_XDNA_STREAM_PORT_CORE);
      EXPECT_EQ(shared.destination_port, LOOM_XDNA_STREAM_PORT_SOUTH);
      EXPECT_EQ(shared.destination_channel, 0);
      const auto& receiver = routing.routes[first + 1];
      EXPECT_EQ(receiver.coordinate.row, 1);
      EXPECT_EQ(receiver.source_port, LOOM_XDNA_STREAM_PORT_NORTH);
      EXPECT_EQ(receiver.source_channel, 0);
      EXPECT_EQ(receiver.destination_port,
                row ? LOOM_XDNA_STREAM_PORT_TILE_CONTROL
                    : LOOM_XDNA_STREAM_PORT_SOUTH);
      EXPECT_EQ(receiver.destination_channel, 0);
      const auto& terminal = routing.routes[routing.route_count - 1];
      EXPECT_EQ(terminal.coordinate.row, row);
      EXPECT_EQ(terminal.destination_port, LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
    }
    const size_t other_first = routing.route_count;
    // A second arithmetic core shares the same packet namespace and must
    // retain the already reserved continuation after the payload receiver.
    ASSERT_TRUE(loom_aie2p_array_route_control(&routing, 7, &trees[7], {0, 3},
                                               {0, 0}, &sharing));
    ASSERT_EQ(routing.route_count - other_first, 4u);
    for (size_t i = other_first + 1; i < other_first + 3; ++i) {
      EXPECT_EQ(routing.routes[i].destination_port,
                LOOM_XDNA_STREAM_PORT_SOUTH);
      EXPECT_EQ(routing.routes[i].destination_channel, 0);
    }
    const auto& other_terminal = routing.routes[routing.route_count - 1];
    EXPECT_EQ(other_terminal.coordinate.row, 0);
    EXPECT_EQ(other_terminal.source_channel, 0);
    EXPECT_EQ(other_terminal.destination_port,
              LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
    EXPECT_TRUE(sharing.used);
    EXPECT_EQ(routing.link_channels.southbound[family->column_count], 15u);
    EXPECT_EQ(routing.link_channels.southbound[0], 1u);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

TEST(Aie2pRouteTest, GrantedControlRequiresFreeContinuationBeyondReceiver) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&blocks, &arena);
  loom_aie2p_array_route_builder_t routing;
  loom_aie2p_array_route_tree_t trees[8] = {};
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      loom_xdna_npu2_array_family(), 64, &arena, &routing));
  size_t carrier_count = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &routing, i, &trees[i], {0, static_cast<uint16_t>(i + 2)}, 0, {0, 1}, i,
        false));
    if (!i) {
      carrier_count = routing.route_count;
    }
  }
  loom_aie2p_array_route_sharing_t sharing;
  IREE_ASSERT_OK(loom_aie2p_array_route_sharing_initialize(
      &routing, 0, carrier_count, {0, 1}, &arena, &sharing));
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &routing, 4, &trees[4], {0, 1}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0,
      {0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0, true));
  const size_t first = routing.route_count;
  // SRAM north/south passthrough preserves the lane. The carrier's lane zero
  // cannot reach the shim now; another free SRAM-to-shim lane is insufficient.
  EXPECT_FALSE(loom_aie2p_array_route_control(&routing, 5, &trees[5], {0, 2},
                                              {0, 0}, &sharing));
  EXPECT_EQ(routing.route_count, first);
  EXPECT_FALSE(sharing.used);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&blocks);
}

}  // namespace
