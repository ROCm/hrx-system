// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"

#include <array>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/target/arch/amd/xdna/array/registers.h"
#include "loom/testing/diagnostic_matchers.h"

namespace {

class Aie2pSwitchTest : public ::testing::TestWithParam<loom_xdna_tile_kind_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    pass_.arena = &arena_;
    pass_.diagnostic_emitter = diagnostics_.emitter();
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tiles_.resize(context_.family->column_count * context_.family->row_count);
    context_.tiles = tiles_.data();
    coordinate_ = {1,
                   loom_xdna_array_tile_kind_facts(context_.family, GetParam())
                       ->first_row};
    auto& tile = tiles_[coordinate_.column * context_.family->row_count +
                        coordinate_.row];
    tile.coordinate = coordinate_;
    tile.facts = loom_xdna_array_tile_facts(context_.family, coordinate_);
    // Each test uses at most one route per packet ID, with no inter-tile hops.
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        context_.family, 32, 32, &arena_, &context_.routing));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void AddRoute(loom_xdna_stream_port_t source, uint8_t source_channel,
                loom_xdna_stream_port_t destination,
                uint8_t destination_channel, uint8_t packet) {
    const auto index = static_cast<uint32_t>(routes_.size());
    ASSERT_TRUE(loom_aie2p_array_route_stream(
        &context_.routing, index, index, coordinate_, source, source_channel,
        coordinate_, destination, destination_channel));
    ASSERT_EQ(context_.routing.route_count, index + 1);
    routes_.push_back({&context_.routing.routes[index], packet});
  }

  void Build(bool expected_valid) {
    context_.routes = routes_.data();
    bool valid = false;
    IREE_ASSERT_OK(
        loom_aie2p_native_switches_build(&context_, nullptr, &valid));
    ASSERT_EQ(valid, expected_valid);
    if (expected_valid) {
      ASSERT_NE(context_.switches, nullptr);
      EXPECT_EQ(context_.switches->next, nullptr);
      EXPECT_TRUE(diagnostics_.emissions.empty());
    } else {
      ASSERT_EQ(diagnostics_.emissions.size(), 1u);
    }
  }

  uint16_t Ordinal(loom_xdna_stream_direction_t direction,
                   loom_xdna_stream_port_t port, uint8_t channel) {
    return loom_xdna_array_stream_port_range(context_.family, GetParam(),
                                             direction, port)
               ->ordinal +
           channel;
  }

  uint32_t Word(loom_aie2p_native_switch_bank_t bank, uint16_t ordinal) {
    return context_.switches->banks[bank].words[ordinal];
  }

  // Register fields have the same layout for all three tile roles.
  uint32_t Decode(uint32_t word, loom_xdna_register_field_id_t field) {
    loom_xdna_register_field_info_t info = {};
    IREE_CHECK_OK(loom_xdna_register_field_info(field, &info));
    return (word >> info.least_significant_bit) &
           (UINT32_MAX >> (32 - info.bit_width));
  }

  // Backing storage for this compiler pass's retained plans.
  iree_arena_block_pool_t pool_ = {};
  // Allocation lifetime shared by routing and switch admission.
  iree_arena_allocator_t arena_ = {};
  // Real diagnostic sink for target-resource rejection.
  loom::testing::DiagnosticEmissionCapture diagnostics_;
  // Pass services consumed by native switch admission.
  loom_pass_t pass_ = {};
  // Physical route and retained switch owners.
  loom_aie2p_native_context_t context_ = {};
  // Column-major tile inventory, with the selected switch initialized.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Packet identities attached to the physical route builder's edges.
  std::vector<loom_aie2p_native_route_t> routes_;
  // Interior column on the selected tile role's first row.
  loom_xdna_tile_coordinate_t coordinate_ = {};
};

TEST_P(Aie2pSwitchTest, AllDestinationsUsePhysicalArbitersAndReuseSelections) {
  std::array<uint8_t, 32> source_uses = {};
  const auto* family = context_.family;
  for (uint8_t i = 0; i < family->stream_port_count; ++i) {
    const auto& destination = family->stream_ports[i];
    if (destination.tile_kind != GetParam() ||
        destination.direction != LOOM_XDNA_STREAM_DIRECTION_MASTER) {
      continue;
    }
    for (uint8_t channel = 0; channel < destination.count; ++channel) {
      // Choose an actual legal source with room for another packet filter.
      const loom_xdna_stream_port_range_t* source = nullptr;
      uint8_t source_channel = 0;
      uint8_t least_uses = 4;
      for (uint8_t j = 0; j < family->stream_port_count; ++j) {
        const auto& candidate = family->stream_ports[j];
        if (candidate.tile_kind != GetParam() ||
            candidate.direction != LOOM_XDNA_STREAM_DIRECTION_SLAVE) {
          continue;
        }
        for (uint8_t k = 0; k < candidate.count; ++k) {
          if (source_uses[candidate.ordinal + k] < least_uses &&
              (loom_xdna_array_stream_destination_channels(family, &candidate,
                                                           k, &destination) &
               (uint32_t{1} << channel))) {
            source = &candidate;
            source_channel = k;
            least_uses = source_uses[candidate.ordinal + k];
          }
        }
      }
      ASSERT_NE(source, nullptr);
      AddRoute(source->port, source_channel, destination.port, channel,
               static_cast<uint8_t>(routes_.size()));
      ++source_uses[source->ordinal + source_channel];
    }
  }
  const size_t destination_count = routes_.size();
  const size_t expected_counts[] = {22, 17, 23};
  ASSERT_EQ(destination_count, expected_counts[GetParam() - 1]);

  // Another packet ID targeting an existing destination keeps its selection.
  const auto first = *routes_.front().edge;
  ASSERT_LT(source_uses[Ordinal(LOOM_XDNA_STREAM_DIRECTION_SLAVE,
                                first.source_port, first.source_channel)],
            4u);
  AddRoute(first.source_port, first.source_channel, first.destination_port,
           first.destination_channel, 31);
  ASSERT_NO_FATAL_FAILURE(Build(true));

  // Six physical arbiters, each with four selections. The seventh, thirteenth,
  // and nineteenth destinations reuse arbiter zero instead of encoding 6/7.
  const std::array<uint8_t, 24> expected_arbiters = {
      0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  const std::array<uint8_t, 24> expected_selections = {
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3};
  source_uses.fill(0);
  for (size_t i = 0; i < routes_.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& route = routes_[i];
    const auto& edge = *route.edge;
    const size_t selection = i == destination_count ? 0 : i;
    const auto arbiter = expected_arbiters[selection];
    const auto master_select = expected_selections[selection];
    const auto master =
        Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER, edge.destination_port,
                edge.destination_channel);
    const auto slave = Ordinal(LOOM_XDNA_STREAM_DIRECTION_SLAVE,
                               edge.source_port, edge.source_channel);
    const auto master_word =
        Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, master);
    EXPECT_EQ(
        Decode(
            master_word,
            LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_MASTER_CONFIG_CONFIGURATION),
        arbiter | (uint32_t{1} << (master_select + 3)));
    const auto filter_word = Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS,
                                  slave * 4 + source_uses[slave]++);
    EXPECT_EQ(Decode(filter_word,
                     LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_ARBITER),
              arbiter);
    EXPECT_EQ(
        Decode(filter_word,
               LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_MASTER_SELECT),
        master_select);
    EXPECT_EQ(Decode(filter_word,
                     LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_PACKET_ID),
              route.packet);
    EXPECT_EQ(Decode(filter_word,
                     LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_ENABLE),
              1u);
  }
}

TEST_P(Aie2pSwitchTest, FifthPacketFilterRejectsTheRoutePlan) {
  for (uint8_t packet = 0; packet < 5; ++packet) {
    AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0,
             packet);
  }
  ASSERT_NO_FATAL_FAILURE(Build(false));
  EXPECT_EQ(diagnostics_.emissions.front().string_params,
            std::vector<std::string>{
                "available packet filters and unicast master selections"});
}

INSTANTIATE_TEST_SUITE_P(TileKinds, Aie2pSwitchTest,
                         ::testing::Values(LOOM_XDNA_TILE_KIND_SHIM_NOC,
                                           LOOM_XDNA_TILE_KIND_MEMORY,
                                           LOOM_XDNA_TILE_KIND_COMPUTE));

}  // namespace
