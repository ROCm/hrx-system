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
    // Include all 32 packet IDs and their multicast branches.
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        context_.family, 128, 128, &arena_, &context_.routing));
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
        coordinate_, destination, destination_channel, packet != UINT8_MAX));
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

  uint32_t PacketDestinations(loom_xdna_stream_port_t source,
                              uint8_t source_channel, uint8_t packet,
                              uint32_t expected_matches = 1) {
    const uint16_t slave =
        Ordinal(LOOM_XDNA_STREAM_DIRECTION_SLAVE, source, source_channel);
    uint32_t destinations = 0;
    uint32_t matches = 0;
    for (uint16_t slot = 0; slot < 4; ++slot) {
      const uint32_t filter =
          Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS, slave * 4 + slot);
      const uint32_t mask = (filter >> 16) & 31;
      if (!(filter & (1u << 8)) || ((filter >> 24) & mask) != (packet & mask)) {
        continue;
      }
      ++matches;
      const uint32_t arbiter = filter & 7;
      const uint32_t selection = (filter >> 4) & 3;
      const auto& masters =
          context_.switches->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS];
      for (uint32_t master = 0; master < masters.word_count; ++master) {
        const uint32_t word = masters.words[master];
        if ((word & (1u << 30)) && (word & 7) == arbiter &&
            (word & (1u << (selection + 3)))) {
          destinations |= 1u << master;
        }
      }
    }
    EXPECT_EQ(matches, expected_matches);
    return destinations;
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

TEST_P(Aie2pSwitchTest, OneInputSharesAnArbiterAcrossDestinationSets) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  for (uint8_t packet = 0; packet < 4; ++packet) {
    AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, packet, packet);
  }
  // A second identity for the first set compresses into the same input filter.
  AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0, 16);
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  const uint32_t arbiter =
      Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, north) & 7;
  EXPECT_LT(arbiter, 6u);
  for (uint8_t channel = 0; channel < 4; ++channel) {
    EXPECT_EQ(Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, north + channel) & 7,
              arbiter);
  }
  for (uint8_t packet = 0; packet < 32; ++packet) {
    SCOPED_TRACE(packet);
    const bool admitted = packet < 4 || packet == 16;
    EXPECT_EQ(PacketDestinations(source, 0, packet, admitted ? 1 : 0),
              admitted ? 1u << (north + (packet & 3)) : 0);
  }
}

TEST_P(Aie2pSwitchTest, IndependentInputsKeepIndependentArbiters) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  for (uint8_t channel = 0; channel < 6; ++channel) {
    AddRoute(source, channel, LOOM_XDNA_STREAM_PORT_NORTH, channel, 0);
  }
  // A seventh destination must share with its own input, not with the first
  // arbiter that happens to have an unused master selection.
  AddRoute(source, 1, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0, 1);
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  uint32_t arbiters = 0;
  for (uint8_t channel = 0; channel < 6; ++channel) {
    const uint32_t arbiter =
        Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, north + channel) & 7;
    ASSERT_LT(arbiter, 6u);
    EXPECT_EQ(arbiters & (1u << arbiter), 0u);
    arbiters |= 1u << arbiter;
    EXPECT_EQ(PacketDestinations(source, channel, 0), 1u << (north + channel));
  }
  const uint16_t control = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                   LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  EXPECT_EQ(Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, control) & 7,
            Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, north + 1) & 7);
  EXPECT_EQ(PacketDestinations(source, 1, 1), 1u << control);
}

TEST_P(Aie2pSwitchTest, MultipleInputComponentsCannotShareArbiters) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  for (uint8_t input = 0; input < 2; ++input) {
    AddRoute(source, input, LOOM_XDNA_STREAM_PORT_NORTH, 0, 0);
    AddRoute(source, input, LOOM_XDNA_STREAM_PORT_NORTH, 1, 1);
  }
  AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, 2, 2);
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  uint32_t arbiters = 0;
  for (uint8_t channel = 0; channel < 3; ++channel) {
    const uint32_t arbiter =
        Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, north + channel) & 7;
    ASSERT_LT(arbiter, 6u);
    EXPECT_EQ(arbiters & (1u << arbiter), 0u);
    arbiters |= 1u << arbiter;
    EXPECT_EQ(PacketDestinations(source, 0, channel), 1u << (north + channel));
    if (channel < 2) {
      EXPECT_EQ(PacketDestinations(source, 1, channel),
                1u << (north + channel));
    }
  }
}

TEST_P(Aie2pSwitchTest, SevenIndependentInputsRejectTheRoutePlan) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  for (uint8_t channel = 0; channel < 6; ++channel) {
    AddRoute(source, channel, LOOM_XDNA_STREAM_PORT_NORTH, channel, 0);
  }
  AddRoute(LOOM_XDNA_STREAM_PORT_NORTH, 0, LOOM_XDNA_STREAM_PORT_TILE_CONTROL,
           0, 0);
  ASSERT_NO_FATAL_FAILURE(Build(false));
  EXPECT_EQ(diagnostics_.emissions.front().string_params,
            std::vector<std::string>{
                "available packet filters and independent multicast arbiters"});
}

TEST_P(Aie2pSwitchTest,
       SharedDestinationCompressesPacketIdsWithoutOvermatching) {
  for (uint8_t packet = 0; packet < 5; ++packet) {
    AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0,
             packet);
  }
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  for (uint8_t packet = 0; packet < 32; ++packet) {
    SCOPED_TRACE(packet);
    const bool admitted = packet < 5;
    EXPECT_EQ(PacketDestinations(LOOM_XDNA_STREAM_PORT_SOUTH, 0, packet,
                                 admitted ? 1 : 0),
              admitted ? 1u << north : 0);
  }
}

TEST_P(Aie2pSwitchTest, FullPacketNamespaceUsesOneFilter) {
  for (uint8_t packet = 0; packet < 32; ++packet) {
    AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0,
             packet);
  }
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  const uint16_t south =
      Ordinal(LOOM_XDNA_STREAM_DIRECTION_SLAVE, LOOM_XDNA_STREAM_PORT_SOUTH, 0);
  EXPECT_EQ(Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS, south * 4), 1u << 8);
  for (uint8_t packet = 0; packet < 32; ++packet) {
    SCOPED_TRACE(packet);
    EXPECT_EQ(PacketDestinations(LOOM_XDNA_STREAM_PORT_SOUTH, 0, packet),
              1u << north);
  }
}

TEST_P(Aie2pSwitchTest, MaskedFiltersPreserveCompleteMulticastSets) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  std::array<uint32_t, 32> destinations = {};
  destinations[0] = destinations[1] = 3;
  destinations[2] = destinations[3] = 4;
  destinations[16] = 7;
  // Interleave destination branches: compression must wait until every
  // packet ID's entire multicast set is known.
  for (uint8_t channel = 0; channel < 3; ++channel) {
    for (uint8_t packet = 0; packet < 32; ++packet) {
      if (destinations[packet] & (1u << channel)) {
        AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, channel, packet);
      }
    }
  }
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  for (uint8_t packet = 0; packet < 32; ++packet) {
    SCOPED_TRACE(packet);
    EXPECT_EQ(
        PacketDestinations(source, 0, packet, destinations[packet] ? 1 : 0),
        destinations[packet] << north);
  }
}

TEST_P(Aie2pSwitchTest, NoncontiguousMaskBitsKeepInputsIndependent) {
  for (uint8_t packet = 0; packet < 32; ++packet) {
    if ((packet & 5) == 1) {
      AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0,
               packet);
    }
  }
  AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 1, LOOM_XDNA_STREAM_PORT_NORTH, 1, 31);
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  for (uint8_t packet = 0; packet < 32; ++packet) {
    SCOPED_TRACE(packet);
    const bool admitted = (packet & 5) == 1;
    EXPECT_EQ(PacketDestinations(LOOM_XDNA_STREAM_PORT_SOUTH, 0, packet,
                                 admitted ? 1 : 0),
              admitted ? 1u << north : 0);
    EXPECT_EQ(PacketDestinations(LOOM_XDNA_STREAM_PORT_SOUTH, 1, packet,
                                 packet == 31 ? 1 : 0),
              packet == 31 ? 1u << (north + 1) : 0);
  }
}

TEST_P(Aie2pSwitchTest, FiveIrreducibleFiltersRejectTheRoutePlan) {
  // No pair differs in only one ID bit, so every ID needs its own filter.
  for (uint8_t packet : {0, 3, 5, 6, 9}) {
    AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0,
             packet);
  }
  ASSERT_NO_FATAL_FAILURE(Build(false));
  EXPECT_EQ(diagnostics_.emissions.front().string_params,
            std::vector<std::string>{
                "available packet filters and independent multicast arbiters"});
}

TEST_P(Aie2pSwitchTest, MinimumCoverCanReconsiderAnEarlierMask) {
  // Greedily covering IDs 0/1 first leaves four singletons. Covering 0/8 and
  // 1/3 instead leaves only IDs 5/6, fitting the four physical filters.
  constexpr uint32_t packets = 0x16b;
  for (uint8_t packet = 0; packet < 32; ++packet) {
    if (packets & (1u << packet)) {
      AddRoute(LOOM_XDNA_STREAM_PORT_SOUTH, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0,
               packet);
    }
  }
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  for (uint8_t packet = 0; packet < 32; ++packet) {
    SCOPED_TRACE(packet);
    const bool admitted = (packets & (1u << packet)) != 0;
    EXPECT_EQ(PacketDestinations(LOOM_XDNA_STREAM_PORT_SOUTH, 0, packet,
                                 admitted ? 1 : 0),
              admitted ? 1u << north : 0);
  }
}

TEST_P(Aie2pSwitchTest, OverlappingMulticastsKeepCompleteDestinationSets) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  const std::array<uint32_t, 3> masks = {3, 4, 7};
  for (uint8_t packet = 0; packet < masks.size(); ++packet) {
    for (uint8_t channel = 0; channel < 3; ++channel) {
      if (masks[packet] & (1u << channel)) {
        AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, channel, packet);
      }
    }
  }
  // A shared physical branch does not allocate a second packet filter.
  AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0, 0);
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  for (uint8_t packet = 0; packet < masks.size(); ++packet) {
    EXPECT_EQ(PacketDestinations(source, 0, packet), masks[packet] << north);
  }
}

TEST_P(Aie2pSwitchTest, OnlyDmaPayloadEndpointsRemovePacketHeaders) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  const bool payload = GetParam() != LOOM_XDNA_TILE_KIND_SHIM_NOC;
  const auto endpoint =
      payload ? LOOM_XDNA_STREAM_PORT_DMA : LOOM_XDNA_STREAM_PORT_TILE_CONTROL;
  AddRoute(source, 0, endpoint, 0, 1);
  AddRoute(source, 0, LOOM_XDNA_STREAM_PORT_NORTH, 0, 1);
  ASSERT_NO_FATAL_FAILURE(Build(true));
  const uint16_t dma = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER, endpoint, 0);
  const uint16_t north = Ordinal(LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                 LOOM_XDNA_STREAM_PORT_NORTH, 0);
  EXPECT_EQ(PacketDestinations(source, 0, 1), (1u << dma) | (1u << north));
  EXPECT_EQ(Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, dma) & (1u << 7),
            payload ? 1u << 7 : 0u);
  EXPECT_EQ(Word(LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS, north) & (1u << 7), 0u);
}

TEST_P(Aie2pSwitchTest, ConnectedDestinationSetsMustFitOneArbiter) {
  const auto source = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                          ? LOOM_XDNA_STREAM_PORT_DMA
                          : LOOM_XDNA_STREAM_PORT_SOUTH;
  const std::array<uint32_t, 5> masks = {1, 3, 5, 7, 9};
  for (uint8_t packet = 0; packet < masks.size(); ++packet) {
    for (uint8_t channel = 0; channel < 4; ++channel) {
      if (masks[packet] & (1u << channel)) {
        AddRoute(source, packet / 4, LOOM_XDNA_STREAM_PORT_NORTH, channel,
                 packet);
      }
    }
  }
  ASSERT_NO_FATAL_FAILURE(Build(false));
  EXPECT_EQ(diagnostics_.emissions.front().string_params,
            std::vector<std::string>{
                "available packet filters and independent multicast arbiters"});
}

INSTANTIATE_TEST_SUITE_P(TileKinds, Aie2pSwitchTest,
                         ::testing::Values(LOOM_XDNA_TILE_KIND_SHIM_NOC,
                                           LOOM_XDNA_TILE_KIND_MEMORY,
                                           LOOM_XDNA_TILE_KIND_COMPUTE));

}  // namespace
