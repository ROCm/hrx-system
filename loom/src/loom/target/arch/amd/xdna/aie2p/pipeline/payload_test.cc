// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

#include <set>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_queues.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/testing/diagnostic_matchers.h"

namespace {

class PayloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tiles_.resize(context_.family->column_count * context_.family->row_count);
    for (size_t i = 0; i < tiles_.size(); ++i) {
      auto& tile = tiles_[i];
      tile.coordinate = {static_cast<uint16_t>(i / context_.family->row_count),
                         static_cast<uint16_t>(i % context_.family->row_count)};
      tile.facts = loom_xdna_array_tile_facts(context_.family, tile.coordinate);
      tile.memory_layout = tile.coordinate.row == 0 ? nullptr
                           : tile.coordinate.row == 1
                               ? &loom_aie2p_sram_memory
                               : &loom_aie2p_compute_memory;
    }
    context_.tiles = tiles_.data();
    context_.workers = workers_;
    realization_.invocations.worker_count = IREE_ARRAYSIZE(workers_);
    realization_.invocations.channel_count = 2;
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  void Form() {
    bool valid = false;
    IREE_ASSERT_OK(
        loom_aie2p_native_form_payloads(&context_, &realization_, &valid));
    ASSERT_TRUE(valid);
  }
  void Ordinary(size_t worker, size_t local_row, bool ingress) {
    paths_[worker].local = &tiles_[local_row];
    paths_[worker].shim = &tiles_[0];
    paths_[worker].ingress = ingress;
    workers_[worker].transfer_paths = &paths_[worker];
  }

  // Arena owner for retained physical destination sets.
  iree_arena_block_pool_t blocks_;
  // One planning invocation's temporary storage.
  iree_arena_allocator_t arena_;
  // Compiler allocation boundary.
  loom_pass_t pass_ = {};
  // Complete tile and worker inventory.
  loom_aie2p_native_context_t context_ = {};
  // Hardware coordinates indexed by column then row.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Physical route labels retained alongside the native context.
  std::vector<loom_aie2p_native_route_t> routes_;
  // Already assigned physical paths in disjoint construction lifetimes.
  loom_aie2p_native_dma_path_t paths_[3] = {};
  // Native endpoint families consumed by payload formation.
  loom_aie2p_native_worker_t workers_[3] = {};
  // Common retained counts; formation never traverses source IR.
  loom_pipeline_realization_t realization_ = {};
};

TEST_F(PayloadTest, RepeatedPhysicalEndpointsKeepOneUnframedClass) {
  Ordinary(0, 2, true);
  Ordinary(1, 2, true);
  Form();
  EXPECT_EQ(paths_[0].payload, paths_[1].payload);
  EXPECT_EQ(paths_[0].payload->source->class_count, 1);
  EXPECT_FALSE(paths_[0].payload->source->framed);
  uint32_t words[8] = {17, 23, 5};
  loom_aie2p_native_frame_payload(paths_[0].payload, words);
  EXPECT_EQ(words[0], 17u);
  EXPECT_EQ(words[1], 23u);
  EXPECT_EQ(words[2], 5u);
}

TEST_F(PayloadTest, RepeatedPacketClassRetainsEachConstructionUse) {
  constexpr auto none = IREE_HOST_SIZE_MAX;
  const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 2, none, 4},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 2, none, 5},
      {LOOM_PIPELINE_LIFETIME_OWNER, 3, 2, none, none},
  };
  const iree_host_size_t owners[] = {1, 3, 4, 5};
  loom_pipeline_invocation_t invocations[3] = {};
  loom_pipeline_invocation_worker_t workers[3] = {};
  for (size_t i = 0; i < 3; ++i) {
    invocations[i].ordinal = i + 1;
    workers[i].invocation = &invocations[i];
    Ordinary(i, i == 1 ? 3 : 2, true);
  }
  realization_.invocations.workers = workers;
  realization_.invocations.lifetimes = {
      .nodes = nodes,
      .count = IREE_ARRAYSIZE(nodes),
      .owners = owners,
      .has_serial = true,
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(
      &arena_, &realization_.invocations.lifetimes));
  Form();
  ASSERT_EQ(paths_[0].payload, paths_[2].payload);
  ASSERT_NE(paths_[0].payload, paths_[1].payload);
  // The repeated class is used before and after the other class. Its set
  // must retain both uses, not just the first owner or their SERIAL ancestor.
  for (size_t i = 0; i < 2; ++i) {
    std::set<size_t> uses;
    for (size_t use = paths_[i].payload->lifetime_head; use;
         use = context_.route_lifetimes[use].next) {
      uses.insert(context_.route_lifetimes[use].lifetime);
    }
    EXPECT_EQ(uses, i ? std::set<size_t>({4}) : std::set<size_t>({3, 5}));
  }
}

TEST_F(PayloadTest, ShimPhasesShareOnePacketNamespace) {
  Ordinary(0, 2, true);
  Ordinary(1, 3, true);
  Form();
  EXPECT_EQ(paths_[0].payload->source, paths_[1].payload->source);
  EXPECT_EQ(paths_[0].payload->packet, 0);
  EXPECT_EQ(paths_[1].payload->packet, 1);
  EXPECT_TRUE(paths_[0].payload->source->framed);
  uint32_t words[8] = {17, 23, 5};
  loom_aie2p_native_frame_payload(paths_[1].payload, words);
  EXPECT_EQ(words[0], 17u);
  EXPECT_EQ(words[1], 23u);
  EXPECT_EQ(
      words[2],
      5u |
          loom_xdna_register_field_encode(
              LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_ENABLE_PACKET, 1) |
          loom_xdna_register_field_encode(
              LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_PACKET_ID, 1));
}

TEST_F(PayloadTest, ReceiverFanInFramesEveryPhysicalSender) {
  Ordinary(0, 1, false);
  Ordinary(1, 2, false);
  Form();
  EXPECT_NE(paths_[0].payload->source, paths_[1].payload->source);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(paths_[i].payload->source->framed);
    const auto* layout = paths_[i].local->memory_layout;
    uint32_t words[8] = {};
    loom_aie2p_native_frame_payload(paths_[i].payload, words);
    EXPECT_EQ(words[layout->packet.word],
              loom_xdna_register_field_encode(layout->packet.enable, 1));
  }
}

TEST_F(PayloadTest, SourceOwnsPrefixesIndependentlyOfRouteLabels) {
  Ordinary(0, 2, true);
  Ordinary(1, 3, true);
  Form();
  // Two destinations share one physical sender. Large destination labels
  // cannot select a prefix root or consume additional root-table capacity.
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      context_.family, 16, &arena_, &context_.routing));
  routes_.resize(16);
  context_.routes = routes_.data();
  uint32_t next_source = 1000;
  iree_host_size_t root = 0;
  for (size_t i = 0; i < 2; ++i) {
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_route_payload(
        &context_, paths_[i].payload, nullptr, &next_source, &valid));
    ASSERT_TRUE(valid);
    EXPECT_EQ(next_source, 1002);
    EXPECT_NE(paths_[i].payload->source->tree.root_index, 0u);
    if (i == 0) {
      root = paths_[i].payload->source->tree.root_index;
    }
    EXPECT_EQ(paths_[i].payload->source->tree.root_index, root);
  }
  for (size_t i = 0; i < 2; ++i) {
    const auto* payload = paths_[i].payload;
    const auto& terminal =
        routes_[payload->route_first + payload->route_count - 1];
    EXPECT_EQ(terminal.edge->coordinate.row, 2 + i);
    EXPECT_TRUE(terminal.edge->dma_destination);
    EXPECT_EQ(terminal.packet, i);
  }
}

TEST_F(PayloadTest, PhasedInputAndOrdinaryIngressUseTheSamePhysicalClass) {
  Ordinary(0, 2, true);
  loom_aie2p_native_tile_t* pools[] = {&tiles_[2]};
  context_.pool_tiles = pools;
  loom_aie2p_native_channel_t destination = {};
  loom_pipeline_transport_input_t source = {.phase_count = 1};
  loom_aie2p_native_input_phase_t phase = {.destination = &destination};
  loom_aie2p_native_input_path_t path = {.shim = &tiles_[0]};
  loom_aie2p_native_input_t input = {
      .source = &source,
      .phases = &phase,
      .path = &path,
  };
  workers_[1].input = &input;
  Form();
  EXPECT_EQ(paths_[0].payload, phase.payload);
  EXPECT_TRUE(phase.payload->source->framed);
  EXPECT_EQ(phase.payload->source->class_count, 1);
  EXPECT_FALSE(phase.payload->distribution_only);
}

TEST_F(PayloadTest, ColumnIngressFitsBothSRAMQueuesAndAllComputeRows) {
  // One shim queue serves both SRAM queues and all four compute rows. In
  // discovery order the northbound IDs would be 1..4, requiring three
  // filters in addition to the two SRAM destinations on the same input.
  const loom_aie2p_native_connection_endpoint_t destinations[] = {
      {{0, 1}, LOOM_XDNA_STREAM_PORT_DMA, 0},
      {{0, 2}, LOOM_XDNA_STREAM_PORT_DMA, 0},
      {{0, 3}, LOOM_XDNA_STREAM_PORT_DMA, 0},
      {{0, 4}, LOOM_XDNA_STREAM_PORT_DMA, 0},
      {{0, 5}, LOOM_XDNA_STREAM_PORT_DMA, 0},
      {{0, 1}, LOOM_XDNA_STREAM_PORT_DMA, 4},
  };
  loom_aie2p_native_dma_path_t ingress[IREE_ARRAYSIZE(destinations)] = {};
  for (size_t i = IREE_ARRAYSIZE(ingress); i > 0; --i) {
    auto& path = ingress[i - 1];
    path.shim = &tiles_[0];
    path.local = &tiles_[destinations[i - 1].coordinate.row];
    path.local_engine = destinations[i - 1].channel;
    path.ingress = true;
    path.next_demand = workers_[0].transfer_paths;
    workers_[0].transfer_paths = &path;
  }
  Form();
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      context_.family, 64, &arena_, &context_.routing));
  routes_.resize(64);
  context_.routes = routes_.data();
  uint32_t next_source = 0;
  bool valid = false;
  IREE_ASSERT_OK(loom_aie2p_native_route_payload(
      &context_, ingress[0].payload, nullptr, &next_source, &valid));
  ASSERT_TRUE(valid);
  const loom_aie2p_native_switch_input_t input = {
      .routing = &context_.routing,
      .routes = context_.routes,
  };
  iree_string_view_t requirement = iree_string_view_empty();
  IREE_ASSERT_OK(loom_aie2p_native_switches_build(
      &input, &arena_, &context_.switches, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  loom_aie2p_native_connectivity_t* connectivity = nullptr;
  IREE_ASSERT_OK(loom_aie2p_native_connectivity_allocate(
      context_.family, context_.switches, context_.routing.routes,
      context_.routing.route_count, &arena_, &connectivity));
  const loom_aie2p_native_connection_endpoint_t shim = {
      {0, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  for (size_t i = 0; i < IREE_ARRAYSIZE(ingress); ++i) {
    EXPECT_EQ(loom_aie2p_native_connectivity_check(connectivity, shim,
                                                   ingress[i].payload->packet,
                                                   &destinations[i], 1, false)
                  .failure,
              LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  }
  for (uint8_t packet = IREE_ARRAYSIZE(ingress); packet < 32; ++packet) {
    EXPECT_EQ(loom_aie2p_native_connectivity_check(connectivity, shim, packet,
                                                   nullptr, 0, false)
                  .failure,
              LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END);
  }
}

TEST_F(PayloadTest, LaterPassthroughConstrainsAnEarlierShortIngress) {
  loom::testing::DiagnosticEmissionCapture diagnostics;
  pass_.diagnostic_emitter = diagnostics.emitter();
  Ordinary(0, 1, true);
  Ordinary(1, 3, true);
  Form();
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      context_.family, 32, &arena_, &context_.routing));
  routes_.resize(32);
  context_.routes = routes_.data();
  // The SRAM sender occupies north lane zero. A shim packet terminating in
  // SRAM may arrive on south lane zero, but that lane cannot subsequently
  // pass through: the SRAM switch pairs south/north passthrough lanes.
  loom_aie2p_array_route_tree_t sram = {};
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &context_.routing, 0, &sram, {0, 1}, LOOM_XDNA_STREAM_PORT_DMA, 0, {0, 3},
      LOOM_XDNA_STREAM_PORT_DMA, 1, true));
  loom_aie2p_native_route_label(&context_, 0, 7, false, UINT32_MAX);
  uint32_t next_source = 1;
  for (size_t i = 0; i < 2; ++i) {
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_route_payload(
        &context_, paths_[i].payload, nullptr, &next_source, &valid));
    ASSERT_TRUE(valid);
  }
  const loom_aie2p_native_switch_input_t input = {
      .routing = &context_.routing,
      .routes = context_.routes,
  };
  iree_string_view_t requirement = iree_string_view_empty();
  IREE_ASSERT_OK(loom_aie2p_native_switches_build(
      &input, &arena_, &context_.switches, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  loom_aie2p_native_connectivity_t* connectivity = nullptr;
  IREE_ASSERT_OK(loom_aie2p_native_connectivity_allocate(
      context_.family, context_.switches, context_.routing.routes,
      context_.routing.route_count, &arena_, &connectivity));
  const loom_aie2p_native_connection_endpoint_t shim = {
      {0, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  for (size_t i = 0; i < 2; ++i) {
    const loom_aie2p_native_connection_endpoint_t destination = {
        paths_[i].local->coordinate, LOOM_XDNA_STREAM_PORT_DMA, 0};
    EXPECT_EQ(loom_aie2p_native_connectivity_check(connectivity, shim,
                                                   paths_[i].payload->packet,
                                                   &destination, 1, false)
                  .failure,
              LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  }
  const loom_aie2p_native_connection_endpoint_t sender = {
      {0, 1}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const loom_aie2p_native_connection_endpoint_t consumer = {
      {0, 3}, LOOM_XDNA_STREAM_PORT_DMA, 1};
  EXPECT_EQ(loom_aie2p_native_connectivity_check(connectivity, sender, 7,
                                                 &consumer, 1, false)
                .failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  EXPECT_TRUE(diagnostics.emissions.empty());
}

TEST_F(PayloadTest, MulticastClassesIgnoreDestinationEnumerationOrder) {
  loom_aie2p_native_distribution_t distributions[3] = {};
  loom_aie2p_native_distribution_path_t paths[3] = {};
  loom_aie2p_native_distribution_plan_t plans[3] = {};
  loom_aie2p_native_distribution_endpoint_t receivers[3][2] = {};
  for (size_t i = 0; i < 3; ++i) {
    distributions[i].sender.tile = &tiles_[1];
    receivers[i][0].tile = &tiles_[i == 1 ? 3 : 2];
    receivers[i][1].tile = &tiles_[i == 1 ? 2 : 3];
    distributions[i].receivers = receivers[i];
    distributions[i].receiver_count = i == 2 ? 1 : 2;
    distributions[i].path = &paths[i];
    paths[i].first = &distributions[i];
    plans[i].geometries = &distributions[i];
    workers_[i].distribution_plan = &plans[i];
  }
  Form();
  EXPECT_EQ(paths[0].payloads[0], paths[1].payloads[0]);
  EXPECT_NE(paths[0].payloads[0], paths[2].payloads[0]);
  EXPECT_EQ(paths[0].payloads[0]->source, paths[2].payloads[0]->source);
  EXPECT_EQ(paths[0].payloads[0]->source->class_count, 2);
  EXPECT_TRUE(paths[0].payloads[0]->source->framed);
  EXPECT_TRUE(paths[0].payloads[0]->distribution_only);
  EXPECT_EQ(paths[0].payloads[0]->destinations[0].tile, &tiles_[3]);
}

TEST_F(PayloadTest, SRAMDistributionLeavesPassthroughForShimCommands) {
  // A single SRAM queue sends one packet class up to a worker and another
  // down to the shim. Queue and packet identities span both directions.
  Ordinary(0, 1, false);
  loom_aie2p_native_distribution_path_t path = {};
  loom_aie2p_native_distribution_endpoint_t receiver = {.tile = &tiles_[2]};
  loom_aie2p_native_distribution_t distribution = {};
  distribution.sender.tile = &tiles_[1];
  distribution.receivers = &receiver;
  distribution.receiver_count = 1;
  distribution.path = &path;
  path.first = &distribution;
  loom_aie2p_native_distribution_plan_t plan = {.geometries = &distribution};
  workers_[1].distribution_plan = &plan;
  Form();
  const auto* egress = paths_[0].payload;
  ASSERT_EQ(egress->source, path.payloads[0]->source);
  EXPECT_NE(egress->packet, path.payloads[0]->packet);

  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      context_.family, 64, &arena_, &context_.routing));
  routes_.resize(64);
  context_.routes = routes_.data();
  uint32_t next_source = 0;
  bool valid = false;
  IREE_ASSERT_OK(loom_aie2p_native_route_payload(
      &context_, path.payloads[0], nullptr, &next_source, &valid));
  ASSERT_TRUE(valid);

  // All four compute rows return data to SRAM. The first row grants its
  // return lane to a command that continues through SRAM to the shim.
  loom_aie2p_array_route_tree_t returns[4] = {};
  const size_t first = context_.routing.route_count;
  for (uint8_t i = 0; i < 4; ++i) {
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &context_.routing, next_source++, &returns[i],
        {0, static_cast<uint16_t>(2 + i)}, 0, {0, 1}, i, i == 0));
  }
  loom_aie2p_array_route_sharing_t sharing = {};
  IREE_ASSERT_OK(loom_aie2p_array_route_sharing_initialize(
      &context_.routing, first, 2, {0, 1},
      LOOM_AIE2P_ARRAY_ROUTE_CARRIER_GRANTED_PAYLOAD, &sharing));
  loom_aie2p_array_route_tree_t command = {};
  ASSERT_TRUE(loom_aie2p_array_route_control(
      &context_.routing, next_source++, &command, {0, 2},
      LOOM_XDNA_STREAM_PORT_CORE, 0, {0, 0}, &sharing));
  ASSERT_TRUE(sharing.used);
  IREE_ASSERT_OK(loom_aie2p_native_route_payload(
      &context_, paths_[0].payload, nullptr, &next_source, &valid));
  ASSERT_TRUE(valid);
  EXPECT_TRUE(egress->routed);
  EXPECT_TRUE(path.payloads[0]->routed);
}

class MulticastPayloadTest
    : public PayloadTest,
      public ::testing::WithParamInterface<std::tuple<unsigned, bool>> {
 protected:
  void SetUp() override {
    PayloadTest::SetUp();
    pass_.diagnostic_emitter = diagnostics_.emitter();
    const auto [sources, queries] = GetParam();
    multicast_workers_.resize(sources);
    invocations_.resize(sources);
    ownership_.resize(sources);
    nodes_.resize(sources + 3);
    owners_.resize(sources + 1);
    distributions_.resize(sources);
    multicast_paths_.resize(sources);
    receivers_.resize(sources);
    packets_.resize(sources);
    packet_receivers_.resize(sources);
    plans_.resize(sources);
    // Root-owned state coexists with the complete serial child sequence.
    constexpr auto none = IREE_HOST_SIZE_MAX;
    nodes_[0] = {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none};
    nodes_[1] = {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2};
    nodes_[2] = {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none};
    owners_[0] = 1;
    for (unsigned source = 0; source < sources; ++source) {
      const size_t owner = source + 3;
      nodes_[owner] = {LOOM_PIPELINE_LIFETIME_OWNER, source + 1, 2, none,
                       source + 1 < sources ? owner + 1 : none};
      owners_[source + 1] = owner;
      invocations_[source].ordinal = source + 1;
      ownership_[source].invocation = &invocations_[source];
      multicast_workers_[source].distribution_plan = &plans_[source];
      plans_[source].geometries = &distributions_[source];
      auto& distribution = distributions_[source];
      auto& receivers = receivers_[source];
      auto& path = multicast_paths_[source];
      receivers.resize(32);
      for (unsigned column = 0; column < 8; ++column) {
        for (unsigned row = 2; row < 6; ++row) {
          receivers[column * 4 + row - 2].tile = &tiles_[column * 6 + row];
        }
      }
      distribution.invocation = &invocations_[source];
      distribution.sender.tile = &tiles_[source * 6 + 1];
      distribution.receivers = receivers.data();
      distribution.receiver_count = receivers.size();
      distribution.path = &path;
      path.first = &distribution;
      if (queries) {
        // One ordered DMA path selects four private queries and one broadcast.
        // Class order deliberately differs from the selected shared namespace.
        packets_[source].resize(5);
        packet_receivers_[source].resize(5);
        for (unsigned c = 0; c < 5; ++c) {
          auto& indices = packet_receivers_[source][c];
          if (c < 4) {
            indices = {source * 4 + c};
          } else {
            for (unsigned i = 0; i < 32; ++i) {
              indices.push_back(i);
            }
          }
          packets_[source][c].receivers = indices.data();
          packets_[source][c].destination_count = indices.size();
        }
        path.packet_count = 5;
        path.packets = packets_[source].data();
      }
    }
    context_.workers = multicast_workers_.data();
    context_.dma_queue_count = sources * 33;
    realization_.invocations.workers = ownership_.data();
    realization_.invocations.worker_count = sources;
    realization_.invocations.channel_count = 32;
    realization_.invocations.lifetimes = {
        .nodes = nodes_.data(),
        .count = nodes_.size(),
        .owners = owners_.data(),
        .has_serial = true,
    };
    IREE_ASSERT_OK(loom_pipeline_lifetimes_index(
        &arena_, &realization_.invocations.lifetimes));
    const size_t route_capacity = sources * (32 + (queries ? 4 : 0)) * 16;
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        context_.family, route_capacity, &arena_, &context_.routing));
    routes_.resize(route_capacity);
    context_.routes = routes_.data();
  }

  void Assign(bool expected_valid = true) {
    std::vector<iree_host_size_t> order;
    for (size_t i = 0; i < multicast_workers_.size(); ++i) {
      order.push_back(i);
    }
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_assign_dma_queues(
        &context_, &realization_, order.data(), order.size(), &valid));
    ASSERT_EQ(valid, expected_valid);
  }

  void Route() {
    uint32_t next_source = 0;
    for (auto& path : multicast_paths_) {
      for (unsigned c = 0; c < (path.packet_count ? path.packet_count : 1);
           ++c) {
        bool valid = false;
        IREE_ASSERT_OK(loom_aie2p_native_route_payload(
            &context_, path.payloads[c], nullptr, &next_source, &valid));
        ASSERT_TRUE(valid);
      }
    }
    const loom_aie2p_native_switch_input_t input = {
        .routing = &context_.routing,
        .routes = context_.routes,
        .lifetimes = &realization_.invocations.lifetimes,
        .route_lifetimes = context_.route_lifetimes,
        .control_sources = context_.control_sources,
        .control_destination_sources = context_.control_destination_sources,
    };
    iree_string_view_t requirement = iree_string_view_empty();
    IREE_ASSERT_OK(loom_aie2p_native_switches_build(
        &input, &arena_, &context_.switches, &requirement));
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    EXPECT_TRUE(diagnostics_.emissions.empty());
    switches_.resize(tiles_.size());
    for (auto* sw = context_.switches; sw; sw = sw->next) {
      switches_[sw->coordinate.column * 6 + sw->coordinate.row] = sw;
    }
  }

  void ExpectDelivery(uint16_t source, size_t packet_index) {
    const auto* payload = multicast_paths_[source].payloads[packet_index];
    const uint8_t packet = payload->packet;
    std::vector<unsigned> expected(tiles_.size() * 2);
    const auto& path = multicast_paths_[source];
    const size_t count = path.packet_count
                             ? packets_[source][packet_index].destination_count
                             : receivers_[source].size();
    for (size_t i = 0; i < count; ++i) {
      const size_t index =
          path.packet_count ? packet_receivers_[source][packet_index][i] : i;
      const auto& receiver = receivers_[source][index];
      ++expected[(receiver.tile - tiles_.data()) * 2 + receiver.engine];
    }
    using Arrival =
        std::tuple<uint16_t, uint16_t, loom_xdna_stream_port_t, uint8_t>;
    std::vector<Arrival> pending = {{source, 1, LOOM_XDNA_STREAM_PORT_DMA, 0}};
    std::set<Arrival> visited;
    std::vector<unsigned> deliveries(tiles_.size() * 2);
    while (!pending.empty()) {
      const auto arrival = pending.back();
      pending.pop_back();
      ASSERT_TRUE(visited.insert(arrival).second)
          << "duplicate arrival or loop";
      const auto [column, row, port, channel] = arrival;
      ASSERT_LT(column, context_.family->column_count);
      ASSERT_LT(row, context_.family->row_count);
      const auto* sw = switches_[column * 6 + row];
      ASSERT_NE(sw, nullptr);
      const auto* facts = tiles_[column * 6 + row].facts;
      const auto* input = loom_xdna_array_stream_port_range(
          context_.family, facts->kind, LOOM_XDNA_STREAM_DIRECTION_SLAVE, port);
      const uint32_t slave = input->ordinal + channel;
      const uint32_t input_word =
          sw->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].words[slave];
      ASSERT_NE(input_word & (1u << 31), 0u);
      const bool packet_mode = (input_word & (1u << 30)) != 0;
      unsigned matches = 0;
      uint32_t filter = 0;
      for (unsigned slot = 0; slot < 4; ++slot) {
        const uint32_t word = sw->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS]
                                  .words[slave * 4 + slot];
        const uint32_t mask = (word >> 16) & 31;
        if ((word & (1u << 8)) && ((word >> 24) & mask) == (packet & mask)) {
          ++matches;
          filter = word;
        }
      }
      ASSERT_EQ(matches, packet_mode ? 1u : 0u);
      unsigned destinations = 0;
      for (size_t range = 0; range < context_.family->stream_port_count;
           ++range) {
        const auto& output = context_.family->stream_ports[range];
        if (output.tile_kind != facts->kind ||
            output.direction != LOOM_XDNA_STREAM_DIRECTION_MASTER) {
          continue;
        }
        for (uint8_t lane = 0; lane < output.count; ++lane) {
          const uint32_t word = sw->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]
                                    .words[output.ordinal + lane];
          if (!(word & (1u << 31)) ||
              ((word & (1u << 30)) != 0) != packet_mode) {
            continue;
          }
          if (packet_mode ? (word & 7) != (filter & 7) ||
                                !(word & (1u << (((filter >> 4) & 3) + 3)))
                          : (word & 31) != slave) {
            continue;
          }
          ++destinations;
          switch (output.port) {
            case LOOM_XDNA_STREAM_PORT_DMA:
              ASSERT_LT(lane, 2);
              EXPECT_TRUE(packet_mode);
              EXPECT_NE(word & (1u << 7), 0u) << "receiver must strip header";
              ++deliveries[(column * 6 + row) * 2 + lane];
              break;
            case LOOM_XDNA_STREAM_PORT_NORTH:
              pending.emplace_back(column, row + 1, LOOM_XDNA_STREAM_PORT_SOUTH,
                                   lane);
              break;
            case LOOM_XDNA_STREAM_PORT_SOUTH:
              pending.emplace_back(column, row - 1, LOOM_XDNA_STREAM_PORT_NORTH,
                                   lane);
              break;
            case LOOM_XDNA_STREAM_PORT_EAST:
              pending.emplace_back(column + 1, row, LOOM_XDNA_STREAM_PORT_WEST,
                                   lane);
              break;
            case LOOM_XDNA_STREAM_PORT_WEST:
              pending.emplace_back(column - 1, row, LOOM_XDNA_STREAM_PORT_EAST,
                                   lane);
              break;
            default:
              FAIL() << "unexpected terminal port";
          }
        }
      }
      ASSERT_NE(destinations, 0u);
    }
    EXPECT_EQ(deliveries, expected);
  }

  // Native distribution demand owned by each child construction.
  std::vector<loom_aie2p_native_worker_t> multicast_workers_;
  // Distinct child constructions participating in the common lifetime tree.
  std::vector<loom_pipeline_invocation_t> invocations_;
  // Construction ownership consumed by physical DMA queue allocation.
  std::vector<loom_pipeline_invocation_worker_t> ownership_;
  // Retained lifetime tree; changing its serial node changes queue allocation.
  std::vector<loom_pipeline_lifetime_t> nodes_;
  // Construction-to-lifetime mapping retained by common analysis.
  std::vector<iree_host_size_t> owners_;
  // One ordered distribution program per physical source.
  std::vector<loom_aie2p_native_distribution_t> distributions_;
  // Formed demand paths awaiting physical queue and packet assignment.
  std::vector<loom_aie2p_native_distribution_path_t> multicast_paths_;
  // Stable receiver storage owned by each geometry.
  std::vector<std::vector<loom_aie2p_native_distribution_endpoint_t>>
      receivers_;
  // Optional private-query and broadcast packet classes in each source program.
  std::vector<std::vector<loom_aie2p_native_distribution_packet_t>> packets_;
  // Authored destination subsets, independent of payload interning and routing.
  std::vector<std::vector<std::vector<uint32_t>>> packet_receivers_;
  // Per-worker geometry lists.
  std::vector<loom_aie2p_native_distribution_plan_t> plans_;
  // Emitted switch banks indexed by physical tile.
  std::vector<const loom_aie2p_native_switch_t*> switches_;
  // Resource rejection evidence from the actual native planner.
  loom::testing::DiagnosticEmissionCapture diagnostics_;
};

TEST_P(MulticastPayloadTest, SerializedSourcesPreserveEveryPacketDestination) {
  ASSERT_NO_FATAL_FAILURE(Assign());
  Form();
  ASSERT_NO_FATAL_FAILURE(Route());
  for (uint16_t source = 0; source < multicast_paths_.size(); ++source) {
    const auto& path = multicast_paths_[source];
    for (unsigned c = 0; c < (path.packet_count ? path.packet_count : 1); ++c) {
      SCOPED_TRACE(::testing::Message()
                   << "source=" << source << " class=" << c);
      if (path.payloads[c]->source->framed) {
        ExpectDelivery(source, c);
      }
    }
  }
  if (std::get<0>(GetParam()) > 1) {
    // Each directed horizontal link uses one payload lane, independent of
    // sender count; private query classes never occupy horizontal links.
    for (unsigned column = 0; column < 7; ++column) {
      EXPECT_EQ(context_.routing.link_channels.eastbound[5 * 8 + column], 1u);
      EXPECT_EQ(context_.routing.link_channels.eastbound[2 * 8 + column], 0u);
    }
  }
}

TEST_P(MulticastPayloadTest, ConcurrentSourcesKeepIndependentReceiverQueues) {
  nodes_[2].kind = LOOM_PIPELINE_LIFETIME_CONCURRENT;
  realization_.invocations.lifetimes.has_serial = false;
  const auto [sources, queries] = GetParam();
  // A core has two receive queues. More concurrent senders are rejected by
  // queue allocation before payload routing can consume those identities.
  ASSERT_NO_FATAL_FAILURE(Assign(sources <= 2));
  if (sources > 2) {
    return;
  }
  Form();
  ASSERT_NO_FATAL_FAILURE(Route());
  for (uint16_t source = 0; source < sources; ++source) {
    for (const auto& receiver : receivers_[source]) {
      EXPECT_EQ(receiver.engine, source);
    }
    for (unsigned c = 0; c < (queries ? 5 : 1); ++c) {
      EXPECT_EQ(multicast_paths_[source].payloads[c]->packet, c);
      if (queries) {
        ExpectDelivery(source, c);
      }
    }
  }
  if (sources == 2) {
    EXPECT_EQ(context_.routing.link_channels.eastbound[2 * 8 + 1], 3u);
  }
}

TEST_P(MulticastPayloadTest, DifferentReceiverSetsNeverMergePacketIdentity) {
  // The first source omits one destination. Shared queues alone prove
  // serialization, but only the complete destination set defines packet ID.
  auto& distribution = distributions_[0];
  --distribution.receiver_count;
  --context_.dma_queue_count;
  receivers_[0].pop_back();
  if (std::get<1>(GetParam())) {
    --packets_[0][4].destination_count;
    packet_receivers_[0][4].pop_back();
  }
  ASSERT_NO_FATAL_FAILURE(Assign());
  Form();
  const auto& path = multicast_paths_[0];
  for (unsigned c = 0; c < (path.packet_count ? path.packet_count : 1); ++c) {
    EXPECT_EQ(path.payloads[c]->packet, c);
  }
  uint32_t next_source = 0;
  bool valid = false;
  IREE_ASSERT_OK(loom_aie2p_native_route_payload(
      &context_, path.payloads[path.packet_count ? 4 : 0], nullptr,
      &next_source, &valid));
  ASSERT_TRUE(valid);
  // An independent sender retains the ordinary per-receiver-row geometry.
  EXPECT_EQ(context_.routing.link_channels.eastbound[2 * 8 + 1], 1u);
}

INSTANTIATE_TEST_SUITE_P(SourceCountAndPrivateQueries, MulticastPayloadTest,
                         ::testing::Combine(::testing::Values(1u, 2u, 3u, 4u,
                                                              5u, 6u, 7u, 8u),
                                            ::testing::Bool()));

}  // namespace
