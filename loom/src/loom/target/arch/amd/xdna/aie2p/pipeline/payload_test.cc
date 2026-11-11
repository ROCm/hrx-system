// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"

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

TEST_F(PayloadTest, DestinationsOwnTheirRoutingIdentityBudget) {
  Ordinary(0, 2, true);
  Ordinary(1, 3, true);
  Form();
  // Two destinations share a physical sender. The route inventory budgets
  // one identity per destination, including the canonical prefix-tree root.
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      context_.family, 2, 16, &arena_, &context_.routing));
  loom_aie2p_native_route_t routes[16] = {};
  context_.routes = routes;
  uint32_t next_source = 0;
  for (size_t i = 0; i < 2; ++i) {
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_route_payload(
        &context_, paths_[i].payload, nullptr, &next_source, &valid));
    ASSERT_TRUE(valid);
    EXPECT_EQ(next_source, i + 1);
    EXPECT_EQ(paths_[i].payload->source->route, 0u);
  }
  EXPECT_NE(context_.routing.prefixes.roots[0], IREE_HOST_SIZE_MAX);
  EXPECT_EQ(context_.routing.prefixes.roots[1], IREE_HOST_SIZE_MAX);
  const auto& terminal = routes[context_.routing.route_count - 1];
  EXPECT_EQ(terminal.edge->coordinate.row, 3);
  EXPECT_TRUE(terminal.edge->dma_destination);
  EXPECT_EQ(terminal.packet, 1);
}

TEST_F(PayloadTest, PhasedInputAndOrdinaryIngressUseTheSamePhysicalClass) {
  Ordinary(0, 2, true);
  loom_aie2p_native_tile_t* pools[] = {&tiles_[2]};
  context_.pool_tiles = pools;
  loom_aie2p_native_channel_t destination = {};
  loom_pipeline_transport_input_t source = {};
  source.phase_count = 1;
  loom_aie2p_native_input_phase_t phase = {};
  phase.destination = &destination;
  loom_aie2p_native_input_path_t path = {};
  path.shim = &tiles_[0];
  loom_aie2p_native_input_t input = {};
  input.source = &source;
  input.path = &path;
  input.phases = &phase;
  workers_[1].input = &input;
  Form();
  EXPECT_EQ(paths_[0].payload, phase.payload);
  EXPECT_TRUE(phase.payload->source->framed);
  EXPECT_EQ(phase.payload->source->class_count, 1);
  EXPECT_FALSE(phase.payload->distribution_only);
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

}  // namespace
