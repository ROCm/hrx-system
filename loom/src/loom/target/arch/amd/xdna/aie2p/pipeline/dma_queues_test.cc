// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_queues.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

namespace {

class DmaQueuesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tiles_.resize(context_.family->column_count * context_.family->row_count);
    context_.tiles = tiles_.data();
    for (size_t i = 0; i < tiles_.size(); ++i) {
      tiles_[i].coordinate = {
          static_cast<uint16_t>(i / context_.family->row_count),
          static_cast<uint16_t>(i % context_.family->row_count)};
      tiles_[i].facts =
          loom_xdna_array_tile_facts(context_.family, tiles_[i].coordinate);
    }
    pools_[0] = &tiles_[1];
    context_.pool_tiles = pools_;
    context_.workers = workers_;
    context_.dma_queue_count = 6;
    realization_.invocations.workers = ownership_;
    realization_.invocations.worker_count = IREE_ARRAYSIZE(workers_);
    realization_.invocations.lifetimes = {
        .nodes = nodes_,
        .count = IREE_ARRAYSIZE(nodes_),
        .owners = owners_,
        .has_serial = true,
    };
    IREE_ASSERT_OK(loom_pipeline_lifetimes_index(
        &arena_, &realization_.invocations.lifetimes));
    for (size_t i = 0; i < IREE_ARRAYSIZE(workers_); ++i) {
      invocations_[i].ordinal = i;
      ownership_[i].invocation = &invocations_[i];
    }
    // The root's ordinary ingress remains live while either child runs.
    ordinary_.local = &tiles_[1];
    ordinary_.shim = &tiles_[0];
    ordinary_.ingress = true;
    workers_[0].transfer_paths = &ordinary_;
    // The first child has two geometries using one multicast path.
    for (size_t i = 0; i < 2; ++i) {
      distributions_[i].sender.tile = &tiles_[2];
      distributions_[i].receivers = &receivers_[i];
      distributions_[i].receiver_count = 1;
      distributions_[i].path = &distribution_path_;
      receivers_[i].tile = &tiles_[1];
    }
    distributions_[0].next_demand = &distributions_[1];
    distribution_path_.first = &distributions_[0];
    distribution_plan_.geometries = distributions_;
    workers_[1].distribution_plan = &distribution_plan_;
    // The second child streams two input phases through one receive window.
    input_source_.phase_count = 2;
    input_.source = &input_source_;
    input_.phases = phases_;
    input_.window_capacity = 1;
    input_.path = &input_path_;
    input_path_.shim = &tiles_[0];
    phases_[0].destination = phases_[1].destination = &channel_;
    workers_[2].input = &input_;
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  void Assign() {
    std::vector<iree_host_size_t> distributions;
    for (size_t i = 0; i < IREE_ARRAYSIZE(workers_); ++i) {
      if (workers_[i].distribution_plan) {
        distributions.push_back(i);
      }
    }
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_assign_dma_queues(
        &context_, &realization_, distributions.data(), distributions.size(),
        &valid));
    ASSERT_TRUE(valid);
  }

  // Root state coexists with two children of one serial scope.
  static constexpr auto none = IREE_HOST_SIZE_MAX;
  // Retained common lifetime tree; node 2 controls sibling coexistence.
  loom_pipeline_lifetime_t nodes_[7] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  // Canonical construction-owner indices.
  const iree_host_size_t owners_[3] = {1, 5, 6};
  // Construction identities referenced by workers.
  loom_pipeline_invocation_t invocations_[3] = {};
  // Retained worker-to-construction correspondence.
  loom_pipeline_invocation_worker_t ownership_[3] = {};
  // Native requests in root, first child, and second child order.
  loom_aie2p_native_worker_t workers_[3] = {};
  // Root's ordinary shim/local endpoint pair.
  loom_aie2p_native_dma_path_t ordinary_ = {};
  // Logical multicast geometries sharing one physical demand.
  loom_aie2p_native_distribution_t distributions_[2] = {};
  // Receive endpoints associated with those geometries.
  loom_aie2p_native_distribution_endpoint_t receivers_[2] = {};
  // Shared multicast topology within the first child.
  loom_aie2p_native_distribution_path_t distribution_path_ = {};
  // Occurrence-owned multicast demand.
  loom_aie2p_native_distribution_plan_t distribution_plan_ = {};
  // Retained input phase inventory.
  loom_pipeline_transport_input_t input_source_ = {};
  // Native phased input in the second child.
  loom_aie2p_native_input_t input_ = {};
  // Input's shim demand before physical interning.
  loom_aie2p_native_input_path_t input_path_ = {};
  // Two receiver phases sharing the selected window.
  loom_aie2p_native_input_phase_t phases_[2] = {};
  // Destination's canonical backing-pool identity.
  loom_aie2p_native_channel_t channel_ = {};
  // Source storage's placed memory tile.
  loom_aie2p_native_tile_t* pools_[1] = {};
  // Complete tile array required by the selected target family.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Compiler-owned construction and worker inventory.
  loom_pipeline_realization_t realization_ = {};
  // Pass arena used by layout; successful selection emits no diagnostic.
  loom_pass_t pass_ = {};
  // Native resource state under test.
  loom_aie2p_native_context_t context_ = {};
  // Backing for one test's retained allocation result.
  iree_arena_block_pool_t blocks_;
  // Arena owning that result.
  iree_arena_allocator_t arena_;
};

TEST_F(DmaQueuesTest, TransportFamiliesShareOnlySerialQueueLifetimes) {
  Assign();
  EXPECT_EQ(ordinary_.local_engine, 0);
  EXPECT_EQ(ordinary_.shim_engine, 0);
  EXPECT_EQ(receivers_[0].engine, 1);
  EXPECT_EQ(receivers_[1].engine, receivers_[0].engine);
  EXPECT_EQ(phases_[0].receiver.engine, receivers_[0].engine);
  EXPECT_EQ(phases_[1].receiver.engine, phases_[0].receiver.engine);
  EXPECT_EQ(input_path_.input_engine, 1);
  EXPECT_EQ(tiles_[1].s2mm_count, 2);
}

TEST_F(DmaQueuesTest, ConcurrentTransportsKeepIndependentQueues) {
  nodes_[2].kind = LOOM_PIPELINE_LIFETIME_CONCURRENT;
  realization_.invocations.lifetimes.has_serial = false;
  Assign();
  EXPECT_EQ(receivers_[0].engine, 1);
  EXPECT_EQ(phases_[0].receiver.engine, 2);
  EXPECT_EQ(tiles_[1].s2mm_count, 3);
}

TEST_F(DmaQueuesTest, IndependentInputPhasesReserveSeparateReceivers) {
  input_.window_capacity = 0;
  ++context_.dma_queue_count;
  Assign();
  EXPECT_EQ(phases_[0].receiver.engine, 1);
  EXPECT_EQ(phases_[1].receiver.engine, 2);
  EXPECT_EQ(tiles_[1].s2mm_count, 3);
}

TEST_F(DmaQueuesTest, SameTileCopiesPrecedeOneWayQueueDemand) {
  ownership_[0].invocation = &invocations_[1];
  for (size_t row : {1, 2}) {
    ordinary_.local = &tiles_[row];
    for (size_t i = 0; i < 2; ++i) {
      distributions_[i].sender.tile = &tiles_[row];
      receivers_[i].tile = &tiles_[row];
    }
    Assign();
    EXPECT_EQ(distributions_[0].sender.engine, 0);
    EXPECT_EQ(receivers_[0].engine, distributions_[0].sender.engine);
    EXPECT_EQ(ordinary_.local_engine, 1);
  }
}

TEST_F(DmaQueuesTest, CompletionOwnersFollowPhysicalQueueLifetimes) {
  loom_aie2p_native_dma_path_t paths[3] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
    workers_[i] = {};
    paths[i].local = &tiles_[1];
    paths[i].shim = &tiles_[0];
    paths[i].ingress = true;
    paths[i].completion_tokens = 2;
    paths[i].completion_packet = 3 + i;
    workers_[i].paths = workers_[i].transfer_paths = &paths[i];
  }
  Assign();
  loom_aie2p_native_select_dma_controllers(&context_, &realization_);
  // Parent queues coexist with both children and keep one immutable owner.
  EXPECT_EQ(tiles_[0].dma_controls[0][0].controller, 3);
  EXPECT_FALSE(tiles_[0].dma_controls[0][0].dynamic);
  EXPECT_EQ(tiles_[1].dma_controls[1][0].controller, 3);
  EXPECT_FALSE(tiles_[1].dma_controls[1][0].dynamic);
  // Both serial children share the second pair but receive their own tokens.
  EXPECT_EQ(paths[1].shim_engine, paths[2].shim_engine);
  EXPECT_EQ(paths[1].local_engine, paths[2].local_engine);
  EXPECT_EQ(tiles_[0].dma_controls[0][1].controller, 4);
  EXPECT_TRUE(tiles_[0].dma_controls[0][1].dynamic);
  EXPECT_EQ(tiles_[1].dma_controls[1][1].controller, 4);
  EXPECT_TRUE(tiles_[1].dma_controls[1][1].dynamic);
}

TEST_F(DmaQueuesTest, TokenFreeUsersDoNotChangeCompletionDestination) {
  loom_aie2p_native_dma_path_t paths[3] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
    workers_[i] = {};
    paths[i].local = &tiles_[1];
    paths[i].shim = &tiles_[0];
    paths[i].completion_tokens = i == 1 ? 1 : 0;
    paths[i].completion_packet = i == 1 ? 7 : 0;
    workers_[i].paths = workers_[i].transfer_paths = &paths[i];
  }
  Assign();
  loom_aie2p_native_select_dma_controllers(&context_, &realization_);
  EXPECT_EQ(tiles_[0].dma_controls[1][1].controller, 7);
  EXPECT_FALSE(tiles_[0].dma_controls[1][1].dynamic);
  EXPECT_EQ(tiles_[0].dma_controls[1][0].controller, UINT8_MAX);
  EXPECT_EQ(tiles_[1].dma_controls[0][1].controller, UINT8_MAX);
}

TEST_F(DmaQueuesTest, ShimOverflowPreservesCompleteSerialLifetimeClasses) {
  for (bool ingress : {false, true}) {
    for (uint16_t column : {0, 3, 7}) {
      const size_t tile_index = column * context_.family->row_count;
      const uint16_t neighboring_column = column ? column - 1 : 1;
      const size_t neighbor = neighboring_column * context_.family->row_count;
      loom_aie2p_native_dma_path_t paths[4] = {};
      for (size_t i = 0; i < IREE_ARRAYSIZE(workers_); ++i) {
        workers_[i] = {};
      }
      for (auto& path : paths) {
        path.local = &tiles_[tile_index + 1];
        path.shim = &tiles_[tile_index];
        path.ingress = ingress;
      }
      // The root has two independent local endpoints. Its two shim queues
      // coexist with either serial child, whose shared third queue must move.
      paths[1].local = &tiles_[tile_index + 2];
      paths[0].next_demand = &paths[1];
      workers_[0].transfer_paths = paths;
      workers_[1].transfer_paths = &paths[2];
      workers_[2].transfer_paths = &paths[3];
      context_.dma_queue_count = 8;
      Assign();
      EXPECT_EQ(paths[0].shim, &tiles_[tile_index]);
      EXPECT_EQ(paths[1].shim, &tiles_[tile_index]);
      EXPECT_EQ(paths[0].shim_engine, 0);
      EXPECT_EQ(paths[1].shim_engine, 1);
      EXPECT_EQ(paths[2].shim, &tiles_[neighbor]);
      EXPECT_EQ(paths[3].shim, paths[2].shim);
      EXPECT_EQ(paths[2].shim_engine, 0);
      EXPECT_EQ(paths[3].shim_engine, paths[2].shim_engine);
      EXPECT_EQ(paths[2].local_engine, paths[3].local_engine);
      EXPECT_EQ(
          ingress ? tiles_[neighbor].mm2s_count : tiles_[neighbor].s2mm_count,
          1);
    }
  }
}

TEST_F(DmaQueuesTest, OrdinaryAndPhasedIngressShareRelocatedSerialQueue) {
  loom_aie2p_native_dma_path_t extra = ordinary_;
  extra.local = &tiles_[2];
  ordinary_.next_demand = &extra;
  loom_aie2p_native_dma_path_t child = ordinary_;
  child.next_demand = nullptr;
  workers_[1] = {};
  workers_[1].transfer_paths = &child;
  context_.dma_queue_count = 8;
  Assign();
  const size_t neighbor = context_.family->row_count;
  EXPECT_EQ(ordinary_.shim, &tiles_[0]);
  EXPECT_EQ(extra.shim, &tiles_[0]);
  EXPECT_EQ(child.shim, &tiles_[neighbor]);
  EXPECT_EQ(input_path_.shim, child.shim);
  EXPECT_EQ(input_path_.input_engine, child.shim_engine);
  EXPECT_EQ(tiles_[neighbor].mm2s_count, 1);
  EXPECT_EQ(phases_[0].receiver.engine, child.local_engine);
  EXPECT_EQ(phases_[1].receiver.engine, child.local_engine);
}

TEST_F(DmaQueuesTest, ConcurrentShimOverflowKeepsIndependentQueues) {
  nodes_[2].kind = LOOM_PIPELINE_LIFETIME_CONCURRENT;
  realization_.invocations.lifetimes.has_serial = false;
  loom_aie2p_native_dma_path_t paths[4] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(workers_); ++i) {
    workers_[i] = {};
  }
  for (auto& path : paths) {
    path.local = &tiles_[1];
    path.shim = &tiles_[0];
  }
  paths[1].local = &tiles_[2];
  paths[0].next_demand = &paths[1];
  workers_[0].transfer_paths = paths;
  workers_[1].transfer_paths = &paths[2];
  workers_[2].transfer_paths = &paths[3];
  context_.dma_queue_count = 8;
  Assign();
  const size_t neighbor = context_.family->row_count;
  EXPECT_EQ(paths[2].shim, &tiles_[neighbor]);
  EXPECT_EQ(paths[3].shim, paths[2].shim);
  EXPECT_NE(paths[2].shim_engine, paths[3].shim_engine);
  EXPECT_EQ(tiles_[neighbor].s2mm_count, 2);
}

TEST_F(DmaQueuesTest, OverflowPreservesPreferredQueuesInOtherColumns) {
  const size_t neighbor = context_.family->row_count;
  loom_aie2p_native_dma_path_t paths[4] = {};
  for (auto& worker : workers_) {
    worker = {};
  }
  for (size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
    paths[i].local = &tiles_[i + 2];
    paths[i].shim = &tiles_[i == 3 ? neighbor : 0];
    if (i + 1 < IREE_ARRAYSIZE(paths)) {
      paths[i].next_demand = &paths[i + 1];
    }
  }
  workers_[0].transfer_paths = paths;
  context_.dma_queue_count = 8;
  Assign();
  // Column one's preferred demand is reserved before placing column zero's
  // third queue. Earlier overflow cannot displace a later preferred request.
  EXPECT_EQ(paths[3].shim, &tiles_[neighbor]);
  EXPECT_EQ(paths[3].shim_engine, 0);
  EXPECT_EQ(paths[2].shim, &tiles_[neighbor]);
  EXPECT_EQ(paths[2].shim_engine, 1);
  EXPECT_EQ(tiles_[neighbor].s2mm_count, 2);
}

}  // namespace
