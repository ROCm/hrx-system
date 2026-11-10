// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"

namespace {

class Aie2pSharingTest
    : public ::testing::TestWithParam<loom_xdna_tile_kind_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tiles_.resize(context_.family->column_count * context_.family->row_count);
    tiles_[1].coordinate = {0, 1};
    tiles_[3].coordinate = {0, 3};
    tiles_[1].memory_layout = &loom_aie2p_sram_memory;
    tiles_[3].memory_layout = &loom_aie2p_compute_memory;
    const bool sram_sender = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY;
    sender_index_ = sram_sender ? 1 : 3;
    receiver_index_ = sram_sender ? 3 : 1;
    context_.tiles = tiles_.data();
    context_.channels = channels_;
    context_.workers = &worker_;
    for (size_t i = 0; i < 2; ++i) {
      resources_[i].capacity = 1;
      channels_[i].source = &resources_[i];
    }
    records_.admissions = records_.transfers = 2;
    batch_.records = &records_;
    batch_.record_count = 1;
    distribution_.source = &batch_;
    distribution_.repetitions = 2;
    distribution_.sender.channel = &channels_[0];
    distribution_.sender.tile = &tiles_[sender_index_];
    receiver_.channel = &channels_[1];
    receiver_.tile = &tiles_[receiver_index_];
    channels_[1].configuration_returns.count = 2;
    distribution_.receivers = &receiver_;
    distribution_.receiver_count = 1;
    worker_.distributions = &distribution_;
    worker_.refills = &refill_;
    realization_.invocations.worker_count = 1;
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        context_.family, 1, 8, &arena_, &context_.routing));
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &context_.routing, 0, 0, tiles_[sender_index_].coordinate, 0,
        tiles_[receiver_index_].coordinate, 0, false));
    distribution_.routes.count = context_.routing.route_count;
    for (size_t i = 0; i < context_.routing.route_count; ++i) {
      routes_.push_back({&context_.routing.routes[i], UINT8_MAX, UINT32_MAX});
    }
    context_.routes = routes_.data();
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Prepare(bool eligible) {
    IREE_ASSERT_OK(loom_aie2p_native_prepare_sharing(&context_, &realization_));
    ASSERT_NE(context_.deliveries, nullptr);
    EXPECT_EQ(context_.deliveries[receiver_index_] != nullptr, eligible);
  }

  // Retained native planning storage.
  iree_arena_block_pool_t pool_ = {};
  // Lifetime shared by route and delivery facts.
  iree_arena_allocator_t arena_ = {};
  // Allocation service consumed by sharing selection.
  loom_pass_t pass_ = {};
  // Canonical physical plans, without worker bodies or instruction selection.
  loom_aie2p_native_context_t context_ = {};
  // Source tile in the column-major inventory.
  size_t sender_index_ = 0;
  // Destination tile in the column-major inventory.
  size_t receiver_index_ = 0;
  // Column-major physical tile identities.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Native packet classifications over an actual routed unicast path.
  std::vector<loom_aie2p_native_route_t> routes_;
  // Source and destination storage capacities.
  loom_pipeline_resource_channel_t resources_[2] = {};
  // Distinct protocol instances, independent of payload addresses.
  loom_aie2p_native_channel_t channels_[2] = {};
  // Source admission and transfer counts owned by worker analysis.
  loom_pipeline_transport_cohort_t records_ = {};
  // One source record per delivery.
  loom_pipeline_transport_batch_t batch_ = {};
  // Source DMA plan and its receiver.
  loom_aie2p_native_distribution_t distribution_ = {};
  // Independently publishing destination.
  loom_aie2p_native_distribution_endpoint_t receiver_ = {};
  // Existing compute reader needing refill control traffic.
  loom_aie2p_native_refill_t refill_ = {};
  // Selected transport and control owner.
  loom_aie2p_native_worker_t worker_ = {};
  // Bound worker inventory consumed by preparation.
  loom_pipeline_realization_t realization_ = {};
};

TEST_P(Aie2pSharingTest, StartupOnlyStreamsDoNotAllocateSharingState) {
  worker_.refills = nullptr;
  IREE_ASSERT_OK(loom_aie2p_native_prepare_sharing(&context_, &realization_));
  EXPECT_EQ(context_.deliveries, nullptr);
}

TEST_P(Aie2pSharingTest, RepeatedDeliveryRequiresExactReceiverReturns) {
  channels_[1].configuration_returns.count = 0;
  Prepare(false);
  channels_[1].configuration_returns.count = 2;
  Prepare(true);
}

TEST_P(Aie2pSharingTest, OneDeliveryUsesTheInitiallyFreeSlot) {
  records_.admissions = records_.transfers = distribution_.repetitions = 1;
  channels_[1].configuration_returns.count = 0;
  Prepare(true);
}

TEST_P(Aie2pSharingTest, ResidentReceiverRequiresExactSameInvocationReturns) {
  loom_pipeline_invocation_t parent = {};
  loom_pipeline_invocation_t child = {.parent = &parent};
  loom_pipeline_invocation_worker_t invocations[] = {{.invocation = &parent},
                                                     {.invocation = &parent}};
  loom_aie2p_native_worker_t workers[] = {
      worker_, {.execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE}};
  context_.workers = workers;
  realization_.invocations.worker_count = IREE_ARRAYSIZE(workers);
  realization_.invocations.workers = invocations;
  channels_[1].configuration_returns.count = 0;
  auto& reader = channels_[1].cursor.reader;
  reader.worker = UINT32_MAX;
  loom_aie2p_native_cursor_bind(&reader, 1, 1, 2, true, false);
  ASSERT_NO_FATAL_FAILURE(Prepare(true));
  ASSERT_NE(context_.deliveries[receiver_index_], nullptr);
  EXPECT_EQ(context_.deliveries[receiver_index_]->reader_worker, 1u);

  // A representative worker's count does not cover other readers, nor does
  // an upper bound prove that every read actually returns its credit.
  reader.shared = true;
  Prepare(false);
  reader.shared = false;
  reader.exact_admissions = false;
  Prepare(false);
  reader.exact_admissions = true;
  reader.maximum_admissions = 1;
  Prepare(false);
  reader.maximum_admissions = 2;

  // A child may execute a different number of times from this distribution.
  // Only the distribution's own finite receiver can issue its continuation.
  invocations[1].invocation = &child;
  Prepare(false);
  invocations[1].invocation = &parent;
  workers[1].execution = LOOM_AIE2P_NATIVE_EXECUTION_DMA;
  Prepare(false);
  workers[1].execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE;
  Prepare(true);
}

TEST_P(Aie2pSharingTest, MultiSlotAndRetainedPayloadsKeepTheirOwnProtocol) {
  for (auto& resource : resources_) {
    resource.capacity = 2;
    Prepare(false);
    resource.capacity = 1;
  }
  for (auto& channel : channels_) {
    channel.replay.record_count = 1;
    channel.replay.credits = 1;
    Prepare(false);
    channel.replay = {};
  }
  Prepare(true);
}

TEST_P(Aie2pSharingTest, UnusedCandidateDoesNotChangePacketOrQueuePolicy) {
  ASSERT_NO_FATAL_FAILURE(Prepare(true));
  loom_aie2p_native_commit_sharing(&context_);
  EXPECT_FALSE(distribution_.sender.wait_for_idle);
  EXPECT_EQ(channels_[1].delivery, nullptr);
  for (const auto& route : routes_) {
    EXPECT_EQ(route.packet, UINT8_MAX);
  }
}

TEST_P(Aie2pSharingTest, UsedPathAdmitsExactlyOneRecordPerReceiverGrant) {
  ASSERT_NO_FATAL_FAILURE(Prepare(true));
  auto* delivery = context_.deliveries[receiver_index_];
  ASSERT_NE(delivery, nullptr);
  delivery->sharing.used = true;
  const auto& layout = *distribution_.sender.tile->memory_layout;
  // A finite two-record startup chain is a valid input. Sharing must break
  // the chain so one granted destination slot cannot admit both records.
  uint32_t words[2][8] = {};
  words[0][0] = words[1][0] = 16;
  words[0][layout.lock_word] = words[1][layout.lock_word] =
      loom_xdna_register_field_encode(layout.locks.valid, 1);
  words[0][layout.chain.word] =
      loom_xdna_register_field_encode(layout.chain.enable, 1) |
      loom_xdna_register_field_encode(layout.chain.next, 3);
  uint32_t tasks[] = {
      loom_xdna_register_field_encode(layout.directions[0].start, 2)};
  auto& sender = distribution_.sender;
  sender.sequence.descriptor = 2;
  sender.sequence.descriptor_count = 2;
  sender.sequence.words = words;
  sender.sequence.task_count = 1;
  sender.sequence.tasks = tasks;
  loom_aie2p_native_commit_sharing(&context_);
  EXPECT_EQ(sender.sequence.descriptor_count, 1u);
  EXPECT_EQ(sender.sequence.task_count, 1u);
  EXPECT_EQ(tasks[0], 2u);
  EXPECT_EQ(delivery->task, tasks[0]);
  EXPECT_EQ(words[0][layout.chain.word] &
                loom_xdna_register_field_encode(layout.chain.enable, 1),
            0u);
  const auto length = loom_xdna_register_field_info(layout.descriptor_base);
  EXPECT_EQ((words[0][0] >> length.least_significant_bit) &
                (UINT32_MAX >> (32 - length.bit_width)),
            16u);
  EXPECT_TRUE(sender.wait_for_idle);
  EXPECT_EQ(channels_[1].delivery, delivery);
  for (const auto& route : routes_) {
    EXPECT_EQ(route.packet, 16u);
  }
  EXPECT_NE(words[0][layout.packet.word] &
                loom_xdna_register_field_encode(layout.packet.enable, 1),
            0u);
  const uint16_t engine = sender.engine;
  EXPECT_EQ(delivery->address, loom_xdna_register_field_address(
                                   context_.family, layout.directions[0].start,
                                   sender.tile->coordinate, &engine));
}

INSTANTIATE_TEST_SUITE_P(OnChipMemory, Aie2pSharingTest,
                         ::testing::Values(LOOM_XDNA_TILE_KIND_COMPUTE,
                                           LOOM_XDNA_TILE_KIND_MEMORY));

TEST(Aie2pSharingTest, IndependentCoresAgreeOnSharedPacketTerminals) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {0, 1}};
  const loom_aie2p_native_tile_t shim = {.coordinate = {0, 0}};
  loom_aie2p_native_delivery_t delivery = {};
  // One core reaches SRAM before the shim; another reaches the shim first.
  // Their independent source ordinals cannot define the shared packet IDs.
  const uint8_t memory_packet = loom_aie2p_native_delivery_packet(
      &delivery, 0, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  const uint8_t shim_packet = loom_aie2p_native_delivery_packet(
      &delivery, UINT32_C(1) << memory_packet, &shim,
      LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  ASSERT_NE(memory_packet, shim_packet);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                &delivery, 0, &shim, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            shim_packet);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                &delivery, UINT32_C(1) << shim_packet, &memory,
                LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            memory_packet);

  const uint32_t occupied =
      (UINT32_C(1) << memory_packet) | (UINT32_C(1) << shim_packet);
  const uint8_t credit_packet = loom_aie2p_native_delivery_packet(
      &delivery, occupied, &shim, LOOM_XDNA_STREAM_PORT_DMA, 0);
  EXPECT_NE(credit_packet, memory_packet);
  EXPECT_NE(credit_packet, shim_packet);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(&delivery, 0, &shim,
                                              LOOM_XDNA_STREAM_PORT_DMA, 0),
            credit_packet);
  EXPECT_NE(loom_aie2p_native_delivery_packet(&delivery, 0, &shim,
                                              LOOM_XDNA_STREAM_PORT_DMA, 1),
            credit_packet);
}

TEST(Aie2pSharingTest, UnsharedCorePacketsKeepIndependentIdentities) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {0, 1}};
  const loom_aie2p_native_tile_t shim = {.coordinate = {0, 0}};
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                nullptr, 0, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            0);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                nullptr, 0, &shim, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            0);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                nullptr, 1, &shim, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            1);
}

TEST(Aie2pSharingTest, CollectivesPreserveSparseControlIdentities) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {7, 1}};
  loom_aie2p_native_tile_t terminals[30] = {};
  loom_aie2p_native_delivery_t delivery = {};
  // Independent core streams populate the shared path's destination namespace.
  // The next core's first control route therefore receives a sparse high ID.
  for (uint8_t i = 0; i < 30; ++i) {
    terminals[i].coordinate = {static_cast<uint16_t>(i / 6),
                               static_cast<uint16_t>(i % 6)};
    ASSERT_NE(
        loom_aie2p_native_delivery_packet(
            &delivery, 0, &terminals[i], LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
        UINT8_MAX);
  }
  const uint8_t control = loom_aie2p_native_delivery_packet(
      &delivery, 0, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  ASSERT_EQ(control, 31);
  uint32_t occupied = UINT32_C(1) << control;
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), 30);
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), 29);
  EXPECT_EQ(occupied, UINT32_C(0xe0000000));
}

TEST(Aie2pSharingTest, CollectivesReserveEveryNonPayloadIdentityOnce) {
  const uint32_t payload = UINT32_C(1) << LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID;
  uint32_t occupied = payload;
  uint32_t selected = 0;
  for (uint8_t i = 0; i < 31; ++i) {
    const uint8_t packet = loom_aie2p_native_collective_packet(&occupied);
    ASSERT_LT(packet, 32);
    EXPECT_NE(packet, LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID);
    const uint32_t bit = UINT32_C(1) << packet;
    EXPECT_EQ(selected & bit, 0u);
    selected |= bit;
    EXPECT_EQ(occupied, selected | payload);
  }
  EXPECT_EQ(selected, UINT32_C(0xfffeffff));
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), UINT8_MAX);
  EXPECT_EQ(occupied, selected | payload);
}

TEST(Aie2pSharingTest, UnsharedCollectivesCanUseEveryPacketIdentity) {
  uint32_t occupied = 0;
  for (uint8_t i = 0; i < 32; ++i) {
    const uint32_t previous = occupied;
    const uint8_t packet = loom_aie2p_native_collective_packet(&occupied);
    ASSERT_LT(packet, 32);
    EXPECT_EQ(previous & (UINT32_C(1) << packet), 0u);
    EXPECT_EQ(occupied, previous | (UINT32_C(1) << packet));
  }
  EXPECT_EQ(occupied, UINT32_MAX);
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), UINT8_MAX);
  EXPECT_EQ(occupied, UINT32_MAX);
}

TEST(Aie2pSharingTest, ControlSelectionPreservesCollectiveIdentities) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {0, 1}};
  // The shared payload and all but two other IDs are assigned on this core.
  uint32_t occupied = UINT32_C(0x7ffffffe);
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), 31);
  const uint8_t control = loom_aie2p_native_delivery_packet(
      nullptr, occupied, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  ASSERT_EQ(control, 0);
  occupied |= UINT32_C(1) << control;
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), UINT8_MAX);
  EXPECT_EQ(
      loom_aie2p_native_delivery_packet(nullptr, occupied, &memory,
                                        LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
      UINT8_MAX);
}

}  // namespace
