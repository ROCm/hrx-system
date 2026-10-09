// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/channel_resources.h"

#include "iree/testing/gtest.h"

namespace {

const loom_xdna_tile_facts_t* ComputeFacts() {
  return loom_xdna_array_tile_kind_facts(loom_xdna_npu2_array_family(),
                                         LOOM_XDNA_TILE_KIND_COMPUTE);
}

const loom_xdna_tile_facts_t* ShimFacts() {
  return loom_xdna_array_tile_kind_facts(loom_xdna_npu2_array_family(),
                                         LOOM_XDNA_TILE_KIND_SHIM_NOC);
}

loom_aie2p_array_tile_resources_t MakeResources(uint32_t* bank_cursors) {
  loom_aie2p_array_tile_resources_t resources = {
      .facts = ComputeFacts(),
      .bank_cursors = bank_cursors,
      .flags = LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_WORKER};
  return resources;
}

loom_aie2p_array_compute_endpoint_request_t MakeRequest(
    loom_xdna_dma_direction_t direction) {
  loom_aie2p_array_compute_endpoint_request_t request = {
      .coordinate = {0, 2},
      .load_address_base = ComputeFacts()->memory.local_load_base,
      .record_count = 2,
      .record_byte_length = 64,
      .direction = direction};
  return request;
}

TEST(Aie2pArrayChannelResourcesTest, ComputeProposalRetainsExactTransition) {
  uint32_t bank_cursors[4] = {0};
  loom_aie2p_array_tile_resources_t resources = MakeResources(bank_cursors);
  const loom_aie2p_array_compute_endpoint_request_t request =
      MakeRequest(LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM);
  loom_aie2p_array_compute_endpoint_proposal_t proposal = {};

  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_compute(
                &resources, /*predecessor=*/nullptr, &request, &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE);
  EXPECT_EQ(resources.next_memory_to_stream_channel, 0u);
  EXPECT_EQ(resources.next_buffer_descriptor, 0u);
  EXPECT_EQ(resources.next_lock, 0u);
  EXPECT_EQ(resources.next_bank, 0u);
  for (uint32_t cursor : bank_cursors) {
    EXPECT_EQ(cursor, 0u);
  }

  EXPECT_EQ(proposal.dma_channel, 0u);
  EXPECT_EQ(proposal.buffer_descriptor_start, 0u);
  EXPECT_EQ(proposal.buffer_descriptor_count, 2u);
  EXPECT_EQ(proposal.ring.owner_offsets[0], 0u);
  EXPECT_EQ(proposal.ring.owner_offsets[1], 16u * 1024u);
  EXPECT_EQ(proposal.ring.credit_lock, 0u);
  EXPECT_EQ(proposal.ring.ready_lock, 1u);

  loom_aie2p_array_channel_resources_commit_compute(&proposal, &resources);
  EXPECT_EQ(resources.next_memory_to_stream_channel, 1u);
  EXPECT_EQ(resources.next_buffer_descriptor, 2u);
  EXPECT_EQ(resources.next_lock, 2u);
  EXPECT_EQ(resources.next_bank, 2u);
  EXPECT_EQ(bank_cursors[0], 64u);
  EXPECT_EQ(bank_cursors[1], 64u);
}

TEST(Aie2pArrayChannelResourcesTest,
     SameTileReceiverComposesRetainedSourceTransition) {
  uint32_t bank_cursors[4] = {0};
  loom_aie2p_array_tile_resources_t resources = MakeResources(bank_cursors);
  const loom_aie2p_array_compute_endpoint_request_t source_request =
      MakeRequest(LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM);
  loom_aie2p_array_compute_endpoint_proposal_t source = {};
  ASSERT_EQ(loom_aie2p_array_channel_resources_propose_compute(
                &resources, /*predecessor=*/nullptr, &source_request, &source),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE);

  loom_aie2p_array_compute_endpoint_request_t receiver_request =
      MakeRequest(LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY);
  receiver_request.loopback_source_dma_channel = source.dma_channel;
  receiver_request.flags =
      LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_REQUEST_FLAG_REQUIRE_LOOPBACK;
  loom_aie2p_array_compute_endpoint_proposal_t receiver = {};
  ASSERT_EQ(loom_aie2p_array_channel_resources_propose_compute(
                &resources, &source, &receiver_request, &receiver),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE);

  EXPECT_EQ(receiver.dma_channel, 0u);
  EXPECT_EQ(receiver.buffer_descriptor_start, 2u);
  EXPECT_EQ(receiver.ring.owner_offsets[0], 32u * 1024u);
  EXPECT_EQ(receiver.ring.owner_offsets[1], 48u * 1024u);
  EXPECT_EQ(receiver.ring.credit_lock, 2u);
  EXPECT_EQ(receiver.ring.ready_lock, 3u);
  for (uint32_t cursor : bank_cursors) {
    EXPECT_EQ(cursor, 0u);
  }

  loom_aie2p_array_channel_resources_commit_compute(&source, &resources);
  loom_aie2p_array_channel_resources_commit_compute(&receiver, &resources);
  EXPECT_EQ(resources.next_memory_to_stream_channel, 1u);
  EXPECT_EQ(resources.next_stream_to_memory_channel, 1u);
  EXPECT_EQ(resources.next_buffer_descriptor, 4u);
  EXPECT_EQ(resources.next_lock, 4u);
  EXPECT_EQ(resources.next_bank, 0u);
  for (uint32_t cursor : bank_cursors) {
    EXPECT_EQ(cursor, 64u);
  }
}

TEST(Aie2pArrayChannelResourcesTest, LoopbackMismatchRejectsWithoutMutation) {
  uint32_t bank_cursors[4] = {0};
  loom_aie2p_array_tile_resources_t resources = MakeResources(bank_cursors);
  loom_aie2p_array_compute_endpoint_request_t request =
      MakeRequest(LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY);
  request.loopback_source_dma_channel = 1;
  request.flags =
      LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_REQUEST_FLAG_REQUIRE_LOOPBACK;
  loom_aie2p_array_compute_endpoint_proposal_t proposal = {};

  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_compute(
                &resources, /*predecessor=*/nullptr, &request, &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_LOOPBACK);
  EXPECT_EQ(resources.next_stream_to_memory_channel, 0u);
  EXPECT_EQ(resources.next_buffer_descriptor, 0u);
  EXPECT_EQ(resources.next_lock, 0u);
  for (uint32_t cursor : bank_cursors) {
    EXPECT_EQ(cursor, 0u);
  }
}

TEST(Aie2pArrayChannelResourcesTest, NeighborReportsOwningResource) {
  uint32_t bank_cursors[4] = {0};
  loom_aie2p_array_tile_resources_t resources = MakeResources(bank_cursors);
  loom_aie2p_array_ring_resource_proposal_t proposal = {};

  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_neighbor(
                &resources, {0, 2}, ComputeFacts()->memory.local_load_base,
                /*record_count=*/2,
                /*record_byte_length=*/ComputeFacts()->memory.local_capacity,
                &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_RING_STORAGE);
  EXPECT_EQ(resources.next_lock, 0u);
  for (uint32_t cursor : bank_cursors) {
    EXPECT_EQ(cursor, 0u);
  }

  resources.next_lock = ComputeFacts()->lock_count - 1u;
  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_neighbor(
                &resources, {0, 2}, ComputeFacts()->memory.local_load_base,
                /*record_count=*/1, /*record_byte_length=*/64, &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_LOCK_PAIR);
  EXPECT_EQ(resources.next_lock, ComputeFacts()->lock_count - 1u);
  for (uint32_t cursor : bank_cursors) {
    EXPECT_EQ(cursor, 0u);
  }
}

TEST(Aie2pArrayChannelResourcesTest, ShimProposalRetainsExactTransition) {
  loom_aie2p_array_tile_resources_t resources = {
      .facts = ShimFacts(),
      .next_buffer_descriptor = 7,
      .next_memory_to_stream_channel = 1};
  loom_aie2p_array_shim_endpoint_proposal_t proposal = {};

  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_shim(
                &resources, {3, 0}, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                /*descriptor_count=*/2, &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE);
  EXPECT_EQ(resources.next_memory_to_stream_channel, 1u);
  EXPECT_EQ(resources.next_stream_to_memory_channel, 0u);
  EXPECT_EQ(resources.next_buffer_descriptor, 7u);
  EXPECT_EQ(proposal.coordinate.column, 3u);
  EXPECT_EQ(proposal.coordinate.row, 0u);
  EXPECT_EQ(proposal.dma_channel, 1u);
  EXPECT_EQ(proposal.buffer_descriptor_start, 7u);
  EXPECT_EQ(proposal.buffer_descriptor_count, 2u);

  loom_aie2p_array_channel_resources_commit_shim(&proposal, &resources);
  EXPECT_EQ(resources.next_memory_to_stream_channel, 2u);
  EXPECT_EQ(resources.next_stream_to_memory_channel, 0u);
  EXPECT_EQ(resources.next_buffer_descriptor, 9u);
}

TEST(Aie2pArrayChannelResourcesTest, ShimProposalRejectsWithoutMutation) {
  loom_aie2p_array_tile_resources_t resources =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  resources.facts = ShimFacts();
  resources.next_memory_to_stream_channel =
      ShimFacts()->dma.channel_count_per_direction;
  loom_aie2p_array_shim_endpoint_proposal_t proposal = {};

  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_shim(
                &resources, {0, 0}, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                /*descriptor_count=*/1, &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_CHANNEL);
  EXPECT_EQ(resources.next_memory_to_stream_channel,
            ShimFacts()->dma.channel_count_per_direction);
  EXPECT_EQ(resources.next_buffer_descriptor, 0u);

  resources.next_memory_to_stream_channel = 0;
  resources.next_buffer_descriptor =
      ShimFacts()->dma.buffer_descriptor_count - 1u;
  EXPECT_EQ(loom_aie2p_array_channel_resources_propose_shim(
                &resources, {0, 0}, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                /*descriptor_count=*/2, &proposal),
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_DESCRIPTORS);
  EXPECT_EQ(resources.next_memory_to_stream_channel, 0u);
  EXPECT_EQ(resources.next_buffer_descriptor,
            ShimFacts()->dma.buffer_descriptor_count - 1u);
}

}  // namespace
