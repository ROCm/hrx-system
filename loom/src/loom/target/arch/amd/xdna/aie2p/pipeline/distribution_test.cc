// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/distribution.h"

#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

namespace {

class DistributionReuseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Binding occurrences may have the same source value but still own
    // distinct storage and synchronization. Identity is the retained object.
    for (auto& identity : identities_) {
      identity.value_id = 17;
    }
    destinations_[0][0].channel = &identities_[0];
    destinations_[0][1].channel = &identities_[1];
    destinations_[1][0].channel = &identities_[2];
    destinations_[1][1].channel = &identities_[1];
    for (size_t i = 0; i < 2; ++i) {
      records_[i].destinations = destinations_[i];
      records_[i].destination_count = 2;
      records_[i].admissions = 1;
      records_[i].transfers = 2;
      batches_[i].records = &records_[i];
      batches_[i].record_count = 1;
      batches_[i].replays = 2;
      distributions_[i].source = &batches_[i];
      distributions_[i].transport = &transport_;
      distributions_[i].sender.channel = &source_;
      distributions_[i].receiver_count = 2;
      distributions_[i].repetitions = 2;
      distributions_[i].path = &path_;
    }
    path_.first = &distributions_[0];
    distributions_[0].next = &distributions_[1];
    owner_.distributions = &distributions_[0];
  }

  auto* Reuse(const loom_pipeline_transport_batch_t& batch,
              uint32_t repetitions = 2) {
    return loom_aie2p_native_reusable_distribution(
        &owner_, &source_, &transport_, &batch, repetitions);
  }

  // Distinct receiver protocols despite equal source value IDs.
  loom_channel_identity_t identities_[3] = {};
  // Two destination sets on one already admitted physical route.
  loom_pipeline_transport_destination_t destinations_[2][2] = {};
  // Canonical cohort facts for each descriptor geometry.
  loom_pipeline_transport_cohort_t records_[2] = {};
  // Retained source lifetimes consumed by geometry selection.
  loom_pipeline_transport_batch_t batches_[2] = {};
  // Shared transport facts; these uniform batches need no loop table.
  loom_pipeline_transport_t transport_ = {};
  // Native source protocol, shared across phases.
  loom_aie2p_native_channel_t source_ = {};
  // Two descriptor sets whose source and transfer count are identical.
  loom_aie2p_native_distribution_t distributions_[2] = {};
  // Physical route shared by the two descriptor geometries.
  loom_aie2p_native_distribution_path_t path_ = {};
  // Ordered transport owner retaining both geometries.
  loom_aie2p_native_worker_t owner_ = {};
};

TEST_F(DistributionReuseTest,
       ReturningToEachDestinationSetSelectsItsDescriptors) {
  EXPECT_EQ(Reuse(batches_[0]), &distributions_[0]);
  EXPECT_EQ(Reuse(batches_[1]), &distributions_[1]);
  EXPECT_EQ(Reuse(batches_[0]), &distributions_[0]);
  // A later occurrence has its own cohort storage but the same protocols.
  auto records = records_[1];
  auto batch = batches_[1];
  batch.records = &records;
  EXPECT_EQ(Reuse(batch), &distributions_[1]);
}

TEST_F(DistributionReuseTest, ChangedTransferCountNeedsDifferentDescriptors) {
  auto records = records_[0];
  records.transfers = 4;
  auto batch = batches_[0];
  batch.records = &records;
  batch.replays = 4;
  EXPECT_EQ(Reuse(batch, 4), nullptr);
  EXPECT_EQ(Reuse(batches_[0]), &distributions_[0]);
}

TEST_F(DistributionReuseTest, SourceStorageIdentityIsPartOfGeometry) {
  auto another_source = source_;
  EXPECT_EQ(loom_aie2p_native_reusable_distribution(
                &owner_, &another_source, &transport_, &batches_[0], 2),
            nullptr);
}

TEST_F(DistributionReuseTest, ReceiverCountAndOrderArePartOfCircuitGeometry) {
  auto records = records_[0];
  auto batch = batches_[0];
  batch.records = &records;
  records.destination_count = 1;
  EXPECT_EQ(Reuse(batch), nullptr);
  records.destination_count = 2;
  loom_pipeline_transport_destination_t reversed[] = {destinations_[0][1],
                                                      destinations_[0][0]};
  records.destinations = reversed;
  EXPECT_EQ(Reuse(batch), nullptr);
}

TEST_F(DistributionReuseTest, NewOwnerHasNoReusableGeometry) {
  owner_.distributions = nullptr;
  EXPECT_EQ(Reuse(batches_[0]), nullptr);
}

TEST_F(DistributionReuseTest, PacketReusePreservesCopyOrderAndRepetition) {
  owner_.distribution_packets = true;
  owner_.distributions->next = nullptr;
  distributions_[0].repetitions = 0;
  loom_pipeline_transport_copy_t copies[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    copies[i].source = &records_[0];
    copies[i].destination = &destinations_[0][i];
    copies[i].executions = 2;
  }
  batches_[0].copies = copies;
  batches_[0].copy_count = 2;
  auto batch = batches_[0];
  loom_pipeline_transport_copy_t reordered[] = {copies[0], copies[1]};
  batch.copies = reordered;
  EXPECT_EQ(Reuse(batch), &distributions_[0]);
  std::swap(reordered[0], reordered[1]);
  EXPECT_EQ(Reuse(batch), nullptr);
  std::swap(reordered[0], reordered[1]);
  auto records = records_[0];
  records.transfers = 4;
  batch.records = &records;
  batch.replays = 4;
  for (auto& copy : reordered) {
    copy.source = &records;
    copy.executions = 4;
  }
  EXPECT_EQ(Reuse(batch, 4), nullptr);
}

// Common execution has admitted three serial children. Each has one source
// controller, two receiver controllers, and two independent transport strands.
// Physical storage may alias; channel identities and ownership remain distinct.
class DistributionPathTest
    : public ::testing::TestWithParam<std::tuple<uint8_t, uint8_t>> {
 protected:
  static constexpr size_t kChildren = 3;
  static constexpr size_t kStreams = kChildren * 2;
  static constexpr size_t kWorkers = 1 + kChildren * 5;
  static constexpr size_t kChannels = kStreams * 3;

  static size_t Producer(size_t stream) { return 1 + (stream / 2) * 5; }
  static size_t Worker(size_t stream) {
    return Producer(stream) + 3 + stream % 2;
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    const size_t tile_count =
        context_.family->column_count * context_.family->row_count;
    tiles_.resize(tile_count);
    context_.tiles = tiles_.data();
    for (size_t i = 0; i < tile_count; ++i) {
      auto& tile = tiles_[i];
      tile.coordinate = {static_cast<uint8_t>(i / context_.family->row_count),
                         static_cast<uint8_t>(i % context_.family->row_count)};
      tile.facts = loom_xdna_array_tile_facts(context_.family, tile.coordinate);
      tile.memory_layout = tile.coordinate.row == 1
                               ? &loom_aie2p_sram_memory
                               : &loom_aie2p_compute_memory;
    }
    pools_[0] = &tiles_[std::get<0>(GetParam())];
    pools_[1] = &tiles_[context_.family->row_count + std::get<1>(GetParam())];
    pools_[2] = &tiles_[context_.family->row_count + 3];
    context_.pool_tiles = pools_.data();
    context_.channels = channels_.data();
    context_.workers = workers_.data();
    context_.core_engines = core_engines_.data();
    context_.distribution_engines = distribution_engines_.data();
    context_.distribution_execution.workers = representatives_.data();
    context_.distribution_execution.engine_count = kStreams;
    realization_.workers = source_workers_.data();
    auto& invocations = realization_.invocations;
    invocations.entries = entries_.data();
    invocations.count = entries_.size();
    invocations.channels = channel_owners_.data();
    invocations.channel_count = kChannels;
    invocations.channel_order = channel_order_.data();
    invocations.workers = worker_owners_.data();
    invocations.worker_count = kWorkers;
    invocations.lifetimes.has_serial = true;
    for (size_t i = 0; i < entries_.size(); ++i) {
      constructions_[i].ordinal = i;
      entries_[i] = &constructions_[i];
    }
    core_engines_.fill(IREE_HOST_SIZE_MAX);
    distribution_engines_.fill(IREE_HOST_SIZE_MAX);
    for (size_t i = 0; i < kWorkers; ++i) {
      worker_owners_[i].invocation = &constructions_[i ? 1 + (i - 1) / 5 : 0];
    }
    for (size_t i = 0; i < kStreams; ++i) {
      representatives_[i] = Worker(i);
      distribution_engines_[Worker(i)] = i;
      workers_[Worker(i)].execution = LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION;
      source_workers_[Worker(i)].source = &strands_[i];
      for (size_t j = 0; j < 3; ++j) {
        const size_t controller = Producer(i) + j;
        workers_[controller].execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE;
        core_engines_[controller] = j ? context_.family->row_count + 1 + j : 2;
        const size_t index = i * 3 + j;
        channel_order_[index] = index;
        sources_[index].capacity = 1;
        channels_[index].source = &sources_[index];
        channels_[index].pool_index = j;
        channels_[index].byte_offset = (i % 2) * 256;
        channels_[index].record_byte_length = i / 2 == 1 ? 32 : 64;
        channels_[index].byte_stride = channels_[index].record_byte_length;
        channels_[index].record_dense = true;
        channels_[index].free_lock = (i % 2) * 2;
        channels_[index].ready_lock = (i % 2) * 2 + 1;
        channels_[index].cursor.writer.worker = j ? Worker(i) : controller;
        channels_[index].cursor.reader.worker = j ? controller : Worker(i);
        channel_owners_[index].source = &sources_[index];
        channel_owners_[index].invocation = &constructions_[1 + i / 2];
      }
      admissions_[i].channel = &sources_[i * 3].identity;
      cohorts_[i].admission = &admissions_[i];
      cohorts_[i].destinations = destinations_[i].data();
      cohorts_[i].admission_scope = UINT16_MAX;
      cohorts_[i].copy_scope = UINT16_MAX;
      for (size_t j = 0; j < 3; ++j) {
        destinations_[i][j].channel = &sources_[i * 3 + 1 + j % 2].identity;
        copies_[i][j].source = &cohorts_[i];
        copies_[i][j].destination = &destinations_[i][j];
        copies_[i][j].executions = 1;
        copies_[i][j].scope = UINT16_MAX;
      }
      batches_[i].records = &cohorts_[i];
      batches_[i].record_count = 1;
      batches_[i].copies = copies_[i].data();
      batches_[i].replays = 1;
      source_workers_[Worker(i)].transport.batches = &batches_[i];
      source_workers_[Worker(i)].transport.batch_count = 1;
    }
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        context_.family, 128, routes_.size(), &arena_, &context_.routing));
    context_.routes = routes_.data();
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  const loom_aie2p_native_distribution_t* Form(size_t stream,
                                               uint32_t records = 2,
                                               size_t receivers = 1,
                                               bool packet = false,
                                               bool repeat_first = false) {
    workers_[Worker(stream)].distribution_packets = packet;
    auto& cohort = cohorts_[stream];
    cohort.destination_count = receivers + (repeat_first ? 1 : 0);
    cohort.admissions = packet ? 1 : records;
    cohort.transfers = packet ? 1 : records;
    cohort.release = packet ? &release_ : nullptr;
    batches_[stream].copy_count = cohort.destination_count;
    batches_[stream].replays = packet ? 0 : 1;
    for (size_t i = 0; i < cohort.destination_count; ++i) {
      copies_[stream][i].executions = packet ? 1 : records;
    }
    for (size_t i = 0; i <= receivers; ++i) {
      auto& cursor = i ? channels_[stream * 3 + i].cursor.reader
                       : channels_[stream * 3].cursor.reader;
      cursor.maximum_admissions = packet ? 1 : records;
      cursor.exact_admissions = true;
    }
    bool valid = false;
    IREE_EXPECT_OK(loom_aie2p_native_form_worker_distributions(
        &context_, &realization_, Worker(stream), &valid));
    EXPECT_TRUE(valid);
    return workers_[Worker(stream)].distribution_plan->batches[0].distribution;
  }

  const loom_aie2p_native_distribution_t* SelectPrepared(size_t stream) {
    bool valid = false;
    IREE_EXPECT_OK(loom_aie2p_native_select_worker_distributions(
        &context_, &realization_, Worker(stream), &next_source_, &valid));
    EXPECT_TRUE(valid);
    return workers_[Worker(stream)].distributions;
  }

  const loom_aie2p_native_distribution_t* Select(size_t stream,
                                                 uint32_t records = 2,
                                                 size_t receivers = 1,
                                                 bool packet = false,
                                                 bool repeat_first = false) {
    Form(stream, records, receivers, packet, repeat_first);
    return SelectPrepared(stream);
  }

  void CheckCompleteDemandFormation(bool packet) {
    for (size_t i = 0; i < kStreams; ++i) {
      const auto* distribution = Form(i, 2, 2, packet);
      ASSERT_NE(distribution, nullptr);
      EXPECT_EQ(distribution->sender.channel, &channels_[i * 3]);
      EXPECT_EQ(distribution->sender.tile, pools_[0]);
      ASSERT_EQ(distribution->receiver_count, 2u);
      for (size_t j = 0; j < 2; ++j) {
        EXPECT_EQ(distribution->receivers[j].channel,
                  &channels_[i * 3 + j + 1]);
        EXPECT_EQ(distribution->receivers[j].tile, pools_[j + 1]);
      }
    }
    EXPECT_EQ(context_.routing.route_count, 0u);
    for (const auto* tile : pools_) {
      EXPECT_EQ(tile->next_mm2s, 0);
      EXPECT_EQ(tile->next_s2mm, 0);
      EXPECT_EQ(tile->descriptor_counts[0], 0);
      EXPECT_EQ(tile->descriptor_counts[1], 0);
    }
    for (size_t i = 0; i < kStreams; ++i) {
      EXPECT_NE(SelectPrepared(i), nullptr);
    }
    EXPECT_EQ(pools_[0]->next_mm2s, 2);
    EXPECT_EQ(pools_[1]->next_s2mm, 2);
    EXPECT_EQ(pools_[2]->next_s2mm, 2);
    EXPECT_GT(context_.routing.route_count, 0u);
  }

  // Selection-owned temporary memory.
  iree_arena_block_pool_t blocks_ = {};
  // Arena holding native paths, geometry, and routes.
  iree_arena_allocator_t arena_ = {};
  // Compiler ownership of that arena.
  loom_pass_t pass_ = {};
  // Native selection state under test.
  loom_aie2p_native_context_t context_ = {};
  // Common retained resource and occurrence inventory.
  loom_pipeline_realization_t realization_ = {};
  // Complete physical coordinate domain.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Source and two receiver memory locations.
  std::array<loom_aie2p_native_tile_t*, 3> pools_ = {};
  // Root and three serial construction occurrences.
  std::array<loom_pipeline_invocation_t, kChildren + 1> constructions_ = {};
  // Common construction index.
  std::array<const loom_pipeline_invocation_t*, kChildren + 1> entries_ = {};
  // Admitted instruction workers and communication strands.
  std::array<loom_aie2p_native_worker_t, kWorkers> workers_ = {};
  // Corresponding retained common workers.
  std::array<loom_pipeline_worker_t, kWorkers> source_workers_ = {};
  // Construction ownership of all workers.
  std::array<loom_pipeline_invocation_worker_t, kWorkers> worker_owners_ = {};
  // Already-admitted physical instruction engines.
  std::array<iree_host_size_t, kWorkers> core_engines_ = {};
  // Independent logical distribution engines.
  std::array<iree_host_size_t, kWorkers> distribution_engines_ = {};
  // Their representative transport workers.
  std::array<iree_host_size_t, kStreams> representatives_ = {};
  // Transport declarations referenced by retained plans.
  std::array<loom_pipeline_resource_strand_t, kStreams> strands_ = {};
  // Logical channel sources, in identity-address order.
  std::array<loom_pipeline_resource_channel_t, kChannels> sources_ = {};
  // Selected storage and cursor facts for those channels.
  std::array<loom_aie2p_native_channel_t, kChannels> channels_ = {};
  // Common construction ownership of every channel.
  std::array<loom_pipeline_invocation_channel_t, kChannels> channel_owners_ =
      {};
  // Identity-address index into the common channel table.
  std::array<iree_host_size_t, kChannels> channel_order_ = {};
  // Source admission action for each stream.
  std::array<loom_channel_plan_action_t, kStreams> admissions_ = {};
  // Source-record ownership intervals.
  std::array<loom_pipeline_transport_cohort_t, kStreams> cohorts_ = {};
  // Structural destination sites, including an optional repeated first
  // receiver.
  std::array<std::array<loom_pipeline_transport_destination_t, 3>, kStreams>
      destinations_ = {};
  // Retained copy visits through those sites.
  std::array<std::array<loom_pipeline_transport_copy_t, 3>, kStreams> copies_ =
      {};
  // Complete finite transfer programs.
  std::array<loom_pipeline_transport_batch_t, kStreams> batches_ = {};
  // Retained release identity for packet source ownership.
  loom_op_t release_ = {};
  // Physical route annotations selected alongside the route builder.
  std::array<loom_aie2p_native_route_t, 512> routes_ = {};
  // Next physical source identity in that builder.
  uint32_t next_source_ = 0;
};

TEST_P(DistributionPathTest, FreshChannelsReuseRetiredQueuesAndRoutes) {
  const auto* first = Select(0, 4);
  ASSERT_NE(first, nullptr);
  const auto route_count = context_.routing.route_count;
  const auto* tail = Select(2, 2);
  const auto* final = Select(4, 6);
  ASSERT_NE(tail, nullptr);
  ASSERT_NE(final, nullptr);
  EXPECT_EQ(first->path, tail->path);
  EXPECT_EQ(first->path, final->path);
  EXPECT_NE(first->sender.channel, tail->sender.channel);
  EXPECT_NE(first->receivers[0].channel, tail->receivers[0].channel);
  EXPECT_EQ(context_.routing.route_count, route_count);
  EXPECT_EQ(pools_[0]->next_mm2s, 1);
  EXPECT_EQ(pools_[1]->next_s2mm, 1);
  EXPECT_EQ(first->repetitions, 4u);
  EXPECT_EQ(tail->repetitions, 2u);
  EXPECT_NE(first->sender.sequence.words[0][0],
            tail->sender.sequence.words[0][0]);
}

TEST_P(DistributionPathTest, SimultaneousStreamsKeepIndependentPaths) {
  const auto* first = Select(0);
  const auto* second = Select(1);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first->path, second->path);
  const auto* later_first = Select(2);
  const auto* later_second = Select(3);
  ASSERT_NE(later_first, nullptr);
  ASSERT_NE(later_second, nullptr);
  EXPECT_NE(later_first->path, later_second->path);
  EXPECT_EQ(first->path, later_first->path);
  EXPECT_EQ(second->path, later_second->path);
  EXPECT_EQ(pools_[0]->next_mm2s, 2);
  EXPECT_EQ(pools_[1]->next_s2mm, 2);
}

TEST_P(DistributionPathTest, CircuitDemandPrecedesPhysicalAllocation) {
  CheckCompleteDemandFormation(false);
}

TEST_P(DistributionPathTest, PacketDemandPrecedesPhysicalAllocation) {
  CheckCompleteDemandFormation(true);
}

TEST_P(DistributionPathTest, PacketItinerariesRetainNewDestinationClasses) {
  const auto* first = Form(0, 1, 2, true);
  const auto* second = Form(2, 1, 2, true, true);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(first->path, second->path);
  ASSERT_EQ(first->path->packet_count, 2);
  const auto& first_program =
      workers_[Worker(0)].distribution_plan->batches[0].packet_program;
  const auto& second_program =
      workers_[Worker(2)].distribution_plan->batches[0].packet_program;
  ASSERT_EQ(first_program.range_count, 1);
  EXPECT_EQ(first_program.task_count, 1);
  EXPECT_EQ(first_program.first_packet, 0);
  EXPECT_EQ(first_program.packet_count, 1);
  EXPECT_EQ(first_program.ranges[0].first, 0u);
  EXPECT_EQ(first_program.ranges[0].count, 1u);
  EXPECT_EQ(first_program.ranges[0].packet, 0);
  ASSERT_EQ(second_program.range_count, 2);
  EXPECT_EQ(second_program.first_packet, 1);
  EXPECT_EQ(second_program.packet_count, 1);
  EXPECT_EQ(second_program.ranges[0].packet, 0);
  EXPECT_EQ(second_program.ranges[1].packet, 1);
  EXPECT_EQ(context_.routing.route_count, 0u);
  EXPECT_EQ(pools_[0]->next_mm2s, 0);
  EXPECT_EQ(pools_[0]->descriptor_counts[0], 0);

  ASSERT_NE(SelectPrepared(0), nullptr);
  const auto first_routes = context_.routing.route_count;
  EXPECT_GT(first_routes, 0u);
  ASSERT_NE(SelectPrepared(2), nullptr);
  EXPECT_GT(context_.routing.route_count, first_routes);
  EXPECT_EQ(first->sender.sequence.descriptor_count, 1);
  EXPECT_EQ(second->sender.sequence.descriptor_count, 2);
}

TEST_P(DistributionPathTest, EarlierSelectionSweepCannotReuseItsOwnPath) {
  const auto* first = Select(0);
  const auto* later = Select(2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(later, nullptr);
  EXPECT_EQ(first->path, later->path);
  const auto* concurrent = Select(1);
  ASSERT_NE(concurrent, nullptr);
  EXPECT_NE(first->path, concurrent->path);
}

TEST_P(DistributionPathTest, ImportedEndpointsKeepResidentPaths) {
  const auto* first = Select(0);
  channel_owners_[2 * 3 + 1].invocation = &constructions_[0];
  const auto* imported = Select(2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(imported, nullptr);
  EXPECT_NE(first->path, imported->path);
}

TEST_P(DistributionPathTest, IndependentControllersKeepIndependentPaths) {
  const auto* first = Select(0);
  core_engines_[Producer(2)] = 3;
  const auto* independent = Select(2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(independent, nullptr);
  EXPECT_NE(first->path, independent->path);
}

TEST_P(DistributionPathTest, DifferentDestinationsKeepIndependentRoutes) {
  const auto* first = Select(0);
  channels_[7].pool_index = 2;
  const auto* different = Select(2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(different, nullptr);
  EXPECT_NE(first->path, different->path);
}

TEST_P(DistributionPathTest, CircuitAndPacketProgramsKeepSeparateFraming) {
  const auto* circuit = Select(0, 1);
  const auto* packet = Select(2, 1, 1, true);
  ASSERT_NE(circuit, nullptr);
  ASSERT_NE(packet, nullptr);
  EXPECT_NE(circuit->path, packet->path);
}

TEST_P(DistributionPathTest, PacketClassesNameReceiversAcrossFreshChannels) {
  const auto* first = Select(0, 1, 2, true);
  ASSERT_NE(first, nullptr);
  const auto route_count = context_.routing.route_count;
  const auto* second = Select(2, 1, 2, true);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->path, second->path);
  EXPECT_EQ(first->path->packet_count, 1);
  EXPECT_EQ(context_.routing.route_count, route_count);
  const auto* selective = Select(4, 1, 2, true, true);
  ASSERT_NE(selective, nullptr);
  EXPECT_EQ(first->path, selective->path);
  EXPECT_EQ(first->path->packet_count, 2);
  EXPECT_EQ(selective->sender.sequence.descriptor_count, 2);
  EXPECT_EQ(pools_[0]->next_mm2s, 1);
  EXPECT_EQ(pools_[1]->next_s2mm, 1);
  EXPECT_EQ(pools_[2]->next_s2mm, 1);
}

TEST_P(DistributionPathTest, EveryGeometryOwnsItsReceiverGrant) {
  const auto* first = Select(0, 4);
  const auto* second = Select(2, 2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  workers_[Producer(0)].channels.remote_count = 1;
  IREE_ASSERT_OK(loom_aie2p_native_prepare_sharing(&context_, &realization_));
  auto* delivery = context_.deliveries[pools_[1] - context_.tiles];
  ASSERT_NE(delivery, nullptr);
  loom_aie2p_native_commit_sharing(&context_);
  EXPECT_EQ(channels_[1].grant, nullptr);
  EXPECT_EQ(channels_[7].grant, nullptr);
  delivery->sharing.used = true;
  loom_aie2p_native_commit_sharing(&context_);
  ASSERT_NE(channels_[1].grant, nullptr);
  ASSERT_NE(channels_[7].grant, nullptr);
  EXPECT_NE(channels_[1].grant, channels_[7].grant);
  EXPECT_EQ(channels_[1].grant->distribution, first);
  EXPECT_EQ(channels_[7].grant->distribution, second);
  EXPECT_EQ(first->grant_reader, Producer(0) + 1);
  EXPECT_EQ(second->grant_reader, Producer(2) + 1);
  const auto* layout = pools_[0]->memory_layout;
  for (const auto* geometry : {first, second}) {
    EXPECT_EQ(geometry->sender.sequence.task_count, 1);
    EXPECT_EQ(geometry->sender.sequence.descriptor_count, 1);
    EXPECT_EQ(geometry->sender.sequence.words[0][layout->packet.word] &
                  loom_xdna_register_field_encode(layout->packet.id, -1),
              loom_xdna_register_field_encode(
                  layout->packet.id, LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID));
  }
  for (size_t i = first->path->routes.first;
       i < first->path->routes.first + first->path->routes.count; ++i) {
    EXPECT_EQ(routes_[i].packet, LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID);
  }
}

TEST_P(DistributionPathTest, EveryGeometryMustSupportSharedFraming) {
  const auto* first = Select(0, 4);
  sources_[6].capacity = 2;
  const auto* second = Select(2, 2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(first->path, second->path);
  workers_[Producer(0)].channels.remote_count = 1;
  IREE_ASSERT_OK(loom_aie2p_native_prepare_sharing(&context_, &realization_));
  EXPECT_EQ(context_.deliveries[pools_[1] - context_.tiles], nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    MemoryModules, DistributionPathTest,
    ::testing::Combine(::testing::Values(uint8_t{1}, uint8_t{2}),
                       ::testing::Values(uint8_t{1}, uint8_t{2})));

}  // namespace
