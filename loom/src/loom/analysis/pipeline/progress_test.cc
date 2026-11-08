// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/progress.h"

#include <initializer_list>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class PipelineProgressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    graph_.channels = channels_;
    graph_.channel_count = 4;
    graph_.channel_order = order_;
    for (size_t i = 0; i < 4; ++i) {
      channels_[i].source = &resources_[i];
      endpoints_[i].reads = endpoints_[i].writes = 8;
    }
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  void Copy(uint32_t source, std::initializer_list<uint32_t> destinations) {
    const size_t index = graph_.worker_count++;
    admissions_[index].channel = &resources_[source].identity;
    auto& cohort = cohorts_[index];
    cohort.admission = &admissions_[index];
    cohort.admissions = cohort.transfers = endpoints_[source].reads;
    cohort.destinations = destinations_[index];
    for (uint32_t destination : destinations) {
      destinations_[index][cohort.destination_count++].channel =
          &resources_[destination].identity;
    }
    batches_[index].records = &cohort;
    batches_[index].record_count = 1;
    auto& transport = workers_[index].transport;
    transport.cohorts = &cohort;
    transport.cohort_count = 1;
    transport.batches = &batches_[index];
    transport.batch_count = 1;
  }
  void Expect(std::initializer_list<uint32_t> expected) {
    const loom_pipeline_progress_callback_t callback = {
        .cost =
            [](void* user_data, uint32_t source, uint32_t reader) {
              return static_cast<PipelineProgressTest*>(user_data)
                  ->costs_[source][reader];
            },
        .user_data = this,
    };
    uint32_t* readers = nullptr;
    IREE_ASSERT_OK(loom_pipeline_progress_select(&graph_, workers_, endpoints_,
                                                 callback, &arena_, &readers));
    ASSERT_EQ(expected.size(), graph_.channel_count);
    size_t index = 0;
    for (uint32_t reader : expected) {
      EXPECT_EQ(readers[index++], reader);
    }
  }
  // Arena backing the selection result, not any source program.
  iree_arena_block_pool_t pool_ = {};
  // One analysis invocation's allocation lifetime.
  iree_arena_allocator_t arena_ = {};
  // Four distinct canonical protocol identities in address order.
  loom_pipeline_resource_channel_t resources_[4] = {};
  // Complete inventory references used by canonical lookup.
  loom_pipeline_invocation_channel_t channels_[4] = {};
  // Index of the sorted contiguous identity array.
  const iree_host_size_t order_[4] = {0, 1, 2, 3};
  // Static channel and transport inventory supplied by the owning analysis.
  loom_pipeline_invocations_t graph_ = {};
  // Physical owners' exact endpoint counts and continuation eligibility.
  loom_pipeline_progress_endpoint_t endpoints_[4] = {};
  // Retained copy workers; ordinary reader bodies are not needed by selection.
  loom_pipeline_worker_t workers_[4] = {};
  // Source admissions retaining protocol identity.
  loom_channel_plan_action_t admissions_[4] = {};
  // Destination identities retained per source copy site.
  loom_pipeline_transport_destination_t destinations_[4][3] = {};
  // Equal-record copy facts, one cohort per transport worker.
  loom_pipeline_transport_cohort_t cohorts_[4] = {};
  // Source lifetime boundaries consumed by progress selection.
  loom_pipeline_transport_batch_t batches_[4] = {};
  // Placement costs supplied by the target, independent of source discovery.
  uint64_t costs_[4][4] = {};
};

TEST_F(PipelineProgressTest, EmptyInventoryHasNoObservers) {
  loom_pipeline_invocations_t empty = {};
  uint32_t* readers = nullptr;
  IREE_ASSERT_OK(loom_pipeline_progress_select(&empty, nullptr, nullptr, {},
                                               &arena_, &readers));
  EXPECT_EQ(readers, nullptr);
}

TEST_F(PipelineProgressTest, DownstreamReadProvesEachEqualOrdinalUpstream) {
  Copy(0, {1});
  Copy(1, {2});
  endpoints_[2].observer = true;
  Expect({2, 2, 2, UINT32_MAX});
  // Record counts affect eligibility, not the number of graph nodes. A long
  // epoch has exactly the same four-channel observer mapping.
  for (auto& endpoint : endpoints_) {
    endpoint.reads = endpoint.writes = UINT32_MAX;
  }
  for (size_t i = 0; i < graph_.worker_count; ++i) {
    cohorts_[i].admissions = cohorts_[i].transfers = UINT32_MAX;
  }
  Expect({2, 2, 2, UINT32_MAX});
}

TEST_F(PipelineProgressTest, FanoutUsesEligibleCostThenStableOrdinal) {
  Copy(0, {2, 1});
  endpoints_[1].observer = endpoints_[2].observer = true;
  Expect({1, 1, 2, UINT32_MAX});
  costs_[0][1] = 5;
  Expect({2, 1, 2, UINT32_MAX});
  endpoints_[2].observer = false;
  Expect({1, 1, UINT32_MAX, UINT32_MAX});
}

TEST_F(PipelineProgressTest, DifferentRecordCountsCannotEstablishProgress) {
  Copy(0, {1});
  Copy(1, {2});
  endpoints_[2].observer = true;
  endpoints_[1].writes = 9;
  Expect({UINT32_MAX, 2, 2, UINT32_MAX});
  endpoints_[1].writes = 8;
  endpoints_[2].reads = 7;
  Expect({UINT32_MAX, UINT32_MAX, 2, UINT32_MAX});
}

TEST_F(PipelineProgressTest, ReplayedCopiesDoNotPreserveRecordOrdinals) {
  Copy(0, {1});
  const loom_op_t release = {};
  cohorts_[0].release = &release;
  cohorts_[0].transfers = 16;
  endpoints_[1].reads = endpoints_[1].writes = 16;
  endpoints_[1].observer = true;
  Expect({UINT32_MAX, 1, UINT32_MAX, UINT32_MAX});
}

TEST_F(PipelineProgressTest, CyclesNeedDistanceProofEvenWithLocalObservers) {
  Copy(0, {1});
  Copy(1, {0});
  for (auto& endpoint : endpoints_) {
    endpoint.observer = true;
  }
  Expect({UINT32_MAX, UINT32_MAX, 2, 3});
}

}  // namespace
