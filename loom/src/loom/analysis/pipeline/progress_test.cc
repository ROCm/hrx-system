// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/progress.h"

#include <deque>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// Source analysis has already established these transport cohorts and endpoint
// counts. The progress owner consumes their identities and cardinalities, not
// source operations, storage addresses, or an expanded list of record visits.
class ProgressGraph {
 public:
  ProgressGraph(size_t channel_count, uint64_t records)
      : channels_(channel_count),
        occurrences_(channel_count),
        channel_order_(channel_count),
        endpoints_(channel_count),
        preferred_(channel_count, UINT32_MAX) {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    for (size_t i = 0; i < channel_count; ++i) {
      occurrences_[i].source = &channels_[i];
      channel_order_[i] = i;
      endpoints_[i] = {.reads = records, .writes = records};
    }
  }

  ~ProgressGraph() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void Observe(uint32_t channel, bool eligible = true) {
    endpoints_[channel].observer = eligible;
  }

  void Prefer(uint32_t source, uint32_t observer) {
    preferred_[source] = observer;
  }

  void SetReads(uint32_t channel, uint64_t records) {
    endpoints_[channel].reads = records;
  }

  void SetWrites(uint32_t channel, uint64_t records) {
    endpoints_[channel].writes = records;
  }

  void Copy(uint32_t source, const std::vector<uint32_t>& destinations,
            bool retain_source = false) {
    copies_.emplace_back();
    CopyGroup& copy = copies_.back();
    copy.worker = copies_.size() - 1;
    copy.admission.channel = &channels_[source].identity;
    for (uint32_t destination : destinations) {
      copy.destinations.push_back({
          .channel = &channels_[destination].identity,
      });
    }
    copy.cohort = {
        .admission = &copy.admission,
        .destinations = copy.destinations.data(),
        .destination_count = copy.destinations.size(),
        .release = retain_source ? &copy.release : nullptr,
        .admissions = endpoints_[source].reads,
        .transfers = endpoints_[source].reads,
    };
    copy.batch = {.records = &copy.cohort, .record_count = 1};
  }

  void CopyPhases(uint32_t source, const std::vector<uint32_t>& destinations,
                  const std::vector<uint64_t>& records) {
    const size_t worker = copies_.size();
    for (uint64_t count : records) {
      Copy(source, destinations);
      copies_.back().worker = worker;
      copies_.back().cohort.admissions = count;
      copies_.back().cohort.transfers = count;
    }
  }

  void Select(bool retain_observations = true,
              std::vector<uint32_t> observation_sources = {}) {
    if (retain_observations && observation_sources.empty()) {
      for (uint32_t i = 0; i < channels_.size(); ++i) {
        observation_sources.push_back(i);
      }
    }
    std::vector<loom_pipeline_worker_t> workers(copies_.size());
    std::vector<std::vector<loom_pipeline_transport_cohort_t>> cohorts(
        workers.size());
    std::vector<std::vector<loom_pipeline_transport_batch_t>> batches(
        workers.size());
    for (const auto& copy : copies_) {
      cohorts[copy.worker].push_back(copy.cohort);
      batches[copy.worker].push_back(copy.batch);
    }
    for (size_t i = 0; i < workers.size(); ++i) {
      for (size_t j = 0; j < batches[i].size(); ++j) {
        batches[i][j].records = &cohorts[i][j];
      }
      workers[i].transport.cohorts = cohorts[i].data();
      workers[i].transport.cohort_count = cohorts[i].size();
      workers[i].transport.batches = batches[i].data();
      workers[i].transport.batch_count = batches[i].size();
    }
    const loom_pipeline_invocations_t invocations = {
        .worker_count = workers.size(),
        .channels = occurrences_.data(),
        .channel_count = channels_.size(),
        .channel_order = channel_order_.data(),
    };
    IREE_ASSERT_OK(loom_pipeline_progress_select(
        &invocations, workers.data(), endpoints_.data(),
        {ObservationCost, this}, observation_sources.data(),
        observation_sources.size(), &arena_, &readers_,
        retain_observations ? &observations_ : nullptr));
  }

  uint32_t Reader(uint32_t source) const { return readers_[source]; }

  bool HasReaders() const { return readers_ != nullptr; }

  std::vector<uint32_t> Observations(uint32_t source) const {
    const auto& row = observations_[source];
    if (!row.count) {
      return {};
    }
    return {row.channels, row.channels + row.count};
  }

  size_t Bytes() const { return arena_.used_allocation_size; }

 private:
  struct CopyGroup {
    // Physical transport owner; several phases may share this FIFO cursor.
    size_t worker = 0;
    // Source identity captured by the retained acquisition action.
    loom_channel_plan_action_t admission = {};
    // Optional independently retained source obligation.
    loom_op_t release = {};
    // Independently published fanout destination identities.
    std::vector<loom_pipeline_transport_destination_t> destinations;
    // Equal-record copy group established by the transport analysis.
    loom_pipeline_transport_cohort_t cohort = {};
    // Complete ownership interval of this one-record group.
    loom_pipeline_transport_batch_t batch = {};
  };

  static uint64_t ObservationCost(void* user_data, uint32_t source,
                                  uint32_t observer) {
    const auto& graph = *static_cast<const ProgressGraph*>(user_data);
    return graph.preferred_[source] == observer ? 0 : 1;
  }

  // Stable protocol identities, independent of their unused storage geometry.
  std::vector<loom_pipeline_resource_channel_t> channels_;
  // Canonical occurrence rows in source construction order.
  std::vector<loom_pipeline_invocation_channel_t> occurrences_;
  // Sorted identity index; channels_ is one fixed contiguous allocation.
  std::vector<iree_host_size_t> channel_order_;
  // Physical admission counts and continuation capabilities.
  std::vector<loom_pipeline_progress_endpoint_t> endpoints_;
  // Optional preferred observation per source for conflicting-cost cases.
  std::vector<uint32_t> preferred_;
  // Stable cohorts whose addresses are borrowed by the retained batch tables.
  std::deque<CopyGroup> copies_;
  // Arena backing for the production analysis result.
  iree_arena_block_pool_t blocks_;
  // Analysis allocation lifetime and byte accounting.
  iree_arena_allocator_t arena_;
  // Existing single-observer selection in canonical channel order.
  uint32_t* readers_ = nullptr;
  // Complete observation sets retained for joint phase selection.
  loom_pipeline_progress_observations_t* observations_ = nullptr;
};

TEST(PipelineProgressTest, EmptyInventoryHasNoObservers) {
  ProgressGraph graph(0, 0);
  graph.Select();
  EXPECT_FALSE(graph.HasReaders());
}

TEST(PipelineProgressTest, DownstreamReadProvesEachEqualOrdinalUpstream) {
  for (uint64_t records : std::initializer_list<uint64_t>{8, UINT32_MAX}) {
    ProgressGraph graph(4, records);
    graph.Copy(0, {1});
    graph.Copy(1, {2});
    graph.Observe(2);
    graph.Select(false);
    EXPECT_EQ(graph.Reader(0), 2u);
    EXPECT_EQ(graph.Reader(1), 2u);
    EXPECT_EQ(graph.Reader(2), 2u);
    EXPECT_EQ(graph.Reader(3), UINT32_MAX);
  }
}

TEST(PipelineProgressTest, FanoutUsesEligibleCostThenStableOrdinal) {
  ProgressGraph graph(4, 8);
  graph.Copy(0, {2, 1});
  graph.Observe(1);
  graph.Observe(2);
  graph.Select(false);
  EXPECT_EQ(graph.Reader(0), 1u);
  graph.Prefer(0, 2);
  graph.Select(false);
  EXPECT_EQ(graph.Reader(0), 2u);
  graph.Observe(2, false);
  graph.Select(false);
  EXPECT_EQ(graph.Reader(0), 1u);
}

TEST(PipelineProgressTest, DifferentWriteCountCannotEstablishProgress) {
  ProgressGraph graph(4, 8);
  graph.Copy(0, {1});
  graph.Copy(1, {2});
  graph.Observe(2);
  graph.SetWrites(1, 9);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_EQ(graph.Reader(1), 2u);
  EXPECT_TRUE(graph.Observations(0).empty());
}

TEST(PipelineProgressTest, RetainsAllObserversThroughNestedFanout) {
  ProgressGraph graph(6, 11);
  graph.Copy(0, {1, 2});
  graph.Copy(1, {3});
  graph.Copy(2, {4, 5});
  for (uint32_t channel : {3u, 4u, 5u}) {
    graph.Observe(channel);
  }
  graph.Select();
  EXPECT_EQ(graph.Observations(0), (std::vector<uint32_t>{3, 4, 5}));
  EXPECT_EQ(graph.Observations(1), (std::vector<uint32_t>{3}));
  EXPECT_EQ(graph.Observations(2), (std::vector<uint32_t>{4, 5}));
  EXPECT_EQ(graph.Reader(0), 3u);
}

TEST(PipelineProgressTest, EqualCountsPreserveIndependentPhaseIdentities) {
  ProgressGraph graph(6, 11);
  graph.Copy(0, {2, 3});
  graph.Copy(1, {4, 5});
  for (uint32_t channel : {2u, 3u, 4u, 5u}) {
    graph.Observe(channel);
  }
  graph.Prefer(0, 2);
  graph.Prefer(1, 5);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), 2u);
  EXPECT_EQ(graph.Reader(1), 5u);
  EXPECT_EQ(graph.Observations(0), (std::vector<uint32_t>{2, 3}));
  EXPECT_EQ(graph.Observations(1), (std::vector<uint32_t>{4, 5}));
}

TEST(PipelineProgressTest, DifferentCardinalityDoesNotProveRecordProgress) {
  ProgressGraph graph(2, 11);
  graph.SetReads(1, 22);
  graph.Copy(0, {1});
  graph.Observe(1);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_TRUE(graph.Observations(0).empty());
  EXPECT_EQ(graph.Observations(1), (std::vector<uint32_t>{1}));
}

TEST(PipelineProgressTest, CompletePhasesComposeThroughFanout) {
  ProgressGraph graph(4, 6144);
  graph.CopyPhases(0, {1, 2}, {3072, 3072});
  graph.Copy(2, {3});
  graph.Observe(1);
  graph.Observe(3);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), 1u);
  EXPECT_EQ(graph.Observations(0), (std::vector<uint32_t>{1, 3}));
}

TEST(PipelineProgressTest, MissingPhaseDoesNotProveProgress) {
  ProgressGraph graph(2, 6144);
  graph.CopyPhases(0, {1}, {2048, 3072});
  graph.Observe(1);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_TRUE(graph.Observations(0).empty());
}

TEST(PipelineProgressTest, IndependentPhaseSourcesDoNotCompose) {
  ProgressGraph graph(3, 3072);
  graph.SetReads(2, 6144);
  graph.SetWrites(2, 6144);
  graph.CopyPhases(0, {2}, {1024, 2048});
  graph.CopyPhases(1, {2}, {2048, 1024});
  graph.Observe(2);
  graph.Select();
  EXPECT_TRUE(graph.Observations(0).empty());
  EXPECT_TRUE(graph.Observations(1).empty());
}

TEST(PipelineProgressTest, IndependentWorkersDoNotComposePartialCopies) {
  ProgressGraph graph(2, 6144);
  graph.CopyPhases(0, {1}, {3072});
  graph.CopyPhases(0, {1}, {3072});
  graph.Observe(1);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_TRUE(graph.Observations(0).empty());
}

TEST(PipelineProgressTest, DisabledEndpointDoesNotComposePartialCopies) {
  ProgressGraph graph(2, 6144);
  graph.CopyPhases(0, {1}, {3072, 3072});
  graph.SetWrites(1, 0);
  graph.Observe(1);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_TRUE(graph.Observations(0).empty());
}

TEST(PipelineProgressTest, PhaseCardinalityDoesNotExpandStorage) {
  size_t baseline = 0;
  for (uint64_t records : {2ull, 6144ull, 6144000000000ull}) {
    ProgressGraph graph(3, records);
    graph.CopyPhases(0, {1, 2}, {records / 2, records / 2});
    graph.Observe(1);
    graph.Observe(2);
    graph.Select();
    EXPECT_EQ(graph.Observations(0), (std::vector<uint32_t>{1, 2}));
    if (!baseline) {
      baseline = graph.Bytes();
    }
    EXPECT_EQ(graph.Bytes(), baseline);
  }
}

TEST(PipelineProgressTest, RetainedSourceObligationIsNotCopyCompletion) {
  ProgressGraph graph(2, 11);
  graph.Copy(0, {1}, /*retain_source=*/true);
  graph.Observe(1);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_TRUE(graph.Observations(0).empty());
}

TEST(PipelineProgressTest, IndependentPublicationsDoNotShareARecordOrdinal) {
  ProgressGraph graph(3, 11);
  graph.SetReads(2, 22);
  graph.SetWrites(2, 22);
  graph.Copy(0, {2});
  graph.Copy(1, {2});
  graph.Observe(2);
  graph.Select();
  EXPECT_TRUE(graph.Observations(0).empty());
  EXPECT_TRUE(graph.Observations(1).empty());
  EXPECT_EQ(graph.Observations(2), (std::vector<uint32_t>{2}));
}

TEST(PipelineProgressTest, CyclesRequireARecordDistanceProof) {
  ProgressGraph graph(4, 11);
  graph.Copy(0, {1});
  graph.Copy(1, {0});
  graph.Copy(2, {3});
  graph.Observe(0);
  graph.Observe(1);
  graph.Observe(3);
  graph.Select();
  EXPECT_EQ(graph.Reader(0), UINT32_MAX);
  EXPECT_EQ(graph.Reader(1), UINT32_MAX);
  EXPECT_TRUE(graph.Observations(0).empty());
  EXPECT_TRUE(graph.Observations(1).empty());
  EXPECT_EQ(graph.Observations(2), (std::vector<uint32_t>{3}));
}

TEST(PipelineProgressTest, ObservationPopulationIsNotLimitedByMachineWordSize) {
  ProgressGraph graph(131, 11);
  std::vector<uint32_t> destinations;
  for (uint32_t channel = 130; channel > 0; --channel) {
    destinations.push_back(channel);
    graph.Observe(channel);
  }
  graph.Copy(0, destinations);
  graph.Select();
  std::vector<uint32_t> expected;
  for (uint32_t channel = 1; channel <= 130; ++channel) {
    expected.push_back(channel);
  }
  EXPECT_EQ(graph.Observations(0), expected);
  EXPECT_EQ(graph.Reader(0), 1u);
}

TEST(PipelineProgressTest, EpochLengthDoesNotExpandObservationStorage) {
  size_t baseline = 0;
  for (uint64_t records : {11ull, 1100000ull, 1100000000000ull}) {
    ProgressGraph graph(4, records);
    graph.Copy(0, {1});
    graph.Copy(1, {2, 3});
    graph.Observe(2);
    graph.Observe(3);
    graph.Select();
    EXPECT_EQ(graph.Observations(0), (std::vector<uint32_t>{2, 3}));
    if (!baseline) {
      baseline = graph.Bytes();
    }
    EXPECT_EQ(graph.Bytes(), baseline);
  }
}

TEST(PipelineProgressTest, OnePhaseDoesNotRetainIntermediateTransitiveSets) {
  size_t smaller_bytes = 0;
  for (uint32_t count : {64u, 128u}) {
    ProgressGraph graph(count, 11);
    for (uint32_t i = 0; i < count; ++i) {
      graph.Observe(i);
      if (i + 1 < count) {
        graph.Copy(i, {i + 1});
      }
    }
    graph.Select(true, {0, 0});
    EXPECT_EQ(graph.Observations(0).size(), count);
    EXPECT_TRUE(graph.Observations(1).empty());
    if (!smaller_bytes) {
      smaller_bytes = graph.Bytes();
    } else {
      EXPECT_LT(graph.Bytes(), 3 * smaller_bytes);
    }
  }
}

TEST(PipelineProgressTest, SingleObserverSelectionNeedsNoPopulationStorage) {
  ProgressGraph ordinary(3, 11);
  ProgressGraph joint(3, 11);
  for (auto* graph : {&ordinary, &joint}) {
    graph->Copy(0, {1, 2});
    graph->Observe(1);
    graph->Observe(2);
    graph->Prefer(0, 2);
  }
  ordinary.Select(false);
  joint.Select();
  EXPECT_EQ(ordinary.Reader(0), 2u);
  EXPECT_EQ(joint.Reader(0), ordinary.Reader(0));
  EXPECT_LT(ordinary.Bytes(), joint.Bytes());
}

}  // namespace
}  // namespace loom
