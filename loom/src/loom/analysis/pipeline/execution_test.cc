// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/execution.h"

#include <deque>
#include <initializer_list>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// These fixtures are the retained fork/join tables consumed by the analysis.
// Worker bodies and source operations have already been classified by their
// owners and are deliberately unavailable to the execution planner.
struct Invocation {
  // Resource counts and direct concurrency edges for this occurrence.
  loom_pipeline_resources_t resources = {};
  // Bound occurrence stored in parent-before-child order.
  loom_pipeline_invocation_t source = {};
  // Concurrent child occurrence pointers.
  std::vector<const loom_pipeline_invocation_t*> children;
  // Bound independent serial scopes.
  std::vector<loom_pipeline_invocation_schedule_t> schedules;
};

struct Scope {
  // Canonical CFG header; this fixture has one block ending in yield.
  loom_value_fact_cfg_region_t control = {};
  // Classified composition and nested-scope sites, followed by yield.
  std::vector<loom_pipeline_schedule_step_t> steps;
  // Occurrence bindings parallel to steps.
  std::vector<loom_pipeline_invocation_step_t> bindings;
  // Single block's retained steps.
  loom_pipeline_schedule_block_t block = {};
  // Source scope retaining the classified CFG.
  loom_pipeline_schedule_t source = {};
  // Bound block pointer table.
  const loom_pipeline_invocation_step_t* bound_block = nullptr;
  // Bound scope used by nested serial sites.
  loom_pipeline_invocation_schedule_t bound = {};
};

class PipelineExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  Invocation& AddInvocation(std::initializer_list<iree_host_size_t> engines) {
    invocations_.emplace_back();
    auto& invocation = invocations_.back();
    invocation.source.ordinal = invocations_.size() - 1;
    invocation.source.first_worker = engines_.size();
    invocation.source.resources = &invocation.resources;
    invocation.resources.strand_count = engines.size();
    engines_.insert(engines_.end(), engines);
    return invocation;
  }

  Scope& AddScope(std::initializer_list<Invocation*> children,
                  Scope* nested = nullptr) {
    scopes_.emplace_back();
    auto& scope = scopes_.back();
    for (auto* child : children) {
      scope.steps.push_back({
          .kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION,
      });
      scope.bindings.push_back({.child = &child->source});
    }
    if (nested) {
      scope.steps.push_back({
          .nested = &nested->source,
          .kind = LOOM_PIPELINE_SCHEDULE_STEP_SERIAL,
      });
      scope.bindings.push_back({.nested = &nested->bound});
    }
    scope.steps.push_back({
        .kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD,
    });
    scope.bindings.push_back({});
    scope.control.graph.block_count = 1;
    scope.block.steps = scope.steps.data();
    scope.block.step_count = scope.steps.size();
    scope.source.control = &scope.control;
    scope.source.blocks = &scope.block;
    scope.bound_block = scope.bindings.data();
    scope.bound.source = &scope.source;
    scope.bound.blocks = &scope.bound_block;
    return scope;
  }

  void Build(iree_host_size_t engine_count) {
    std::vector<const loom_pipeline_invocation_t*> entries;
    for (auto& invocation : invocations_) {
      invocation.resources.composition_count = invocation.children.size();
      invocation.source.compositions = invocation.children.data();
      invocation.resources.schedule_count = invocation.schedules.size();
      invocation.source.schedules = invocation.schedules.data();
      entries.push_back(&invocation.source);
    }
    loom_pipeline_invocations_t graph = {
        .entries = entries.data(),
        .count = entries.size(),
        .worker_count = engines_.size(),
    };
    IREE_ASSERT_OK(loom_pipeline_execution_build(&graph, engines_.data(),
                                                 engine_count, &arena_,
                                                 &execution_, &conflict_));
  }

  iree_host_size_t Participant(const Invocation& invocation,
                               iree_host_size_t engine) {
    return loom_pipeline_participants_worker(
        execution_.invocations[invocation.source.ordinal], engine);
  }

  // Arena pool backing only retained execution results.
  iree_arena_block_pool_t pool_;
  // Retained analysis allocation lifetime.
  iree_arena_allocator_t arena_;
  // Stable occurrence addresses used by direct and serial child edges.
  std::deque<Invocation> invocations_;
  // Stable source and bound addresses for arbitrarily nested serial scopes.
  std::deque<Scope> scopes_;
  // Physical engine selection in complete worker inventory order.
  std::vector<iree_host_size_t> engines_;
  // Last computed participant rows.
  loom_pipeline_execution_t execution_ = {};
  // Last conflict, with invalid second worker on success.
  loom_pipeline_execution_conflict_t conflict_ = {};
};

TEST_F(PipelineExecutionTest, IndependentStrandsNeedIndependentProgress) {
  AddInvocation({0, 0});
  Build(1);
  EXPECT_EQ(conflict_.first_worker, 0u);
  EXPECT_EQ(conflict_.second_worker, 1u);
}

TEST_F(PipelineExecutionTest, SharedResourcesRequireOrderedWorkerLifetimes) {
  auto& root = AddInvocation({});
  auto& first = AddInvocation({0});
  auto& second = AddInvocation({1});
  auto& third = AddInvocation({2});
  const loom_pipeline_execution_sharing_t sharing[] = {
      {2, 1},
      {0, 1},
      {1, 2},
      {0, 0},
  };
  iree_host_size_t* representatives = nullptr;
  IREE_ASSERT_OK(loom_pipeline_execution_coalesce(
      3, sharing, IREE_ARRAYSIZE(sharing), &arena_, &representatives));
  engines_.assign(representatives, representatives + 3);
  EXPECT_EQ(engines_, (std::vector<iree_host_size_t>{0, 0, 0}));
  auto& scope = AddScope({&first, &second, &third});
  root.schedules.push_back(scope.bound);
  Build(3);
  EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(Participant(root, 0), 0u);

  // Equal physical resources cannot serialize independently progressing work.
  root.schedules.clear();
  root.children = {&first.source, &second.source, &third.source};
  Build(3);
  EXPECT_EQ(conflict_.first_worker, 0u);
  EXPECT_EQ(conflict_.second_worker, 1u);
}

TEST_F(PipelineExecutionTest,
       DistinctResourceComponentsKeepIndependentEngines) {
  const loom_pipeline_execution_sharing_t sharing[] = {
      {4, 2},
      {3, 1},
      {6, 4},
  };
  iree_host_size_t* representatives = nullptr;
  IREE_ASSERT_OK(loom_pipeline_execution_coalesce(
      7, sharing, IREE_ARRAYSIZE(sharing), &arena_, &representatives));
  EXPECT_EQ(
      (std::vector<iree_host_size_t>(representatives, representatives + 7)),
      (std::vector<iree_host_size_t>{0, 1, 2, 1, 2, 5, 2}));
}

TEST_F(PipelineExecutionTest, SerialChildrenReuseEnginesAcrossNestedScopes) {
  auto& root = AddInvocation({});
  auto& first = AddInvocation({0, 1});
  auto& second = AddInvocation({0, 1});
  auto& nested = AddScope({&second});
  auto& outer = AddScope({&first}, &nested);
  root.schedules.push_back(outer.bound);
  Build(3);
  EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(Participant(root, 0), 0u);
  EXPECT_EQ(Participant(root, 1), 1u);
  EXPECT_EQ(Participant(root, 2), IREE_HOST_SIZE_MAX);
  EXPECT_EQ(Participant(second, 0), 2u);
  EXPECT_EQ(Participant(second, 1), 3u);
  EXPECT_EQ(execution_.schedule_count, 2u);
  const auto* nested_workers =
      loom_pipeline_execution_schedule(&execution_, &nested.bound)
          ->participants;
  EXPECT_EQ(loom_pipeline_participants_worker(nested_workers, 0), 2u);
  EXPECT_EQ(loom_pipeline_participants_worker(nested_workers, 1), 3u);
  const auto* outer_workers =
      loom_pipeline_execution_schedule(&execution_, &root.schedules[0])
          ->participants;
  EXPECT_EQ(loom_pipeline_participants_worker(outer_workers, 0), 0u);
  EXPECT_EQ(loom_pipeline_participants_worker(outer_workers, 1), 1u);
}

TEST_F(PipelineExecutionTest, ConcurrentChildrenCannotReuseAnEngine) {
  auto& root = AddInvocation({});
  auto& first = AddInvocation({0});
  auto& second = AddInvocation({0});
  root.children = {&first.source, &second.source};
  Build(1);
  EXPECT_EQ(conflict_.first_worker, 0u);
  EXPECT_EQ(conflict_.second_worker, 1u);
}

TEST_F(PipelineExecutionTest, CompactSlotsIncludeOnlyActualParticipants) {
  auto& root = AddInvocation({7});
  auto& left = AddInvocation({129});
  auto& empty = AddInvocation({IREE_HOST_SIZE_MAX});
  auto& right = AddInvocation({4095});
  root.children = {&left.source, &empty.source, &right.source};
  Build(4096);
  ASSERT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(execution_.participant_count, 5u);
  EXPECT_EQ((std::vector<iree_host_size_t>(execution_.offsets,
                                           execution_.offsets + 5)),
            (std::vector<iree_host_size_t>{0, 3, 4, 4, 5}));
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 0, 7), 0u);
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 0, 129), 1u);
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 0, 4095), 2u);
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 1, 129), 3u);
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 3, 4095), 4u);
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 1, 7),
            IREE_HOST_SIZE_MAX);
  EXPECT_EQ(loom_pipeline_execution_index(&execution_, 2, 129),
            IREE_HOST_SIZE_MAX);
  EXPECT_EQ(execution_.workers[7], 0u);
  EXPECT_EQ(execution_.workers[129], 1u);
  EXPECT_EQ(execution_.workers[4095], 3u);
  EXPECT_EQ(execution_.workers[128], IREE_HOST_SIZE_MAX);
}

TEST_F(PipelineExecutionTest, AnOuterStrandOverlapsEverySerialVisit) {
  auto& root = AddInvocation({0});
  auto& child = AddInvocation({0});
  root.schedules.push_back(AddScope({&child}).bound);
  Build(1);
  EXPECT_EQ(conflict_.first_worker, 0u);
  EXPECT_EQ(conflict_.second_worker, 1u);
}

TEST_F(PipelineExecutionTest, IndependentSerialScopesProgressConcurrently) {
  auto& root = AddInvocation({});
  auto& first = AddInvocation({0});
  auto& second = AddInvocation({0});
  root.schedules.push_back(AddScope({&first}).bound);
  root.schedules.push_back(AddScope({&second}).bound);
  Build(1);
  EXPECT_EQ(conflict_.first_worker, 1u);
  EXPECT_EQ(conflict_.second_worker, 0u);
}

TEST_F(PipelineExecutionTest, SerialParentsRetainConcurrentChildConflicts) {
  auto& root = AddInvocation({});
  auto& phase = AddInvocation({1});
  auto& concurrent = AddInvocation({1});
  root.schedules.push_back(AddScope({&phase}).bound);
  phase.children = {&concurrent.source};
  Build(2);
  EXPECT_EQ(conflict_.first_worker, 0u);
  EXPECT_EQ(conflict_.second_worker, 1u);
}

TEST_F(PipelineExecutionTest, AutonomousActorsDoNotOccupyInstructionEngines) {
  auto& root = AddInvocation({IREE_HOST_SIZE_MAX, 0, IREE_HOST_SIZE_MAX, 1});
  Build(2);
  EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(Participant(root, 0), 1u);
  EXPECT_EQ(Participant(root, 1), 3u);
}

TEST_F(PipelineExecutionTest, EmptyAssignmentRetainsEmptyScopeParticipation) {
  auto& root = AddInvocation({IREE_HOST_SIZE_MAX});
  auto& child = AddInvocation({IREE_HOST_SIZE_MAX});
  root.schedules.push_back(AddScope({&child}).bound);
  Build(0);
  EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(execution_.engine_count, 0u);
  EXPECT_EQ(execution_.schedule_count, 1u);
  EXPECT_EQ(loom_pipeline_execution_schedule(&execution_, &root.schedules[0])
                ->participants,
            execution_.schedules[0].participants);
}

TEST_F(PipelineExecutionTest, ParticipantSetsPropagateThroughNestedChildren) {
  auto& root = AddInvocation({0});
  auto& wrapper = AddInvocation({1});
  auto& first = AddInvocation({2});
  auto& second = AddInvocation({2, 3});
  root.children = {&wrapper.source};
  wrapper.schedules.push_back(AddScope({&first, &second}).bound);
  Build(4);
  EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(Participant(root, 0), 0u);
  EXPECT_EQ(Participant(root, 1), 1u);
  EXPECT_EQ(Participant(root, 2), 2u);
  EXPECT_EQ(Participant(root, 3), 4u);
}

TEST_F(PipelineExecutionTest, DeepSerialNestingDoesNotUseTheCallStack) {
  auto& root = AddInvocation({});
  auto& child = AddInvocation({0});
  Scope* scope = &AddScope({&child});
  for (int i = 0; i < 2048; ++i) {
    scope = &AddScope({}, scope);
  }
  root.schedules.push_back(scope->bound);
  Build(1);
  EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(Participant(root, 0), 0u);
  EXPECT_EQ(execution_.schedule_count, 2049u);
  EXPECT_EQ(execution_.invocations[0], execution_.invocations[1]);
  for (size_t i = 0; i < execution_.schedule_count; ++i) {
    EXPECT_EQ(execution_.schedules[i].participants, execution_.invocations[1]);
  }
}

}  // namespace
}  // namespace loom
