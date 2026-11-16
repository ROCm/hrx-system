// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/pipeline/execution.h"
#include "loom/analysis/pipeline/lifetime_layout.h"

namespace {

enum class Shape { kBranches, kPhases, kNested };

// The source owner has already bound these occurrence/control tables. Native
// instruction size and runtime loop trip counts do not affect this inventory.
class PipelineFixture {
 public:
  PipelineFixture(Shape shape, size_t count, size_t engine_count)
      : resources_(count + 1), invocations_(count + 1), entries_(count + 1) {
    for (size_t i = 0; i < entries_.size(); ++i) {
      auto& invocation = invocations_[i];
      invocation.ordinal = i;
      invocation.parent =
          i ? &invocations_[shape == Shape::kNested ? i - 1 : 0] : nullptr;
      invocation.resources = &resources_[i];
      invocation.first_worker = engines_.size();
      entries_[i] = &invocation;
      const size_t worker_count =
          shape == Shape::kPhases     ? (i ? engine_count : 0)
          : shape == Shape::kBranches ? (i ? 1 : 0)
                                      : (i == count ? 1 : 0);
      resources_[i].strand_count = worker_count;
      for (size_t j = 0; j < worker_count; ++j) {
        engines_.push_back(shape == Shape::kBranches ? i - 1 : j);
        workers_.push_back({.invocation = &invocation});
      }
    }
    if (shape == Shape::kPhases) {
      steps_.resize(count + 1);
      bindings_.resize(count + 1);
      for (size_t i = 0; i < count; ++i) {
        steps_[i].kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION;
        bindings_[i].child = &invocations_[i + 1];
      }
      steps_[count].kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD;
      control_.graph.block_count = 1;
      block_ = {.steps = steps_.data(), .step_count = steps_.size()};
      scope_ = {.control = &control_, .blocks = &block_};
      bound_block_ = bindings_.data();
      schedule_ = {.source = &scope_, .blocks = &bound_block_};
      resources_[0].schedule_count = 1;
      invocations_[0].schedules = &schedule_;
    } else if (shape == Shape::kBranches) {
      resources_[0].composition_count = count;
      invocations_[0].compositions = entries_.data() + 1;
    } else {
      for (size_t i = 0; i < count; ++i) {
        resources_[i].composition_count = 1;
        invocations_[i].compositions = &entries_[i + 1];
      }
    }
    graph_ = {
        .entries = entries_.data(),
        .count = entries_.size(),
        .workers = workers_.data(),
        .worker_count = workers_.size(),
    };
    requests_.resize(workers_.size());
  }

  void Run(iree_arena_allocator_t* arena, size_t engine_count) {
    IREE_CHECK_OK(
        loom_pipeline_lifetimes_build(&graph_, arena, &graph_.lifetimes));
    loom_pipeline_execution_t execution = {};
    loom_pipeline_execution_conflict_t conflict = {};
    IREE_CHECK_OK(loom_pipeline_execution_build(
        &graph_, engines_.data(), engine_count, arena, &execution, &conflict));
    IREE_ASSERT_EQ(conflict.second_worker, IREE_HOST_SIZE_MAX);

    // Control projection consumes every participating occurrence/engine pair.
    // Include that iteration so a compact producer cannot hide dense consumers.
    uint64_t checksum = 0;
    for (size_t i = 0; i < graph_.count; ++i) {
      loom_pipeline_participant_iterator_t iterator;
      loom_pipeline_participants_begin(execution.invocations[i], &iterator);
      loom_pipeline_participant_t participant;
      while (loom_pipeline_participants_next(&iterator, &participant)) {
        checksum += participant.worker + participant.engine + 1;
      }
    }

    for (size_t i = 0; i < workers_.size(); ++i) {
      requests_[i] = {
          .lifetime = graph_.lifetimes.owners[workers_[i].invocation->ordinal],
          .pool = engines_[i],
          .length = 64 + (i % 3) * 16,
          .alignment = 16,
      };
    }
    const loom_pipeline_lifetime_extent_t* extents = nullptr;
    IREE_CHECK_OK(loom_pipeline_lifetime_layout(
        &graph_.lifetimes, requests_.data(), requests_.size(), engine_count,
        arena, &extents));
    for (const auto& request : requests_) {
      checksum += request.offset;
    }
    for (size_t i = 0; i < engine_count; ++i) {
      checksum += extents[i].length;
    }
    benchmark::DoNotOptimize(checksum);
  }

 private:
  // Occurrence-local resource counts, stable through all repetitions.
  std::vector<loom_pipeline_resources_t> resources_;
  // Parent-before-child occurrences, with distinct logical identities.
  std::vector<loom_pipeline_invocation_t> invocations_;
  // Canonical occurrence inventory and concurrent child pointer ranges.
  std::vector<const loom_pipeline_invocation_t*> entries_;
  // One binding per source worker; only the selected engines participate.
  std::vector<size_t> engines_;
  // Source worker ownership used to form physical storage requests.
  std::vector<loom_pipeline_invocation_worker_t> workers_;
  // Classified sites in the repeated-phase serial scope.
  std::vector<loom_pipeline_schedule_step_t> steps_;
  // Source occurrence binding for each classified site.
  std::vector<loom_pipeline_invocation_step_t> bindings_;
  // Retained CFG header for the single serial block.
  loom_value_fact_cfg_region_t control_ = {};
  // Ordered source sites, without expanding dynamic repetitions.
  loom_pipeline_schedule_block_t block_ = {};
  // Static serial scope retaining its control and sites.
  loom_pipeline_schedule_t scope_ = {};
  // Bound sites for that block.
  const loom_pipeline_invocation_step_t* bound_block_ = nullptr;
  // Occurrence-specific serial scope.
  loom_pipeline_invocation_schedule_t schedule_ = {};
  // Input inventory; each run replaces its arena-owned lifetime result.
  loom_pipeline_invocations_t graph_ = {};
  // One resource request per worker, overlaid across ordered phases.
  std::vector<loom_pipeline_lifetime_request_t> requests_;
};

static void RunPlanning(benchmark::State& state, Shape shape) {
  const size_t count = state.range(0);
  const size_t engine_count = state.range(1);
  PipelineFixture fixture(shape, count, engine_count);
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  size_t arena_bytes = 0, reserved_bytes = 0;
  for (auto _ : state) {
    fixture.Run(&arena, engine_count);
    arena_bytes = std::max(arena_bytes, arena.used_allocation_size);
    reserved_bytes = std::max(reserved_bytes, arena.total_allocation_size);
    iree_arena_reset(&arena);
  }
  state.counters["arena_bytes"] = double(arena_bytes);
  state.counters["reserved_bytes"] = double(reserved_bytes);
  state.SetItemsProcessed(state.iterations() * count);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

static void BM_Branches(benchmark::State& state) {
  RunPlanning(state, Shape::kBranches);
}
BENCHMARK(BM_Branches)->Args({32, 32})->Args({256, 256})->Args({1024, 1024});

static void BM_Phases(benchmark::State& state) {
  RunPlanning(state, Shape::kPhases);
}
BENCHMARK(BM_Phases)->Args({8, 32})->Args({64, 32})->Args({256, 32});

static void BM_Nested(benchmark::State& state) {
  RunPlanning(state, Shape::kNested);
}
BENCHMARK(BM_Nested)->Args({32, 32})->Args({256, 256})->Args({1024, 1024});

// Each serial phase owns two concurrent resource users. Source enumeration
// alternates their order, while the caller requests the same role first in
// every phase. This measures preserving reusable resource identities through
// both compact and sparse physical pool domains.
static void BM_LifetimeReuse(benchmark::State& state) {
  const size_t phase_count = state.range(0);
  const size_t pool_count = state.range(1);
  constexpr auto none = IREE_HOST_SIZE_MAX;
  std::vector<loom_pipeline_lifetime_t> nodes(3 + phase_count * 3);
  nodes[0] = {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none};
  nodes[1] = {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2};
  nodes[2] = {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none};
  for (size_t phase = 0; phase < phase_count; ++phase) {
    const size_t index = 3 + phase * 3;
    nodes[index] = {LOOM_PIPELINE_LIFETIME_CONCURRENT, phase * 2 + 1, 2,
                    index + 1, phase + 1 < phase_count ? index + 3 : none};
    nodes[index + 1] = {LOOM_PIPELINE_LIFETIME_OWNER, phase * 2 + 1, index,
                        none, index + 2};
    nodes[index + 2] = {LOOM_PIPELINE_LIFETIME_OWNER, phase * 2 + 2, index,
                        none, none};
  }
  std::vector<loom_pipeline_lifetime_request_t> requests;
  requests.reserve(phase_count * pool_count * 2);
  for (size_t role = 0; role < 2; ++role) {
    for (size_t phase = 0; phase < phase_count; ++phase) {
      for (size_t pool = 0; pool < pool_count; ++pool) {
        requests.push_back({
            .lifetime = 4 + phase * 3 + ((phase + role) % 2),
            .pool = pool,
            .length = 1,
            .alignment = 1,
        });
      }
    }
  }
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(32768, iree_allocator_system(), &blocks);
  iree_arena_allocator_t source_arena, arena;
  iree_arena_initialize(&blocks, &source_arena);
  iree_arena_initialize(&blocks, &arena);
  loom_pipeline_lifetimes_t lifetimes = {
      .nodes = nodes.data(),
      .count = nodes.size(),
      .has_serial = true,
  };
  IREE_CHECK_OK(loom_pipeline_lifetimes_index(&source_arena, &lifetimes));
  size_t arena_bytes = 0;
  for (auto _ : state) {
    const loom_pipeline_lifetime_extent_t* extents = nullptr;
    IREE_CHECK_OK(loom_pipeline_lifetime_layout(&lifetimes, requests.data(),
                                                requests.size(), pool_count,
                                                &arena, &extents));
    uint64_t checksum = 0;
    for (const auto& request : requests) {
      checksum += request.offset;
    }
    benchmark::DoNotOptimize(checksum);
    benchmark::DoNotOptimize(extents);
    arena_bytes = std::max(arena_bytes, arena.used_allocation_size);
    iree_arena_reset(&arena);
  }
  state.counters["arena_bytes"] = double(arena_bytes);
  state.SetItemsProcessed(state.iterations() * requests.size());
  iree_arena_deinitialize(&arena);
  iree_arena_deinitialize(&source_arena);
  iree_arena_block_pool_deinitialize(&blocks);
}
BENCHMARK(BM_LifetimeReuse)
    ->Args({8, 32})
    ->Args({64, 32})
    ->Args({256, 32})
    ->Args({64, 1024});

}  // namespace
