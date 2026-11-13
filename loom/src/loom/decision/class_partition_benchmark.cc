// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures observed launch classes independently of provider vocabulary size.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/decision/class_partition.h"

namespace {

void Refine(loom_decision_class_partition_t* partition,
            const std::vector<uint32_t>& outcomes, uint32_t outcome_count) {
  loom_decision_class_partition_begin(partition, outcome_count);
  for (iree_host_size_t i = 0; i < outcomes.size(); ++i) {
    const bool accepted =
        loom_decision_class_partition_record(partition, i, outcomes[i]);
    IREE_ASSERT(accepted);
  }
  loom_decision_class_partition_commit(partition);
  benchmark::DoNotOptimize(partition->site_classes);
  benchmark::DoNotOptimize(partition->class_count);
}

void Run(benchmark::State& state, bool initialize_each_iteration) {
  const iree_host_size_t site_count = state.range(0);
  const uint32_t outcome_count = state.range(1);
  constexpr loom_decision_class_ordinal_t kClassLimit = 128;
  std::vector<uint32_t> outcomes(site_count);
  for (iree_host_size_t i = 0; i < site_count; ++i) {
    outcomes[i] =
        outcome_count - 1 - i % std::min<uint32_t>(outcome_count, 128);
  }
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_decision_class_partition_t partition;
  IREE_CHECK_OK(loom_decision_class_partition_initialize(
      site_count, kClassLimit, outcome_count, &arena, &partition));
  Refine(&partition, outcomes, outcome_count);
  const iree_host_size_t used_bytes = arena.used_allocation_size;
  const iree_host_size_t owned_bytes = arena.total_allocation_size;
  if (initialize_each_iteration) {
    iree_arena_reset(&arena);
  }
  iree_arena_block_pool_statistics_t before;
  iree_arena_block_pool_query_statistics(&pool, &before);
  for (auto _ : state) {
    if (initialize_each_iteration) {
      IREE_CHECK_OK(loom_decision_class_partition_initialize(
          site_count, kClassLimit, outcome_count, &arena, &partition));
    }
    Refine(&partition, outcomes, outcome_count);
    if (initialize_each_iteration) {
      iree_arena_reset(&arena);
    }
  }
  iree_arena_block_pool_statistics_t after;
  iree_arena_block_pool_query_statistics(&pool, &after);
  state.counters["arena_used_bytes"] = used_bytes;
  state.counters["arena_owned_bytes"] = owned_bytes;
  state.counters["pool_steady_allocations"] =
      after.block_system_allocation_count -
      before.block_system_allocation_count;
  state.counters["oversized_allocations_per_iteration"] =
      double(after.oversized_allocation_count -
             before.oversized_allocation_count) /
      state.iterations();
  state.SetItemsProcessed(state.iterations() * site_count);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

void BM_RefineClasses(benchmark::State& state) { Run(state, false); }

void BM_CollectClasses(benchmark::State& state) { Run(state, true); }

BENCHMARK(BM_RefineClasses)->ArgsProduct({{2, 128, 4096}, {2, 64, 4098}});
BENCHMARK(BM_CollectClasses)->ArgsProduct({{2, 128, 4096}, {2, 64, 4098}});

}  // namespace
