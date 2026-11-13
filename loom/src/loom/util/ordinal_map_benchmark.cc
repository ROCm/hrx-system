// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/util/ordinal_map.h"

namespace {

static void SelectedScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

static void SparseSelectedScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(16)->Arg(64)->Arg(256)->Arg(1024)->Arg(2048);
}

// The largest source ordinal remains below the 24-bit wire identity limit.
static constexpr uint32_t kSparseOrdinalStride = 8192;

static void InsertReachedFacts(benchmark::State& state, uint32_t stride) {
  const uint32_t count = static_cast<uint32_t>(state.range(0));
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool);
  for (auto _ : state) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool, &arena);
    loom_ordinal_map_t projection;
    loom_ordinal_map_initialize(&arena, &projection);
    for (uint32_t i = 0; i < count; ++i) {
      const auto domain = static_cast<uint8_t>(i % 8u);
      IREE_CHECK_OK(
          loom_ordinal_map_insert(&projection, domain, i * stride, count - i));
    }
    benchmark::DoNotOptimize(projection.buckets.values);
    state.PauseTiming();
    iree_arena_deinitialize(&arena);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetComplexityN(count);
  iree_arena_block_pool_deinitialize(&pool);
}

static void BM_InsertReachedFacts(benchmark::State& state) {
  InsertReachedFacts(state, 1);
}
BENCHMARK(BM_InsertReachedFacts)->Apply(SelectedScales)->Complexity();

static void BM_InsertSparseReachedFacts(benchmark::State& state) {
  InsertReachedFacts(state, kSparseOrdinalStride);
}
BENCHMARK(BM_InsertSparseReachedFacts)
    ->Apply(SparseSelectedScales)
    ->Complexity();

static void LookupReachedFacts(benchmark::State& state, uint32_t stride) {
  const uint32_t count = static_cast<uint32_t>(state.range(0));
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_ordinal_map_t projection;
  loom_ordinal_map_initialize(&arena, &projection);
  for (uint32_t i = 0; i < count; ++i) {
    const auto domain = static_cast<uint8_t>(i % 8u);
    IREE_CHECK_OK(
        loom_ordinal_map_insert(&projection, domain, i * stride, count - i));
  }
  for (uint32_t i = 0; i < count; ++i) {
    const auto domain = static_cast<uint8_t>(i % 8u);
    uint32_t target_id = 0;
    if (!loom_ordinal_map_lookup(&projection, domain, i * stride, &target_id) ||
        target_id != count - i) {
      state.SkipWithError(
          "reached-fact lookup must return its inserted target");
      iree_arena_deinitialize(&arena);
      iree_arena_block_pool_deinitialize(&pool);
      return;
    }
  }

  uint32_t ordinal = 0;
  for (auto _ : state) {
    const uint32_t source_ordinal = ordinal++ % count;
    const auto domain = static_cast<uint8_t>(source_ordinal % 8u);
    uint32_t target_id = 0;
    bool found = loom_ordinal_map_lookup(&projection, domain,
                                         source_ordinal * stride, &target_id);
    benchmark::DoNotOptimize(found);
    benchmark::DoNotOptimize(target_id);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["arena_bytes"] =
      static_cast<double>(arena.total_allocation_size);
  state.counters["used_bytes"] =
      static_cast<double>(arena.used_allocation_size);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

static void BM_LookupReachedFacts(benchmark::State& state) {
  LookupReachedFacts(state, 1);
}
BENCHMARK(BM_LookupReachedFacts)->Apply(SelectedScales);

static void BM_LookupSparseReachedFacts(benchmark::State& state) {
  LookupReachedFacts(state, kSparseOrdinalStride);
}
BENCHMARK(BM_LookupSparseReachedFacts)->Apply(SparseSelectedScales);

}  // namespace

BENCHMARK_MAIN();
