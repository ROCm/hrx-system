// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for loom_symbol_map_t insert/find/erase throughput.
//
// Measures performance at sizes representative of production ML
// compilation: N in {32, 256, 1024, 4096, 16384}. At production
// scale (Llama 70B), symbol counts are in the low thousands.

#include <cstdint>
#include <cstring>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/symbol_map.h"

namespace {

// Arena setup shared across benchmarks.
struct BenchmarkArena {
  iree_arena_block_pool_t block_pool;
  iree_arena_allocator_t arena;
  BenchmarkArena() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool);
    iree_arena_initialize(&block_pool, &arena);
  }
  ~BenchmarkArena() {
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&block_pool);
  }
  void Reset() {
    iree_arena_deinitialize(&arena);
    iree_arena_initialize(&block_pool, &arena);
  }
};

// Populates a map with N sequential entries.
static void PopulateMap(loom_symbol_map_t* map, iree_arena_allocator_t* arena,
                        int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(
        loom_symbol_map_insert(map, arena, (loom_string_id_t)i, (uint16_t)i));
  }
}

static void BM_Insert(benchmark::State& state) {
  int64_t count = state.range(0);
  BenchmarkArena bench;
  for (auto _ : state) {
    state.PauseTiming();
    bench.Reset();
    loom_symbol_map_t map;
    memset(&map, 0, sizeof(map));
    state.ResumeTiming();
    PopulateMap(&map, &bench.arena, count);
    benchmark::DoNotOptimize(map.count);
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_Insert)->Arg(32)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_FindHit(benchmark::State& state) {
  int64_t count = state.range(0);
  BenchmarkArena bench;
  loom_symbol_map_t map;
  memset(&map, 0, sizeof(map));
  PopulateMap(&map, &bench.arena, count);

  int64_t i = 0;
  for (auto _ : state) {
    uint16_t result = loom_symbol_map_find(&map, (loom_string_id_t)(i % count));
    benchmark::DoNotOptimize(result);
    ++i;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FindHit)->Arg(32)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_FindMiss(benchmark::State& state) {
  int64_t count = state.range(0);
  BenchmarkArena bench;
  loom_symbol_map_t map;
  memset(&map, 0, sizeof(map));
  PopulateMap(&map, &bench.arena, count);

  // Look up keys that don't exist (offset past the populated range).
  int64_t i = 0;
  for (auto _ : state) {
    uint16_t result =
        loom_symbol_map_find(&map, (loom_string_id_t)(count + (i % count)));
    benchmark::DoNotOptimize(result);
    ++i;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FindMiss)->Arg(32)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_FindOrInsert(benchmark::State& state) {
  // Simulates the parser pattern: each symbol is defined once (insert)
  // then referenced several times (find). We use N symbols and cycle
  // through them, so ~1/N of calls are inserts and the rest are finds.
  int64_t count = state.range(0);
  BenchmarkArena bench;
  for (auto _ : state) {
    state.PauseTiming();
    bench.Reset();
    loom_symbol_map_t map;
    memset(&map, 0, sizeof(map));
    state.ResumeTiming();
    // Each symbol referenced ~4 times on average.
    int64_t total_ops = count * 4;
    for (int64_t i = 0; i < total_ops; ++i) {
      loom_string_id_t name_id = (loom_string_id_t)(i % count);
      uint16_t symbol_id = 0;
      IREE_CHECK_OK(loom_symbol_map_find_or_insert(
          &map, &bench.arena, name_id, (uint16_t)name_id, &symbol_id));
      benchmark::DoNotOptimize(symbol_id);
    }
  }
  state.SetItemsProcessed(state.iterations() * count * 4);
}
BENCHMARK(BM_FindOrInsert)->Arg(32)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_Erase(benchmark::State& state) {
  int64_t count = state.range(0);
  BenchmarkArena bench;
  for (auto _ : state) {
    state.PauseTiming();
    bench.Reset();
    loom_symbol_map_t map;
    memset(&map, 0, sizeof(map));
    PopulateMap(&map, &bench.arena, count);
    state.ResumeTiming();
    for (int64_t i = 0; i < count; ++i) {
      bool erased = loom_symbol_map_erase(&map, (loom_string_id_t)i);
      benchmark::DoNotOptimize(erased);
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_Erase)->Arg(32)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384);

}  // namespace
