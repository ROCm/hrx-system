// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/program_source_test_util.h"

namespace {

static void RunAdmission(benchmark::State& state, bool diamond) {
  const size_t helper_count = state.range(0);
  const size_t root_count = state.range(1);
  std::vector<std::vector<loom_symbol_id_t>> edges(root_count + helper_count);
  for (size_t i = 0; i < root_count; ++i) {
    if (diamond) {
      for (size_t j = root_count; j + 1 < edges.size(); ++j) {
        edges[i].push_back(static_cast<loom_symbol_id_t>(j));
      }
    } else {
      edges[i].push_back(static_cast<loom_symbol_id_t>(root_count));
    }
  }
  for (size_t i = root_count; i + 1 < edges.size(); ++i) {
    edges[i].push_back(
        static_cast<loom_symbol_id_t>(diamond ? edges.size() - 1 : i + 1));
  }
  loom::aie2p::testing::SourceInventory fixture(edges);
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  size_t arena_bytes = 0, reserved_bytes = 0, helper_references = 0;
  for (auto _ : state) {
    fixture.Reset(&arena);
    for (size_t i = 0; i < root_count; ++i) {
      fixture.Admit(i);
    }
    helper_references = fixture.RetainedHelperCount();
    benchmark::DoNotOptimize(helper_references);
    arena_bytes = std::max(arena_bytes, arena.used_allocation_size);
    reserved_bytes = std::max(reserved_bytes, arena.total_allocation_size);
    iree_arena_reset(&arena);
  }
  state.counters["arena_bytes"] = double(arena_bytes);
  state.counters["reserved_bytes"] = double(reserved_bytes);
  state.counters["helper_references"] = double(helper_references);
  state.SetItemsProcessed(state.iterations() * root_count * helper_count);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

static void BM_Chain(benchmark::State& state) { RunAdmission(state, false); }
BENCHMARK(BM_Chain)
    ->Args({8, 1})
    ->Args({32, 1})
    ->Args({256, 1})
    ->Args({1024, 1})
    ->Args({32, 32})
    ->Args({256, 32});

static void BM_Diamond(benchmark::State& state) { RunAdmission(state, true); }
BENCHMARK(BM_Diamond)->Args({8, 1})->Args({32, 1})->Args({256, 1});

}  // namespace
