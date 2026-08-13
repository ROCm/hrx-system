// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "loom/format/bytecode/reader/selected_projection.h"

namespace {

static void SelectedScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

static void BM_InsertReachedFacts(benchmark::State& state) {
  const uint32_t count = static_cast<uint32_t>(state.range(0));
  for (auto _ : state) {
    loom_bytecode_selected_projection_t projection;
    loom_bytecode_selected_projection_initialize(iree_allocator_system(),
                                                 &projection);
    for (uint32_t i = 0; i < count; ++i) {
      const auto domain =
          static_cast<loom_bytecode_selected_projection_domain_t>(i % 5u);
      IREE_CHECK_OK(loom_bytecode_selected_projection_insert(
          &projection, domain, i, count - i));
    }
    benchmark::DoNotOptimize(projection.slots.values);
    state.PauseTiming();
    loom_bytecode_selected_projection_deinitialize(&projection);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetComplexityN(count);
}
BENCHMARK(BM_InsertReachedFacts)
    ->Apply(SelectedScales)
    ->Complexity(benchmark::oN);

static void BM_LookupReachedFacts(benchmark::State& state) {
  const uint32_t count = static_cast<uint32_t>(state.range(0));
  loom_bytecode_selected_projection_t projection;
  loom_bytecode_selected_projection_initialize(iree_allocator_system(),
                                               &projection);
  for (uint32_t i = 0; i < count; ++i) {
    const auto domain =
        static_cast<loom_bytecode_selected_projection_domain_t>(i & 3u);
    IREE_CHECK_OK(loom_bytecode_selected_projection_insert(&projection, domain,
                                                           i, count - i));
  }

  uint32_t ordinal = 0;
  for (auto _ : state) {
    const uint32_t source_ordinal = ordinal++ % count;
    const auto domain = static_cast<loom_bytecode_selected_projection_domain_t>(
        source_ordinal % 5u);
    uint32_t target_id = 0;
    loom_bytecode_selected_projection_lookup(&projection, domain,
                                             source_ordinal, &target_id);
    benchmark::DoNotOptimize(target_id);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["retained_bytes"] =
      static_cast<double>(projection.slots.capacity * sizeof(uint64_t));
  loom_bytecode_selected_projection_deinitialize(&projection);
}
BENCHMARK(BM_LookupReachedFacts)->Apply(SelectedScales);

}  // namespace

BENCHMARK_MAIN();
