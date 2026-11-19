// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "loom/runtime/storage/backing_directory.h"

namespace {

static std::vector<iree_host_size_t> Populate(
    iree_host_size_t count, loom_runtime_backing_directory_t* directory) {
  std::vector<iree_host_size_t> indexes;
  indexes.reserve(count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t index = i * 17;
    loom_runtime_backing_group_t* entry = nullptr;
    IREE_CHECK_OK(loom_runtime_backing_directory_insert(
        directory, index / LOOM_RUNTIME_BACKING_GROUP_SIZE, &entry));
    entry->in_flight |= UINT64_C(1)
                        << (index % LOOM_RUNTIME_BACKING_GROUP_SIZE);
    indexes.push_back(index);
  }
  return indexes;
}

// Models a warm footprint query over already resident state. Each item is one
// logical slot lookup and mark plus the corresponding cleanup lookup.
static void BM_MarkResidentSlots(benchmark::State& state) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  const auto indexes = Populate(state.range(0), &directory);
  for (auto _ : state) {
    for (const iree_host_size_t index : indexes) {
      loom_runtime_backing_group_t* entry = nullptr;
      IREE_CHECK_OK(loom_runtime_backing_directory_insert(
          &directory, index / LOOM_RUNTIME_BACKING_GROUP_SIZE, &entry));
      benchmark::DoNotOptimize(entry);
      entry->selected |= UINT64_C(1)
                         << (index % LOOM_RUNTIME_BACKING_GROUP_SIZE);
    }
    for (const iree_host_size_t index : indexes) {
      auto* entry = loom_runtime_backing_directory_find(
          &directory, index / LOOM_RUNTIME_BACKING_GROUP_SIZE);
      benchmark::DoNotOptimize(entry);
      entry->selected &=
          ~(UINT64_C(1) << (index % LOOM_RUNTIME_BACKING_GROUP_SIZE));
    }
  }
  state.SetItemsProcessed(state.iterations() * indexes.size());
  loom_runtime_backing_directory_deinitialize(&directory);
}
BENCHMARK(BM_MarkResidentSlots)->Arg(16)->Arg(256)->Arg(4096);

// Models a capacity probe that discovers missing slots and is then denied.
// The live directory must return to its prior footprint after each query.
static void BM_DeniedSparseGrowth(benchmark::State& state) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  auto live_indexes = Populate(state.range(0), &directory);
  const iree_host_size_t missing_count = state.range(1);
  const iree_host_size_t missing_origin = state.range(0) * 17 + 1;
  for (auto _ : state) {
    for (iree_host_size_t i = 0; i < missing_count; ++i) {
      const iree_host_size_t index = missing_origin + i * 17;
      loom_runtime_backing_group_t* entry = nullptr;
      IREE_CHECK_OK(loom_runtime_backing_directory_insert(
          &directory, index / LOOM_RUNTIME_BACKING_GROUP_SIZE, &entry));
      entry->pending_commit |= UINT64_C(1)
                               << (index % LOOM_RUNTIME_BACKING_GROUP_SIZE);
    }
    for (iree_host_size_t i = 0; i < missing_count; ++i) {
      const iree_host_size_t index = missing_origin + i * 17;
      auto* entry = loom_runtime_backing_directory_find(
          &directory, index / LOOM_RUNTIME_BACKING_GROUP_SIZE);
      entry->pending_commit &=
          ~(UINT64_C(1) << (index % LOOM_RUNTIME_BACKING_GROUP_SIZE));
      loom_runtime_backing_directory_prune(&directory, entry);
    }
    IREE_CHECK_OK(loom_runtime_backing_directory_shrink(&directory));
  }
  benchmark::DoNotOptimize(live_indexes);
  state.SetItemsProcessed(state.iterations() * missing_count);
  loom_runtime_backing_directory_deinitialize(&directory);
}
BENCHMARK(BM_DeniedSparseGrowth)->Args({16, 16})->Args({256, 256});

}  // namespace
