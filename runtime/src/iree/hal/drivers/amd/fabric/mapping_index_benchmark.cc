// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/fabric/mapping_index.h"

namespace {

// Measures host metadata only, including prepare and commit. Native mapping,
// queue admission, synchronization, and payload I/O are deliberately excluded.
class MappingIndex {
 public:
  ~MappingIndex() { ClearStorage(); }

  amdf_memory_mapping_update_t Mapping(uint64_t position) {
    return {
        .target_byte_offset = position * 4096,
        .byte_length = 4096,
        .memory = reinterpret_cast<amdf_memory_t*>(&source_),
        .memory_access_ordinal = 0,
        .access = AMDF_MEMORY_ACCESS_READ,
        .source_byte_offset = 0,
    };
  }

  void Apply(size_t count, const amdf_memory_mapping_update_t* native,
             const iree_hal_memory_mapping_update_t* updates) {
    IREE_CHECK_OK(iree_hal_amd_fabric_mapping_index_prepare(
        &index_, count, native, iree_allocator_system()));
    iree_hal_amd_fabric_mapping_index_apply(&index_, count, updates, native);
  }

  void Fill(const std::vector<uint64_t>& positions) {
    const auto update = Update();
    for (uint64_t position : positions) {
      const auto native = Mapping(position);
      Apply(1, &native, &update);
    }
  }

  iree_hal_memory_mapping_update_t Update() {
    iree_hal_memory_mapping_update_t update = {};
    update.source.mapping_source.descriptor = &source_;
    return update;
  }

  void ClearStorage() {
    iree_hal_amd_fabric_mapping_index_deinitialize(&index_,
                                                   iree_allocator_system());
  }

  const iree_hal_amd_fabric_mapping_index_t* index() const { return &index_; }

 private:
  // Production metadata; no alternate benchmark implementation.
  iree_hal_amd_fabric_mapping_index_t index_ = {};
  // Stable borrowed identity, never dereferenced by the index.
  iree_hal_amd_fabric_mapping_source_t source_ = {};
};

std::vector<uint64_t> Positions(size_t count, int order = 0) {
  std::vector<uint64_t> positions(count);
  std::iota(positions.begin(), positions.end(), 0);
  if (order == 1) {
    std::reverse(positions.begin(), positions.end());
  }
  if (order == 2) {
    std::mt19937 random(42);
    std::shuffle(positions.begin(), positions.end(), random);
  }
  return positions;
}

// One-range calls build the full index. Destruction is outside the timed
// region; SetItemsProcessed reports inserted ranges rather than whole index
// builds.
void BM_IncrementalInsertion(benchmark::State& state) {
  const size_t count = state.range(0);
  const auto positions = Positions(count, state.range(1));
  MappingIndex index;
  for (auto _ : state) {
    index.Fill(positions);
    state.PauseTiming();
    index.ClearStorage();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_IncrementalInsertion)
    ->ArgsProduct({{64, 512, 4096, 16384}, {0, 1, 2}});

void BM_Query(benchmark::State& state) {
  const uint64_t count = state.range(0);
  MappingIndex index;
  index.Fill(Positions(count));
  uint64_t position = 1;
  iree_hal_amd_fabric_virtual_mapping_t mapping;
  for (auto _ : state) {
    position = position * UINT64_C(6364136223846793005) + 1;
    benchmark::DoNotOptimize(iree_hal_amd_fabric_mapping_index_query(
        index.index(), (position % count) * 4096 + 123, 4096, &mapping));
    benchmark::DoNotOptimize(mapping);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Query)->Arg(64)->Arg(512)->Arg(4096)->Arg(16384);

// Repeated exact replacement of a fixed cohort. This exercises node reuse and
// sorted-batch locality without changing the live run count or allocating.
void BM_Replacement(benchmark::State& state) {
  const size_t count = state.range(0);
  const size_t batch_count = state.range(1);
  MappingIndex index;
  index.Fill(Positions(count));
  std::vector<amdf_memory_mapping_update_t> native(batch_count);
  std::vector<iree_hal_memory_mapping_update_t> updates(batch_count,
                                                        index.Update());
  for (size_t i = 0; i < batch_count; ++i) {
    native[i] = index.Mapping(i);
  }
  for (auto _ : state) {
    for (auto& mapping : native) {
      mapping.access ^= AMDF_MEMORY_ACCESS_WRITE;
    }
    index.Apply(batch_count, native.data(), updates.data());
  }
  state.SetItemsProcessed(state.iterations() * batch_count);
}
BENCHMARK(BM_Replacement)->ArgsProduct({{64, 512, 4096, 16384}, {1, 48}});

// Includes removal and coalescing work but excludes preparation of the initial
// mappings and release of pooled metadata. The native detach call is excluded.
void BM_Clear(benchmark::State& state) {
  const size_t count = state.range(0);
  const auto positions = Positions(count);
  MappingIndex index;
  auto native = index.Mapping(0);
  native.byte_length = count * 4096;
  native.memory = nullptr;
  const iree_hal_memory_mapping_update_t update = {};
  for (auto _ : state) {
    state.PauseTiming();
    index.Fill(positions);
    state.ResumeTiming();
    index.Apply(1, &native, &update);
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_Clear)->Arg(64)->Arg(512)->Arg(4096)->Arg(16384);

}  // namespace
