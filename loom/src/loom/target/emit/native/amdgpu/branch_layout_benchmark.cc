// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/emit/native/amdgpu/branch_layout.h"

namespace {

static void AbortOnError(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

static void BM_SharedLongEdges(benchmark::State& state) {
  const uint32_t anchor_count = static_cast<uint32_t>(state.range(0));
  const uint32_t edge_count = static_cast<uint32_t>(state.range(1));
  const uint64_t anchor_spacing = static_cast<uint64_t>(state.range(2));
  const uint64_t byte_length = (uint64_t{anchor_count} + 1u) * anchor_spacing;
  const loom_amdgpu_branch_layout_block_t target_block = {
      .byte_offset = byte_length,
  };
  std::vector<loom_amdgpu_branch_layout_input_edge_t> edges(edge_count);
  for (uint32_t i = 0; i < edge_count; ++i) {
    edges[i] = {
        .source_byte_offset = uint64_t{i} * 4u,
        .target_block_index = 0,
    };
  }
  std::vector<loom_amdgpu_branch_layout_anchor_t> anchors(anchor_count);
  for (uint32_t i = 0; i < anchor_count; ++i) {
    anchors[i] = {
        .byte_offset = uint64_t{i + 1u} * anchor_spacing,
        .packet_index = i,
    };
  }
  const loom_amdgpu_branch_layout_input_t input = {
      .byte_length = byte_length,
      .blocks = &target_block,
      .block_count = 1,
      .edges = edges.data(),
      .edge_count = edges.size(),
      .anchors = anchors.data(),
      .anchor_count = anchors.size(),
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_host_size_t island_count = 0;
  iree_host_size_t group_count = 0;
  for (auto _ : state) {
    loom_amdgpu_branch_layout_t layout = {};
    AbortOnError(loom_amdgpu_branch_layout_build(&input, &arena, &layout));
    island_count = layout.island_count;
    group_count = layout.group_count;
    benchmark::DoNotOptimize(layout.edges);
    benchmark::DoNotOptimize(layout.islands);
    state.PauseTiming();
    iree_arena_reset(&arena);
    state.ResumeTiming();
  }
  state.counters["anchors"] = anchor_count;
  state.counters["edges"] = edge_count;
  state.counters["islands"] = island_count;
  state.counters["groups"] = group_count;
  state.SetItemsProcessed(state.iterations() * island_count);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

BENCHMARK(BM_SharedLongEdges)
    ->Args({4096, 64, 64})
    ->Args({8192, 128, 64})
    ->Args({16384, 256, 64})
    ->Args({318000, 5000, 16});

}  // namespace
