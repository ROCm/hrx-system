// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "common/graph.h"
#include "hrx_internal.h"
#include "iree/base/internal/arena.h"

namespace {

enum class Topology {
  kIndependent,
  kChain,
  kDiamond,
  kLayered,
  kHostBoundaries,
};
enum class EdgeStorage { kEmbedded, kAdditional, kReverseAdditional };

// Both frontends accept these dependency shapes. Fixture construction is
// excluded from timing; analysis and returning its arena storage are included.
// No device, command buffer, or dispatch is created by these CPU-only rows.
std::vector<std::pair<uint32_t, uint32_t>> MakeEdges(uint32_t count,
                                                     Topology topology) {
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (uint32_t i = 1; i < count; ++i) {
    switch (topology) {
      case Topology::kIndependent:
        break;
      case Topology::kChain:
      case Topology::kHostBoundaries:
        edges.emplace_back(i - 1, i);
        break;
      case Topology::kDiamond:
        // Four-node diamonds share their join with the next diamond's root.
        if (i % 3 == 0) {
          edges.emplace_back(i - 2, i);
          edges.emplace_back(i - 1, i);
        } else {
          edges.emplace_back(i - i % 3, i);
        }
        break;
      case Topology::kLayered:
        // Each node in a 16-wide layer consumes two distinct preceding nodes.
        if (i >= 16) {
          const uint32_t previous_layer = (i / 16 - 1) * 16;
          edges.emplace_back(previous_layer + i % 16, i);
          edges.emplace_back(previous_layer + (i + 1) % 16, i);
        }
        break;
    }
  }
  return edges;
}

struct NativeGraph {
  using Node = hrx_graph_node_s;
  using Block = hrx_graph_node_block_t;
  using Edge = hrx_graph_edge_t;
  using Schedule = hrx_graph_schedule_t;
  static constexpr auto kNodeType = HRX_GRAPH_NODE_TYPE_INTERNAL_MEMSET;
  static constexpr auto kHostType = HRX_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL;

  static void Link(Edge* edge, Node* from, Node* to, Edge** head) {
    *edge = {.next = *head, .from = from, .to = to};
    *head = edge;
  }

  static iree_status_t Analyze(Block* block, Edge* edges,
                               iree_arena_allocator_t* arena,
                               Schedule* out_schedule) {
    return hrx_graph_schedule_nodes(block, block->count, edges, arena,
                                    out_schedule);
  }
};

struct BindingGraph {
  using Node = iree_hal_streaming_graph_node_t;
  using Block = iree_hal_streaming_node_block_t;
  using Edge = iree_hal_streaming_graph_dependency_t;
  using Schedule = hrx_graph_schedule_t;
  static constexpr auto kNodeType = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET;
  static constexpr auto kHostType =
      IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL;

  static void Link(Edge* edge, Node* from, Node* to, Edge**) {
    *edge = {.next = to->additional_dependencies, .from = from};
    to->additional_dependencies = edge;
  }

  static iree_status_t Analyze(Block* block, Edge*,
                               iree_arena_allocator_t* arena,
                               Schedule* out_schedule) {
    return iree_hal_streaming_graph_schedule_nodes(
        block, block->count, nullptr, 0, nullptr, arena, out_schedule);
  }
};

template <typename Graph>
class GraphInput {
 public:
  GraphInput(uint32_t count,
             const std::vector<std::pair<uint32_t, uint32_t>>& edges,
             Topology topology, EdgeStorage storage) {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_CHECK_OK(iree_arena_allocate(&arena_,
                                      sizeof(Block) + count * sizeof(Node*),
                                      reinterpret_cast<void**>(&block_)));
    std::memset(block_, 0, sizeof(*block_));
    block_->capacity = count;
    block_->count = count;
    std::vector<uint32_t> dependency_counts(count);
    if (storage == EdgeStorage::kEmbedded) {
      for (const auto& edge : edges) {
        ++dependency_counts[edge.second];
      }
    }
    for (uint32_t i = 0; i < count; ++i) {
      Node* node = nullptr;
      IREE_CHECK_OK(iree_arena_allocate(
          &arena_, sizeof(Node) + dependency_counts[i] * sizeof(Node*),
          reinterpret_cast<void**>(&node)));
      std::memset(node, 0, sizeof(*node));
      node->type = topology == Topology::kHostBoundaries && i % 32 == 31
                       ? Graph::kHostType
                       : Graph::kNodeType;
      node->node_index = i;
      block_->nodes[i] = node;
    }
    for (const auto& edge : edges) {
      Node* from = block_->nodes[edge.first];
      Node* to = block_->nodes[edge.second];
      if (storage == EdgeStorage::kEmbedded) {
        to->dependencies[to->dependency_count++] = from;
      } else {
        Edge* added = nullptr;
        IREE_CHECK_OK(iree_arena_allocate(&arena_, sizeof(*added),
                                          reinterpret_cast<void**>(&added)));
        Graph::Link(added, from, to, &edges_);
      }
    }
  }

  ~GraphInput() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t Analyze(iree_arena_allocator_t* arena,
                        typename Graph::Schedule* out_schedule) {
    return Graph::Analyze(block_, edges_, arena, out_schedule);
  }

 private:
  using Node = typename Graph::Node;
  using Block = typename Graph::Block;
  using Edge = typename Graph::Edge;

  // Owns immutable fixture nodes independently of the timed analysis arena.
  iree_arena_block_pool_t pool_;
  // Storage for frontend nodes and their original dependency representation.
  iree_arena_allocator_t arena_;
  // Original creation order accepted by the frontend analyzer.
  Block* block_ = nullptr;
  // Native HRX's graph-wide edges; the binding keeps edges on destination
  // nodes.
  Edge* edges_ = nullptr;
};

template <typename Graph, Topology topology, EdgeStorage storage>
void GraphAnalysis(benchmark::State& state) {
  const uint32_t count = static_cast<uint32_t>(state.range(0));
  auto edges = MakeEdges(count, topology);
  if (storage == EdgeStorage::kReverseAdditional) {
    for (auto& edge : edges) {
      edge.first = count - 1 - edge.first;
      edge.second = count - 1 - edge.second;
    }
  }
  GraphInput<Graph> input(count, edges, topology, storage);
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Validate dependency ordering before timing and warm reusable arena blocks.
  typename Graph::Schedule schedule;
  IREE_CHECK_OK(input.Analyze(&arena, &schedule));
  std::vector<uint32_t> ranks(count);
  for (uint32_t i = 0; i < count; ++i) {
    ranks[schedule.order[i]] = i;
  }
  for (const auto& edge : edges) {
    if (ranks[edge.first] >= ranks[edge.second]) {
      std::fputs("graph analysis violated a dependency\n", stderr);
      std::abort();
    }
  }
  const auto used_bytes = arena.used_allocation_size;
  const auto reserved_bytes = arena.total_allocation_size;
  const auto partition_count = schedule.partition_count;
  iree_arena_reset(&arena);

  for (auto _ : state) {
    IREE_CHECK_OK(input.Analyze(&arena, &schedule));
    benchmark::DoNotOptimize(schedule.order);
    benchmark::DoNotOptimize(schedule.partitions);
    iree_arena_reset(&arena);
  }
  state.counters["nodes"] = count;
  state.counters["edges"] = edges.size();
  state.counters["partitions"] = partition_count;
  state.counters["arena_used_bytes"] = used_bytes;
  state.counters["arena_reserved_bytes"] = reserved_bytes;
  state.SetItemsProcessed(state.iterations() * count);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

#define HRX_GRAPH_ANALYSIS_BENCHMARK(graph, topology, storage)    \
  BENCHMARK_TEMPLATE(GraphAnalysis, graph, Topology::k##topology, \
                     EdgeStorage::k##storage)                     \
      ->Name("GraphAnalysis/" #graph "/" #topology "/" #storage)  \
      ->ArgName("nodes")                                          \
      ->Arg(16)                                                   \
      ->Arg(256)                                                  \
      ->Arg(1024)                                                 \
      ->Arg(4096)                                                 \
      ->Unit(benchmark::kMicrosecond)

HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, Independent, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, Independent, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, Chain, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, Chain, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, Chain, Additional);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, Chain, Additional);
HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, Chain, ReverseAdditional);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, Chain, ReverseAdditional);
HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, Diamond, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, Diamond, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, Layered, Additional);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, Layered, Additional);
HRX_GRAPH_ANALYSIS_BENCHMARK(NativeGraph, HostBoundaries, Embedded);
HRX_GRAPH_ANALYSIS_BENCHMARK(BindingGraph, HostBoundaries, Embedded);

#undef HRX_GRAPH_ANALYSIS_BENCHMARK

}  // namespace
