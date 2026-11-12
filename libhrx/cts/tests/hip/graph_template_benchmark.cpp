// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void Check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "HIP operation failed: %d\n",
                 static_cast<int>(result));
    std::abort();
  }
}

// Public graph operations use one initialized frontend/context for the process.
// These rows construct CPU-side templates and submit no GPU work.
struct GraphApi {
  GraphApi() {
    if (!library.Open()) {
      Require(false, library.error().c_str());
    }
    Check(Resolve<decltype(&hipInit)>("hipInit")(0));
    create = Resolve<decltype(create)>("hipGraphCreate");
    destroy = Resolve<decltype(destroy)>("hipGraphDestroy");
    add_empty = Resolve<decltype(add_empty)>("hipGraphAddEmptyNode");
    add_dependencies =
        Resolve<decltype(add_dependencies)>("hipGraphAddDependencies");
    clone = Resolve<decltype(clone)>("hipGraphClone");
    get_nodes = Resolve<decltype(get_nodes)>("hipGraphGetNodes");
    get_edges = Resolve<decltype(get_edges)>("hipGraphGetEdges");
    get_roots = Resolve<decltype(get_roots)>("hipGraphGetRootNodes");
  }

  ~GraphApi() {
    Check(Resolve<decltype(&hipHALDeinit)>("hipHALDeinit")());
    if (!library.Close()) {
      Require(false, library.error().c_str());
    }
  }

  template <typename T>
  T Resolve(const char* name) {
    T function = library.Resolve<T>(name);
    Require(function != nullptr, library.error().c_str());
    return function;
  }

  // Exact build-selected frontend; symbol lookup stays outside timing.
  hrx::hip::testing::HipDso library;
  // Allocates one empty template in the current context.
  decltype(&hipGraphCreate) create = nullptr;
  // Releases a template outside the measured construction/clone interval.
  decltype(&hipGraphDestroy) destroy = nullptr;
  // Publishes a dependency-only node with embedded incoming edges.
  decltype(&hipGraphAddEmptyNode) add_empty = nullptr;
  // Attaches edges after their endpoint nodes exist.
  decltype(&hipGraphAddDependencies) add_dependencies = nullptr;
  // Creates an independently mutable template from the supplied source.
  decltype(&hipGraphClone) clone = nullptr;
  // Queries public node count for untimed fixture verification.
  decltype(&hipGraphGetNodes) get_nodes = nullptr;
  // Queries public edge count for untimed fixture verification.
  decltype(&hipGraphGetEdges) get_edges = nullptr;
  // Enumerates roots, including dependencies attached after node construction.
  decltype(&hipGraphGetRootNodes) get_roots = nullptr;
};

GraphApi& Api() {
  static GraphApi api;
  return api;
}

enum class Topology { kIndependent, kChain, kFanIn };
enum class Edges { kEmbedded, kBatch, kSingle };
enum class Operation { kConstruct, kClone, kRoots };

class GraphInput {
 public:
  GraphInput(size_t count, Topology topology, Edges edges)
      : topology_(topology),
        edges_(edges),
        nodes_(count),
        from_(topology == Topology::kIndependent ? 0 : count - 1),
        to_(from_.size()),
        roots_(count) {}

  hipGraph_t Build() {
    auto& api = Api();
    hipGraph_t graph = nullptr;
    Check(api.create(&graph, 0));
    for (size_t i = 0; i < nodes_.size(); ++i) {
      hipGraphNode_t* dependencies = nullptr;
      size_t dependency_count = 0;
      if (edges_ == Edges::kEmbedded) {
        if (topology_ == Topology::kChain && i) {
          dependencies = &nodes_[i - 1];
          dependency_count = 1;
        } else if (topology_ == Topology::kFanIn && i + 1 == nodes_.size()) {
          dependencies = nodes_.data();
          dependency_count = i;
        }
      }
      Check(api.add_empty(&nodes_[i], graph, dependencies, dependency_count));
    }
    if (edges_ != Edges::kEmbedded) {
      for (size_t i = 0; i < from_.size(); ++i) {
        from_[i] = nodes_[i];
        to_[i] = topology_ == Topology::kChain ? nodes_[i + 1] : nodes_.back();
      }
      if (edges_ == Edges::kBatch) {
        Check(api.add_dependencies(graph, from_.data(), to_.data(),
                                   from_.size()));
      } else {
        for (size_t i = 0; i < from_.size(); ++i) {
          Check(api.add_dependencies(graph, &from_[i], &to_[i], 1));
        }
      }
    }
    return graph;
  }

  void Verify(hipGraph_t graph) {
    size_t node_count = 0;
    Check(Api().get_nodes(graph, nullptr, &node_count));
    Require(node_count == nodes_.size(), "graph node count mismatch");
    size_t edge_count = 0;
    Check(Api().get_edges(graph, nullptr, nullptr, &edge_count));
    Require(edge_count == from_.size(), "graph edge count mismatch");
    const size_t expected_roots = topology_ == Topology::kChain
                                      ? 1
                                      : nodes_.size() - (edge_count ? 1 : 0);
    Require(QueryRoots(graph) == expected_roots, "graph root count mismatch");
  }

  size_t QueryRoots(hipGraph_t graph) {
    size_t count = roots_.size();
    Check(Api().get_roots(graph, roots_.data(), &count));
    return count;
  }

  size_t edge_count() const { return from_.size(); }

 private:
  // Dependency shape shared by the construction, clone, and query rows.
  Topology topology_;
  // Public insertion form; all forms construct the same topology.
  Edges edges_;
  // Caller-owned node handles, reused across timed template constructions.
  std::vector<hipGraphNode_t> nodes_;
  // Caller-owned predecessor array supplied to additional-edge insertion.
  std::vector<hipGraphNode_t> from_;
  // Caller-owned successor array supplied to additional-edge insertion.
  std::vector<hipGraphNode_t> to_;
  // Caller-owned root-query result array, allocated outside timing.
  std::vector<hipGraphNode_t> roots_;
};

template <Operation operation, Topology topology, Edges edges>
void GraphTemplate(benchmark::State& state) {
  auto& api = Api();
  GraphInput input(state.range(0), topology, edges);
  hipGraph_t source = input.Build();
  input.Verify(source);
  if constexpr (operation == Operation::kClone) {
    hipGraph_t copy = nullptr;
    Check(api.clone(&copy, source));
    input.Verify(copy);
    Check(api.destroy(copy));
  }
  for (auto _ : state) {
    if constexpr (operation == Operation::kRoots) {
      benchmark::DoNotOptimize(input.QueryRoots(source));
    } else {
      hipGraph_t graph = nullptr;
      if constexpr (operation == Operation::kConstruct) {
        graph = input.Build();
      } else {
        Check(api.clone(&graph, source));
      }
      benchmark::DoNotOptimize(graph);
      state.PauseTiming();
      Check(api.destroy(graph));
      state.ResumeTiming();
    }
  }
  Check(api.destroy(source));
  state.counters["nodes"] = state.range(0);
  state.counters["edges"] = input.edge_count();
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

#define GRAPH_TEMPLATE_BENCHMARK(operation, topology, edges)       \
  BENCHMARK_TEMPLATE(GraphTemplate, Operation::k##operation,       \
                     Topology::k##topology, Edges::k##edges)       \
      ->Name("GraphTemplate/" #operation "/" #topology "/" #edges) \
      ->ArgName("nodes")                                           \
      ->Arg(16)                                                    \
      ->Arg(256)                                                   \
      ->Arg(1024)                                                  \
      ->Arg(4096)                                                  \
      ->Unit(benchmark::kMicrosecond)

#define GRAPH_TEMPLATE_OPERATIONS(topology, edges)      \
  GRAPH_TEMPLATE_BENCHMARK(Construct, topology, edges); \
  GRAPH_TEMPLATE_BENCHMARK(Clone, topology, edges);     \
  GRAPH_TEMPLATE_BENCHMARK(Roots, topology, edges)

GRAPH_TEMPLATE_OPERATIONS(Independent, Embedded);
GRAPH_TEMPLATE_OPERATIONS(Chain, Embedded);
GRAPH_TEMPLATE_OPERATIONS(Chain, Batch);
GRAPH_TEMPLATE_OPERATIONS(Chain, Single);
GRAPH_TEMPLATE_OPERATIONS(FanIn, Embedded);
GRAPH_TEMPLATE_OPERATIONS(FanIn, Batch);
GRAPH_TEMPLATE_OPERATIONS(FanIn, Single);

#undef GRAPH_TEMPLATE_OPERATIONS
#undef GRAPH_TEMPLATE_BENCHMARK

}  // namespace
