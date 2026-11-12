// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
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
    remove_dependencies =
        Resolve<decltype(remove_dependencies)>("hipGraphRemoveDependencies");
    add_host = Resolve<decltype(add_host)>("hipGraphAddHostNode");
    clone = Resolve<decltype(clone)>("hipGraphClone");
    instantiate =
        Resolve<decltype(instantiate)>("hipGraphInstantiateWithFlags");
    destroy_executable =
        Resolve<decltype(destroy_executable)>("hipGraphExecDestroy");
    update = Resolve<decltype(update)>("hipGraphExecUpdate");
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
  // Removes a final dependency to form a late-rejected update.
  decltype(&hipGraphRemoveDependencies) remove_dependencies = nullptr;
  // Publishes a host node whose parameters differ between replacements.
  decltype(&hipGraphAddHostNode) add_host = nullptr;
  // Creates an independently mutable template from the supplied source.
  decltype(&hipGraphClone) clone = nullptr;
  // Compiles the source before measuring executable replacement.
  decltype(&hipGraphInstantiateWithFlags) instantiate = nullptr;
  // Releases the final executable outside the update interval.
  decltype(&hipGraphExecDestroy) destroy_executable = nullptr;
  // Validates, compiles, and publishes a whole-graph replacement.
  decltype(&hipGraphExecUpdate) update = nullptr;
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

enum class Topology { kIndependent, kChain, kFanIn, kFanOut };
enum class Edges { kEmbedded, kBatch, kBatchReverse, kBatchShuffled, kSingle };
enum class Operation { kConstruct, kClone, kRoots };

class GraphInput {
 public:
  GraphInput(size_t count, Topology topology, Edges edges,
             uint32_t* terminal_host_value = nullptr)
      : topology_(topology),
        edges_(edges),
        terminal_host_value_(terminal_host_value),
        nodes_(count),
        from_(topology == Topology::kIndependent ? 0 : count - 1),
        to_(from_.size()),
        roots_(count),
        edge_order_(from_.size()) {
    std::iota(edge_order_.begin(), edge_order_.end(), size_t{0});
    if (edges == Edges::kBatchReverse) {
      std::reverse(edge_order_.begin(), edge_order_.end());
    } else if (edges == Edges::kBatchShuffled) {
      std::mt19937 random(0);
      std::shuffle(edge_order_.begin(), edge_order_.end(), random);
    }
  }

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
        } else if (topology_ == Topology::kFanOut && i) {
          dependencies = nodes_.data();
          dependency_count = 1;
        }
      }
      if (terminal_host_value_ && i + 1 == nodes_.size()) {
        hipHostNodeParams parameters = {
            .fn = +[](void* value) { ++*static_cast<uint32_t*>(value); },
            .userData = terminal_host_value_,
        };
        Check(api.add_host(&nodes_[i], graph, dependencies, dependency_count,
                           &parameters));
      } else {
        Check(api.add_empty(&nodes_[i], graph, dependencies, dependency_count));
      }
    }
    if (edges_ != Edges::kEmbedded) {
      for (size_t i = 0; i < from_.size(); ++i) {
        const size_t edge_index = edge_order_[i];
        from_[i] = topology_ == Topology::kFanOut ? nodes_.front()
                                                  : nodes_[edge_index];
        to_[i] = topology_ == Topology::kFanIn ? nodes_.back()
                                               : nodes_[edge_index + 1];
      }
      if (edges_ != Edges::kSingle) {
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
    const size_t expected_roots =
        topology_ == Topology::kChain || topology_ == Topology::kFanOut
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

  // Changes only the final node's predecessor set and returns its error handle.
  hipGraphNode_t ChangeLastDependency(hipGraph_t graph) {
    hipGraphNode_t from = topology_ == Topology::kFanOut
                              ? nodes_.front()
                              : nodes_[nodes_.size() - 2];
    hipGraphNode_t to = nodes_.back();
    if (topology_ == Topology::kIndependent) {
      Check(Api().add_dependencies(graph, &from, &to, 1));
    } else {
      Check(Api().remove_dependencies(graph, &from, &to, 1));
    }
    return to;
  }

 private:
  // Dependency shape shared by the construction, clone, and query rows.
  Topology topology_;
  // Public insertion form; all forms construct the same topology.
  Edges edges_;
  // Optional caller-owned value giving the final host node distinct parameters.
  uint32_t* terminal_host_value_;
  // Caller-owned node handles, reused across timed template constructions.
  std::vector<hipGraphNode_t> nodes_;
  // Caller-owned predecessor array supplied to additional-edge insertion.
  std::vector<hipGraphNode_t> from_;
  // Caller-owned successor array supplied to additional-edge insertion.
  std::vector<hipGraphNode_t> to_;
  // Caller-owned root-query result array, allocated outside timing.
  std::vector<hipGraphNode_t> roots_;
  // Fixed insertion permutation, prepared outside timing for every topology.
  std::vector<size_t> edge_order_;
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
GRAPH_TEMPLATE_OPERATIONS(FanOut, Embedded);
GRAPH_TEMPLATE_OPERATIONS(FanOut, Batch);
GRAPH_TEMPLATE_OPERATIONS(FanOut, Single);

#define GRAPH_TEMPLATE_BATCH_ORDER(topology, edges)     \
  GRAPH_TEMPLATE_BENCHMARK(Construct, topology, edges); \
  GRAPH_TEMPLATE_BENCHMARK(Clone, topology, edges)

GRAPH_TEMPLATE_BATCH_ORDER(Chain, BatchReverse);
GRAPH_TEMPLATE_BATCH_ORDER(Chain, BatchShuffled);
GRAPH_TEMPLATE_BATCH_ORDER(FanIn, BatchReverse);
GRAPH_TEMPLATE_BATCH_ORDER(FanIn, BatchShuffled);
GRAPH_TEMPLATE_BATCH_ORDER(FanOut, BatchReverse);
GRAPH_TEMPLATE_BATCH_ORDER(FanOut, BatchShuffled);

#undef GRAPH_TEMPLATE_BATCH_ORDER

#undef GRAPH_TEMPLATE_OPERATIONS
#undef GRAPH_TEMPLATE_BENCHMARK

template <bool accepted, Topology topology, Edges edges>
void GraphUpdate(benchmark::State& state) {
  auto& api = Api();
  uint32_t values[3] = {};
  GraphInput original(state.range(0), topology, Edges::kBatch, &values[0]);
  GraphInput first(state.range(0), topology, edges, &values[1]);
  // Embedded predecessors and reverse batch insertion both produce ascending
  // lists. Alternate with the opposite order so timed updates compare sets,
  // including transitions between embedded and separately attached edges.
  constexpr Edges alternate_edges =
      edges == Edges::kEmbedded || edges == Edges::kBatchReverse
          ? Edges::kBatch
          : Edges::kBatchReverse;
  GraphInput second(state.range(0), topology, alternate_edges, &values[2]);
  hipGraph_t source = original.Build();
  hipGraph_t replacements[] = {first.Build(), second.Build()};
  original.Verify(source);
  first.Verify(replacements[0]);
  second.Verify(replacements[1]);
  hipGraphNode_t expected_errors[2] = {};
  if constexpr (!accepted) {
    expected_errors[0] = first.ChangeLastDependency(replacements[0]);
    expected_errors[1] = second.ChangeLastDependency(replacements[1]);
  }
  hipGraphExec_t executable = nullptr;
  Check(api.instantiate(&executable, source, 0));
  const auto update = [&](size_t index) {
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    hipError_t error =
        api.update(executable, replacements[index], &error_node, &result);
    if constexpr (accepted) {
      Check(error);
      Require(result == hipGraphExecUpdateSuccess && error_node == nullptr,
              "compatible update rejected");
    } else {
      Require(error == hipErrorGraphExecUpdateFailure &&
                  result == hipGraphExecUpdateErrorTopologyChanged &&
                  error_node == expected_errors[index],
              "incompatible update lost its failure boundary");
    }
  };
  update(0);
  update(1);
  size_t index = 0;
  for (auto _ : state) {
    // Alternate real parameter changes, including replacement and old-state
    // release in timing. No graph is launched by these rows.
    update(index);
    index ^= 1;
  }
  Check(api.destroy_executable(executable));
  for (hipGraph_t graph : replacements) {
    Check(api.destroy(graph));
  }
  Check(api.destroy(source));
  state.counters["nodes"] = state.range(0);
  state.counters["edges"] = original.edge_count();
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

#define GRAPH_UPDATE_BENCHMARK(accepted, name, topology, edges)    \
  BENCHMARK_TEMPLATE(GraphUpdate, accepted, Topology::k##topology, \
                     Edges::k##edges)                              \
      ->Name("GraphUpdate/" name "/" #topology "/" #edges)         \
      ->ArgName("nodes")                                           \
      ->Arg(16)                                                    \
      ->Arg(256)                                                   \
      ->Arg(1024)                                                  \
      ->Arg(4096)                                                  \
      ->Unit(benchmark::kMicrosecond)

#define GRAPH_UPDATE_SHAPE(topology, edges)                  \
  GRAPH_UPDATE_BENCHMARK(true, "Accepted", topology, edges); \
  GRAPH_UPDATE_BENCHMARK(false, "Rejected", topology, edges)

GRAPH_UPDATE_SHAPE(Independent, Embedded);
GRAPH_UPDATE_SHAPE(Chain, Embedded);
GRAPH_UPDATE_SHAPE(Chain, Batch);
GRAPH_UPDATE_SHAPE(Chain, BatchReverse);
GRAPH_UPDATE_SHAPE(Chain, BatchShuffled);
GRAPH_UPDATE_SHAPE(FanIn, Embedded);
GRAPH_UPDATE_SHAPE(FanIn, Batch);
GRAPH_UPDATE_SHAPE(FanIn, BatchReverse);
GRAPH_UPDATE_SHAPE(FanIn, BatchShuffled);
GRAPH_UPDATE_SHAPE(FanOut, Embedded);
GRAPH_UPDATE_SHAPE(FanOut, Batch);
GRAPH_UPDATE_SHAPE(FanOut, BatchReverse);
GRAPH_UPDATE_SHAPE(FanOut, BatchShuffled);

#undef GRAPH_UPDATE_SHAPE
#undef GRAPH_UPDATE_BENCHMARK

}  // namespace
