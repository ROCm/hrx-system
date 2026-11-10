// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/call_fields.h"

#include <algorithm>
#include <vector>

#include "loom/analysis/scc.h"

namespace loom::cxx_import {
namespace {

struct Scratch {
  // SCC result storage, retired before source signature construction.
  iree_arena_allocator_t arena;
  explicit Scratch(iree_arena_block_pool_t* pool) {
    iree_arena_initialize(pool, &arena);
  }
  ~Scratch() { iree_arena_deinitialize(&arena); }
};

}  // namespace

iree_status_t resolve_call_fields(
    const std::unordered_map<cxx::FunctionSymbol*,
                             std::unique_ptr<ControlFlow>>& functions,
    std::span<const AnalyzedCall> calls, iree_arena_block_pool_t* block_pool) {
  const size_t count = std::count_if(
      functions.begin(), functions.end(),
      [](const auto& entry) { return entry.second->has_field_effects(); });
  if (!count) {
    return iree_ok_status();
  }
  std::vector<ControlFlow*> nodes;
  std::unordered_map<ControlFlow*, size_t> ordinals;
  nodes.reserve(count);
  for (const auto& [symbol, analysis] : functions) {
    if (!analysis->has_field_effects()) {
      continue;
    }
    ordinals.emplace(analysis.get(), nodes.size());
    nodes.push_back(analysis.get());
    analysis->bind_field_calls(functions);
  }
  std::vector<std::vector<size_t>> successors(nodes.size());
  for (const auto& call : calls) {
    auto caller = ordinals.find(call.caller);
    auto callee = ordinals.find(call.callee);
    // An empty effect graph has a fixed contract and needs no solver node.
    if (caller != ordinals.end() && callee != ordinals.end()) {
      successors[caller->second].push_back(callee->second);
    }
  }
  Scratch scratch(block_pool);
  const loom_scc_graph_t graph = {
      .node_count = nodes.size(),
      .visit_successors =
          {
              [](void* user_data, size_t node,
                 loom_scc_successor_callback_t successor) -> iree_status_t {
                const auto& edges =
                    *static_cast<const std::vector<std::vector<size_t>>*>(
                        user_data);
                for (size_t callee : edges[node]) {
                  IREE_RETURN_IF_ERROR(
                      successor.fn(successor.user_data, callee));
                }
                return iree_ok_status();
              },
              &successors,
          },
  };
  loom_scc_list_t components;
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, nullptr, &scratch.arena, &components));
  auto propagate = [&](auto update) {
    for (size_t index = 0; index < components.count; ++index) {
      const auto& component = components.values[index];
      bool changed;
      do {
        changed = false;
        for (size_t ordinal :
             std::span(component.nodes, component.node_count)) {
          changed = (nodes[ordinal]->*update)() || changed;
        }
      } while (component.is_cycle && changed);
    }
  };
  propagate(&ControlFlow::propagate_field_outputs);
  for (auto* node : nodes) {
    node->resolve_field_flow();
  }
  propagate(&ControlFlow::propagate_field_inputs);
  return iree_ok_status();
}

}  // namespace loom::cxx_import
