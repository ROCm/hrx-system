// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_CFG_GRAPH_TEST_UTIL_H_
#define LOOM_UTIL_CFG_GRAPH_TEST_UTIL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "loom/util/cfg_graph.h"

namespace loom::testing {

// Adjacency, DFS, SCC and entry-reachability fixture for graph-only analyses.
// Authored IR and extraction contracts are exercised by loom-check fixtures.
class CfgGraph {
 public:
  explicit CfgGraph(const std::vector<std::vector<uint16_t>>& successors)
      : blocks_(successors.size()) {
    std::vector<std::vector<uint32_t>> incoming(blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i) {
      auto& block = blocks_[i];
      block.component = UINT16_MAX;
      block.preorder = UINT16_MAX;
      block.parent = UINT16_MAX;
      block.successor_start = successor_indices_.size();
      block.successor_edge_start = successor_indices_.size();
      block.successor_count = successors[i].size();
      for (uint16_t target : successors[i]) {
        graph_.backward_edge_count += target <= i;
        uint32_t edge = edges_.size();
        incoming[target].push_back(edge);
        loom_cfg_edge_info_t edge_info = {};
        edge_info.source_block_index = static_cast<uint16_t>(i);
        edge_info.target_block_index = target;
        edge_info.successor_index =
            static_cast<uint16_t>(edge - block.successor_start);
        edges_.push_back(edge_info);
        successor_indices_.push_back(target);
        successor_edges_.push_back(edge);
      }
    }
    for (size_t i = 0; i < blocks_.size(); ++i) {
      auto& block = blocks_[i];
      block.predecessor_start = predecessor_indices_.size();
      block.predecessor_edge_start = predecessor_indices_.size();
      block.predecessor_count = incoming[i].size();
      for (uint32_t edge : incoming[i]) {
        predecessor_indices_.push_back(edges_[edge].source_block_index);
        predecessor_edges_.push_back(edge);
      }
    }
    if (!blocks_.empty()) {
      std::vector<std::pair<uint16_t, size_t>> pending{{0, 0}};
      uint16_t preorder = 0;
      blocks_[0].reachable = true;
      blocks_[0].preorder = preorder++;
      while (!pending.empty()) {
        auto& frame = pending.back();
        if (frame.second == successors[frame.first].size()) {
          blocks_[frame.first].preorder_end = preorder;
          reverse_postorder_.push_back(frame.first);
          pending.pop_back();
          continue;
        }
        uint16_t target = successors[frame.first][frame.second++];
        if (!blocks_[target].reachable) {
          blocks_[target].reachable = true;
          blocks_[target].preorder = preorder++;
          blocks_[target].parent = frame.first;
          pending.emplace_back(target, 0);
        } else if (blocks_[target].preorder_end == 0) {
          blocks_[target].is_dfs_backedge_target = true;
          graph_.has_cycles = true;
        }
      }
      std::reverse(reverse_postorder_.begin(), reverse_postorder_.end());
    }
    // The reverse-graph traversal completes the same SCC contract as the
    // production graph builder, independently of its Tarjan implementation.
    uint16_t component_count = 0;
    for (uint16_t root : reverse_postorder_) {
      if (blocks_[root].component != UINT16_MAX) {
        continue;
      }
      std::vector<uint16_t> members{root};
      blocks_[root].component = component_count;
      for (size_t i = 0; i < members.size(); ++i) {
        for (uint32_t edge : incoming[members[i]]) {
          uint16_t predecessor = edges_[edge].source_block_index;
          if (blocks_[predecessor].reachable &&
              blocks_[predecessor].component == UINT16_MAX) {
            blocks_[predecessor].component = component_count;
            members.push_back(predecessor);
          }
        }
      }
      const bool cyclic =
          members.size() > 1 ||
          std::find(successors[root].begin(), successors[root].end(), root) !=
              successors[root].end();
      for (uint16_t member : members) {
        blocks_[member].component_is_cyclic = cyclic;
      }
      ++component_count;
    }
    for (auto& block : blocks_) {
      if (block.reachable) {
        block.component = component_count - block.component - 1;
      }
    }
    graph_.blocks = blocks_.data();
    graph_.block_count = blocks_.size();
    graph_.edges = edges_.data();
    graph_.edge_count = edges_.size();
    graph_.successor_indices = successor_indices_.data();
    graph_.predecessor_indices = predecessor_indices_.data();
    graph_.successor_edge_indices = successor_edges_.data();
    graph_.predecessor_edge_indices = predecessor_edges_.data();
    graph_.reverse_postorder = {reverse_postorder_.data(),
                                reverse_postorder_.size()};
  }

  CfgGraph(const CfgGraph&) = delete;
  CfgGraph& operator=(const CfgGraph&) = delete;
  CfgGraph(CfgGraph&&) = delete;
  CfgGraph& operator=(CfgGraph&&) = delete;

  const loom_cfg_graph_t* get() const { return &graph_; }

 private:
  // Per-block adjacency and entry reachability.
  std::vector<loom_cfg_block_info_t> blocks_;
  // Stable edge identities and source/target ordinals.
  std::vector<loom_cfg_edge_info_t> edges_;
  // Dense outgoing block adjacency.
  std::vector<uint16_t> successor_indices_;
  // Dense incoming block adjacency.
  std::vector<uint16_t> predecessor_indices_;
  // Dense outgoing edge adjacency.
  std::vector<uint32_t> successor_edges_;
  // Dense incoming edge adjacency.
  std::vector<uint32_t> predecessor_edges_;
  // Reachable blocks in reverse DFS postorder.
  std::vector<uint16_t> reverse_postorder_;
  // Non-owning graph view over the fixture arrays.
  loom_cfg_graph_t graph_ = {};
};

}  // namespace loom::testing

#endif  // LOOM_UTIL_CFG_GRAPH_TEST_UTIL_H_
