// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_SOURCE_TEST_UTIL_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_SOURCE_TEST_UTIL_H_

#include <vector>

#include "loom/target/arch/amd/xdna/aie2p/program_source.h"

namespace loom::aie2p::testing {

// Resolved immutable requirements at the source-inventory boundary. These
// fixtures exercise closure ownership without parsing or rebuilding native IR;
// authored admission and emitted programs are covered by the Loom tests.
class SourceInventory {
 public:
  explicit SourceInventory(
      const std::vector<std::vector<loom_symbol_id_t>>& edges)
      : functions_(edges.size()), calls_(edges.size()), index_(edges.size()) {
    module_.symbols.count = edges.size();
    for (size_t i = 0; i < edges.size(); ++i) {
      functions_[i].symbol.symbol_id = static_cast<loom_symbol_id_t>(i);
      for (loom_symbol_id_t callee : edges[i]) {
        calls_[i].push_back({.symbol = {.symbol_id = callee}});
      }
      functions_[i].requirements.direct_callees = {.values = calls_[i].data(),
                                                   .count = calls_[i].size()};
      index_[i] = &functions_[i];
    }
    table_.module = &module_;
    table_.functions_by_symbol = index_.data();
  }

  void Reset(iree_arena_allocator_t* arena) {
    table_.arena = arena;
    for (auto& function : functions_) {
      function.helpers_admitted = false;
      function.helpers = {};
      function.program_data = &function.requirements.read_only_data;
    }
  }

  void Admit(size_t root) {
    bool admitted = false;
    IREE_CHECK_OK(loom_aie2p_program_source_admit_helpers(
        &table_, &functions_[root], &admitted));
    if (!admitted) {
      IREE_CHECK_UNREACHABLE();
    }
  }

  const loom_aie2p_program_source_function_t& function(size_t index) const {
    return functions_[index];
  }

  size_t RetainedHelperCount() const {
    size_t count = 0;
    for (const auto& function : functions_) {
      count += function.helpers.count;
    }
    return count;
  }

 private:
  // Immutable symbol domain; every source record is already resolved below.
  loom_module_t module_ = {};
  // Stable requirement records, with per-request closure products resettable.
  std::vector<loom_aie2p_program_source_function_t> functions_;
  // Unique direct dependencies in each function's source order.
  std::vector<std::vector<loom_low_direct_callee_requirement_t>> calls_;
  // Source inventory indexed in the module's symbol domain.
  std::vector<loom_aie2p_program_source_function_t*> index_;
  // Ordinary source owner consuming the resolved requirement records.
  loom_aie2p_program_source_table_t table_ = {};
};

}  // namespace loom::aie2p::testing

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_SOURCE_TEST_UTIL_H_
