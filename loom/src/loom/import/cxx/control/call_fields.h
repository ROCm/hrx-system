// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_CALL_FIELDS_H_
#define LOOM_IMPORT_CXX_CONTROL_CALL_FIELDS_H_

#include <memory>
#include <span>
#include <unordered_map>

#include "iree/base/internal/arena.h"
#include "loom/import/cxx/control/analysis.h"

namespace loom::cxx_import {

// One resolved edge of the source graph, produced during callable discovery.
struct AnalyzedCall {
  // Owner of the call and its source-origin mappings.
  ControlFlow* caller;
  // Retained source call within caller.
  const DirectCall* call;
  // Resolved definition supplying the field contract.
  ControlFlow* callee;
};

// Resolves outputs, source availability, and input requirements in that order.
// Acyclic definitions run once per phase; only recursive components iterate.
// Every function and edge has already passed source admission and identity
// propagation. The only status failure is allocation by the shared SCC utility.
iree_status_t resolve_call_fields(
    const std::unordered_map<cxx::FunctionSymbol*,
                             std::unique_ptr<ControlFlow>>& functions,
    std::span<const AnalyzedCall> calls, iree_arena_block_pool_t* block_pool);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_CALL_FIELDS_H_
