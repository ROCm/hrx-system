// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_LIFETIME_ORIGINS_H_
#define LOOM_IMPORT_CXX_LIFETIME_ORIGINS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstddef>
#include <unordered_map>
#include <variant>
#include <vector>

#include "loom/import/cxx/lifetime/destruction.h"

namespace loom::cxx_import {

class Types;

// Nontrivial result-object and reference lifetime facts from the owning source
// walk. Construction syntax and its implicit conversion can describe the same
// object; equal payloads from separate constructions never establish identity.
// Source nodes identify construction sites, not dynamic evaluation instances.
class TemporaryOrigins {
 public:
  TemporaryOrigins(cxx::TranslationUnit& unit, Types& types)
      : unit_(unit), types_(types) {}

  // Visits one expression after its evaluated children. Forwarding consumes
  // their retained facts rather than walking the source expression again.
  void retain(cxx::ExpressionAST* expression);
  // Extends the materializations selected by one reference initializer. This
  // consumes the retained lifetime graph, including both conditional arms.
  void extend(cxx::ExpressionAST* initializer);
  // A named aggregate owns the temporaries bound to its reference members,
  // independently of whether the aggregate itself has a destructor.
  void extend_members(cxx::ExpressionAST* initializer);
  // Propagates marked roots over the completed postorder dependency graph.
  // Runs once after the owning source walk and before cleanup queries.
  void finish();

  // Nontrivial result supplied by a class/array prvalue or initializer wrapper.
  // A glvalue or subobject access does not supply an owning result.
  cxx::ExpressionAST* result(cxx::ExpressionAST* expression) const;
  // Complete temporary reached through materialization, without changing its
  // lifetime. Reference-returning function calls are not lifetime extenders.
  cxx::ExpressionAST* materialization(cxx::ExpressionAST* expression) const;
  // Whether a reference declaration extends this construction's lifetime.
  bool extended(cxx::ExpressionAST* origin) const;
  // Frontend-selected destruction retained when the origin was established.
  const Destruction& destruction(cxx::ExpressionAST* origin) const;

 private:
  struct Construction {
    // Selected complete-object destruction, including any array extent.
    Destruction destruction;
    // Lifetime dependencies of references initialized inside this result.
    size_t members = 0;
  };
  struct Dependencies {
    // First member or conditional alternative's one-based lifetime node.
    size_t first;
    // Second member or conditional alternative's one-based lifetime node.
    size_t second;
  };
  struct Lifetime {
    // Construction or joined dependencies, each pointing to earlier nodes.
    std::variant<Construction, Dependencies> value;
    // Whether a lexical scope owns this node and its dependent cleanups.
    bool extended = false;
  };
  struct Facts {
    // Complete owning result, absent on a glvalue or scalar expression.
    cxx::ExpressionAST* result = nullptr;
    // One complete materialized object, absent for a selected glvalue.
    cxx::ExpressionAST* materialization = nullptr;
    // Shared lifetime-extension path; forwarding introduces no additional node.
    size_t lifetime = 0;
    // New reference bindings in the result's aggregate initialization. Copies
    // and glvalues carry no new member lifetimes.
    size_t members = 0;
  };

  Facts facts(cxx::ExpressionAST* expression) const;
  Facts classify(cxx::ExpressionAST* expression);
  size_t aggregate_members(cxx::ExpressionAST* initializer,
                           cxx::List<cxx::ExpressionAST*>* elements,
                           cxx::VariableSymbol* implicit_element,
                           bool extend_references);
  size_t combine(size_t first, size_t second);
  void extend(size_t lifetime);

  // Verified source type/category facts; source storage outlives this analysis.
  cxx::TranslationUnit& unit_;
  // Canonical admitted type partitions, shared with control and value lowering.
  Types& types_;
  // Sparse facts; ordinary scalar expressions allocate no entries.
  std::unordered_map<cxx::ExpressionAST*, Facts> expressions_;
  // One-based nodes for constructions and joins; zero denotes no extension.
  std::vector<Lifetime> lifetimes_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_LIFETIME_ORIGINS_H_
