// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_LIFETIME_ORIGINS_H_
#define LOOM_IMPORT_CXX_LIFETIME_ORIGINS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>

#include <cstddef>
#include <unordered_map>
#include <variant>
#include <vector>

#include "loom/import/cxx/lifetime/destruction.h"

namespace loom::cxx_import {

// Nontrivial result-object and reference lifetime facts from the owning source
// walk. Construction syntax and its implicit conversion can describe the same
// object; equal payloads from separate constructions never establish identity.
// Source nodes identify construction sites, not dynamic evaluation instances.
class TemporaryOrigins {
 public:
  explicit TemporaryOrigins(cxx::TranslationUnit& unit) : unit_(unit) {}

  // Visits one expression after its evaluated children. Forwarding consumes
  // their retained facts rather than walking the source expression again.
  void retain(cxx::ExpressionAST* expression);
  // Extends the materializations selected by one reference initializer. This
  // consumes the retained lifetime graph, including both conditional arms.
  void extend(cxx::ExpressionAST* initializer);

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
    // Whether a lexical reference scope owns this construction's cleanup.
    bool extended = false;
  };
  struct Selection {
    // First conditional alternative's one-based lifetime node.
    size_t first;
    // Second conditional alternative's one-based lifetime node.
    size_t second;
  };
  using Lifetime = std::variant<Construction, Selection>;
  struct Facts {
    // Complete owning result, absent on a glvalue or scalar expression.
    cxx::ExpressionAST* result = nullptr;
    // One complete materialized object, absent for a selected glvalue.
    cxx::ExpressionAST* materialization = nullptr;
    // Shared lifetime-extension path; forwarding introduces no additional node.
    size_t lifetime = 0;
  };

  Facts facts(cxx::ExpressionAST* expression) const;
  Facts classify(cxx::ExpressionAST* expression);
  void extend(size_t lifetime);

  // Verified source type/category facts; source storage outlives this analysis.
  cxx::TranslationUnit& unit_;
  // Sparse facts; ordinary scalar expressions allocate no entries.
  std::unordered_map<cxx::ExpressionAST*, Facts> expressions_;
  // One-based nodes for constructions and joins; zero denotes no extension.
  std::vector<Lifetime> lifetimes_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_LIFETIME_ORIGINS_H_
