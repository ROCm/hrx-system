// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_PREDICATES_H_
#define LOOM_IMPORT_CXX_BINDING_PREDICATES_H_

#include <cxx/types_fwd.h>

#include <array>
#include <cstdint>
#include <vector>

#include "loom/import/cxx/source/source.h"
#include "loom/ir/attribute.h"

namespace loom::cxx_import {

// Semantic origin of one value operand retained by a projected predicate.
enum class PredicateValueOrigin {
  // A resolved source binding, optionally followed by record members.
  Binding,
  // A materialized scalar expression retained as its resulting SSA value.
  Expression,
  // The typed result placeholder of an enclosing callable contract.
  Result,
  // The implicit value defined by an enclosing config declaration.
  Subject,
};

// One automatic source binding read by a retained scalar expression. Both
// references borrow the owning Source and allow the body translator to verify
// that the binding still has an owned SSA identity at the assumption site.
struct PredicateDependency {
  // Resolved source binding read by the expression.
  cxx::Symbol* binding;
  // Direct source reference owning any binding-lifetime diagnostic.
  cxx::ExpressionAST* source;
};

// One source value referenced by a projected predicate. AST and symbol
// references borrow the owning Source. The converted expression preserves the
// implicit conversions selected by C++ and is absent only for Subject.
struct PredicateValue {
  // Semantic root kind selected before any consumer-specific substitution.
  PredicateValueOrigin origin;
  // Resolved root symbol for Binding, otherwise null.
  cxx::Symbol* binding;
  // Direct semantic field selections from the root to the scalar leaf.
  std::vector<cxx::FieldSymbol*> members;
  // Runtime bindings read by Expression values, in source traversal order.
  std::vector<PredicateDependency> dependencies;
  // Unconverted source identity owning diagnostics and its declared type.
  cxx::ExpressionAST* source;
  // Complete operand after frontend-selected implicit conversions.
  cxx::ExpressionAST* converted;
  // Unconverted root type before direct record-member selection.
  const cxx::Type* root_type;
};

// One exactly representable predicate and its semantic value operands. Before
// native IR construction, each VALUE payload stores an ordinal into values;
// the owning resolver substitutes the appropriate SSA identity.
struct ProjectedPredicate {
  // Semantic values referenced by VALUE arguments, in ordinal order.
  std::array<PredicateValue, 3> values;
  // Number of active entries in values.
  uint8_t value_count;
  // Frontend-selected source type for each explicit predicate argument. An
  // implicit config subject has no source argument and leaves its entry null.
  std::array<const cxx::Type*, 3> argument_types;
  // Predicate template whose VALUE payloads index values.
  loom_predicate_t predicate;
};

// Selects whether predicate helpers spell every argument or omit the first
// value supplied by an enclosing config declaration.
enum class PredicateSubject {
  Explicit,
  Implicit,
};

// Selects whether predicate operands must name retained values directly or may
// also form side-effect-free expressions whose current values can be
// materialized. Function and config contracts use direct identities; body-local
// assumptions can retain derived SSA identities within their dominance scope.
enum class PredicateExpressionPolicy {
  RetainedValues,
  MaterializableValues,
};

// Returns whether |kind| accepts a scalar value with the resolved source type.
// Consumers with implicit subjects validate that subject through this query.
bool predicate_accepts_source_type(cxx::TranslationUnit& unit,
                                   loom_predicate_kind_t kind,
                                   const cxx::Type* type);

// Projects a pure C++ conjunction into Loom predicate templates. Resolution
// follows semantic symbols and frontend-selected implicit conversions. Direct
// constant-evaluated calls are accepted; runtime calls, mutation, volatile
// reads, overloaded operators, and unrepresentable constants diagnose before
// any result is returned.
std::vector<ProjectedPredicate> project_predicates(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::ExpressionAST* expression,
    PredicateSubject subject = PredicateSubject::Explicit,
    PredicateExpressionPolicy expression_policy =
        PredicateExpressionPolicy::RetainedValues);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_PREDICATES_H_
