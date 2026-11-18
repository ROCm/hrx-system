// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_LOOPS_COUNTED_H_
#define LOOM_IMPORT_CXX_CONTROL_LOOPS_COUNTED_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstdint>
#include <optional>
#include <variant>

#include "loom/ir/scalar_type.h"

namespace loom::cxx_import {

class Types;

// A strict, increasing source interval whose last increment cannot overflow.
// Recognition uses the checked header alone. ControlFlow separately proves
// that the body preserves the induction and that the upper bound is invariant.
struct CountedLoop {
  struct Bound {
    // Referenced source identifier, or null for a native address literal.
    cxx::IdExpressionAST* identifier;
    // Complete expression to emit once, including its source conversions.
    cxx::ExpressionAST* expression;
  };

  static std::optional<CountedLoop> recognize(cxx::TranslationUnit& unit,
                                              Types& types,
                                              cxx::ForStatementAST* loop);

  cxx::IdExpressionAST* upper_identifier() const {
    auto* bound = std::get_if<Bound>(&upper);
    return bound ? bound->identifier : nullptr;
  }

  // Source variable replaced by the structured induction argument.
  cxx::Symbol* induction;
  // Folded C++ int/unsigned int, or the exact source expression to evaluate.
  std::variant<std::int64_t, Bound> upper;
  // Positive source increment; native index and offset require a unit step.
  unsigned step;
  // Logical signed index or physical unsigned offset used by scf.for.
  loom_scalar_type_t domain;
  // Native source values already inhabit the indicated High address domain.
  bool native;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_LOOPS_COUNTED_H_
