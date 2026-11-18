// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_EVALUATION_H_
#define LOOM_IMPORT_CXX_VALUE_EVALUATION_H_

#include <cxx/ast_fwd.h>

#include <cstddef>
#include <optional>

#include "loom/import/cxx/value/representation.h"

namespace loom::cxx_import {

// Saves the current cleanup scope while a nested C++ full expression runs.
struct FullExpression {
  // Temporary prefix retained by this full expression.
  size_t mark;
  // Enclosing full expression restored after this one completes.
  std::optional<size_t> parent_mark;
  // Whether the enclosing initializer transfers its surviving temporaries
  // to lexical cleanup instead of destroying them at expression completion.
  bool parent_extends_temporaries = false;
};

// Evaluates a source expression at the current insertion point, including its
// side effects. Operand consumers request each expression in C++ source order.
class ExpressionEvaluator {
 public:
  virtual Value expression(cxx::ExpressionAST* source) = 0;

 protected:
  ~ExpressionEvaluator() = default;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_EVALUATION_H_
