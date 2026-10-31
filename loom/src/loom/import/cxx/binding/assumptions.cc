// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/assumptions.h"

#include <cxx/ast.h>

namespace loom::cxx_import {

std::vector<ProjectedPredicate> assumption_predicates(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::CallExpressionAST* call) {
  if (!call->expressionList || call->expressionList->next) {
    diagnostics.reject(unit, call, "assume requires exactly one condition");
  }
  return project_predicates(unit, diagnostics, call->expressionList->value,
                            PredicateSubject::Explicit,
                            PredicateExpressionPolicy::MaterializableValues);
}

}  // namespace loom::cxx_import
