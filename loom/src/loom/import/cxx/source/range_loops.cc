// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/range_loops.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>

namespace loom::cxx_import {
namespace {

cxx::InitDeclaratorAST* first_declarator(cxx::StatementAST* statement) {
  auto* declaration = cxx::ast_cast<cxx::DeclarationStatementAST>(statement);
  auto* simple =
      declaration
          ? cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration)
          : nullptr;
  return simple ? simple->initDeclaratorList->value : nullptr;
}

}  // namespace

cxx::CompoundStatementAST* RangeLoops::expand(
    cxx::ForRangeStatementAST* source) {
  auto* scope = source->range.statement;
  auto* declarations = scope->statementList;
  if (source->initializer) {
    declarations = declarations->next;
  }
  initializers_.insert(first_declarator(declarations->value));
  return scope;
}

cxx::CompoundStatementAST* RangeLoops::scope(
    cxx::ForRangeStatementAST* source) const {
  return source->range.statement;
}

bool RangeLoops::initializes_range(cxx::InitDeclaratorAST* declaration) const {
  return initializers_.contains(declaration);
}

}  // namespace loom::cxx_import
