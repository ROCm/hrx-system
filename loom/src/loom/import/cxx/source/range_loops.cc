// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/range_loops.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

namespace loom::cxx_import {
namespace {

cxx::DeclarationStatementAST* declaration(cxx::Arena* arena,
                                          cxx::DeclarationAST* source) {
  auto* statement = cxx::DeclarationStatementAST::create(arena);
  statement->declaration = source;
  return statement;
}

cxx::InitDeclaratorAST* initialized(cxx::Arena* arena,
                                    cxx::InitDeclaratorAST* source,
                                    cxx::ExpressionAST* initializer) {
  return cxx::InitDeclaratorAST::create(arena, source->declarator,
                                        source->requiresClause, initializer,
                                        source->symbol);
}

cxx::SimpleDeclarationAST* local(cxx::Arena* arena,
                                 cxx::VariableSymbol* variable,
                                 cxx::SourceLocation location) {
  auto* declarator = cxx::InitDeclaratorAST::create(arena);
  declarator->symbol = variable;
  declarator->initializer = variable->initializer();
  auto* simple = cxx::SimpleDeclarationAST::create(arena);
  simple->initDeclaratorList = cxx::make_list_node(arena, declarator);
  simple->semicolonLoc = location;
  return simple;
}

cxx::DeclarationAST* element(cxx::Arena* arena,
                             cxx::ForRangeStatementAST* range) {
  if (auto* source =
          cxx::ast_cast<cxx::SimpleDeclarationAST>(range->rangeDeclaration)) {
    auto* result = cxx::SimpleDeclarationAST::create(arena);
    *result = *source;
    result->initDeclaratorList = cxx::make_list_node(
        arena,
        initialized(arena, source->initDeclaratorList->value, range->element));
    return result;
  }
  auto* source = static_cast<cxx::StructuredBindingDeclarationAST*>(
      range->rangeDeclaration);
  auto* result = cxx::StructuredBindingDeclarationAST::create(arena);
  *result = *source;
  result->hiddenVariable =
      initialized(arena, source->hiddenVariable, range->element);
  return result;
}

}  // namespace

cxx::CompoundStatementAST* RangeLoops::expand(
    cxx::ForRangeStatementAST* source) {
  auto* arena = unit_.arena();
  auto* scope = cxx::CompoundStatementAST::create(arena);
  scope->lbraceLoc = source->forLoc;
  scope->rbraceLoc = source->statement->lastSourceLocation().previous();
  scope->symbol = source->symbol;
  auto** tail = &scope->statementList;
  auto append = [&](cxx::StatementAST* statement) {
    *tail = cxx::make_list_node(arena, statement);
    tail = &(*tail)->next;
  };
  if (source->initializer) {
    append(source->initializer);
  }
  auto* range = local(arena, source->rangeVariable, source->colonLoc);
  initializers_.insert(range->initDeclaratorList->value);
  append(declaration(arena, range));
  append(declaration(arena,
                     local(arena, source->beginVariable, source->colonLoc)));
  append(
      declaration(arena, local(arena, source->endVariable, source->colonLoc)));

  auto* body = cxx::CompoundStatementAST::create(arena);
  body->lbraceLoc = source->rangeDeclaration->firstSourceLocation();
  body->rbraceLoc = scope->rbraceLoc;
  body->statementList = cxx::make_list_node<cxx::StatementAST>(
      arena, declaration(arena, element(arena, source)));
  body->statementList->next = cxx::make_list_node(arena, source->statement);

  auto* loop = cxx::ForStatementAST::create(arena);
  loop->attributeList = source->attributeList;
  loop->forLoc = source->forLoc;
  loop->lparenLoc = source->lparenLoc;
  loop->condition = source->condition;
  loop->semicolonLoc = source->colonLoc;
  loop->expression = source->increment;
  loop->rparenLoc = source->rparenLoc;
  loop->statement = body;
  append(loop);
  scopes_.emplace(source, scope);
  return scope;
}

cxx::CompoundStatementAST* RangeLoops::scope(
    cxx::ForRangeStatementAST* source) const {
  return scopes_.at(source);
}

bool RangeLoops::initializes_range(cxx::InitDeclaratorAST* declaration) const {
  return initializers_.contains(declaration);
}

}  // namespace loom::cxx_import
