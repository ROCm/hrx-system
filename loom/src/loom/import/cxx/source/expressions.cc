// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/expressions.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace loom::cxx_import {

cxx::ExpressionAST* strip_implicit_casts(cxx::ExpressionAST* expression) {
  while (auto* cast =
             cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
    expression = cast->expression;
  }
  return expression;
}

cxx::ExpressionAST* unwrap_expression(cxx::ExpressionAST* expression) {
  expression = strip_implicit_casts(expression);
  while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
    expression = strip_implicit_casts(nested->expression);
  }
  return expression;
}

bool is_current_object(cxx::ExpressionAST* expression) {
  expression = unwrap_expression(expression);
  if (cxx::ast_cast<cxx::ThisExpressionAST>(expression)) {
    return true;
  }
  auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(expression);
  return unary && !unary->symbol && unary->op == cxx::TokenKind::T_STAR &&
         is_current_object(unary->expression);
}

cxx::FunctionSymbol* direct_callee(cxx::CallExpressionAST* call) {
  auto* base = unwrap_expression(call ? call->baseExpression : nullptr);
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(base)) {
    return cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol);
  }
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base)) {
    return cxx::symbol_cast<cxx::FunctionSymbol>(member->symbol);
  }
  return nullptr;
}

const cxx::Type* implicit_object_type(cxx::TranslationUnit& unit,
                                      cxx::FunctionSymbol* function) {
  auto* record = function
                     ? cxx::symbol_cast<cxx::ClassSymbol>(function->parent())
                     : nullptr;
  auto* signature =
      function ? cxx::type_cast<cxx::FunctionType>(function->type()) : nullptr;
  if (!function || !function->isImplicitObjectMemberFunction() || !record ||
      !signature) {
    return nullptr;
  }
  auto qualifiers = signature->cvQualifiers();
  return qualifiers == cxx::CvQualifiers::kNone
             ? record->type()
             : unit.control()->getQualType(record->type(), qualifiers);
}

bool implicit_object_may_mutate(cxx::FunctionSymbol* function) {
  auto* record = function
                     ? cxx::symbol_cast<cxx::ClassSymbol>(function->parent())
                     : nullptr;
  auto* signature =
      function ? cxx::type_cast<cxx::FunctionType>(function->type()) : nullptr;
  if (!function || !function->isImplicitObjectMemberFunction() || !record ||
      !signature) {
    return false;
  }
  if (!cxx::has_const(signature->cvQualifiers())) {
    return true;
  }
  for (auto* member : record->members()) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
    if (field && !field->isStatic() && field->isMutable()) {
      return true;
    }
  }
  return false;
}

cxx::ConditionExpressionAST* condition_declaration(
    cxx::ExpressionAST* expression) {
  return cxx::ast_cast<cxx::ConditionExpressionAST>(
      strip_implicit_casts(expression));
}

}  // namespace loom::cxx_import
