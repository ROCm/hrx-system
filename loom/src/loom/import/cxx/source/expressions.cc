// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/expressions.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/initialization.h>
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
  for (;;) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      expression = strip_implicit_casts(nested->expression);
      continue;
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(expression)) {
      expression = strip_implicit_casts(initializer->expression);
      continue;
    }
    return expression;
  }
}

cxx::ImplicitCastExpressionAST* constructor_conversion(
    cxx::ExpressionAST* initializer) {
  for (;;) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(initializer)) {
      initializer = nested->expression;
      continue;
    }
    if (auto* equal = cxx::ast_cast<cxx::EqualInitializerAST>(initializer)) {
      initializer = equal->expression;
      continue;
    }
    if (auto* source =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(initializer)) {
      initializer = source->expression;
      continue;
    }
    auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(initializer);
    return cast && cast->conversionFunction &&
                   cast->conversionFunction->isConstructor()
               ? cast
               : nullptr;
  }
}

bool same_function(cxx::FunctionSymbol* left, cxx::FunctionSymbol* right) {
  return left && right && left->canonical() == right->canonical();
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

const cxx::StringLiteral* string_initializer(cxx::ExpressionAST* initializer) {
  auto arguments = cxx::Initializer(initializer).arguments();
  auto* literal = arguments.size() == 1
                      ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(
                            unwrap_expression(arguments.front()))
                      : nullptr;
  return literal ? literal->literal : nullptr;
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

std::optional<DirectOperatorCall> direct_operator_call(
    cxx::ExpressionAST* expression) {
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(expression);
      binary && binary->symbol) {
    const bool member = binary->symbol->isImplicitObjectMemberFunction();
    return DirectOperatorCall{
        .function = binary->symbol,
        .receiver = member ? binary->leftExpression : nullptr,
        .arguments =
            member
                ? std::array<cxx::ExpressionAST*, 2>{binary->rightExpression,
                                                     nullptr}
                : std::array<cxx::ExpressionAST*, 2>{binary->leftExpression,
                                                     binary->rightExpression},
        .argument_count = member ? 1u : 2u,
        .evaluation_order = OperatorEvaluationOrder::Forward,
        .has_postfix_dummy = false,
        .is_virtual_dispatch = binary->isVirtualDispatch,
    };
  }
  if (auto* assignment =
          cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(expression);
      assignment && assignment->symbol) {
    const bool member = assignment->symbol->isImplicitObjectMemberFunction();
    return DirectOperatorCall{
        .function = assignment->symbol,
        .receiver = member ? assignment->targetExpression : nullptr,
        .arguments = member
                         ? std::array<cxx::ExpressionAST*,
                                      2>{assignment->rightExpression, nullptr}
                         : std::array<cxx::ExpressionAST*,
                                      2>{assignment->targetExpression,
                                         assignment->rightExpression},
        .argument_count = member ? 1u : 2u,
        .evaluation_order = member
                                ? OperatorEvaluationOrder::ArgumentsThenReceiver
                                : OperatorEvaluationOrder::ReverseArguments,
        .has_postfix_dummy = false,
        .is_virtual_dispatch = assignment->isVirtualDispatch,
    };
  }
  if (auto* postfix = cxx::ast_cast<cxx::PostIncrExpressionAST>(expression);
      postfix && postfix->symbol) {
    const bool member = postfix->symbol->isImplicitObjectMemberFunction();
    return DirectOperatorCall{
        .function = postfix->symbol,
        .receiver = member ? postfix->baseExpression : nullptr,
        .arguments =
            member ? std::array<cxx::ExpressionAST*, 2>{nullptr, nullptr}
                   : std::array<cxx::ExpressionAST*, 2>{postfix->baseExpression,
                                                        nullptr},
        .argument_count = member ? 0u : 1u,
        .evaluation_order = OperatorEvaluationOrder::Forward,
        .has_postfix_dummy = true,
        .is_virtual_dispatch = postfix->isVirtualDispatch,
    };
  }
  if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(expression);
      unary && unary->symbol) {
    const bool member = unary->symbol->isImplicitObjectMemberFunction();
    return DirectOperatorCall{
        .function = unary->symbol,
        .receiver = member ? unary->expression : nullptr,
        .arguments = member
                         ? std::array<cxx::ExpressionAST*, 2>{nullptr, nullptr}
                         : std::array<cxx::ExpressionAST*, 2>{unary->expression,
                                                              nullptr},
        .argument_count = member ? 0u : 1u,
        .evaluation_order = OperatorEvaluationOrder::Forward,
        .has_postfix_dummy = false,
        .is_virtual_dispatch = unary->isVirtualDispatch,
    };
  }
  return std::nullopt;
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
