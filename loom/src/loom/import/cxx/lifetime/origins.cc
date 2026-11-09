// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/lifetime/origins.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>

namespace loom::cxx_import {

TemporaryOrigins::Facts TemporaryOrigins::facts(
    cxx::ExpressionAST* expression) const {
  auto found = expressions_.find(expression);
  return found == expressions_.end() ? Facts{} : found->second;
}

cxx::ExpressionAST* TemporaryOrigins::result(
    cxx::ExpressionAST* expression) const {
  return facts(expression).result;
}

cxx::ExpressionAST* TemporaryOrigins::materialization(
    cxx::ExpressionAST* expression) const {
  return facts(expression).materialization;
}

bool TemporaryOrigins::extended(cxx::ExpressionAST* origin) const {
  return std::get<Construction>(lifetimes_[facts(origin).lifetime - 1])
      .extended;
}

const Destruction& TemporaryOrigins::destruction(
    cxx::ExpressionAST* origin) const {
  return std::get<Construction>(lifetimes_[facts(origin).lifetime - 1])
      .destruction;
}

void TemporaryOrigins::retain(cxx::ExpressionAST* expression) {
  auto retained = classify(expression);
  if (retained.result || retained.materialization || retained.lifetime) {
    expressions_.insert_or_assign(expression, retained);
  }
}

TemporaryOrigins::Facts TemporaryOrigins::classify(
    cxx::ExpressionAST* expression) {
  auto traits = unit_.typeTraits();
  switch (expression->kind()) {
    case cxx::ASTKind::NestedExpression:
      return facts(
          static_cast<cxx::NestedExpressionAST*>(expression)->expression);
    case cxx::ASTKind::EqualInitializer:
      return facts(
          static_cast<cxx::EqualInitializerAST*>(expression)->expression);
    case cxx::ASTKind::DefaultInitializerExpression:
      return facts(
          static_cast<cxx::DefaultInitializerExpressionAST*>(expression)
              ->expression);
    case cxx::ASTKind::ImplicitCastExpression: {
      auto* cast = static_cast<cxx::ImplicitCastExpressionAST*>(expression);
      auto child = facts(cast->expression);
      if (cast->conversionFunction) {
        // The frontend's constructor conversion wraps its completed argument
        // initializer. Both nodes denote the one constructed result object.
        if (cast->conversionFunction->isConstructor() && child.result &&
            traits.remove_cv(cast->expression->type) ==
                traits.remove_cv(cast->type)) {
          return child;
        }
        break;
      }
      if (cast->valueCategory != cxx::ValueCategory::kPrValue) {
        child.result = nullptr;
        return child;
      }
      if (traits.is_class(cast->type) && child.result) {
        return child;
      }
      return {};
    }
    case cxx::ASTKind::BinaryExpression: {
      auto* binary = static_cast<cxx::BinaryExpressionAST*>(expression);
      if (!binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
        return facts(binary->rightExpression);
      }
      if (!binary->symbol && binary->op == cxx::TokenKind::T_DOT_STAR &&
          !traits.is_reference(binary->type)) {
        return {.lifetime = facts(binary->leftExpression).lifetime};
      }
      break;
    }
    case cxx::ASTKind::MemberExpression: {
      auto* member = static_cast<cxx::MemberExpressionAST*>(expression);
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      return member->accessOp == cxx::TokenKind::T_DOT && field &&
                     !traits.is_reference(field->type())
                 ? Facts{.lifetime = facts(member->baseExpression).lifetime}
                 : Facts{};
    }
    case cxx::ASTKind::SubscriptExpression: {
      auto* subscript = static_cast<cxx::SubscriptExpressionAST*>(expression);
      if (subscript->symbol) {
        break;
      }
      auto* base = traits.is_array(subscript->baseExpression->type)
                       ? subscript->baseExpression
                       : subscript->indexExpression;
      return traits.is_array(base->type)
                 ? Facts{.lifetime = facts(base).lifetime}
                 : Facts{};
    }
    case cxx::ASTKind::CppCastExpression: {
      auto* cast = static_cast<cxx::CppCastExpressionAST*>(expression);
      if (cast->valueCategory != cxx::ValueCategory::kPrValue) {
        auto child = facts(cast->expression);
        child.result = nullptr;
        return child;
      }
      break;
    }
    case cxx::ASTKind::ConditionalExpression: {
      if (expression->valueCategory == cxx::ValueCategory::kPrValue) {
        break;
      }
      auto* conditional =
          static_cast<cxx::ConditionalExpressionAST*>(expression);
      auto first = facts(conditional->iftrueExpression).lifetime;
      auto second = facts(conditional->iffalseExpression).lifetime;
      if (!first) {
        return {.lifetime = second};
      }
      if (!second || first == second) {
        return {.lifetime = first};
      }
      lifetimes_.push_back(Selection{first, second});
      return {.lifetime = lifetimes_.size()};
    }
    default:
      break;
  }
  if (!expression->type ||
      expression->valueCategory != cxx::ValueCategory::kPrValue ||
      (!traits.is_class(expression->type) &&
       !traits.is_array(expression->type))) {
    return {};
  }
  auto destruction = destruction_for(unit_, expression->type);
  if (!destruction) {
    return {};
  }
  lifetimes_.push_back(Construction{*destruction});
  return {expression, expression, lifetimes_.size()};
}

void TemporaryOrigins::extend(cxx::ExpressionAST* initializer) {
  extend(facts(initializer).lifetime);
}

void TemporaryOrigins::extend(size_t index) {
  if (!index) {
    return;
  }
  auto& lifetime = lifetimes_[index - 1];
  if (auto* construction = std::get_if<Construction>(&lifetime)) {
    construction->extended = true;
    return;
  }
  const auto& selection = std::get<Selection>(lifetime);
  extend(selection.first);
  extend(selection.second);
}

}  // namespace loom::cxx_import
