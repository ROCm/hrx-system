// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/lifetime/origins.h"

#include <cxx/ast.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>

#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

TemporaryOrigins::Facts TemporaryOrigins::facts(
    cxx::ExpressionAST* expression, bool retain_materialization) const {
  auto found = expressions_.find(expression);
  Facts result;
  if (found != expressions_.end()) {
    result.result = found->second.result;
    result.lifetime = found->second.lifetime;
    result.members = found->second.members;
  }
  if (retain_materialization) {
    result.materialization = materialization(expression);
  }
  return result;
}

cxx::ExpressionAST* TemporaryOrigins::result(
    cxx::ExpressionAST* expression) const {
  return facts(expression).result;
}

cxx::ExpressionAST* TemporaryOrigins::materialization(
    cxx::ExpressionAST* expression) const {
  auto found = materializations_.find(expression);
  if (found != materializations_.end()) {
    return found->second;
  }
  auto retained = expressions_.find(expression);
  if (retained == expressions_.end() || !retained->second.lifetime) {
    return nullptr;
  }
  const auto& lifetime = lifetimes_[retained->second.lifetime - 1];
  const auto* construction = std::get_if<Construction>(&lifetime.value);
  return construction ? construction->origin : nullptr;
}

bool TemporaryOrigins::extended(cxx::ExpressionAST* origin) const {
  return lifetimes_[facts(origin).lifetime - 1].extended;
}

const Destruction& TemporaryOrigins::destruction(
    cxx::ExpressionAST* origin) const {
  return std::get<Construction>(lifetimes_[facts(origin).lifetime - 1].value)
      .destruction;
}

void TemporaryOrigins::retain(cxx::ExpressionAST* expression,
                              bool retain_materialization) {
  auto retained = classify(expression, retain_materialization);
  if (retained.result || retained.lifetime || retained.members) {
    expressions_.insert_or_assign(
        expression,
        StoredFacts{retained.result, retained.lifetime, retained.members});
  }
  if (retained.materialization && !result(retained.materialization)) {
    materializations_.insert_or_assign(expression, retained.materialization);
  }
}

TemporaryOrigins::Facts TemporaryOrigins::classify(
    cxx::ExpressionAST* expression, bool retain_materialization) {
  auto traits = unit_.typeTraits();
  size_t members = 0;
  switch (expression->kind()) {
    case cxx::ASTKind::NestedExpression:
      return facts(
          static_cast<cxx::NestedExpressionAST*>(expression)->expression,
          retain_materialization);
    case cxx::ASTKind::EqualInitializer:
      return facts(
          static_cast<cxx::EqualInitializerAST*>(expression)->expression,
          retain_materialization);
    case cxx::ASTKind::DefaultInitializerExpression:
      return facts(
          static_cast<cxx::DefaultInitializerExpressionAST*>(expression)
              ->expression,
          retain_materialization);
    case cxx::ASTKind::ConstExpression: {
      auto* constant = static_cast<cxx::ConstExpressionAST*>(expression);
      if (constant->constValue &&
          constant_contains_address(*constant->constValue)) {
        return facts(constant->expression, retain_materialization);
      }
      break;
    }
    case cxx::ASTKind::ImplicitCastExpression: {
      auto* cast = static_cast<cxx::ImplicitCastExpressionAST*>(expression);
      auto child = facts(cast->expression, retain_materialization);
      if (cast->conversionFunction) {
        // The frontend's constructor conversion wraps its completed argument
        // initializer. Both nodes denote the one constructed result object.
        if (cast->conversionFunction->isConstructor() &&
            (child.result || child.materialization) &&
            traits.remove_cv(cast->expression->type) ==
                traits.remove_cv(cast->type)) {
          return child;
        }
        break;
      }
      if (cast->valueCategory != cxx::ValueCategory::kPrValue) {
        child.result = nullptr;
        child.members = 0;
        return child;
      }
      if (traits.is_class(cast->type) &&
          (child.result || child.materialization)) {
        return child;
      }
      if (retain_materialization && traits.is_pointer(cast->type)) {
        // A converted pointer is a new value object, independent of the
        // pointer or subobject from which the conversion reads its value.
        break;
      }
      return {};
    }
    case cxx::ASTKind::BinaryExpression: {
      auto* binary = static_cast<cxx::BinaryExpressionAST*>(expression);
      if (!binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
        return facts(binary->rightExpression, retain_materialization);
      }
      if (!binary->symbol && binary->op == cxx::TokenKind::T_DOT_STAR &&
          !traits.is_reference(binary->type)) {
        auto retained = facts(binary->leftExpression, retain_materialization);
        retained.result = nullptr;
        retained.members = 0;
        return retained;
      }
      break;
    }
    case cxx::ASTKind::MemberExpression: {
      auto* member = static_cast<cxx::MemberExpressionAST*>(expression);
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (member->accessOp != cxx::TokenKind::T_DOT || !field ||
          traits.is_reference(field->type())) {
        return {};
      }
      auto retained = facts(member->baseExpression, retain_materialization);
      retained.result = nullptr;
      retained.members = 0;
      return retained;
    }
    case cxx::ASTKind::SubscriptExpression: {
      auto* subscript = static_cast<cxx::SubscriptExpressionAST*>(expression);
      if (subscript->symbol) {
        break;
      }
      auto* index = subscript_index(subscript);
      auto* base = traits.is_array(subscript->baseExpression->type)
                       ? subscript->baseExpression
                       : index;
      if (!traits.is_array(base->type)) {
        return {};
      }
      auto retained = facts(base, retain_materialization);
      retained.result = nullptr;
      retained.members = 0;
      return retained;
    }
    case cxx::ASTKind::CppCastExpression: {
      auto* cast = static_cast<cxx::CppCastExpressionAST*>(expression);
      if (cast->valueCategory != cxx::ValueCategory::kPrValue) {
        auto child = facts(cast->expression, retain_materialization);
        child.result = nullptr;
        child.members = 0;
        return child;
      }
      break;
    }
    case cxx::ASTKind::ConditionalExpression: {
      auto* conditional =
          static_cast<cxx::ConditionalExpressionAST*>(expression);
      auto first = facts(conditional->iftrueExpression, retain_materialization);
      auto second =
          facts(conditional->iffalseExpression, retain_materialization);
      if (expression->valueCategory == cxx::ValueCategory::kPrValue) {
        members = combine(first.members, second.members);
        break;
      }
      return {.lifetime = combine(first.lifetime, second.lifetime)};
    }
    case cxx::ASTKind::BracedInitList: {
      auto* list = static_cast<cxx::BracedInitListAST*>(expression);
      members = aggregate_members(
          list, cxx::Initializer(list).expressionList(), list->implicitElement,
          !list->initialization.isParenthesizedAggregate);
      break;
    }
    case cxx::ASTKind::ParenInitializer:
      // Parenthesized initializer syntax does not itself construct a result,
      // and parenthesized aggregate reference members do not extend their
      // referents beyond the full expression.
      return {};
    case cxx::ASTKind::BracedTypeConstruction: {
      auto* construction =
          static_cast<cxx::BracedTypeConstructionAST*>(expression);
      if (!construction->constructorSymbol) {
        members = facts(construction->bracedInitList).members;
      }
      break;
    }
    case cxx::ASTKind::TypeConstruction: {
      auto* construction = static_cast<cxx::TypeConstructionAST*>(expression);
      if (!construction->constructorSymbol) {
        members = aggregate_members(construction, construction->expressionList,
                                    nullptr,
                                    /*extend_references=*/false);
      }
      break;
    }
    default:
      break;
  }
  if (!expression->type ||
      expression->valueCategory != cxx::ValueCategory::kPrValue ||
      (!traits.is_class(expression->type) &&
       !traits.is_array(expression->type) &&
       (!retain_materialization ||
        !types_.requires_value_transport(expression->type)))) {
    return {};
  }
  auto destruction = destruction_for(unit_, expression->type);
  if (!destruction) {
    return {
        .materialization =
            retain_materialization &&
                    types_.requires_value_transport(expression->type)
                ? expression
                : nullptr,
        .lifetime = members,
        .members = members,
    };
  }
  lifetimes_.push_back({Construction{*destruction, expression, members}});
  return {
      .result = expression,
      .lifetime = lifetimes_.size(),
      .members = members,
  };
}

void TemporaryOrigins::retain_initializer_list(
    cxx::BracedInitListAST* expression, cxx::BracedInitListAST* backing,
    bool retain_materialization) {
  auto child = facts(backing);
  if (child.lifetime) {
    // The list is not the backing array's result object. Its dependency
    // permits extension without letting copies adopt the array's destructor.
    lifetimes_.push_back({Dependencies{child.lifetime, 0}});
    expressions_.insert_or_assign(
        expression, StoredFacts{nullptr, lifetimes_.size(), lifetimes_.size()});
  }
  if (retain_materialization) {
    materializations_.insert_or_assign(expression, expression);
  }
}

size_t TemporaryOrigins::aggregate_members(
    cxx::ExpressionAST* initializer, cxx::List<cxx::ExpressionAST*>* elements,
    cxx::VariableSymbol* implicit_element, bool extend_references) {
  auto traits = unit_.typeTraits();
  size_t members = 0;
  if (traits.is_array(initializer->type)) {
    for (auto* element : cxx::ListView{elements}) {
      members = combine(members, facts(element).members);
    }
    if (implicit_element) {
      members =
          combine(members, facts(implicit_element->initializer()).members);
    }
    return members;
  }
  auto* record = types_.record(initializer->type, initializer);
  if (!record || !record->is_aggregate) {
    return 0;
  }
  // Copy-list initialization forwards an existing object; copying a reference
  // member does not establish another temporary lifetime.
  if (elements && !elements->next &&
      types_.unqualified(elements->value->type) ==
          types_.unqualified(initializer->type)) {
    return facts(elements->value).members;
  }
  for (const auto& member : record->members) {
    if (!elements) {
      break;
    }
    auto child = facts(elements->value);
    members = combine(members, traits.is_reference(member.type())
                                   ? (extend_references ? child.lifetime : 0)
                                   : child.members);
    elements = elements->next;
  }
  return members;
}

size_t TemporaryOrigins::combine(size_t first, size_t second) {
  if (!first) {
    return second;
  }
  if (!second || first == second) {
    return first;
  }
  lifetimes_.push_back({Dependencies{first, second}});
  return lifetimes_.size();
}

void TemporaryOrigins::extend(cxx::ExpressionAST* initializer) {
  extend(facts(initializer).lifetime);
}

void TemporaryOrigins::extend_members(cxx::ExpressionAST* initializer) {
  extend(facts(initializer).members);
}

void TemporaryOrigins::extend(size_t index) {
  if (index) {
    lifetimes_[index - 1].extended = true;
  }
}

void TemporaryOrigins::finish() {
  // Every edge points backward in source postorder. One reverse pass reaches
  // shared dependencies once, without recursive walks or per-root worklists.
  for (auto lifetime = lifetimes_.rbegin(); lifetime != lifetimes_.rend();
       ++lifetime) {
    if (!lifetime->extended) {
      continue;
    }
    if (const auto* construction =
            std::get_if<Construction>(&lifetime->value)) {
      extend(construction->members);
    } else {
      const auto& dependencies = std::get<Dependencies>(lifetime->value);
      extend(dependencies.first);
      extend(dependencies.second);
    }
  }
}

}  // namespace loom::cxx_import
