// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/object_identity.h"

#include <cxx/ast.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <utility>

#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

size_t ObjectIdentity::add(const void* key) {
  auto [found, inserted] = identities_.try_emplace(key, groups_.size());
  if (inserted) {
    groups_.push_back({found->second});
  }
  return found->second;
}

size_t ObjectIdentity::root(size_t index) const {
  while (groups_[index].parent != index) {
    index = groups_[index].parent;
  }
  return index;
}

void ObjectIdentity::join(size_t first, size_t second) {
  first = root(first);
  second = root(second);
  if (first == second) {
    return;
  }
  if (groups_[first].size < groups_[second].size) {
    std::swap(first, second);
  }
  groups_[second].parent = first;
  groups_[first].size += groups_[second].size;
  groups_[first].storage |= groups_[second].storage;
}

void ObjectIdentity::join(const void* destination, cxx::ExpressionAST* source) {
  auto found = identities_.find(source);
  if (found == identities_.end()) {
    return;
  }
  const size_t source_group = found->second;
  join(add(destination), source_group);
}

void ObjectIdentity::alias(cxx::ExpressionAST* expression, const void* source) {
  auto found = identities_.find(source);
  if (found != identities_.end()) {
    identities_.insert_or_assign(expression, found->second);
  }
}

void ObjectIdentity::elements(cxx::ExpressionAST* object,
                              cxx::List<cxx::ExpressionAST*>* initializers) {
  size_t ordinal = 0;
  auto add_element = [&](const cxx::Type* type, cxx::Symbol* member) {
    if (!initializers) {
      return;
    }
    auto* element = initializers->value;
    if (!unit_.typeTraits().is_reference(type) &&
        element->valueCategory == cxx::ValueCategory::kPrValue) {
      auto child = identities_.find(element);
      if (child != identities_.end()) {
        const size_t child_index = child->second;
        if (types_.requires_value_transport(object->type) &&
            (!member || types_.record(object->type, object)) &&
            types_.unqualified_value(object->type) !=
                types_.unqualified_value(type)) {
          if (!members_) {
            members_ = std::make_unique<AggregateMembers>();
          }
          const size_t offset =
              member ? types_.member(member, object).component_offset
                     : ordinal * types_.partition(type, object).component_count;
          members_->elements.push_back(
              {add(object), child_index, offset, element});
        } else {
          join(add(object), child_index);
        }
      }
    }
    ++ordinal;
    initializers = initializers->next;
  };
  auto* type = types_.unqualified(object->type);
  if (auto* record = cxx::type_cast<cxx::ClassType>(type)) {
    for (auto* base : record->definition()->baseClasses()) {
      add_element(source_object_type(base), base);
    }
    for (auto* member : record->definition()->members()) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
      if (field && !field->isStatic()) {
        add_element(field->type(), field);
      }
    }
  } else {
    while (initializers) {
      add_element(unit_.typeTraits().get_element_type(type), nullptr);
    }
  }
}

void ObjectIdentity::retain(cxx::ExpressionAST* expression) {
  auto traits = unit_.typeTraits();
  switch (expression->kind()) {
    case cxx::ASTKind::IdExpression:
      alias(expression, static_cast<cxx::IdExpressionAST*>(expression)->symbol);
      return;
    case cxx::ASTKind::NestedExpression:
      alias(expression,
            static_cast<cxx::NestedExpressionAST*>(expression)->expression);
      return;
    case cxx::ASTKind::EqualInitializer:
      alias(expression,
            static_cast<cxx::EqualInitializerAST*>(expression)->expression);
      return;
    case cxx::ASTKind::DefaultInitializerExpression:
      alias(expression,
            static_cast<cxx::DefaultInitializerExpressionAST*>(expression)
                ->expression);
      return;
    case cxx::ASTKind::MemberExpression: {
      auto* member = static_cast<cxx::MemberExpressionAST*>(expression);
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (member->accessOp == cxx::TokenKind::T_DOT && field &&
          !traits.is_reference(field->type())) {
        alias(expression, member->baseExpression);
      }
      return;
    }
    case cxx::ASTKind::ImplicitCastExpression: {
      auto* cast = static_cast<cxx::ImplicitCastExpressionAST*>(expression);
      if ((!cast->conversionFunction &&
           (cast->valueCategory != cxx::ValueCategory::kPrValue ||
            cast->expression->valueCategory == cxx::ValueCategory::kPrValue)) ||
          (cast->conversionFunction &&
           cast->conversionFunction->isConstructor() &&
           cast->expression->valueCategory == cxx::ValueCategory::kPrValue &&
           traits.remove_cv(cast->expression->type) ==
               traits.remove_cv(cast->type))) {
        alias(expression, cast->expression);
        return;
      }
      break;
    }
    case cxx::ASTKind::BinaryExpression: {
      auto* binary = static_cast<cxx::BinaryExpressionAST*>(expression);
      if (!binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
        alias(expression, binary->rightExpression);
        return;
      }
      break;
    }
    default:
      break;
  }
  if (!expression->type ||
      expression->valueCategory != cxx::ValueCategory::kPrValue) {
    return;
  }
  const bool array = traits.is_array(expression->type);
  if (!array && !traits.is_class(expression->type)) {
    return;
  }
  add(expression);
  if (types_.has_volatile_subobject(expression->type)) {
    require_storage(expression);
  }
  if (auto* conditional =
          cxx::ast_cast<cxx::ConditionalExpressionAST>(expression)) {
    join(expression, conditional->iftrueExpression);
    join(expression, conditional->iffalseExpression);
  } else if (auto* construction =
                 cxx::ast_cast<cxx::BracedTypeConstructionAST>(expression);
             construction && !construction->constructorSymbol) {
    join(expression, construction->bracedInitList);
  } else if (auto* construction =
                 cxx::ast_cast<cxx::TypeConstructionAST>(expression);
             construction && !construction->constructorSymbol) {
    elements(expression, construction->expressionList);
  } else if (auto* list = cxx::ast_cast<cxx::BracedInitListAST>(expression);
             list && (array || traits.is_aggregate(expression->type))) {
    elements(expression, cxx::Initializer(list).expressionList());
  } else if (auto* list = cxx::ast_cast<cxx::ParenInitializerAST>(expression);
             list && (array || traits.is_aggregate(expression->type))) {
    elements(expression, cxx::Initializer(list).expressionList());
  }
}

void ObjectIdentity::bind(cxx::Symbol* binding, cxx::ExpressionAST* initializer,
                          bool storage) {
  add(binding);
  auto* source = initializer ? cxx::Initializer(initializer).clause() : nullptr;
  if (source && source->valueCategory == cxx::ValueCategory::kPrValue &&
      types_.unqualified(source->type) ==
          types_.unqualified(source_object_type(binding))) {
    join(binding, source);
    if (members_ && types_.requires_value_transport(source->type)) {
      members_->bindings.emplace_back(add(binding), binding);
    }
  }
  if (storage || types_.has_volatile_subobject(source_object_type(binding))) {
    require_storage(binding);
  }
}

void ObjectIdentity::returned(cxx::FunctionSymbol* function,
                              cxx::ExpressionAST* expression) {
  add(function);
  const auto* signature =
      static_cast<const cxx::FunctionType*>(function->type());
  if (types_.has_volatile_subobject(signature->returnType())) {
    require_storage(function);
  }
  if (expression && expression->valueCategory == cxx::ValueCategory::kPrValue) {
    join(function, expression);
  }
}

std::span<const ObjectIdentity::MemberConstruction>
ObjectIdentity::member_constructions() const {
  return members_ ? std::span<const MemberConstruction>(members_->constructions)
                  : std::span<const MemberConstruction>{};
}

const Destination* ObjectIdentity::aggregate_destination(
    cxx::ExpressionAST* initializer) const {
  if (!members_) {
    return nullptr;
  }
  auto identity = identities_.find(initializer);
  if (identity == identities_.end()) {
    return nullptr;
  }
  auto owner = members_->destinations.find(root(identity->second));
  return owner == members_->destinations.end() ? nullptr : &owner->second;
}

const ObjectIdentity::MemberConstruction* ObjectIdentity::member_construction(
    cxx::ExpressionAST* initializer) const {
  if (!members_) {
    return nullptr;
  }
  auto found = members_->by_initializer.find(initializer);
  return found == members_->by_initializer.end()
             ? nullptr
             : &members_->constructions[found->second];
}

void ObjectIdentity::finish_members() {
  struct Owner {
    // Construction group rooted at this subobject.
    size_t group;
    // Named source object that supplies final storage.
    cxx::Symbol* binding;
    // First component of the subobject within that binding.
    size_t offset;
  };
  std::unordered_map<size_t, std::vector<const Element*>> children;
  for (const auto& child : members_->elements) {
    children[root(child.parent)].push_back(&child);
  }
  std::vector<Owner> owners;
  for (auto [group, binding] : members_->bindings) {
    owners.push_back({root(group), binding, 0});
  }
  while (!owners.empty()) {
    const auto owner = owners.back();
    owners.pop_back();
    auto found = children.find(owner.group);
    members_->destinations.emplace(
        owner.group, Destination{owner.binding, owner.offset, nullptr});
    if (found == children.end()) {
      continue;
    }
    for (auto* edge : found->second) {
      const size_t offset = owner.offset + edge->offset;
      if (types_.requires_value_transport(edge->initializer->type)) {
        owners.push_back({root(edge->child), owner.binding, offset});
      } else {
        members_->by_initializer.emplace(edge->initializer,
                                         members_->constructions.size());
        members_->constructions.push_back(
            {owner.binding, offset, edge->initializer});
      }
    }
    children.erase(found);
  }
  // Unnamed result objects still use the owning callable's complete-object
  // identity. Only a resolved local can supply independent member lifetimes.
  for (const auto& [parent, edges] : children) {
    for (auto* edge : edges) {
      join(parent, edge->child);
    }
  }
  members_->elements.clear();
  members_->bindings.clear();
}

void ObjectIdentity::finish() {
  if (members_) {
    finish_members();
  }
  for (auto& [key, index] : identities_) {
    index = root(index);
  }
  for (size_t index = 0; index < groups_.size(); ++index) {
    groups_[index].parent = root(index);
  }
}

bool ObjectIdentity::requires_storage(const void* key) const {
  auto found = identities_.find(key);
  return found != identities_.end() && groups_[root(found->second)].storage;
}

bool ObjectIdentity::require_storage(const void* key) {
  auto found = identities_.find(key);
  if (found == identities_.end()) {
    return false;
  }
  auto& group = groups_[root(found->second)];
  return !std::exchange(group.storage, true);
}

}  // namespace loom::cxx_import
