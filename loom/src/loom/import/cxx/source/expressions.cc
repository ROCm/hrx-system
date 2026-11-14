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
#include <cxx/type_checker.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/objects.h"

namespace loom::cxx_import {

std::vector<cxx::ExpressionAST*> constructor_arguments(
    cxx::ExpressionAST* initializer) {
  cxx::List<cxx::ExpressionAST*>* arguments = nullptr;
  if (auto* construction =
          cxx::ast_cast<cxx::TypeConstructionAST>(initializer)) {
    arguments = construction->expressionList;
  } else if (auto* construction =
                 cxx::ast_cast<cxx::BracedTypeConstructionAST>(initializer)) {
    auto* semantic = construction->initialization.expression
                         ? construction->initialization.expression
                         : construction->bracedInitList;
    return cxx::Initializer(semantic).arguments();
  } else {
    return cxx::Initializer(initializer).arguments();
  }
  std::vector<cxx::ExpressionAST*> result;
  for (auto* argument : cxx::ListView{arguments}) {
    result.push_back(argument);
  }
  return result;
}

std::vector<cxx::ExpressionAST*> complete_default_arguments(
    cxx::TranslationUnit& unit, cxx::ScopeSymbol* scope,
    cxx::FunctionSymbol* function, std::vector<cxx::ExpressionAST*> arguments,
    cxx::SourceLocation location) {
  if (arguments.size() >= function->parameters().size()) {
    return arguments;
  }
  cxx::List<cxx::ExpressionAST*>* list = nullptr;
  auto** tail = &list;
  for (auto* argument : arguments) {
    *tail = cxx::make_list_node(unit.arena(), argument);
    tail = &(*tail)->next;
  }
  cxx::TypeChecker checker(&unit);
  checker.setScope(scope);
  checker.append_default_arguments(function, &list, location);
  arguments.clear();
  for (auto* argument : cxx::ListView{list}) {
    arguments.push_back(argument);
  }
  return arguments;
}

std::optional<ArrayCopyInitialization> array_copy_initialization(
    cxx::TranslationUnit& unit, cxx::ScopeSymbol* scope, const cxx::Type* type,
    cxx::ExpressionAST* initializer, cxx::FunctionSymbol* constructor,
    cxx::AST* owner) {
  if (!unit.typeTraits().is_array(type)) {
    return std::nullopt;
  }
  cxx::Initializer source(initializer);
  auto* expression = source.singleExpression();
  if (!cxx::isWholeArrayCopy(unit.typeTraits(), expression, type)) {
    return std::nullopt;
  }
  ArrayCopyInitialization result{expression, constructor, {}};
  if (result.constructor->parameters().size() > 1) {
    // Selection belongs to array initialization in the frontend. Complete
    // only its omitted arguments, without repeating overload resolution.
    auto arguments =
        complete_default_arguments(unit, scope, result.constructor,
                                   {expression}, owner->firstSourceLocation());
    result.default_arguments.assign(arguments.begin() + 1, arguments.end());
  }
  return result;
}

cxx::ExpressionAST* strip_implicit_casts(cxx::ExpressionAST* expression) {
  while (auto* cast =
             cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
    if (is_class_adjustment(cast)) {
      break;
    }
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

cxx::ExpressionAST* reference_initializer(cxx::TranslationUnit& unit,
                                          cxx::ExpressionAST* initializer) {
  auto* clause = cxx::Initializer(initializer).clause();
  if (auto* list = cxx::ast_cast<cxx::BracedInitListAST>(clause)) {
    if (auto* element = cxx::listInitializedElement(unit.typeTraits(), list)) {
      return element;
    }
    return list;
  }
  if (!cxx::ast_cast<cxx::ParenInitializerAST>(clause)) {
    return clause;
  }
  auto arguments = cxx::Initializer(clause).arguments();
  return arguments.size() == 1 ? arguments.front() : nullptr;
}

cxx::ExpressionAST* subscript_index(cxx::SubscriptExpressionAST* expression) {
  return expression->expressionList->value;
}

std::optional<MemberObjectClasses> member_object_classes(
    cxx::TranslationUnit& unit, cxx::MemberExpressionAST* expression) {
  if (!expression->baseExpression || !expression->baseExpression->type ||
      !expression->symbol) {
    return std::nullopt;
  }
  auto* declaring =
      cxx::symbol_cast<cxx::ClassSymbol>(expression->symbol->parent());
  if (!declaring) {
    return std::nullopt;
  }
  auto traits = unit.typeTraits();
  auto* object_type = cxx::unaligned_type(traits.remove_cv(
      traits.remove_reference(expression->baseExpression->type)));
  if (expression->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
    object_type = traits.remove_pointer(object_type);
  }
  auto* object = cxx::type_cast<cxx::ClassType>(
      cxx::unaligned_type(traits.remove_cv(object_type)));
  if (!object) {
    return std::nullopt;
  }
  return MemberObjectClasses{
      .object = object->definition(),
      .declaring = declaring->resolvedDefinition(),
  };
}

cxx::ExpressionAST* explicit_reference_operand(cxx::TranslationUnit& unit,
                                               cxx::ExpressionAST* expression) {
  auto* cast = cxx::ast_cast<cxx::CppCastExpressionAST>(expression);
  return cast && cast->castOp == cxx::TokenKind::T_STATIC_CAST &&
                 cast->valueCategory != cxx::ValueCategory::kPrValue &&
                 cast->typeId &&
                 unit.typeTraits().is_reference(cast->typeId->type)
             ? cast->expression
             : nullptr;
}

cxx::ExpressionAST* unwrap_object_identity(cxx::TranslationUnit& unit,
                                           cxx::ExpressionAST* expression) {
  for (;;) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      expression = nested->expression;
      continue;
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(expression)) {
      expression = initializer->expression;
      continue;
    }
    if (auto* operand = explicit_reference_operand(unit, expression)) {
      expression = operand;
      continue;
    }
    auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression);
    if (!cast || cast->conversionFunction || is_class_adjustment(cast) ||
        cast->valueCategory == cxx::ValueCategory::kPrValue) {
      return expression;
    }
    expression = cast->expression;
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

bool is_current_object(cxx::TranslationUnit& unit,
                       cxx::ExpressionAST* expression) {
  expression = unwrap_object_identity(unit, expression);
  if (cxx::ast_cast<cxx::ThisExpressionAST>(expression)) {
    return true;
  }
  auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(expression);
  return unary && !unary->symbol && unary->op == cxx::TokenKind::T_STAR &&
         is_current_object(unit, unary->expression);
}

std::optional<CapturedObjectMember> captured_object_member(
    cxx::TranslationUnit& unit, cxx::ClassSymbol* closure,
    cxx::Symbol* member) {
  auto* pointer_field = closure && closure->isClosureType()
                            ? closure->capturedThisField()
                            : nullptr;
  auto* member_owner =
      is_direct_subobject(member)
          ? cxx::symbol_cast<cxx::ClassSymbol>(member->parent())
          : nullptr;
  if (!pointer_field || !member_owner) {
    return std::nullopt;
  }
  auto traits = unit.typeTraits();
  auto* object_type = cxx::type_cast<cxx::ClassType>(cxx::unaligned_type(
      traits.remove_cv(traits.get_element_type(pointer_field->type()))));
  auto* object_definition = object_type ? object_type->definition() : nullptr;
  if (!object_definition || object_definition->resolvedDefinition() !=
                                member_owner->resolvedDefinition()) {
    return std::nullopt;
  }
  return CapturedObjectMember{
      .pointer_field = pointer_field,
      .object_type = object_type,
  };
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
  auto binary =
      [](cxx::FunctionSymbol* function, cxx::ExpressionAST* left,
         cxx::ExpressionAST* right, bool right_first,
         bool is_virtual_dispatch) -> std::optional<DirectOperatorCall> {
    if (!function) {
      return std::nullopt;
    }
    const bool member = function->isImplicitObjectMemberFunction();
    return DirectOperatorCall{
        .function = function,
        .receiver = member ? left : nullptr,
        .arguments = member ? std::array<cxx::ExpressionAST*, 2>{right, nullptr}
                            : std::array<cxx::ExpressionAST*, 2>{left, right},
        .argument_count = member ? 1u : 2u,
        .evaluation_order = !right_first ? OperatorEvaluationOrder::Forward
                            : member
                                ? OperatorEvaluationOrder::ArgumentsThenReceiver
                                : OperatorEvaluationOrder::ReverseArguments,
        .has_postfix_dummy = false,
        .is_virtual_dispatch = is_virtual_dispatch,
    };
  };
  auto unary =
      [](cxx::FunctionSymbol* function, cxx::ExpressionAST* operand,
         bool postfix,
         bool is_virtual_dispatch) -> std::optional<DirectOperatorCall> {
    if (!function) {
      return std::nullopt;
    }
    const bool member = function->isImplicitObjectMemberFunction();
    return DirectOperatorCall{
        .function = function,
        .receiver = member ? operand : nullptr,
        .arguments = member
                         ? std::array<cxx::ExpressionAST*, 2>{nullptr, nullptr}
                         : std::array<cxx::ExpressionAST*, 2>{operand, nullptr},
        .argument_count = member ? 0u : 1u,
        .evaluation_order = OperatorEvaluationOrder::Forward,
        .has_postfix_dummy = postfix,
        .is_virtual_dispatch = is_virtual_dispatch,
    };
  };
  switch (expression->kind()) {
    case cxx::ASTKind::ImplicitCastExpression: {
      auto* ast = static_cast<cxx::ImplicitCastExpressionAST*>(expression);
      auto* function = ast->conversionFunction;
      return function && !function->isConstructor()
                 ? unary(function, ast->expression, false,
                         ast->isVirtualDispatch)
                 : std::nullopt;
    }
    case cxx::ASTKind::BinaryExpression: {
      auto* ast = static_cast<cxx::BinaryExpressionAST*>(expression);
      return binary(ast->symbol, ast->leftExpression, ast->rightExpression,
                    false, ast->isVirtualDispatch);
    }
    case cxx::ASTKind::AssignmentExpression: {
      auto* ast = static_cast<cxx::AssignmentExpressionAST*>(expression);
      return binary(ast->symbol, ast->leftExpression, ast->rightExpression,
                    true, ast->isVirtualDispatch);
    }
    case cxx::ASTKind::CompoundAssignmentExpression: {
      auto* ast =
          static_cast<cxx::CompoundAssignmentExpressionAST*>(expression);
      return binary(ast->symbol, ast->targetExpression, ast->rightExpression,
                    true, ast->isVirtualDispatch);
    }
    case cxx::ASTKind::PostIncrExpression: {
      auto* ast = static_cast<cxx::PostIncrExpressionAST*>(expression);
      return unary(ast->symbol, ast->baseExpression, true,
                   ast->isVirtualDispatch);
    }
    case cxx::ASTKind::UnaryExpression: {
      auto* ast = static_cast<cxx::UnaryExpressionAST*>(expression);
      return unary(ast->symbol, ast->expression, false, ast->isVirtualDispatch);
    }
    default:
      return std::nullopt;
  }
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

cxx::ConditionExpressionAST* condition_declaration(
    cxx::ExpressionAST* expression) {
  if (auto* constant = cxx::ast_cast<cxx::ConstExpressionAST>(expression)) {
    expression = constant->expression;
  }
  return cxx::ast_cast<cxx::ConditionExpressionAST>(
      strip_implicit_casts(expression));
}

}  // namespace loom::cxx_import
