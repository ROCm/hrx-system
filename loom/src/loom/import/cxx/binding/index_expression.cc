// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/index_expression.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/initialization.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <string>

#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

class PureIndexExpression final : private cxx::ASTVisitor {
 public:
  PureIndexExpression(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                      Types& types, std::string_view purpose)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        purpose_(purpose) {}

  cxx::ExpressionAST* admit(cxx::ExpressionAST* expression) {
    if (!unit_.typeTraits().is_integral_or_enum(expression->type) &&
        !types_.is_index(expression->type, expression)) {
      diagnostics_.reject(
          unit_, expression,
          std::string(purpose_) + " value requires an integer expression");
    }
    accept(expression);
    return expression;
  }

 private:
  bool preVisit(cxx::AST* ast) override {
    auto* expression = cxx::ast_cast<cxx::ExpressionAST>(ast);
    if (!expression) {
      return false;
    }
    if (unit_.typeTraits().is_volatile(expression->type)) {
      diagnostics_.reject(
          unit_, ast,
          std::string(purpose_) + " expressions cannot read volatile values");
    }
    switch (ast->kind()) {
      case cxx::ASTKind::IntLiteralExpression:
        if (!cxx::ast_cast<cxx::IntLiteralExpressionAST>(ast)
                 ->literalOperatorCall) {
          return false;
        }
        break;
      case cxx::ASTKind::FloatLiteralExpression:
        if (!cxx::ast_cast<cxx::FloatLiteralExpressionAST>(ast)
                 ->literalOperatorCall) {
          return false;
        }
        break;
      case cxx::ASTKind::CharLiteralExpression:
        if (!cxx::ast_cast<cxx::CharLiteralExpressionAST>(ast)
                 ->literalOperatorCall) {
          return false;
        }
        break;
      case cxx::ASTKind::IdExpression:
      case cxx::ASTKind::BoolLiteralExpression:
      case cxx::ASTKind::ConstExpression:
      case cxx::ASTKind::SizeofExpression:
      case cxx::ASTKind::SizeofTypeExpression:
      case cxx::ASTKind::AlignofTypeExpression:
        return false;
      case cxx::ASTKind::NestedExpression:
      case cxx::ASTKind::CastExpression:
      case cxx::ASTKind::CppCastExpression:
      case cxx::ASTKind::ConditionalExpression:
      case cxx::ASTKind::MemberExpression:
        return true;
      case cxx::ASTKind::ImplicitCastExpression:
        if (!cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast)
                 ->conversionFunction) {
          return true;
        }
        break;
      case cxx::ASTKind::TypeConstruction:
        if (!cxx::ast_cast<cxx::TypeConstructionAST>(ast)->constructorSymbol) {
          return true;
        }
        break;
      case cxx::ASTKind::SubscriptExpression:
        if (!cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)->symbol) {
          return true;
        }
        break;
      case cxx::ASTKind::UnaryExpression: {
        auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast);
        if (!unary->symbol && unary->op != cxx::TokenKind::T_PLUS_PLUS &&
            unary->op != cxx::TokenKind::T_MINUS_MINUS) {
          return true;
        }
        break;
      }
      case cxx::ASTKind::BinaryExpression: {
        auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast);
        if (!binary->symbol && binary->op != cxx::TokenKind::T_COMMA &&
            binary->op != cxx::TokenKind::T_DOT_STAR &&
            binary->op != cxx::TokenKind::T_MINUS_GREATER_STAR) {
          return true;
        }
        break;
      }
      default:
        break;
    }
    diagnostics_.reject(
        unit_, ast,
        std::string(purpose_) +
            " expressions require pure values without calls, mutation, or "
            "overloaded operations");
  }

  cxx::TranslationUnit& unit_;
  Diagnostics& diagnostics_;
  Types& types_;
  std::string_view purpose_;
};

bool is_indices(cxx::TranslationUnit& unit, const cxx::Type* type) {
  auto* class_type =
      cxx::type_cast<cxx::ClassType>(unit.typeTraits().remove_cv(type));
  auto* definition = class_type ? class_type->definition() : nullptr;
  auto* primary = definition ? cxx::class_template_of(definition) : nullptr;
  auto* type_namespace =
      primary ? cxx::symbol_cast<cxx::NamespaceSymbol>(primary->parent())
              : nullptr;
  auto* loom_namespace =
      type_namespace
          ? cxx::symbol_cast<cxx::NamespaceSymbol>(type_namespace->parent())
          : nullptr;
  return primary && primary->name() && type_namespace &&
         type_namespace->name() && loom_namespace && loom_namespace->name() &&
         cxx::to_string(primary->name()) == "indices" &&
         cxx::to_string(type_namespace->name()) == "type" &&
         cxx::to_string(loom_namespace->name()) == "loom" &&
         (!loom_namespace->parent() || !loom_namespace->parent()->name());
}

cxx::ExpressionAST* unwrap_index_construction(Types& types,
                                              cxx::ExpressionAST* expression) {
  auto* source = unwrap_expression(expression);
  if (!types.is_index(source->type, source)) {
    return expression;
  }
  cxx::List<cxx::ExpressionAST*>* arguments = nullptr;
  if (auto* construction =
          cxx::ast_cast<cxx::BracedTypeConstructionAST>(source)) {
    arguments =
        construction->bracedInitList
            ? cxx::Initializer(construction->bracedInitList).expressionList()
            : nullptr;
  } else if (auto* construction =
                 cxx::ast_cast<cxx::TypeConstructionAST>(source)) {
    arguments = construction->expressionList;
  }
  // index deliberately requires explicit integer construction. The wrapper
  // establishes the source type but contributes no separate computation to an
  // operation-level static/dynamic index list.
  return arguments && !arguments->next ? unwrap_expression(arguments->value)
                                       : expression;
}

cxx::List<cxx::ExpressionAST*>* index_elements(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::BracedTypeConstructionAST* construction, std::string_view purpose) {
  auto* aggregate = construction->bracedInitList;
  auto* members =
      aggregate ? cxx::Initializer(aggregate).expressionList() : nullptr;
  if (!members) {
    return nullptr;
  }
  // indices is an aggregate with one fixed-size array member. CXX retains that
  // member as a nested braced list after normalizing aggregate initialization;
  // the nested expressions are the logical axes exposed by the facade.
  auto* array = !members->next ? cxx::ast_cast<cxx::BracedInitListAST>(
                                     unwrap_expression(members->value))
                               : nullptr;
  if (!array) {
    diagnostics.reject(
        unit, construction,
        std::string(purpose) + " requires every index value to be explicit");
  }
  return cxx::Initializer(array).expressionList();
}

}  // namespace

cxx::ExpressionAST* IndexExpression::admit(cxx::TranslationUnit& unit,
                                           Diagnostics& diagnostics,
                                           Types& types,
                                           cxx::ExpressionAST* expression,
                                           std::string_view purpose) {
  return PureIndexExpression(unit, diagnostics, types, purpose)
      .admit(expression);
}

loom_value_id_t IndexExpression::build(cxx::TranslationUnit& unit,
                                       loom_builder_t* builder,
                                       cxx::ExpressionAST* expression,
                                       loom_value_id_t value,
                                       loom_location_id_t location) {
  auto type = loom_module_value_type(builder->module, value);
  auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(type, index_type)) {
    return value;
  }
  loom_op_t* op;
  auto wide_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  if (!unit.typeTraits().integral_representation(expression->type)->isSigned &&
      !loom_type_equal(type, wide_type)) {
    check(loom_scalar_extui_build(builder, value, type, wide_type, location,
                                  &op));
    value = loom_op_results(op)[0];
    type = wide_type;
  }
  check(loom_index_cast_build(builder, value, type, index_type, location, &op));
  return loom_op_results(op)[0];
}

bool IndexList::is_type(cxx::TranslationUnit& unit, const cxx::Type* type) {
  return is_indices(unit, type);
}

IndexList IndexList::admit(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                           Types& types, cxx::ExpressionAST* expression,
                           std::string_view purpose) {
  auto* source = unwrap_expression(expression);
  auto* construction = cxx::ast_cast<cxx::BracedTypeConstructionAST>(source);
  if (!construction || !construction->bracedInitList ||
      !is_indices(unit, source->type)) {
    diagnostics.reject(unit, expression,
                       std::string(purpose) +
                           " requires a direct loom::type::indices{...} "
                           "value");
  }
  IndexList result;
  for (auto* element : cxx::ListView{
           index_elements(unit, diagnostics, construction, purpose)}) {
    element = unwrap_index_construction(types, element);
    IndexExpression::admit(unit, diagnostics, types, element, purpose);
    auto constant = integer_constant(unit, element);
    if (constant && *constant != INT64_MIN) {
      result.static_values_.push_back(static_cast<int64_t>(*constant));
    } else {
      result.static_values_.push_back(INT64_MIN);
      result.dynamic_expressions_.push_back(element);
    }
  }
  return result;
}

}  // namespace loom::cxx_import
