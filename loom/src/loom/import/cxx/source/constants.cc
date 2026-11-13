// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/constants.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_visitor.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <utility>

#include "iree/base/internal/unicode.h"
#include "loom/import/cxx/source/frontend.h"

namespace loom::cxx_import {
namespace {

enum class ConstantDomain { Integer, Scalar };

class PureConstant final : private cxx::ASTVisitor {
 public:
  PureConstant(cxx::TranslationUnit& unit, ConstantDomain domain)
      : unit_(unit), domain_(domain) {}

  std::optional<cxx::ConstValue> evaluate(cxx::ExpressionAST* expression) {
    accept(expression);
    if (!pure_) {
      return std::nullopt;
    }
    cxx::ASTInterpreter interpreter(&unit_);
    return interpreter.evaluate(expression);
  }

 private:
  bool preVisit(cxx::AST* ast) override {
    auto* expression = cxx::ast_cast<cxx::ExpressionAST>(ast);
    if (!pure_ || !expression) {
      return false;
    }
    auto traits = unit_.typeTraits();
    if ((!traits.is_integral_or_enum(expression->type) &&
         !(domain_ == ConstantDomain::Scalar &&
           traits.is_floating_point(expression->type))) ||
        traits.is_volatile(expression->type)) {
      pure_ = false;
      return false;
    }
    switch (ast->kind()) {
      case cxx::ASTKind::IntLiteralExpression:
        pure_ = !cxx::ast_cast<cxx::IntLiteralExpressionAST>(ast)
                     ->literalOperatorCall;
        return false;
      case cxx::ASTKind::FloatLiteralExpression:
        pure_ = !cxx::ast_cast<cxx::FloatLiteralExpressionAST>(ast)
                     ->literalOperatorCall;
        return false;
      case cxx::ASTKind::CharLiteralExpression:
        pure_ = !cxx::ast_cast<cxx::CharLiteralExpressionAST>(ast)
                     ->literalOperatorCall;
        return false;
      case cxx::ASTKind::IdExpression: {
        auto* symbol = cxx::ast_cast<cxx::IdExpressionAST>(ast)->symbol;
        auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(symbol);
        pure_ =
            cxx::symbol_cast<cxx::EnumeratorSymbol>(symbol) ||
            (variable && variable->constValue() &&
             (variable->isConstexpr() || traits.is_const(variable->type())));
        return false;
      }
      case cxx::ASTKind::BoolLiteralExpression:
      case cxx::ASTKind::ConstExpression:
      case cxx::ASTKind::SizeofExpression:
      case cxx::ASTKind::SizeofTypeExpression:
      case cxx::ASTKind::AlignofTypeExpression:
      case cxx::ASTKind::BuiltinOffsetofExpression:
        return false;
      case cxx::ASTKind::NestedExpression:
      case cxx::ASTKind::CastExpression:
      case cxx::ASTKind::CppCastExpression:
      case cxx::ASTKind::ConditionalExpression:
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
      case cxx::ASTKind::UnaryExpression: {
        auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast);
        if (!unary->symbol && (unary->op == cxx::TokenKind::T_PLUS ||
                               unary->op == cxx::TokenKind::T_MINUS ||
                               unary->op == cxx::TokenKind::T_TILDE ||
                               unary->op == cxx::TokenKind::T_EXCLAIM)) {
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
    pure_ = false;
    return false;
  }

  // Resolved source types and retained frontend constants for this invocation.
  cxx::TranslationUnit& unit_;
  // Source grammar requested by the owning bound or literal admission.
  ConstantDomain domain_;
  // Admission result; unsupported syntax prevents any interpretation.
  bool pure_ = true;
};

}  // namespace

std::optional<ObjectRepresentation> object_representation(
    cxx::TranslationUnit& unit, const cxx::Type* type,
    const cxx::ConstValue& value, const cxx::StringLiteral* string) {
  cxx::ASTInterpreter interpreter(&unit);
  auto bytes = interpreter.objectRepresentation(value, type);
  if (!bytes) {
    return std::nullopt;
  }
  auto traits = unit.typeTraits();
  const bool is_utf8 =
      string && string->codeUnitSize() == 1 && traits.is_array(type) &&
      traits.is_char_type(traits.remove_cv(traits.get_element_type(type))) &&
      iree_unicode_utf8_validate(iree_make_string_view(
          reinterpret_cast<const char*>(bytes->data()), bytes->size()));
  return ObjectRepresentation{std::move(*bytes), is_utf8};
}

std::optional<cxx::ConstValue> scalar_constant(cxx::TranslationUnit& unit,
                                               cxx::ExpressionAST* expression) {
  return PureConstant(unit, ConstantDomain::Scalar).evaluate(expression);
}

std::optional<cxx::ConstValue> floating_builtin_constant(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::CallExpressionAST* call) {
  auto* base = call->baseExpression;
  while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(base)) {
    base = nested->expression;
  }
  auto* callee = cxx::ast_cast<cxx::IdExpressionAST>(base);
  using Builtin = cxx::BuiltinFunctionKind;
  switch (cxx::resolveBuiltinFunctionKind(callee)) {
    case Builtin::T___BUILTIN_INF:
    case Builtin::T___BUILTIN_INFF:
    case Builtin::T___BUILTIN_INFL:
    case Builtin::T___BUILTIN_HUGE_VAL:
    case Builtin::T___BUILTIN_HUGE_VALF:
    case Builtin::T___BUILTIN_HUGE_VALL:
    case Builtin::T___BUILTIN_NAN:
    case Builtin::T___BUILTIN_NANF:
    case Builtin::T___BUILTIN_NANL:
      break;
    case Builtin::T___BUILTIN_NANS:
    case Builtin::T___BUILTIN_NANSF:
    case Builtin::T___BUILTIN_NANSL:
      diagnostics.reject(unit, call,
                         "signaling NaN builtins require frontend constant "
                         "evaluation support; use __builtin_bit_cast to "
                         "specify the representation bits");
    default:
      return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit);
  auto value = interpreter.evaluate(call);
  if (!value) {
    diagnostics.reject(
        unit, call,
        "NaN builtin requires a constant null-terminated integer payload "
        "string");
  }
  return value;
}

bool null_pointer_expression(cxx::TranslationUnit& unit,
                             cxx::ExpressionAST* expression) {
  auto traits = unit.typeTraits();
  for (;;) {
    if (traits.is_null_pointer(expression->type)) {
      return true;
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      expression = nested->expression;
      continue;
    }
    if (!traits.is_pointer(expression->type)) {
      auto value = scalar_constant(unit, expression);
      auto* integer = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
      return integer && integer->isZero();
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression);
        cast && !cast->conversionFunction) {
      expression = cast->expression;
      continue;
    }
    if (auto* cast = cxx::ast_cast<cxx::CastExpressionAST>(expression)) {
      expression = cast->expression;
      continue;
    }
    if (auto* cast = cxx::ast_cast<cxx::CppCastExpressionAST>(expression)) {
      expression = cast->expression;
      continue;
    }
    if (auto* construction =
            cxx::ast_cast<cxx::TypeConstructionAST>(expression)) {
      if (!construction->expressionList) {
        return true;
      }
      expression = construction->expressionList->value;
      continue;
    }
    if (auto* construction =
            cxx::ast_cast<cxx::BracedTypeConstructionAST>(expression)) {
      auto* elements = construction->bracedInitList->expressionList;
      if (!elements) {
        return true;
      }
      expression = elements->value;
      continue;
    }
    const cxx::ConstValue* value = nullptr;
    if (auto* constant = cxx::ast_cast<cxx::ConstExpressionAST>(expression)) {
      value = constant->constValue ? &*constant->constValue : nullptr;
    } else if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
      auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(id->symbol);
      if (variable && variable->constValue() &&
          (variable->isConstexpr() || traits.is_const(variable->type()))) {
        value = &*variable->constValue();
      }
    }
    auto* integer = value ? std::get_if<cxx::ConstInt>(value) : nullptr;
    return integer && integer->isZero();
  }
}

std::optional<std::intmax_t> integer_constant(cxx::TranslationUnit& unit,
                                              cxx::ExpressionAST* expression) {
  auto evaluated =
      PureConstant(unit, ConstantDomain::Integer).evaluate(expression);
  cxx::ASTInterpreter interpreter(&unit);
  auto value = evaluated ? interpreter.toInt(*evaluated) : std::nullopt;
  if (value && *value < 0 &&
      !unit.typeTraits().integral_representation(expression->type)->isSigned) {
    return std::nullopt;
  }
  return value;
}

}  // namespace loom::cxx_import
