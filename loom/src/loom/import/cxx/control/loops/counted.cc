// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/loops/counted.h"

#include <cxx/ast.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

// CXX materializes trivial by-value class arguments as an implicit copy around
// a one-element paren initializer. Peel only those same-type copies so loop
// admission still reasons about the resolved source binding identity.
cxx::IdExpressionAST* address_identity(cxx::TranslationUnit& unit,
                                       const AddressPartition& address,
                                       cxx::ExpressionAST* expression) {
  while (expression) {
    if (auto* cast =
            cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
      if (cast->conversionFunction &&
          !same_function(cast->conversionFunction,
                         address.source->copyConstructor()) &&
          !same_function(cast->conversionFunction,
                         address.source->moveConstructor())) {
        return nullptr;
      }
      expression = cast->expression;
      continue;
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      expression = nested->expression;
      continue;
    }
    if (auto* copy = cxx::ast_cast<cxx::ParenInitializerAST>(expression)) {
      auto* arguments = cxx::Initializer(copy).expressionList();
      if (!arguments || arguments->next) {
        return nullptr;
      }
      expression = arguments->value;
      continue;
    }
    break;
  }
  auto* identity = cxx::ast_cast<cxx::IdExpressionAST>(expression);
  if (!identity) {
    return nullptr;
  }
  cxx::TypeTraits traits(&unit);
  return traits.is_same(traits.remove_cv(identity->type),
                        address.source->type())
             ? identity
             : nullptr;
}

std::optional<std::intmax_t> address_constant(cxx::TranslationUnit& unit,
                                              const AddressPartition& address,
                                              cxx::ExpressionAST* expression) {
  expression = unwrap_expression(expression);
  cxx::TypeTraits traits(&unit);
  if (!traits.is_same(traits.remove_cv(expression->type),
                      address.source->type())) {
    return std::nullopt;
  }
  cxx::List<cxx::ExpressionAST*>* arguments = nullptr;
  if (auto* construction =
          cxx::ast_cast<cxx::BracedTypeConstructionAST>(expression)) {
    arguments =
        construction->bracedInitList
            ? cxx::Initializer(construction->bracedInitList).expressionList()
            : nullptr;
  } else if (auto* construction =
                 cxx::ast_cast<cxx::TypeConstructionAST>(expression)) {
    arguments = construction->expressionList;
  }
  return arguments && !arguments->next
             ? integer_constant(unit, arguments->value)
             : std::nullopt;
}

bool is_address_unit(cxx::TranslationUnit& unit,
                     const AddressPartition& address,
                     cxx::ExpressionAST* expression) {
  auto value = address_constant(unit, address, expression);
  return value && *value == 1;
}

struct IntegerIncrement {
  // Identifier updated by the source step.
  cxx::IdExpressionAST* induction;
  // Exact positive increment in source arithmetic.
  unsigned step;
};

std::optional<IntegerIncrement> classify_integer_increment(
    cxx::TranslationUnit& unit, cxx::ExpressionAST* expression,
    std::uintmax_t maximum_step) {
  expression = unwrap_expression(expression);
  if (auto* prefix = cxx::ast_cast<cxx::UnaryExpressionAST>(expression);
      prefix && !prefix->symbol && prefix->op == cxx::TokenKind::T_PLUS_PLUS) {
    return IntegerIncrement{cxx::ast_cast<cxx::IdExpressionAST>(
                                unwrap_expression(prefix->expression)),
                            1};
  }
  if (auto* postfix = cxx::ast_cast<cxx::PostIncrExpressionAST>(expression);
      postfix && !postfix->symbol &&
      postfix->op == cxx::TokenKind::T_PLUS_PLUS) {
    return IntegerIncrement{cxx::ast_cast<cxx::IdExpressionAST>(
                                unwrap_expression(postfix->baseExpression)),
                            1};
  }
  auto* compound =
      cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(expression);
  if (!compound || compound->symbol ||
      compound->op != cxx::TokenKind::T_PLUS_EQUAL) {
    return std::nullopt;
  }
  auto amount = integer_constant(unit, compound->rightExpression);
  if (!amount || *amount <= 0 ||
      static_cast<std::uintmax_t>(*amount) > maximum_step) {
    return std::nullopt;
  }
  return IntegerIncrement{cxx::ast_cast<cxx::IdExpressionAST>(
                              unwrap_expression(compound->targetExpression)),
                          static_cast<unsigned>(*amount)};
}

std::optional<CountedLoop> integer_interval(
    cxx::TranslationUnit& unit, cxx::ForStatementAST* loop,
    cxx::Symbol* induction, cxx::TypeKind kind,
    cxx::BinaryExpressionAST* condition) {
  cxx::TypeTraits traits(&unit);
  auto source_kind = [&](const cxx::Type* type) {
    return cxx::unattributed_type(traits.remove_cv(type))->kind();
  };
  auto* left = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_expression(condition->leftExpression));
  if (condition->symbol || !left || left->symbol != induction ||
      source_kind(condition->leftExpression->type) != kind ||
      source_kind(condition->rightExpression->type) != kind) {
    return std::nullopt;
  }
  const bool is_signed = kind == cxx::TypeKind::kInt;
  if (is_signed) {
    auto* compound = cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(
        unwrap_expression(loop->expression));
    if (compound && source_kind(compound->rightExpression->type) != kind) {
      return std::nullopt;
    }
  }

  // An executed induction can reach upper - 1. Even on the last iteration
  // the source step must fit, independently of where the interval starts.
  const std::int64_t maximum = is_signed ? std::int64_t{INT32_MAX} : UINT32_MAX;
  const std::int64_t minimum = is_signed ? INT32_MIN : 0;
  const loom_scalar_type_t domain =
      is_signed ? LOOM_SCALAR_TYPE_INDEX : LOOM_SCALAR_TYPE_OFFSET;
  auto increment = classify_integer_increment(unit, loop->expression, maximum);
  if (!increment || !increment->induction ||
      increment->induction->symbol != induction) {
    return std::nullopt;
  }
  if (auto upper = integer_constant(unit, condition->rightExpression)) {
    if (*upper < minimum || *upper > maximum - increment->step + 1) {
      return std::nullopt;
    }
    return CountedLoop{induction, *upper, increment->step, domain,
                       /*native=*/false};
  }
  auto* bound = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_expression(condition->rightExpression));
  if (increment->step != 1 || !bound ||
      !traits.is_integral_or_enum(bound->type) ||
      traits.is_volatile(bound->type)) {
    return std::nullopt;
  }
  return CountedLoop{induction,
                     CountedLoop::Bound{bound, condition->rightExpression}, 1,
                     domain, /*native=*/false};
}

std::optional<CountedLoop> address_interval(
    cxx::TranslationUnit& unit, Types& types, cxx::ForStatementAST* loop,
    cxx::Symbol* induction, cxx::BinaryExpressionAST* condition) {
  const auto& partition = types.partition(induction->type(), loop);
  if (partition.kind != ValueKind::Address) {
    return std::nullopt;
  }
  const auto& address = static_cast<const AddressPartition&>(partition);
  auto domain = address.element_type;
  if (domain != LOOM_SCALAR_TYPE_INDEX && domain != LOOM_SCALAR_TYPE_OFFSET) {
    return std::nullopt;
  }
  auto predicate = domain == LOOM_SCALAR_TYPE_INDEX ? "slt" : "ult";
  auto* left = address_identity(unit, address, condition->leftExpression);
  auto* bound = address_identity(unit, address, condition->rightExpression);
  if (!left || left->symbol != induction ||
      !has_op_binding(condition->symbol, "index.cmp", predicate) ||
      (!bound &&
       !address_constant(unit, address, condition->rightExpression))) {
    return std::nullopt;
  }
  auto* assignment = cxx::ast_cast<cxx::AssignmentExpressionAST>(
      unwrap_expression(loop->expression));
  auto* increment = assignment
                        ? cxx::ast_cast<cxx::IdExpressionAST>(
                              unwrap_expression(assignment->leftExpression))
                        : nullptr;
  auto* addition = assignment
                       ? cxx::ast_cast<cxx::BinaryExpressionAST>(
                             unwrap_expression(assignment->rightExpression))
                       : nullptr;
  if (!assignment || assignment->op != cxx::TokenKind::T_EQUAL || !increment ||
      increment->symbol != induction || !addition ||
      !has_op_binding(addition->symbol, "index.add")) {
    return std::nullopt;
  }
  auto* addition_left =
      address_identity(unit, address, addition->leftExpression);
  auto* addition_right =
      address_identity(unit, address, addition->rightExpression);
  cxx::ExpressionAST* amount = nullptr;
  if (addition_left && addition_left->symbol == induction) {
    amount = addition->rightExpression;
  } else if (addition_right && addition_right->symbol == induction) {
    amount = addition->leftExpression;
  } else {
    return std::nullopt;
  }
  if (!is_address_unit(unit, address, amount)) {
    return std::nullopt;
  }

  if (bound && cxx::TypeTraits(&unit).is_volatile(bound->type)) {
    return std::nullopt;
  }
  return CountedLoop{induction,
                     CountedLoop::Bound{bound, condition->rightExpression}, 1,
                     domain, /*native=*/true};
}

}  // namespace

std::optional<CountedLoop> CountedLoop::recognize(cxx::TranslationUnit& unit,
                                                  Types& types,
                                                  cxx::ForStatementAST* loop) {
  auto* declaration =
      cxx::ast_cast<cxx::DeclarationStatementAST>(loop->initializer);
  auto* initial =
      declaration
          ? cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration)
          : nullptr;
  auto* condition = cxx::ast_cast<cxx::BinaryExpressionAST>(
      unwrap_expression(loop->condition));
  if (!initial || !initial->initDeclaratorList ||
      initial->initDeclaratorList->next || !condition ||
      condition->op != cxx::TokenKind::T_LESS) {
    return std::nullopt;
  }
  auto* induction = initial->initDeclaratorList->value->symbol;
  cxx::TypeTraits traits(&unit);
  if (!induction || traits.is_volatile(induction->type())) {
    return std::nullopt;
  }
  auto kind =
      cxx::unattributed_type(traits.remove_cv(induction->type()))->kind();
  switch (kind) {
    case cxx::TypeKind::kInt:
    case cxx::TypeKind::kUnsignedInt:
      return integer_interval(unit, loop, induction, kind, condition);
    case cxx::TypeKind::kClass:
      return address_interval(unit, types, loop, induction, condition);
    default:
      return std::nullopt;
  }
}

}  // namespace loom::cxx_import
