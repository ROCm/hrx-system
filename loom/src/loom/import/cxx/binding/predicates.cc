// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/predicates.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/attributes.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/token.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/expressions.h"

namespace loom::cxx_import {
namespace {

struct PredicateConstant {
  // Raw value after conversion to the predicate's promoted integer type.
  uint64_t unsigned_value;
  // The same bits interpreted in Loom's signed carrier fact domain.
  int64_t signed_value;
};

using PredicateOperand = std::variant<PredicateValue, PredicateConstant>;

int64_t signed_payload(uint64_t value, int bit_count) {
  if (bit_count == 64) {
    return std::bit_cast<int64_t>(value);
  }
  const uint64_t mask = (UINT64_C(1) << bit_count) - 1;
  value &= mask;
  if (value & (UINT64_C(1) << (bit_count - 1))) {
    value |= ~mask;
  }
  return std::bit_cast<int64_t>(value);
}

std::optional<PredicateConstant> integer_predicate_constant(
    cxx::TranslationUnit& unit, cxx::ExpressionAST* expression, int bit_count) {
  auto constant = scalar_constant(unit, expression);
  if (!constant) {
    return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit);
  auto number = interpreter.toInt(*constant);
  if (!number) {
    return std::nullopt;
  }
  uint64_t raw_value = static_cast<uint64_t>(*number);
  if (bit_count < 64) {
    raw_value &= (UINT64_C(1) << bit_count) - 1;
  }
  return PredicateConstant{
      .unsigned_value = raw_value,
      .signed_value = signed_payload(raw_value, bit_count),
  };
}

std::optional<PredicateConstant> float_predicate_constant(
    cxx::TranslationUnit& unit, cxx::ExpressionAST* expression) {
  auto constant = scalar_constant(unit, expression);
  if (!constant) {
    return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit);
  auto number = interpreter.toDouble(*constant);
  if (!number || !std::isfinite(*number) || std::trunc(*number) != *number ||
      *number < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      *number > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  const auto value = static_cast<int64_t>(*number);
  return PredicateConstant{
      .unsigned_value = static_cast<uint64_t>(value),
      .signed_value = value,
  };
}

std::optional<PredicateConstant> predicate_constant(
    cxx::TranslationUnit& unit, cxx::ExpressionAST* expression,
    const cxx::Type* carrier_type) {
  const auto traits = unit.typeTraits();
  if (traits.is_floating_point(carrier_type)) {
    return float_predicate_constant(unit, expression);
  }
  auto representation = traits.integral_representation(carrier_type);
  if (!representation || representation->bits <= 0 ||
      representation->bits > 64) {
    return std::nullopt;
  }
  return integer_predicate_constant(unit, expression, representation->bits);
}

std::optional<bool> constant_truth(cxx::TranslationUnit& unit,
                                   cxx::ExpressionAST* expression) {
  auto constant = scalar_constant(unit, expression);
  if (!constant) {
    return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit);
  return interpreter.toBool(*constant);
}

std::optional<loom_predicate_kind_t> helper_kind(cxx::TranslationUnit& unit,
                                                 Diagnostics& diagnostics,
                                                 cxx::CallExpressionAST* call) {
  auto* attribute = annotation(direct_callee(call), "predicate");
  if (!attribute) {
    return std::nullopt;
  }
  loom_predicate_kind_t kind;
  if (attribute->arguments.size() != 1 ||
      !loom_predicate_kind_parse(view(attribute->arguments[0]->name()),
                                 &kind)) {
    diagnostics.reject(
        unit, call,
        "predicate helper requires one canonical Loom predicate name");
  }
  return kind;
}

bool is_result_call(cxx::CallExpressionAST* call) {
  return annotation(direct_callee(call), "predicate_result");
}

std::optional<PredicateValue> predicate_value(cxx::TranslationUnit& unit,
                                              Diagnostics& diagnostics,
                                              cxx::ExpressionAST* expression) {
  auto* identity = unwrap_expression(expression);
  std::vector<cxx::FieldSymbol*> members;
  while (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(identity)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    if (!field || field->isStatic() ||
        member->accessOp != cxx::TokenKind::T_DOT) {
      diagnostics.reject(
          unit, member,
          "predicate member paths require direct non-static record fields");
    }
    members.push_back(field);
    identity = unwrap_expression(member->baseExpression);
  }
  std::reverse(members.begin(), members.end());

  PredicateValue value = {};
  value.members = std::move(members);
  value.source = unwrap_expression(expression);
  value.converted = expression;
  value.root_type = identity->type;
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(identity)) {
    value.origin = PredicateValueOrigin::Binding;
    value.binding = id->symbol;
  } else if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(identity);
             call && is_result_call(call)) {
    if (call->expressionList) {
      diagnostics.reject(unit, call,
                         "predicate result placeholder takes no arguments");
    }
    value.origin = PredicateValueOrigin::Result;
  } else {
    return std::nullopt;
  }

  if (unit.typeTraits().is_volatile(value.source->type)) {
    diagnostics.reject(
        unit, value.source,
        "predicate expressions cannot read volatile values because no "
        "runtime comparison is emitted");
  }
  return value;
}

PredicateOperand predicate_operand(cxx::TranslationUnit& unit,
                                   Diagnostics& diagnostics,
                                   cxx::ExpressionAST* expression,
                                   const cxx::Type* carrier_type) {
  if (auto constant = predicate_constant(unit, expression, carrier_type)) {
    return *constant;
  }
  if (auto value = predicate_value(unit, diagnostics, expression)) {
    return std::move(*value);
  }
  diagnostics.reject(
      unit, expression,
      "predicate operands must be retained scalar values or pure constants "
      "without calls, mutation, or value-changing casts");
}

bool same_value(const PredicateValue& left, const PredicateValue& right) {
  return left.origin == right.origin && left.binding == right.binding &&
         left.members == right.members;
}

void reject_false(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                  cxx::ExpressionAST* expression) {
  diagnostics.reject(unit, expression,
                     "predicate condition is statically false; Loom has no "
                     "unreachable-path contract representation");
}

void append_constant_predicate(loom_predicate_kind_t kind, PredicateValue value,
                               int64_t constant,
                               std::vector<ProjectedPredicate>& output) {
  ProjectedPredicate projected = {};
  projected.values[0] = std::move(value);
  projected.value_count = 1;
  projected.predicate = {
      .kind = kind,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .args = {0, constant},
  };
  output.push_back(std::move(projected));
}

void append_range_predicate(PredicateValue value, int64_t lower, int64_t upper,
                            std::vector<ProjectedPredicate>& output) {
  ProjectedPredicate projected = {};
  projected.values[0] = std::move(value);
  projected.value_count = 1;
  projected.predicate = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {0, lower, upper},
  };
  output.push_back(std::move(projected));
}

void append_value_predicate(loom_predicate_kind_t kind, PredicateValue left,
                            PredicateValue right,
                            std::vector<ProjectedPredicate>& output) {
  ProjectedPredicate projected = {};
  projected.values[0] = std::move(left);
  projected.values[1] = std::move(right);
  projected.value_count = 2;
  projected.predicate = {
      .kind = kind,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .args = {0, 1},
  };
  output.push_back(std::move(projected));
}

loom_predicate_kind_t relation_kind(cxx::TranslationUnit& unit,
                                    Diagnostics& diagnostics,
                                    cxx::BinaryExpressionAST* comparison) {
  if (comparison->symbol) {
    diagnostics.reject(unit, comparison,
                       "predicate comparisons require builtin operators");
  }
  switch (comparison->op) {
    case cxx::TokenKind::T_EQUAL_EQUAL:
      return LOOM_PREDICATE_EQ;
    case cxx::TokenKind::T_EXCLAIM_EQUAL:
      return LOOM_PREDICATE_NE;
    case cxx::TokenKind::T_LESS:
      return LOOM_PREDICATE_LT;
    case cxx::TokenKind::T_LESS_EQUAL:
      return LOOM_PREDICATE_LE;
    case cxx::TokenKind::T_GREATER:
      return LOOM_PREDICATE_GT;
    case cxx::TokenKind::T_GREATER_EQUAL:
      return LOOM_PREDICATE_GE;
    default:
      diagnostics.reject(
          unit, comparison,
          "predicate expressions require comparisons, predicate helpers, "
          "or conjunctions joined by &&");
  }
}

loom_predicate_kind_t swap_relation(loom_predicate_kind_t kind) {
  switch (kind) {
    case LOOM_PREDICATE_LT:
      return LOOM_PREDICATE_GT;
    case LOOM_PREDICATE_LE:
      return LOOM_PREDICATE_GE;
    case LOOM_PREDICATE_GT:
      return LOOM_PREDICATE_LT;
    case LOOM_PREDICATE_GE:
      return LOOM_PREDICATE_LE;
    default:
      return kind;
  }
}

loom_predicate_kind_t unsigned_relation_kind(loom_predicate_kind_t kind) {
  switch (kind) {
    case LOOM_PREDICATE_LT:
      return LOOM_PREDICATE_ULT;
    case LOOM_PREDICATE_LE:
      return LOOM_PREDICATE_ULE;
    case LOOM_PREDICATE_GT:
      return LOOM_PREDICATE_UGT;
    case LOOM_PREDICATE_GE:
      return LOOM_PREDICATE_UGE;
    default:
      return kind;
  }
}

void append_unsigned_constant_predicate(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::ExpressionAST* expression, loom_predicate_kind_t kind,
    PredicateValue value, PredicateConstant constant, int bit_count,
    std::vector<ProjectedPredicate>& predicates) {
  const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
  const uint64_t signed_maximum = sign_bit - 1;
  const uint64_t unsigned_maximum =
      bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
  const uint64_t bound = constant.unsigned_value;
  switch (kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
      append_constant_predicate(kind, std::move(value), constant.signed_value,
                                predicates);
      return;
    case LOOM_PREDICATE_LT:
      if (bound == 0) {
        reject_false(unit, diagnostics, expression);
      }
      if (bound <= sign_bit) {
        append_range_predicate(std::move(value), 0,
                               static_cast<int64_t>(bound - 1), predicates);
        return;
      }
      if (bound == unsigned_maximum) {
        append_constant_predicate(LOOM_PREDICATE_NE, std::move(value),
                                  signed_payload(unsigned_maximum, bit_count),
                                  predicates);
        return;
      }
      break;
    case LOOM_PREDICATE_LE:
      if (bound <= signed_maximum) {
        append_range_predicate(std::move(value), 0, static_cast<int64_t>(bound),
                               predicates);
        return;
      }
      if (bound == unsigned_maximum) {
        return;
      }
      if (bound == unsigned_maximum - 1) {
        append_constant_predicate(LOOM_PREDICATE_NE, std::move(value),
                                  signed_payload(unsigned_maximum, bit_count),
                                  predicates);
        return;
      }
      break;
    case LOOM_PREDICATE_GT:
      if (bound == 0) {
        append_constant_predicate(LOOM_PREDICATE_NE, std::move(value), 0,
                                  predicates);
        return;
      }
      if (bound == unsigned_maximum) {
        reject_false(unit, diagnostics, expression);
      }
      if (bound >= signed_maximum) {
        append_range_predicate(std::move(value),
                               signed_payload(bound + 1, bit_count), -1,
                               predicates);
        return;
      }
      break;
    case LOOM_PREDICATE_GE:
      if (bound == 0) {
        return;
      }
      if (bound == 1) {
        append_constant_predicate(LOOM_PREDICATE_NE, std::move(value), 0,
                                  predicates);
        return;
      }
      if (bound >= sign_bit) {
        append_range_predicate(std::move(value), constant.signed_value, -1,
                               predicates);
        return;
      }
      break;
    default:
      break;
  }
  append_constant_predicate(unsigned_relation_kind(kind), std::move(value),
                            constant.signed_value, predicates);
}

void append_helper_predicate(cxx::TranslationUnit& unit,
                             Diagnostics& diagnostics,
                             cxx::CallExpressionAST* call,
                             loom_predicate_kind_t kind,
                             PredicateSubject subject,
                             std::vector<ProjectedPredicate>& output) {
  const uint8_t expected_count = loom_predicate_kind_argument_count(kind);
  size_t source_count = 0;
  for (auto* ignored : cxx::ListView{call->expressionList}) {
    (void)ignored;
    ++source_count;
  }
  const size_t implicit_count = subject == PredicateSubject::Implicit ? 1 : 0;
  if (expected_count < implicit_count ||
      source_count != expected_count - implicit_count) {
    diagnostics.reject(
        unit, call,
        std::string(loom_predicate_kind_name(kind)) +
            (subject == PredicateSubject::Implicit
                 ? " config predicate has the wrong argument count"
                 : " predicate has the wrong argument count"));
  }

  ProjectedPredicate projected = {};
  projected.predicate.kind = kind;
  projected.predicate.arg_count = expected_count;
  if (subject == PredicateSubject::Implicit) {
    projected.values[0].origin = PredicateValueOrigin::Subject;
    projected.predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
    projected.predicate.args[0] = 0;
    projected.value_count = 1;
  }

  const cxx::Type* carrier_type =
      call->expressionList ? call->expressionList->value->type : nullptr;
  size_t argument_index = implicit_count;
  for (auto* expression : cxx::ListView{call->expressionList}) {
    if (!carrier_type) {
      carrier_type = expression->type;
    }
    auto operand =
        predicate_operand(unit, diagnostics, expression, carrier_type);
    projected.argument_types[argument_index] = expression->type;
    if (auto* value = std::get_if<PredicateValue>(&operand)) {
      if (!predicate_accepts_source_type(unit, kind, value->source->type)) {
        diagnostics.reject(unit, value->source,
                           std::string(loom_predicate_kind_name(kind)) +
                               " does not accept this C++ value type");
      }
      projected.predicate.arg_tags[argument_index] = LOOM_PRED_ARG_VALUE;
      projected.predicate.args[argument_index] = projected.value_count;
      projected.values[projected.value_count++] = std::move(*value);
    } else {
      auto constant = std::get<PredicateConstant>(operand);
      projected.predicate.arg_tags[argument_index] = LOOM_PRED_ARG_CONST;
      projected.predicate.args[argument_index] = constant.signed_value;
    }
    ++argument_index;
  }
  if (projected.predicate.arg_tags[0] != LOOM_PRED_ARG_VALUE) {
    diagnostics.reject(unit, call,
                       "a predicate's first argument must be a retained "
                       "value or an implicit config subject");
  }
  if (kind == LOOM_PREDICATE_MULTIPLE_OF &&
      projected.predicate.arg_tags[1] == LOOM_PRED_ARG_CONST &&
      projected.predicate.args[1] <= 0) {
    diagnostics.reject(unit, call, "multiple_of requires a positive divisor");
  }
  if (kind == LOOM_PREDICATE_RANGE &&
      projected.predicate.arg_tags[1] == LOOM_PRED_ARG_CONST &&
      projected.predicate.arg_tags[2] == LOOM_PRED_ARG_CONST &&
      projected.predicate.args[1] > projected.predicate.args[2]) {
    reject_false(unit, diagnostics, call);
  }
  output.push_back(std::move(projected));
}

void collect_predicates(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                        cxx::ExpressionAST* expression,
                        PredicateSubject subject,
                        std::vector<ProjectedPredicate>& predicates) {
  auto* condition =
      cxx::ast_cast<cxx::BinaryExpressionAST>(unwrap_expression(expression));
  if (condition && !condition->symbol &&
      condition->op == cxx::TokenKind::T_AMP_AMP) {
    collect_predicates(unit, diagnostics, condition->leftExpression, subject,
                       predicates);
    collect_predicates(unit, diagnostics, condition->rightExpression, subject,
                       predicates);
    return;
  }
  if (auto truth = constant_truth(unit, expression)) {
    if (*truth) {
      return;
    }
    reject_false(unit, diagnostics, expression);
  }

  if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(
          unwrap_expression(expression))) {
    if (auto kind = helper_kind(unit, diagnostics, call)) {
      append_helper_predicate(unit, diagnostics, call, *kind, subject,
                              predicates);
      return;
    }
  }

  const auto traits = unit.typeTraits();
  if (auto value = predicate_value(unit, diagnostics, expression);
      value && traits.is_integral(value->source->type)) {
    append_constant_predicate(LOOM_PREDICATE_NE, std::move(*value), 0,
                              predicates);
    return;
  }
  if (auto* unary =
          cxx::ast_cast<cxx::UnaryExpressionAST>(unwrap_expression(expression));
      unary && !unary->symbol && unary->op == cxx::TokenKind::T_EXCLAIM) {
    if (auto value = predicate_value(unit, diagnostics, unary->expression);
        value && traits.is_integral(value->source->type)) {
      append_constant_predicate(LOOM_PREDICATE_EQ, std::move(*value), 0,
                                predicates);
      return;
    }
  }

  if (!condition) {
    diagnostics.reject(
        unit, expression,
        "predicate expressions require comparisons, predicate helpers, or "
        "conjunctions joined by &&");
  }
  auto kind = relation_kind(unit, diagnostics, condition);
  const bool floating =
      traits.is_floating_point(condition->leftExpression->type);
  if (floating) {
    if (!traits.is_floating_point(condition->rightExpression->type) ||
        (kind != LOOM_PREDICATE_EQ && kind != LOOM_PREDICATE_NE)) {
      diagnostics.reject(unit, condition,
                         "floating predicate comparisons support only == and "
                         "!=; use classification helpers for other facts");
    }
  } else if (!traits.is_integral(condition->leftExpression->type) ||
             !traits.is_integral(condition->rightExpression->type)) {
    diagnostics.reject(unit, condition,
                       "predicate comparisons require numeric scalar "
                       "operands");
  }
  auto representation =
      traits.integral_representation(condition->leftExpression->type);
  if (!floating && (!representation || representation->bits <= 0 ||
                    representation->bits > 64)) {
    diagnostics.reject(unit, condition,
                       "predicate integer carriers must be at most 64 bits");
  }
  auto left = predicate_operand(unit, diagnostics, condition->leftExpression,
                                condition->leftExpression->type);
  auto right = predicate_operand(unit, diagnostics, condition->rightExpression,
                                 condition->leftExpression->type);
  if (std::holds_alternative<PredicateConstant>(left) &&
      std::holds_alternative<PredicateValue>(right)) {
    std::swap(left, right);
    kind = swap_relation(kind);
  }
  auto* left_value = std::get_if<PredicateValue>(&left);
  auto* right_value = std::get_if<PredicateValue>(&right);
  if (!left_value) {
    reject_false(unit, diagnostics, expression);
  }
  if (right_value) {
    if (same_value(*left_value, *right_value)) {
      if (kind == LOOM_PREDICATE_EQ || kind == LOOM_PREDICATE_LE ||
          kind == LOOM_PREDICATE_GE) {
        return;
      }
      reject_false(unit, diagnostics, expression);
    }
    if (!floating && !representation->isSigned) {
      kind = unsigned_relation_kind(kind);
    }
    append_value_predicate(kind, std::move(*left_value),
                           std::move(*right_value), predicates);
    return;
  }

  const auto& right_constant = std::get<PredicateConstant>(right);
  if (floating || representation->isSigned) {
    append_constant_predicate(kind, std::move(*left_value),
                              right_constant.signed_value, predicates);
    return;
  }
  append_unsigned_constant_predicate(unit, diagnostics, expression, kind,
                                     std::move(*left_value), right_constant,
                                     representation->bits, predicates);
}

}  // namespace

bool predicate_accepts_source_type(cxx::TranslationUnit& unit,
                                   loom_predicate_kind_t kind,
                                   const cxx::Type* type) {
  const auto traits = unit.typeTraits();
  switch (kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
      return traits.is_arithmetic(type) || traits.is_enum(type);
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return traits.is_floating_point(type);
    default: {
      auto representation = traits.integral_representation(type);
      return representation && representation->bits > 0 &&
             representation->bits <= 64;
    }
  }
}

std::vector<ProjectedPredicate> project_predicates(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::ExpressionAST* expression, PredicateSubject subject) {
  std::vector<ProjectedPredicate> predicates;
  collect_predicates(unit, diagnostics, expression, subject, predicates);
  return predicates;
}

}  // namespace loom::cxx_import
