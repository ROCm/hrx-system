// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/target_requirements.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/const_int.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cstdint>
#include <optional>
#include <utility>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/source.h"
#include "loom/ops/target/ops.h"

namespace loom::cxx_import {
namespace {

std::optional<TargetRequirementKind> target_query(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::ExpressionAST* expression) {
  auto* call =
      cxx::ast_cast<cxx::CallExpressionAST>(unwrap_expression(expression));
  if (!has_op_binding(direct_callee(call), "target.subgroup.size")) {
    return std::nullopt;
  }
  auto representation = unit.typeTraits().integral_representation(call->type);
  if (call->expressionList || !representation || representation->isSigned ||
      representation->bits != 32) {
    diagnostics.reject(
        unit, call,
        "target.subgroup.size requirements require an unsigned() query");
  }
  return TargetRequirementKind::SubgroupSize;
}

}  // namespace

std::optional<ProjectedTargetRequirement> project_target_requirement(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::ExpressionAST* expression) {
  auto* comparison =
      cxx::ast_cast<cxx::BinaryExpressionAST>(unwrap_expression(expression));
  if (!comparison) {
    if (target_query(unit, diagnostics, expression)) {
      diagnostics.reject(
          unit, expression,
          "target requirements require an exact equality comparison");
    }
    return std::nullopt;
  }

  auto left = target_query(unit, diagnostics, comparison->leftExpression);
  auto right = target_query(unit, diagnostics, comparison->rightExpression);
  if (!left && !right) {
    return std::nullopt;
  }
  if (comparison->symbol || comparison->op != cxx::TokenKind::T_EQUAL_EQUAL) {
    diagnostics.reject(
        unit, comparison,
        "target requirements require an exact equality comparison");
  }
  if (left && right) {
    diagnostics.reject(unit, comparison,
                       "a target requirement must compare one query to one "
                       "pure integer constant");
  }

  auto* constant_expression =
      left ? comparison->rightExpression : comparison->leftExpression;
  auto evaluated = scalar_constant(unit, constant_expression);
  auto* value = evaluated ? std::get_if<cxx::ConstInt>(&*evaluated) : nullptr;
  if (!value) {
    diagnostics.reject(unit, constant_expression,
                       "a target requirement must compare one query to one "
                       "pure integer constant");
  }
  if (value->isNegative() || value->isZero() || value->toUWide() > UINT32_MAX) {
    diagnostics.reject(
        unit, constant_expression,
        "target.subgroup.size requires a nonzero unsigned 32-bit constant");
  }
  return ProjectedTargetRequirement{
      .kind = left ? *left : *right,
      .value = static_cast<int64_t>(value->toUIntMax()),
      .source = expression,
  };
}

loom_attribute_t materialize_target_requirement(
    loom_module_t* module, const ProjectedTargetRequirement& requirement) {
  loom_attribute_t attribute;
  switch (requirement.kind) {
    case TargetRequirementKind::SubgroupSize:
      check(loom_target_subgroup_size_attr_make(module, requirement.value,
                                                &attribute));
      return attribute;
  }
  std::unreachable();
}

}  // namespace loom::cxx_import
