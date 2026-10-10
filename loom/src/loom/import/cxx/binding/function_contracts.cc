// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/function_contracts.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <utility>

#include "loom/import/cxx/binding/predicates.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
bool FunctionContracts::predicate_equal(const Predicate& left,
                                        const Predicate& right) {
  if (left.value_count != right.value_count ||
      left.predicate.kind != right.predicate.kind ||
      left.predicate.arg_count != right.predicate.arg_count) {
    return false;
  }
  for (size_t index = 0; index < left.predicate.arg_count; ++index) {
    if (left.predicate.arg_tags[index] != right.predicate.arg_tags[index] ||
        left.predicate.args[index] != right.predicate.args[index]) {
      return false;
    }
  }
  for (size_t index = 0; index < left.value_count; ++index) {
    const auto& left_value = left.values[index];
    const auto& right_value = right.values[index];
    if (left_value.root != right_value.root ||
        left_value.parameter_ordinal != right_value.parameter_ordinal ||
        left_value.members != right_value.members ||
        left_value.root_type != right_value.root_type ||
        left_value.source_type != right_value.source_type ||
        left_value.converted_type != right_value.converted_type) {
      return false;
    }
  }
  return true;
}

bool FunctionContracts::contract_equal(const Contract& left,
                                       const Contract& right) {
  if (left.predicates.size() != right.predicates.size() ||
      left.requirements.size() != right.requirements.size()) {
    return false;
  }
  std::vector<bool> matched_predicates(right.predicates.size());
  for (const auto& predicate : left.predicates) {
    auto found = right.predicates.end();
    for (auto candidate = right.predicates.begin();
         candidate != right.predicates.end(); ++candidate) {
      const size_t ordinal = candidate - right.predicates.begin();
      if (!matched_predicates[ordinal] &&
          predicate_equal(predicate, *candidate)) {
        found = candidate;
        matched_predicates[ordinal] = true;
        break;
      }
    }
    if (found == right.predicates.end()) {
      return false;
    }
  }
  std::vector<bool> matched_requirements(right.requirements.size());
  for (const auto& requirement : left.requirements) {
    auto found = right.requirements.end();
    for (auto candidate = right.requirements.begin();
         candidate != right.requirements.end(); ++candidate) {
      const size_t ordinal = candidate - right.requirements.begin();
      if (!matched_requirements[ordinal] &&
          requirement.kind == candidate->kind &&
          requirement.value == candidate->value) {
        found = candidate;
        matched_requirements[ordinal] = true;
        break;
      }
    }
    if (found == right.requirements.end()) {
      return false;
    }
  }
  return true;
}

void FunctionContracts::declaration(
    cxx::FunctionSymbol* function, cxx::FunctionDeclaratorChunkAST* prototype) {
  if (!prototype) {
    return;
  }
  cxx::AttributeAST* source = nullptr;
  cxx::ExpressionAST* expression = nullptr;
  visit_loom_attributes(
      unit_, prototype->attributeList,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "where") {
          return;
        }
        if (source) {
          diagnostics_.reject(unit_, attribute,
                              "one where contract per declaration is allowed");
        }
        source = attribute;
        auto* clause = attribute->attributeArgumentClause;
        auto* arguments = clause ? clause->expressionList : nullptr;
        if (!arguments || arguments->next) {
          diagnostics_.reject(unit_, attribute,
                              "where requires one predicate expression");
        }
        expression = arguments->value;
      });
  if (!source) {
    return;
  }

  auto parameters = function->parameters();
  Contract contract;
  auto collect = [&](auto&& self, cxx::ExpressionAST* condition) -> void {
    auto* binary =
        cxx::ast_cast<cxx::BinaryExpressionAST>(unwrap_expression(condition));
    if (binary && !binary->symbol && binary->op == cxx::TokenKind::T_AMP_AMP) {
      self(self, binary->leftExpression);
      self(self, binary->rightExpression);
      return;
    }
    if (auto requirement =
            project_target_requirement(unit_, diagnostics_, condition)) {
      contract.requirements.push_back(*requirement);
      return;
    }
    for (auto projected : project_predicates(unit_, diagnostics_, condition)) {
      Predicate predicate = {};
      predicate.value_count = projected.value_count;
      predicate.predicate = projected.predicate;
      for (size_t index = 0; index < projected.value_count; ++index) {
        auto& projected_value = projected.values[index];
        auto& value = predicate.values[index];
        value.members = std::move(projected_value.members);
        value.root_type = projected_value.root_type;
        value.source_type = projected_value.source->type;
        value.converted_type = projected_value.converted->type;
        value.source = projected_value.source;
        switch (projected_value.origin) {
          case PredicateValueOrigin::Binding: {
            auto parameter = std::find(parameters.begin(), parameters.end(),
                                       projected_value.binding);
            if (parameter == parameters.end()) {
              diagnostics_.reject(
                  unit_, projected_value.source,
                  "function predicates may reference only parameters and the "
                  "typed result placeholder");
            }
            value.root = RootKind::Parameter;
            value.parameter_ordinal = parameter - parameters.begin();
            break;
          }
          case PredicateValueOrigin::Result:
            value.root = RootKind::Result;
            value.parameter_ordinal = std::numeric_limits<size_t>::max();
            break;
          case PredicateValueOrigin::Subject:
            diagnostics_.reject(
                unit_, projected_value.source,
                "implicit predicate subjects require a config declaration");
        }
      }
      contract.predicates.push_back(std::move(predicate));
    }
  };
  collect(collect, expression);

  auto previous = contracts_.find(function->canonical());
  if (previous == contracts_.end()) {
    contracts_.emplace(function->canonical(), std::move(contract));
    return;
  }
  if (!contract_equal(previous->second, contract)) {
    diagnostics_.reject(unit_, source,
                        "conflicting where contracts across redeclarations");
  }
}

bool FunctionContracts::has_predicates(cxx::FunctionSymbol* function) const {
  auto found = contracts_.find(function->canonical());
  return found != contracts_.end() && !found->second.predicates.empty();
}

bool FunctionContracts::has_requirements(cxx::FunctionSymbol* function) const {
  auto found = contracts_.find(function->canonical());
  return found != contracts_.end() && !found->second.requirements.empty();
}

std::vector<loom_predicate_t> FunctionContracts::bind(
    cxx::FunctionSymbol* function, Types& types,
    std::span<const loom_value_id_t> identities,
    FunctionContractSignature signature, cxx::AST* owner) const {
  auto found = contracts_.find(function->canonical());
  if (found == contracts_.end() || found->second.predicates.empty()) {
    return {};
  }

  const auto parameters = function->parameters();
  auto* function_type = cxx::type_cast<cxx::FunctionType>(function->type());
  std::vector<size_t> parameter_offsets(parameters.size());
  size_t component_count = 0;
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    parameter_offsets[ordinal] = component_count;
    component_count +=
        signature == FunctionContractSignature::Kernel
            ? 1
            : types
                  .partition(types.unqualified(parameters[ordinal]->type()),
                             owner)
                  .component_count;
  }
  const size_t result_offset = component_count;
  const bool returns_void =
      function_type->returnType()->kind() == cxx::TypeKind::kVoid;
  if (!returns_void && signature == FunctionContractSignature::Flattened) {
    component_count +=
        types.partition(types.unqualified(function_type->returnType()), owner)
            .component_count;
  }
  if (identities.size() != component_count) {
    diagnostics_.reject(unit_, owner,
                        "callable predicate identities do not match the "
                        "projected source signature");
  }

  auto resolve = [&](const Value& value) -> loom_value_id_t {
    const cxx::Type* root_type = nullptr;
    size_t component_offset = 0;
    if (value.root == RootKind::Parameter) {
      root_type =
          types.unqualified(parameters[value.parameter_ordinal]->type());
      component_offset = parameter_offsets[value.parameter_ordinal];
      if (types.unqualified(value.root_type) != root_type) {
        diagnostics_.reject(unit_, value.source,
                            "predicate parameter type changed across source "
                            "redeclarations");
      }
    } else {
      if (returns_void || signature == FunctionContractSignature::Kernel) {
        diagnostics_.reject(unit_, value.source,
                            "result predicates require a non-void ordinary "
                            "callable");
      }
      root_type = types.unqualified(function_type->returnType());
      component_offset = result_offset;
      if (types.unqualified(value.root_type) != root_type) {
        diagnostics_.reject(
            unit_, value.source,
            "predicate result placeholder must use the callable result type");
      }
    }

    const Partition* partition = &types.partition(root_type, owner);
    for (auto* field : value.members) {
      if (partition->kind != ValueKind::Record ||
          static_cast<const RecordPartition*>(partition)->source !=
              field->parent()) {
        diagnostics_.reject(unit_, value.source,
                            "predicate member path does not match its source "
                            "record type");
      }
      const auto& member = types.member(field, value.source);
      component_offset += member.component_offset;
      partition = member.partition;
    }
    if (partition->kind != ValueKind::SSA || partition->component_count != 1) {
      diagnostics_.reject(unit_, value.source,
                          "predicate values must select one scalar source "
                          "component");
    }
    if (types.unqualified(value.source_type) !=
        types.unqualified(value.converted_type)) {
      diagnostics_.reject(
          unit_, value.source,
          "function predicates cannot retain a value-changing implicit "
          "conversion; use a type-matched predicate helper");
    }
    return identities[component_offset];
  };

  std::vector<loom_predicate_t> predicates;
  predicates.reserve(found->second.predicates.size());
  for (const auto& admitted : found->second.predicates) {
    auto predicate = admitted.predicate;
    for (size_t index = 0; index < predicate.arg_count; ++index) {
      if (predicate.arg_tags[index] == LOOM_PRED_ARG_VALUE) {
        const auto ordinal = static_cast<size_t>(predicate.args[index]);
        predicate.args[index] = resolve(admitted.values[ordinal]);
      }
    }
    predicates.push_back(predicate);
  }
  return predicates;
}

std::vector<loom_attribute_t> FunctionContracts::bind_requirements(
    cxx::FunctionSymbol* function, loom_module_t* module) const {
  auto found = contracts_.find(function->canonical());
  if (found == contracts_.end() || found->second.requirements.empty()) {
    return {};
  }
  std::vector<loom_attribute_t> requirements;
  requirements.reserve(found->second.requirements.size());
  for (const auto& requirement : found->second.requirements) {
    requirements.push_back(materialize_target_requirement(module, requirement));
  }
  return requirements;
}

}  // namespace loom::cxx_import
