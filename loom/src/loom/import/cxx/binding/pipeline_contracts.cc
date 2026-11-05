// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/pipeline_contracts.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <string_view>
#include <utility>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

void PipelineContracts::declaration(
    cxx::FunctionSymbol* function,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    cxx::FunctionDeclaratorChunkAST* prototype, cxx::AST* owner) {
  if (!function) {
    return;
  }
  cxx::AttributeAST* pipeline_source = nullptr;
  bool kernel_scope = false;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "pipeline") {
          return;
        }
        if (pipeline_source) {
          diagnostics_.reject(
              unit_, attribute,
              "one pipeline annotation is allowed per declaration");
        }
        pipeline_source = attribute;
        auto* clause = attribute->attributeArgumentClause;
        if (!clause) {
          return;
        }
        auto* arguments = clause->expressionList;
        auto* literal = arguments && !arguments->next
                            ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(
                                  arguments->value)
                            : nullptr;
        if (!literal || literal->literal->stringValue() != "kernel") {
          diagnostics_.reject(
              unit_, attribute,
              "pipeline accepts no argument clause or the product scope "
              "\"kernel\"");
        }
        kernel_scope = true;
      });

  auto found = contracts_.find(function->canonical());
  if (!pipeline_source && found == contracts_.end() &&
      !annotated(function, "pipeline")) {
    return;
  }

  cxx::AttributeAST* launch_source = nullptr;
  std::optional<size_t> launch_parameter_ordinal;
  std::vector<cxx::ParameterDeclarationAST*> parameter_sources;
  if (prototype && prototype->parameterDeclarationClause) {
    size_t ordinal = 0;
    for (auto* parameter : cxx::ListView{
             prototype->parameterDeclarationClause->parameterDeclarationList}) {
      parameter_sources.push_back(parameter);
      visit_loom_attributes(
          unit_, parameter->attributeList,
          [&](std::string_view name, cxx::AttributeAST* attribute) {
            if (name != "launch") {
              return;
            }
            if (launch_source) {
              diagnostics_.reject(
                  unit_, attribute,
                  "one launch boundary is allowed per pipeline declaration");
            }
            if (attribute->attributeArgumentClause) {
              diagnostics_.reject(unit_, attribute,
                                  "launch takes no argument clause");
            }
            launch_source = attribute;
            launch_parameter_ordinal = ordinal;
          });
      ++ordinal;
    }
  }

  auto& contract = contracts_[function->canonical()];
  if (pipeline_source) {
    if (contract.source && contract.kernel_scope != kernel_scope) {
      diagnostics_.reject(
          unit_, pipeline_source,
          "conflicting artifact scopes across pipeline redeclarations");
    }
    if (!contract.source) {
      contract.source = pipeline_source;
    }
    contract.kernel_scope = kernel_scope;
  }
  if (launch_source) {
    if (contract.launch_parameter_ordinal &&
        contract.launch_parameter_ordinal != launch_parameter_ordinal) {
      diagnostics_.reject(
          unit_, launch_source,
          "conflicting launch boundaries across pipeline redeclarations");
    }
    contract.launch_source = launch_source;
    contract.launch_parameter_ordinal = launch_parameter_ordinal;
  }
  if (!parameter_sources.empty()) {
    contract.parameter_sources = std::move(parameter_sources);
  }
}

void PipelineContracts::finish() {
  for (auto& [function, contract] : contracts_) {
    if (!contract.source) {
      diagnostics_.reject(unit_, contract.launch_source,
                          "pipeline launch boundaries require a pipeline "
                          "function declaration");
    }
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    const bool valid_owner =
        cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent()) &&
        !function->isTemplatePattern() &&
        !function->isImplicitObjectMemberFunction();
    if (!valid_owner || !signature || signature->isVariadic() ||
        !unit_.typeTraits().is_void(signature->returnType())) {
      diagnostics_.reject(
          unit_, contract.source,
          "pipeline requires a concrete non-variadic void namespace-scope "
          "function declaration");
    }
    if (annotated(function, "kernel") ||
        annotated(function, "command_program") ||
        annotated(function, "device") || annotated(function, "op") ||
        annotated(function, "assume") || annotated(function, "check_case") ||
        annotated(function, "check_scenario") ||
        annotated(function, "check_benchmark") ||
        annotated(function, "template_def") || annotated(function, "inline") ||
        annotated(function, "import")) {
      diagnostics_.reject(
          unit_, contract.source,
          "pipeline cannot be combined with another callable kind");
    }
    const size_t parameter_count = function->parameters().size();
    if (contract.launch_parameter_ordinal &&
        *contract.launch_parameter_ordinal >= parameter_count) {
      diagnostics_.reject(unit_, contract.launch_source,
                          "launch boundary must name a source parameter");
    }
  }
}

bool PipelineContracts::contains(cxx::FunctionSymbol* function) const {
  if (!function) {
    return false;
  }
  auto found = contracts_.find(function->canonical());
  return found != contracts_.end() && found->second.source;
}

bool PipelineContracts::has_kernel_scope(cxx::FunctionSymbol* function) const {
  return contracts_.at(function->canonical()).kernel_scope;
}

cxx::AST* PipelineContracts::parameter_source(const Contract& contract,
                                              size_t ordinal,
                                              cxx::AST* fallback) const {
  return ordinal < contract.parameter_sources.size()
             ? static_cast<cxx::AST*>(contract.parameter_sources[ordinal])
             : fallback;
}

const StagedSignatureProjection& PipelineContracts::project(
    cxx::FunctionSymbol* function, Types& types, cxx::AST* owner) {
  auto found = contracts_.find(function->canonical());
  if (found == contracts_.end() || !found->second.source) {
    diagnostics_.reject(unit_, owner,
                        "pipeline signature requires a pipeline contract");
  }
  auto& contract = found->second;
  if (contract.projection) {
    return *contract.projection;
  }

  auto parameters = function->parameters();
  const size_t launch_parameter_ordinal =
      contract.launch_parameter_ordinal.value_or(parameters.size());
  auto projection = std::make_unique<StagedSignatureProjection>();
  projection->launch_parameter_ordinal = launch_parameter_ordinal;
  projection->source_types.reserve(parameters.size());
  projection->parameters.reserve(parameters.size());
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    auto* source = parameter_source(contract, ordinal, owner);
    if (unit_.typeTraits().is_reference(parameter->type()) ||
        unit_.typeTraits().is_volatile(parameter->type())) {
      diagnostics_.reject(
          unit_, source,
          "pipeline parameters require non-reference, non-volatile value "
          "types");
    }
    const auto* source_type = types.unqualified_value(parameter->type());
    const auto& partition = types.partition(source_type, source);
    if (partition.kind == ValueKind::Pointer ||
        partition.kind == ValueKind::Record ||
        partition.kind == ValueKind::Array) {
      diagnostics_.reject(
          unit_, source,
          "pipeline contracts require direct Loom value types instead of "
          "source pointers, records, or arrays");
    }
    const bool requires_binding = types.requires_binding(source_type, source);
    projection->requires_binding |= requires_binding;
    const bool binding = ordinal >= launch_parameter_ordinal;
    size_t parameter_specializations = partition.component_count;
    size_t parameter_bindings = 0;
    if (binding) {
      parameter_specializations =
          requires_binding ? partition.component_count - 1 : 0;
      parameter_bindings =
          partition.component_count - parameter_specializations;
    }
    IREE_ASSERT_EQ(parameter_specializations + parameter_bindings,
                   partition.component_count);
    projection->append(source_type, partition.component_count,
                       parameter_specializations);
  }
  contract.projection = std::move(projection);
  return *contract.projection;
}

BoundStagedSignature PipelineContracts::bind(
    cxx::FunctionSymbol* function, Types& types, cxx::AST* owner,
    loom_builder_t* builder, SignatureIdentityRequirement identities) {
  return bind_staged_signature(project(function, types, owner), types, owner,
                               builder, identities);
}

}  // namespace loom::cxx_import
