// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/parameter_contracts.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <string>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/buffer/ops.h"

namespace loom::cxx_import {

loom_value_id_t apply_parameter_contract(const ParameterContract& contract,
                                         loom_value_id_t buffer,
                                         Locations& locations,
                                         loom_builder_t* builder) {
  const auto buffer_type = loom_type_buffer();
  if (contract.alignment.source) {
    loom_op_t* assumption;
    check(loom_buffer_assume_alignment_build(
        builder, &buffer, 1, contract.alignment.minimum_alignment, &buffer_type,
        1, locations.get(contract.alignment.source), &assumption));
    buffer = loom_op_results(assumption)[0];
  }
  if (contract.noalias_source) {
    loom_op_t* assumption;
    check(loom_buffer_assume_noalias_build(
        builder, &buffer, 1, &buffer_type, 1,
        locations.get(contract.noalias_source), &assumption));
    buffer = loom_op_results(assumption)[0];
  }
  return buffer;
}

void ParameterContracts::declaration(
    cxx::FunctionSymbol* function, cxx::FunctionDeclaratorChunkAST* prototype) {
  if (!prototype || !prototype->parameterDeclarationClause) {
    return;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  size_t ordinal = 0;
  for (auto* parameter : cxx::ListView{
           prototype->parameterDeclarationClause->parameterDeclarationList}) {
    ParameterContract declared;
    visit_loom_attributes(
        unit_, parameter->attributeList,
        [&](std::string_view name, cxx::AttributeAST* attribute) {
          if (name == "access") {
            if (declared.access_source) {
              diagnostics_.reject(unit_, attribute,
                                  "one access bound per parameter is allowed");
            }
            auto* clause = attribute->attributeArgumentClause;
            auto* arguments = clause ? clause->expressionList : nullptr;
            auto* literal =
                arguments && !arguments->next
                    ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(
                          arguments->value)
                    : nullptr;
            const auto spelling =
                literal ? literal->literal->stringValue() : std::string_view{};
            declared.access =
                spelling == "read"         ? LOOM_ARGUMENT_ACCESS_READ
                : spelling == "write"      ? LOOM_ARGUMENT_ACCESS_WRITE
                : spelling == "read_write" ? LOOM_ARGUMENT_ACCESS_READ_WRITE
                : spelling == "none"       ? LOOM_ARGUMENT_ACCESS_NONE
                                           : LOOM_ARGUMENT_ACCESS_UNSPECIFIED;
            if (declared.access == LOOM_ARGUMENT_ACCESS_UNSPECIFIED) {
              diagnostics_.reject(unit_, attribute,
                                  "access requires read, write, read_write, or "
                                  "none as one string literal");
            }
            declared.access_source = attribute;
            return;
          }
          if (name != "assume_aligned" && name != "noalias") {
            return;
          }
          if (!unit_.typeTraits().is_pointer(parameter->type)) {
            diagnostics_.reject(
                unit_, attribute,
                std::string(name) + " requires a pointer parameter");
          }
          if (name == "noalias") {
            if (declared.noalias_source) {
              diagnostics_.reject(
                  unit_, attribute,
                  "one noalias contract per parameter is allowed");
            }
            if (attribute->attributeArgumentClause) {
              diagnostics_.reject(unit_, attribute,
                                  "noalias takes no argument clause");
            }
            declared.noalias_source = attribute;
            return;
          }
          if (declared.alignment.source) {
            diagnostics_.reject(
                unit_, attribute,
                "one assume_aligned contract per parameter is allowed");
          }
          auto* clause = attribute->attributeArgumentClause;
          auto* arguments = clause ? clause->expressionList : nullptr;
          auto alignment = arguments && !arguments->next
                               ? integer_constant(unit_, arguments->value)
                               : std::nullopt;
          if (!alignment || *alignment <= 0 ||
              (*alignment & (*alignment - 1)) != 0) {
            diagnostics_.reject(
                unit_, attribute,
                "assume_aligned requires one positive power-of-two integer "
                "constant without calls or mutation");
          }
          declared.alignment = {*alignment, attribute};
        });
    if (declared.alignment.source || declared.noalias_source ||
        declared.access_source) {
      auto& parameters = contracts_[function->canonical()];
      if (parameters.empty()) {
        parameters.resize(signature->parameterTypes().size());
      }
      auto& previous = parameters[ordinal];
      if (declared.access_source) {
        if (previous.access_source && previous.access != declared.access) {
          diagnostics_.reject(
              unit_, declared.access_source,
              "conflicting parameter access across declarations");
        }
        previous.access = declared.access;
        previous.access_source = declared.access_source;
      }
      if (declared.alignment.source) {
        if (previous.alignment.minimum_alignment &&
            previous.alignment.minimum_alignment !=
                declared.alignment.minimum_alignment) {
          diagnostics_.reject(
              unit_, declared.alignment.source,
              "conflicting parameter alignment across declarations");
        }
        previous.alignment = declared.alignment;
      }
      if (declared.noalias_source) {
        previous.noalias_source = declared.noalias_source;
      }
    }
    ++ordinal;
  }
}

std::span<const ParameterContract> ParameterContracts::get(
    cxx::FunctionSymbol* function) const {
  auto found = contracts_.find(function->canonical());
  return found == contracts_.end()
             ? std::span<const ParameterContract>{}
             : std::span<const ParameterContract>{found->second};
}

}  // namespace loom::cxx_import
