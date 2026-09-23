// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/parameter_alignment.h"

#include <cxx/symbols.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"

namespace loom::cxx_import {

void ParameterAlignments::declaration(
    cxx::FunctionSymbol* function, cxx::FunctionDeclaratorChunkAST* prototype) {
  if (!prototype || !prototype->parameterDeclarationClause) {
    return;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  size_t ordinal = 0;
  for (auto* parameter : cxx::ListView{
           prototype->parameterDeclarationClause->parameterDeclarationList}) {
    bool found = false;
    visit_loom_attributes(
        unit_, parameter->attributeList,
        [&](std::string_view name, cxx::AttributeAST* attribute) {
          if (name != "assume_aligned") {
            return;
          }
          if (found) {
            diagnostics_.reject(
                unit_, attribute,
                "one assume_aligned contract per parameter is allowed");
          }
          found = true;
          if (!unit_.typeTraits().is_pointer(parameter->type)) {
            diagnostics_.reject(unit_, attribute,
                                "assume_aligned requires a pointer parameter");
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
          auto& parameters = contracts_[function->canonical()];
          if (parameters.empty()) {
            parameters.resize(signature->parameterTypes().size());
          }
          auto& previous = parameters[ordinal];
          if (previous.minimum_alignment &&
              previous.minimum_alignment != *alignment) {
            diagnostics_.reject(
                unit_, attribute,
                "conflicting parameter alignment across declarations");
          }
          previous = {*alignment, attribute};
        });
    ++ordinal;
  }
}

std::span<const ParameterAlignment> ParameterAlignments::get(
    cxx::FunctionSymbol* function) const {
  auto found = contracts_.find(function->canonical());
  return found == contracts_.end()
             ? std::span<const ParameterAlignment>{}
             : std::span<const ParameterAlignment>{found->second};
}

}  // namespace loom::cxx_import
