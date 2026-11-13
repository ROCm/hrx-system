// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/template_definitions.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/attributes.h>
#include <cxx/const_int.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <limits>

#include "loom/import/cxx/binding/template_inputs.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

cxx::FunctionSymbol* parse_family(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics,
                                  cxx::AttributeAST* attribute) {
  auto* clause = attribute->attributeArgumentClause;
  auto* arguments = clause ? clause->expressionList : nullptr;
  auto* id = arguments && !arguments->next
                 ? cxx::ast_cast<cxx::IdExpressionAST>(arguments->value)
                 : nullptr;
  auto* family =
      id ? cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol) : nullptr;
  if (id) {
    if (auto* overloads =
            cxx::symbol_cast<cxx::OverloadSetSymbol>(id->symbol)) {
      auto functions = overloads->functions();
      if (functions.size() == 1) {
        family = functions.front();
      }
    }
  }
  if (!family) {
    diagnostics.reject(
        unit, attribute,
        "template definition requires one unambiguous family function name");
  }
  return family->canonical();
}

int64_t parse_priority(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                       cxx::AttributeAST* attribute) {
  auto* clause = attribute->attributeArgumentClause;
  auto* arguments = clause ? clause->expressionList : nullptr;
  auto* expression = arguments && !arguments->next ? arguments->value : nullptr;
  auto evaluated = expression ? cxx::ASTInterpreter(&unit).evaluate(expression)
                              : std::nullopt;
  auto* value = evaluated ? std::get_if<cxx::ConstInt>(&*evaluated) : nullptr;
  constexpr auto kMinimum = std::numeric_limits<int64_t>::min();
  constexpr auto kMaximum = std::numeric_limits<int64_t>::max();
  bool in_range = value;
  if (value && value->isNegative()) {
    in_range = value->toWide() >= static_cast<cxx::ConstInt::Wide>(kMinimum);
  } else if (value) {
    in_range = value->toUWide() <= static_cast<cxx::ConstInt::UWide>(kMaximum);
  }
  if (!in_range) {
    diagnostics.reject(unit, attribute,
                       "template priority requires one i64 integer constant "
                       "expression");
  }
  return static_cast<int64_t>(value->toIntMax());
}

}  // namespace

TemplateDefinitions::Provider& TemplateDefinitions::get_or_create(
    cxx::FunctionSymbol* function, cxx::AST* source) {
  auto canonical = function->canonical();
  auto [entry, inserted] =
      provider_indices_.try_emplace(canonical, providers_.size());
  if (inserted) {
    providers_.push_back({canonical, nullptr, source});
  }
  return providers_[entry->second];
}

void TemplateDefinitions::declaration(
    cxx::FunctionSymbol* function,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes) {
  bool found_family = false;
  bool found_priority = false;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "template_def" && name != "priority") {
          return;
        }
        if (!function || function->isTemplatePattern() ||
            !cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent())) {
          diagnostics_.reject(
              unit_, attribute,
              "template definitions require concrete namespace-scope "
              "functions");
        }
        auto& provider = get_or_create(function, attribute);
        if (name == "template_def") {
          if (found_family) {
            diagnostics_.reject(
                unit_, attribute,
                "one template definition binding is allowed per declaration");
          }
          found_family = true;
          auto* family = parse_family(unit_, diagnostics_, attribute);
          if (provider.family && provider.family != family) {
            diagnostics_.reject(
                unit_, attribute,
                "conflicting template families across redeclarations");
          }
          provider.family = family;
          provider.family_source = attribute;
        } else {
          if (found_priority) {
            diagnostics_.reject(unit_, attribute,
                                "one template priority is allowed per "
                                "declaration");
          }
          found_priority = true;
          auto priority = parse_priority(unit_, diagnostics_, attribute);
          if (provider.priority && *provider.priority != priority) {
            diagnostics_.reject(
                unit_, attribute,
                "conflicting template priorities across redeclarations");
          }
          provider.priority = priority;
          provider.priority_source = attribute;
        }
      });
}

const TemplateDefinitions::Provider* TemplateDefinitions::lookup(
    cxx::FunctionSymbol* function) const {
  if (!function) {
    return nullptr;
  }
  auto found = provider_indices_.find(function->canonical());
  return found == provider_indices_.end() ? nullptr
                                          : &providers_[found->second];
}

void TemplateDefinitions::validate(const Provider& provider,
                                   cxx::FunctionSymbol* family,
                                   Types& types) const {
  auto* source =
      provider.family_source ? provider.family_source : provider.source;
  if (!provider.family) {
    diagnostics_.reject(unit_, source,
                        "template priority requires a template definition");
  }
  auto* implementation_type =
      cxx::type_cast<cxx::FunctionType>(provider.function->type());
  auto* family_type = cxx::type_cast<cxx::FunctionType>(family->type());
  bool matches = implementation_type && family_type &&
                 !implementation_type->isVariadic() &&
                 !family_type->isVariadic() &&
                 implementation_type->parameterTypes().size() ==
                     family_type->parameterTypes().size() &&
                 types.unqualified(implementation_type->returnType()) ==
                     types.unqualified(family_type->returnType());
  if (matches) {
    for (size_t i = 0; i < implementation_type->parameterTypes().size(); ++i) {
      matches &= template_input_compatible(
          types, family_type->parameterTypes()[i],
          implementation_type->parameterTypes()[i], source);
    }
  }
  if (!matches) {
    diagnostics_.reject(unit_, source,
                        "template definition signature must match its family "
                        "carrier and result types");
  }
  // A provider inherits its family's calling convention. A device annotation
  // is accepted as redundant documentation, but cannot turn a host family into
  // a device provider.
  if (annotated(provider.function, "device") && !annotated(family, "device")) {
    diagnostics_.reject(
        unit_, source,
        "template definition calling convention must match its family");
  }
}

}  // namespace loom::cxx_import
