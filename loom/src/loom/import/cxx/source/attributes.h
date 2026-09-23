// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_ATTRIBUTES_H_
#define LOOM_IMPORT_CXX_SOURCE_ATTRIBUTES_H_

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <string_view>

namespace loom::cxx_import {

class Diagnostics;

// Source positions with distinct binding owners. Declaration bindings own
// namespace-scope symbols/configs; parameter bindings own entry preconditions.
enum class BindingAttributeScope { Declaration, Parameter, Local };

// Rejects binding annotations outside their owning source positions.
void reject_misplaced_binding_attributes(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    BindingAttributeScope scope = BindingAttributeScope::Local);

// Checks declarator/type positions once. Only the selected function prototype
// admits parameter bindings; nested function types have no binding owner.
void reject_misplaced_binding_declarator(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::DeclaratorAST* declarator,
    cxx::FunctionDeclaratorChunkAST* parameter_owner = nullptr);

// Checks one statement's leading attributes during its owner's body walk.
void reject_misplaced_binding_statement(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics,
                                        cxx::StatementAST* statement);

// Queries a semantic Loom annotation attached to a resolved source symbol.
inline bool annotated(cxx::Symbol* symbol, std::string_view spelling) {
  if (!symbol) {
    return false;
  }
  const auto* attributes = symbol->canonical()->attributes();
  if (!attributes) {
    return false;
  }
  for (const auto& attribute : *attributes) {
    if (attribute.attributeNamespace && attribute.name &&
        attribute.attributeNamespace->name() == "loom" &&
        attribute.name->name() == spelling) {
      return true;
    }
  }
  return false;
}

// Visits raw C++ Loom attributes, including duplicates and numeric arguments
// that the frontend's semantic string-attribute map cannot preserve. Admission
// owners retain their checked interpretation for subsequent source lowering.
template <typename Visitor>
void visit_loom_attributes(cxx::TranslationUnit& unit,
                           cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                           Visitor visitor) {
  for (auto* specifier : cxx::ListView{attributes}) {
    auto* list = cxx::ast_cast<cxx::CxxAttributeAST>(specifier);
    if (!list) {
      continue;
    }
    auto* prefix = list->attributeUsingPrefix;
    for (auto* attribute : cxx::ListView{list->attributeList}) {
      const cxx::Identifier* space =
          prefix ? unit.identifier(prefix->attributeNamespaceLoc) : nullptr;
      const cxx::Identifier* name = nullptr;
      if (auto* scoped = cxx::ast_cast<cxx::ScopedAttributeTokenAST>(
              attribute->attributeToken)) {
        space = scoped->attributeNamespace;
        name = scoped->identifier;
      } else if (auto* simple = cxx::ast_cast<cxx::SimpleAttributeTokenAST>(
                     attribute->attributeToken)) {
        name = simple->identifier;
      }
      if (space && name && space->name() == "loom") {
        visitor(name->name(), attribute);
      }
    }
  }
}

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_ATTRIBUTES_H_
