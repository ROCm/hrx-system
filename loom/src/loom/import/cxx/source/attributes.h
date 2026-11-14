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

// Returns a semantic attribute attached to a resolved source symbol, or null
// when the canonical declaration has no attribute with this spelling.
inline const cxx::Attribute* semantic_attribute(
    cxx::Symbol* symbol, std::string_view attribute_namespace,
    std::string_view spelling) {
  if (!symbol) {
    return nullptr;
  }
  const auto* attributes = symbol->canonical()->attributes();
  if (!attributes) {
    return nullptr;
  }
  for (const auto& attribute : *attributes) {
    if (attribute.attributeNamespace && attribute.name &&
        attribute.attributeNamespace->name() == attribute_namespace &&
        attribute.name->name() == spelling) {
      return &attribute;
    }
  }
  return nullptr;
}

// Returns a semantic Loom annotation attached to a resolved source symbol, or
// null when the canonical declaration has no annotation with this spelling.
inline const cxx::Attribute* annotation(cxx::Symbol* symbol,
                                        std::string_view spelling) {
  return semantic_attribute(symbol, "loom", spelling);
}

// Source positions with distinct binding owners. Namespace declarations,
// function prototypes, parameters and automatic declarations each admit their
// own leading attributes; all other source positions are local and unowned.
enum class BindingAttributeScope {
  Declaration,
  Function,
  Parameter,
  StagedParameter,
  LocalDeclaration,
  Statement,
  CheckStatement,
  Local,
};

// Returns the raw leading attribute list carried by one statement or automatic
// declaration. C++ represents a declaration statement's leading attributes on
// its nested simple declaration.
inline cxx::List<cxx::AttributeSpecifierAST*>* statement_attributes(
    cxx::StatementAST* statement) {
  if (auto* declaration =
          cxx::ast_cast<cxx::DeclarationStatementAST>(statement)) {
    if (auto* simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(
            declaration->declaration)) {
      return simple->attributeList;
    }
  }
  cxx::List<cxx::AttributeSpecifierAST*>* attributes = nullptr;
  cxx::visit(
      [&](auto* node) {
        if constexpr (requires { node->attributeList; }) {
          attributes = node->attributeList;
        }
      },
      statement);
  return attributes;
}

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
    cxx::FunctionDeclaratorChunkAST* parameter_owner = nullptr,
    BindingAttributeScope parameter_scope = BindingAttributeScope::Parameter);

// Checks one statement's leading attributes during its owner's body walk.
// Scope distinguishes ordinary translated statements from check records,
// whose bodies admit a different binding vocabulary.
void reject_misplaced_binding_statement(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics,
                                        cxx::StatementAST* statement,
                                        BindingAttributeScope scope);

// Queries a semantic Loom annotation attached to a resolved source symbol.
inline bool annotated(cxx::Symbol* symbol, std::string_view spelling) {
  return annotation(symbol, spelling) != nullptr;
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
