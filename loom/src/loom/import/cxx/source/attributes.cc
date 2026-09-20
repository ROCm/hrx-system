// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/attributes.h"

#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {

void reject_global_binding_attributes(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes) {
  visit_loom_attributes(
      unit, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name == "symbol") {
          diagnostics.reject(
              unit, attribute,
              "symbol bindings require namespace-scope functions "
              "with a leading attribute");
        }
        if (name == "config") {
          diagnostics.reject(unit, attribute,
                             "config bindings require namespace-scope scalar "
                             "variables with a leading attribute");
        }
      });
}

void reject_global_binding_declarator(cxx::TranslationUnit& unit,
                                      Diagnostics& diagnostics,
                                      cxx::DeclaratorAST* declarator) {
  if (!declarator) {
    return;
  }
  for (auto* pointer : cxx::ListView{declarator->ptrOpList}) {
    cxx::visit(
        [&](auto* op) {
          reject_global_binding_attributes(unit, diagnostics,
                                           op->attributeList);
        },
        pointer);
  }
  if (auto* id =
          cxx::ast_cast<cxx::IdDeclaratorAST>(declarator->coreDeclarator)) {
    reject_global_binding_attributes(unit, diagnostics, id->attributeList);
  } else if (auto* nested = cxx::ast_cast<cxx::NestedDeclaratorAST>(
                 declarator->coreDeclarator)) {
    reject_global_binding_declarator(unit, diagnostics, nested->declarator);
  } else if (auto* bitfield = cxx::ast_cast<cxx::BitfieldDeclaratorAST>(
                 declarator->coreDeclarator)) {
    reject_global_binding_attributes(unit, diagnostics,
                                     bitfield->attributeList);
    reject_global_binding_attributes(unit, diagnostics,
                                     bitfield->trailingAttributeList);
  }
  for (auto* chunk : cxx::ListView{declarator->declaratorChunkList}) {
    cxx::visit(
        [&](auto* part) {
          reject_global_binding_attributes(unit, diagnostics,
                                           part->attributeList);
        },
        chunk);
    auto* function = cxx::ast_cast<cxx::FunctionDeclaratorChunkAST>(chunk);
    if (!function || !function->parameterDeclarationClause) {
      continue;
    }
    for (auto* parameter : cxx::ListView{
             function->parameterDeclarationClause->parameterDeclarationList}) {
      reject_global_binding_attributes(unit, diagnostics,
                                       parameter->attributeList);
      reject_global_binding_declarator(unit, diagnostics,
                                       parameter->declarator);
    }
  }
}

}  // namespace loom::cxx_import
