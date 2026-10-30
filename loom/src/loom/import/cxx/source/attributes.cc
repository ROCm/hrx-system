// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/attributes.h"

#include <string>

#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {

void reject_misplaced_binding_attributes(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    BindingAttributeScope scope) {
  visit_loom_attributes(
      unit, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name == "symbol" && scope != BindingAttributeScope::Declaration) {
          diagnostics.reject(
              unit, attribute,
              "symbol bindings require namespace-scope functions "
              "with a leading attribute");
        }
        if (name == "import" && scope != BindingAttributeScope::Declaration) {
          diagnostics.reject(
              unit, attribute,
              "import bindings require namespace-scope functions "
              "with a leading attribute");
        }
        if (name == "command_program" &&
            scope != BindingAttributeScope::Declaration) {
          diagnostics.reject(
              unit, attribute,
              "command_program requires a namespace-scope function with a "
              "leading attribute");
        }
        if (name == "config" && scope != BindingAttributeScope::Declaration) {
          diagnostics.reject(unit, attribute,
                             "config bindings require namespace-scope scalar "
                             "variables with a leading attribute");
        }
        if (name == "target" && scope != BindingAttributeScope::Declaration &&
            scope != BindingAttributeScope::Statement) {
          diagnostics.reject(unit, attribute,
                             "target bindings require a leading attribute on a "
                             "namespace-scope callable or a pipeline strand");
        }
        if (name == "retain" && scope != BindingAttributeScope::Declaration) {
          diagnostics.reject(
              unit, attribute,
              "retain requires a leading attribute on a namespace-scope "
              "callable");
        }
        if ((name == "template_def" || name == "priority") &&
            scope != BindingAttributeScope::Declaration) {
          diagnostics.reject(
              unit, attribute,
              "template definitions require namespace-scope functions with a "
              "leading attribute");
        }
        if ((name == "assume_aligned" || name == "noalias") &&
            scope != BindingAttributeScope::Parameter &&
            scope != BindingAttributeScope::StagedParameter) {
          diagnostics.reject(
              unit, attribute,
              std::string(name) +
                  " requires a leading pointer parameter attribute");
        }
        if (name == "launch" &&
            scope != BindingAttributeScope::StagedParameter) {
          diagnostics.reject(
              unit, attribute,
              "launch requires a leading command_program or pipeline "
              "parameter attribute");
        }
        if (name == "constexpr" &&
            scope != BindingAttributeScope::LocalDeclaration) {
          diagnostics.reject(
              unit, attribute,
              "constexpr requires a leading automatic const scalar "
              "declaration attribute");
        }
        if (name == "where" && scope != BindingAttributeScope::Declaration &&
            scope != BindingAttributeScope::Function) {
          diagnostics.reject(
              unit, attribute,
              "where requires a trailing function attribute or a leading "
              "config declaration attribute");
        }
        if (name == "fastmath" && scope != BindingAttributeScope::Declaration &&
            scope != BindingAttributeScope::Function &&
            scope != BindingAttributeScope::LocalDeclaration &&
            scope != BindingAttributeScope::Statement) {
          if (scope == BindingAttributeScope::CheckStatement) {
            diagnostics.reject(
                unit, attribute,
                "check statements cannot carry fastmath contracts");
          }
          diagnostics.reject(
              unit, attribute,
              "fastmath requires a function, automatic declaration, or "
              "leading statement attribute");
        }
        if ((name == "serial" || name == "concurrent" || name == "strand") &&
            scope != BindingAttributeScope::Statement) {
          diagnostics.reject(
              unit, attribute,
              "region attributes require a leading compound-statement "
              "attribute");
        }
      });
}

void reject_misplaced_binding_declarator(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::DeclaratorAST* declarator,
    cxx::FunctionDeclaratorChunkAST* parameter_owner,
    BindingAttributeScope parameter_scope) {
  if (!declarator) {
    return;
  }
  for (auto* pointer : cxx::ListView{declarator->ptrOpList}) {
    cxx::visit(
        [&](auto* op) {
          reject_misplaced_binding_attributes(unit, diagnostics,
                                              op->attributeList);
        },
        pointer);
  }
  if (auto* id =
          cxx::ast_cast<cxx::IdDeclaratorAST>(declarator->coreDeclarator)) {
    reject_misplaced_binding_attributes(unit, diagnostics, id->attributeList);
  } else if (auto* nested = cxx::ast_cast<cxx::NestedDeclaratorAST>(
                 declarator->coreDeclarator)) {
    reject_misplaced_binding_declarator(unit, diagnostics, nested->declarator,
                                        parameter_owner, parameter_scope);
  } else if (auto* bitfield = cxx::ast_cast<cxx::BitfieldDeclaratorAST>(
                 declarator->coreDeclarator)) {
    reject_misplaced_binding_attributes(unit, diagnostics,
                                        bitfield->attributeList);
    reject_misplaced_binding_attributes(unit, diagnostics,
                                        bitfield->trailingAttributeList);
  }
  for (auto* chunk : cxx::ListView{declarator->declaratorChunkList}) {
    auto* function = cxx::ast_cast<cxx::FunctionDeclaratorChunkAST>(chunk);
    cxx::visit(
        [&](auto* part) {
          reject_misplaced_binding_attributes(
              unit, diagnostics, part->attributeList,
              function && function == parameter_owner
                  ? BindingAttributeScope::Function
                  : BindingAttributeScope::Local);
        },
        chunk);
    if (!function || !function->parameterDeclarationClause) {
      continue;
    }
    for (auto* parameter : cxx::ListView{
             function->parameterDeclarationClause->parameterDeclarationList}) {
      reject_misplaced_binding_attributes(
          unit, diagnostics, parameter->attributeList,
          function == parameter_owner ? parameter_scope
                                      : BindingAttributeScope::Local);
      reject_misplaced_binding_declarator(unit, diagnostics,
                                          parameter->declarator);
    }
  }
}

void reject_misplaced_binding_statement(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics,
                                        cxx::StatementAST* statement,
                                        BindingAttributeScope scope) {
  if (cxx::ast_cast<cxx::DeclarationStatementAST>(statement)) {
    scope = BindingAttributeScope::LocalDeclaration;
  }
  reject_misplaced_binding_attributes(unit, diagnostics,
                                      statement_attributes(statement), scope);
}

}  // namespace loom::cxx_import
