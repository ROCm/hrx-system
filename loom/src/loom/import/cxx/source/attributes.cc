// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/attributes.h"

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>

#include <algorithm>
#include <initializer_list>

#include "loom/import/cxx/source/diagnostics.h"

namespace loom::cxx_import {

const cxx::Attribute* semantic_attribute(cxx::Symbol* symbol,
                                         std::string_view attribute_namespace,
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

bool has_op_binding(cxx::Symbol* symbol, std::string_view operation,
                    std::string_view selector) {
  const auto* binding = annotation(symbol, "op");
  return binding && binding->arguments.size() == (selector.empty() ? 1u : 2u) &&
         binding->arguments[0]->name() == operation &&
         (selector.empty() || binding->arguments[1]->name() == selector);
}

const cxx::Attribute* template_annotation(cxx::ClassSymbol* source,
                                          std::string_view spelling) {
  if (auto* attribute = annotation(source, spelling)) {
    return attribute;
  }
  auto* primary = cxx::class_template_of(source);
  return primary == source ? nullptr : annotation(primary, spelling);
}

cxx::List<cxx::AttributeSpecifierAST*>* statement_attributes(
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

cxx::AttributeAST* LoomAttributes::next(std::string_view& spelling) {
  for (;;) {
    while (attributes_) {
      auto* attribute = attributes_->value;
      attributes_ = attributes_->next;
      const cxx::Identifier* space = prefix_;
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
        spelling = name->name();
        return attribute;
      }
    }
    if (!specifiers_) {
      return nullptr;
    }
    auto* list = cxx::ast_cast<cxx::CxxAttributeAST>(specifiers_->value);
    specifiers_ = specifiers_->next;
    if (list) {
      attributes_ = list->attributeList;
      prefix_ = list->attributeUsingPrefix
                    ? unit_.identifier(
                          list->attributeUsingPrefix->attributeNamespaceLoc)
                    : nullptr;
    }
  }
}

namespace {
using Scope = BindingAttributeScope;

constexpr unsigned scope_bit(Scope scope) {
  return 1u << static_cast<unsigned>(scope);
}

struct Placement {
  constexpr Placement(std::string_view name,
                      std::initializer_list<Scope> allowed,
                      std::string_view diagnostic)
      : name(name), diagnostic(diagnostic) {
    for (Scope scope : allowed) {
      scopes |= scope_bit(scope);
    }
  }
  // Exact public attribute spelling.
  std::string_view name;
  // Source positions with an admission owner for this attribute.
  unsigned scopes = 0;
  // Source-level explanation for a misplaced spelling.
  std::string_view diagnostic;
};

// The owning binder checks argument values and more specific requirements.
constexpr Placement kPlacements[] = {
    {"access",
     {Scope::Parameter},
     "access requires a leading ordinary parameter attribute"},
    {"assume_aligned",
     {Scope::Parameter, Scope::StagedParameter},
     "assume_aligned requires a leading pointer parameter attribute"},
    {"cold",
     {Scope::Declaration},
     "temperature hint requires a leading function declaration attribute"},
    {"command_program",
     {Scope::Declaration},
     "command_program requires a namespace-scope function with a leading "
     "attribute"},
    {"concurrent",
     {Scope::Statement},
     "region attributes require a leading compound-statement attribute"},
    {"config",
     {Scope::Declaration},
     "config bindings require namespace-scope scalar variables with a leading "
     "attribute"},
    {"constexpr",
     {Scope::LocalDeclaration},
     "constexpr requires a leading automatic const scalar declaration "
     "attribute"},
    {"fastmath",
     {Scope::Declaration, Scope::Function, Scope::LocalDeclaration,
      Scope::Statement},
     "fastmath requires a function, automatic declaration, or leading "
     "statement attribute"},
    {"hot",
     {Scope::Declaration},
     "temperature hint requires a leading function declaration attribute"},
    {"import",
     {Scope::Declaration},
     "import bindings require namespace-scope functions with a leading "
     "attribute"},
    {"inline",
     {Scope::Declaration},
     "inline policy requires a leading function declaration attribute"},
    {"launch",
     {Scope::StagedParameter},
     "launch requires a leading command_program or pipeline parameter "
     "attribute"},
    {"noalias",
     {Scope::Parameter, Scope::StagedParameter},
     "noalias requires a leading pointer parameter attribute"},
    {"noinline",
     {Scope::Declaration},
     "inline policy requires a leading function declaration attribute"},
    {"priority",
     {Scope::Declaration},
     "template definitions require namespace-scope functions with a leading "
     "attribute"},
    {"retain",
     {Scope::Declaration},
     "retain requires a leading attribute on a namespace-scope callable"},
    {"serial",
     {Scope::Statement},
     "region attributes require a leading compound-statement attribute"},
    {"strand",
     {Scope::Statement},
     "region attributes require a leading compound-statement attribute"},
    {"symbol",
     {Scope::Declaration},
     "symbol bindings require namespace-scope functions with a leading "
     "attribute"},
    {"target",
     {Scope::Declaration, Scope::Statement},
     "target bindings require a leading attribute on a namespace-scope "
     "callable or a pipeline strand"},
    {"template_def",
     {Scope::Declaration},
     "template definitions require namespace-scope functions with a leading "
     "attribute"},
    {"where",
     {Scope::Declaration, Scope::Function},
     "where requires a trailing function attribute or a leading config "
     "declaration attribute"},
};
static_assert(std::ranges::is_sorted(kPlacements, {}, &Placement::name));

}  // namespace

void reject_misplaced_binding_attributes(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    BindingAttributeScope scope) {
  visit_loom_attributes(
      unit, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        const auto* rule =
            std::ranges::lower_bound(kPlacements, name, {}, &Placement::name);
        if (rule == std::end(kPlacements) || rule->name != name ||
            (rule->scopes & scope_bit(scope))) {
          return;
        }
        if (scope == BindingAttributeScope::CheckStatement &&
            name == "fastmath") {
          diagnostics.reject(
              unit, attribute,
              "check statements cannot carry fastmath contracts");
        }
        diagnostics.reject(unit, attribute, rule->diagnostic);
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
