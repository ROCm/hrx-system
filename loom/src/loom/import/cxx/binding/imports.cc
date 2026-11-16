// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/imports.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>

#include <string_view>
#include <vector>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {

void Imports::declaration(cxx::Symbol* symbol,
                          cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                          cxx::AST* owner) {
  auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(symbol);
  auto* object = cxx::symbol_cast<cxx::ClassSymbol>(symbol);
  cxx::AttributeAST* source = nullptr;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "import") {
          return;
        }
        if (source) {
          diagnostics_.reject(unit_, attribute,
                              "one import binding is allowed per declaration");
        }
        source = attribute;
        if (object) {
          if (object->definition() || object->isUnion() ||
              object->isTemplatePattern() ||
              !cxx::symbol_cast<cxx::NamespaceSymbol>(object->parent())) {
            diagnostics_.reject(
                unit_, attribute,
                "object imports require incomplete namespace-scope classes");
          }
        } else if (!function || function->isTemplatePattern() ||
                   !cxx::symbol_cast<cxx::NamespaceSymbol>(
                       function->parent())) {
          diagnostics_.reject(
              unit_, attribute,
              "import bindings require concrete namespace-scope functions "
              "or incomplete object classes");
        }
        if (cxx::has_internal_linkage(symbol)) {
          diagnostics_.reject(unit_, attribute,
                              "import bindings require external linkage");
        }
        std::vector<std::string> arguments;
        auto* clause = attribute->attributeArgumentClause;
        for (auto* argument :
             cxx::ListView{clause ? clause->expressionList : nullptr}) {
          auto* literal =
              cxx::ast_cast<cxx::StringLiteralExpressionAST>(argument);
          auto spelling =
              literal ? literal->literal->stringValue() : std::string_view{};
          if (!literal || spelling.empty() ||
              spelling.find('\0') != std::string_view::npos) {
            diagnostics_.reject(
                unit_, argument,
                "import names require nonempty NUL-free string literals");
          }
          arguments.emplace_back(spelling);
        }
        if (arguments.empty() || arguments.size() > 2) {
          diagnostics_.reject(
              unit_, attribute,
              "import requires a module string and optional symbol string");
        }
        const std::string module = arguments[0];
        const std::optional<std::string> external_name =
            arguments.size() == 2 ? std::optional<std::string>{arguments[1]}
                                  : std::nullopt;
        if (object) {
          if (!external_name) {
            diagnostics_.reject(
                unit_, attribute,
                "object imports require a namespace and payload name");
          }
          auto [index, inserted] =
              object_indices_.try_emplace(object->canonical(), objects_.size());
          if (inserted) {
            objects_.push_back(
                {object->canonical(), {module, external_name, attribute}});
          } else {
            const auto& previous = objects_[index->second].binding;
            if (previous.module != module || previous.symbol != external_name) {
              diagnostics_.reject(
                  unit_, owner,
                  "conflicting import bindings across object redeclarations");
            }
          }
          return;
        }
        auto [previous, inserted] = imports_.try_emplace(
            function->canonical(),
            ImportBinding{module, external_name, attribute});
        if (!inserted && (previous->second.module != module ||
                          previous->second.symbol != external_name)) {
          diagnostics_.reject(
              unit_, owner,
              "conflicting import bindings across function redeclarations");
        }
      });
}

const ImportBinding* Imports::lookup(cxx::FunctionSymbol* function) const {
  auto found = imports_.find(function->canonical());
  return found == imports_.end() ? nullptr : &found->second;
}

}  // namespace loom::cxx_import
