// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/functions.h"

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <cctype>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"

namespace loom::cxx_import {

void Functions::select(std::span<const iree_string_view_t> roots) {
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(unit_.ast());
  if (!root) {
    diagnostics_.reject(unit_, unit_.ast(),
                        "expected an ordinary translation unit");
  }
  std::vector<cxx::FunctionSymbol*> definitions;
  collect(root->declarationList, DeclarationScope::Namespace, definitions);
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  configs_.build(&builder);
  for (const auto& [function, source] : check_cases_) {
    if (!definition(function)) {
      diagnostics_.reject(unit_, source, "check cases require a definition");
    }
  }
  for (const auto& benchmark : benchmarks_) {
    if (!is_check_case(benchmark.case_function)) {
      diagnostics_.reject(unit_, benchmark.source,
                          "check benchmark must reference a check case");
    }
  }
  if (!roots.empty()) {
    std::unordered_map<std::string, std::vector<cxx::FunctionSymbol*>>
        candidates;
    for (auto* symbol : definitions) {
      candidates[qualified_name(symbol)].push_back(symbol);
    }
    for (size_t i = 0; i < roots.size(); ++i) {
      auto spelling = cxx_import::string(roots[i]);
      auto found = candidates.find(spelling);
      if (found == candidates.end()) {
        diagnostics_.reject(unit_, root,
                            "root has no concrete definition: " + spelling);
      }
      if (found->second.size() != 1) {
        diagnostics_.reject(unit_, found->second.front()->declaration(),
                            "ambiguous root: " + spelling);
      }
      auto* selected = found->second.front();
      exported_.insert(selected);
      declare(selected);
    }
  } else {
    for (auto* symbol : definitions) {
      bool visible = !symbol->isStatic();
      auto* visibility =
          cxx::attributeArgument(symbol->attributes(), "visibility");
      if (visibility && visibility->name() == "hidden") {
        visible = false;
      }
      for (auto* owner : symbol->enclosingSymbols()) {
        if (auto* space = cxx::symbol_cast<cxx::NamespaceSymbol>(owner)) {
          if (space->parent() && !space->name()) {
            visible = false;
          }
        }
      }
      if (visible || is_check_case(symbol)) {
        exported_.insert(symbol);
        declare(symbol);
      }
    }
  }
  // A case can only be selected as a root, never reached through a call.
  // Reserve its public benchmark names before discovering private helpers.
  for (const auto& benchmark : benchmarks_) {
    if (callees_.contains(benchmark.case_function->canonical())) {
      exported_.insert(benchmark.function);
      create_symbol(benchmark.function, benchmark.source);
    }
  }
}

void Functions::collect(cxx::List<cxx::DeclarationAST*>* declarations,
                        DeclarationScope scope,
                        std::vector<cxx::FunctionSymbol*>& definitions) {
  for (auto* declaration : cxx::ListView{declarations}) {
    collect(declaration, scope, definitions);
  }
}

void Functions::collect(cxx::DeclarationAST* declaration,
                        DeclarationScope scope,
                        std::vector<cxx::FunctionSymbol*>& definitions) {
  if (auto* function = cxx::ast_cast<cxx::FunctionDefinitionAST>(declaration)) {
    admit_declaration(function->symbol, function->attributeList, function,
                      scope);
    if (scope == DeclarationScope::Namespace &&
        !function->symbol->isTemplatePattern()) {
      intrinsics_.declaration(function->symbol, function->attributeList,
                              function);
      launches_.declaration(function->symbol, function->attributeList);
      definitions_.emplace(function->symbol->canonical(), function->symbol);
      definitions.push_back(function->symbol);
    }
  } else if (auto* space =
                 cxx::ast_cast<cxx::NamespaceDefinitionAST>(declaration)) {
    admit_declaration(nullptr, space->attributeList, space, scope);
    admit_declaration(nullptr, space->extraAttributeList, space, scope);
    collect(space->declarationList, scope, definitions);
  } else if (auto* linkage =
                 cxx::ast_cast<cxx::LinkageSpecificationAST>(declaration)) {
    collect(linkage->declarationList, scope, definitions);
  } else if (auto* pattern =
                 cxx::ast_cast<cxx::TemplateDeclarationAST>(declaration)) {
    collect(pattern->declaration,
            pattern->templateParameterList ? DeclarationScope::Nested : scope,
            definitions);
  } else if (auto* alias =
                 cxx::ast_cast<cxx::AliasDeclarationAST>(declaration)) {
    admit_declaration(nullptr, alias->attributeList, alias, scope);
    reject_global_binding_attributes(unit_, diagnostics_,
                                     alias->typeId->attributeList);
    reject_global_binding_declarator(unit_, diagnostics_,
                                     alias->typeId->declarator);
  } else if (auto* attribute =
                 cxx::ast_cast<cxx::AttributeDeclarationAST>(declaration)) {
    admit_declaration(nullptr, attribute->attributeList, attribute, scope);
  } else if (auto* enumeration =
                 cxx::ast_cast<cxx::OpaqueEnumDeclarationAST>(declaration)) {
    admit_declaration(nullptr, enumeration->attributeList, enumeration, scope);
  } else if (auto* simple =
                 cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration)) {
    if (!simple->initDeclaratorList) {
      admit_declaration(nullptr, simple->attributeList, simple, scope);
    }
    for (auto* specifier : cxx::ListView{simple->declSpecifierList}) {
      if (auto* record = cxx::ast_cast<cxx::ClassSpecifierAST>(specifier)) {
        admit_declaration(nullptr, record->attributeList, record,
                          DeclarationScope::Nested);
        collect(record->declarationList, DeclarationScope::Nested, definitions);
      } else if (auto* enumeration =
                     cxx::ast_cast<cxx::EnumSpecifierAST>(specifier)) {
        admit_declaration(nullptr, enumeration->attributeList, enumeration,
                          scope);
        for (auto* enumerator : cxx::ListView{enumeration->enumeratorList}) {
          admit_declaration(nullptr, enumerator->attributeList, enumerator,
                            scope);
        }
      }
    }
    for (auto* declarator : cxx::ListView{simple->initDeclaratorList}) {
      auto* function =
          cxx::symbol_cast<cxx::FunctionSymbol>(declarator->symbol);
      bool is_config = admit_declaration(
          declarator->symbol, simple->attributeList, declarator, scope);
      if (scope != DeclarationScope::Namespace) {
        continue;
      }
      if (function) {
        intrinsics_.declaration(function, simple->attributeList, declarator);
        launches_.declaration(function, simple->attributeList);
      }
      auto* variable =
          cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
      if (variable && !is_config && !variable->isExtern() &&
          !(variable->isConstexpr() ||
            (unit_.typeTraits().is_const(variable->type()) &&
             variable->constValue()))) {
        diagnostics_.reject(
            unit_, declarator,
            "global storage definitions require a global-storage projection");
      }
    }
  }
}

bool Functions::admit_declaration(
    cxx::Symbol* symbol, cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    cxx::AST* owner, DeclarationScope scope) {
  if (auto* declarator = cxx::ast_cast<cxx::InitDeclaratorAST>(owner)) {
    reject_global_binding_declarator(unit_, diagnostics_,
                                     declarator->declarator);
  } else if (auto* function =
                 cxx::ast_cast<cxx::FunctionDefinitionAST>(owner)) {
    reject_global_binding_declarator(unit_, diagnostics_, function->declarator);
  }
  bool is_config = configs_.declaration(symbol, attributes, owner);
  admit_symbol(symbol, attributes);
  auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(symbol);
  auto require_namespace_function = [&] {
    if (!function || scope != DeclarationScope::Namespace ||
        function->isTemplatePattern() ||
        !cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent())) {
      diagnostics_.reject(
          unit_, owner,
          "check declarations require non-template namespace-scope functions");
    }
  };
  bool found = false;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "check_case" && name != "check_benchmark") {
          return;
        }
        require_namespace_function();
        auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
        if (found || annotated(function, "kernel") ||
            annotated(function, "device") || annotated(function, "op") ||
            signature->isVariadic() || !signature->parameterTypes().empty() ||
            signature->returnType()->kind() != cxx::TypeKind::kVoid) {
          diagnostics_.reject(
              unit_, owner,
              "check declarations require one check annotation on void()");
        }
        found = true;
        auto* clause = attribute->attributeArgumentClause;
        if (name == "check_case") {
          if (clause) {
            diagnostics_.reject(unit_, attribute,
                                "check_case takes no arguments");
          }
          check_cases_.try_emplace(function->canonical(), owner);
          return;
        }
        if (function->declaration()) {
          diagnostics_.reject(unit_, owner,
                              "check benchmarks are declarations referencing a "
                              "case, without a body");
        }
        auto* arguments = clause ? clause->expressionList : nullptr;
        auto* id = arguments && !arguments->next
                       ? cxx::ast_cast<cxx::IdExpressionAST>(arguments->value)
                       : nullptr;
        auto* target =
            id ? cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol) : nullptr;
        if (id) {
          if (auto* overloads =
                  cxx::symbol_cast<cxx::OverloadSetSymbol>(id->symbol)) {
            auto functions = overloads->functions();
            if (functions.size() == 1) {
              target = functions.front();
            }
          }
        }
        if (!target) {
          diagnostics_.reject(unit_, attribute,
                              "check_benchmark requires one unambiguous check "
                              "case function name");
        }
        for (const auto& previous : benchmarks_) {
          if (previous.function->canonical() == function->canonical()) {
            diagnostics_.reject(unit_, owner,
                                "check benchmark is already declared");
          }
        }
        benchmarks_.push_back({function, target, attribute});
      });
  if (!found &&
      ((annotated(function, "check_case") && !is_check_case(function)) ||
       annotated(function, "check_benchmark"))) {
    require_namespace_function();
    diagnostics_.reject(unit_, owner,
                        "check annotations must precede the declaration");
  }
  return is_config;
}

void Functions::admit_symbol(
    cxx::Symbol* symbol, cxx::List<cxx::AttributeSpecifierAST*>* attributes) {
  bool found = false;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "symbol") {
          return;
        }
        if (found) {
          diagnostics_.reject(unit_, attribute,
                              "one symbol binding is allowed per declaration");
        }
        found = true;
        auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(symbol);
        if (!function || function->isTemplatePattern() ||
            !cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent())) {
          diagnostics_.reject(
              unit_, attribute,
              "symbol bindings require concrete namespace-scope functions");
        }
        if (annotated(function, "op") || annotated(function, "assume")) {
          diagnostics_.reject(unit_, attribute,
                              "intrinsic bindings do not define Loom symbols");
        }
        auto* clause = attribute->attributeArgumentClause;
        auto* arguments = clause ? clause->expressionList : nullptr;
        auto* literal = arguments && !arguments->next
                            ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(
                                  arguments->value)
                            : nullptr;
        auto spelling =
            literal ? literal->literal->stringValue() : std::string_view{};
        if (!is_symbol_name(spelling)) {
          diagnostics_.reject(unit_, attribute,
                              "symbol binding requires one valid Loom symbol "
                              "string without the @ prefix");
        }
        auto [binding, inserted] =
            explicit_names_.try_emplace(function->canonical(), spelling);
        if (inserted) {
          names_.reserve(spelling, attribute);
        } else if (binding->second != spelling) {
          diagnostics_.reject(unit_, attribute,
                              "conflicting symbol names across redeclarations");
        }
      });
}

cxx::FunctionSymbol* Functions::definition(
    cxx::FunctionSymbol* function) const {
  auto found = definitions_.find(function->canonical());
  if (found != definitions_.end()) {
    return found->second;
  }
  return function->declaration() ? function : nullptr;
}

bool Functions::is_check_case(cxx::FunctionSymbol* function) const {
  return check_cases_.contains(function->canonical());
}

void Functions::build_benchmarks(Locations& locations,
                                 loom_builder_t* builder) {
  for (const auto& benchmark : benchmarks_) {
    auto target = callees_.find(benchmark.case_function->canonical());
    if (target == callees_.end()) {
      continue;
    }
    auto symbol = callees_.at(benchmark.function->canonical());
    loom_op_t* op;
    check(loom_check_benchmark_build(
        builder, LOOM_CHECK_BENCHMARK_BUILD_FLAG_HAS_BENCHMARK, target->second,
        symbol, {}, locations.get(benchmark.source), &op));
  }
}

const std::string& Functions::qualified_name(cxx::FunctionSymbol* symbol) {
  auto found = qualified_names_.find(symbol);
  if (found != qualified_names_.end()) {
    return found->second;
  }
  std::string spelling = cxx::to_string(symbol->name());
  for (auto* owner : symbol->enclosingSymbols()) {
    if (auto* space = cxx::symbol_cast<cxx::NamespaceSymbol>(owner)) {
      if (space->name()) {
        spelling = cxx::to_string(space->name()) + "::" + spelling;
      }
    }
  }
  return qualified_names_.emplace(symbol, std::move(spelling)).first->second;
}

loom_symbol_ref_t Functions::declare(cxx::FunctionSymbol* function) {
  if (auto found = callees_.find(function->canonical());
      found != callees_.end()) {
    return found->second;
  }
  if (!function->templateArguments().empty() && function->declaration()) {
    launches_.declaration(function, function->declaration()->attributeList);
  }
  auto* body = definition(function);
  auto callee = create_symbol(function, body->declaration());
  pending_.push_back(body);
  return callee;
}

loom_symbol_ref_t Functions::create_symbol(cxx::FunctionSymbol* function,
                                           cxx::AST* source) {
  std::string spelling;
  if (auto found = explicit_names_.find(function->canonical());
      found != explicit_names_.end()) {
    spelling = found->second;
  } else if (exported_.contains(function)) {
    bool overloaded = false;
    for (auto* candidate : function->parent()->find(function->name())) {
      if (auto* overloads =
              cxx::symbol_cast<cxx::OverloadSetSymbol>(candidate)) {
        overloaded = overloads->declaredFunctions().size() > 1;
      }
    }
    if (overloaded || !function->templateArguments().empty() ||
        !cxx::name_cast<cxx::Identifier>(function->name())) {
      diagnostics_.reject(unit_, source,
                          "exported overloads, templates and operators require "
                          "an explicit loom::symbol name");
    }
    spelling = function->hasCLinkage() ? cxx::to_string(function->name())
                                       : qualified_name(function);
    for (size_t position = 0; position < spelling.size(); ++position) {
      if (spelling[position] == ':') {
        spelling.replace(position, 2, ".");
      }
    }
    if (!is_symbol_name(spelling)) {
      diagnostics_.reject(unit_, source,
                          "exported source name requires an explicit "
                          "loom::symbol spelling");
    }
    names_.reserve(spelling, source);
  } else {
    spelling = qualified_name(function);
    for (const auto& argument : function->templateArguments()) {
      spelling += "_" + cxx::to_string(argument);
    }
    for (char& ch : spelling) {
      if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
        ch = '_';
      }
    }
    spelling = names_.unique(std::move(spelling));
  }
  loom_string_id_t name;
  check(loom_module_intern_string(module_, view(spelling), &name));
  loom_symbol_id_t id;
  check(loom_module_add_symbol(module_, name, &id));
  loom_symbol_ref_t callee = {0, id};
  callees_[function->canonical()] = callee;
  return callee;
}

FunctionBody Functions::define(cxx::FunctionSymbol* symbol, Types& types,
                               Locations& locations, loom_builder_t* builder) {
  auto* definition = symbol->declaration();
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
      definition->functionBody);
  if (!body) {
    diagnostics_.reject(unit_, definition, "unsupported function body");
  }
  auto parameters = symbol->parameters();
  bool kernel = annotated(symbol, "kernel");
  bool check_case = is_check_case(symbol);
  std::vector<loom_type_t> arguments;
  for (auto* parameter : parameters) {
    if (kernel) {
      arguments.push_back(types.get(parameter->type(), definition));
    } else {
      types.append(parameter->type(), definition, arguments);
    }
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(symbol->type());
  if (!signature || signature->isVariadic()) {
    diagnostics_.reject(unit_, definition,
                        "variadic functions are not admitted");
  }
  bool returns_void = signature->returnType()->kind() == cxx::TypeKind::kVoid;
  loom_op_t* op;
  if (check_case) {
    launches_.reject_ordinary_function(symbol);
    check(loom_check_case_build(
        builder, LOOM_CHECK_CASE_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_CHECK_CASE_VISIBILITY_PUBLIC, callees_.at(symbol->canonical()),
        locations.get(definition), &op));
  } else if (kernel) {
    if (!returns_void) {
      diagnostics_.reject(unit_, definition, "kernel must return void");
    }
    auto callee = callees_.at(symbol->canonical());
    auto name_id = module_->symbols.entries[callee.symbol_id].name_id;
    check(loom_kernel_def_build(builder, 0, 0, {}, LOOM_STRING_ID_INVALID, 0,
                                callee, nullptr, 0, arguments.data(),
                                arguments.size(), nullptr, 0,
                                locations.get(definition), &op));
    auto saved =
        loom_builder_enter_region(builder, op, loom_kernel_def_config(op));
    auto spelling = module_->strings.entries[name_id];
    launches_.build(symbol, {spelling.data, spelling.size}, names_, builder,
                    locations.get(definition));
    loom_builder_restore(builder, saved);
  } else {
    launches_.reject_ordinary_function(symbol);
    std::vector<loom_type_t> results;
    if (!returns_void) {
      types.append(signature->returnType(), definition, results);
    }
    check(loom_func_def_build(
        builder,
        (annotated(symbol, "device") ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC : 0) |
            (exported_.contains(symbol)
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY
                 : 0) |
            (annotated(symbol, "force_inline")
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY
                 : 0),
        exported_.contains(symbol) ? LOOM_FUNC_VISIBILITY_PUBLIC : 0, 0,
        annotated(symbol, "device") ? LOOM_FUNC_CC_DEVICE : 0, 0, 0,
        annotated(symbol, "force_inline") ? LOOM_INLINE_POLICY_INLINE : 0, {},
        0, {}, 0, {}, callees_.at(symbol->canonical()), arguments.data(),
        arguments.size(), results.data(), results.size(), nullptr, 0, nullptr,
        0, locations.get(definition), &op));
  }
  auto* region = check_case ? loom_check_case_body(op)
                 : kernel   ? loom_kernel_def_body(op)
                            : loom_func_def_body(op);
  return {definition,
          body->statement,
          op,
          region,
          signature->returnType(),
          check_case ? FunctionKind::CheckCase
          : kernel   ? FunctionKind::Kernel
                     : FunctionKind::Ordinary};
}

}  // namespace loom::cxx_import
