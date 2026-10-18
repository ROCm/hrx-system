// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/functions.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/attributes.h>
#include <cxx/const_int.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <array>
#include <cctype>
#include <limits>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/template/ops.h"

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
  target_definitions_.build(&builder);
  configs_.build(&builder);
  for (const auto& [function, source] : check_cases_) {
    if (!definition(function)) {
      diagnostics_.reject(unit_, source, "check cases require a definition");
    }
  }
  for (const auto& [function, scenario] : check_scenarios_) {
    if (!definition(function)) {
      diagnostics_.reject(unit_, scenario.source,
                          "check scenarios require a definition");
    }
  }
  for (const auto& benchmark : benchmarks_) {
    if (!is_check_record(benchmark.record_function)) {
      diagnostics_.reject(unit_, benchmark.source,
                          "check benchmark must reference a check record");
    }
  }
  if (!roots.empty()) {
    std::unordered_map<std::string, std::vector<cxx::FunctionSymbol*>>
        candidates;
    for (auto* symbol : definitions) {
      if (template_definitions_.lookup(symbol)) {
        continue;
      }
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
      if (launches_.is_configuration(symbol) ||
          template_definitions_.lookup(symbol)) {
        continue;
      }
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
      if (visible || is_check_record(symbol)) {
        exported_.insert(symbol);
        declare(symbol);
      }
    }
  }
  for (const auto& provider : template_definitions_.providers()) {
    auto* source =
        provider.family_source ? provider.family_source : provider.source;
    if (!provider.family) {
      diagnostics_.reject(unit_, source,
                          "template priority requires a template definition");
    }
    if (annotated(provider.function, "kernel") ||
        annotated(provider.function, "op") ||
        annotated(provider.function, "check_case") ||
        annotated(provider.function, "check_scenario") ||
        annotated(provider.function, "check_benchmark") ||
        launches_.is_configuration(provider.function)) {
      diagnostics_.reject(
          unit_, source,
          "template definitions require ordinary source functions");
    }
    auto* body = definition(provider.function);
    if (!body || !body->declaration()) {
      diagnostics_.reject(unit_, source,
                          "template definitions require a function body");
    }
    retain(body, source);
  }
  // A check record can only be selected as a root, never reached through a
  // call. Reserve its public benchmark names before discovering private
  // helpers.
  for (const auto& benchmark : benchmarks_) {
    if (callees_.contains(benchmark.record_function->canonical())) {
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
    // Function template patterns are not ordinary callable definitions, but
    // their leading operation attribute owns admission for every concrete
    // specialization reached later. Class members and nested templates remain
    // outside the namespace-scope intrinsic surface.
    if (scope == DeclarationScope::Namespace &&
        pattern->templateParameterList) {
      if (auto* simple =
              cxx::ast_cast<cxx::SimpleDeclarationAST>(pattern->declaration)) {
        for (auto* declarator : cxx::ListView{simple->initDeclaratorList}) {
          if (auto* function =
                  cxx::symbol_cast<cxx::FunctionSymbol>(declarator->symbol)) {
            intrinsics_.declaration(function, simple->attributeList,
                                    declarator);
          }
        }
      } else if (auto* function = cxx::ast_cast<cxx::FunctionDefinitionAST>(
                     pattern->declaration)) {
        intrinsics_.declaration(function->symbol, function->attributeList,
                                function);
      }
    }
    collect(pattern->declaration,
            pattern->templateParameterList ? DeclarationScope::Nested : scope,
            definitions);
  } else if (auto* alias =
                 cxx::ast_cast<cxx::AliasDeclarationAST>(declaration)) {
    admit_declaration(nullptr, alias->attributeList, alias, scope);
    reject_misplaced_binding_attributes(unit_, diagnostics_,
                                        alias->typeId->attributeList);
    reject_misplaced_binding_declarator(unit_, diagnostics_,
                                        alias->typeId->declarator);
  } else if (auto* directive =
                 cxx::ast_cast<cxx::UsingDirectiveAST>(declaration)) {
    reject_misplaced_binding_attributes(unit_, diagnostics_,
                                        directive->attributeList);
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
      bool is_module_metadata = admit_declaration(
          declarator->symbol, simple->attributeList, declarator, scope);
      if (scope != DeclarationScope::Namespace) {
        continue;
      }
      if (function) {
        intrinsics_.declaration(function, simple->attributeList, declarator);
        launches_.declaration(function, simple->attributeList);
        if (!function->isTemplatePattern()) {
          declarations_.try_emplace(function->canonical(),
                                    FunctionDeclaration{function, declarator});
        }
      }
      auto* variable =
          cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
      if (variable && !is_module_metadata && !variable->isExtern() &&
          (unit_.typeTraits().is_volatile(variable->type()) ||
           !(variable->isConstexpr() ||
             (unit_.typeTraits().is_const(variable->type()) &&
              variable->constValue())))) {
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
  auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(symbol);
  cxx::DeclaratorAST* source_declarator = nullptr;
  if (auto* declarator = cxx::ast_cast<cxx::InitDeclaratorAST>(owner)) {
    source_declarator = declarator->declarator;
  } else if (auto* definition =
                 cxx::ast_cast<cxx::FunctionDefinitionAST>(owner)) {
    source_declarator = definition->declarator;
  }
  function_imports_.declaration(function, attributes, owner);
  if (source_declarator) {
    auto* prototype =
        function ? cxx::getFunctionPrototype(source_declarator) : nullptr;
    reject_misplaced_binding_declarator(unit_, diagnostics_, source_declarator,
                                        prototype);
    if (function && !function->isTemplatePattern()) {
      parameter_contracts_.declaration(function, prototype);
      fastmath_.declaration(function, attributes, prototype);
      function_contracts_.declaration(function, prototype);
      if (const auto* import = function_imports_.lookup(function);
          import && (cxx::ast_cast<cxx::FunctionDefinitionAST>(owner) ||
                     definitions_.contains(function->canonical()))) {
        diagnostics_.reject(unit_, import->source,
                            "import bindings require bodyless functions");
      }
      if (const auto* import = function_imports_.lookup(function);
          import &&
          (annotated(function, "kernel") || annotated(function, "op") ||
           annotated(function, "assume") || annotated(function, "check_case") ||
           annotated(function, "check_scenario") ||
           annotated(function, "check_benchmark") ||
           annotated(function, "template_def") ||
           annotated(function, "priority") ||
           annotated(function, "force_inline"))) {
        diagnostics_.reject(
            unit_, import->source,
            "import bindings require ordinary non-inline functions");
      }
    }
  }
  reject_misplaced_binding_attributes(unit_, diagnostics_, attributes,
                                      BindingAttributeScope::Declaration);
  if (!function) {
    visit_loom_attributes(
        unit_, attributes,
        [&](std::string_view name, cxx::AttributeAST* attribute) {
          if (name == "fastmath") {
            diagnostics_.reject(
                unit_, attribute,
                "fastmath requires a function declaration or statement");
          }
        });
  }
  if (function) {
    visit_loom_attributes(
        unit_, attributes,
        [&](std::string_view name, cxx::AttributeAST* attribute) {
          if (name == "where") {
            diagnostics_.reject(
                unit_, attribute,
                "function where contracts must follow the parameter list");
          }
        });
  }
  template_definitions_.declaration(function, attributes);
  bool is_module_metadata = configs_.declaration(symbol, attributes, owner);
  is_module_metadata |=
      target_definitions_.declaration(symbol, attributes, owner);
  admit_symbol(symbol, attributes);
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
        if (name != "check_case" && name != "check_scenario" &&
            name != "check_benchmark") {
          return;
        }
        require_namespace_function();
        auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
        if (found || annotated(function, "kernel") ||
            annotated(function, "device") || annotated(function, "op") ||
            signature->isVariadic() ||
            signature->returnType()->kind() != cxx::TypeKind::kVoid) {
          diagnostics_.reject(
              unit_, owner,
              "check declarations require one check annotation on a "
              "non-variadic void function");
        }
        found = true;
        auto* clause = attribute->attributeArgumentClause;
        if (name == "check_case") {
          if (clause || !signature->parameterTypes().empty()) {
            diagnostics_.reject(unit_, attribute,
                                "check_case requires void() and no arguments");
          }
          check_cases_.try_emplace(function->canonical(), owner);
          return;
        }
        if (name == "check_scenario") {
          std::optional<int64_t> configuration_count;
          auto* arguments = clause ? clause->expressionList : nullptr;
          if (arguments) {
            cxx::ASTInterpreter interpreter(&unit_);
            auto value = !arguments->next
                             ? interpreter.evaluate(arguments->value)
                             : std::nullopt;
            auto* count = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
            if (!count || count->isNegative() || count->isZero() ||
                count->toUWide() > static_cast<cxx::ConstInt::UWide>(
                                       std::numeric_limits<int64_t>::max())) {
              diagnostics_.reject(
                  unit_, attribute,
                  "configured check_scenario requires one positive i64 "
                  "configuration count");
            }
            configuration_count = static_cast<int64_t>(count->toUIntMax());
          }
          size_t expected_parameters = configuration_count ? 2 : 0;
          if (signature->parameterTypes().size() != expected_parameters) {
            diagnostics_.reject(
                unit_, attribute,
                configuration_count
                    ? "configured check_scenario requires void(ordinal, "
                      "entropy)"
                    : "unconfigured check_scenario requires void()");
          }
          check_scenarios_.try_emplace(
              function->canonical(),
              CheckScenario{attribute, configuration_count});
          return;
        }
        if (!signature->parameterTypes().empty()) {
          diagnostics_.reject(unit_, attribute,
                              "check_benchmark requires void()");
        }
        if (function->declaration()) {
          diagnostics_.reject(unit_, owner,
                              "check benchmarks are declarations referencing a "
                              "record, without a body");
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
                              "record function name");
        }
        for (const auto& previous : benchmarks_) {
          if (previous.function->canonical() == function->canonical()) {
            diagnostics_.reject(unit_, owner,
                                "check benchmark is already declared");
          }
        }
        benchmarks_.push_back({function, target, attribute});
      });
  if (found) {
    if (auto* source = fastmath_.function_source(function)) {
      diagnostics_.reject(unit_, source,
                          "check declarations cannot carry fastmath contracts");
    }
  }
  if (!found &&
      ((annotated(function, "check_case") && !is_check_case(function)) ||
       annotated(function, "check_scenario") ||
       annotated(function, "check_benchmark"))) {
    require_namespace_function();
    diagnostics_.reject(unit_, owner,
                        "check annotations must precede the declaration");
  }
  return is_module_metadata;
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

bool Functions::is_check_record(cxx::FunctionSymbol* function) const {
  auto* canonical = function->canonical();
  return check_cases_.contains(canonical) ||
         check_scenarios_.contains(canonical);
}

void Functions::build_benchmarks(Locations& locations,
                                 loom_builder_t* builder) {
  for (const auto& benchmark : benchmarks_) {
    auto target = callees_.find(benchmark.record_function->canonical());
    if (target == callees_.end()) {
      continue;
    }
    auto symbol = callees_.at(benchmark.function->canonical());
    loom_op_t* op;
    check(loom_check_benchmark_build(builder, /*build_flags=*/0, target->second,
                                     symbol, {},
                                     locations.get(benchmark.source), &op));
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

loom_symbol_ref_t Functions::declare(cxx::FunctionSymbol* function,
                                     cxx::AST* owner) {
  if (auto* provider = template_definitions_.lookup(function)) {
    diagnostics_.reject(
        unit_, owner ? owner : provider->source,
        "template definition implementations cannot be selected or called "
        "directly");
  }
  return retain(function, owner);
}

loom_symbol_ref_t Functions::retain(cxx::FunctionSymbol* function,
                                    cxx::AST* owner) {
  if (auto found = callees_.find(function->canonical());
      found != callees_.end()) {
    return found->second;
  }
  if (launches_.is_configuration(function)) {
    diagnostics_.reject(
        unit_, owner ? owner : function->declaration(),
        "kernel configuration functions are owned by their kernels and "
        "cannot be selected or called");
  }
  auto* body = definition(function);
  if (!body) {
    auto found = declarations_.find(function->canonical());
    if (found == declarations_.end()) {
      diagnostics_.reject(
          unit_, owner,
          "call requires a concrete namespace-scope function declaration");
    }
    const auto& declaration = found->second;
    if (cxx::has_internal_linkage(declaration.function)) {
      diagnostics_.reject(
          unit_, owner,
          "called function with internal linkage requires a definition");
    }
    external_.insert(function->canonical());
    admit_reference_parameters(declaration.function,
                               owner ? owner : declaration.source);
    auto callee = create_symbol(declaration.function, declaration.source);
    pending_declarations_.push_back(declaration);
    return callee;
  }
  // Reaching a specialization through a kernel launch need not odr-use its
  // body in C++. Importing it still requires the completed semantic definition.
  if (body->hasPendingBody()) {
    cxx::ASTRewriter::completePendingBodyFor(&unit_, body);
    diagnostics_.finish();
  }
  if (!function->templateArguments().empty() && function->declaration()) {
    launches_.declaration(function, function->declaration()->attributeList);
    if (!definitions_.contains(function->canonical())) {
      auto* prototype =
          cxx::getFunctionPrototype(function->declaration()->declarator);
      parameter_contracts_.declaration(function, prototype);
      fastmath_.declaration(function, function->declaration()->attributeList,
                            prototype);
      function_contracts_.declaration(function, prototype);
    }
  }
  admit_reference_parameters(function, owner ? owner : body->declaration());
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
  } else if (exported_.contains(function) ||
             external_.contains(function->canonical())) {
    bool overloaded = false;
    for (auto* candidate : function->parent()->find(function->name())) {
      if (auto* overloads =
              cxx::symbol_cast<cxx::OverloadSetSymbol>(candidate)) {
        overloaded = overloads->declaredFunctions().size() > 1;
      }
    }
    if (overloaded || !function->templateArguments().empty() ||
        !cxx::name_cast<cxx::Identifier>(function->name())) {
      diagnostics_.reject(
          unit_, source,
          "linkable overloads, templates and operators require an explicit "
          "loom::symbol name");
    }
    spelling = function->hasCLinkage() ? cxx::to_string(function->name())
                                       : qualified_name(function);
    for (size_t position = 0; position < spelling.size(); ++position) {
      if (spelling[position] == ':') {
        spelling.replace(position, 2, ".");
      }
    }
    if (!is_symbol_name(spelling)) {
      diagnostics_.reject(
          unit_, source,
          "linkable source name requires an explicit loom::symbol spelling");
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

void Functions::admit_reference_parameters(cxx::FunctionSymbol* symbol,
                                           cxx::AST* source) {
  auto* canonical = symbol->canonical();
  auto* signature = cxx::type_cast<cxx::FunctionType>(symbol->type());
  if (!signature) {
    diagnostics_.reject(unit_, source,
                        "value references require a function type");
  }
  auto traits = unit_.typeTraits();
  if (traits.is_reference(signature->returnType())) {
    diagnostics_.reject(
        unit_, source,
        "runtime reference returns require object identity; return an "
        "admitted value instead");
  }

  std::vector<ValueReferenceParameter> references;
  auto parameters = symbol->parameters();
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter_type = parameters[ordinal]->type();
    if (traits.is_rvalue_reference(parameter_type)) {
      diagnostics_.reject(
          unit_, source,
          "rvalue-reference parameters require consuming object identity");
    }
    if (!traits.is_lvalue_reference(parameter_type)) {
      continue;
    }
    auto* value_type = traits.remove_reference(parameter_type);
    if (traits.is_volatile(value_type)) {
      diagnostics_.reject(unit_, source,
                          "volatile reference parameters require addressable "
                          "storage");
    }
    if (traits.is_array(value_type)) {
      diagnostics_.reject(
          unit_, source,
          "array reference parameters require addressable storage; pass a "
          "view or pointer instead");
    }
    if (traits.is_function(value_type)) {
      diagnostics_.reject(unit_, source,
                          "function reference parameters are not callable "
                          "value references");
    }
    references.push_back(
        {ordinal, !traits.is_const(traits.remove_reference(parameter_type))});
  }

  if (!references.empty()) {
    auto* body = definition(symbol);
    if (!body || exported_.contains(body) || external_.contains(canonical) ||
        annotated(body, "kernel") || is_check_record(body) ||
        launches_.is_configuration(body) ||
        template_definitions_.lookup(body)) {
      diagnostics_.reject(
          unit_, source,
          "runtime reference parameters require an internal ordinary "
          "function definition");
    }
    reference_parameters_.emplace(canonical, std::move(references));
  }
}

std::span<const ValueReferenceParameter> Functions::reference_parameters(
    cxx::FunctionSymbol* symbol) const {
  auto found = reference_parameters_.find(symbol->canonical());
  return found == reference_parameters_.end()
             ? std::span<const ValueReferenceParameter>{}
             : std::span<const ValueReferenceParameter>{found->second};
}

BoundSignature Functions::bind_callable_signature(
    cxx::FunctionSymbol* symbol, cxx::AST* source, Types& types,
    loom_builder_t* builder, std::vector<loom_type_t>& arguments,
    std::vector<loom_type_t>& results,
    std::vector<loom_tied_result_t>& tied_results,
    std::vector<loom_predicate_t>& predicates) {
  auto* signature = cxx::type_cast<cxx::FunctionType>(symbol->type());
  if (!signature || signature->isVariadic()) {
    diagnostics_.reject(unit_, source, "variadic functions are not admitted");
  }
  for (const auto& contract : parameter_contracts_.get(symbol)) {
    if (contract.alignment.source) {
      diagnostics_.reject(
          unit_, contract.alignment.source,
          "assume_aligned on ordinary helper parameters requires "
          "pointer-origin alignment support; kernel parameters are supported");
    }
    if (contract.noalias_source) {
      diagnostics_.reject(
          unit_, contract.noalias_source,
          "noalias on ordinary helper parameters requires scoped alias "
          "contracts; kernel parameters are supported");
    }
  }
  std::vector<const cxx::Type*> sources;
  auto parameters = symbol->parameters();
  auto references = reference_parameters(symbol);
  const bool returns_void =
      signature->returnType()->kind() == cxx::TypeKind::kVoid;
  sources.reserve(parameters.size() + !returns_void + references.size());
  size_t argument_count = 0;
  std::vector<size_t> argument_offsets;
  if (!references.empty()) {
    argument_offsets.resize(parameters.size());
  }
  size_t reference_index = 0;
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    const bool is_reference = reference_index < references.size() &&
                              references[reference_index].ordinal == ordinal;
    auto* value_type = is_reference ? types.unqualified_value(parameter->type())
                                    : types.unqualified(parameter->type());
    if (!references.empty()) {
      argument_offsets[ordinal] = argument_count;
    }
    sources.push_back(value_type);
    argument_count += types.partition(value_type, source).component_count;
    reference_index += is_reference;
  }
  size_t ordinary_result_count = 0;
  if (!returns_void) {
    sources.push_back(signature->returnType());
    ordinary_result_count =
        types.partition(signature->returnType(), source).component_count;
  }
  for (const auto& reference : references) {
    if (reference.mutable_access) {
      sources.push_back(
          types.unqualified_value(parameters[reference.ordinal]->type()));
    }
  }
  const bool has_predicates = function_contracts_.has_predicates(symbol);
  auto callable = bind_signature(
      types, sources, source, builder,
      has_predicates ? SignatureIdentityRequirement::Required
                     : SignatureIdentityRequirement::DependentTypes);
  arguments.assign(callable.types.begin(),
                   callable.types.begin() + argument_count);
  results.assign(callable.types.begin() + argument_count, callable.types.end());
  size_t result_offset = ordinary_result_count;
  for (const auto& reference : references) {
    if (!reference.mutable_access) {
      continue;
    }
    const auto component_count =
        types
            .partition(
                types.unqualified_value(parameters[reference.ordinal]->type()),
                source)
            .component_count;
    const size_t argument_offset = argument_offsets[reference.ordinal];
    if (argument_offset + component_count > UINT16_MAX ||
        result_offset + component_count > UINT16_MAX) {
      diagnostics_.reject(unit_, source,
                          "value-reference signature exceeds native operand "
                          "or result limits");
    }
    for (size_t component = 0; component < component_count; ++component) {
      tied_results.push_back({
          .result_index = static_cast<uint16_t>(result_offset + component),
          .operand_index = static_cast<uint16_t>(argument_offset + component),
          .has_type_change =
              !loom_type_equal(arguments[argument_offset + component],
                               results[result_offset + component]),
      });
    }
    result_offset += component_count;
  }
  auto predicate_identities = std::span(callable.identities);
  if (!predicate_identities.empty()) {
    predicate_identities =
        predicate_identities.first(argument_count + ordinary_result_count);
  }
  predicates =
      function_contracts_.bind(symbol, types, predicate_identities,
                               FunctionContractSignature::Flattened, source);
  return callable;
}

void Functions::build_declarations(Types& types, Locations& locations,
                                   loom_builder_t* builder) {
  for (const auto& declaration : pending_declarations_) {
    auto* symbol = declaration.function;
    auto* source = declaration.source;
    launches_.reject_ordinary_function(symbol);
    if (function_contracts_.has_requirements(symbol)) {
      diagnostics_.reject(
          unit_, source,
          "target requirements require a template family or definition");
    }
    if (auto* contract = fastmath_.function_source(symbol)) {
      diagnostics_.reject(
          unit_, contract,
          "bodyless function declarations cannot carry fastmath contracts");
    }
    std::vector<loom_type_t> arguments;
    std::vector<loom_type_t> results;
    std::vector<loom_tied_result_t> tied_results;
    std::vector<loom_predicate_t> predicates;
    // Dependent type payloads in this signature must outlive the declaration
    // builder call below.
    auto bound_signature =
        bind_callable_signature(symbol, source, types, builder, arguments,
                                results, tied_results, predicates);

    loom_func_decl_build_flags_t flags = 0;
    loom_string_id_t import_module = LOOM_STRING_ID_INVALID;
    loom_string_id_t import_symbol = LOOM_STRING_ID_INVALID;
    if (const auto* import = function_imports_.lookup(symbol)) {
      flags |= LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE;
      check(loom_builder_intern_string(builder, view(import->module),
                                       &import_module));
      if (import->symbol) {
        flags |= LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_SYMBOL;
        check(loom_builder_intern_string(builder, view(*import->symbol),
                                         &import_symbol));
      }
    }
    if (annotated(symbol, "device")) {
      flags |= LOOM_FUNC_DECL_BUILD_FLAG_HAS_CC;
    }
    if (annotated(symbol, "force_inline")) {
      flags |= LOOM_FUNC_DECL_BUILD_FLAG_HAS_INLINE_POLICY;
    }
    if (!predicates.empty()) {
      flags |= LOOM_FUNC_DECL_BUILD_FLAG_HAS_PREDICATES;
    }
    loom_op_t* op;
    check(loom_func_decl_build(
        builder, flags, /*visibility=*/0, /*retain=*/0, import_module,
        import_symbol, annotated(symbol, "device") ? LOOM_FUNC_CC_DEVICE : 0,
        /*effect_contract=*/0, /*temperature=*/0,
        annotated(symbol, "force_inline") ? LOOM_INLINE_POLICY_INLINE : 0, {},
        /*abi=*/0, {}, LOOM_STRING_ID_INVALID, {},
        callees_.at(symbol->canonical()), arguments.data(), arguments.size(),
        results.data(), results.size(), tied_results.data(),
        tied_results.size(), predicates.data(), predicates.size(),
        locations.get(source), &op));
  }
}

DefinedFunction Functions::define(cxx::FunctionSymbol* symbol, Types& types,
                                  Locations& locations,
                                  loom_builder_t* builder) {
  auto* definition = symbol->declaration();
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
      definition->functionBody);
  if (!body) {
    diagnostics_.reject(unit_, definition, "unsupported function body");
  }
  auto parameters = symbol->parameters();
  bool kernel = annotated(symbol, "kernel");
  bool check_case = is_check_case(symbol);
  auto scenario = check_scenarios_.find(symbol->canonical());
  bool check_scenario = scenario != check_scenarios_.end();
  auto* template_definition = template_definitions_.lookup(symbol);
  auto parameter_contracts = parameter_contracts_.get(symbol);
  auto* signature = cxx::type_cast<cxx::FunctionType>(symbol->type());
  if (!signature || signature->isVariadic()) {
    diagnostics_.reject(unit_, definition,
                        "variadic functions are not admitted");
  }
  bool returns_void = signature->returnType()->kind() == cxx::TypeKind::kVoid;
  bool has_predicates = function_contracts_.has_predicates(symbol);
  bool has_requirements = function_contracts_.has_requirements(symbol);
  if ((check_case || check_scenario) && has_predicates) {
    diagnostics_.reject(unit_, definition,
                        "check records cannot carry callable predicates");
  }
  if (has_requirements && !template_definition) {
    diagnostics_.reject(
        unit_, definition,
        "target requirements require a template family or definition");
  }
  std::vector<loom_type_t> arguments;
  std::vector<loom_type_t> results;
  std::vector<loom_tied_result_t> tied_results;
  std::vector<loom_predicate_t> predicates;
  std::optional<LaunchConfiguration> launch_configuration;
  BoundSignature configuration_signature;
  BoundSignature callable_signature;
  if (kernel) {
    for (auto* parameter : parameters) {
      arguments.push_back(
          types.get(types.unqualified(parameter->type()), definition));
    }
    if (auto* configuration = launches_.configuration_function(symbol)) {
      auto* configuration_symbol = this->definition(configuration);
      if (!configuration_symbol || !configuration_symbol->declaration()) {
        diagnostics_.reject(
            unit_, definition,
            "kernel configuration function requires a definition");
      }
      launch_configuration =
          launches_.bind_configuration(configuration_symbol, types);
    }
  } else if (!check_case && !check_scenario) {
    callable_signature =
        bind_callable_signature(symbol, definition, types, builder, arguments,
                                results, tied_results, predicates);
  }
  loom_op_t* op;
  if (check_case) {
    launches_.reject_ordinary_function(symbol);
    check(loom_check_case_build(
        builder, LOOM_CHECK_CASE_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_CHECK_VISIBILITY_PUBLIC, callees_.at(symbol->canonical()),
        locations.get(definition), &op));
  } else if (check_scenario) {
    launches_.reject_ordinary_function(symbol);
    std::array<loom_type_t, 2> configuration_arguments = {};
    loom_check_scenario_build_flags_t flags =
        LOOM_CHECK_SCENARIO_BUILD_FLAG_HAS_VISIBILITY;
    int64_t configuration_count = 0;
    size_t configuration_argument_count = 0;
    if (scenario->second.configuration_count) {
      if (parameters.size() != 2 ||
          !types.is_index(parameters[0]->type(), definition)) {
        diagnostics_.reject(unit_, scenario->second.source,
                            "configured check_scenario first parameter must be "
                            "loom::check::ordinal");
      }
      auto entropy_type = types.get(parameters[1]->type(), definition);
      auto entropy_name = loom_type_is_dialect(entropy_type)
                              ? loom_type_dialect_name_id(entropy_type)
                              : LOOM_STRING_ID_INVALID;
      if (entropy_name == LOOM_STRING_ID_INVALID ||
          !iree_string_view_equal(
              loom_string_table_get(&module_->strings, entropy_name),
              IREE_SV("check.entropy"))) {
        diagnostics_.reject(
            unit_, scenario->second.source,
            "configured check_scenario second parameter must be "
            "loom::check::entropy");
      }
      flags |= LOOM_CHECK_SCENARIO_BUILD_FLAG_HAS_CONFIGURATION_COUNT;
      configuration_count = *scenario->second.configuration_count;
      configuration_arguments = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                 entropy_type};
      configuration_argument_count = configuration_arguments.size();
    }
    check(loom_check_scenario_build(
        builder, flags, LOOM_CHECK_VISIBILITY_PUBLIC,
        callees_.at(symbol->canonical()), configuration_count,
        configuration_arguments.data(), configuration_argument_count,
        locations.get(definition), &op));
  } else if (kernel) {
    if (!returns_void) {
      diagnostics_.reject(unit_, definition, "kernel must return void");
    }
    auto callee = callees_.at(symbol->canonical());
    auto target = target_definitions_.reference(symbol);
    auto name_id = module_->symbols.entries[callee.symbol_id].name_id;
    size_t configuration_identity_count = 0;
    bool configuration_requires_binding = false;
    if (launch_configuration) {
      for (const auto* workload : launch_configuration->workloads) {
        configuration_identity_count +=
            types.partition(workload, definition).component_count;
        configuration_requires_binding |=
            types.requires_binding(workload, definition);
      }
    }
    std::vector<loom_value_id_t> kernel_identities;
    if (configuration_requires_binding || has_predicates) {
      kernel_identities.resize(configuration_identity_count +
                               (has_predicates ? arguments.size() : 0));
      if (!kernel_identities.empty()) {
        check(loom_builder_reserve_values(builder, kernel_identities.size(),
                                          kernel_identities.data()));
      }
      if (launch_configuration) {
        configuration_signature = bind_signature_to(
            types, launch_configuration->workloads, definition,
            std::span<const loom_value_id_t>(kernel_identities)
                .first(configuration_identity_count));
      }
    } else if (launch_configuration) {
      configuration_signature = bind_signature(
          types, launch_configuration->workloads, definition, builder);
    }
    std::span<const loom_type_t> configuration_arguments =
        launch_configuration ? std::span(configuration_signature.types)
                             : std::span<const loom_type_t>{};
    if (has_predicates) {
      predicates = function_contracts_.bind(
          symbol, types,
          std::span<const loom_value_id_t>(kernel_identities)
              .subspan(configuration_identity_count),
          FunctionContractSignature::Kernel, definition);
    }
    check(loom_kernel_def_build(
        builder,
        (loom_symbol_ref_is_valid(target)
             ? LOOM_KERNEL_DEF_BUILD_FLAG_HAS_TARGET
             : 0) |
            (predicates.empty() ? 0
                                : LOOM_KERNEL_DEF_BUILD_FLAG_HAS_PREDICATES),
        0, target, LOOM_STRING_ID_INVALID, 0, callee,
        configuration_arguments.data(), configuration_arguments.size(),
        arguments.data(), arguments.size(), predicates.data(),
        predicates.size(), locations.get(definition), &op));
    auto saved =
        loom_builder_enter_region(builder, op, loom_kernel_def_config(op));
    auto spelling = loom_string_table_get(&module_->strings, name_id);
    launches_.build(symbol, {spelling.data, spelling.size}, names_, builder,
                    locations.get(definition));
    loom_builder_restore(builder, saved);
  } else if (template_definition) {
    launches_.reject_ordinary_function(symbol);
    auto family = intrinsics_.bind_template_family(
        template_definition->family, template_definition->family_source,
        builder);
    template_definitions_.validate(*template_definition, family.function,
                                   types);
    auto target = target_definitions_.reference(symbol);
    bool is_device = annotated(family.function, "device");
    loom_template_def_build_flags_t flags = 0;
    if (is_device) {
      flags |= LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_CC;
    }
    if (loom_symbol_ref_is_valid(target)) {
      flags |= LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_TARGET;
    }
    if (template_definition->priority) {
      flags |= LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_PRIORITY;
    }
    if (!predicates.empty()) {
      flags |= LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_PREDICATES;
    }
    auto requirements = function_contracts_.bind_requirements(symbol, module_);
    if (!requirements.empty()) {
      flags |= LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_REQUIRES;
    }
    check(loom_template_def_build(
        builder, flags, family.reference, /*visibility=*/0, /*retain=*/0,
        is_device ? LOOM_TEMPLATE_CC_DEVICE : 0,
        /*effect_contract=*/0, /*temperature=*/0, target,
        loom_make_parameterized_attr_array(requirements.data(),
                                           requirements.size()),
        template_definition->priority.value_or(0),
        callees_.at(symbol->canonical()), arguments.data(), arguments.size(),
        results.data(), results.size(), tied_results.data(),
        tied_results.size(), predicates.data(), predicates.size(),
        locations.get(definition), &op));
  } else {
    launches_.reject_ordinary_function(symbol);
    check(loom_func_def_build(
        builder,
        (annotated(symbol, "device") ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC : 0) |
            (exported_.contains(symbol)
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY
                 : 0) |
            (annotated(symbol, "force_inline")
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY
                 : 0) |
            (predicates.empty() ? 0 : LOOM_FUNC_DEF_BUILD_FLAG_HAS_PREDICATES),
        exported_.contains(symbol) ? LOOM_FUNC_VISIBILITY_PUBLIC : 0, 0,
        annotated(symbol, "device") ? LOOM_FUNC_CC_DEVICE : 0, 0, 0,
        annotated(symbol, "force_inline") ? LOOM_INLINE_POLICY_INLINE : 0, {},
        0, {}, 0, {}, callees_.at(symbol->canonical()), arguments.data(),
        arguments.size(), results.data(), results.size(), tied_results.data(),
        tied_results.size(), predicates.data(), predicates.size(),
        locations.get(definition), &op));
  }
  auto* region = check_case            ? loom_check_case_body(op)
                 : check_scenario      ? loom_check_scenario_body(op)
                 : kernel              ? loom_kernel_def_body(op)
                 : template_definition ? loom_template_def_body(op)
                                       : loom_func_def_body(op);
  FunctionBody function_body = {
      definition,
      body->statement,
      op,
      region,
      signature->returnType(),
      check_case            ? FunctionKind::CheckCase
      : check_scenario      ? FunctionKind::CheckScenario
      : kernel              ? FunctionKind::Kernel
      : template_definition ? FunctionKind::TemplateDefinition
                            : FunctionKind::Ordinary,
      parameter_contracts,
      reference_parameters(symbol),
  };
  std::optional<FunctionBody> configuration;
  if (launch_configuration) {
    configuration = FunctionBody{
        launch_configuration->source,
        launch_configuration->body,
        op,
        loom_kernel_def_config(op),
        launch_configuration->result_type,
        launch_configuration->kind == LaunchConfigurationKind::Clustered
            ? FunctionKind::ClusteredLaunchConfiguration
            : FunctionKind::LaunchConfiguration,
        {},
        {},
    };
  }
  return {function_body, configuration};
}

}  // namespace loom::cxx_import
