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
#include <deque>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/symbol/immutable_objects.h"
#include "loom/import/cxx/symbol/source_signature.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/template/ops.h"

namespace loom::cxx_import {
namespace {

std::string_view private_operator_name(cxx::FunctionSymbol* function) {
  auto* operator_id = cxx::name_cast<cxx::OperatorId>(function->name());
  if (!operator_id) {
    return {};
  }
  const size_t operand_count =
      function->parameters().size() +
      (function->isImplicitObjectMemberFunction() ? 1u : 0u);
  switch (operator_id->op()) {
    case cxx::TokenKind::T_PLUS:
      return operand_count == 1 ? "operator_positive" : "operator_add";
    case cxx::TokenKind::T_MINUS:
      return operand_count == 1 ? "operator_negate" : "operator_subtract";
    case cxx::TokenKind::T_STAR:
      return operand_count == 1 ? "operator_dereference" : "operator_multiply";
    case cxx::TokenKind::T_AMP:
      return operand_count == 1 ? "operator_address" : "operator_bit_and";
    case cxx::TokenKind::T_PLUS_PLUS:
      return operand_count == 1 ? "operator_pre_increment"
                                : "operator_post_increment";
    case cxx::TokenKind::T_MINUS_MINUS:
      return operand_count == 1 ? "operator_pre_decrement"
                                : "operator_post_decrement";
    case cxx::TokenKind::T_SLASH:
      return "operator_divide";
    case cxx::TokenKind::T_PERCENT:
      return "operator_remainder";
    case cxx::TokenKind::T_CARET:
      return "operator_bit_xor";
    case cxx::TokenKind::T_BAR:
      return "operator_bit_or";
    case cxx::TokenKind::T_TILDE:
      return "operator_complement";
    case cxx::TokenKind::T_EXCLAIM:
      return "operator_not";
    case cxx::TokenKind::T_EQUAL:
      return "operator_assign";
    case cxx::TokenKind::T_LESS:
      return "operator_less";
    case cxx::TokenKind::T_GREATER:
      return "operator_greater";
    case cxx::TokenKind::T_PLUS_EQUAL:
      return "operator_add_assign";
    case cxx::TokenKind::T_MINUS_EQUAL:
      return "operator_subtract_assign";
    case cxx::TokenKind::T_STAR_EQUAL:
      return "operator_multiply_assign";
    case cxx::TokenKind::T_SLASH_EQUAL:
      return "operator_divide_assign";
    case cxx::TokenKind::T_PERCENT_EQUAL:
      return "operator_remainder_assign";
    case cxx::TokenKind::T_CARET_EQUAL:
      return "operator_bit_xor_assign";
    case cxx::TokenKind::T_AMP_EQUAL:
      return "operator_bit_and_assign";
    case cxx::TokenKind::T_BAR_EQUAL:
      return "operator_bit_or_assign";
    case cxx::TokenKind::T_LESS_LESS:
      return "operator_left_shift";
    case cxx::TokenKind::T_GREATER_GREATER:
      return "operator_right_shift";
    case cxx::TokenKind::T_GREATER_GREATER_EQUAL:
      return "operator_right_shift_assign";
    case cxx::TokenKind::T_LESS_LESS_EQUAL:
      return "operator_left_shift_assign";
    case cxx::TokenKind::T_EQUAL_EQUAL:
      return "operator_equal";
    case cxx::TokenKind::T_EXCLAIM_EQUAL:
      return "operator_not_equal";
    case cxx::TokenKind::T_LESS_EQUAL:
      return "operator_less_equal";
    case cxx::TokenKind::T_GREATER_EQUAL:
      return "operator_greater_equal";
    case cxx::TokenKind::T_LESS_EQUAL_GREATER:
      return "operator_compare";
    case cxx::TokenKind::T_AMP_AMP:
      return "operator_logical_and";
    case cxx::TokenKind::T_BAR_BAR:
      return "operator_logical_or";
    case cxx::TokenKind::T_COMMA:
      return "operator_comma";
    case cxx::TokenKind::T_MINUS_GREATER_STAR:
      return "operator_member_pointer";
    case cxx::TokenKind::T_MINUS_GREATER:
      return "operator_arrow";
    case cxx::TokenKind::T_LPAREN:
      return "operator_call";
    case cxx::TokenKind::T_LBRACKET:
      return "operator_subscript";
    case cxx::TokenKind::T_NEW:
      return "operator_new";
    case cxx::TokenKind::T_NEW_ARRAY:
      return "operator_new_array";
    case cxx::TokenKind::T_DELETE:
      return "operator_delete";
    case cxx::TokenKind::T_DELETE_ARRAY:
      return "operator_delete_array";
    case cxx::TokenKind::T_CO_AWAIT:
      return "operator_co_await";
    default:
      return {};
  }
}

}  // namespace

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
          template_definitions_.lookup(symbol) ||
          symbol->isImplicitObjectMemberFunction()) {
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
    retain({body->canonical(), ImplicitObjectABI::None}, source);
  }
  // A check record can only be selected as a root, never reached through a
  // call. Reserve its public benchmark names before discovering private
  // helpers.
  for (const auto& benchmark : benchmarks_) {
    if (callees_.contains({benchmark.record_function->canonical(),
                           ImplicitObjectABI::None})) {
      exported_.insert(benchmark.function);
      create_symbol({benchmark.function->canonical(), ImplicitObjectABI::None},
                    benchmark.source);
    }
  }
}

bool Functions::ordinary_analysis_call(const DirectCall& call) {
  auto* function = call.function;
  if (!function || annotated(function, "kernel") ||
      launches_.is_configuration(function) || is_check_record(function) ||
      annotated(function, "check_benchmark") ||
      intrinsics_.lookup(function, call.source)) {
    return false;
  }
  return definition(function) != nullptr;
}

void Functions::prepare_analysis(cxx::FunctionSymbol* function, Types& types,
                                 std::vector<AnalyzedFunction>& functions,
                                 std::vector<AnalyzedCall>& calls) {
  auto* canonical = function->canonical();
  if (control_flows_.contains(canonical)) {
    return;
  }
  auto* concrete = definition(function);
  if (!concrete || !concrete->declaration()) {
    return;
  }
  if (concrete->hasPendingBody()) {
    cxx::ASTRewriter::completePendingBodyFor(&unit_, concrete);
    diagnostics_.finish();
  }
  auto* definition_ast = concrete->declaration();
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
      definition_ast->functionBody);
  if (!body) {
    return;
  }
  auto analysis = std::make_unique<ControlFlow>(unit_, diagnostics_, types,
                                                body->statement, concrete);
  auto* retained = analysis.get();
  control_flows_.emplace(canonical, std::move(analysis));
  functions.emplace_back(canonical, retained);
  for (const auto& call : retained->calls()) {
    if (ordinary_analysis_call(call)) {
      prepare_analysis(call.function, types, functions, calls);
      if (auto* callee = mutable_control_flow(call.function)) {
        calls.push_back({retained, &call, callee});
      }
    }
  }
}

void Functions::analyze(Types& types) {
  std::vector<AnalyzedFunction> analyzed_functions;
  std::vector<AnalyzedCall> analyzed_calls;
  for (const auto& instance : pending_) {
    prepare_analysis(instance.symbol, types, analyzed_functions,
                     analyzed_calls);
  }

  std::unordered_map<ControlFlow*, std::vector<size_t>> incident_calls;
  std::deque<size_t> pending_calls;
  std::vector<bool> queued(analyzed_calls.size(), true);
  for (size_t index = 0; index < analyzed_calls.size(); ++index) {
    const auto& edge = analyzed_calls[index];
    incident_calls[edge.caller].push_back(index);
    if (edge.callee != edge.caller) {
      incident_calls[edge.callee].push_back(index);
    }
    pending_calls.push_back(index);
  }
  auto enqueue_incident = [&](ControlFlow* function) {
    for (size_t index : incident_calls.at(function)) {
      if (!queued[index]) {
        queued[index] = true;
        pending_calls.push_back(index);
      }
    }
  };
  while (!pending_calls.empty()) {
    const size_t index = pending_calls.front();
    pending_calls.pop_front();
    queued[index] = false;
    const auto& edge = analyzed_calls[index];
    auto changed = edge.caller->propagate_call(*edge.call, *edge.callee);
    if (changed.caller_changed) {
      enqueue_incident(edge.caller);
    }
    if (changed.callee_changed) {
      enqueue_incident(edge.callee);
    }
  }

  reference_parameters_.clear();
  for (const auto& [canonical, analysis] : analyzed_functions) {
    analysis->finalize_callable_effects();
    auto* function = definition(canonical);
    if (!function) {
      continue;
    }
    std::vector<ReferenceParameter> references;
    auto parameters = function->parameters();
    for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
      auto* parameter = parameters[ordinal];
      if (!unit_.typeTraits().is_lvalue_reference(parameter->type())) {
        continue;
      }
      references.push_back(
          {ordinal,
           (analysis->reference_requires_storage(parameter) ||
            unit_.typeTraits().is_volatile(
                unit_.typeTraits().remove_reference(parameter->type())) ||
            unit_.typeTraits().is_array(
                unit_.typeTraits().remove_reference(parameter->type())))
               ? ReferenceABI::Storage
               : ReferenceABI::Value,
           analysis->reference_written(parameter)});
    }
    if (!references.empty()) {
      reference_parameters_.emplace(canonical, std::move(references));
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
      if (variable && !is_module_metadata) {
        if (!variable->isExtern() &&
            (unit_.typeTraits().is_volatile(variable->type()) ||
             !(variable->isConstexpr() ||
               (unit_.typeTraits().is_const(variable->type()) &&
                variable->constValue())))) {
          diagnostics_.reject(unit_, declarator,
                              "global storage definitions require a "
                              "global-storage projection");
        }
        immutable_objects_.retain_definition(variable, declarator);
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
            !unit_.typeTraits().is_void(signature->returnType())) {
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
    auto target = callees_.find(
        {benchmark.record_function->canonical(), ImplicitObjectABI::None});
    if (target == callees_.end()) {
      continue;
    }
    auto symbol =
        callees_.at({benchmark.function->canonical(), ImplicitObjectABI::None});
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
    } else if (auto* record = cxx::symbol_cast<cxx::ClassSymbol>(owner)) {
      if (record->name()) {
        spelling = cxx::to_string(record->name()) + "::" + spelling;
      }
    }
  }
  return qualified_names_.emplace(symbol, std::move(spelling)).first->second;
}

loom_symbol_ref_t Functions::declare(cxx::FunctionSymbol* function,
                                     cxx::AST* owner,
                                     ImplicitObjectABI receiver_abi) {
  if (auto* provider = template_definitions_.lookup(function)) {
    diagnostics_.reject(
        unit_, owner ? owner : provider->source,
        "template definition implementations cannot be selected or called "
        "directly");
  }
  return retain({function->canonical(), receiver_abi}, owner);
}

loom_symbol_ref_t Functions::retain(FunctionInstance instance,
                                    cxx::AST* owner) {
  auto* function = instance.symbol;
  const bool implicit = function->isImplicitObjectMemberFunction();
  if (implicit == (instance.receiver_abi == ImplicitObjectABI::None)) {
    diagnostics_.reject(
        unit_, owner ? owner : function->declaration(),
        implicit
            ? "non-static member call requires an implicit object receiver"
            : "implicit object receiver requires a non-static member function");
  }
  if (auto found = callees_.find(instance); found != callees_.end()) {
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
    if (instance.receiver_abi != ImplicitObjectABI::None) {
      diagnostics_.reject(
          unit_, owner, "non-static member call requires a source definition");
    }
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
    auto callee = create_symbol(
        {declaration.function->canonical(), ImplicitObjectABI::None},
        declaration.source);
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
  auto retained = FunctionInstance{body, instance.receiver_abi};
  auto callee = create_symbol(retained, body->declaration());
  pending_.push_back(retained);
  return callee;
}

loom_symbol_ref_t Functions::create_symbol(FunctionInstance instance,
                                           cxx::AST* source) {
  auto* function = instance.symbol;
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
    const std::string source_name = cxx::to_string(function->name());
    if (auto operator_name = private_operator_name(function);
        !operator_name.empty()) {
      spelling.replace(spelling.size() - source_name.size(), source_name.size(),
                       operator_name);
    }
    for (const auto& argument : function->templateArguments()) {
      spelling += "_" + cxx::to_string(argument);
    }
    for (char& ch : spelling) {
      if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
        ch = '_';
      }
    }
    if (instance.receiver_abi == ImplicitObjectABI::Value) {
      spelling += "_value";
    } else if (instance.receiver_abi == ImplicitObjectABI::Storage) {
      spelling += "_storage";
    }
    spelling = names_.unique(std::move(spelling));
  }
  loom_string_id_t name;
  check(loom_module_intern_string(module_, view(spelling), &name));
  loom_symbol_id_t id;
  check(loom_module_add_symbol(module_, name, &id));
  loom_symbol_ref_t callee = {0, id};
  callees_[{function->canonical(), instance.receiver_abi}] = callee;
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
  if (traits.is_rvalue_reference(signature->returnType())) {
    diagnostics_.reject(unit_, source,
                        "rvalue-reference returns require consuming object "
                        "identity");
  }
  if (traits.is_lvalue_reference(signature->returnType()) &&
      traits.is_function(traits.remove_reference(signature->returnType()))) {
    diagnostics_.reject(unit_, source,
                        "function reference returns require callable values");
  }

  bool has_runtime_reference =
      traits.is_lvalue_reference(signature->returnType());
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
    if (traits.is_function(value_type)) {
      diagnostics_.reject(unit_, source,
                          "function reference parameters are not callable "
                          "value references");
    }
    has_runtime_reference = true;
  }

  if (has_runtime_reference) {
    auto* body = definition(symbol);
    if (!body || exported_.contains(body) || external_.contains(canonical) ||
        annotated(body, "kernel") || is_check_record(body) ||
        launches_.is_configuration(body) ||
        template_definitions_.lookup(body)) {
      diagnostics_.reject(
          unit_, source,
          "runtime reference signatures require an internal ordinary "
          "function definition");
    }
  }
}

std::span<const ReferenceParameter> Functions::reference_parameters(
    cxx::FunctionSymbol* symbol) const {
  auto found = reference_parameters_.find(symbol->canonical());
  return found == reference_parameters_.end()
             ? std::span<const ReferenceParameter>{}
             : std::span<const ReferenceParameter>{found->second};
}

const ControlFlow* Functions::control_flow(cxx::FunctionSymbol* symbol) const {
  auto found = control_flows_.find(symbol->canonical());
  return found == control_flows_.end() ? nullptr : found->second.get();
}

ControlFlow* Functions::mutable_control_flow(cxx::FunctionSymbol* symbol) {
  auto* concrete = definition(symbol);
  auto found = concrete ? control_flows_.find(concrete->canonical())
                        : control_flows_.end();
  return found == control_flows_.end() ? nullptr : found->second.get();
}

bool Functions::receiver_requires_storage(cxx::FunctionSymbol* symbol) const {
  auto* analysis = control_flow(symbol);
  return analysis && analysis->receiver_requires_storage();
}

bool Functions::returns_current_receiver_reference(
    cxx::FunctionSymbol* symbol) const {
  auto* analysis = control_flow(symbol);
  return analysis && analysis->returns_current_receiver_reference();
}

std::span<cxx::FieldSymbol* const> Functions::receiver_writes(
    cxx::FunctionSymbol* symbol) const {
  auto* analysis = control_flow(symbol);
  return analysis ? analysis->receiver_writes()
                  : std::span<cxx::FieldSymbol* const>{};
}

std::optional<ImplicitObjectParameter> Functions::implicit_object(
    FunctionInstance instance, cxx::AST* source) const {
  auto* symbol = instance.symbol;
  if (instance.receiver_abi == ImplicitObjectABI::None) {
    return std::nullopt;
  }
  auto* object_type = cxx_import::implicit_object_type(unit_, symbol);
  if (!object_type) {
    diagnostics_.reject(unit_, source,
                        "implicit object receiver requires a non-static "
                        "member function of a class");
  }
  auto* analysis = control_flow(symbol);
  if (!analysis) {
    diagnostics_.reject(unit_, source,
                        "implicit object requires retained callable analysis");
  }
  if (instance.receiver_abi == ImplicitObjectABI::Value &&
      analysis->receiver_requires_storage()) {
    diagnostics_.reject(unit_, source,
                        "member function requires a storage-backed receiver");
  }
  return ImplicitObjectParameter{
      instance.receiver_abi, object_type,
      instance.receiver_abi == ImplicitObjectABI::Value
          ? analysis->receiver_writes()
          : std::span<cxx::FieldSymbol* const>{}};
}

BoundSignature Functions::bind_callable_signature(
    FunctionInstance instance, cxx::AST* source, Types& types,
    loom_builder_t* builder, std::vector<loom_type_t>& arguments,
    std::vector<loom_type_t>& results,
    std::vector<loom_tied_result_t>& tied_results,
    std::vector<loom_predicate_t>& predicates) {
  auto* symbol = instance.symbol;
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
  auto receiver = implicit_object(instance, source);
  const bool returns_void = unit_.typeTraits().is_void(signature->returnType());
  const bool returns_reference =
      unit_.typeTraits().is_lvalue_reference(signature->returnType());
  const bool returns_receiver_alias =
      returns_reference && receiver &&
      receiver->abi == ImplicitObjectABI::Value &&
      returns_current_receiver_reference(symbol);
  sources.reserve(receiver.has_value() + parameters.size() + !returns_void +
                  references.size() +
                  (receiver ? receiver->written_fields.size() : 0));
  size_t argument_count = 0;
  size_t receiver_argument_count = 0;
  if (receiver) {
    const cxx::Type* receiver_type =
        receiver->abi == ImplicitObjectABI::Value
            ? types.unqualified(receiver->object_type)
            : unit_.typeTraits().add_pointer(receiver->object_type);
    sources.push_back(receiver_type);
    receiver_argument_count =
        types.partition(receiver_type, source).component_count;
    argument_count = receiver_argument_count;
  }
  std::vector<size_t> argument_offsets;
  if (!references.empty()) {
    argument_offsets.resize(parameters.size());
  }
  size_t reference_index = 0;
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    const bool is_reference = reference_index < references.size() &&
                              references[reference_index].ordinal == ordinal;
    auto* value_type =
        !is_reference ? types.unqualified(parameter->type())
        : references[reference_index].abi == ReferenceABI::Storage
            ? unit_.typeTraits().add_pointer(
                  unit_.typeTraits().remove_reference(parameter->type()))
            : types.unqualified_value(parameter->type());
    if (!references.empty()) {
      argument_offsets[ordinal] = argument_count;
    }
    sources.push_back(value_type);
    argument_count += types.partition(value_type, source).component_count;
    reference_index += is_reference;
  }
  size_t ordinary_result_count = 0;
  if (!returns_void && !returns_receiver_alias) {
    auto* result_type =
        returns_reference
            ? unit_.typeTraits().add_pointer(
                  unit_.typeTraits().remove_reference(signature->returnType()))
            : signature->returnType();
    sources.push_back(result_type);
    ordinary_result_count =
        types.partition(result_type, source).component_count;
  }
  if (receiver && receiver->abi == ImplicitObjectABI::Value) {
    for (auto* field : receiver->written_fields) {
      sources.push_back(types.unqualified_value(field->type()));
    }
  }
  for (const auto& reference : references) {
    if (reference.abi == ReferenceABI::Value && reference.mutable_access) {
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
  if (receiver && receiver->abi == ImplicitObjectABI::Value) {
    for (auto* field : receiver->written_fields) {
      const auto& member = types.member(field, source);
      const size_t component_count = member.partition->component_count;
      if (member.component_offset + component_count > UINT16_MAX ||
          result_offset + component_count > UINT16_MAX) {
        diagnostics_.reject(unit_, source,
                            "implicit object signature exceeds native operand "
                            "or result limits");
      }
      for (size_t component = 0; component < component_count; ++component) {
        const size_t operand = member.component_offset + component;
        tied_results.push_back({
            .result_index = static_cast<uint16_t>(result_offset + component),
            .operand_index = static_cast<uint16_t>(operand),
            .has_type_change = !loom_type_equal(
                arguments[operand], results[result_offset + component]),
        });
      }
      result_offset += component_count;
    }
  }
  for (const auto& reference : references) {
    if (reference.abi != ReferenceABI::Value || !reference.mutable_access) {
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
  std::vector<loom_value_id_t> projected_predicate_identities;
  if (receiver && !predicate_identities.empty()) {
    projected_predicate_identities.reserve(
        argument_count - receiver_argument_count + ordinary_result_count);
    auto explicit_arguments = predicate_identities.subspan(
        receiver_argument_count, argument_count - receiver_argument_count);
    projected_predicate_identities.insert(projected_predicate_identities.end(),
                                          explicit_arguments.begin(),
                                          explicit_arguments.end());
    auto ordinary_results =
        predicate_identities.subspan(argument_count, ordinary_result_count);
    projected_predicate_identities.insert(projected_predicate_identities.end(),
                                          ordinary_results.begin(),
                                          ordinary_results.end());
    predicate_identities = projected_predicate_identities;
  } else if (!predicate_identities.empty()) {
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
    auto bound_signature = bind_callable_signature(
        {symbol->canonical(), ImplicitObjectABI::None}, source, types, builder,
        arguments, results, tied_results, predicates);
    const loom_attribute_t source_signature = build_c_source_signature(
        unit_, diagnostics_, module_, symbol, source, types);
    const auto target = target_definitions_.reference(symbol);

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
    if (loom_symbol_ref_is_valid(target)) {
      flags |= LOOM_FUNC_DECL_BUILD_FLAG_HAS_TARGET;
    }
    loom_op_t* op;
    check(loom_func_decl_build(
        builder, flags, /*visibility=*/0, /*retain=*/0, import_module,
        import_symbol, annotated(symbol, "device") ? LOOM_FUNC_CC_DEVICE : 0,
        /*effect_contract=*/0, /*temperature=*/0,
        annotated(symbol, "force_inline") ? LOOM_INLINE_POLICY_INLINE : 0,
        target, source_signature, /*abi=*/0, {}, LOOM_STRING_ID_INVALID, {},
        callees_.at({symbol->canonical(), ImplicitObjectABI::None}),
        arguments.data(), arguments.size(), results.data(), results.size(),
        tied_results.data(), tied_results.size(), predicates.data(),
        predicates.size(), locations.get(source), &op));
  }
}

DefinedFunction Functions::define(FunctionInstance instance, Types& types,
                                  Locations& locations,
                                  loom_builder_t* builder) {
  auto* symbol = instance.symbol;
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
  bool returns_void = unit_.typeTraits().is_void(signature->returnType());
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
      target_definitions_.reject_launch_configuration_function(
          configuration_symbol);
      launch_configuration =
          launches_.bind_configuration(configuration_symbol, types);
    }
  } else if (!check_case && !check_scenario) {
    callable_signature =
        bind_callable_signature(instance, definition, types, builder, arguments,
                                results, tied_results, predicates);
  }
  loom_op_t* op;
  if (check_case) {
    launches_.reject_ordinary_function(symbol);
    check(loom_check_case_build(
        builder, LOOM_CHECK_CASE_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_CHECK_VISIBILITY_PUBLIC,
        callees_.at({symbol->canonical(), instance.receiver_abi}),
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
        callees_.at({symbol->canonical(), instance.receiver_abi}),
        configuration_count, configuration_arguments.data(),
        configuration_argument_count, locations.get(definition), &op));
  } else if (kernel) {
    if (!returns_void) {
      diagnostics_.reject(unit_, definition, "kernel must return void");
    }
    auto callee = callees_.at({symbol->canonical(), instance.receiver_abi});
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
    const bool link_once = symbol->canonical()->isInline();
    check(loom_kernel_def_build(
        builder,
        (loom_symbol_ref_is_valid(target)
             ? LOOM_KERNEL_DEF_BUILD_FLAG_HAS_TARGET
             : 0) |
            (link_once ? LOOM_KERNEL_DEF_BUILD_FLAG_HAS_LINKAGE : 0) |
            (predicates.empty() ? 0
                                : LOOM_KERNEL_DEF_BUILD_FLAG_HAS_PREDICATES),
        link_once ? LOOM_SYMBOL_LINKAGE_LINK_ONCE : 0, /*retain=*/0, target,
        LOOM_STRING_ID_INVALID, /*export_linkage=*/0, callee,
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
        callees_.at({symbol->canonical(), instance.receiver_abi}),
        arguments.data(), arguments.size(), results.data(), results.size(),
        tied_results.data(), tied_results.size(), predicates.data(),
        predicates.size(), locations.get(definition), &op));
  } else {
    launches_.reject_ordinary_function(symbol);
    const auto target = target_definitions_.reference(symbol);
    const bool link_once =
        exported_.contains(symbol) && symbol->canonical()->isInline();
    const loom_attribute_t source_signature = build_c_source_signature(
        unit_, diagnostics_, module_, symbol, definition, types);
    check(loom_func_def_build(
        builder,
        (annotated(symbol, "device") ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC : 0) |
            (exported_.contains(symbol)
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY
                 : 0) |
            (annotated(symbol, "force_inline")
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY
                 : 0) |
            (link_once ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_LINKAGE : 0) |
            (loom_symbol_ref_is_valid(target)
                 ? LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET
                 : 0) |
            (predicates.empty() ? 0 : LOOM_FUNC_DEF_BUILD_FLAG_HAS_PREDICATES),
        exported_.contains(symbol) ? LOOM_FUNC_VISIBILITY_PUBLIC : 0,
        link_once ? LOOM_SYMBOL_LINKAGE_LINK_ONCE : 0, /*retain=*/0,
        annotated(symbol, "device") ? LOOM_FUNC_CC_DEVICE : 0,
        /*effect_contract=*/0, /*temperature=*/0,
        annotated(symbol, "force_inline") ? LOOM_INLINE_POLICY_INLINE : 0,
        target, source_signature, 0, {}, 0, {},
        callees_.at({symbol->canonical(), instance.receiver_abi}),
        arguments.data(), arguments.size(), results.data(), results.size(),
        tied_results.data(), tied_results.size(), predicates.data(),
        predicates.size(), locations.get(definition), &op));
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
      implicit_object(instance, definition),
      control_flow(symbol),
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
        std::nullopt,
        nullptr,
    };
  }
  return {function_body, configuration};
}

}  // namespace loom::cxx_import
