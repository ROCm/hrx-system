// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/launch.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/dependent_types.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <span>
#include <utility>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"

namespace loom::cxx_import {
namespace {

bool has_unsigned_scalar_leaves(const RecordPartition& record, Types& types) {
  for (const auto& member : record.members) {
    if (member.partition->kind == ValueKind::Record) {
      if (!has_unsigned_scalar_leaves(
              static_cast<const RecordPartition&>(*member.partition), types)) {
        return false;
      }
      continue;
    }
    if (member.partition->kind != ValueKind::SSA ||
        member.partition->component_count != 1 ||
        !types.is_unsigned(types.unqualified(member.type()))) {
      return false;
    }
  }
  return true;
}

}  // namespace

LaunchContracts::Arguments LaunchContracts::read_arguments(
    Annotation annotation) {
  Arguments result{};
  auto* clause = annotation.source->attributeArgumentClause;
  auto* next = clause ? clause->expressionList : nullptr;
  const size_t arity = annotation.arity();
  size_t count = 0;
  while (next && count < arity) {
    result[count++] = next->value;
    next = next->next;
  }
  if (count != arity || next) {
    if (annotation.field == Field::Configuration) {
      diagnostics_.reject(
          unit_, annotation.source,
          "kernel configuration requires one unambiguous function name");
    }
    diagnostics_.reject(
        unit_, annotation.source,
        annotation.form == Form::Exact
            ? "exact launch dimensions require three constants: x, y, z"
            : "launch ranges require six constants: xmin, xmax, ymin, ymax, "
              "zmin, zmax");
  }
  return result;
}

LaunchContracts::Dimensions LaunchContracts::parse(
    Annotation annotation, std::span<cxx::ExpressionAST* const> arguments) {
  std::array<int32_t, 6> values;
  cxx::ASTInterpreter interpreter(&unit_);
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto evaluated =
        arguments[i] ? interpreter.evaluate(arguments[i]) : std::nullopt;
    auto* value = evaluated ? std::get_if<cxx::ConstInt>(&*evaluated) : nullptr;
    if (!value || value->isNegative() || value->isZero() ||
        value->toUWide() > INT32_MAX) {
      diagnostics_.reject(unit_, annotation.source,
                          "launch dimensions require positive i32 integer "
                          "constant expressions");
    }
    values[i] = static_cast<int32_t>(value->toUIntMax());
  }
  Dimensions dimensions = {
      .form = annotation.form,
      .source = annotation.source,
  };
  for (size_t axis = 0; axis < 3; ++axis) {
    dimensions.axes[axis] = annotation.form == Form::Exact
                                ? Axis{values[axis], values[axis]}
                                : Axis{values[axis * 2], values[axis * 2 + 1]};
    if (dimensions.axes[axis].lower > dimensions.axes[axis].upper) {
      diagnostics_.reject(unit_, annotation.source,
                          "launch range lower bound exceeds its upper bound");
    }
  }
  return dimensions;
}

cxx::FunctionSymbol* LaunchContracts::parse_configuration(
    cxx::AttributeAST* attribute, cxx::ExpressionAST* expression) {
  auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression);
  auto* function =
      id ? cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol) : nullptr;
  if (id) {
    if (auto* overloads =
            cxx::symbol_cast<cxx::OverloadSetSymbol>(id->symbol)) {
      auto functions = overloads->functions();
      if (functions.size() == 1) {
        function = functions.front();
      }
    }
  }
  if (!function) {
    diagnostics_.reject(
        unit_, attribute,
        "kernel configuration requires one unambiguous function name");
  }
  if (function->isTemplatePattern()) {
    diagnostics_.reject(
        unit_, attribute,
        "kernel configuration function template requires concrete template "
        "arguments");
  }
  return function->canonical();
}

void LaunchContracts::apply(Contract& contract, Annotation annotation,
                            const Arguments& arguments) {
  auto* source = annotation.source;
  auto conflict = [&](cxx::AttributeAST* other, std::string_view message) {
    auto here = source->firstSourceLocation();
    auto there = other->firstSourceLocation();
    // Constant annotations may have been checked before dependent ones.
    if (here < there) {
      std::swap(here, there);
    }
    diagnostics_.reject(here, message, there, "previous launch contract");
  };
  constexpr std::string_view mixed =
      "kernel configuration functions cannot be combined with launch dimension "
      "annotations";
  if (annotation.field == Field::Configuration) {
    if (contract.count || contract.size) {
      conflict(contract.count ? contract.count->source : contract.size->source,
               mixed);
    }
    auto* configuration = parse_configuration(source, arguments[0]);
    if (contract.configuration && contract.configuration != configuration) {
      conflict(
          contract.configuration_source,
          "conflicting kernel configuration functions across declarations");
    }
    if (!contract.configuration) {
      contract.configuration = configuration;
      contract.configuration_source = source;
    }
  } else {
    if (contract.configuration) {
      conflict(contract.configuration_source, mixed);
    }
    auto& previous =
        annotation.field == Field::Count ? contract.count : contract.size;
    auto next =
        parse(annotation, std::span(arguments).first(annotation.arity()));
    if (previous &&
        (previous->form != next.form || previous->axes != next.axes)) {
      conflict(previous->source,
               "conflicting launch contracts across declarations");
    }
    if (!previous) {
      previous = next;
    }
  }
}

void LaunchContracts::declaration(
    cxx::FunctionSymbol* function,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    cxx::TemplateDeclarationAST* pattern) {
  if (!attributes) {
    return;
  }
  auto* canonical = function->canonical();
  // With a single declaration CXX already substitutes its attributes when it
  // instantiates the definition. Only redeclarations require a merged contract.
  if (pattern && canonical->redeclarations().empty()) {
    return;
  }
  std::array<bool, 3> seen = {};
  visit_loom_attributes(
      unit_, attributes, [&](std::string_view name, cxx::AttributeAST* source) {
        Field field;
        Form form = Form::Exact;
        if (name == "kernel") {
          field = Field::Configuration;
        } else if (name == "workgroup_count" ||
                   name == "workgroup_count_range") {
          field = Field::Count;
          if (name == "workgroup_count_range") {
            form = Form::Range;
          }
        } else if (name == "workgroup_size" || name == "workgroup_size_range") {
          field = Field::Size;
          if (name == "workgroup_size_range") {
            form = Form::Range;
          }
        } else {
          return;
        }
        if (std::exchange(seen[static_cast<size_t>(field)], true)) {
          diagnostics_.reject(
              unit_, source,
              field == Field::Configuration
                  ? "one kernel annotation is allowed on a declaration"
                  : "one launch contract per dimension group is allowed on a "
                    "declaration");
        }
        auto* clause = source->attributeArgumentClause;
        if (field == Field::Configuration && !clause) {
          return;
        }
        Annotation annotation{field, form, source};
        auto arguments = read_arguments(annotation);
        if (pattern) {
          auto& contract = templates_[canonical];
          DependentAnnotation pending{annotation, pattern, arguments, {}};
          for (size_t i = 0; i < annotation.arity(); ++i) {
            pending.dependent.set(i, cxx::isDependent(&unit_, arguments[i]));
          }
          if (pending.dependent.any()) {
            contract.dependent.push_back(pending);
          } else {
            apply(contract.fixed, annotation, arguments);
          }
        } else {
          auto& contract = contracts_[canonical];
          apply(contract, annotation, arguments);
          if (field == Field::Configuration) {
            configurations_.insert(contract.configuration);
          }
        }
      });
}

const LaunchContracts::Contract* LaunchContracts::resolve(
    cxx::FunctionSymbol* function) {
  function = function->canonical();
  if (auto found = contracts_.find(function); found != contracts_.end()) {
    return &found->second;
  }
  auto* primary = function->primaryTemplateSymbol();
  if (!primary || function->isExplicitSpecialization()) {
    return nullptr;
  }
  if (primary->redeclarations().empty()) {
    auto* definition = function->declaration();
    if (!definition || !definition->attributeList) {
      return nullptr;
    }
    declaration(function, definition->attributeList);
    auto found = contracts_.find(function);
    return found == contracts_.end() ? nullptr : &found->second;
  }
  auto found = templates_.find(primary);
  if (found == templates_.end()) {
    return nullptr;
  }
  auto& pattern = found->second;
  auto& concrete = contracts_.emplace(function, pattern.fixed).first->second;
  const auto index = function->templateSpecializationIndex();
  for (const auto& pending : pattern.dependent) {
    auto operands = pending.arguments;
    for (size_t i = 0; i < pending.annotation.arity(); ++i) {
      if (pending.dependent[i]) {
        operands[i] = cxx::ASTRewriter::substituteDefaultExpression(
            &unit_, operands[i], primary->specializations()[index].arguments,
            pending.declaration->depth, pending.declaration->symbol->parent());
      }
    }
    apply(concrete, pending.annotation, operands);
  }
  if (concrete.configuration) {
    configurations_.insert(concrete.configuration);
  }
  return &concrete;
}

void LaunchContracts::reject_ordinary_function(cxx::FunctionSymbol* function) {
  if (const auto* contract = resolve(function)) {
    auto* source = contract->configuration_source
                       ? contract->configuration_source
                       : (contract->count ? contract->count->source
                                          : contract->size->source);
    diagnostics_.reject(unit_, source,
                        "launch contracts require a kernel function");
  }
}

cxx::FunctionSymbol* LaunchContracts::configuration_function(
    cxx::FunctionSymbol* function) {
  const auto* contract = resolve(function);
  return contract ? contract->configuration : nullptr;
}

LaunchConfiguration LaunchContracts::bind_configuration(
    cxx::FunctionSymbol* selected, cxx::AST* kernel, Types& types) {
  auto* function = selected->resolvedDefinition();
  // Naming a specialization in an attribute does not odr-use its body.
  // CXX can attach a later pattern definition when this kernel needs it.
  if (function->isSpecialization()) {
    cxx::ASTRewriter::completePendingBodyFor(&unit_, function);
    diagnostics_.finish();
  }
  auto* definition = function->declaration();
  if (!definition) {
    diagnostics_.reject(unit_, kernel,
                        "kernel configuration function requires a definition");
  }
  if (!cxx::has_internal_linkage(function) ||
      !cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent()) ||
      is_kernel_function(function) || annotated(function, "device") ||
      annotated(function, "op") || annotated(function, "check_case") ||
      annotated(function, "check_benchmark")) {
    diagnostics_.reject(
        unit_, definition,
        "kernel configuration requires an ordinary namespace-scope function "
        "with internal linkage");
  }
  auto* source_body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
      definition->functionBody);
  auto* body = source_body ? source_body->statement : nullptr;
  if (!body) {
    diagnostics_.reject(unit_, definition,
                        "kernel configuration requires an ordinary compound "
                        "function body");
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (!signature || signature->isVariadic()) {
    diagnostics_.reject(unit_, definition,
                        "kernel configuration requires a fixed function "
                        "signature");
  }
  std::vector<const cxx::Type*> workloads;
  for (auto* parameter : function->parameters()) {
    auto* parameter_type = types.unqualified(parameter->type());
    if (unit_.typeTraits().is_reference(parameter->type()) ||
        unit_.typeTraits().is_volatile(parameter->type())) {
      diagnostics_.reject(
          unit_, definition,
          "kernel configuration workloads require non-reference "
          "non-volatile value parameters");
    }
    types.partition(parameter_type, definition);
    workloads.push_back(parameter_type);
  }

  auto* result_type = signature->returnType();
  auto* result_class =
      cxx::type_cast<cxx::ClassType>(types.unqualified(result_type));
  auto* result_symbol = result_class ? result_class->definition() : nullptr;
  bool ordinary = annotated(result_symbol, "launch_config");
  bool clustered = annotated(result_symbol, "clustered_launch_config");
  if (ordinary == clustered) {
    diagnostics_.reject(
        unit_, definition,
        "kernel configuration must return loom::kernel::configuration or "
        "loom::kernel::clustered_configuration");
  }
  const auto& result_partition = types.partition(result_type, definition);
  auto* record = result_partition.kind == ValueKind::Record
                     ? static_cast<const RecordPartition*>(&result_partition)
                     : nullptr;
  static constexpr std::string_view ordinary_names[] = {
      "workgroup_count_x", "workgroup_count_y", "workgroup_count_z",
      "workgroup_size_x",  "workgroup_size_y",  "workgroup_size_z",
  };
  static constexpr std::string_view clustered_names[] = {
      "workgroup_count_x",        "workgroup_count_y",
      "workgroup_count_z",        "workgroup_size_x",
      "workgroup_size_y",         "workgroup_size_z",
      "workgroup_cluster_size_x", "workgroup_cluster_size_y",
      "workgroup_cluster_size_z",
  };
  auto expected_names = clustered
                            ? std::span<const std::string_view>(clustered_names)
                            : std::span<const std::string_view>(ordinary_names);
  bool valid_result = record &&
                      record->component_names.size() == expected_names.size() &&
                      has_unsigned_scalar_leaves(*record, types);
  if (valid_result) {
    for (size_t i = 0; i < expected_names.size(); ++i) {
      valid_result &= record->component_names[i] == expected_names[i];
    }
  }
  std::vector<loom_type_t> result_types;
  if (valid_result) {
    types.append(result_type, definition, result_types);
    valid_result = result_types.size() == expected_names.size();
    for (auto type : result_types) {
      valid_result &=
          loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
    }
  }
  if (!valid_result) {
    diagnostics_.reject(
        unit_, definition,
        "kernel launch configuration aggregates require unsigned x, y, and "
        "z fields for each dimension group");
  }
  return {
      definition,
      body,
      result_type,
      std::move(workloads),
      clustered ? LaunchConfigurationKind::Clustered
                : LaunchConfigurationKind::Standard,
  };
}

bool LaunchContracts::is_configuration(cxx::FunctionSymbol* function) const {
  return configurations_.contains(function->canonical());
}

std::array<loom_value_id_t, 3> LaunchContracts::build_dimensions(
    const std::optional<Dimensions>& dimensions, std::string_view prefix,
    std::string_view name, SymbolNames& names, cxx::AST* source,
    loom_builder_t* builder, loom_location_id_t location) {
  std::array<loom_value_id_t, 3> values;
  loom_builder_t declaration_builder;
  auto* module = builder->module;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &declaration_builder);
  for (size_t axis = 0; axis < 3; ++axis) {
    loom_op_t* op;
    if (dimensions && dimensions->form == Form::Exact) {
      check(loom_index_constant_build(
          builder, loom_attr_i64(dimensions->axes[axis].lower),
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &op));
    } else {
      auto spelling =
          std::string(prefix) + "." + std::string(name) + "." + "xyz"[axis];
      names.reserve(spelling, source);
      loom_string_id_t name_id;
      check(loom_builder_intern_string(builder, view(spelling), &name_id));
      loom_symbol_id_t id;
      check(loom_module_add_symbol(module, name_id, &id));
      loom_symbol_ref_t reference = {0, id};
      loom_op_t* declaration;
      loom_predicate_t predicate = {};
      loom_value_id_t value;
      if (dimensions) {
        check(loom_builder_reserve_values(&declaration_builder, 1, &value));
        predicate = {
            .kind = LOOM_PREDICATE_RANGE,
            .arg_count = 3,
            .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                         LOOM_PRED_ARG_CONST},
            .args = {value, dimensions->axes[axis].lower,
                     dimensions->axes[axis].upper},
        };
      }
      check(loom_config_decl_build(
          &declaration_builder,
          dimensions ? LOOM_CONFIG_DECL_BUILD_FLAG_HAS_PREDICATES : 0,
          reference, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), nullptr, 0,
          dimensions ? &predicate : nullptr, dimensions ? 1 : 0, location,
          &declaration));
      check(loom_config_get_build(builder, reference,
                                  loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                  location, &op));
      auto hint = std::string(name) + "_" + "xyz"[axis];
      loom_string_id_t hint_id;
      check(loom_builder_intern_string(builder, view(hint), &hint_id));
      check(
          loom_module_set_value_name(module, loom_op_results(op)[0], hint_id));
    }
    values[axis] = loom_op_results(op)[0];
  }
  return values;
}

void LaunchContracts::build(cxx::FunctionSymbol* function,
                            std::string_view symbol, SymbolNames& names,
                            loom_builder_t* builder,
                            loom_location_id_t location) {
  const auto found = contracts_.find(function->canonical());
  const Contract absent;
  const auto& contract = found == contracts_.end() ? absent : found->second;
  if (contract.configuration) {
    return;
  }
  auto count =
      build_dimensions(contract.count, symbol, "workgroup_count", names,
                       function->declaration(), builder, location);
  auto size = build_dimensions(contract.size, symbol, "workgroup_size", names,
                               function->declaration(), builder, location);
  loom_op_t* launch;
  check(loom_kernel_launch_config_build(builder, 0, count[0], count[1],
                                        count[2], size[0], size[1], size[2], 0,
                                        0, 0, location, &launch));
}

}  // namespace loom::cxx_import
