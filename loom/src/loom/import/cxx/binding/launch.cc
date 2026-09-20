// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/launch.h"

#include <cxx/ast_interpreter.h>
#include <cxx/symbols.h>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"

namespace loom::cxx_import {

LaunchContracts::Dimensions LaunchContracts::parse(
    Form form, cxx::AttributeAST* attribute) {
  auto* clause = attribute->attributeArgumentClause;
  size_t expected = form == Form::Exact ? 3 : 6;
  std::array<int32_t, 6> values;
  size_t count = 0;
  if (clause) {
    cxx::ASTInterpreter interpreter(&unit_);
    for (auto* expression : cxx::ListView{clause->expressionList}) {
      auto evaluated = interpreter.evaluate(expression);
      auto* value =
          evaluated ? std::get_if<std::intmax_t>(&*evaluated) : nullptr;
      if (count == expected || !value || *value <= 0 || *value > INT32_MAX) {
        diagnostics_.reject(unit_, attribute,
                            "launch dimensions require positive i32 integer "
                            "constant expressions");
      }
      values[count++] = static_cast<int32_t>(*value);
    }
  }
  if (count != expected) {
    diagnostics_.reject(
        unit_, attribute,
        form == Form::Exact
            ? "exact launch dimensions require three constants: x, y, z"
            : "launch ranges require six constants: xmin, xmax, ymin, ymax, "
              "zmin, zmax");
  }
  Dimensions dimensions = {.form = form, .source = attribute};
  for (size_t axis = 0; axis < 3; ++axis) {
    dimensions.axes[axis] = form == Form::Exact
                                ? Axis{values[axis], values[axis]}
                                : Axis{values[axis * 2], values[axis * 2 + 1]};
    if (dimensions.axes[axis].lower > dimensions.axes[axis].upper) {
      diagnostics_.reject(unit_, attribute,
                          "launch range lower bound exceeds its upper bound");
    }
  }
  return dimensions;
}

void LaunchContracts::merge(std::optional<Dimensions>& previous,
                            const std::optional<Dimensions>& next) {
  if (!next) {
    return;
  }
  if (previous &&
      (previous->form != next->form || previous->axes != next->axes)) {
    diagnostics_.reject(unit_, next->source,
                        "conflicting launch contracts across declarations");
  }
  previous = next;
}

void LaunchContracts::declaration(
    cxx::FunctionSymbol* function,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes) {
  Contract contract;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        std::optional<Dimensions>* dimensions = nullptr;
        Form form = Form::Exact;
        if (name == "workgroup_count" || name == "workgroup_count_range") {
          dimensions = &contract.count;
          form = name == "workgroup_count" ? Form::Exact : Form::Range;
        } else if (name == "workgroup_size" || name == "workgroup_size_range") {
          dimensions = &contract.size;
          form = name == "workgroup_size" ? Form::Exact : Form::Range;
        } else {
          return;
        }
        if (*dimensions) {
          diagnostics_.reject(
              unit_, attribute,
              "one launch contract per dimension group is allowed "
              "on a declaration");
        }
        *dimensions = parse(form, attribute);
      });
  if (contract.count || contract.size) {
    auto& previous = contracts_[function->canonical()];
    merge(previous.count, contract.count);
    merge(previous.size, contract.size);
  }
}

void LaunchContracts::reject_ordinary_function(cxx::FunctionSymbol* function) {
  if (contracts_.contains(function->canonical())) {
    diagnostics_.reject(unit_, function->declaration(),
                        "launch contracts require a kernel function");
  }
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
        check(loom_builder_reserve_results(&declaration_builder, 1, &value));
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
