// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/config.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>

#include <algorithm>
#include <span>
#include <string>

#include "loom/analysis/symbol_value_constraints.h"
#include "loom/import/cxx/binding/predicates.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

bool predicate_equal(const loom_predicate_t& left,
                     const loom_predicate_t& right) {
  if (left.kind != right.kind || left.arg_count != right.arg_count) {
    return false;
  }
  for (size_t index = 0; index < left.arg_count; ++index) {
    if (left.arg_tags[index] != right.arg_tags[index] ||
        left.args[index] != right.args[index]) {
      return false;
    }
  }
  return true;
}

void append_unique(std::vector<loom_predicate_t>& destination,
                   std::span<const loom_predicate_t> source) {
  for (const auto& predicate : source) {
    if (std::none_of(destination.begin(), destination.end(),
                     [&](const loom_predicate_t& existing) {
                       return predicate_equal(existing, predicate);
                     })) {
      destination.push_back(predicate);
    }
  }
}

}  // namespace

bool Configs::declaration(cxx::Symbol* symbol,
                          cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                          cxx::AST* owner) {
  cxx::AttributeAST* selected = nullptr;
  cxx::AttributeAST* where = nullptr;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name == "where") {
          if (where) {
            diagnostics_.reject(
                unit_, attribute,
                "one where contract per declaration is allowed");
          }
          where = attribute;
          return;
        }
        if (name != "config") {
          return;
        }
        if (selected) {
          diagnostics_.reject(unit_, attribute,
                              "one config binding is allowed per declaration");
        }
        selected = attribute;
      });
  if (!selected) {
    if (where) {
      diagnostics_.reject(
          unit_, where, "leading where contracts require a config declaration");
    }
    return false;
  }
  auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(symbol);
  if (!variable || variable->isTemplatePattern() ||
      !cxx::symbol_cast<cxx::NamespaceSymbol>(variable->parent())) {
    diagnostics_.reject(
        unit_, selected,
        "config bindings require namespace-scope scalar variables");
  }
  auto* clause = selected->attributeArgumentClause;
  auto* arguments = clause ? clause->expressionList : nullptr;
  auto* literal =
      arguments && !arguments->next
          ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(arguments->value)
          : nullptr;
  auto name = literal ? literal->literal->stringValue() : std::string_view{};
  if (!is_symbol_name(name)) {
    diagnostics_.reject(
        unit_, selected,
        "config binding requires one valid Loom symbol string key");
  }
  const auto& traits = unit_.typeTraits();
  auto* source_type = variable->type();
  if (!traits.is_const(source_type) || traits.is_volatile(source_type) ||
      variable->isThreadLocal() ||
      !(traits.is_arithmetic(source_type) || traits.is_enum(source_type))) {
    diagnostics_.reject(unit_, owner,
                        "config bindings require nonvolatile const numeric "
                        "scalars without thread-local storage");
  }
  auto type = types_.get(source_type, owner);
  std::vector<loom_predicate_t> predicates;
  if (where) {
    auto* clause = where->attributeArgumentClause;
    auto* arguments = clause ? clause->expressionList : nullptr;
    if (!arguments || arguments->next) {
      diagnostics_.reject(unit_, where,
                          "where requires one predicate expression");
    }
    auto projected = project_predicates(unit_, diagnostics_, arguments->value,
                                        PredicateSubject::Implicit);
    for (const auto& predicate : projected) {
      if (!predicate_accepts_source_type(unit_, predicate.predicate.kind,
                                         types_.unqualified(source_type))) {
        diagnostics_.reject(
            unit_, where,
            std::string(loom_predicate_kind_name(predicate.predicate.kind)) +
                " does not accept this config source type");
      }
      if (predicate.value_count != 1 ||
          predicate.values[0].origin != PredicateValueOrigin::Subject) {
        cxx::AST* source = where;
        for (size_t index = 0; index < predicate.value_count; ++index) {
          if (predicate.values[index].origin != PredicateValueOrigin::Subject &&
              predicate.values[index].source) {
            source = predicate.values[index].source;
            break;
          }
        }
        diagnostics_.reject(
            unit_, source,
            "config predicates may reference only the implicit config value "
            "and pure constants");
      }
      for (const auto* argument_type : predicate.argument_types) {
        if (argument_type && types_.unqualified(argument_type) !=
                                 types_.unqualified(source_type)) {
          diagnostics_.reject(
              unit_, where,
              "config predicate arguments must match the declared source "
              "type");
        }
      }
      predicates.push_back(predicate.predicate);
    }
  }
  std::optional<loom_attribute_t> value;
  if (variable->initializer()) {
    if (!variable->constValue()) {
      diagnostics_.reject(unit_, owner,
                          "config definitions require a constant initializer");
    }
    value = scalars_.constant_attribute(*variable->constValue(), source_type,
                                        owner);
  } else if (!variable->isExtern()) {
    diagnostics_.reject(
        unit_, owner,
        "config declarations without an initializer require extern");
  }
  auto [named, inserted] = names_.try_emplace(name, bindings_.size());
  if (inserted) {
    symbol_names_.reserve(name, owner);
    bindings_.push_back({name, types_.unqualified(source_type), type,
                         std::move(predicates), value, owner,
                         loom_symbol_ref_null()});
  } else {
    auto& binding = bindings_[named->second];
    if (binding.source_type != types_.unqualified(source_type)) {
      diagnostics_.reject(unit_, owner,
                          "conflicting source types for config key");
    }
    if (binding.value && value &&
        !loom_attribute_equal(&*binding.value, &*value)) {
      diagnostics_.reject(unit_, owner,
                          "conflicting definitions for config key");
    }
    if (value) {
      binding.value = value;
      binding.owner = owner;
    }
    append_unique(binding.predicates, predicates);
  }
  auto [source, fresh] =
      variables_.try_emplace(variable->canonical(), named->second);
  if (!fresh && source->second != named->second) {
    diagnostics_.reject(unit_, owner,
                        "conflicting config keys across redeclarations");
  }
  declarations_.insert(variable);
  return true;
}

void Configs::build(loom_builder_t* builder) {
  for (const auto& [canonical, index] : variables_) {
    auto require_annotation = [&](cxx::VariableSymbol* declaration) {
      if (!declarations_.contains(declaration)) {
        diagnostics_.reject(unit_.tokenAt(declaration->location()),
                            "every config redeclaration requires its explicit "
                            "loom::config attribute");
      }
    };
    require_annotation(canonical);
    for (auto* declaration : canonical->redeclarations()) {
      require_annotation(declaration);
    }
  }
  for (auto& binding : bindings_) {
    loom_string_id_t name;
    check(loom_builder_intern_string(builder, view(binding.name), &name));
    check(loom_module_add_symbol(builder->module, name,
                                 &binding.reference.symbol_id));
    binding.reference.module_id = 0;
    loom_op_t* op;
    if (binding.value) {
      if (!binding.predicates.empty()) {
        iree_status_t status = loom_symbol_value_constraints_check_exact(
            view(binding.name), binding.type, /*contract_value=*/0,
            *binding.value,
            loom_attr_predicate_list(binding.predicates.data(),
                                     binding.predicates.size()));
        if (!iree_status_is_ok(status)) {
          auto message = iree_status_message(status);
          std::string text(message.data, message.size);
          iree_status_free(status);
          diagnostics_.reject(unit_, binding.owner, text);
        }
      }
      check(loom_config_def_build(builder, 0, binding.reference, *binding.value,
                                  binding.type, nullptr, 0, nullptr, 0,
                                  locations_.get(binding.owner), &op));
    } else {
      auto predicates = binding.predicates;
      loom_config_decl_build_flags_t flags = 0;
      loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
      if (!predicates.empty()) {
        check(loom_builder_reserve_values(builder, 1, &identity));
        for (auto& predicate : predicates) {
          for (size_t index = 0; index < predicate.arg_count; ++index) {
            if (predicate.arg_tags[index] == LOOM_PRED_ARG_VALUE) {
              predicate.args[index] = identity;
            }
          }
        }
        flags |= LOOM_CONFIG_DECL_BUILD_FLAG_HAS_PREDICATES;
      }
      check(loom_config_decl_build(builder, flags, binding.reference,
                                   binding.type, nullptr, 0, predicates.data(),
                                   predicates.size(),
                                   locations_.get(binding.owner), &op));
    }
  }
}

std::optional<loom_value_id_t> Configs::read(cxx::VariableSymbol* variable,
                                             loom_builder_t* builder,
                                             loom_location_id_t location) {
  auto found = variables_.find(variable->canonical());
  if (found == variables_.end()) {
    return std::nullopt;
  }
  const auto& binding = bindings_[found->second];
  loom_op_t* op;
  if (binding.value) {
    check(loom_scalar_constant_build(builder, *binding.value, binding.type,
                                     location, &op));
  } else {
    check(loom_config_get_build(builder, binding.reference, binding.type,
                                location, &op));
  }
  return loom_op_results(op)[0];
}

}  // namespace loom::cxx_import
