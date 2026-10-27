// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/target_definitions.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/const_value.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "loom/import/cxx/binding/constant_attributes.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/source.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {
namespace {

cxx::ClassSymbol* class_definition(cxx::TranslationUnit& unit,
                                   const cxx::Type* type) {
  auto* unqualified = unit.typeTraits().remove_cv(type);
  auto* class_type = cxx::type_cast<cxx::ClassType>(unqualified);
  return class_type ? class_type->definition() : nullptr;
}

bool is_target_optional(cxx::TranslationUnit& unit, const cxx::Type* type) {
  auto* definition = class_definition(unit, type);
  if (!definition) {
    return false;
  }
  auto* primary = cxx::class_template_of(definition);
  if (!primary || !primary->name() ||
      cxx::to_string(primary->name()) != "optional") {
    return false;
  }
  auto* target_namespace =
      cxx::symbol_cast<cxx::NamespaceSymbol>(primary->parent());
  auto* loom_namespace =
      target_namespace
          ? cxx::symbol_cast<cxx::NamespaceSymbol>(target_namespace->parent())
          : nullptr;
  return target_namespace && target_namespace->name() && loom_namespace &&
         loom_namespace->name() &&
         cxx::to_string(target_namespace->name()) == "target" &&
         cxx::to_string(loom_namespace->name()) == "loom" &&
         (!loom_namespace->parent() || !loom_namespace->parent()->name());
}

const cxx::FieldSymbol* field_symbol(const cxx::ConstObject::Member& member) {
  return member.symbol && member.symbol->kind() == cxx::SymbolKind::kField
             ? static_cast<const cxx::FieldSymbol*>(member.symbol)
             : nullptr;
}

struct PresentValue {
  const cxx::Type* type;
  const cxx::ConstValue* value;
};

std::optional<PresentValue> unwrap_optional(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const loom_attr_descriptor_t& descriptor, std::string_view field_name,
    const cxx::Type* type, const cxx::ConstValue& value, cxx::AST* owner) {
  if (!is_target_optional(unit, type)) {
    return PresentValue{type, &value};
  }
  if (!iree_any_bit_set(descriptor.flags, LOOM_ATTR_OPTIONAL)) {
    diagnostics.reject(unit, owner,
                       "required target field '" + std::string(field_name) +
                           "' cannot use loom::target::optional");
  }
  auto* object = std::get_if<std::shared_ptr<cxx::ConstObject>>(&value);
  if (!object || !*object || (*object)->members().size() != 2) {
    diagnostics.reject(unit, owner,
                       "loom::target::optional has an invalid constant "
                       "representation");
  }
  const cxx::ConstObject::Member* present_member = nullptr;
  const cxx::ConstObject::Member* value_member = nullptr;
  const cxx::FieldSymbol* value_field = nullptr;
  for (const auto& member : (*object)->members()) {
    auto* field = field_symbol(member);
    auto name = field && field->name() ? cxx::to_string(field->name()) : "";
    if (name == "present_") {
      present_member = &member;
    } else if (name == "value_") {
      value_member = &member;
      value_field = field;
    }
  }
  if (!present_member || !value_member || !value_field) {
    diagnostics.reject(unit, owner,
                       "loom::target::optional has an invalid constant "
                       "representation");
  }
  cxx::ASTInterpreter interpreter(&unit);
  auto present = interpreter.toBool(present_member->value);
  if (!present) {
    diagnostics.reject(unit, owner,
                       "loom::target::optional presence must be a constant "
                       "boolean");
  }
  if (!*present) {
    return std::nullopt;
  }
  return PresentValue{value_field->type(), &value_member->value};
}

loom_attribute_t decode_signed_enum_set(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const loom_attr_descriptor_t& descriptor, const cxx::Type* type,
    const cxx::ConstValue& value, cxx::AST* owner, loom_builder_t* builder) {
  auto* object = std::get_if<std::shared_ptr<cxx::ConstObject>>(&value);
  if (!object || !*object) {
    diagnostics.reject(unit, owner,
                       "signed target fields require a constant aggregate");
  }
  std::array<uint64_t, LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT * 2> words = {};
  bool has_assertion = false;
  for (const auto& member : (*object)->members()) {
    auto* field = field_symbol(member);
    if (!field || !field->name()) {
      diagnostics.reject(unit, owner,
                         "signed target fields require named members");
    }
    uint8_t value_ordinal;
    auto field_name = cxx::to_string(field->name());
    if (!loom_attr_descriptor_find_enum_case(&descriptor, view(field_name),
                                             &value_ordinal)) {
      diagnostics.reject(unit, owner,
                         "unknown signed target field '" + field_name + "'");
    }
    auto state = constant_enum_name(unit, field->type(), member.value);
    if (!state || (*state != "unspecified" && *state != "enabled" &&
                   *state != "disabled")) {
      diagnostics.reject(
          unit, owner,
          "signed target fields require unspecified, enabled, or disabled");
    }
    if (*state == "unspecified") {
      continue;
    }
    const size_t word_index = value_ordinal / 64u;
    const size_t polarity = *state == "disabled" ? 1 : 0;
    words[polarity * LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT + word_index] |=
        UINT64_C(1) << (value_ordinal % 64u);
    has_assertion = true;
  }
  if (!has_assertion) {
    return loom_attr_absent();
  }
  size_t word_count = descriptor.enum_max_value / 64u + 1;
  std::array<uint64_t, LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT * 2> packed = {};
  for (size_t i = 0; i < word_count; ++i) {
    packed[i] = words[i];
    packed[word_count + i] = words[LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT + i];
  }
  const uint64_t* storage;
  uint16_t stored_word_count;
  check(loom_builder_copy_signed_enum_set_attr_storage(
      builder, loom_make_signed_enum_set(packed.data(), word_count),
      loom_attr_descriptor_name(&descriptor), &storage, &stored_word_count));
  return loom_attr_signed_enum_set(storage, stored_word_count);
}

}  // namespace

bool TargetDefinitions::declaration(
    cxx::Symbol* symbol, cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    cxx::AST* owner) {
  if (auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(symbol)) {
    admit_function(function, attributes, owner);
    return false;
  }
  if (cxx::ast_cast<cxx::ClassSpecifierAST>(owner)) {
    cxx::AttributeAST* selected = nullptr;
    visit_loom_attributes(
        unit_, attributes,
        [&](std::string_view name, cxx::AttributeAST* attribute) {
          if (name != "target") {
            return;
          }
          if (selected) {
            diagnostics_.reject(
                unit_, attribute,
                "one target schema binding is allowed per type");
          }
          selected = attribute;
          auto* clause = attribute->attributeArgumentClause;
          auto* arguments = clause ? clause->expressionList : nullptr;
          auto* literal = arguments && !arguments->next
                              ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(
                                    arguments->value)
                              : nullptr;
          if (!literal || literal->literal->stringValue().empty()) {
            diagnostics_.reject(
                unit_, attribute,
                "target facade type requires one target operation name");
          }
        });
    return false;
  }
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name == "target") {
          diagnostics_.reject(
              unit_, attribute,
              "target bindings require concrete namespace-scope functions");
        }
      });
  auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(symbol);
  return variable ? admit_definition(variable, owner) : false;
}

void TargetDefinitions::admit_function(
    cxx::FunctionSymbol* function,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes, cxx::AST* owner) {
  cxx::AttributeAST* selected = nullptr;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "target") {
          return;
        }
        if (selected) {
          diagnostics_.reject(unit_, attribute,
                              "one target binding is allowed per declaration");
        }
        selected = attribute;
      });
  if (!selected) {
    return;
  }
  const bool emits_non_callable_op = annotated(function, "op") ||
                                     annotated(function, "assume") ||
                                     annotated(function, "check_case") ||
                                     annotated(function, "check_scenario") ||
                                     annotated(function, "check_benchmark");
  if (function->isTemplatePattern() ||
      !cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent()) ||
      (emits_non_callable_op && !annotated(function, "kernel") &&
       !annotated(function, "template_def"))) {
    diagnostics_.reject(
        unit_, selected,
        "target bindings require ordinary functions, kernels, or template "
        "definitions");
  }
  auto* clause = selected->attributeArgumentClause;
  auto* arguments = clause ? clause->expressionList : nullptr;
  auto* id = arguments && !arguments->next
                 ? cxx::ast_cast<cxx::IdExpressionAST>(arguments->value)
                 : nullptr;
  auto* variable =
      id ? cxx::symbol_cast<cxx::VariableSymbol>(id->symbol) : nullptr;
  if (!variable) {
    diagnostics_.reject(unit_, selected,
                        "target binding requires one target object name");
  }
  auto [binding, inserted] = functions_.try_emplace(
      function->canonical(), FunctionBinding{variable->canonical(), selected});
  if (!inserted && binding->second.variable != variable->canonical()) {
    diagnostics_.reject(unit_, selected,
                        "conflicting target bindings across declarations");
  }
}

bool TargetDefinitions::admit_definition(cxx::VariableSymbol* variable,
                                         cxx::AST* owner) {
  auto* definition = class_definition(unit_, variable->type());
  auto* marker = annotation(definition, "target");
  if (!marker) {
    return false;
  }
  if (marker->arguments.size() != 1 || marker->arguments[0]->name().empty()) {
    diagnostics_.reject(
        unit_, owner, "target facade type requires one target operation name");
  }
  if (variable->isTemplatePattern() || variable->isThreadLocal() ||
      !variable->isConstexpr() ||
      !cxx::symbol_cast<cxx::NamespaceSymbol>(variable->parent())) {
    diagnostics_.reject(
        unit_, owner,
        "target definitions require namespace-scope constexpr objects");
  }
  auto* object = variable->constValue()
                     ? std::get_if<std::shared_ptr<cxx::ConstObject>>(
                           &*variable->constValue())
                     : nullptr;
  auto* class_type = cxx::type_cast<cxx::ClassType>(
      unit_.typeTraits().remove_cv(variable->type()));
  if (!object || !*object || !definition || definition->isUnion() ||
      !definition->baseClasses().empty() ||
      !unit_.typeTraits().is_aggregate(class_type)) {
    diagnostics_.reject(
        unit_, owner,
        "target definitions require a constant aggregate without "
        "bases or unions");
  }
  loom_op_kind_t operation_kind;
  auto operation_name = marker->arguments[0]->name();
  auto* vtable = loom_context_lookup_op_by_name(
      module_->context, view(operation_name), &operation_kind);
  constexpr uint8_t kVariableShapeFlags = LOOM_OP_VTABLE_VARIADIC_OPERANDS |
                                          LOOM_OP_VTABLE_VARIADIC_RESULTS |
                                          LOOM_OP_VTABLE_VARIADIC_REGIONS;
  if (!vtable || !vtable->target_like || !vtable->attr_descriptors ||
      vtable->fixed_operand_count || vtable->fixed_result_count ||
      vtable->region_count ||
      iree_any_bit_set(vtable->vtable_flags, kVariableShapeFlags)) {
    diagnostics_.reject(
        unit_, owner,
        "target facade requires one registered attribute-only TargetLike "
        "operation: " +
            std::string(operation_name));
  }
  auto canonical = variable->canonical();
  if (variables_.contains(canonical)) {
    diagnostics_.reject(unit_, owner,
                        "target objects require one constexpr definition");
  }
  auto name = qualified_name(variable);
  symbol_names_.reserve(name, owner);
  variables_.emplace(canonical, definitions_.size());
  definitions_.push_back({canonical, owner, operation_kind, vtable,
                          std::move(name), loom_symbol_ref_null()});
  return true;
}

std::string TargetDefinitions::qualified_name(
    cxx::VariableSymbol* variable) const {
  std::string spelling = cxx::to_string(variable->name());
  for (auto* owner : variable->enclosingSymbols()) {
    auto* space = cxx::symbol_cast<cxx::NamespaceSymbol>(owner);
    if (space && space->name()) {
      spelling = cxx::to_string(space->name()) + "." + spelling;
    }
  }
  if (!is_symbol_name(spelling)) {
    diagnostics_.reject(unit_, variable->declaration(),
                        "target object requires a valid Loom symbol name");
  }
  return spelling;
}

void TargetDefinitions::build(loom_builder_t* builder) {
  for (auto& definition : definitions_) {
    loom_string_id_t name;
    check(loom_builder_intern_string(builder, view(definition.name), &name));
    check(
        loom_module_add_symbol(module_, name, &definition.reference.symbol_id));
    definition.reference.module_id = 0;
    emit_definition(definition, builder);
  }
  for (const auto& [function, binding] : functions_) {
    if (!variables_.contains(binding.variable)) {
      diagnostics_.reject(unit_, binding.owner,
                          "target binding must name a constexpr target object");
    }
  }
}

void TargetDefinitions::emit_definition(Definition& definition,
                                        loom_builder_t* builder) {
  auto object = std::get<std::shared_ptr<cxx::ConstObject>>(
      *definition.variable->constValue());
  std::vector<loom_attribute_t> attributes(definition.vtable->attribute_count,
                                           loom_attr_absent());
  const auto* target = definition.vtable->target_like;
  attributes[target->symbol_attr_index] =
      loom_attr_symbol(definition.reference);
  for (const auto& member : object->members()) {
    auto* field = field_symbol(member);
    if (!field || !field->name()) {
      diagnostics_.reject(unit_, definition.owner,
                          "target fields require names");
    }
    auto field_name = cxx::to_string(field->name());
    uint8_t attribute_index;
    auto* descriptor = loom_attr_descriptor_find_by_name(
        definition.vtable->attr_descriptors, definition.vtable->attribute_count,
        view(field_name), &attribute_index);
    if (!descriptor || attribute_index == target->symbol_attr_index) {
      diagnostics_.reject(unit_, definition.owner,
                          "unknown target field '" + field_name + "'");
    }
    auto present =
        unwrap_optional(unit_, diagnostics_, *descriptor, field_name,
                        field->type(), member.value, definition.owner);
    if (!present) {
      continue;
    }
    loom_attribute_t attribute;
    if (descriptor->attr_kind == LOOM_ATTR_SIGNED_ENUM_SET) {
      attribute = decode_signed_enum_set(unit_, diagnostics_, *descriptor,
                                         present->type, *present->value,
                                         definition.owner, builder);
      if (loom_attr_is_absent(attribute)) {
        continue;
      }
    } else {
      auto decoded = decode_constant_attribute(unit_, module_, *descriptor,
                                               present->type, *present->value);
      if (!decoded) {
        std::string message = "target field '" + field_name + "' ";
        if (decoded.error == ConstantAttributeError::EnumKeyword) {
          message += "does not name a declared target keyword";
        } else if (decoded.error == ConstantAttributeError::UnsupportedKind) {
          message += "uses an unsupported target attribute kind";
        } else {
          message +=
              "requires a matching integer, boolean, enum, or string "
              "constant";
        }
        diagnostics_.reject(unit_, definition.owner, message);
      }
      attribute = decoded.value;
    }
    attributes[attribute_index] = attribute;
  }
  if (loom_attr_is_absent(attributes[target->selector_attr_index])) {
    diagnostics_.reject(unit_, definition.owner,
                        "target definition requires its selector field");
  }
  loom_op_t* op;
  check(loom_builder_allocate_op(builder, definition.operation_kind,
                                 /*operand_count=*/0,
                                 /*result_count=*/0,
                                 /*region_count=*/0, /*tied_result_count=*/0,
                                 definition.vtable->attribute_count,
                                 locations_.get(definition.owner), &op));
  std::copy(attributes.begin(), attributes.end(), loom_op_attrs(op));
  check(loom_builder_finalize_op(builder, op));
}

loom_symbol_ref_t TargetDefinitions::reference(
    cxx::FunctionSymbol* function) const {
  auto binding = functions_.find(function->canonical());
  if (binding == functions_.end()) {
    return loom_symbol_ref_null();
  }
  auto definition = variables_.find(binding->second.variable);
  IREE_ASSERT(definition != variables_.end());
  return definitions_[definition->second].reference;
}

void TargetDefinitions::reject_launch_configuration_function(
    cxx::FunctionSymbol* function) const {
  auto binding = functions_.find(function->canonical());
  if (binding != functions_.end()) {
    diagnostics_.reject(
        unit_, binding->second.owner,
        "kernel launch configuration functions cannot carry target bindings");
  }
}

}  // namespace loom::cxx_import
