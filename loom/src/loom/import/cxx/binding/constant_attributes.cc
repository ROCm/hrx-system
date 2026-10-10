// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/constant_attributes.h"

#include <cxx/ast_interpreter.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <limits>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/source.h"
#include "loom/ir/module.h"

namespace loom::cxx_import {

std::optional<std::string_view> constant_string(
    const cxx::ConstValue& source_value) {
  if (auto* literal = std::get_if<const cxx::StringLiteral*>(&source_value)) {
    return *literal ? std::optional((*literal)->stringValue()) : std::nullopt;
  }
  auto* address =
      std::get_if<std::shared_ptr<cxx::ConstAddress>>(&source_value);
  if (!address || !*address || (*address)->offset() != 0 ||
      !(*address)->stringLiteral()) {
    return std::nullopt;
  }
  return (*address)->stringLiteral()->stringValue();
}

std::optional<std::string> constant_enum_name(
    cxx::TranslationUnit& unit, const cxx::Type* source_type,
    const cxx::ConstValue& source_value) {
  auto* type = unit.typeTraits().remove_cv(source_type);
  cxx::ScopeSymbol* scope = nullptr;
  if (auto* enumeration = cxx::type_cast<cxx::EnumType>(type)) {
    scope = enumeration->symbol();
  } else if (auto* enumeration = cxx::type_cast<cxx::ScopedEnumType>(type)) {
    scope = enumeration->symbol();
  }
  if (!scope) {
    return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit);
  auto integer = interpreter.toInt(source_value);
  std::optional<std::string> selected;
  for (auto* member : scope->members()) {
    auto* enumerator = cxx::symbol_cast<cxx::EnumeratorSymbol>(member);
    if (!enumerator || !enumerator->value() || !integer ||
        interpreter.toInt(*enumerator->value()) != integer) {
      continue;
    }
    auto spelling = cxx::to_string(enumerator->name());
    if (selected && *selected != spelling) {
      return std::nullopt;
    }
    selected = spelling;
  }
  return selected;
}

ConstantAttributeResult decode_constant_attribute(
    cxx::TranslationUnit& unit, loom_module_t* module,
    const loom_attr_descriptor_t& descriptor, const cxx::Type* source_type,
    const cxx::ConstValue& source_value) {
  auto* type = unit.typeTraits().remove_cv(source_type);
  auto traits = unit.typeTraits();
  cxx::ASTInterpreter interpreter(&unit);
  if (descriptor.attr_kind == LOOM_ATTR_BOOL &&
      type->kind() == cxx::TypeKind::kBool) {
    if (auto flag = interpreter.toBool(source_value)) {
      return {.value = loom_attr_bool(*flag)};
    }
  } else if (descriptor.attr_kind == LOOM_ATTR_I64 &&
             traits.is_integral(type) && type->kind() != cxx::TypeKind::kBool) {
    if (traits.is_unsigned(type)) {
      auto integer = interpreter.toUInt(source_value);
      if (integer && *integer <= std::numeric_limits<int64_t>::max()) {
        return {.value = loom_attr_i64(static_cast<int64_t>(*integer))};
      }
    } else if (auto integer = interpreter.toInt(source_value)) {
      return {.value = loom_attr_i64(*integer)};
    }
  } else if (descriptor.attr_kind == LOOM_ATTR_ENUM) {
    if (auto keyword = constant_string(source_value)) {
      uint8_t value;
      if (!loom_attr_descriptor_find_enum_case(&descriptor, view(*keyword),
                                               &value)) {
        return {.error = ConstantAttributeError::EnumKeyword};
      }
      return {.value = loom_attr_enum(value)};
    }
    cxx::ScopeSymbol* scope = nullptr;
    if (auto* enumeration = cxx::type_cast<cxx::EnumType>(type)) {
      scope = enumeration->symbol();
    } else if (auto* enumeration = cxx::type_cast<cxx::ScopedEnumType>(type)) {
      scope = enumeration->symbol();
    }
    auto integer = interpreter.toInt(source_value);
    std::optional<uint8_t> selected;
    if (scope && integer) {
      for (auto* member : scope->members()) {
        auto* enumerator = cxx::symbol_cast<cxx::EnumeratorSymbol>(member);
        if (!enumerator || !enumerator->value() ||
            interpreter.toInt(*enumerator->value()) != integer) {
          continue;
        }
        uint8_t candidate;
        if (!loom_attr_descriptor_find_enum_case(
                &descriptor, view(cxx::to_string(enumerator->name())),
                &candidate) ||
            (selected && *selected != candidate)) {
          return {.error = ConstantAttributeError::EnumKeyword};
        }
        selected = candidate;
      }
    }
    return selected
               ? ConstantAttributeResult{.value = loom_attr_enum(*selected)}
               : ConstantAttributeResult{
                     .error = ConstantAttributeError::SourceType,
                 };
  } else if (descriptor.attr_kind == LOOM_ATTR_STRING) {
    auto string = constant_string(source_value);
    if (!string) {
      return {.error = ConstantAttributeError::SourceType};
    }
    loom_string_id_t id;
    check(loom_module_intern_string(module, view(*string), &id));
    return {.value = loom_attr_string(id)};
  } else {
    return {.error = ConstantAttributeError::UnsupportedKind};
  }
  return {.error = ConstantAttributeError::SourceType};
}

}  // namespace loom::cxx_import
