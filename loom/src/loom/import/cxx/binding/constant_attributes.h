// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_CONSTANT_ATTRIBUTES_H_
#define LOOM_IMPORT_CXX_BINDING_CONSTANT_ATTRIBUTES_H_

#include <cxx/const_value.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <string>
#include <string_view>

#include "loom/ir/attribute_schema.h"

namespace cxx {
class TranslationUnit;
}

namespace loom::cxx_import {

// Why a source constant could not inhabit one descriptor-backed scalar field.
enum class ConstantAttributeError {
  None,
  SourceType,
  EnumKeyword,
  UnsupportedKind,
};

// Result of projecting one source constant through a registered IR field
// descriptor. Successful values own no frontend storage; strings are interned
// into |module| before return.
struct ConstantAttributeResult {
  loom_attribute_t value = loom_attr_absent();
  ConstantAttributeError error = ConstantAttributeError::None;

  explicit operator bool() const {
    return error == ConstantAttributeError::None;
  }
};

// Decodes integer, boolean, enum, and string fields. Enum values may be named
// C++ enumerators or exact string keywords; C++ enum ordinals never become IR
// ordinals. Other descriptor kinds report UnsupportedKind.
ConstantAttributeResult decode_constant_attribute(
    cxx::TranslationUnit& unit, loom_module_t* module,
    const loom_attr_descriptor_t& descriptor, const cxx::Type* source_type,
    const cxx::ConstValue& source_value);

// Returns an exact zero-offset string literal payload, including a literal
// reached through retained array-decay steps or a constexpr pointer. Other
// addresses and values return empty.
std::optional<std::string_view> constant_string(
    const cxx::ConstValue& source_value);

// Returns the unique source enumerator spelling selected by |source_value|.
// Numeric values without one exact declared spelling return empty.
std::optional<std::string> constant_enum_name(
    cxx::TranslationUnit& unit, const cxx::Type* source_type,
    const cxx::ConstValue& source_value);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_CONSTANT_ATTRIBUTES_H_
