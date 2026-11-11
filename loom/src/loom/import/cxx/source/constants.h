// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_CONSTANTS_H_
#define LOOM_IMPORT_CXX_SOURCE_CONSTANTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/cxx_fwd.h>
#include <cxx/literals_fwd.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::cxx_import {

// Exact source-ABI object bytes, independent of frontend constant storage.
struct ObjectRepresentation {
  // Complete initialized byte extent, including terminators and zero fill.
  std::vector<uint8_t> bytes;
  // A one-byte string initializer whose complete representation is valid UTF8.
  bool is_utf8;
};

// Serializes a fully initialized, trivially copyable, address-free object.
// The optional string origin preserves text form without inferring it for
// arbitrary character arrays or changing any source code units.
std::optional<ObjectRepresentation> object_representation(
    cxx::TranslationUnit& unit, const cxx::Type* type,
    const cxx::ConstValue& value, const cxx::StringLiteral* string = nullptr);

// Evaluates a pure scalar constant, retaining the frontend's typed value.
// Calls, mutation, overloaded operations, and volatile/runtime reads reject.
// Unsigned payloads retain all bits, including values above INTMAX_MAX.
std::optional<cxx::ConstValue> scalar_constant(cxx::TranslationUnit& unit,
                                               cxx::ExpressionAST* expression);

// Whether an already evaluated source expression denotes null. This includes
// nullptr_t expressions, retained constexpr pointers, and converted zero
// constants. It does not establish that evaluating the expression is pure.
bool null_pointer_expression(cxx::TranslationUnit& unit,
                             cxx::ExpressionAST* expression);

// Evaluates the pure integer grammar used by source bounds. Source conversions
// and arithmetic widths are preserved by cxx. Calls, overloaded operations,
// mutation, and runtime or volatile reads are excluded even in unselected arms.
// Retained constants and unevaluated sizeof/alignof operands are leaves.
// Returns no value for unsupported expressions, invalid constant arithmetic,
// or a final value outside the signed intmax_t range.
// The owning source analysis evaluates each bound once and retains the result;
// downstream consumers do not repeat admission or evaluation.
std::optional<std::intmax_t> integer_constant(cxx::TranslationUnit& unit,
                                              cxx::ExpressionAST* expression);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_CONSTANTS_H_
