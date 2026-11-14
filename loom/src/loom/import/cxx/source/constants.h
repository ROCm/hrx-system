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

class Diagnostics;

// Exact source-ABI object bytes, independent of frontend constant storage.
struct ObjectRepresentation {
  // Complete initialized byte extent, including terminators and zero fill.
  std::vector<uint8_t> bytes;
  // A one-byte string initializer whose complete representation is valid UTF8.
  bool is_utf8;
};

// Whether a retained constant contains source storage identity at any depth.
// Such a value cannot be flattened into ordinary Loom SSA constants; its
// source expression must establish the referenced storage at runtime.
bool constant_contains_address(const cxx::ConstValue& value);

// Returns a constexpr variable's address-free frontend value when its
// initializer can be materialized without any runtime evaluation. Constants
// containing source storage identity retain their source initialization path.
const cxx::ConstValue* folded_variable_constant(cxx::VariableSymbol* variable);

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

// Evaluates the infinity, huge-value, and quiet-NaN builtins in their source
// types. NaN payloads must be constant strings; evaluation failures and
// unsupported signaling NaNs diagnose at the call. Other builtin families
// return no value without attempting evaluation.
std::optional<cxx::ConstValue> floating_builtin_constant(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::CallExpressionAST* call);

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
