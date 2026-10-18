// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_EXPRESSIONS_H_
#define LOOM_IMPORT_CXX_SOURCE_EXPRESSIONS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

namespace loom::cxx_import {

// Returns the source expression beneath any implicit conversion sequence.
cxx::ExpressionAST* strip_implicit_casts(cxx::ExpressionAST* expression);

// Returns the semantic expression beneath implicit conversions and redundant
// parenthesized nesting.
cxx::ExpressionAST* unwrap_expression(cxx::ExpressionAST* expression);

// Whether an expression denotes the implicit object parameter, including
// transparent parentheses, conversions, and an explicit dereference of this.
bool is_current_object(cxx::ExpressionAST* expression);

// Returns the directly named function called by |call|, or null for indirect
// calls and other callable expressions.
cxx::FunctionSymbol* direct_callee(cxx::CallExpressionAST* call);

// Returns the cv-qualified implicit object type of a non-static member
// function, or null when the function has no implicit object parameter.
const cxx::Type* implicit_object_type(cxx::TranslationUnit& unit,
                                      cxx::FunctionSymbol* function);

// Whether a call may replace value components of its implicit object. Const
// functions only qualify when their record declares mutable fields.
bool implicit_object_may_mutate(cxx::FunctionSymbol* function);

// Returns the declaration syntax beneath a condition's contextual conversion,
// or null when the condition is an ordinary expression.
cxx::ConditionExpressionAST* condition_declaration(
    cxx::ExpressionAST* expression);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_EXPRESSIONS_H_
