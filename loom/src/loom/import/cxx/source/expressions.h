// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_EXPRESSIONS_H_
#define LOOM_IMPORT_CXX_SOURCE_EXPRESSIONS_H_

#include <cxx/ast_fwd.h>
#include <cxx/literals_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <array>
#include <cstddef>
#include <optional>

namespace loom::cxx_import {

// Returns the source expression beneath any implicit conversion sequence.
cxx::ExpressionAST* strip_implicit_casts(cxx::ExpressionAST* expression);

// Returns the semantic expression beneath implicit conversions, redundant
// parentheses, and frontend wrappers around selected member initializers.
cxx::ExpressionAST* unwrap_expression(cxx::ExpressionAST* expression);

// Returns a frontend-selected constructor conversion at the root of an
// initializer, including transparent source initializer wrappers. Constructor
// arguments are retained on the cast's expression in resolved parameter order.
cxx::ImplicitCastExpressionAST* constructor_conversion(
    cxx::ExpressionAST* initializer);

// Whether two resolved source functions denote the same canonical entity.
bool same_function(cxx::FunctionSymbol* left, cxx::FunctionSymbol* right);

// Returns the string literal initializing one object, including a single
// braced clause and transparent expression wrappers, or null otherwise.
const cxx::StringLiteral* string_initializer(cxx::ExpressionAST* initializer);

// Whether an expression denotes the implicit object parameter, including
// transparent parentheses, conversions, and an explicit dereference of this.
bool is_current_object(cxx::ExpressionAST* expression);

// Returns the directly named function called by |call|, or null for indirect
// calls and other callable expressions.
cxx::FunctionSymbol* direct_callee(cxx::CallExpressionAST* call);

// Evaluation sequence imposed by the source operator notation. Forward visits
// the receiver followed by explicit arguments, or only the explicit arguments
// when no receiver exists. Compound assignment is the only admitted form that
// evaluates in another order.
enum class OperatorEvaluationOrder {
  Forward,
  ArgumentsThenReceiver,
  ReverseArguments,
};

// One ordinary function selected by operator syntax. The frontend has already
// performed overload resolution and inserted conversions into each retained
// expression. The receiver and explicit arguments describe the callable ABI;
// the evaluation order preserves the distinct sequencing rules of operator
// notation without requiring later users to classify the AST again.
struct DirectOperatorCall {
  // Concrete function selected by source overload resolution.
  cxx::FunctionSymbol* function;
  // Implicit object expression for a non-static member, or null otherwise.
  cxx::ExpressionAST* receiver;
  // Explicit parameter expressions in declaration order.
  std::array<cxx::ExpressionAST*, 2> arguments;
  // Number of populated explicit argument expressions.
  size_t argument_count;
  // Source evaluation sequence, independently of callable ABI order.
  OperatorEvaluationOrder evaluation_order;
  // Whether a postfix increment/decrement contributes an implicit zero.
  bool has_postfix_dummy;
  // Whether overload resolution selected virtual member dispatch.
  bool is_virtual_dispatch;
};

// Returns the directly selected ordinary function and source operand contract
// for unary, binary, compound-assignment, and postfix operator syntax. Builtin
// operators and call/subscript syntax return no value.
std::optional<DirectOperatorCall> direct_operator_call(
    cxx::ExpressionAST* expression);

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
