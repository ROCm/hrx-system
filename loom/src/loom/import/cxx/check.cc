// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/check.h"

#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"

namespace loom::cxx_import {
namespace {

class CheckBody {
 public:
  CheckBody(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
            Functions& functions, Intrinsics& intrinsics, Types& types,
            Scalars& scalars, Locations& locations, loom_builder_t& builder)
      : unit_(unit),
        diagnostics_(diagnostics),
        functions_(functions),
        intrinsics_(intrinsics),
        types_(types),
        scalars_(scalars),
        locations_(locations),
        builder_(builder) {}

  void translate(const FunctionBody& body) {
    cxx::AST* end = body.body;
    for (auto* remaining = body.body->statementList; remaining;
         remaining = remaining->next) {
      auto* statement = remaining->value;
      if (auto* returned = cxx::ast_cast<cxx::ReturnStatementAST>(statement)) {
        if (returned->expression || remaining->next) {
          fail(statement, "check case return must be bare and final");
        }
        end = returned;
        break;
      }
      if (auto* declaration =
              cxx::ast_cast<cxx::DeclarationStatementAST>(statement)) {
        if (observing_) {
          fail(statement, "check expectations must be terminal");
        }
        local(declaration);
        continue;
      }
      auto* expression_statement =
          cxx::ast_cast<cxx::ExpressionStatementAST>(statement);
      if (!expression_statement) {
        fail(statement,
             "check cases support immutable scalars, direct calls, and "
             "terminal expectations");
      }
      if (!expression_statement->expression) {
        continue;
      }
      auto* call = cxx::ast_cast<cxx::CallExpressionAST>(
          expression_statement->expression);
      if (!call) {
        fail(statement, "check case statements must be direct calls");
      }
      auto* function = callee(call);
      if (intrinsics_.expectation_type(function)) {
        std::vector<loom_value_id_t> arguments;
        for (auto* argument : cxx::ListView{call->expressionList}) {
          arguments.push_back(expression(argument));
        }
        loom_op_t* op;
        check(loom_check_expect_equal_build(
            &builder_, arguments[0], arguments[1], locations_.get(call), &op));
        observing_ = true;
      } else {
        invocation(call, function);
      }
    }
    loom_op_t* terminator;
    check(loom_check_return_build(&builder_, locations_.get(end), &terminator));
  }

 private:
  [[noreturn]] void fail(cxx::AST* source, const char* message) {
    diagnostics_.reject(unit_, source, message);
  }

  loom_type_t scalar_type(const cxx::Type* type, cxx::AST* source) {
    auto result = types_.get(type, source);
    if (loom_type_kind(result) != LOOM_TYPE_SCALAR ||
        unit_.typeTraits().is_volatile(type)) {
      fail(source, "check values require non-volatile scalar types");
    }
    return result;
  }

  loom_value_id_t literal(const cxx::ConstValue& value,
                          cxx::ExpressionAST* source) {
    auto type = scalar_type(source->type, source);
    loom_op_t* op;
    check(loom_check_literal_build(
        &builder_, scalars_.constant_attribute(value, source->type, source),
        type, locations_.get(source), &op));
    return loom_op_results(op)[0];
  }

  cxx::FunctionSymbol* callee(cxx::CallExpressionAST* call) {
    auto* id = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
    auto* function =
        id ? cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol) : nullptr;
    if (!function) {
      fail(call, "check calls must resolve to a function declaration");
    }
    return function;
  }

  loom_op_t* invocation(cxx::CallExpressionAST* call,
                        cxx::FunctionSymbol* function) {
    if (observing_) {
      fail(call, "check expectations must be terminal");
    }
    if (!functions_.definition(function) ||
        functions_.is_check_case(function) || annotated(function, "kernel") ||
        annotated(function, "check_benchmark")) {
      fail(call, "check invocations require a defined ordinary function");
    }
    std::vector<loom_value_id_t> arguments;
    for (auto* argument : cxx::ListView{call->expressionList}) {
      arguments.push_back(expression(argument));
    }
    std::vector<loom_type_t> results;
    if (types_.unqualified(call->type)->kind() != cxx::TypeKind::kVoid) {
      results.push_back(scalar_type(call->type, call));
    }
    auto symbol = functions_.declare(function);
    loom_op_t* op;
    check(loom_func_call_build(&builder_, 0, 0, 0, 0, symbol, arguments.data(),
                               arguments.size(), results.data(), results.size(),
                               nullptr, 0, locations_.get(call), &op));
    return op;
  }

  loom_value_id_t expression(cxx::ExpressionAST* source) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
      return expression(nested->expression);
    }
    if (auto* initializer = cxx::ast_cast<cxx::EqualInitializerAST>(source)) {
      return expression(initializer->expression);
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
      return expression(initializer->expression);
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(source)) {
      return loom_op_results(invocation(call, callee(call)))[0];
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(source)) {
      auto value = values_.find(id->symbol);
      if (value != values_.end()) {
        return value->second;
      }
    }
    if (auto constant = scalar_constant(unit_, source)) {
      return literal(*constant, source);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source)) {
      if (!cast->conversionFunction) {
        auto input = scalar_type(cast->expression->type, source);
        auto output = scalar_type(cast->type, source);
        if (loom_type_equal(input, output)) {
          return expression(cast->expression);
        }
      }
      fail(source, "runtime conversions are not supported in check cases");
    }
    fail(source,
         "check values require scalar constants, immutable bindings, or direct "
         "calls");
  }

  void local(cxx::DeclarationStatementAST* statement) {
    auto* declaration =
        cxx::ast_cast<cxx::SimpleDeclarationAST>(statement->declaration);
    if (!declaration) {
      fail(statement, "check locals require immutable scalar bindings");
    }
    reject_global_binding_attributes(unit_, diagnostics_,
                                     declaration->attributeList);
    for (auto* declarator : cxx::ListView{declaration->initDeclaratorList}) {
      reject_global_binding_declarator(unit_, diagnostics_,
                                       declarator->declarator);
      auto* variable =
          cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
      if (!variable || !declarator->initializer || variable->isStatic() ||
          variable->isExtern() || variable->isThreadLocal() ||
          !unit_.typeTraits().is_const(variable->type())) {
        fail(declarator,
             "check locals require initialized automatic const scalars");
      }
      scalar_type(variable->type(), declarator);
      auto value = expression(declarator->initializer);
      values_[variable] = value;
      if (loom_module_value(builder_.module, value)->name_id ==
          LOOM_STRING_ID_INVALID) {
        loom_string_id_t name;
        check(loom_module_intern_string(
            builder_.module, view(cxx::to_string(variable->name())), &name));
        check(loom_module_set_value_name(builder_.module, value, name));
      }
    }
  }

  // Source semantic types and resolved declarations, borrowed for this body.
  cxx::TranslationUnit& unit_;
  // Source rejection boundary for unsupported harness semantics.
  Diagnostics& diagnostics_;
  // Shared symbol owner retaining direct callees and pending definitions.
  Functions& functions_;
  // Admitted declaration bindings, including equality expectations.
  Intrinsics& intrinsics_;
  // Source scalar representation admission.
  Types& types_;
  // Shared constant payload encoding with ordinary functions.
  Scalars& scalars_;
  // Source locations retained in the output module.
  Locations& locations_;
  // Caller-owned insertion point within the check case.
  loom_builder_t& builder_;
  // Immutable source locals mapped to native SSA values.
  std::unordered_map<cxx::Symbol*, loom_value_id_t> values_;
  // The first expectation closes the invocation stage of this case.
  bool observing_ = false;
};

}  // namespace

void translate_check_body(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          Functions& functions, Intrinsics& intrinsics,
                          Types& types, Scalars& scalars, Locations& locations,
                          loom_builder_t& builder, const FunctionBody& body) {
  CheckBody(unit, diagnostics, functions, intrinsics, types, scalars, locations,
            builder)
      .translate(body);
}

}  // namespace loom::cxx_import
