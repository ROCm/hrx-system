// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/check.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"

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
    reject_misplaced_binding_statement(unit_, diagnostics_, body.body);
    cxx::AST* end = body.body;
    for (auto* remaining = body.body->statementList; remaining;
         remaining = remaining->next) {
      auto* statement = remaining->value;
      reject_misplaced_binding_statement(unit_, diagnostics_, statement);
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
             "check cases support immutable values, direct calls, and "
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
      if (auto* binding = intrinsics_.check_binding(function, call)) {
        check_call(call, *binding);
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

  loom_string_id_t intern(std::string_view text) {
    loom_string_id_t result;
    check(loom_module_intern_string(builder_.module, view(text), &result));
    return result;
  }

  std::string_view string_literal(cxx::ExpressionAST* source) {
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source)) {
      if (!cast->conversionFunction) {
        return string_literal(cast->expression);
      }
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
      return string_literal(nested->expression);
    }
    auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(source);
    if (!literal) {
      fail(source, "check metadata requires a string literal");
    }
    return literal->literal->stringValue();
  }

  loom_attribute_t constant(cxx::ExpressionAST* source) {
    if (auto value = scalar_constant(unit_, source)) {
      return scalars_.constant_attribute(*value, source->type, source);
    }
    fail(source, "check generation and metadata require pure scalar constants");
  }

  void metadata(cxx::CallExpressionAST* call, const CheckIntrinsic& binding) {
    auto* argument = call->expressionList;
    auto provider = intern(string_literal(argument->value));
    std::vector<loom_named_attr_t> attributes;
    std::unordered_set<loom_string_id_t> names;
    for (argument = argument->next; argument; argument = argument->next->next) {
      auto name = intern(string_literal(argument->value));
      if (!names.insert(name).second) {
        fail(argument->value, "duplicate check metadata attribute");
      }
      auto* source = argument->next->value;
      loom_attribute_t value;
      if (unit_.typeTraits().is_pointer(source->type)) {
        value = loom_attr_string(intern(string_literal(source)));
      } else {
        value = constant(source);
        if (types_.unqualified(source->type)->kind() == cxx::TypeKind::kBool) {
          value = loom_attr_bool(loom_attr_as_i64(value) != 0);
        }
      }
      attributes.push_back({name, 0, value});
    }
    loom_named_attr_slice_t attrs{attributes.data(), attributes.size()};
    loom_op_t* op;
    if (binding.operation == CheckIntrinsic::Operation::Requires) {
      check(loom_check_requires_build(&builder_, provider, attrs,
                                      locations_.get(call), &op));
    } else {
      check(loom_check_expect_event_build(
          &builder_, LOOM_CHECK_EXPECT_EVENT_BUILD_FLAG_HAS_ATTRS, provider,
          attrs, locations_.get(call), &op));
    }
  }

  std::optional<Value> check_call(cxx::CallExpressionAST* call,
                                  const CheckIntrinsic& binding) {
    using Operation = CheckIntrinsic::Operation;
    auto* first = call->expressionList;
    auto location = locations_.get(call);
    loom_op_t* op;
    switch (binding.operation) {
      case Operation::Fill: {
        check(loom_check_generate_fill_build(&builder_, constant(first->value),
                                             binding.result_tensor->type,
                                             location, &op));
        return value_arena_.capture(*binding.result_tensor,
                                    {loom_op_results(op), 1});
      }
      case Operation::Slice: {
        auto source = expression(first->value);
        auto offset = integer_constant(unit_, first->next->value);
        auto source_count =
            loom_type_dim_static_size_at(binding.source_tensor->type, 0);
        auto result_count =
            loom_type_dim_static_size_at(binding.result_tensor->type, 0);
        if (!offset || *offset < 0 || result_count > source_count ||
            *offset > source_count - result_count) {
          fail(call,
               "check tensor slice must select an in-bounds constant element "
               "range");
        }
        check(loom_check_tensor_view_build(
            &builder_, source.components()[0],
            *offset * binding.source_tensor->element_bytes,
            binding.result_tensor->type, location, &op));
        return value_arena_.capture(*binding.result_tensor,
                                    {loom_op_results(op), 1});
      }
      case Operation::Equal:
      case Operation::Bitwise: {
        auto actual = expression(first->value);
        auto expected = expression(first->next->value);
        auto build = binding.operation == Operation::Equal
                         ? loom_check_expect_equal_build
                         : loom_check_expect_bitwise_build;
        check(build(&builder_, actual.components()[0], expected.components()[0],
                    location, &op));
        break;
      }
      case Operation::Requires:
      case Operation::Event:
        metadata(call, binding);
        break;
      case Operation::Launch: {
        if (observing_) {
          fail(call, "check expectations must be terminal");
        }
        if (!functions_.definition(binding.kernel)) {
          fail(call, "check launches require a defined kernel");
        }
        std::vector<loom_value_id_t> arguments;
        for (auto* argument : cxx::ListView{call->expressionList}) {
          expression(argument).append_to(arguments);
        }
        auto symbol = functions_.declare(binding.kernel);
        check(loom_kernel_launch_build(&builder_, symbol, nullptr, 0,
                                       arguments.data(), arguments.size(),
                                       location, &op));
        break;
      }
    }
    observing_ |= binding.is_observation();
    return std::nullopt;
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
      auto value = expression(argument);
      for (auto component : value.components()) {
        if (loom_type_kind(loom_module_value_type(
                builder_.module, component)) != LOOM_TYPE_SCALAR) {
          fail(
              argument,
              "ordinary check calls require scalar or scalar-record arguments");
        }
        arguments.push_back(component);
      }
    }
    std::vector<loom_type_t> results;
    if (types_.unqualified(call->type)->kind() != cxx::TypeKind::kVoid) {
      types_.append(call->type, call, results);
      for (auto type : results) {
        if (loom_type_kind(type) != LOOM_TYPE_SCALAR) {
          fail(call,
               "check call results require scalars or records of scalars");
        }
      }
    }
    auto symbol = functions_.declare(function);
    loom_op_t* op;
    check(loom_func_call_build(&builder_, 0, 0, 0, 0, symbol, arguments.data(),
                               arguments.size(), results.data(), results.size(),
                               nullptr, 0, locations_.get(call), &op));
    return op;
  }

  Value expression(cxx::ExpressionAST* source) {
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
    if (auto* initializer = cxx::ast_cast<cxx::ParenInitializerAST>(source)) {
      if (initializer->expressionList && !initializer->expressionList->next &&
          types_.unqualified(source->type) ==
              types_.unqualified(initializer->expressionList->value->type)) {
        return expression(initializer->expressionList->value);
      }
    }
    if (auto* initializer = cxx::ast_cast<cxx::BracedInitListAST>(source)) {
      if (initializer->expressionList && !initializer->expressionList->next &&
          types_.unqualified(source->type) ==
              types_.unqualified(initializer->expressionList->value->type)) {
        return expression(initializer->expressionList->value);
      }
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(source)) {
      auto* function = callee(call);
      if (auto* binding = intrinsics_.check_binding(function, call)) {
        auto value = check_call(call, *binding);
        if (!value) {
          fail(call, "void check operation cannot be used as a value");
        }
        return *value;
      }
      auto* op = invocation(call, function);
      return value_arena_.capture(types_.partition(call->type, call),
                                  {loom_op_results(op), op->result_count});
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
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(source)) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (!field || field->isStatic()) {
        fail(source, "record value access requires a non-static data member");
      }
      auto value = expression(member->baseExpression);
      const auto& slice = types_.member(field, source);
      return value.project(*slice.partition, slice.component_offset);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source)) {
      if (cast->conversionFunction) {
        types_.admit_copy(cast->conversionFunction, cast->type, source);
      }
      if (types_.unqualified(cast->expression->type) ==
          types_.unqualified(cast->type)) {
        return expression(cast->expression);
      }
      auto input = scalar_type(cast->expression->type, source);
      auto output = scalar_type(cast->type, source);
      if (loom_type_equal(input, output)) {
        return expression(cast->expression);
      }
      fail(source, "runtime conversions are not supported in check cases");
    }
    fail(source,
         "check values require scalar constants, immutable bindings, record "
         "members, or direct calls");
  }

  void local(cxx::DeclarationStatementAST* statement) {
    auto* declaration =
        cxx::ast_cast<cxx::SimpleDeclarationAST>(statement->declaration);
    if (!declaration) {
      fail(statement, "check locals require immutable value bindings");
    }
    reject_misplaced_binding_attributes(unit_, diagnostics_,
                                        declaration->attributeList);
    for (auto* declarator : cxx::ListView{declaration->initDeclaratorList}) {
      reject_misplaced_binding_declarator(unit_, diagnostics_,
                                          declarator->declarator);
      auto* variable =
          cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
      if (!variable || !declarator->initializer || variable->isStatic() ||
          variable->isExtern() || variable->isThreadLocal() ||
          !unit_.typeTraits().is_const(variable->type()) ||
          unit_.typeTraits().is_volatile(variable->type())) {
        fail(declarator,
             "check locals require initialized automatic non-volatile const "
             "values");
      }
      types_.admit_copy(variable->constructor(), variable->type(), declarator);
      auto value = expression(declarator->initializer);
      values_[variable] = value;
      auto hint = cxx::to_string(variable->name());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        auto component = components[index];
        if (loom_module_value(builder_.module, component)->name_id ==
            LOOM_STRING_ID_INVALID) {
          auto component_name = hint;
          if (value.is_record()) {
            const auto& partition =
                static_cast<const RecordPartition&>(value.partition());
            component_name += "_" + partition.component_names[index];
          }
          loom_string_id_t name;
          check(loom_module_intern_string(builder_.module, view(component_name),
                                          &name));
          check(loom_module_set_value_name(builder_.module, component, name));
        }
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
  // Source representation admission and retained record member slices.
  Types& types_;
  // Shared constant payload encoding with ordinary functions.
  Scalars& scalars_;
  // Source locations retained in the output module.
  Locations& locations_;
  // Caller-owned insertion point within the check case.
  loom_builder_t& builder_;
  // Immutable flattened bindings with storage owned by this check translation.
  ValueArena value_arena_;
  // Source locals retaining their source partition and component identities.
  std::unordered_map<cxx::Symbol*, Value> values_;
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
