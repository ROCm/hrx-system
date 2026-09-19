// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/translation.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/initialization.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/assumptions.h"
#include "loom/import/cxx/binding/intrinsics.h"
#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/loop_schedule.h"
#include "loom/import/cxx/control/analysis.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/symbol/functions.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/import/cxx/value/storage.h"
#include "loom/import/cxx/value/types.h"
#include "loom/import/cxx/value/vector.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {
class Translator {
 public:
  Translator(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
             loom_module_t* module, const loom_cxx_import_options_t& options)
      : unit_(unit),
        diagnostics_(diagnostics),
        module_(module),
        locations_(unit, diagnostics, module),
        types_(unit, diagnostics),
        scalars_(unit, diagnostics, types_, locations_, builder_),
        vectors_(unit, diagnostics, types_, scalars_, locations_, builder_),
        storage_(unit, diagnostics, types_, scalars_, locations_, builder_),
        intrinsics_(unit, diagnostics, types_),
        launches_(unit, diagnostics),
        functions_(unit, diagnostics, module, intrinsics_, launches_),
        options_(options),
        math_flags_(iree_any_bit_set(options.flags,
                                     LOOM_CXX_IMPORT_FLAG_APPROXIMATE_FUNCTIONS)
                        ? LOOM_SCALAR_FASTMATHFLAGS_AFN
                        : 0) {
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void translate() {
    functions_.select({options_.roots, options_.root_count});
    for (size_t index = 0; index < functions_.pending().size(); ++index) {
      function(functions_.pending()[index]);
    }
  }

 private:
  [[noreturn]] void fail(cxx::AST* ast, const std::string& message) {
    diagnostics_.reject(unit_, ast, message);
  }

  Value convert(cxx::ExpressionAST* input_ast, const cxx::Type* output_type,
                cxx::AST* owner) {
    auto value = expression(input_ast);
    if (value.is_pointer()) {
      if (loom_type_kind(types_.get(output_type, owner)) != LOOM_TYPE_BUFFER) {
        fail(owner, "conversion must preserve pointer representation");
      }
      return value;
    }
    return numeric_convert(value.ssa(), input_ast->type, output_type, owner);
  }

  loom_value_id_t numeric_convert(loom_value_id_t value,
                                  const cxx::Type* input_type,
                                  const cxx::Type* output_type,
                                  cxx::AST* owner) {
    return types_.vector(input_type) || types_.vector(output_type)
               ? vectors_.convert(value, input_type, output_type, owner)
               : scalars_.convert(value, input_type, output_type, owner);
  }

  loom_value_id_t binary(cxx::TokenKind token, loom_value_id_t left,
                         loom_value_id_t right, const cxx::Type* input_type,
                         const cxx::Type* output_type, cxx::AST* owner) {
    return types_.vector(input_type)
               ? vectors_.binary(token, left, right, input_type, output_type,
                                 owner)
               : scalars_.binary(token, left, right, input_type, output_type,
                                 owner);
  }

  loom_type_t value_type(loom_value_id_t value) {
    return loom_module_value_type(module_, value);
  }

  loom_string_id_t string(const std::string& value) {
    loom_string_id_t result;
    check(loom_builder_intern_string(
        &builder_, iree_make_cstring_view(value.c_str()), &result));
    return result;
  }

  loom_value_id_t name(loom_value_id_t value, const std::string& hint) {
    // A C++ alias of an existing SSA value keeps the original value's name.
    if (loom_module_value(module_, value)->name_id == LOOM_STRING_ID_INVALID) {
      check(loom_module_set_value_name(module_, value, string(hint)));
    }
    return value;
  }

  Value name(Value value, const std::string& hint) {
    if (value.is_pointer()) {
      auto pointer = value.pointer();
      name(pointer.root, hint);
      name(pointer.byte_offset, hint + "_byte_offset");
    } else {
      name(value.ssa(), hint);
    }
    return value;
  }

  Value region_value(loom_region_t* region, size_t& index, size_t count) {
    loom_value_id_t components[2];
    for (size_t component = 0; component < count; ++component) {
      components[component] = loom_region_entry_arg_id(region, index++);
    }
    return Value({components, count});
  }

  loom_value_id_t result(loom_op_t* op, const std::string& hint = "") {
    auto value = loom_op_results(op)[0];
    return hint.empty() ? value : name(value, hint);
  }

  // Source tails are borrowed stack frames. A returning branch consumes its
  // remaining source in at most one arm, so no continuation is duplicated.
  struct ReturnSequence {
    // Remaining statements in the current lexical block.
    cxx::List<cxx::StatementAST*>* remaining;
    // Enclosing block's tail, or null at the function boundary.
    const ReturnSequence* continuation;
  };

  struct Returned {
    // Native return operands; empty for a void source function.
    std::vector<loom_value_id_t> values;
    // Return or branch whose source location owns the resulting exit.
    cxx::AST* source;
  };

  void function(cxx::FunctionSymbol* symbol) {
    current_function_ =
        functions_.define(symbol, types_, locations_, &builder_);
    const auto& defined = current_function_;
    auto* body = defined.body;
    auto* op = defined.operation;
    auto* region = defined.region;
    auto parameters = symbol->parameters();
    control_.emplace(unit_, body);
    auto saved = loom_builder_enter_region(&builder_, op, region);
    values_.clear();
    size_t argument_index = 0;
    for (auto* parameter : parameters) {
      auto type = types_.get(parameter->type(), defined.source);
      bool pointer = loom_type_kind(type) == LOOM_TYPE_BUFFER;
      bool kernel = defined.kind == FunctionKind::Kernel;
      auto value =
          region_value(region, argument_index, pointer && !kernel ? 2 : 1);
      if (pointer && kernel) {
        value = storage_.root(value.ssa(), defined.source);
      }
      values_[parameter] = name(value, cxx::to_string(parameter->name()));
    }
    auto returned = return_sequence({body->statementList, nullptr});
    loom_op_t* terminator;
    if (defined.kind == FunctionKind::Kernel) {
      check(loom_kernel_return_build(&builder_, locations_.get(returned.source),
                                     &terminator));
    } else {
      check(loom_func_return_build(
          &builder_, returned.values.data(), returned.values.size(),
          locations_.get(returned.source), &terminator));
    }
    loom_builder_restore(&builder_, saved);
  }

  Returned return_sequence(ReturnSequence sequence) {
    for (;;) {
      if (!sequence.remaining) {
        if (sequence.continuation) {
          sequence = *sequence.continuation;
          continue;
        }
        if (current_function_.return_type->kind() != cxx::TypeKind::kVoid) {
          fail(current_function_.source,
               "non-void function can reach its end without returning a value");
        }
        return {{}, current_function_.source};
      }
      auto* child = sequence.remaining->value;
      sequence.remaining = sequence.remaining->next;
      if (control_->returns(child) != ReturnFlow::None) {
        return returning_statement(child, sequence);
      }
      statement(child);
    }
  }

  Returned returning_statement(cxx::StatementAST* ast,
                               const ReturnSequence& continuation) {
    if (auto* ret = cxx::ast_cast<cxx::ReturnStatementAST>(ast)) {
      bool returns_void =
          current_function_.return_type->kind() == cxx::TypeKind::kVoid;
      if (!ret->expression) {
        if (!returns_void) {
          fail(ast, "non-void return requires a value");
        }
        return {{}, ret};
      }
      if (ret->expression->type->kind() == cxx::TypeKind::kVoid) {
        if (!returns_void) {
          fail(ast, "non-void return requires a value");
        }
        effect(ret->expression);
        return {{}, ret};
      }
      if (returns_void) {
        fail(ast, "void return cannot carry a value");
      }
      Returned returned = {{}, ret};
      convert(ret->expression, current_function_.return_type, ret)
          .append_to(returned.values);
      return returned;
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return return_sequence({compound->statementList, &continuation});
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch && control_->returns(branch) != ReturnFlow::None) {
      if (branch->initializer || branch->constexprLoc) {
        fail(ast, "if initializer/constexpr is outside this slice");
      }
      if (control_->returns(branch->statement) != ReturnFlow::All &&
          control_->returns(branch->elseStatement) != ReturnFlow::All) {
        fail(ast,
             "conditional returns require at most one fallthrough arm; "
             "shared continuations need a scoped exit projection");
      }
      auto condition = expression(branch->condition).ssa();
      auto outer_values = values_;
      std::vector<loom_type_t> result_types;
      if (current_function_.return_type->kind() != cxx::TypeKind::kVoid) {
        types_.append(current_function_.return_type, ast, result_types);
      }
      loom_op_t* op;
      auto source = locations_.get(ast);
      check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                              condition, result_types.data(),
                              result_types.size(), nullptr, 0, source, &op));
      auto saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
      auto returned = returning_statement(branch->statement, continuation);
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, returned.values.data(),
                                 returned.values.size(),
                                 locations_.get(returned.source), &yield));
      loom_builder_restore(&builder_, saved);
      values_ = outer_values;
      saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
      returned = returning_statement(branch->elseStatement, continuation);
      check(loom_scf_yield_build(&builder_, returned.values.data(),
                                 returned.values.size(),
                                 locations_.get(returned.source), &yield));
      loom_builder_restore(&builder_, saved);
      values_ = outer_values;
      returned.values.assign(loom_op_results(op),
                             loom_op_results(op) + result_types.size());
      returned.source = ast;
      return returned;
    }
    if (control_->returns(ast) != ReturnFlow::None) {
      fail(ast,
           "returns inside loops require a structured loop-exit projection");
    }
    if (ast) {
      statement(ast);
    }
    return return_sequence(continuation);
  }

  StorageAccess address(cxx::ExpressionAST* ast) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return address(nested->expression);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      if (subscript->symbol) {
        fail(ast, "overloaded indexing is not admitted");
      }
      auto base = expression(subscript->baseExpression);
      auto index = expression(subscript->indexExpression);
      auto* base_type = subscript->baseExpression->type;
      auto* index_type = subscript->indexExpression->type;
      if (types_.vector(base_type) || types_.vector(index_type)) {
        fail(ast, "vector lane subscripts require a value projection");
      }
      if (!base.is_pointer()) {
        std::swap(base, index);
        std::swap(base_type, index_type);
      }
      return storage_.subscript(base.pointer(), index.ssa(), base_type,
                                index_type, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
        auto base = expression(unary->expression);
        return storage_.dereference(base.pointer(), ast->type, ast);
      }
    }
    fail(ast, "memory access requires a builtin subscript or dereference");
  }

  Value address_of(cxx::ExpressionAST* ast) {
    ast = cxx::Initializer::stripImplicitCasts(ast);
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return address_of(nested->expression);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      if (subscript->symbol) {
        fail(ast, "overloaded indexing is not admitted");
      }
      auto base = expression(subscript->baseExpression);
      auto index = expression(subscript->indexExpression);
      auto* base_type = subscript->baseExpression->type;
      auto* index_type = subscript->indexExpression->type;
      if (types_.vector(base_type) || types_.vector(index_type)) {
        fail(ast, "vector lanes do not have an independent storage address");
      }
      if (!base.is_pointer()) {
        std::swap(base, index);
        std::swap(base_type, index_type);
      }
      return storage_.advance(base.pointer(), index.ssa(), base_type,
                              index_type, cxx::TokenKind::T_PLUS, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
        return expression(unary->expression);
      }
    }
    fail(ast, "address-of requires an existing storage-backed element");
  }

  loom_value_id_t load(cxx::ExpressionAST* ast) {
    auto access = address(ast);
    loom_op_t* op;
    int64_t selector = access.index ? INT64_MIN : 0;
    auto build = types_.vector(ast->type) ? loom_vector_load_build
                                          : loom_view_load_build;
    check(build(&builder_, 0, 0, access.view,
                access.index ? &*access.index : nullptr, access.index ? 1 : 0,
                &selector, 1, 0, 0, types_.get(ast->type, ast),
                locations_.get(ast), &op));
    return result(op);
  }

  enum class IncrementResult { Previous, Updated };

  Value increment(cxx::ExpressionAST* destination, cxx::TokenKind token,
                  IncrementResult selected, cxx::AST* owner) {
    while (auto* nested =
               cxx::ast_cast<cxx::NestedExpressionAST>(destination)) {
      destination = nested->expression;
    }
    auto* id = cxx::ast_cast<cxx::IdExpressionAST>(destination);
    if (!id || !values_.contains(id->symbol)) {
      fail(owner, "increment requires an owned automatic source binding");
    }
    types_.require_mutable(id->type, id);
    auto previous = values_.at(id->symbol);
    auto source = locations_.get(owner);
    auto operation = token == cxx::TokenKind::T_MINUS_MINUS
                         ? cxx::TokenKind::T_MINUS
                         : cxx::TokenKind::T_PLUS;
    Value updated;
    if (auto* pointer =
            cxx::type_cast<cxx::PointerType>(types_.unqualified(id->type))) {
      auto one = scalars_.integer(1, LOOM_SCALAR_TYPE_I32, source);
      updated =
          storage_.advance(previous.pointer(), one, pointer,
                           unit_.control()->getIntType(), operation, owner);
    } else {
      if (!unit_.typeTraits().is_integral(id->type) ||
          types_.unqualified(id->type)->kind() == cxx::TypeKind::kBool) {
        fail(owner, "increment requires an integer or pointer binding");
      }
      // Builtin ++/-- performs promoted arithmetic before converting back to
      // the lvalue type, just like compound assignment with an integer one.
      auto* promoted = unit_.typeTraits().promoted_integer_type(id->type);
      auto value = scalars_.convert(previous.ssa(), id->type, promoted, owner);
      auto one = scalars_.integer(1, loom_type_element_type(value_type(value)),
                                  source);
      value = scalars_.binary(operation, value, one, promoted, promoted, owner);
      updated = scalars_.convert(value, promoted, id->type, owner);
    }
    values_[id->symbol] = name(updated, cxx::to_string(id->symbol->name()));
    return selected == IncrementResult::Previous ? previous : updated;
  }

  // The condition has already executed. Each arm starts from that same state
  // and yields its value followed by the bindings it may have changed.
  template <typename Then, typename Else>
  Value conditional_value(cxx::ExpressionAST* ast, loom_value_id_t condition,
                          Then then_value, Else else_value) {
    std::vector<loom_type_t> outputs;
    types_.append(ast->type, ast, outputs);
    auto value_count = outputs.size();
    auto written = live_mutations(ast);
    for (auto* symbol : written) {
      types_.append(symbol->type(), ast, outputs);
    }
    auto initial = current(written);
    auto source = locations_.get(ast);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.data(), outputs.size(), nullptr,
                            0, source, &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    std::vector<loom_value_id_t> yielded;
    then_value().append_to(yielded);
    auto mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    bind_values(written, initial.data());
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    yielded.clear();
    else_value().append_to(yielded);
    mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    bind_values(written, loom_op_results(op) + value_count);
    return Value({loom_op_results(op), value_count});
  }

  Value expression(cxx::ExpressionAST* ast) {
    if (!ast) {
      throw std::runtime_error("missing expression");
    }
    auto source = locations_.get(ast);
    if (auto* constant = cxx::ast_cast<cxx::ConstExpressionAST>(ast)) {
      return scalars_.constant(*constant->constValue, ast->type, ast);
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return expression(nested->expression);
    }
    if (auto* equal = cxx::ast_cast<cxx::EqualInitializerAST>(ast)) {
      return expression(equal->expression);
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(ast)) {
      return expression(initializer->expression);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast)) {
      if (cast->conversionFunction) {
        fail(ast, "user-defined conversions are not admitted");
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::BuiltinBitCastExpressionAST>(ast)) {
      auto input = types_.get(cast->expression->type, ast);
      auto output = types_.get(cast->type, ast);
      if (loom_type_kind(input) != loom_type_kind(output) ||
          (loom_type_kind(input) != LOOM_TYPE_SCALAR &&
           loom_type_kind(input) != LOOM_TYPE_VECTOR) ||
          unit_.control()->memoryLayout()->sizeOf(cast->expression->type) !=
              unit_.control()->memoryLayout()->sizeOf(cast->type)) {
        fail(ast,
             "bit_cast requires equal-width scalars or equal-width vectors");
      }
      auto value = expression(cast->expression).ssa();
      loom_op_t* op;
      auto build = types_.vector(cast->type) ? loom_vector_bitcast_build
                                             : loom_scalar_bitcast_build;
      check(build(&builder_, value, input, output, source, &op));
      return result(op);
    }
    if (auto* cast = cxx::ast_cast<cxx::CastExpressionAST>(ast)) {
      auto input = loom_type_kind(types_.get(cast->expression->type, ast));
      auto output = loom_type_kind(types_.get(cast->type, ast));
      if (input != output) {
        fail(ast,
             "explicit casts require two scalars, two vectors or two pointers");
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::CppCastExpressionAST>(ast)) {
      auto input = loom_type_kind(types_.get(cast->expression->type, ast));
      auto output = loom_type_kind(types_.get(cast->type, ast));
      if (input != output ||
          (cast->castOp != cxx::TokenKind::T_STATIC_CAST &&
           !(cast->castOp == cxx::TokenKind::T_REINTERPRET_CAST &&
             (input == LOOM_TYPE_BUFFER || input == LOOM_TYPE_VECTOR)))) {
        fail(ast,
             "casts require numeric static_cast or pointer/vector "
             "reinterpret_cast");
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::TypeConstructionAST>(ast)) {
      auto output = types_.get(ast->type, ast);
      if (cast->constructorSymbol ||
          loom_type_kind(output) != LOOM_TYPE_SCALAR ||
          (cast->expressionList && cast->expressionList->next)) {
        fail(ast, "functional casts require a scalar and at most one argument");
      }
      if (cast->expressionList) {
        return convert(cast->expressionList->value, ast->type, ast);
      }
      loom_op_t* op;
      check(loom_scalar_constant_build(
          &builder_,
          types_.is_float(ast->type) ? loom_attr_f64(0.0) : loom_attr_i64(0),
          output, source, &op));
      return result(op);
    }
    if (auto* construction =
            cxx::ast_cast<cxx::BracedTypeConstructionAST>(ast)) {
      if (construction->constructorSymbol || !types_.vector(ast->type)) {
        fail(ast, "braced value construction requires an explicit vector type");
      }
      return expression(construction->bracedInitList);
    }
    if (auto* initializer = cxx::ast_cast<cxx::BracedInitListAST>(ast)) {
      auto* vector = types_.vector(ast->type);
      if (!vector) {
        fail(ast,
             "aggregate value initializers require an explicit vector type");
      }
      std::vector<loom_value_id_t> elements;
      for (auto* element : cxx::ListView{initializer->expressionList}) {
        elements.push_back(convert(element, vector->elementType(), ast).ssa());
      }
      return vectors_.construct(elements, ast->type, ast);
    }
    if (auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(ast)) {
      if (types_.vector(select->condition->type)) {
        fail(ast, "vector conditions require lane-wise selection");
      }
      auto condition = expression(select->condition).ssa();
      return conditional_value(
          ast, condition, [&] { return expression(select->iftrueExpression); },
          [&] { return expression(select->iffalseExpression); });
    }

    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = values_.find(id->symbol); found != values_.end()) {
        return found->second;
      }
      if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(id->symbol)) {
        if (variable->constValue() &&
            (variable->isConstexpr() ||
             unit_.typeTraits().is_const(variable->type()))) {
          return name(
              scalars_.constant(*variable->constValue(), ast->type, ast),
              cxx::to_string(variable->name()));
        }
      }
      fail(ast, "unbound source value: " +
                    cxx::to_string(id->symbol ? id->symbol->name() : nullptr));
    }
    if (auto* literal = cxx::ast_cast<cxx::BoolLiteralExpressionAST>(ast)) {
      return scalars_.integer(literal->isTrue, LOOM_SCALAR_TYPE_I1, source);
    }
    if (cxx::ast_cast<cxx::IntLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::FloatLiteralExpressionAST>(ast)) {
      cxx::ASTInterpreter interpreter(&unit_);
      auto value = interpreter.evaluate(ast);
      if (!value) {
        fail(ast, "literal has no constant value");
      }
      return scalars_.constant(*value, ast->type, ast);
    }
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      auto* base = cxx::ast_cast<cxx::IdExpressionAST>(member->baseExpression);
      if (!base || !member->symbol ||
          member->accessOp != cxx::TokenKind::T_DOT) {
        fail(ast, "only topology member access is admitted");
      }
      auto axis = cxx::to_string(member->symbol->name());
      loom_kernel_dimension_t dimension;
      if (axis == "x") {
        dimension = LOOM_KERNEL_DIMENSION_X;
      } else if (axis == "y") {
        dimension = LOOM_KERNEL_DIMENSION_Y;
      } else if (axis == "z") {
        dimension = LOOM_KERNEL_DIMENSION_Z;
      } else {
        fail(ast, "unknown topology axis");
      }
      auto build = annotated(base->symbol, "workitem_id")
                       ? loom_kernel_workitem_id_build
                   : annotated(base->symbol, "workgroup_id")
                       ? loom_kernel_workgroup_id_build
                   : annotated(base->symbol, "workgroup_size")
                       ? loom_kernel_workgroup_size_build
                   : annotated(base->symbol, "workgroup_count")
                       ? loom_kernel_workgroup_count_build
                       : nullptr;
      if (!build) {
        fail(ast, "member base is not an owned topology intrinsic");
      }
      loom_op_t* op;
      auto coordinate = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      check(build(&builder_, dimension, coordinate, source, &op));
      auto value =
          name(result(op), cxx::to_string(base->symbol->name()) + "_" + axis);
      check(loom_index_cast_build(&builder_, value, coordinate,
                                  types_.get(ast->type, ast), source, &op));
      return result(op);
    }
    if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast)) {
      if (binary->symbol) {
        fail(ast, "overloaded arithmetic is not admitted");
      }
      auto left = expression(binary->leftExpression);
      if (binary->op == cxx::TokenKind::T_AMP_AMP ||
          binary->op == cxx::TokenKind::T_BAR_BAR) {
        if (types_.vector(ast->type)) {
          fail(ast, "vector logical operators require lane-wise evaluation");
        }
        // The semantic AST supplies both contextual boolean conversions.
        // The skipped arm has a known boolean result and never evaluates the
        // right operand, including its memory accesses and binding updates.
        auto evaluate_right = [&] {
          return expression(binary->rightExpression);
        };
        auto skipped = [&] {
          return Value(scalars_.integer(binary->op == cxx::TokenKind::T_BAR_BAR,
                                        LOOM_SCALAR_TYPE_I1, source));
        };
        return binary->op == cxx::TokenKind::T_AMP_AMP
                   ? conditional_value(ast, left.ssa(), evaluate_right, skipped)
                   : conditional_value(ast, left.ssa(), skipped,
                                       evaluate_right);
      }
      auto right = expression(binary->rightExpression);
      if (left.is_pointer() || right.is_pointer()) {
        if (left.is_pointer() && !right.is_pointer()) {
          return storage_.advance(
              left.pointer(), right.ssa(), binary->leftExpression->type,
              binary->rightExpression->type, binary->op, ast);
        }
        if (right.is_pointer() && !left.is_pointer() &&
            binary->op == cxx::TokenKind::T_PLUS) {
          return storage_.advance(
              right.pointer(), left.ssa(), binary->rightExpression->type,
              binary->leftExpression->type, binary->op, ast);
        }
        fail(ast,
             "pointer arithmetic requires a pointer and an integral "
             "displacement");
      }
      // C++ promotes shift operands independently. Loom's shift operands have
      // one width. Every defined source shift count fits the promoted left
      // width; normalize that count without changing any defined execution.
      if (binary->op == cxx::TokenKind::T_LESS_LESS ||
          binary->op == cxx::TokenKind::T_GREATER_GREATER) {
        right = numeric_convert(right.ssa(), binary->rightExpression->type,
                                binary->leftExpression->type, ast);
      }
      return this->binary(binary->op, left.ssa(), right.ssa(),
                          binary->leftExpression->type, ast->type, ast);
    }
    if (auto* postfix = cxx::ast_cast<cxx::PostIncrExpressionAST>(ast)) {
      if (postfix->symbol) {
        fail(ast, "only builtin increment/decrement is admitted");
      }
      return increment(postfix->baseExpression, postfix->op,
                       IncrementResult::Previous, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (unary->symbol) {
        fail(ast, "overloaded unary operations are not supported");
      }
      if (unary->op == cxx::TokenKind::T_PLUS_PLUS ||
          unary->op == cxx::TokenKind::T_MINUS_MINUS) {
        return increment(unary->expression, unary->op, IncrementResult::Updated,
                         ast);
      }
      if (unary->op == cxx::TokenKind::T_AMP) {
        return address_of(unary->expression);
      }
      if (unary->op == cxx::TokenKind::T_STAR) {
        return load(ast);
      }
      if (unary->op != cxx::TokenKind::T_PLUS &&
          unary->op != cxx::TokenKind::T_MINUS &&
          unary->op != cxx::TokenKind::T_TILDE &&
          unary->op != cxx::TokenKind::T_EXCLAIM) {
        fail(ast, "unsupported unary value expression");
      }
      auto operand = expression(unary->expression);
      if (types_.vector(unary->expression->type)) {
        return vectors_.unary(unary->op, operand.ssa(), unary->expression->type,
                              ast->type, ast);
      }
      if (unary->op == cxx::TokenKind::T_PLUS) {
        return operand;
      }
      loom_op_t* op;
      auto output = types_.get(ast->type, ast);
      auto value = operand.ssa();
      if (unary->op == cxx::TokenKind::T_MINUS) {
        if (types_.is_float(ast->type)) {
          check(
              loom_scalar_negf_build(&builder_, 0, value, output, source, &op));
        } else {
          auto zero =
              scalars_.integer(0, loom_type_element_type(output), source);
          check(loom_scalar_subi_build(&builder_, 0, zero, value, output,
                                       source, &op));
        }
        return result(op);
      }
      if (unary->op == cxx::TokenKind::T_TILDE ||
          unary->op == cxx::TokenKind::T_EXCLAIM) {
        value =
            scalars_.convert(value, unary->expression->type, ast->type, ast);
        auto mask =
            scalars_.integer(unary->op == cxx::TokenKind::T_EXCLAIM ? 1 : -1,
                             loom_type_element_type(output), source);
        check(loom_scalar_xori_build(&builder_, value, mask, output, source,
                                     &op));
        return result(op);
      }
      fail(ast, "unsupported unary value expression");
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      auto* base_type = subscript->baseExpression->type;
      auto* index_type = subscript->indexExpression->type;
      if (types_.vector(base_type) || types_.vector(index_type)) {
        if (subscript->symbol) {
          fail(ast, "overloaded indexing is not admitted");
        }
        auto base = expression(subscript->baseExpression);
        auto index = expression(subscript->indexExpression);
        if (!types_.vector(base_type)) {
          std::swap(base, index);
          std::swap(base_type, index_type);
        }
        return vectors_.extract(base.ssa(), index.ssa(), base_type, index_type,
                                ast);
      }
      return load(ast);
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(ast)) {
      auto* callee = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
      auto* function =
          callee ? cxx::symbol_cast<cxx::FunctionSymbol>(callee->symbol)
                 : nullptr;
      if (!function) {
        fail(ast, "call must resolve to a function symbol");
      }
      std::vector<loom_value_id_t> arguments;
      for (auto* argument : cxx::ListView{call->expressionList}) {
        expression(argument).append_to(arguments);
      }
      auto result_type = types_.get(ast->type, ast);
      loom_op_t* op;
      if (annotated(function, "subgroup_size")) {
        if (!arguments.empty() ||
            !loom_type_equal(result_type,
                             loom_type_scalar(LOOM_SCALAR_TYPE_I32))) {
          fail(ast, "subgroup_size requires unsigned subgroup_size()");
        }
        auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
        check(loom_kernel_subgroup_size_build(&builder_, index_type, source,
                                              &op));
        auto size = result(op);
        check(loom_index_cast_build(&builder_, size, index_type, result_type,
                                    source, &op));
        return result(op);
      }
      if (annotated(function, "shuffle_xor") && arguments.size() == 3) {
        check(loom_kernel_subgroup_shuffle_build(
            &builder_, LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR, arguments[0],
            arguments[1], arguments[2], result_type, source, &op));
        return result(op);
      }
      if (auto value = intrinsics_.call(function, arguments, math_flags_,
                                        &builder_, source)) {
        return *value;
      }
      if (!function->declaration() || annotated(function, "kernel")) {
        fail(
            ast,
            "call must resolve to an owned intrinsic or defined device helper");
      }
      auto symbol = functions_.declare(function);
      std::vector<loom_type_t> results;
      types_.append(ast->type, ast, results);
      check(loom_func_call_build(
          &builder_, 0, 0, 0, 0, symbol, arguments.data(), arguments.size(),
          results.data(), results.size(), nullptr, 0, source, &op));
      return Value({loom_op_results(op), results.size()});
    }
    fail(ast,
         "unsupported expression: " + std::string(cxx::to_string(ast->kind())));
  }

  void statement(cxx::StatementAST* ast) {
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      for (auto* child : cxx::ListView{compound->statementList}) {
        statement(child);
      }
      return;
    }
    if (auto* declaration = cxx::ast_cast<cxx::DeclarationStatementAST>(ast)) {
      if (cxx::ast_cast<cxx::EmptyDeclarationAST>(declaration->declaration)) {
        return;
      }
      auto* simple =
          cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration);
      if (!simple) {
        fail(ast, "unsupported local declaration");
      }
      for (auto* variable : cxx::ListView{simple->initDeclaratorList}) {
        auto* source_variable =
            cxx::symbol_cast<cxx::VariableSymbol>(variable->symbol);
        if (!source_variable || source_variable->isStatic() ||
            source_variable->isExtern() || source_variable->isThreadLocal()) {
          fail(ast, "local storage duration must be automatic or __shared__");
        }
        if (variable->symbol && annotated(variable->symbol, "workgroup")) {
          auto* array = cxx::type_cast<cxx::BoundedArrayType>(
              types_.unqualified(variable->symbol->type()));
          if (!array || variable->initializer ||
              current_function_.kind != FunctionKind::Kernel) {
            fail(ast,
                 "shared storage must be an uninitialized fixed scalar array "
                 "in the kernel");
          }
          auto allocation = storage_.workgroup(
              array, source_variable->explicitAlignment(), variable);
          auto spelling = cxx::to_string(variable->symbol->name());
          values_[variable->symbol] = name(Value(allocation.pointer), spelling);
          name(allocation.view, spelling + "_view");
          continue;
        }
        if (!variable->initializer || !variable->symbol) {
          fail(ast, "locals require initializers");
        }
        if (cxx::type_cast<cxx::BoundedArrayType>(
                types_.unqualified(variable->symbol->type()))) {
          fail(ast, "local arrays require __shared__ in this slice");
        }
        types_.get(variable->symbol->type(), variable);
        values_[variable->symbol] =
            name(expression(variable->initializer),
                 cxx::to_string(variable->symbol->name()));
      }
      return;
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast)) {
      if (branch->initializer || branch->constexprLoc) {
        fail(ast, "if initializer/constexpr is outside this slice");
      }
      auto condition = expression(branch->condition).ssa();
      auto saved_values = values_;
      auto written = live_mutations(ast);
      std::vector<loom_type_t> types;
      for (auto* symbol : written) {
        types_.append(symbol->type(), ast, types);
      }
      loom_op_t* op;
      auto flags = branch->elseStatement || !written.empty()
                       ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION
                       : 0;
      check(loom_scf_if_build(&builder_, flags, condition, types.data(),
                              types.size(), nullptr, 0, locations_.get(ast),
                              &op));
      auto saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
      statement(branch->statement);
      loom_op_t* yield;
      auto yielded = current(written);
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 locations_.get(ast), &yield));
      values_ = saved_values;
      loom_builder_restore(&builder_, saved);
      if (flags) {
        saved = loom_builder_enter_region(&builder_, op,
                                          loom_scf_if_else_region(op));
        if (branch->elseStatement) {
          statement(branch->elseStatement);
        }
        yielded = current(written);
        check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                   locations_.get(ast), &yield));
        values_ = saved_values;
        loom_builder_restore(&builder_, saved);
      }
      bind_values(written, loom_op_results(op));
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(ast)) {
      LoopSchedule schedule(unit_, diagnostics_, loop->attributeList);
      if (!loop->condition) {
        fail(ast, "for loops require a condition");
      }
      if (loop->initializer) {
        statement(loop->initializer);
      }
      if (auto* counted = control_->counted(loop)) {
        counted_loop(loop, *counted, schedule);
        return;
      }
      if (!schedule.empty()) {
        fail(
            ast,
            "loop scheduling requires a nonwrapping unsigned counted for loop");
      }
      conditional_loop(ast, loop->condition, loop->statement, loop->expression,
                       LoopTest::BeforeBody);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(ast)) {
      if (!LoopSchedule(unit_, diagnostics_, loop->attributeList).empty()) {
        fail(ast, "loop scheduling requires a counted for loop");
      }
      conditional_loop(ast, loop->condition, loop->statement, nullptr,
                       LoopTest::BeforeBody);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(ast)) {
      if (!LoopSchedule(unit_, diagnostics_, loop->attributeList).empty()) {
        fail(ast, "loop scheduling requires a counted for loop");
      }
      conditional_loop(ast, loop->expression, loop->statement, nullptr,
                       LoopTest::AfterBody);
      return;
    }
    if (auto* expression_statement =
            cxx::ast_cast<cxx::ExpressionStatementAST>(ast)) {
      if (expression_statement->expression) {
        effect(expression_statement->expression);
      }
      return;
    }
    fail(ast,
         "unsupported statement: " + std::string(cxx::to_string(ast->kind())));
  }

  enum class LoopTest { BeforeBody, AfterBody };

  // The before region's forwarded values are also the loop's final results.
  // A post-test loop therefore runs its body there and uses an identity after
  // region; the final body mutations survive the false condition.
  void conditional_loop(cxx::StatementAST* ast,
                        cxx::ExpressionAST* condition_expression,
                        cxx::StatementAST* body, cxx::ExpressionAST* step,
                        LoopTest test) {
    auto written = live_mutations(ast);
    auto initial = current(written);
    auto outer_values = values_;
    auto source = locations_.get(ast);
    loom_op_t* op;
    check(loom_scf_while_build(&builder_, initial.data(), initial.size(),
                               nullptr, 0, source, &op));
    auto* before = loom_scf_while_before(op);
    auto saved = loom_builder_enter_region(&builder_, op, before);
    bind(written, before);
    if (test == LoopTest::AfterBody) {
      statement(body);
    }
    auto condition = expression(condition_expression).ssa();
    auto forwarded = current(written);
    loom_op_t* terminator;
    check(loom_scf_condition_build(&builder_, condition, forwarded.data(),
                                   forwarded.size(), source, &terminator));
    loom_builder_restore(&builder_, saved);
    auto* after = loom_scf_while_after(op);
    saved = loom_builder_enter_region(&builder_, op, after);
    values_ = outer_values;
    bind(written, after);
    if (test == LoopTest::BeforeBody) {
      statement(body);
    }
    if (step) {
      effect(step);
    }
    auto yielded = current(written);
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &terminator));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    bind_values(written, loom_op_results(op));
  }

  loom_value_id_t unsigned_offset(loom_value_id_t value,
                                  loom_location_id_t source) {
    loom_op_t* cast;
    check(loom_index_cast_build(&builder_, value, value_type(value),
                                loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                source, &cast));
    return result(cast);
  }

  void counted_loop(cxx::ForStatementAST* loop, const CountedLoop& counted,
                    const LoopSchedule& schedule) {
    auto source = locations_.get(loop);
    auto* induction = counted.induction;
    auto lower = unsigned_offset(values_.at(induction).ssa(), source);
    auto upper = unsigned_offset(expression(counted.upper).ssa(), source);
    auto step = scalars_.integer(counted.step, LOOM_SCALAR_TYPE_OFFSET, source);
    auto written = live_mutations(loop);
    std::erase(written, induction);
    auto initial = current(written);
    auto outer_values = values_;
    auto* op = schedule.build(&builder_, lower, upper, step, initial, source);
    auto* body = loom_scf_for_body(op);
    auto saved = loom_builder_enter_region(&builder_, op, body);
    auto iteration = name(loom_region_entry_arg_id(body, 0),
                          cxx::to_string(induction->name()));
    loom_op_t* cast;
    check(loom_index_cast_build(&builder_, iteration, value_type(iteration),
                                types_.get(induction->type(), loop), source,
                                &cast));
    values_[induction] = result(cast);
    bind(written, body, 1);
    statement(loop->statement);
    auto yielded = current(written);
    loom_op_t* terminator;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &terminator));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    values_.erase(induction);
    bind_values(written, loom_op_results(op));
  }

  std::vector<cxx::Symbol*> live_mutations(cxx::AST* owner) {
    std::vector<cxx::Symbol*> result;
    for (auto* symbol : control_->written(owner)) {
      if (values_.contains(symbol)) {
        result.push_back(symbol);
      }
    }
    return result;
  }
  std::vector<loom_value_id_t> current(
      const std::vector<cxx::Symbol*>& symbols) {
    std::vector<loom_value_id_t> result;
    for (auto* symbol : symbols) {
      values_.at(symbol).append_to(result);
    }
    return result;
  }
  void bind(const std::vector<cxx::Symbol*>& symbols, loom_region_t* region,
            size_t index = 0) {
    for (auto* symbol : symbols) {
      auto count = values_.at(symbol).components().size();
      values_[symbol] = name(region_value(region, index, count),
                             cxx::to_string(symbol->name()));
    }
  }
  void bind_values(const std::vector<cxx::Symbol*>& symbols,
                   const loom_value_id_t* components) {
    size_t index = 0;
    for (auto* symbol : symbols) {
      auto count = values_.at(symbol).components().size();
      values_[symbol] = name(Value({components + index, count}),
                             cxx::to_string(symbol->name()));
      index += count;
    }
  }

  void effect(cxx::ExpressionAST* ast) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      effect(nested->expression);
      return;
    }
    if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast)) {
      if (binary->op == cxx::TokenKind::T_AMP_AMP ||
          binary->op == cxx::TokenKind::T_BAR_BAR) {
        expression(ast);
        return;
      }
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(ast)) {
      auto* id = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
      if (id && annotated(id->symbol, "assume")) {
        auto bounds = assumption_bounds(unit_, diagnostics_, call);
        for (const auto& bound : bounds) {
          auto value = expression(bound.value).ssa();
          auto value_type = types_.get(bound.value->type, ast);
          loom_predicate_t predicate = {
              .kind = LOOM_PREDICATE_RANGE,
              .arg_count = 3,
              .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                           LOOM_PRED_ARG_CONST},
              .args = {value, 0, bound.upper_bound - 1},
          };
          loom_op_t* op;
          check(loom_scalar_assume_build(&builder_, &value, 1, &predicate, 1,
                                         &value_type, 1, locations_.get(ast),
                                         &op));
          values_[bound.binding->symbol] =
              name(numeric_convert(result(op), bound.value->type,
                                   bound.binding->type, ast),
                   cxx::to_string(bound.binding->symbol->name()));
        }
        return;
      }
      if (!id || !annotated(id->symbol, "barrier")) {
        if (types_.unqualified(ast->type)->kind() != cxx::TypeKind::kVoid) {
          expression(ast);
          return;
        }
        auto* function =
            id ? cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol) : nullptr;
        if (!function || !function->declaration() ||
            annotated(function, "kernel")) {
          fail(
              ast,
              "void call requires a defined ordinary function or an intrinsic");
        }
        std::vector<loom_value_id_t> arguments;
        for (auto* argument : cxx::ListView{call->expressionList}) {
          expression(argument).append_to(arguments);
        }
        auto callee = functions_.declare(function);
        loom_op_t* op;
        check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee,
                                   arguments.data(), arguments.size(), nullptr,
                                   0, nullptr, 0, locations_.get(ast), &op));
        return;
      }
      if (call->expressionList) {
        fail(ast, "barrier does not take arguments");
      }
      // HIP __syncthreads covers both global and shared memory. Loom names one
      // memory space per barrier, so preserve both fences explicitly.
      loom_op_t* op;
      check(loom_kernel_barrier_build(
          &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
          LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL,
          locations_.get(ast), &op));
      check(loom_kernel_barrier_build(
          &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
          LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL,
          locations_.get(ast), &op));
      return;
    }
    if (auto* assignment =
            cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(ast)) {
      auto* destination =
          cxx::ast_cast<cxx::IdExpressionAST>(assignment->targetExpression);
      if (!destination || assignment->symbol) {
        fail(ast, "compound assignment requires a builtin local");
      }
      types_.require_mutable(destination->type, destination);
      auto old = expression(destination);
      auto value = expression(assignment->rightExpression);
      if (old.is_pointer()) {
        auto updated = storage_.advance(
            old.pointer(), value.ssa(), destination->type,
            assignment->rightExpression->type,
            cxx::get_underlying_binary_op(assignment->op), ast);
        values_[destination->symbol] =
            name(Value(updated), cxx::to_string(destination->symbol->name()));
        return;
      }
      auto* promoted = assignment->leftExpression->type;
      old = numeric_convert(old.ssa(), destination->type, promoted, ast);
      value = numeric_convert(value.ssa(), assignment->rightExpression->type,
                              promoted, ast);
      auto updated = binary(cxx::get_underlying_binary_op(assignment->op),
                            old.ssa(), value.ssa(), promoted, promoted, ast);
      updated = numeric_convert(updated, promoted, destination->type, ast);
      values_[destination->symbol] =
          name(updated, cxx::to_string(destination->symbol->name()));
      return;
    }
    if (auto* assignment = cxx::ast_cast<cxx::AssignmentExpressionAST>(ast)) {
      if (assignment->symbol || assignment->op != cxx::TokenKind::T_EQUAL) {
        fail(ast, "only builtin plain assignment is admitted here");
      }
      types_.require_mutable(assignment->leftExpression->type,
                             assignment->leftExpression);
      auto value = expression(assignment->rightExpression);
      if (auto* destination =
              cxx::ast_cast<cxx::IdExpressionAST>(assignment->leftExpression)) {
        if (!values_.contains(destination->symbol)) {
          fail(ast, "assignment requires an owned automatic source binding");
        }
        values_[destination->symbol] =
            name(value, cxx::to_string(destination->symbol->name()));
        return;
      }
      auto access = address(assignment->leftExpression);
      int64_t selector = access.index ? INT64_MIN : 0;
      loom_op_t* op;
      auto build = types_.vector(assignment->leftExpression->type)
                       ? loom_vector_store_build
                       : loom_view_store_build;
      check(build(&builder_, 0, 0, value.ssa(), access.view,
                  access.index ? &*access.index : nullptr, access.index ? 1 : 0,
                  &selector, 1, 0, 0, locations_.get(ast), &op));
      return;
    }
    auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast);
    if (cxx::ast_cast<cxx::PostIncrExpressionAST>(ast) ||
        (unary && (unary->op == cxx::TokenKind::T_PLUS_PLUS ||
                   unary->op == cxx::TokenKind::T_MINUS_MINUS))) {
      expression(ast);
      return;
    }
    fail(ast, "unsupported effect expression: " +
                  std::string(cxx::to_string(ast->kind())));
  }

  // Source AST/symbol lifetime ends after construction and verification.
  cxx::TranslationUnit& unit_;
  // Source diagnostics share frontend byte ranges and the caller sink.
  Diagnostics& diagnostics_;
  // Output arena owner.
  loom_module_t* module_;
  // Current insertion point in the structured output.
  loom_builder_t builder_ = {};
  // Retained source ranges copied into the output module.
  Locations locations_;
  // Source type and scalar representation contracts.
  Types types_;
  // Numeric builders consume evaluated operands without AST callbacks.
  Scalars scalars_;
  // Explicit vector builders retain lane widths and full-width source masks.
  Vectors vectors_;
  // Memory representations retain declared array extents and access shape.
  Storage storage_;
  // Retained generated operation bindings for reached source declarations.
  Intrinsics intrinsics_;
  // Admitted launch contracts, including bounds from function redeclarations.
  LaunchContracts launches_;
  // Root selection, native definition contracts and reachable identities.
  Functions functions_;
  // Borrowed source configuration for this invocation.
  const loom_cxx_import_options_t& options_;
  // Explicit source-level permission for approximate math function results.
  uint8_t math_flags_;
  // Source and native definition contracts for the body being translated.
  FunctionBody current_function_ = {};
  // Bound symbols, never identifier spellings, key source-to-SSA mappings.
  std::unordered_map<cxx::Symbol*, Value> values_;
  // Immutable control facts for the function currently being translated.
  std::optional<ControlFlow> control_;
};
}  // namespace

void translate(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
               loom_module_t* module,
               const loom_cxx_import_options_t& options) {
  Translator(unit, diagnostics, module, options).translate();
}

}  // namespace loom::cxx_import
