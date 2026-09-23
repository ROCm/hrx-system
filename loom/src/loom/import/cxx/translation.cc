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

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/assumptions.h"
#include "loom/import/cxx/binding/config.h"
#include "loom/import/cxx/binding/intrinsics.h"
#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/loop_schedule.h"
#include "loom/import/cxx/check.h"
#include "loom/import/cxx/control/analysis.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/symbol/functions.h"
#include "loom/import/cxx/value/bitcast.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/import/cxx/value/signature.h"
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
        names_(unit, diagnostics),
        configs_(unit, diagnostics, types_, scalars_, locations_, names_),
        vectors_(unit, diagnostics, types_, scalars_, locations_, builder_),
        storage_(unit, diagnostics, types_, scalars_, locations_, builder_),
        intrinsics_(unit, diagnostics, types_),
        launches_(unit, diagnostics),
        functions_(unit, diagnostics, module, intrinsics_, launches_, configs_,
                   names_),
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
    functions_.build_benchmarks(locations_, &builder_);
    assembly_fragments_.verify(module_, options_.diagnostic_sink);
  }

 private:
  [[noreturn]] void fail(cxx::AST* ast, const std::string& message) {
    diagnostics_.reject(unit_, ast, message);
  }

  void require_kernel_context(cxx::AST* owner) {
    // Required inlining defers the IR ancestor requirement until expansion.
    if (current_function_.kind == FunctionKind::Kernel ||
        (current_function_.kind == FunctionKind::Ordinary &&
         loom_func_def_inline_policy(current_function_.operation) ==
             LOOM_INLINE_POLICY_INLINE)) {
      return;
    }
    fail(owner, "kernel intrinsic requires a kernel or force-inline helper");
  }

  Value convert(cxx::ExpressionAST* input_ast, const cxx::Type* output_type,
                cxx::AST* owner) {
    auto value = expression(input_ast);
    if (value.is_record() || value.is_encoding() || value.is_view() ||
        value.is_tensor()) {
      if (&types_.partition(output_type, owner) != &value.partition()) {
        fail(owner, "conversion must preserve the source value type");
      }
      return value;
    }
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
    if (value.is_record()) {
      const auto& partition =
          static_cast<const RecordPartition&>(value.partition());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        name(components[index], hint + "_" + partition.component_names[index]);
      }
    } else if (value.is_view()) {
      const auto& partition =
          static_cast<const ViewPartition&>(value.partition());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        auto suffix = partition.component_names[index];
        name(components[index], suffix.empty() ? hint : hint + "_" + suffix);
      }
    } else if (value.is_pointer()) {
      auto pointer = value.pointer();
      name(pointer.root, hint);
      name(pointer.byte_offset, hint + "_byte_offset");
    } else if (value.is_encoding() || value.is_tensor()) {
      name(value.components()[0], hint);
    } else {
      name(value.ssa(), hint);
    }
    return value;
  }

  Value region_value(loom_region_t* region, size_t& index,
                     const Partition& partition) {
    auto* block = loom_region_entry_block(region);
    auto components =
        std::span<const loom_value_id_t>(block->arg_ids, block->arg_count)
            .subspan(index, partition.component_count);
    index += partition.component_count;
    return value_arena_.capture(partition, components);
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
    if (defined.kind == FunctionKind::CheckCase) {
      auto saved = loom_builder_enter_region(&builder_, op, region);
      translate_check_body(unit_, diagnostics_, functions_, intrinsics_, types_,
                           scalars_, locations_, builder_, defined);
      loom_builder_restore(&builder_, saved);
      return;
    }
    auto parameters = symbol->parameters();
    control_.emplace(unit_, diagnostics_, types_, body);
    auto saved = loom_builder_enter_region(&builder_, op, region);
    values_.clear();
    locals_.clear();
    value_arena_.reset();
    size_t argument_index = 0;
    size_t parameter_index = 0;
    for (auto* parameter : parameters) {
      bool kernel = defined.kind == FunctionKind::Kernel;
      const auto& partition =
          types_.partition(parameter->type(), defined.source);
      auto value = region_value(region, argument_index,
                                kernel ? kSSAPartition : partition);
      if (partition.kind == ValueKind::Pointer && kernel) {
        auto buffer = value.ssa();
        if (!defined.parameter_contracts.empty()) {
          buffer = apply_parameter_contract(
              defined.parameter_contracts[parameter_index], buffer, locations_,
              &builder_);
        }
        value = storage_.root(buffer, defined.source);
      }
      value = name(value, cxx::to_string(parameter->name()));
      if (control_->addressed(parameter)) {
        auto access = allocate_local(parameter, 0, defined.source);
        storage_.store(access, value.ssa(), parameter->type(), defined.source);
      } else {
        values_[parameter] = value;
      }
      ++parameter_index;
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
      if (control_->returns(child) != ExitFlow::None) {
        return returning_statement(child, sequence);
      }
      statement(child);
    }
  }

  StorageAccess allocate_local(cxx::Symbol* symbol, int64_t alignment,
                               cxx::AST* owner) {
    auto allocation = storage_.allocate(
        symbol->type(), LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, alignment, owner);
    auto spelling = cxx::to_string(symbol->name());
    name(Value(allocation.pointer), spelling + "_storage");
    name(allocation.view, spelling + "_view");
    locals_[symbol] = allocation;
    return {allocation.view, std::nullopt};
  }

  Value binding(cxx::Symbol* symbol, cxx::AST* owner) {
    auto found = locals_.find(symbol);
    if (found == locals_.end()) {
      return values_.at(symbol);
    }
    if (unit_.typeTraits().is_array(symbol->type())) {
      return found->second.pointer;
    }
    return name(storage_.load({found->second.view, std::nullopt},
                              symbol->type(), owner),
                cxx::to_string(symbol->name()));
  }

  void initialize_variable(cxx::VariableSymbol* variable,
                           cxx::ExpressionAST* initializer, cxx::AST* owner) {
    if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(
            types_.unqualified(variable->type()))) {
      auto access =
          allocate_local(variable, variable->explicitAlignment(), owner);
      if (!initializer) {
        return;
      }
      auto* elements = cxx::Initializer(initializer).expressionListSlot();
      if (!elements) {
        fail(owner, "automatic arrays require element-wise initialization");
      }
      auto* element_type =
          unit_.typeTraits().get_element_type(variable->type());
      auto* next = *elements;
      // The source frontend supplies conversions and explicit element order.
      // Each store precedes the next clause, which may read this same array.
      // Omitted trivial elements are value-initialized, unlike a declaration
      // without an initializer.
      for (size_t index = 0; index < array->size(); ++index) {
        auto value =
            next ? expression(next->value).ssa()
                 : initialize(array->elementType(), nullptr, owner).ssa();
        access.index = scalars_.integer(index, LOOM_SCALAR_TYPE_INDEX,
                                        locations_.get(owner));
        storage_.store(access, value, element_type, owner);
        if (next) {
          next = next->next;
        }
      }
      return;
    }
    if (control_->addressed(variable) ||
        unit_.typeTraits().is_volatile(variable->type())) {
      auto access =
          allocate_local(variable, variable->explicitAlignment(), owner);
      if (initializer) {
        storage_.store(access, expression(initializer).ssa(), variable->type(),
                       owner);
      }
      return;
    }
    if (!initializer) {
      fail(owner, "locals require initializers");
    }
    types_.partition(variable->type(), owner);
    types_.admit_copy(variable->constructor(), variable->type(), owner);
    values_[variable] =
        name(expression(initializer), cxx::to_string(variable->name()));
  }

  void initialize_condition(cxx::VariableSymbol* variable) {
    if (!variable) {
      return;
    }
    auto* declaration = control_->condition_declaration(variable);
    reject_misplaced_binding_attributes(unit_, diagnostics_,
                                        declaration->attributeList);
    reject_misplaced_binding_declarator(unit_, diagnostics_,
                                        declaration->declarator);
    if (variable->isStatic() || variable->isExtern() ||
        variable->isThreadLocal()) {
      fail(declaration, "condition storage duration must be automatic");
    }
    initialize_variable(variable, declaration->initializer, declaration);
  }

  // Source-selected branches are transparent to every statement/exit path.
  // Their initializers still execute once; a discarded arm contributes no IR.
  cxx::StatementAST* selected_statement(cxx::StatementAST* ast) {
    while (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast)) {
      if (!branch->constexprValue.has_value()) {
        break;
      }
      if (branch->initializer) {
        statement(branch->initializer);
      }
      initialize_condition(branch->decisionVariable);
      ast = *branch->constexprValue ? branch->statement : branch->elseStatement;
    }
    return ast;
  }

  loom_value_id_t branch_condition(cxx::IfStatementAST* branch) {
    if (branch->initializer) {
      statement(branch->initializer);
    }
    initialize_condition(branch->decisionVariable);
    return expression(branch->condition).ssa();
  }

  Returned returning_statement(cxx::StatementAST* ast,
                               const ReturnSequence& continuation) {
    ast = selected_statement(ast);
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
        branch && control_->returns(branch) != ExitFlow::None) {
      if (control_->returns(branch->statement) != ExitFlow::All &&
          control_->returns(branch->elseStatement) != ExitFlow::All) {
        fail(ast,
             "conditional returns require at most one fallthrough arm; "
             "shared continuations need a scoped exit projection");
      }
      auto condition = branch_condition(branch);
      auto outer_values = values_;
      BoundSignature result_signature;
      if (current_function_.return_type->kind() != cxx::TypeKind::kVoid) {
        std::array<const cxx::Type*, 1> result_sources = {
            current_function_.return_type};
        result_signature =
            bind_signature(types_, result_sources, ast, &builder_);
      }
      loom_op_t* op;
      auto source = locations_.get(ast);
      check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                              condition, result_signature.types.data(),
                              result_signature.types.size(), nullptr, 0, source,
                              &op));
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
      returned.values.assign(
          loom_op_results(op),
          loom_op_results(op) + result_signature.types.size());
      returned.source = ast;
      return returned;
    }
    if (control_->returns(ast) != ExitFlow::None) {
      fail(ast,
           "returns inside loops require a structured loop-exit projection");
    }
    if (ast) {
      statement(ast);
    }
    return return_sequence(continuation);
  }

  StorageProjection pointer_projection(Value value, const cxx::Type* type,
                                       cxx::AST* owner) {
    auto* pointer = cxx::type_cast<cxx::PointerType>(types_.unqualified(type));
    return storage_.project(value.pointer(),
                            pointer ? pointer->elementType() : type, owner);
  }

  StorageProjection pointer_expression(cxx::ExpressionAST* ast) {
    auto* source = cxx::Initializer::stripImplicitCasts(ast);
    if (unit_.typeTraits().is_array(source->type)) {
      // Direct array indexing retains its enclosing lvalue's alignment. An
      // actual pointer value, including an explicit cast or call result, starts
      // with its pointee contract and carries no hidden alignment component.
      auto projection = object_address(source);
      if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(source)) {
        name(projection.pointer, cxx::to_string(member->symbol->name()));
      }
      return projection;
    }
    return pointer_projection(expression(ast), ast->type, ast);
  }

  struct SubscriptOperands {
    // Evaluated array or pointer origin with its source alignment.
    StorageProjection base;
    // Evaluated integral displacement, before address-width conversion.
    loom_value_id_t index;
    // Source array or pointer type owning the element stride.
    const cxx::Type* base_type;
    // Source displacement type owning its signedness and width.
    const cxx::Type* index_type;
  };

  SubscriptOperands subscript_operands(cxx::SubscriptExpressionAST* ast) {
    if (ast->symbol) {
      fail(ast, "overloaded indexing is not admitted");
    }
    auto* base_type = ast->baseExpression->type;
    auto* index_type = ast->indexExpression->type;
    if (types_.vector(base_type) || types_.vector(index_type)) {
      fail(ast, "vector lane subscripts require a value projection");
    }
    if (unit_.typeTraits().is_pointer(base_type) ||
        unit_.typeTraits().is_array(base_type)) {
      auto base = pointer_expression(ast->baseExpression);
      auto index = expression(ast->indexExpression);
      return {base, index.ssa(), base_type, index_type};
    }
    // The commuted spelling i[p] still evaluates its written base first.
    auto index = expression(ast->baseExpression);
    auto base = pointer_expression(ast->indexExpression);
    return {base, index.ssa(), index_type, base_type};
  }

  StorageAccess address(cxx::ExpressionAST* ast) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return address(nested->expression);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      auto found = locals_.find(id->symbol);
      if (found != locals_.end()) {
        return {found->second.view, std::nullopt};
      }
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      auto operands = subscript_operands(subscript);
      return storage_.subscript(operands.base, operands.index,
                                operands.base_type, operands.index_type, ast);
    }
    return storage_.dereference(object_address(ast), ast->type, ast);
  }

  StorageProjection object_address(cxx::ExpressionAST* ast) {
    ast = cxx::Initializer::stripImplicitCasts(ast);
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return object_address(nested->expression);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = locals_.find(id->symbol); found != locals_.end()) {
        return storage_.project(found->second.pointer, id->type, id);
      }
      if (unit_.typeTraits().is_array(id->type)) {
        return storage_.project(expression(id).pointer(), id->type, id);
      }
    }
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (!field || field->isStatic()) {
        fail(ast, "record storage access requires a non-static data member");
      }
      auto base = member->accessOp == cxx::TokenKind::T_MINUS_GREATER
                      ? pointer_expression(member->baseExpression)
                      : object_address(member->baseExpression);
      return storage_.member(base, field, ast);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      auto operands = subscript_operands(subscript);
      return storage_.advance(operands.base, operands.index, operands.base_type,
                              operands.index_type, cxx::TokenKind::T_PLUS, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
        return pointer_expression(unary->expression);
      }
    }
    fail(ast, "address-of requires an existing storage-backed element");
  }

  Value load(cxx::ExpressionAST* ast) {
    // An array lvalue denotes borrowed storage. Decay and subsequent element
    // projections preserve that origin without loading or copying the array.
    if (unit_.typeTraits().is_array(ast->type)) {
      return object_address(ast).pointer;
    }
    return storage_.load(address(ast), ast->type, ast);
  }

  enum class IncrementResult { Previous, Updated };

  // Resolving the source lvalue executes its address expressions once. Reads
  // and writes then share that location even when another binding changes.
  struct Lvalue {
    // Source element type retains qualifiers and arithmetic conversion rules.
    const cxx::Type* type;
    // An automatic SSA partition or an evaluated storage projection.
    std::variant<Destination, StorageAccess> location;
  };

  Lvalue destination(cxx::ExpressionAST* expression, cxx::AST* owner) {
    types_.require_mutable(expression->type, expression);
    if (auto destination = control_->destination(expression)) {
      if (locals_.contains(destination->binding)) {
        return {expression->type, address(expression)};
      }
      if (!values_.contains(destination->binding)) {
        fail(owner, "mutation requires an owned automatic source binding");
      }
      return {expression->type, *destination};
    }
    return {expression->type, address(expression)};
  }

  Value read(const Lvalue& target, cxx::AST* owner) {
    if (auto* destination = std::get_if<Destination>(&target.location)) {
      auto value = values_.at(destination->binding);
      return destination->member ? value.project(*destination->member,
                                                 destination->component_offset)
                                 : value;
    }
    return storage_.load(std::get<StorageAccess>(target.location), target.type,
                         owner);
  }

  void write(const Lvalue& target, Value value, cxx::AST* owner) {
    if (auto* destination = std::get_if<Destination>(&target.location)) {
      auto& binding = values_.at(destination->binding);
      binding = name(destination->member
                         ? value_arena_.replace(
                               binding, destination->component_offset, value)
                         : value,
                     cxx::to_string(destination->binding->name()));
    } else {
      storage_.store(std::get<StorageAccess>(target.location), value.ssa(),
                     target.type, owner);
    }
  }

  Value increment(cxx::ExpressionAST* destination, cxx::TokenKind token,
                  IncrementResult selected, cxx::AST* owner) {
    auto target = this->destination(destination, owner);
    auto previous = read(target, destination);
    auto source = locations_.get(owner);
    auto operation = token == cxx::TokenKind::T_MINUS_MINUS
                         ? cxx::TokenKind::T_MINUS
                         : cxx::TokenKind::T_PLUS;
    Value updated;
    if (auto* pointer = cxx::type_cast<cxx::PointerType>(
            types_.unqualified(destination->type))) {
      auto one = scalars_.integer(1, LOOM_SCALAR_TYPE_I32, source);
      updated =
          storage_
              .advance(pointer_projection(previous, pointer, owner), one,
                       pointer, unit_.control()->getIntType(), operation, owner)
              .pointer;
    } else {
      if (!unit_.typeTraits().is_integral(destination->type) ||
          types_.unqualified(destination->type)->kind() ==
              cxx::TypeKind::kBool) {
        fail(owner, "increment requires an integer or pointer binding");
      }
      // Builtin ++/-- performs promoted arithmetic before converting back to
      // the lvalue type, just like compound assignment with an integer one.
      auto* promoted =
          unit_.typeTraits().promoted_integer_type(destination->type);
      auto value =
          scalars_.convert(previous.ssa(), destination->type, promoted, owner);
      auto one = scalars_.integer(1, loom_type_element_type(value_type(value)),
                                  source);
      value = scalars_.binary(operation, value, one, promoted, promoted, owner);
      updated = scalars_.convert(value, promoted, destination->type, owner);
    }
    write(target, updated, owner);
    return selected == IncrementResult::Previous ? previous : updated;
  }

  Value compound_assignment(cxx::CompoundAssignmentExpressionAST* assignment) {
    auto* ast = assignment;
    auto* destination = assignment->targetExpression;
    if (assignment->symbol) {
      fail(ast, "overloaded compound assignment is not admitted");
    }
    auto value = expression(assignment->rightExpression);
    // The right operand precedes both the address evaluation and value read.
    auto target = this->destination(destination, ast);
    auto old = read(target, destination);
    if (old.is_pointer()) {
      auto updated =
          storage_
              .advance(pointer_projection(old, destination->type, ast),
                       value.ssa(), destination->type,
                       assignment->rightExpression->type,
                       cxx::get_underlying_binary_op(assignment->op), ast)
              .pointer;
      write(target, Value(updated), ast);
      return Value(updated);
    }
    auto* promoted = assignment->leftExpression->type;
    old = numeric_convert(old.ssa(), destination->type, promoted, ast);
    value = numeric_convert(value.ssa(), assignment->rightExpression->type,
                            promoted, ast);
    auto updated = binary(cxx::get_underlying_binary_op(assignment->op),
                          old.ssa(), value.ssa(), promoted, promoted, ast);
    updated = numeric_convert(updated, promoted, destination->type, ast);
    write(target, updated, ast);
    return updated;
  }

  Value assignment(cxx::AssignmentExpressionAST* assignment) {
    auto* ast = assignment;
    const auto& partition = types_.partition(
        types_.unqualified(assignment->leftExpression->type), ast);
    cxx::ClassSymbol* source = nullptr;
    if (partition.kind == ValueKind::Record) {
      source = static_cast<const RecordPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Encoding) {
      source = static_cast<const EncodingPartition&>(partition).source;
    } else if (partition.kind == ValueKind::View) {
      source = static_cast<const ViewPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Tensor) {
      source = static_cast<const TensorPartition&>(partition).source;
    }
    if ((assignment->symbol &&
         (!source ||
          (assignment->symbol != source->copyAssignmentOperator() &&
           assignment->symbol != source->moveAssignmentOperator()))) ||
        assignment->op != cxx::TokenKind::T_EQUAL) {
      fail(ast, "only builtin plain assignment is admitted here");
    }
    auto value = expression(assignment->rightExpression);
    write(destination(assignment->leftExpression, ast), value, ast);
    return value;
  }

  // The condition has already executed. Each arm starts from that same state
  // and yields its value followed by the bindings it may have changed.
  template <typename Then, typename Else>
  Value conditional_value(cxx::ExpressionAST* ast, loom_value_id_t condition,
                          Then then_value, Else else_value) {
    auto written = live_mutations(ast);
    auto value_count = types_.partition(ast->type, ast).component_count;
    std::vector<const cxx::Type*> sources = {ast->type};
    sources.reserve(1 + written.size());
    for (auto* symbol : written) {
      sources.push_back(symbol->type());
    }
    auto initial = current(written);
    auto source = locations_.get(ast);
    auto outputs = bind_signature(types_, sources, ast, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0, source, &op));
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
    return value_arena_.capture(types_.partition(ast->type, ast),
                                {loom_op_results(op), value_count});
  }

  Value initialize(const cxx::Type* type,
                   cxx::List<cxx::ExpressionAST*>* elements, cxx::AST* owner) {
    if (auto* record = types_.record(type, owner)) {
      // Single same-type arguments are copy-list/direct initialization, not
      // the first aggregate field. The normalized AST retains that distinction.
      if (elements && !elements->next &&
          types_.unqualified(elements->value->type) ==
              types_.unqualified(type)) {
        return expression(elements->value);
      }
      std::vector<loom_value_id_t> components;
      components.reserve(record->component_count);
      for (const auto& member : record->members) {
        convert(elements->value, member.field->type(), owner)
            .append_to(components);
        elements = elements->next;
      }
      return value_arena_.capture(*record, components);
    }
    const auto& partition = types_.partition(type, owner);
    if (partition.kind == ValueKind::Encoding ||
        partition.kind == ValueKind::View ||
        partition.kind == ValueKind::Tensor) {
      if (elements && !elements->next &&
          types_.unqualified(elements->value->type) ==
              types_.unqualified(type)) {
        return expression(elements->value);
      }
      fail(owner,
           "encoding, view and tensor values require an operation result or a "
           "copy");
    }
    if (auto* vector = types_.vector(type)) {
      std::vector<loom_value_id_t> components;
      for (auto* element : cxx::ListView{elements}) {
        components.push_back(
            convert(element, vector->elementType(), owner).ssa());
      }
      return vectors_.construct(components, type, owner);
    }
    auto output = types_.get(type, owner);
    if (elements && !elements->next) {
      return convert(elements->value, type, owner);
    }
    if (elements || loom_type_kind(output) != LOOM_TYPE_SCALAR) {
      fail(owner,
           "value initialization requires a scalar, vector or admitted record");
    }
    loom_op_t* op;
    check(loom_scalar_constant_build(
        &builder_,
        types_.is_float(type) ? loom_attr_f64(0.0) : loom_attr_i64(0), output,
        locations_.get(owner), &op));
    return result(op);
  }

  std::optional<IntrinsicCallResult> atomic_builtin(
      cxx::CallExpressionAST* call) {
    if (auto atomic =
            AtomicBuiltin::resolve(unit_, diagnostics_, types_, call)) {
      std::array<Value, 4> arguments;
      auto* argument = call->expressionList;
      // Admission consumed the constant orderings. Evaluate every remaining
      // operand once, including weak even though High CAS is always strong.
      for (size_t index = 0; index < atomic->argument_count(); ++index) {
        arguments[index] = expression(argument->value);
        argument = argument->next;
      }
      return IntrinsicCallResult{atomic->call(
          std::span(arguments).first(atomic->argument_count()), storage_,
          scalars_, call, &builder_, locations_.get(call))};
    }
    if (auto fence =
            FenceIntrinsic::resolve_builtin(unit_, diagnostics_, call)) {
      fence->call(&builder_, locations_.get(call));
      return IntrinsicCallResult{std::nullopt};
    }
    return std::nullopt;
  }

  Value constant(const cxx::ConstValue& value, cxx::ExpressionAST* ast) {
    return types_.vector(ast->type) ? vectors_.constant(value, ast->type, ast)
                                    : scalars_.constant(value, ast->type, ast);
  }

  Value expression(cxx::ExpressionAST* ast) {
    if (!ast) {
      throw std::runtime_error("missing expression");
    }
    if (auto* assignment = cxx::ast_cast<cxx::AssignmentExpressionAST>(ast)) {
      return this->assignment(assignment);
    }
    if (auto* assignment =
            cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(ast)) {
      return compound_assignment(assignment);
    }
    auto source = locations_.get(ast);
    if (auto* expression = cxx::ast_cast<cxx::ConstExpressionAST>(ast)) {
      return constant(*expression->constValue, ast);
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
        types_.admit_copy(cast->conversionFunction, ast->type, ast);
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast =
            cxx::ast_cast<cxx::BuiltinConvertVectorExpressionAST>(ast)) {
      auto value = expression(cast->expression).ssa();
      return vectors_.convert_elements(value, cast->expression->type,
                                       cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::BuiltinBitCastExpressionAST>(ast)) {
      BitCast conversion(unit_, diagnostics_, types_, cast->expression->type,
                         cast->type, ast);
      auto value = expression(cast->expression).ssa();
      return conversion.build(builder_, value, source);
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
      if (types_.record(ast->type, ast)) {
        types_.admit_copy(cast->constructorSymbol, ast->type, ast);
        return initialize(ast->type, cast->expressionList, ast);
      }
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
      types_.admit_copy(construction->constructorSymbol, ast->type, ast);
      return initialize(ast->type, construction->bracedInitList->expressionList,
                        ast);
    }
    if (auto* initializer = cxx::ast_cast<cxx::BracedInitListAST>(ast)) {
      return initialize(ast->type, initializer->expressionList, ast);
    }
    if (auto* initializer = cxx::ast_cast<cxx::ParenInitializerAST>(ast)) {
      return initialize(ast->type, initializer->expressionList, ast);
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

    if (auto* decision = cxx::ast_cast<cxx::ConditionExpressionAST>(ast)) {
      return binding(decision->symbol, ast);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = values_.find(id->symbol); found != values_.end()) {
        return found->second;
      }
      if (locals_.contains(id->symbol)) {
        return binding(id->symbol, ast);
      }
      if (auto* enumerator =
              cxx::symbol_cast<cxx::EnumeratorSymbol>(id->symbol)) {
        if (!enumerator->value()) {
          fail(ast, "enumerator has no resolved constant value");
        }
        return name(scalars_.constant(*enumerator->value(), ast->type, ast),
                    cxx::to_string(enumerator->name()));
      }
      if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(id->symbol)) {
        if (auto value = configs_.read(variable, &builder_, source)) {
          return name(*value, cxx::to_string(variable->name()));
        }
        if (variable->constValue() &&
            !unit_.typeTraits().is_volatile(variable->type()) &&
            (variable->isConstexpr() ||
             unit_.typeTraits().is_const(variable->type()))) {
          return name(constant(*variable->constValue(), ast),
                      cxx::to_string(variable->name()));
        }
      }
      if (cxx::symbol_cast<cxx::FieldSymbol>(id->symbol)) {
        fail(ast,
             "member-dependent initialization requires source object "
             "initialization semantics");
      }
      fail(ast, "unbound source value: " +
                    cxx::to_string(id->symbol ? id->symbol->name() : nullptr));
    }
    if (auto* literal = cxx::ast_cast<cxx::BoolLiteralExpressionAST>(ast)) {
      return scalars_.integer(literal->isTrue, LOOM_SCALAR_TYPE_I1, source);
    }
    if (cxx::ast_cast<cxx::IntLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::FloatLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::CharLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::SizeofExpressionAST>(ast) ||
        cxx::ast_cast<cxx::SizeofTypeExpressionAST>(ast) ||
        cxx::ast_cast<cxx::AlignofTypeExpressionAST>(ast) ||
        cxx::ast_cast<cxx::BuiltinOffsetofExpressionAST>(ast)) {
      cxx::ASTInterpreter interpreter(&unit_);
      auto value = interpreter.evaluate(ast);
      if (!value) {
        fail(ast, "literal or layout query has no constant value");
      }
      return scalars_.constant(*value, ast->type, ast);
    }
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      auto* base = cxx::ast_cast<cxx::IdExpressionAST>(member->baseExpression);
      if (control_->storage_backed(member)) {
        return name(load(ast), cxx::to_string(member->symbol->name()));
      }
      bool topology = base && (annotated(base->symbol, "workitem_id") ||
                               annotated(base->symbol, "workgroup_id") ||
                               annotated(base->symbol, "workgroup_size") ||
                               annotated(base->symbol, "workgroup_count"));
      if (!topology) {
        auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
        if (!field || field->isStatic()) {
          fail(ast, "record value access requires a non-static data member");
        }
        auto value = expression(member->baseExpression);
        const auto& slice = types_.member(field, ast);
        return value.project(*slice.partition, slice.component_offset);
      }
      require_kernel_context(ast);
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
      if (binary->op == cxx::TokenKind::T_COMMA) {
        effect(binary->leftExpression);
        return expression(binary->rightExpression);
      }
      auto left = expression(binary->leftExpression);
      if (binary->op == cxx::TokenKind::T_AMP_AMP ||
          binary->op == cxx::TokenKind::T_BAR_BAR) {
        if (types_.vector(ast->type)) {
          fail(ast, "vector logical operators require lane-wise evaluation");
        }
        // The semantic AST supplies both contextual boolean conversions.
        // The skipped arm forwards the left value, preserving its identity
        // without evaluating right-side memory accesses or binding updates.
        auto evaluate_right = [&] {
          return expression(binary->rightExpression);
        };
        auto skipped = [&] { return left; };
        return binary->op == cxx::TokenKind::T_AMP_AMP
                   ? conditional_value(ast, left.ssa(), evaluate_right, skipped)
                   : conditional_value(ast, left.ssa(), skipped,
                                       evaluate_right);
      }
      auto right = expression(binary->rightExpression);
      if (left.is_pointer() || right.is_pointer()) {
        if (left.is_pointer() && !right.is_pointer()) {
          return storage_
              .advance(
                  pointer_projection(left, binary->leftExpression->type, ast),
                  right.ssa(), binary->leftExpression->type,
                  binary->rightExpression->type, binary->op, ast)
              .pointer;
        }
        if (right.is_pointer() && !left.is_pointer() &&
            binary->op == cxx::TokenKind::T_PLUS) {
          return storage_
              .advance(
                  pointer_projection(right, binary->rightExpression->type, ast),
                  left.ssa(), binary->rightExpression->type,
                  binary->leftExpression->type, binary->op, ast)
              .pointer;
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
        auto* operand = cxx::Initializer::stripImplicitCasts(unary->expression);
        while (auto* nested =
                   cxx::ast_cast<cxx::NestedExpressionAST>(operand)) {
          operand = cxx::Initializer::stripImplicitCasts(nested->expression);
        }
        if (auto* dereference = cxx::ast_cast<cxx::UnaryExpressionAST>(operand);
            dereference && !dereference->symbol &&
            dereference->op == cxx::TokenKind::T_STAR) {
          // Taking the address of an indirect object preserves the pointer;
          // it neither reads the object nor needs its storage layout.
          return expression(dereference->expression);
        }
        return object_address(unary->expression).pointer;
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
      if (auto called = atomic_builtin(call)) {
        return *called->value;
      }
      auto* callee = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
      auto* function =
          callee ? cxx::symbol_cast<cxx::FunctionSymbol>(callee->symbol)
                 : nullptr;
      if (!function) {
        fail(ast, "call must resolve to a function symbol");
      }
      if (intrinsics_.check_binding(function, ast) ||
          functions_.is_check_case(function) ||
          annotated(function, "check_benchmark")) {
        fail(ast,
             "check declarations cannot be called from ordinary functions");
      }
      auto* binding = intrinsics_.lookup(function, ast);
      if (binding && std::holds_alternative<SubgroupIntrinsic>(*binding)) {
        require_kernel_context(ast);
      }
      auto* assembly =
          binding ? std::get_if<AssemblyIntrinsic>(binding) : nullptr;
      loom_symbol_ref_t fragment = {};
      auto* expressions = call->expressionList;
      if (assembly) {
        fragment =
            assembly->fragment(unit_, diagnostics_, expressions->value, names_,
                               assembly_fragments_, module_, options_);
        expressions = expressions->next;
      }
      std::vector<Value> source_arguments;
      for (auto* argument : cxx::ListView{expressions}) {
        source_arguments.push_back(expression(argument));
      }
      if (assembly) {
        return *assembly->call(fragment, source_arguments, &builder_, source);
      }
      auto flatten_arguments = [&] {
        std::vector<loom_value_id_t> arguments;
        for (auto argument : source_arguments) {
          argument.append_to(arguments);
        }
        return arguments;
      };
      loom_op_t* op;
      if (annotated(function, "shuffle_xor")) {
        require_kernel_context(ast);
        auto arguments = flatten_arguments();
        if (arguments.size() != 3) {
          fail(ast, "shuffle_xor requires three scalar operands");
        }
        check(loom_kernel_subgroup_shuffle_build(
            &builder_, LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR, arguments[0],
            arguments[1], arguments[2], types_.get(ast->type, ast), source,
            &op));
        return result(op);
      }
      if (binding) {
        auto called =
            intrinsics_.call(*binding, source_arguments, value_arena_, storage_,
                             ast, math_flags_, &builder_, source);
        if (!called.value) {
          fail(ast, "void intrinsic cannot be used as a value");
        }
        return *called.value;
      }
      if (!functions_.definition(function) || annotated(function, "kernel")) {
        fail(
            ast,
            "call must resolve to an owned intrinsic or defined device helper");
      }
      auto symbol = functions_.declare(function);
      auto arguments = flatten_arguments();
      std::array<const cxx::Type*, 1> result_sources = {ast->type};
      auto results = bind_signature(types_, result_sources, ast, &builder_);
      check(loom_func_call_build(
          &builder_, 0, 0, 0, 0, symbol, arguments.data(), arguments.size(),
          results.types.data(), results.types.size(), nullptr, 0, source, &op));
      return value_arena_.capture(types_.partition(ast->type, ast),
                                  {loom_op_results(op), results.types.size()});
    }
    fail(ast,
         "unsupported expression: " + std::string(cxx::to_string(ast->kind())));
  }

  // Each arm yields whether its remaining iteration may execute, followed by
  // the live bindings it changed. Arm-local declarations stay inside the arm.
  template <typename Then, typename Else>
  loom_value_id_t continuing_branch(cxx::AST* owner, loom_value_id_t condition,
                                    Then then_statement, Else else_statement) {
    auto written = live_mutations(owner);
    auto outer_values = values_;
    std::vector<const cxx::Type*> sources = {unit_.control()->getBoolType()};
    sources.reserve(1 + written.size());
    for (auto* symbol : written) {
      sources.push_back(symbol->type());
    }
    auto source = locations_.get(owner);
    auto result_signature = bind_signature(types_, sources, owner, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, result_signature.types.data(),
                            result_signature.types.size(), nullptr, 0, source,
                            &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    std::vector<loom_value_id_t> yielded = {then_statement()};
    auto mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    yielded = {else_statement()};
    mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = std::move(outer_values);
    bind_values(written, loom_op_results(op) + 1);
    return loom_op_results(op)[0];
  }

  loom_value_id_t continuing_sequence(
      cxx::CompoundStatementAST* owner,
      cxx::List<cxx::StatementAST*>* remaining) {
    while (remaining) {
      auto* child = remaining->value;
      remaining = remaining->next;
      auto flow = control_->continues(child);
      if (flow == ExitFlow::None) {
        statement(child);
        continue;
      }
      return continuing_tail(owner, child, remaining);
    }
    return scalars_.integer(true, LOOM_SCALAR_TYPE_I1, locations_.get(owner));
  }

  // A tail enters at most one arm directly. When both arms can reach it, join
  // their fallthrough values first and emit the shared source just once.
  loom_value_id_t continuing_tail(cxx::CompoundStatementAST* owner,
                                  cxx::StatementAST* ast,
                                  cxx::List<cxx::StatementAST*>* remaining) {
    ast = selected_statement(ast);
    auto flow = control_->continues(ast);
    if (flow == ExitFlow::None) {
      if (ast) {
        statement(ast);
      }
      return continuing_sequence(owner, remaining);
    }
    if (flow == ExitFlow::All || !remaining) {
      return continuing_statement(ast);
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch &&
        (control_->continues(branch->statement) == ExitFlow::All ||
         control_->continues(branch->elseStatement) == ExitFlow::All)) {
      auto condition = branch_condition(branch);
      return continuing_branch(
          owner, condition,
          [&] { return continuing_tail(owner, branch->statement, remaining); },
          [&] {
            return continuing_tail(owner, branch->elseStatement, remaining);
          });
    }
    auto fallthrough = continuing_statement(ast);
    return continuing_branch(
        owner, fallthrough,
        [&] { return continuing_sequence(owner, remaining); },
        [&] {
          return scalars_.integer(false, LOOM_SCALAR_TYPE_I1,
                                  locations_.get(ast));
        });
  }

  loom_value_id_t continuing_statement(cxx::StatementAST* ast) {
    auto* source = ast;
    ast = selected_statement(ast);
    if (!ast) {
      return scalars_.integer(true, LOOM_SCALAR_TYPE_I1,
                              locations_.get(source));
    }
    if (control_->continues(ast) == ExitFlow::None) {
      statement(ast);
      return scalars_.integer(true, LOOM_SCALAR_TYPE_I1, locations_.get(ast));
    }
    if (cxx::ast_cast<cxx::ContinueStatementAST>(ast)) {
      return scalars_.integer(false, LOOM_SCALAR_TYPE_I1, locations_.get(ast));
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return continuing_sequence(compound, compound->statementList);
    }
    // The retained path analysis propagates iteration exits through blocks
    // and conditionals only; nested loops consume their own continues.
    auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
    auto condition = branch_condition(branch);
    return continuing_branch(
        ast, condition, [&] { return continuing_statement(branch->statement); },
        [&] {
          return branch->elseStatement
                     ? continuing_statement(branch->elseStatement)
                     : scalars_.integer(true, LOOM_SCALAR_TYPE_I1,
                                        locations_.get(ast));
        });
  }

  // Continue ends only this body. The caller still emits the for step or the
  // do condition, and ordinary bodies preserve their existing structured IR.
  void loop_body(cxx::StatementAST* body) {
    if (control_->continues(body) == ExitFlow::None) {
      statement(body);
    } else {
      continuing_statement(body);
    }
  }

  void statement(cxx::StatementAST* ast) {
    ast = selected_statement(ast);
    if (!ast) {
      return;
    }
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
      if (auto* alias = cxx::ast_cast<cxx::AliasDeclarationAST>(
              declaration->declaration)) {
        reject_misplaced_binding_attributes(unit_, diagnostics_,
                                            alias->attributeList);
        reject_misplaced_binding_attributes(unit_, diagnostics_,
                                            alias->typeId->attributeList);
        reject_misplaced_binding_declarator(unit_, diagnostics_,
                                            alias->typeId->declarator);
        return;
      }
      if (auto* directive =
              cxx::ast_cast<cxx::UsingDirectiveAST>(declaration->declaration)) {
        reject_misplaced_binding_attributes(unit_, diagnostics_,
                                            directive->attributeList);
        return;
      }
      if (cxx::ast_cast<cxx::UsingDeclarationAST>(declaration->declaration)) {
        return;
      }
      auto* simple =
          cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration);
      if (!simple) {
        fail(ast, "unsupported local declaration");
      }
      reject_misplaced_binding_attributes(unit_, diagnostics_,
                                          simple->attributeList);
      for (auto* variable : cxx::ListView{simple->initDeclaratorList}) {
        reject_misplaced_binding_declarator(unit_, diagnostics_,
                                            variable->declarator);
        if (cxx::symbol_cast<cxx::TypeAliasSymbol>(variable->symbol)) {
          continue;
        }
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
          auto allocation =
              storage_.allocate(array, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
                                source_variable->explicitAlignment(), variable);
          auto spelling = cxx::to_string(variable->symbol->name());
          values_[variable->symbol] = name(Value(allocation.pointer), spelling);
          name(allocation.view, spelling + "_view");
          continue;
        }
        initialize_variable(source_variable, variable->initializer, variable);
      }
      return;
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast)) {
      auto condition = branch_condition(branch);
      auto saved_values = values_;
      auto written = live_mutations(ast);
      std::vector<const cxx::Type*> sources;
      sources.reserve(written.size());
      for (auto* symbol : written) {
        sources.push_back(symbol->type());
      }
      auto results = bind_signature(types_, sources, ast, &builder_);
      loom_op_t* op;
      auto flags = branch->elseStatement || !written.empty()
                       ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION
                       : 0;
      check(loom_scf_if_build(&builder_, flags, condition, results.types.data(),
                              results.types.size(), nullptr, 0,
                              locations_.get(ast), &op));
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
                       LoopTest::BeforeBody, loop->decisionVariable);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(ast)) {
      if (!LoopSchedule(unit_, diagnostics_, loop->attributeList).empty()) {
        fail(ast, "loop scheduling requires a counted for loop");
      }
      conditional_loop(ast, loop->condition, loop->statement, nullptr,
                       LoopTest::BeforeBody, loop->decisionVariable);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(ast)) {
      if (!LoopSchedule(unit_, diagnostics_, loop->attributeList).empty()) {
        fail(ast, "loop scheduling requires a counted for loop");
      }
      conditional_loop(ast, loop->expression, loop->statement, nullptr,
                       LoopTest::AfterBody, nullptr);
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
                        LoopTest test, cxx::VariableSymbol* decision) {
    initialize_condition(decision);
    auto written = live_mutations(ast);
    if (decision && values_.contains(decision)) {
      std::erase(written, decision);
      written.push_back(decision);
    }
    auto initial = current(written);
    auto outer_values = values_;
    auto source = locations_.get(ast);
    loom_op_t* op;
    check(loom_scf_while_build(&builder_, initial.data(), initial.size(),
                               /*result_types=*/nullptr, nullptr, 0, source,
                               &op));
    auto* before = loom_scf_while_before(op);
    auto saved = loom_builder_enter_region(&builder_, op, before);
    bind(written, before);
    if (test == LoopTest::AfterBody) {
      loop_body(body);
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
      loop_body(body);
    }
    if (step) {
      effect(step);
    }
    // A decision object is recreated after the body and for-loop increment.
    // The preheader supplied its first value, so every check sees a real value.
    if (decision) {
      auto value = expression(decision->initializer());
      if (auto found = locals_.find(decision); found != locals_.end()) {
        storage_.store({found->second.view, std::nullopt}, value.ssa(),
                       decision->type(), ast);
      } else {
        values_[decision] = name(value, cxx::to_string(decision->name()));
      }
    }
    auto yielded = current(written);
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &terminator));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    bind_values(written, loom_op_results(op));
    if (decision) {
      values_.erase(decision);
      locals_.erase(decision);
    }
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
    auto* constant_upper = std::get_if<unsigned>(&counted.upper);
    auto bound =
        constant_upper
            ? scalars_.integer(static_cast<int32_t>(*constant_upper),
                               LOOM_SCALAR_TYPE_I32, source)
            : expression(std::get<CountedLoop::Bound>(counted.upper).expression)
                  .ssa();
    auto upper = unsigned_offset(bound, source);
    auto step = scalars_.integer(counted.step, LOOM_SCALAR_TYPE_OFFSET, source);
    auto depth = schedule.pipeline_depth()
                     ? expression(schedule.pipeline_depth()).ssa()
                     : 0;
    auto factor = schedule.unroll_factor()
                      ? expression(schedule.unroll_factor()).ssa()
                      : 0;
    auto written = live_mutations(loop);
    std::erase(written, induction);
    auto initial = current(written);
    auto outer_values = values_;
    auto* op = schedule.build(&builder_, lower, upper, step, initial, depth,
                              factor, source);
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
    loop_body(loop->statement);
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
      const auto& partition = values_.at(symbol).partition();
      values_[symbol] = name(region_value(region, index, partition),
                             cxx::to_string(symbol->name()));
    }
  }
  void bind_values(const std::vector<cxx::Symbol*>& symbols,
                   const loom_value_id_t* components) {
    for (auto* symbol : symbols) {
      const auto& partition = values_.at(symbol).partition();
      auto count = partition.component_count;
      values_[symbol] =
          name(value_arena_.capture(partition, {components, count}),
               cxx::to_string(symbol->name()));
      if (count) {
        components += count;
      }
    }
  }

  void effect(cxx::ExpressionAST* ast) {
    if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast);
        binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
      effect(binary->leftExpression);
      effect(binary->rightExpression);
      return;
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      effect(nested->expression);
      return;
    }
    if (types_.unqualified(ast->type)->kind() == cxx::TypeKind::kVoid) {
      if (auto* cast = cxx::ast_cast<cxx::CastExpressionAST>(ast)) {
        effect(cast->expression);
        return;
      }
      if (auto* cast = cxx::ast_cast<cxx::CppCastExpressionAST>(ast)) {
        effect(cast->expression);
        return;
      }
      if (auto* cast = cxx::ast_cast<cxx::TypeConstructionAST>(ast)) {
        for (auto* operand : cxx::ListView{cast->expressionList}) {
          effect(operand);
        }
        return;
      }
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(ast)) {
      if (atomic_builtin(call)) {
        return;
      }
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
        if (function && (intrinsics_.check_binding(function, ast) ||
                         functions_.is_check_case(function) ||
                         annotated(function, "check_benchmark"))) {
          fail(ast,
               "check declarations cannot be called from ordinary functions");
        }
        auto* binding = function ? intrinsics_.lookup(function, ast) : nullptr;
        if (binding) {
          auto* assembly = std::get_if<AssemblyIntrinsic>(binding);
          loom_symbol_ref_t fragment = {};
          auto* expressions = call->expressionList;
          if (assembly) {
            fragment = assembly->fragment(
                unit_, diagnostics_, expressions->value, names_,
                assembly_fragments_, module_, options_);
            expressions = expressions->next;
          }
          std::vector<Value> arguments;
          for (auto* argument : cxx::ListView{expressions}) {
            arguments.push_back(expression(argument));
          }
          if (assembly) {
            assembly->call(fragment, arguments, &builder_, locations_.get(ast));
            return;
          }
          auto called =
              intrinsics_.call(*binding, arguments, value_arena_, storage_, ast,
                               math_flags_, &builder_, locations_.get(ast));
          if (called.value) {
            fail(ast, "value-producing intrinsic reached a void call");
          }
          return;
        }
        if (!function || !functions_.definition(function) ||
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
    expression(ast);
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
  // Shared output namespace for callables and configuration symbols.
  SymbolNames names_;
  // Namespace-scope scalar configs retain key identity across source aliases.
  Configs configs_;
  // Explicit vector builders retain lane widths and full-width source masks.
  Vectors vectors_;
  // Memory representations retain declared array extents and access shape.
  Storage storage_;
  // Retained generated operation bindings for reached source declarations.
  Intrinsics intrinsics_;
  // Literal admission records one batch for source-boundary verification.
  AssemblyFragments assembly_fragments_;
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
  // Immutable large source bindings, released after each function's maps.
  ValueArena value_arena_;
  // Bound symbols, never identifier spellings, key source-to-SSA mappings.
  std::unordered_map<cxx::Symbol*, Value> values_;
  // Storage-backed automatic objects retain one allocation across direct and
  // aliased accesses. They are not mutable SSA bindings at region edges.
  std::unordered_map<cxx::Symbol*, StorageAllocation> locals_;
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
