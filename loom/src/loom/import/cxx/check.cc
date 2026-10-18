// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/check.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
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

class SymbolReference final : private cxx::ASTVisitor {
 public:
  static bool find(cxx::AST* source, cxx::Symbol* target) {
    SymbolReference search(target);
    search.accept(source);
    return search.found_;
  }

 private:
  explicit SymbolReference(cxx::Symbol* target) : target_(target) {}

  bool preVisit(cxx::AST* ast) override {
    if (found_) {
      return false;
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast);
        id && id->symbol == target_) {
      found_ = true;
      return false;
    }
    return true;
  }

  cxx::Symbol* target_;
  bool found_ = false;
};

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

  void translate_case(const FunctionBody& body) {
    reject_misplaced_binding_statement(unit_, diagnostics_, body.body,
                                       BindingAttributeScope::CheckStatement);
    cxx::AST* end = body.body;
    for (auto* remaining = body.body->statementList; remaining;
         remaining = remaining->next) {
      auto* statement = remaining->value;
      reject_misplaced_binding_statement(unit_, diagnostics_, statement,
                                         BindingAttributeScope::CheckStatement);
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

  void translate_scenario(const FunctionBody& body) {
    reject_misplaced_binding_statement(unit_, diagnostics_, body.body,
                                       BindingAttributeScope::CheckStatement);
    values_.clear();
    value_arena_.reset();
    observing_ = false;
    bind_scenario_parameters(body);

    cxx::AST* end = body.body;
    bool has_trial = false;
    for (auto* remaining = body.body->statementList; remaining;
         remaining = remaining->next) {
      auto* statement = remaining->value;
      reject_misplaced_binding_statement(unit_, diagnostics_, statement,
                                         BindingAttributeScope::CheckStatement);
      if (auto* returned = cxx::ast_cast<cxx::ReturnStatementAST>(statement)) {
        if (returned->expression || remaining->next) {
          fail(statement, "check scenario return must be bare and final");
        }
        end = returned;
        break;
      }
      auto* expression_statement =
          cxx::ast_cast<cxx::ExpressionStatementAST>(statement);
      if (!expression_statement || !expression_statement->expression) {
        fail(statement,
             "check scenarios require direct trial calls and an optional "
             "final return");
      }
      auto* call = direct_call(expression_statement->expression);
      auto* function = call ? callee(call) : nullptr;
      auto* binding =
          function ? intrinsics_.check_binding(function, call) : nullptr;
      if (!binding || binding->operation != CheckIntrinsic::Operation::Trial) {
        fail(statement, "check scenario statements must be trial calls");
      }
      translate_trial(call, *binding);
      has_trial = true;
    }
    if (!has_trial) {
      fail(body.body, "check scenarios require at least one trial");
    }
    loom_op_t* terminator;
    check(loom_check_return_build(&builder_, locations_.get(end), &terminator));
  }

 private:
  [[noreturn]] void fail(cxx::AST* source, const char* message) {
    diagnostics_.reject(unit_, source, message);
  }

  void bind_scenario_parameters(const FunctionBody& body) {
    auto parameters = body.source->symbol->parameters();
    auto* block = loom_region_entry_block(body.region);
    if (parameters.empty()) {
      if (block->arg_count != 0) {
        fail(body.source,
             "unconfigured check scenario cannot have region arguments");
      }
      return;
    }
    if (parameters.size() != 2 || block->arg_count != 2) {
      fail(body.source,
           "configured check scenario requires ordinal and entropy region "
           "arguments");
    }
    name(block->arg_ids[0], parameters[0]);
    name(block->arg_ids[1], parameters[1]);
    if (SymbolReference::find(body.body, parameters[0])) {
      const auto& ordinal =
          types_.partition(parameters[0]->type(), body.source);
      values_[parameters[0]] = name(
          value_arena_.capture(ordinal, {block->arg_ids, 1}), parameters[0]);
    }
    if (SymbolReference::find(body.body, parameters[1])) {
      const auto& entropy =
          types_.partition(parameters[1]->type(), body.source);
      values_[parameters[1]] =
          name(value_arena_.capture(entropy, {block->arg_ids + 1, 1}),
               parameters[1]);
    }
  }

  cxx::LambdaExpressionAST* lambda_expression(cxx::ExpressionAST* source) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
      return lambda_expression(nested->expression);
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
      return lambda_expression(initializer->expression);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source)) {
      if (!cast->conversionFunction) {
        return lambda_expression(cast->expression);
      }
    }
    return cxx::ast_cast<cxx::LambdaExpressionAST>(source);
  }

  std::unordered_map<cxx::Symbol*, Value> capture_values(
      cxx::LambdaExpressionAST* lambda) {
    std::unordered_map<cxx::Symbol*, Value> result;
    for (auto* capture : cxx::ListView{lambda->captureList}) {
      cxx::FieldSymbol* field = nullptr;
      cxx::ExpressionAST* initializer = nullptr;
      if (auto* simple = cxx::ast_cast<cxx::SimpleLambdaCaptureAST>(capture)) {
        field = simple->symbol;
        initializer = simple->initializer;
      } else if (auto* reference =
                     cxx::ast_cast<cxx::RefLambdaCaptureAST>(capture)) {
        field = reference->symbol;
        initializer = reference->initializer;
      } else if (auto* initialized =
                     cxx::ast_cast<cxx::InitLambdaCaptureAST>(capture)) {
        field = initialized->symbol;
        initializer = initialized->initializer;
      } else if (auto* initialized =
                     cxx::ast_cast<cxx::RefInitLambdaCaptureAST>(capture)) {
        field = initialized->symbol;
        initializer = initialized->initializer;
      } else {
        fail(capture, "check lambdas cannot capture this");
      }
      if (!field || !initializer) {
        fail(capture, "check lambda capture has no resolved initializer");
      }
      result.emplace(field, expression(initializer));
    }
    return result;
  }

  std::vector<cxx::ParameterDeclarationAST*> lambda_parameters(
      cxx::LambdaExpressionAST* lambda) {
    std::vector<cxx::ParameterDeclarationAST*> result;
    auto* clause = lambda->parameterDeclarationClause;
    if (!clause) {
      return result;
    }
    if (clause->isVariadic) {
      fail(clause, "check lambdas cannot be variadic");
    }
    for (auto* parameter : cxx::ListView{clause->parameterDeclarationList}) {
      if (!parameter->symbol || !parameter->type || parameter->isPack) {
        fail(parameter, "check lambda parameters require resolved fixed types");
      }
      result.push_back(parameter);
    }
    return result;
  }

  void translate_trial(cxx::CallExpressionAST* call,
                       const CheckIntrinsic& binding) {
    auto* argument = call->expressionList;
    auto* lambda = argument && !argument->next
                       ? lambda_expression(argument->value)
                       : nullptr;
    if (!lambda || !lambda->statement) {
      fail(call,
           "check.trial requires one inline lambda taking ordinal and "
           "entropy");
    }
    auto parameters = lambda_parameters(lambda);
    if (parameters.size() != 2 ||
        !types_.is_index(parameters[0]->type, parameters[0]) ||
        !types_.is_opaque_dialect(parameters[1]->type, "check.entropy",
                                  parameters[1])) {
      fail(lambda,
           "check.trial lambda requires (loom::check::ordinal, "
           "loom::check::entropy)");
    }

    auto captures = capture_values(lambda);
    auto entropy_type = types_.get(parameters[1]->type, parameters[1]);
    const std::array<loom_type_t, 2> argument_types = {
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), entropy_type};
    loom_op_t* op;
    check(loom_check_trial_build(&builder_, binding.trial_count,
                                 argument_types.data(), argument_types.size(),
                                 locations_.get(call), &op));

    auto outer_values = std::move(values_);
    auto outer_observing = observing_;
    auto outer_in_trial = in_trial_;
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_check_trial_body(op));
    values_ = std::move(captures);
    observing_ = false;
    in_trial_ = true;
    auto* block = loom_region_entry_block(loom_check_trial_body(op));
    name(block->arg_ids[0], parameters[0]->symbol);
    name(block->arg_ids[1], parameters[1]->symbol);
    if (SymbolReference::find(lambda->statement, parameters[0]->symbol)) {
      const auto& ordinal =
          types_.partition(parameters[0]->type, parameters[0]);
      values_[parameters[0]->symbol] =
          name(value_arena_.capture(ordinal, {block->arg_ids, 1}),
               parameters[0]->symbol);
    }
    if (SymbolReference::find(lambda->statement, parameters[1]->symbol)) {
      const auto& entropy_partition =
          types_.partition(parameters[1]->type, parameters[1]);
      values_[parameters[1]->symbol] =
          name(value_arena_.capture(entropy_partition, {block->arg_ids + 1, 1}),
               parameters[1]->symbol);
    }
    translate_trial_statements(lambda);
    loom_builder_restore(&builder_, saved);
    values_ = std::move(outer_values);
    observing_ = outer_observing;
    in_trial_ = outer_in_trial;
  }

  void translate_trial_statements(cxx::LambdaExpressionAST* lambda) {
    bool has_action = false;
    for (auto* remaining = lambda->statement->statementList; remaining;
         remaining = remaining->next) {
      auto* statement = remaining->value;
      reject_misplaced_binding_statement(unit_, diagnostics_, statement,
                                         BindingAttributeScope::CheckStatement);
      if (auto* declaration =
              cxx::ast_cast<cxx::DeclarationStatementAST>(statement)) {
        if (has_action) {
          fail(statement, "check trial action must be final");
        }
        local(declaration);
        continue;
      }
      auto* expression_statement =
          cxx::ast_cast<cxx::ExpressionStatementAST>(statement);
      auto* call = expression_statement && expression_statement->expression
                       ? direct_call(expression_statement->expression)
                       : nullptr;
      auto* function = call ? callee(call) : nullptr;
      auto* binding =
          function ? intrinsics_.check_binding(function, call) : nullptr;
      if (!binding) {
        fail(statement,
             "check trial statements require value-source operations and one "
             "final compare or invoke");
      }
      using Operation = CheckIntrinsic::Operation;
      if (binding->operation == Operation::Compare ||
          binding->operation == Operation::Invoke) {
        if (has_action || remaining->next) {
          fail(statement, "check trial compare or invoke must be final");
        }
        translate_action(call, *binding);
        has_action = true;
        continue;
      }
      if (has_action || binding->is_observation() ||
          binding->operation == Operation::Launch ||
          binding->operation == Operation::Requires ||
          binding->operation == Operation::Trial) {
        fail(statement, "operation is not a check trial value source");
      }
      if (check_call(call, *binding)) {
        fail(statement,
             "check trial value-source results require an immutable binding");
      }
    }
    if (!has_action) {
      fail(lambda, "check trial requires one final compare or invoke");
    }
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

  void name(loom_value_id_t value, std::string_view hint) {
    if (hint.empty() || loom_module_value(builder_.module, value)->name_id !=
                            LOOM_STRING_ID_INVALID) {
      return;
    }
    check(loom_module_set_value_name(builder_.module, value, intern(hint)));
  }

  void name(loom_value_id_t value, cxx::Symbol* symbol) {
    if (symbol && symbol->name()) {
      name(value, cxx::to_string(symbol->name()));
    }
  }

  std::string spelling(cxx::Symbol* symbol, std::string_view suffix = {}) {
    if (!symbol || !symbol->name()) {
      return {};
    }
    return cxx::to_string(symbol->name()) + std::string(suffix);
  }

  Value name(Value value, std::string_view hint) {
    auto components = value.components();
    if (value.is_record()) {
      const auto& record =
          static_cast<const RecordPartition&>(value.partition());
      for (size_t index = 0; index < components.size(); ++index) {
        name(components[index],
             std::string(hint) + "_" + record.component_names[index]);
      }
    } else if (value.is_view()) {
      const auto& view = static_cast<const ViewPartition&>(value.partition());
      for (size_t index = 0; index < components.size(); ++index) {
        const auto& suffix = view.component_names[index];
        if (suffix.empty()) {
          name(components[index], hint);
        } else {
          name(components[index], std::string(hint) + "_" + suffix);
        }
      }
    } else if (value.is_tensor()) {
      const auto& tensor =
          static_cast<const TensorPartition&>(value.partition());
      if (tensor.is_dynamic()) {
        name(components[0], std::string(hint) + "_dim0");
      }
      name(tensor.tensor_value(components), hint);
    } else if (value.is_pointer()) {
      name(value.pointer().root, hint);
      name(value.pointer().byte_offset, std::string(hint) + "_byte_offset");
    } else {
      name(components[0], hint);
    }
    return value;
  }

  loom_value_id_t dynamic_tensor_extent(cxx::ExpressionAST* source) {
    return expression(source).ssa();
  }

  loom_type_t tensor_type(const TensorPartition& tensor,
                          loom_value_id_t dynamic_extent) {
    return tensor.is_dynamic()
               ? tensor.bind_type(
                     std::span<const loom_value_id_t>(&dynamic_extent, 1))
               : tensor.bind_type({});
  }

  Value capture_tensor(const TensorPartition& tensor,
                       loom_value_id_t dynamic_extent, loom_op_t* op) {
    std::array<loom_value_id_t, 2> components;
    size_t count = 0;
    if (tensor.is_dynamic()) {
      components[count++] = dynamic_extent;
    }
    components[count++] = loom_op_results(op)[0];
    return value_arena_.capture(
        tensor, std::span<const loom_value_id_t>(components).first(count));
  }

  void append_subject_value(Value value, std::vector<loom_value_id_t>& output) {
    if (value.is_tensor()) {
      const auto& tensor =
          static_cast<const TensorPartition&>(value.partition());
      output.push_back(tensor.tensor_value(value.components()));
    } else {
      value.append_to(output);
    }
  }

  Value name(Value value, cxx::Symbol* symbol) {
    if (!symbol || !symbol->name()) {
      return value;
    }
    return name(value, cxx::to_string(symbol->name()));
  }

  std::string_view string_literal(cxx::ExpressionAST* source) {
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
      return string_literal(initializer->expression);
    }
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
      fail(source, "check attributes require a string literal");
    }
    return literal->literal->stringValue();
  }

  loom_attribute_t constant(cxx::ExpressionAST* source) {
    if (auto value = scalar_constant(unit_, source)) {
      return scalars_.constant_attribute(*value, source->type, source);
    }
    fail(source, "check attributes require pure scalar constants");
  }

  cxx::CallExpressionAST* direct_call(cxx::ExpressionAST* source) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
      return direct_call(nested->expression);
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
      return direct_call(initializer->expression);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source)) {
      if (!cast->conversionFunction) {
        return direct_call(cast->expression);
      }
    }
    return cxx::ast_cast<cxx::CallExpressionAST>(source);
  }

  const cxx::Type* storage_element_type(const cxx::Type* source_type,
                                        const Partition& partition) {
    if (auto* pointer =
            cxx::type_cast<cxx::PointerType>(types_.unqualified(source_type))) {
      return pointer->elementType();
    }
    return partition.kind == ValueKind::Buffer
               ? static_cast<const BufferPartition&>(partition).element_type
               : nullptr;
  }

  bool tensor_matches_storage(const TensorPartition& tensor,
                              const cxx::Type* source_type,
                              const Partition& partition) {
    auto* element = storage_element_type(source_type, partition);
    if (!element) {
      return false;
    }
    if (types_.unqualified(element) ==
        types_.unqualified(tensor.element_type)) {
      return true;
    }
    return partition.kind == ValueKind::Buffer && tensor.vector_lane_count &&
           types_.unqualified(element) == types_.unqualified(tensor.lane_type);
  }

  void append_workloads(cxx::ExpressionAST* source,
                        const CheckIntrinsic& binding,
                        std::vector<loom_value_id_t>& workloads) {
    auto* call = direct_call(source);
    auto* function = call ? callee(call) : nullptr;
    if (!function || !annotated(function, "workload")) {
      fail(source,
           "kernel workloads require a direct "
           "loom::kernel::workload(...) call");
    }
    auto parameters = binding.configuration->parameters();
    auto* argument = call->expressionList;
    for (auto* parameter : parameters) {
      if (!argument) {
        fail(source,
             "loom::kernel::workload argument count must match the "
             "kernel configuration");
      }
      auto* parameter_type = types_.unqualified(parameter->type());
      const auto& parameter_partition =
          types_.partition(parameter_type, argument->value);
      auto value = expression(argument->value);
      if (value.is_tensor() &&
          (parameter_partition.kind == ValueKind::Pointer ||
           parameter_partition.kind == ValueKind::Buffer)) {
        const auto& tensor =
            static_cast<const TensorPartition&>(value.partition());
        if (!tensor_matches_storage(tensor, parameter_type,
                                    parameter_partition)) {
          fail(argument->value,
               "check tensor element type must match the kernel "
               "configuration storage element type");
        }
        workloads.push_back(tensor.tensor_value(value.components()));
        if (parameter_partition.kind == ValueKind::Pointer) {
          workloads.push_back(scalars_.integer(
              0, LOOM_SCALAR_TYPE_OFFSET, locations_.get(argument->value)));
        }
        argument = argument->next;
        continue;
      }
      if (types_.unqualified(argument->value->type) != parameter_type) {
        fail(argument->value,
             "loom::kernel::workload argument types must match the kernel "
             "configuration parameters");
      }
      if (value.components().size() != parameter_partition.component_count) {
        fail(argument->value,
             "kernel workload representation does not match the kernel "
             "configuration parameter");
      }
      value.append_to(workloads);
      argument = argument->next;
    }
    if (argument) {
      fail(source,
           "loom::kernel::workload argument count must match the "
           "kernel configuration");
    }
  }

  double tolerance(cxx::ExpressionAST* source) {
    double value = loom_attr_as_f64(constant(source));
    if (!std::isfinite(value) || value < 0.0) {
      fail(source, "check tolerances must be finite and non-negative");
    }
    return value;
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

  bool is_kernel(cxx::FunctionSymbol* function) const {
    return annotated(function, "kernel");
  }

  void require_subject(cxx::FunctionSymbol* function, cxx::AST* source,
                       bool allow_kernel) {
    if (!function || !functions_.definition(function) ||
        functions_.is_check_record(function) ||
        annotated(function, "check_benchmark") ||
        (!allow_kernel && is_kernel(function))) {
      fail(source,
           allow_kernel
               ? "check actions require a defined ordinary function or kernel"
               : "check generators require a defined ordinary function");
    }
  }

  std::vector<loom_type_t> subject_results(cxx::FunctionSymbol* function,
                                           cxx::AST* source) {
    require_subject(function, source, /*allow_kernel=*/true);
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    if (!signature || signature->isVariadic()) {
      fail(source, "check subjects require a fixed function signature");
    }
    std::vector<loom_type_t> results;
    if (!unit_.typeTraits().is_void(signature->returnType())) {
      if (is_kernel(function)) {
        fail(source, "check kernels cannot return values");
      }
      types_.append(signature->returnType(), source, results);
    }
    return results;
  }

  void validate_subject_arguments(
      cxx::FunctionSymbol* function,
      std::span<cxx::ExpressionAST* const> source_arguments,
      std::span<const Value> values, cxx::AST* source) {
    auto parameters = function->parameters();
    if (parameters.size() != source_arguments.size() ||
        values.size() != source_arguments.size()) {
      fail(source, "check action arguments must match the subject signature");
    }
    bool kernel = is_kernel(function);
    for (size_t index = 0; index < parameters.size(); ++index) {
      auto* parameter_type = parameters[index]->type();
      auto* argument_type = source_arguments[index]->type;
      auto expected = types_.get(parameter_type, source);
      const auto& expected_partition =
          types_.partition(parameter_type, source_arguments[index]);
      if (values[index].is_tensor() &&
          loom_type_kind(expected) == LOOM_TYPE_BUFFER) {
        const auto& tensor =
            static_cast<const TensorPartition&>(values[index].partition());
        if (!tensor_matches_storage(tensor, parameter_type,
                                    expected_partition)) {
          fail(source_arguments[index],
               "check tensor element type must match the subject buffer "
               "element type");
        }
        if (kernel || expected_partition.kind == ValueKind::Buffer) {
          continue;
        }
      }
      if (kernel && loom_type_kind(expected) == LOOM_TYPE_BUFFER) {
        if (!values[index].is_tensor()) {
          fail(source_arguments[index],
               "kernel buffer arguments require check tensor handles");
        }
        continue;
      }
      if (values[index].components().size() !=
          expected_partition.component_count) {
        if (!kernel && expected_partition.kind == ValueKind::Pointer &&
            values[index].is_tensor()) {
          fail(source_arguments[index],
               "ordinary C++ pointer parameters retain a byte offset and "
               "cannot bind a check tensor handle");
        }
        fail(source_arguments[index],
             "check action argument representation does not match the "
             "subject parameter");
      }
      if (types_.unqualified(parameter_type) !=
          types_.unqualified(argument_type)) {
        fail(source_arguments[index],
             "check action scalar and record arguments require the subject "
             "parameter type");
      }
    }
  }

  struct ActionOperands {
    std::vector<loom_value_id_t> parameters;
    std::vector<loom_value_id_t> arguments;
    std::vector<cxx::ExpressionAST*> sources;
  };

  ActionOperands action_operands(cxx::CallExpressionAST* call,
                                 const CheckIntrinsic& binding,
                                 bool has_comparison_lambda) {
    std::vector<cxx::ExpressionAST*> sources;
    for (auto* argument : cxx::ListView{call->expressionList}) {
      sources.push_back(argument);
    }
    if (has_comparison_lambda) {
      if (sources.empty()) {
        fail(call, "check.compare requires a final comparison lambda");
      }
      sources.pop_back();
    }

    ActionOperands result;
    size_t first_argument = 0;
    if (binding.configuration && !binding.configuration->parameters().empty()) {
      if (sources.empty()) {
        fail(call, "configured check subject requires launch workloads");
      }
      append_workloads(sources[0], binding, result.parameters);
      first_argument = 1;
    }
    std::vector<Value> values;
    values.reserve(sources.size() - first_argument);
    for (size_t index = first_argument; index < sources.size(); ++index) {
      values.push_back(expression(sources[index]));
      append_subject_value(values.back(), result.arguments);
    }
    result.sources.assign(sources.begin() + first_argument, sources.end());
    validate_subject_arguments(binding.subject, result.sources, values, call);
    if (binding.oracle) {
      validate_subject_arguments(binding.oracle, result.sources, values, call);
    }
    return result;
  }

  void bind_comparison_parameters(cxx::LambdaExpressionAST* lambda,
                                  loom_region_t* region,
                                  std::span<const loom_type_t> actual_types,
                                  std::span<const loom_type_t> expected_types) {
    auto parameters = lambda_parameters(lambda);
    if (actual_types.empty() && expected_types.empty()) {
      if (!parameters.empty()) {
        fail(lambda,
             "void check subjects require a parameterless comparison lambda");
      }
      return;
    }
    if (parameters.size() != 2) {
      fail(lambda,
           "value-returning check subjects require actual and expected "
           "comparison parameters");
    }
    auto bind = [&](cxx::ParameterDeclarationAST* parameter,
                    std::span<const loom_type_t> expected, size_t offset) {
      std::vector<loom_type_t> parameter_types;
      types_.append(parameter->type, parameter, parameter_types);
      if (parameter_types.size() != expected.size()) {
        fail(parameter,
             "comparison parameter representation must match the subject "
             "result");
      }
      for (size_t index = 0; index < expected.size(); ++index) {
        if (!loom_type_equal(parameter_types[index], expected[index])) {
          fail(parameter,
               "comparison parameter types must match the subject result");
        }
      }
      const auto& partition = types_.partition(parameter->type, parameter);
      auto* block = loom_region_entry_block(region);
      values_[parameter->symbol] =
          name(value_arena_.capture(partition,
                                    {block->arg_ids + offset, expected.size()}),
               parameter->symbol);
    };
    bind(parameters[0], actual_types, 0);
    bind(parameters[1], expected_types, actual_types.size());
  }

  void translate_comparison(cxx::LambdaExpressionAST* lambda,
                            loom_region_t* region,
                            std::span<const loom_type_t> actual_types,
                            std::span<const loom_type_t> expected_types,
                            std::unordered_map<cxx::Symbol*, Value> captures) {
    auto outer_values = std::move(values_);
    auto outer_observing = observing_;
    values_ = std::move(captures);
    observing_ = false;
    bind_comparison_parameters(lambda, region, actual_types, expected_types);
    cxx::AST* end = lambda->statement;
    for (auto* remaining = lambda->statement->statementList; remaining;
         remaining = remaining->next) {
      auto* statement = remaining->value;
      reject_misplaced_binding_statement(unit_, diagnostics_, statement,
                                         BindingAttributeScope::CheckStatement);
      if (auto* returned = cxx::ast_cast<cxx::ReturnStatementAST>(statement)) {
        if (returned->expression || remaining->next) {
          fail(statement, "check comparison return must be bare and final");
        }
        end = returned;
        break;
      }
      auto* expression_statement =
          cxx::ast_cast<cxx::ExpressionStatementAST>(statement);
      auto* call = expression_statement && expression_statement->expression
                       ? direct_call(expression_statement->expression)
                       : nullptr;
      auto* function = call ? callee(call) : nullptr;
      auto* binding =
          function ? intrinsics_.check_binding(function, call) : nullptr;
      if (!binding || !binding->is_observation()) {
        fail(statement,
             "check comparison bodies require direct expectation calls");
      }
      check_call(call, *binding);
    }
    loom_op_t* terminator;
    check(loom_check_return_build(&builder_, locations_.get(end), &terminator));
    values_ = std::move(outer_values);
    observing_ = outer_observing;
  }

  void translate_action(cxx::CallExpressionAST* call,
                        const CheckIntrinsic& binding) {
    using Operation = CheckIntrinsic::Operation;
    require_subject(binding.subject, call, /*allow_kernel=*/true);
    auto operands =
        action_operands(call, binding, binding.operation == Operation::Compare);
    auto actual_types = subject_results(binding.subject, call);
    loom_op_t* op;
    if (binding.operation == Operation::Invoke) {
      check(loom_check_invoke_build(
          &builder_, functions_.declare(binding.subject, call),
          operands.parameters.data(), operands.parameters.size(),
          operands.arguments.data(), operands.arguments.size(),
          actual_types.data(), actual_types.size(), nullptr, 0,
          locations_.get(call), &op));
      return;
    }

    auto* last = call->expressionList;
    while (last && last->next) {
      last = last->next;
    }
    auto* lambda = last ? lambda_expression(last->value) : nullptr;
    if (!lambda || !lambda->statement) {
      fail(call, "check.compare requires a final inline comparison lambda");
    }
    auto captures = capture_values(lambda);
    auto expected_types =
        binding.oracle ? subject_results(binding.oracle, call) : actual_types;
    if (actual_types.size() != expected_types.size()) {
      fail(call, "check target and oracle result counts must match");
    }
    loom_check_compare_build_flags_t flags = 0;
    loom_symbol_ref_t oracle = {};
    if (binding.oracle) {
      flags |= LOOM_CHECK_COMPARE_BUILD_FLAG_HAS_ORACLE_CALLEE;
      oracle = functions_.declare(binding.oracle, call);
    }
    check(loom_check_compare_build(
        &builder_, flags, functions_.declare(binding.subject, call), oracle,
        operands.parameters.data(), operands.parameters.size(),
        operands.arguments.data(), operands.arguments.size(),
        actual_types.data(), actual_types.size(), expected_types.data(),
        expected_types.size(), locations_.get(call), &op));
    auto saved = loom_builder_enter_region(&builder_, op,
                                           loom_check_compare_comparison(op));
    translate_comparison(lambda, loom_check_compare_comparison(op),
                         actual_types, expected_types, std::move(captures));
    loom_builder_restore(&builder_, saved);
  }

  std::optional<Value> check_call(cxx::CallExpressionAST* call,
                                  const CheckIntrinsic& binding) {
    using Operation = CheckIntrinsic::Operation;
    auto* first = call->expressionList;
    auto location = locations_.get(call);
    loom_op_t* op;
    switch (binding.operation) {
      case Operation::Generate: {
        require_subject(binding.subject, call, /*allow_kernel=*/false);
        std::vector<cxx::ExpressionAST*> sources;
        std::vector<Value> values;
        std::vector<loom_value_id_t> arguments;
        for (auto* argument : cxx::ListView{call->expressionList}) {
          sources.push_back(argument);
          values.push_back(expression(argument));
          append_subject_value(values.back(), arguments);
        }
        validate_subject_arguments(binding.subject, sources, values, call);
        auto results = subject_results(binding.subject, call);
        check(loom_check_generate_build(
            &builder_, functions_.declare(binding.subject, call),
            arguments.data(), arguments.size(), results.data(), results.size(),
            nullptr, 0, location, &op));
        if (types_.unqualified(call->type)->kind() == cxx::TypeKind::kVoid) {
          return std::nullopt;
        }
        return value_arena_.capture(types_.partition(call->type, call),
                                    {loom_op_results(op), op->result_count});
      }
      case Operation::Fill: {
        auto* payload = first;
        loom_value_id_t dynamic_extent = LOOM_VALUE_ID_INVALID;
        if (binding.result_tensor->is_dynamic()) {
          dynamic_extent = dynamic_tensor_extent(payload->value);
          payload = payload->next;
        }
        check(loom_check_generate_fill_build(
            &builder_, constant(payload->value),
            tensor_type(*binding.result_tensor, dynamic_extent), location,
            &op));
        return capture_tensor(*binding.result_tensor, dynamic_extent, op);
      }
      case Operation::Iota: {
        auto* offset = first;
        loom_value_id_t dynamic_extent = LOOM_VALUE_ID_INVALID;
        if (binding.result_tensor->is_dynamic()) {
          dynamic_extent = dynamic_tensor_extent(offset->value);
          offset = offset->next;
        }
        auto* step = offset->next;
        loom_check_generate_iota_build_flags_t flags = 0;
        int64_t period = 0;
        if (step->next) {
          auto value = integer_constant(unit_, step->next->value);
          if (!value || *value <= 0 ||
              static_cast<std::uintmax_t>(*value) >
                  static_cast<std::uintmax_t>(
                      std::numeric_limits<int64_t>::max())) {
            fail(step->next->value,
                 "check iota period must be a positive i64 integer constant");
          }
          flags = LOOM_CHECK_GENERATE_IOTA_BUILD_FLAG_HAS_PERIOD;
          period = static_cast<int64_t>(*value);
        }
        check(loom_check_generate_iota_build(
            &builder_, flags, constant(offset->value), constant(step->value),
            period, tensor_type(*binding.result_tensor, dynamic_extent),
            location, &op));
        return capture_tensor(*binding.result_tensor, dynamic_extent, op);
      }
      case Operation::Slice: {
        auto source = expression(first->value);
        auto* offset_source = first->next;
        auto offset = integer_constant(unit_, offset_source->value);
        loom_value_id_t dynamic_extent = LOOM_VALUE_ID_INVALID;
        if (binding.result_tensor->is_dynamic()) {
          dynamic_extent = dynamic_tensor_extent(offset_source->next->value);
        }
        bool statically_in_bounds = offset && *offset >= 0;
        if (statically_in_bounds && !binding.source_tensor->is_dynamic()) {
          auto source_count = binding.source_tensor->extent;
          statically_in_bounds &= *offset <= source_count;
          if (!binding.result_tensor->is_dynamic()) {
            statically_in_bounds &=
                binding.result_tensor->extent <= source_count - *offset;
          }
        }
        if (!statically_in_bounds) {
          fail(call,
               "check tensor slice must select an in-bounds constant element "
               "range");
        }
        check(loom_check_tensor_view_build(
            &builder_, binding.source_tensor->tensor_value(source.components()),
            *offset * binding.source_tensor->element_bytes,
            tensor_type(*binding.result_tensor, dynamic_extent), location,
            &op));
        return capture_tensor(*binding.result_tensor, dynamic_extent, op);
      }
      case Operation::Equal:
      case Operation::Bitwise: {
        auto actual = expression(first->value);
        auto expected = expression(first->next->value);
        auto build = binding.operation == Operation::Equal
                         ? loom_check_expect_equal_build
                         : loom_check_expect_bitwise_build;
        auto actual_value =
            actual.is_tensor()
                ? static_cast<const TensorPartition&>(actual.partition())
                      .tensor_value(actual.components())
                : actual.ssa();
        auto expected_value =
            expected.is_tensor()
                ? static_cast<const TensorPartition&>(expected.partition())
                      .tensor_value(expected.components())
                : expected.ssa();
        check(build(&builder_, actual_value, expected_value, location, &op));
        break;
      }
      case Operation::Requires:
      case Operation::Event:
        metadata(call, binding);
        break;
      case Operation::Close: {
        auto actual = expression(first->value);
        auto expected = expression(first->next->value);
        auto* absolute = first->next->next;
        auto* relative = absolute->next;
        auto absolute_tolerance = tolerance(absolute->value);
        auto relative_tolerance = tolerance(relative->value);
        auto nan = string_literal(relative->next->value);
        if (nan != "same" && nan != "different") {
          fail(relative->next->value,
               "check NaN policy must be 'same' or 'different'");
        }
        auto actual_value =
            actual.is_tensor()
                ? static_cast<const TensorPartition&>(actual.partition())
                      .tensor_value(actual.components())
                : actual.ssa();
        auto expected_value =
            expected.is_tensor()
                ? static_cast<const TensorPartition&>(expected.partition())
                      .tensor_value(expected.components())
                : expected.ssa();
        check(loom_check_expect_close_build(
            &builder_, actual_value, expected_value, absolute_tolerance,
            relative_tolerance,
            nan == "same" ? LOOM_CHECK_EXPECT_CLOSE_NAN_SAME
                          : LOOM_CHECK_EXPECT_CLOSE_NAN_DIFFERENT,
            location, &op));
        break;
      }
      case Operation::Launch: {
        if (observing_) {
          fail(call, "check expectations must be terminal");
        }
        if (!functions_.definition(binding.kernel)) {
          fail(call, "check launches require a defined kernel");
        }
        std::vector<loom_value_id_t> workloads;
        auto* argument = call->expressionList;
        if (binding.configuration &&
            !binding.configuration->parameters().empty()) {
          append_workloads(argument->value, binding, workloads);
          argument = argument->next;
        }
        std::vector<loom_value_id_t> arguments;
        for (; argument; argument = argument->next) {
          append_subject_value(expression(argument->value), arguments);
        }
        auto symbol = functions_.declare(binding.kernel, call);
        check(loom_kernel_launch_build(&builder_, symbol, workloads.data(),
                                       workloads.size(), arguments.data(),
                                       arguments.size(), location, &op));
        break;
      }
      case Operation::EntropyFork: {
        auto entropy = expression(first->value);
        const auto& partition = types_.partition(call->type, call);
        check(loom_check_entropy_fork_build(
            &builder_, entropy.ssa(),
            intern(string_literal(first->next->value)),
            static_cast<const OpaqueDialectPartition&>(partition).type,
            location, &op));
        return value_arena_.capture(partition, {loom_op_results(op), 1});
      }
      case Operation::EntropyRead: {
        auto entropy = expression(first->value);
        auto* position = first->next->value;
        std::array<int64_t, 1> static_ordinals = {INT64_MIN};
        std::vector<loom_value_id_t> dynamic_ordinals;
        if (auto ordinal = integer_constant(unit_, position)) {
          if (*ordinal < 0) {
            fail(position, "check entropy ordinal must be non-negative");
          }
          static_ordinals[0] = *ordinal;
        } else {
          if (scalar_constant(unit_, position)) {
            fail(position,
                 "check entropy ordinal constant exceeds the supported i64 "
                 "range");
          }
          auto value = expression(position);
          auto value_type =
              loom_module_value_type(builder_.module, value.ssa());
          auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
          if (!loom_type_equal(value_type, index_type)) {
            fail(position,
                 "dynamic check entropy ordinals require "
                 "loom::check::ordinal");
          }
          dynamic_ordinals.push_back(value.ssa());
        }
        check(loom_check_entropy_read_build(
            &builder_, entropy.ssa(), dynamic_ordinals.data(),
            dynamic_ordinals.size(), static_ordinals.data(),
            static_ordinals.size(), loom_type_scalar(LOOM_SCALAR_TYPE_I64),
            location, &op));
        return Value(loom_op_results(op)[0]);
      }
      case Operation::Trial:
      case Operation::Compare:
      case Operation::Invoke:
        fail(call, "structured check operation appears outside its region");
    }
    observing_ |= binding.is_observation();
    return std::nullopt;
  }

  loom_op_t* invocation(cxx::CallExpressionAST* call,
                        cxx::FunctionSymbol* function) {
    if (in_trial_) {
      fail(call,
           "ordinary trial calls must use loom::check::generate<function>");
    }
    if (observing_) {
      fail(call, "check expectations must be terminal");
    }
    if (!functions_.definition(function) ||
        functions_.is_check_record(function) || annotated(function, "kernel") ||
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
    auto symbol = functions_.declare(function, call);
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
        types_.admit_construction(cast->conversionFunction, cast->type, source);
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
      fail(source, "runtime conversions are not supported in check records");
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
      types_.admit_construction(variable->constructor(), variable->type(),
                                declarator);
      auto value = expression(declarator->initializer);
      values_[variable] = value;
      name(value, variable);
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
  // Trial recipes require explicit check.generate calls for runtime work.
  bool in_trial_ = false;
};

}  // namespace

void translate_check_case_body(cxx::TranslationUnit& unit,
                               Diagnostics& diagnostics, Functions& functions,
                               Intrinsics& intrinsics, Types& types,
                               Scalars& scalars, Locations& locations,
                               loom_builder_t& builder,
                               const FunctionBody& body) {
  CheckBody(unit, diagnostics, functions, intrinsics, types, scalars, locations,
            builder)
      .translate_case(body);
}

void translate_check_scenario_body(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Functions& functions,
    Intrinsics& intrinsics, Types& types, Scalars& scalars,
    Locations& locations, loom_builder_t& builder, const FunctionBody& body) {
  CheckBody(unit, diagnostics, functions, intrinsics, types, scalars, locations,
            builder)
      .translate_scenario(body);
}

}  // namespace loom::cxx_import
