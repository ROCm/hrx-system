// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/translation/staged_calls.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include "loom/import/cxx/binding/command.h"
#include "loom/import/cxx/binding/index_expression.h"
#include "loom/import/cxx/binding/kernel_launch.h"
#include "loom/import/cxx/binding/pipeline.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/frontend.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/functions.h"
#include "loom/import/cxx/value/component_names.h"
#include "loom/import/cxx/value/evaluation.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/import/cxx/value/storage.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"

namespace loom::cxx_import {
namespace {

// A user-defined conversion is an evaluated call, even when its result
// has the same type as a workload pack or literal.
cxx::ExpressionAST* direct_operand(cxx::ExpressionAST* source) {
  while (source) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
      source = nested->expression;
      continue;
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
      source = initializer->expression;
      continue;
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source);
        cast && !cast->conversionFunction) {
      source = cast->expression;
      continue;
    }
    return source;
  }
  return nullptr;
}

}  // namespace

void StagedCalls::fail(cxx::AST* owner, std::string_view message) {
  diagnostics_.reject(unit_, owner, message);
}

const cxx::Type* StagedCalls::storage_element_type(const cxx::Type* source_type,
                                                   const Partition& partition) {
  if (auto* pointer =
          cxx::type_cast<cxx::PointerType>(types_.unqualified(source_type))) {
    return pointer->elementType();
  }
  return partition.kind == ValueKind::Buffer
             ? static_cast<const BufferPartition&>(partition).element_type
             : nullptr;
}

bool StagedCalls::storage_elements_compatible(const cxx::Type* expected,
                                              const cxx::Type* actual) {
  if (!expected || !actual ||
      types_.unqualified(expected) != types_.unqualified(actual)) {
    return false;
  }
  auto traits = unit_.typeTraits();
  return !(traits.is_const(actual) && !traits.is_const(expected)) &&
         !(traits.is_volatile(actual) && !traits.is_volatile(expected));
}

void StagedCalls::append_kernel_workloads(
    cxx::ExpressionAST* source, const KernelLaunchIntrinsic& launch,
    std::vector<loom_value_id_t>& workloads) {
  auto* call = cxx::ast_cast<cxx::CallExpressionAST>(direct_operand(source));
  auto* function = call ? direct_callee(call) : nullptr;
  if (!function || !annotated(function, "workload")) {
    fail(source,
         "kernel workloads require a direct loom::kernel::workload(...) "
         "call");
  }
  auto parameters = launch.configuration->parameters();
  auto* argument = call->expressionList;
  for (auto* parameter : parameters) {
    if (!argument) {
      fail(source,
           "loom::kernel::workload argument count must match the kernel "
           "configuration");
    }
    auto* parameter_type = types_.unqualified(parameter->type());
    const auto& expected = types_.partition(parameter_type, argument->value);
    auto value = evaluator_.expression(argument->value);
    const auto& actual = value.partition();
    auto* expected_element = storage_element_type(parameter_type, expected);
    auto* actual_element = storage_element_type(argument->value->type, actual);
    if (expected.kind == ValueKind::Pointer && value.is_buffer()) {
      if (!storage_elements_compatible(expected_element, actual_element)) {
        fail(argument->value,
             "kernel workload buffer element type must match the "
             "configuration pointer element type");
      }
      workloads.push_back(value.ssa());
      auto byte_offset = scalars_.integer(0, LOOM_SCALAR_TYPE_INDEX,
                                          locations_.get(argument->value));
      workloads.push_back(name_value(
          builder_.module, byte_offset,
          cxx::to_string(parameter->name()) + std::string("_byte_offset")));
    } else {
      if (types_.unqualified(argument->value->type) != parameter_type ||
          actual.component_count != expected.component_count) {
        fail(argument->value,
             "loom::kernel::workload arguments must match the kernel "
             "configuration parameters");
      }
      value.append_to(workloads);
    }
    argument = argument->next;
  }
  if (argument) {
    fail(source,
         "loom::kernel::workload argument count must match the kernel "
         "configuration");
  }
}

void StagedCalls::kernel_launch(cxx::CallExpressionAST* call,
                                const KernelLaunchIntrinsic& launch) {
  if (kind_ != FunctionKind::CommandProgram) {
    fail(call, "kernel.launch requires a command_program body");
  }
  std::vector<loom_value_id_t> workloads;
  auto* argument = call->expressionList;
  if (launch.pipeline) {
    std::vector<loom_value_id_t> arguments;
    append_staged_arguments(argument, *launch.pipeline,
                            launch.pipeline_buffer_materializations, workloads,
                            arguments);
    loom_op_t* op;
    check(loom_kernel_launch_build(
        &builder_, functions_.declare(launch.kernel, call), workloads.data(),
        workloads.size(), arguments.data(), arguments.size(),
        locations_.get(call), &op));
    return;
  }
  if (launch.configuration && !launch.configuration->parameters().empty()) {
    append_kernel_workloads(argument->value, launch, workloads);
    argument = argument->next;
  }
  auto* kernel_type = cxx::type_cast<cxx::FunctionType>(launch.kernel->type());
  auto kernel_parameters = kernel_type->parameterTypes();
  std::vector<loom_value_id_t> arguments;
  size_t ordinal = 0;
  for (; argument; argument = argument->next, ++ordinal) {
    auto value = evaluator_.expression(argument->value);
    if (value.is_record()) {
      value.append_to(arguments);
      continue;
    }
    auto expected = types_.get(kernel_parameters[ordinal], argument->value);
    if (loom_type_kind(expected) == LOOM_TYPE_BUFFER) {
      if (!value.is_buffer()) {
        fail(argument->value,
             "command kernel pointer arguments require typed buffers");
      }
      const auto& actual =
          static_cast<const BufferPartition&>(value.partition());
      auto* expected_element = storage_element_type(
          kernel_parameters[ordinal],
          types_.partition(kernel_parameters[ordinal], argument->value));
      if (!storage_elements_compatible(expected_element, actual.element_type)) {
        fail(argument->value,
             "command kernel buffer element type must match the kernel "
             "pointer element type");
      }
    }
    arguments.push_back(value.ssa());
  }
  loom_op_t* op;
  check(loom_kernel_launch_build(
      &builder_, functions_.declare(launch.kernel, call), workloads.data(),
      workloads.size(), arguments.data(), arguments.size(),
      locations_.get(call), &op));
}

void StagedCalls::append_staged_arguments(
    cxx::List<cxx::ExpressionAST*>* source,
    const StagedSignatureProjection& projection,
    std::span<const uint8_t> buffer_materializations,
    std::vector<loom_value_id_t>& specializations,
    std::vector<loom_value_id_t>& bindings) {
  specializations.resize(projection.specialization_count);
  bindings.resize(projection.binding_count);
  size_t ordinal = 0;
  for (auto* argument : cxx::ListView{source}) {
    auto value = evaluator_.expression(argument);
    const auto& parameter = projection.parameters[ordinal];
    auto components = value.components();
    loom_value_id_t storage_operand = LOOM_VALUE_ID_INVALID;
    if (!buffer_materializations.empty() && buffer_materializations[ordinal]) {
      if (value.is_pointer()) {
        storage_operand = storage_.zero_origin_root(value.pointer(), argument);
      } else {
        storage_operand = components.back();
      }
      components = {&storage_operand, 1};
    }
    ++ordinal;
    parameter.write_argument(components, specializations, bindings);
  }
  IREE_ASSERT_EQ(ordinal, projection.parameters.size());
}

std::vector<loom_value_id_t> StagedCalls::dynamic_indices(
    const IndexList& list) {
  std::vector<loom_value_id_t> values;
  values.reserve(list.dynamic_expressions().size());
  for (auto* source : list.dynamic_expressions()) {
    auto value = evaluator_.expression(source).ssa();
    values.push_back(IndexExpression::build(unit_, &builder_, source, value,
                                            locations_.get(source)));
  }
  return values;
}

Value StagedCalls::pipeline_memory(cxx::CallExpressionAST* call,
                                   const PipelineMemoryIntrinsic& binding) {
  if (kind_ != FunctionKind::Pipeline) {
    fail(call, "pipeline.memory requires a pipeline definition body");
  }
  if (in_strand_) {
    fail(call, "pipeline.memory belongs outside pipeline strand bodies");
  }
  auto coordinates =
      IndexList::admit(unit_, diagnostics_, types_, call->expressionList->value,
                       "pipeline.memory coordinates");
  auto dynamic = dynamic_indices(coordinates);
  loom_op_t* op;
  check(loom_pipeline_memory_build(
      &builder_, binding.memory_space, dynamic.data(), dynamic.size(),
      coordinates.static_values().data(), coordinates.static_values().size(),
      types_.get(call->type, call), locations_.get(call), &op));
  auto result = loom_op_results(op)[0];
  return value_arena_.capture(*binding.result, {&result, 1});
}

void StagedCalls::pipeline_compose(cxx::CallExpressionAST* call,
                                   const PipelineComposeIntrinsic& binding) {
  if (kind_ != FunctionKind::Pipeline) {
    fail(call, "pipeline.compose requires a pipeline definition body");
  }
  std::vector<loom_value_id_t> specializations;
  std::vector<loom_value_id_t> bindings;
  append_staged_arguments(call->expressionList, *binding.projection,
                          /*buffer_materializations=*/{}, specializations,
                          bindings);
  loom_op_t* op;
  check(loom_pipeline_compose_build(
      &builder_, functions_.declare(binding.pipeline, call),
      specializations.data(), specializations.size(), bindings.data(),
      bindings.size(), locations_.get(call), &op));
}

void StagedCalls::command_launch(cxx::CallExpressionAST* call,
                                 const CommandLaunchIntrinsic& launch) {
  if (kind_ != FunctionKind::CommandProgram) {
    fail(call, "command.launch requires a command_program body");
  }
  std::vector<loom_value_id_t> specializations;
  std::vector<loom_value_id_t> bindings;
  append_staged_arguments(call->expressionList, *launch.projection,
                          /*buffer_materializations=*/{}, specializations,
                          bindings);
  loom_op_t* op;
  check(loom_command_launch_build(
      &builder_, functions_.declare(launch.program, call),
      specializations.data(), specializations.size(), bindings.data(),
      bindings.size(), locations_.get(call), &op));
}

Value StagedCalls::command_parameter(cxx::CallExpressionAST* call,
                                     const CommandParameterIntrinsic& binding) {
  if (kind_ != FunctionKind::CommandProgram) {
    fail(call, "command parameter operations require a command_program body");
  }
  auto* argument = call->expressionList;
  if (binding.operation == CommandParameterIntrinsic::Operation::Base) {
    auto source = evaluator_.expression(argument->value);
    loom_op_t* op;
    check(loom_command_parameter_base_build(
        &builder_, source.components().back(), types_.get(call->type, call),
        locations_.get(call), &op));
    auto result = loom_op_results(op)[0];
    return value_arena_.capture(*binding.result, {&result, 1});
  }

  auto source = evaluator_.expression(argument->value);
  argument = argument->next;
  std::vector<loom_value_id_t> components;
  components.reserve(binding.binding_component_count + 1);
  for (size_t index = 0; index < binding.binding_component_count; ++index) {
    auto value = evaluator_.expression(argument->value);
    IREE_ASSERT_EQ(value.components().size(), 1);
    components.push_back(value.components()[0]);
    argument = argument->next;
  }
  auto* key = cxx::ast_cast<cxx::StringLiteralExpressionAST>(
      direct_operand(argument->value));
  if (!key) {
    fail(argument->value,
         "command.parameter requires a string literal key pattern");
  }
  auto pattern = key->literal->stringValue();
  argument = argument->next;
  std::vector<loom_value_id_t> substitutions;
  for (; argument; argument = argument->next) {
    substitutions.push_back(evaluator_.expression(argument->value).ssa());
  }

  std::vector<loom_value_id_t> type_identities = components;
  type_identities.push_back(LOOM_VALUE_ID_INVALID);
  BoundTypeStorage type_storage;
  std::vector<loom_type_t> component_types;
  types_.append_bound(call->type, call, type_identities, type_storage,
                      component_types);
  IREE_ASSERT_EQ(component_types.size(), type_identities.size());
  loom_string_id_t pattern_id;
  check(loom_module_intern_string(builder_.module, view(pattern), &pattern_id));
  loom_op_t* op;
  check(loom_command_parameter_build(
      &builder_, source.ssa(), pattern_id, substitutions.data(),
      substitutions.size(), component_types.back(), locations_.get(call), &op));
  components.push_back(loom_op_results(op)[0]);
  return value_arena_.capture(*binding.result, components);
}

}  // namespace loom::cxx_import
