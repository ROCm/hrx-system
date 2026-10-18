// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/check.h"

#include <cxx/ast.h>
#include <cxx/const_int.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <limits>

#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/source/attributes.h"

namespace loom::cxx_import {
namespace {

const TensorPartition& require_tensor(cxx::TranslationUnit& unit,
                                      Diagnostics& diagnostics, Types& types,
                                      const cxx::Type* type, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::Tensor) {
    diagnostics.reject(unit, owner, "check operation requires a tensor handle");
  }
  return static_cast<const TensorPartition&>(partition);
}

bool is_string_type(Types& types, const cxx::Type* type) {
  auto* pointer = cxx::type_cast<cxx::PointerType>(types.unqualified(type));
  return pointer && types.unqualified(pointer->elementType())->kind() ==
                        cxx::TypeKind::kChar;
}

bool is_workload_type(Types& types, const cxx::Type* type) {
  auto* record = cxx::type_cast<cxx::ClassType>(types.unqualified(type));
  return record && annotated(record->definition(), "workload");
}

void require_tensor_extent(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                           Types& types, const cxx::Type* type,
                           cxx::AST* owner) {
  if (!types.is_index(type, owner)) {
    diagnostics.reject(unit, owner,
                       "dynamic tensor extent requires loom::type::index; use "
                       "loom::index::cast in its generating function");
  }
}

}  // namespace

std::optional<CheckIntrinsic::Operation> CheckIntrinsic::parse_operation(
    std::string_view name) {
  if (name == "check.expect.equal") {
    return Operation::Equal;
  }
  if (name == "check.expect.close") {
    return Operation::Close;
  }
  if (name == "check.generate.fill") {
    return Operation::Fill;
  }
  if (name == "check.generate.iota") {
    return Operation::Iota;
  }
  if (name == "check.tensor.view") {
    return Operation::Slice;
  }
  if (name == "check.expect.bitwise") {
    return Operation::Bitwise;
  }
  if (name == "check.requires") {
    return Operation::Requires;
  }
  if (name == "check.expect.event") {
    return Operation::Event;
  }
  if (name == "kernel.launch") {
    return Operation::Launch;
  }
  if (name == "check.trial") {
    return Operation::Trial;
  }
  if (name == "check.generate") {
    return Operation::Generate;
  }
  if (name == "check.compare") {
    return Operation::Compare;
  }
  if (name == "check.invoke") {
    return Operation::Invoke;
  }
  if (name == "check.entropy.fork") {
    return Operation::EntropyFork;
  }
  if (name == "check.entropy.read") {
    return Operation::EntropyRead;
  }
  return std::nullopt;
}

std::optional<CheckIntrinsic> CheckIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    LaunchContracts& launches, cxx::FunctionSymbol* function,
    const cxx::Attribute& attribute, cxx::AST* owner) {
  auto operation = parse_operation(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature->parameterTypes();
  bool returns_void = unit.typeTraits().is_void(signature->returnType());
  auto fail = [&](const char* message) {
    diagnostics.reject(unit, owner, message);
  };
  auto selected_function = [&](size_t index) -> cxx::FunctionSymbol* {
    auto arguments = function->templateArguments();
    auto selected = arguments.size() > index
                        ? cxx::template_argument_value(arguments[index])
                        : std::nullopt;
    auto* address =
        selected ? std::get_if<std::shared_ptr<cxx::ConstAddress>>(&*selected)
                 : nullptr;
    return address && *address
               ? cxx::symbol_cast<cxx::FunctionSymbol>((*address)->symbol())
               : nullptr;
  };
  auto bind_configuration = [&](cxx::FunctionSymbol* selected) {
    if (!selected || !annotated(selected, "kernel")) {
      return static_cast<cxx::FunctionSymbol*>(nullptr);
    }
    if (!selected->templateArguments().empty() && selected->declaration()) {
      launches.declaration(selected, selected->declaration()->attributeList);
    }
    return launches.configuration_function(selected);
  };
  if (attribute.arguments.size() != 1 || signature->isVariadic()) {
    fail("check operations require one binding and a fixed source signature");
  }
  CheckIntrinsic result{*operation};
  switch (*operation) {
    case Operation::Equal: {
      if (!returns_void || parameters.size() != 2 ||
          types.unqualified(parameters[0]) !=
              types.unqualified(parameters[1])) {
        fail("check.expect.equal requires void(T, T) with one scalar type");
      }
      result.scalar_type = types.get(parameters[0], owner);
      if (loom_type_kind(result.scalar_type) != LOOM_TYPE_SCALAR) {
        fail("check.expect.equal requires scalar operands");
      }
      break;
    }
    case Operation::Fill: {
      if (returns_void) {
        fail("check.generate.fill requires one scalar and a tensor result");
      }
      result.result_tensor = &require_tensor(unit, diagnostics, types,
                                             signature->returnType(), owner);
      size_t payload_index = result.result_tensor->is_dynamic() ? 1 : 0;
      if (parameters.size() != payload_index + 1) {
        fail(
            "check.generate.fill requires an optional runtime extent, one "
            "scalar, and a tensor result");
      }
      if (result.result_tensor->is_dynamic()) {
        require_tensor_extent(unit, diagnostics, types, parameters[0], owner);
      }
      if (types.unqualified(parameters[payload_index]) !=
          types.unqualified(result.result_tensor->lane_type)) {
        fail(
            "check.generate.fill payload must match its tensor scalar lane "
            "type");
      }
      break;
    }
    case Operation::Iota: {
      if (returns_void) {
        fail("check.generate.iota requires a tensor result");
      }
      result.result_tensor = &require_tensor(unit, diagnostics, types,
                                             signature->returnType(), owner);
      size_t payload_index = result.result_tensor->is_dynamic() ? 1 : 0;
      if ((parameters.size() != payload_index + 2 &&
           parameters.size() != payload_index + 3)) {
        fail(
            "check.generate.iota requires an optional runtime extent, "
            "offset, step, an optional period, and a tensor result");
      }
      if (result.result_tensor->is_dynamic()) {
        require_tensor_extent(unit, diagnostics, types, parameters[0], owner);
      }
      if (types.unqualified(parameters[payload_index]) !=
              types.unqualified(result.result_tensor->lane_type) ||
          types.unqualified(parameters[payload_index + 1]) !=
              types.unqualified(result.result_tensor->lane_type) ||
          (parameters.size() == payload_index + 3 &&
           !unit.typeTraits().is_integral(parameters[payload_index + 2]))) {
        fail(
            "check.generate.iota offset and step must match the tensor "
            "element type, with an integer period");
      }
      break;
    }
    case Operation::Close: {
      if (!returns_void || parameters.size() != 5 ||
          types.unqualified(parameters[0]) !=
              types.unqualified(parameters[1]) ||
          types.unqualified(parameters[2])->kind() != cxx::TypeKind::kDouble ||
          types.unqualified(parameters[3])->kind() != cxx::TypeKind::kDouble ||
          !is_string_type(types, parameters[4])) {
        fail(
            "check.expect.close requires void(T, T, double, double, const "
            "char*)");
      }
      const auto& partition = types.partition(parameters[0], owner);
      if (partition.kind == ValueKind::SSA && types.is_float(parameters[0])) {
        result.scalar_type = types.get(parameters[0], owner);
      } else if (partition.kind == ValueKind::Tensor &&
                 types.is_float(static_cast<const TensorPartition&>(partition)
                                    .lane_type)) {
        result.source_tensor = static_cast<const TensorPartition*>(&partition);
      } else {
        fail(
            "check.expect.close requires floating-point scalar or tensor "
            "operands");
      }
      break;
    }
    case Operation::Slice: {
      if (returns_void || parameters.size() < 2 ||
          !unit.typeTraits().is_integral(parameters[1])) {
        fail(
            "check.tensor.view requires a tensor, an element offset and a "
            "tensor result");
      }
      result.source_tensor =
          &require_tensor(unit, diagnostics, types, parameters[0], owner);
      result.result_tensor = &require_tensor(unit, diagnostics, types,
                                             signature->returnType(), owner);
      size_t parameter_count = result.result_tensor->is_dynamic() ? 3 : 2;
      if (parameters.size() != parameter_count) {
        fail(
            "check.tensor.view requires a tensor, an element offset, an "
            "optional runtime extent and a tensor result");
      }
      if (result.result_tensor->is_dynamic()) {
        require_tensor_extent(unit, diagnostics, types, parameters[2], owner);
      }
      if (result.source_tensor->element_type !=
          result.result_tensor->element_type) {
        fail("check.tensor.view must preserve its source element type");
      }
      break;
    }
    case Operation::Bitwise: {
      if (!returns_void || parameters.size() != 2 ||
          types.unqualified(parameters[0]) !=
              types.unqualified(parameters[1])) {
        fail("check.expect.bitwise requires two tensors of the same type");
      }
      result.source_tensor =
          &require_tensor(unit, diagnostics, types, parameters[0], owner);
      break;
    }
    case Operation::Requires:
    case Operation::Event: {
      if (!returns_void || parameters.empty() || parameters.size() % 2 != 1 ||
          !is_string_type(types, parameters[0])) {
        fail("check metadata requires a provider followed by name/value pairs");
      }
      for (size_t index = 1; index < parameters.size(); index += 2) {
        if (!is_string_type(types, parameters[index]) ||
            (!is_string_type(types, parameters[index + 1]) &&
             !unit.typeTraits().is_integral_or_enum(parameters[index + 1]) &&
             !unit.typeTraits().is_floating_point(parameters[index + 1]))) {
          fail(
              "check metadata requires string names and scalar or string "
              "values");
        }
      }
      break;
    }
    case Operation::Launch: {
      auto arguments = function->templateArguments();
      auto selected = arguments.empty()
                          ? std::nullopt
                          : cxx::template_argument_value(arguments[0]);
      auto* address =
          selected ? std::get_if<std::shared_ptr<cxx::ConstAddress>>(&*selected)
                   : nullptr;
      result.kernel =
          address && *address
              ? cxx::symbol_cast<cxx::FunctionSymbol>((*address)->symbol())
              : nullptr;
      if (!returns_void || !result.kernel ||
          !annotated(result.kernel, "kernel")) {
        fail("kernel.launch requires a kernel as its first template argument");
      }
      if (!result.kernel->templateArguments().empty() &&
          result.kernel->declaration()) {
        launches.declaration(result.kernel,
                             result.kernel->declaration()->attributeList);
      }
      result.configuration = launches.configuration_function(result.kernel);
      auto* kernel_type =
          cxx::type_cast<cxx::FunctionType>(result.kernel->type());
      auto kernel_parameters = kernel_type->parameterTypes();
      bool has_workload =
          !parameters.empty() && is_workload_type(types, parameters[0]);
      bool requires_workload =
          result.configuration && !result.configuration->parameters().empty();
      if (requires_workload && !has_workload) {
        fail(
            "configured kernel launches require "
            "loom::kernel::workload(...) before the kernel arguments");
      }
      if (!requires_workload && has_workload) {
        fail(
            "loom::kernel::workload(...) requires a kernel configuration with "
            "workload parameters");
      }
      size_t parameter_offset = has_workload ? 1 : 0;
      if (kernel_parameters.size() + parameter_offset != parameters.size()) {
        fail("kernel.launch arguments must match the kernel signature");
      }
      for (size_t index = 0; index < kernel_parameters.size(); ++index) {
        auto expected = types.get(kernel_parameters[index], owner);
        if (loom_type_kind(expected) == LOOM_TYPE_BUFFER) {
          require_tensor(unit, diagnostics, types,
                         parameters[index + parameter_offset], owner);
        } else if (!loom_type_equal(
                       expected, types.get(parameters[index + parameter_offset],
                                           owner))) {
          fail(
              "kernel.launch scalar arguments must match the kernel parameter "
              "types");
        }
      }
      break;
    }
    case Operation::Trial: {
      auto arguments = function->templateArguments();
      auto count_value = arguments.empty()
                             ? std::nullopt
                             : cxx::template_argument_value(arguments[0]);
      auto* count =
          count_value ? std::get_if<cxx::ConstInt>(&*count_value) : nullptr;
      if (!returns_void || parameters.size() != 1 || !count ||
          count->isNegative() || count->isZero() ||
          count->toUWide() > static_cast<cxx::ConstInt::UWide>(
                                 std::numeric_limits<int64_t>::max()) ||
          types.unqualified(parameters[0])->kind() != cxx::TypeKind::kClass) {
        fail("check.trial requires trial<positive_count>(lambda)");
      }
      result.trial_count = static_cast<int64_t>(count->toUIntMax());
      break;
    }
    case Operation::Generate: {
      result.subject = selected_function(0);
      if (!result.subject || signature->isVariadic()) {
        fail("check.generate requires an ordinary function template argument");
      }
      break;
    }
    case Operation::Compare: {
      result.subject = selected_function(0);
      result.oracle = selected_function(1);
      if (!returns_void || !result.subject || parameters.empty() ||
          types.unqualified(parameters.back())->kind() !=
              cxx::TypeKind::kClass) {
        fail("check.compare requires a subject and a final comparison lambda");
      }
      result.configuration = bind_configuration(result.subject);
      auto* oracle_configuration = bind_configuration(result.oracle);
      if (oracle_configuration &&
          oracle_configuration != result.configuration) {
        fail(
            "a configured check.compare oracle must use the target workload "
            "configuration");
      }
      break;
    }
    case Operation::Invoke: {
      result.subject = selected_function(0);
      if (!returns_void || !result.subject) {
        fail("check.invoke requires a subject function template argument");
      }
      result.configuration = bind_configuration(result.subject);
      break;
    }
    case Operation::EntropyFork: {
      if (parameters.size() != 2 || returns_void ||
          !types.is_opaque_dialect(parameters[0], "check.entropy", owner) ||
          !types.is_opaque_dialect(signature->returnType(), "check.entropy",
                                   owner) ||
          !is_string_type(types, parameters[1])) {
        fail("check.entropy.fork requires entropy(entropy, const char*)");
      }
      break;
    }
    case Operation::EntropyRead: {
      if (parameters.size() != 2 || returns_void ||
          !types.is_opaque_dialect(parameters[0], "check.entropy", owner) ||
          types.unqualified(signature->returnType()) !=
              unit.control()->getUnsignedLongLongIntType() ||
          (!types.is_index(parameters[1], owner) &&
           types.unqualified(parameters[1]) !=
               unit.control()->getUnsignedLongLongIntType())) {
        fail(
            "check.entropy.read requires an entropy source, index or constant "
            "ordinal, and an i64 result");
      }
      break;
    }
  }
  return result;
}

bool CheckIntrinsic::equivalent(const CheckIntrinsic& other) const {
  return operation == other.operation &&
         loom_type_equal(scalar_type, other.scalar_type) &&
         source_tensor == other.source_tensor &&
         result_tensor == other.result_tensor && kernel == other.kernel &&
         subject == other.subject && oracle == other.oracle &&
         trial_count == other.trial_count &&
         configuration == other.configuration;
}

}  // namespace loom::cxx_import
