// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/kernel_launch.h"

#include <cxx/ast.h>
#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <memory>
#include <utility>
#include <variant>

#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/pipeline_contracts.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

bool is_workload_type(Types& types, const cxx::Type* type) {
  auto* record = cxx::type_cast<cxx::ClassType>(types.unqualified(type));
  return record && annotated(record->definition(), "workload");
}

}  // namespace

bool KernelLaunchIntrinsic::supports(std::string_view name) {
  return name == "kernel.launch";
}

std::optional<KernelLaunchIntrinsic> KernelLaunchIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    LaunchContracts& launches, PipelineContracts& pipelines,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty() ||
      !supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  auto fail = [&](const char* message) {
    diagnostics.reject(unit, owner, message);
  };
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (attribute.arguments.size() != 1 || !signature ||
      signature->isVariadic() ||
      !unit.typeTraits().is_void(signature->returnType())) {
    fail("kernel.launch requires one binding and a fixed void signature");
  }

  auto arguments = function->templateArguments();
  auto selected = arguments.empty()
                      ? std::nullopt
                      : cxx::template_argument_value(arguments[0]);
  auto* address =
      selected ? std::get_if<std::shared_ptr<cxx::ConstAddress>>(&*selected)
               : nullptr;
  auto* kernel =
      address && *address
          ? cxx::symbol_cast<cxx::FunctionSymbol>((*address)->symbol())
          : nullptr;
  const bool pipeline = kernel && pipelines.contains(kernel);
  if (!kernel || (!annotated(kernel, "kernel") &&
                  !(pipeline && pipelines.has_kernel_scope(kernel)))) {
    fail(
        "kernel.launch requires a kernel or kernel-scoped pipeline as its "
        "first template argument");
  }
  if (pipeline) {
    const auto& projection = pipelines.project(kernel, types, owner);
    auto source_parameters = kernel->parameters();
    auto parameters = signature->parameterTypes();
    std::vector<uint8_t> buffer_materializations(parameters.size());
    if (source_parameters.size() != parameters.size()) {
      fail("kernel.launch arguments must match the pipeline signature");
    }
    for (size_t index = 0; index < source_parameters.size(); ++index) {
      auto* expected =
          types.unqualified_value(source_parameters[index]->type());
      auto* actual = types.unqualified_value(parameters[index]);
      if (expected == actual) {
        continue;
      }
      const auto& expected_partition = types.partition(expected, owner);
      const auto& actual_partition = types.partition(actual, owner);
      const bool materializes_buffer =
          expected_partition.kind == ValueKind::Buffer &&
          (actual_partition.kind == ValueKind::Buffer ||
           actual_partition.kind == ValueKind::View ||
           actual_partition.kind == ValueKind::Tensor ||
           actual_partition.kind == ValueKind::Pointer);
      if (!materializes_buffer) {
        fail("kernel.launch arguments must match the pipeline source types");
      }
      buffer_materializations[index] = 1;
    }
    return KernelLaunchIntrinsic{kernel, /*configuration=*/nullptr, &projection,
                                 std::move(buffer_materializations)};
  }
  if (!kernel->templateArguments().empty() && kernel->declaration()) {
    launches.declaration(kernel, kernel->declaration()->attributeList);
  }
  auto* configuration = launches.configuration_function(kernel);
  auto parameters = signature->parameterTypes();
  const bool has_workload =
      !parameters.empty() && is_workload_type(types, parameters[0]);
  const bool requires_workload =
      configuration && !configuration->parameters().empty();
  if (requires_workload && !has_workload) {
    fail(
        "configured kernel launches require loom::kernel::workload(...) "
        "before the kernel arguments");
  }
  if (!requires_workload && has_workload) {
    fail(
        "loom::kernel::workload(...) requires a kernel configuration with "
        "workload parameters");
  }

  auto* kernel_type = cxx::type_cast<cxx::FunctionType>(kernel->type());
  auto kernel_parameters = kernel_type->parameterTypes();
  const size_t parameter_offset = has_workload ? 1 : 0;
  if (kernel_parameters.size() + parameter_offset != parameters.size()) {
    fail("kernel.launch arguments must match the kernel signature");
  }
  for (size_t index = 0; index < kernel_parameters.size(); ++index) {
    const auto expected = types.get(kernel_parameters[index], owner);
    const auto& actual =
        types.partition(parameters[index + parameter_offset], owner);
    if (loom_type_kind(expected) == LOOM_TYPE_BUFFER) {
      if (actual.kind != ValueKind::Buffer &&
          actual.kind != ValueKind::Tensor) {
        fail(
            "kernel.launch buffer arguments require a typed buffer or check "
            "tensor handle");
      }
    } else if (actual.component_count != 1 ||
               !loom_type_equal(
                   expected,
                   types.get(parameters[index + parameter_offset], owner))) {
      fail(
          "kernel.launch scalar arguments must match the kernel parameter "
          "types");
    }
  }
  return KernelLaunchIntrinsic{kernel, configuration, /*pipeline=*/nullptr,
                               /*pipeline_buffer_materializations=*/{}};
}

}  // namespace loom::cxx_import
