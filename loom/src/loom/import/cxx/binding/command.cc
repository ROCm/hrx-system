// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/command.h"

#include <cxx/ast.h>
#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <memory>
#include <variant>

#include "iree/base/api.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

bool is_string_type(Types& types, const cxx::Type* type) {
  auto* pointer = cxx::type_cast<cxx::PointerType>(types.unqualified(type));
  return pointer && types.unqualified(pointer->elementType())->kind() ==
                        cxx::TypeKind::kChar;
}

}  // namespace

bool CommandLaunchIntrinsic::supports(std::string_view name) {
  return name == "command.launch";
}

std::optional<CommandLaunchIntrinsic> CommandLaunchIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    CommandPrograms& programs, cxx::FunctionSymbol* function,
    const cxx::Attribute& attribute, cxx::AST* owner) {
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
    fail(
        "command.launch requires one binding and a fixed void "
        "signature");
  }

  auto arguments = function->templateArguments();
  auto selected = arguments.empty()
                      ? std::nullopt
                      : cxx::template_argument_value(arguments[0]);
  auto* address =
      selected ? std::get_if<std::shared_ptr<cxx::ConstAddress>>(&*selected)
               : nullptr;
  auto* program =
      address && *address
          ? cxx::symbol_cast<cxx::FunctionSymbol>((*address)->symbol())
          : nullptr;
  if (!program || !programs.contains(program)) {
    fail(
        "command.launch requires a command_program as its first "
        "template argument");
  }
  const auto& projection = programs.project(program, types, owner);
  auto parameters = signature->parameterTypes();
  if (parameters.size() != projection.source_types.size()) {
    fail("command.launch arguments must match the command signature");
  }
  for (size_t index = 0; index < parameters.size(); ++index) {
    if (types.unqualified_value(parameters[index]) !=
        projection.source_types[index]) {
      fail(
          "command.launch arguments must preserve the command source "
          "types");
    }
  }
  return CommandLaunchIntrinsic{program, &projection};
}

bool CommandParameterIntrinsic::supports(std::string_view name) {
  return name == "command.parameter" || name == "command.parameter.base";
}

std::optional<CommandParameterIntrinsic> CommandParameterIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
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
      signature->isVariadic()) {
    fail(
        "command parameter operations require one binding and a fixed "
        "signature");
  }
  auto parameters = signature->parameterTypes();
  if (attribute.arguments[0]->name() == "command.parameter.base") {
    if (parameters.size() != 1) {
      fail("command.parameter.base requires one typed view operand");
    }
    const auto& source = types.partition(parameters[0], owner);
    const auto& result = types.partition(signature->returnType(), owner);
    if (source.kind != ValueKind::View || result.kind != ValueKind::Buffer) {
      fail(
          "command.parameter.base requires a typed view and typed buffer "
          "result");
    }
    const auto& source_view = static_cast<const ViewPartition&>(source);
    const auto& result_buffer = static_cast<const BufferPartition&>(result);
    if (source_view.element_type != result_buffer.element_type) {
      fail("command.parameter.base must preserve its source element type");
    }
    return CommandParameterIntrinsic{Operation::Base, &result, &source, 0};
  }

  if (parameters.empty()) {
    fail("command.parameter requires a typed buffer source");
  }
  const auto& source = types.partition(parameters[0], owner);
  const auto& result = types.partition(signature->returnType(), owner);
  if (source.kind != ValueKind::Buffer || result.kind != ValueKind::View) {
    fail("command.parameter requires a typed buffer and typed view result");
  }
  const auto& view = static_cast<const ViewPartition&>(result);
  if (!unit.typeTraits().is_const(view.element_type)) {
    fail("command.parameter requires a const-qualified result element type");
  }
  const size_t binding_count = view.component_count - 1;
  const size_t pattern_index = 1 + binding_count;
  if (parameters.size() <= pattern_index ||
      !is_string_type(types, parameters[pattern_index])) {
    fail(
        "command.parameter requires its dynamic extents and encoding before "
        "a string key pattern");
  }
  size_t component = 1;
  for (auto extent : view.extents) {
    if (extent < 0 && !types.is_index(parameters[component++], owner)) {
      fail("command.parameter dynamic extents require loom::type::index");
    }
  }
  if (view.encoding &&
      &types.partition(parameters[component++], owner) != view.encoding) {
    fail("command.parameter encoding operand must match its result view");
  }
  IREE_ASSERT_EQ(component, pattern_index);
  for (size_t index = pattern_index + 1; index < parameters.size(); ++index) {
    if (!types.is_index(parameters[index], owner)) {
      fail("command.parameter key substitutions require loom::type::index");
    }
  }
  return CommandParameterIntrinsic{Operation::Parameter, &result, &source,
                                   binding_count};
}

}  // namespace loom::cxx_import
