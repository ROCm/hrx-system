// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/buffer.h"

#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <cstdint>
#include <variant>

#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {

bool BufferIntrinsic::supports(std::string_view name) {
  return name == "buffer.alloca" || name == "buffer.assume.alignment" ||
         name == "buffer.assume.noalias" ||
         name == "buffer.assume.memory_space";
}

std::optional<BufferIntrinsic> BufferIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  auto name = attribute.arguments[0]->name();
  if (!supports(name)) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner, "buffer operation accepts no selectors");
  }

  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature->parameterTypes();
  auto* result = cxx::type_cast<cxx::PointerType>(
      types.unqualified(signature->returnType()));
  if (name == "buffer.assume.memory_space") {
    if (signature->isVariadic() || parameters.size() != 1 ||
        !unit.typeTraits().is_same(
            types.unqualified(parameters[0]),
            types.unqualified(signature->returnType()))) {
      diagnostics.reject(
          unit, owner,
          "buffer.assume.memory_space requires one pointer or typed buffer "
          "and the same result type");
    }
    const cxx::Type* element_type = nullptr;
    if (result) {
      element_type = result->elementType();
    } else {
      const auto& partition = types.partition(parameters[0], owner);
      if (partition.kind != ValueKind::Buffer) {
        diagnostics.reject(
            unit, owner,
            "buffer.assume.memory_space requires a pointer or typed buffer");
      }
      element_type =
          static_cast<const BufferPartition&>(partition).element_type;
    }
    auto arguments = function->templateArguments();
    auto space_value = arguments.size() == 2
                           ? cxx::template_argument_value(arguments[0])
                           : std::nullopt;
    auto* space =
        space_value ? std::get_if<cxx::ConstInt>(&*space_value) : nullptr;
    auto* template_element_type =
        arguments.size() == 2 ? cxx::template_argument_as_type(arguments[1])
                              : nullptr;
    if (!template_element_type ||
        !unit.typeTraits().is_same(template_element_type, element_type)) {
      diagnostics.reject(
          unit, owner,
          "buffer.assume.memory_space requires memory space and element type "
          "template arguments matching the source");
    }
    if (!space || space->isNegative() ||
        space->toUWide() < LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
        space->toUWide() > LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC) {
      diagnostics.reject(
          unit, owner,
          "buffer.assume.memory_space requires a concrete memory space: "
          "global, workgroup, private, constant, host, descriptor, or generic");
    }
    return BufferIntrinsic(
        element_type,
        static_cast<loom_value_fact_memory_space_t>(space->toUIntMax()));
  }
  if (name == "buffer.assume.noalias" || name == "buffer.assume.alignment") {
    if (signature->isVariadic() || parameters.size() != 1 ||
        !unit.typeTraits().is_same(
            types.unqualified(parameters[0]),
            types.unqualified(signature->returnType()))) {
      diagnostics.reject(
          unit, owner,
          "buffer assumption requires one pointer or typed buffer and the "
          "same result type");
    }
    const cxx::Type* source_element_type = nullptr;
    if (auto* source = cxx::type_cast<cxx::PointerType>(
            types.unqualified(parameters[0]))) {
      source_element_type = source->elementType();
    } else {
      const auto& partition = types.partition(parameters[0], owner);
      if (partition.kind != ValueKind::Buffer) {
        diagnostics.reject(
            unit, owner,
            "buffer assumption requires a pointer or typed buffer");
      }
      source_element_type =
          static_cast<const BufferPartition&>(partition).element_type;
    }
    if (name == "buffer.assume.noalias") {
      return BufferIntrinsic(source_element_type);
    }
    auto arguments = function->templateArguments();
    auto alignment_value = arguments.size() == 2
                               ? cxx::template_argument_value(arguments[0])
                               : std::nullopt;
    auto* alignment = alignment_value
                          ? std::get_if<cxx::ConstInt>(&*alignment_value)
                          : nullptr;
    auto* template_element_type =
        arguments.size() == 2 ? cxx::template_argument_as_type(arguments[1])
                              : nullptr;
    if (!alignment || alignment->isNegative() ||
        alignment->toUWide() > INT64_MAX || !template_element_type ||
        !unit.typeTraits().is_same(template_element_type,
                                   source_element_type)) {
      diagnostics.reject(
          unit, owner,
          "buffer.assume.alignment requires an alignment and source element "
          "type");
    }
    auto alignment_bytes = static_cast<uint64_t>(alignment->toUIntMax());
    if (alignment_bytes == 0 ||
        (alignment_bytes & (alignment_bytes - 1)) != 0) {
      diagnostics.reject(
          unit, owner,
          "buffer alignment must be a positive power-of-two byte count");
    }
    return BufferIntrinsic(source_element_type,
                           static_cast<int64_t>(alignment_bytes));
  }
  if (signature->isVariadic() || parameters.size() != 1 || !result) {
    diagnostics.reject(unit, owner,
                       "buffer.alloca requires one count and a pointer result");
  }
  auto count_domain = types.address_domain(parameters[0], owner);
  bool allocates_bytes = count_domain == LOOM_SCALAR_TYPE_OFFSET;
  auto arguments = function->templateArguments();
  size_t space_argument = allocates_bytes ? 0 : 1;
  size_t alignment_argument = allocates_bytes ? 1 : 2;
  const cxx::Type* element_type = nullptr;
  if (allocates_bytes) {
    if (arguments.size() != 2 ||
        !unit.typeTraits().is_same(result->elementType(),
                                   unit.control()->getUnsignedCharType())) {
      diagnostics.reject(
          unit, owner,
          "byte buffer.alloca requires memory space and alignment template "
          "arguments and returns unsigned char*");
    }
  } else {
    auto* size_type = unit.control()->getSizeType();
    if (!unit.typeTraits().is_same(types.unqualified(parameters[0]),
                                   size_type) &&
        count_domain != LOOM_SCALAR_TYPE_INDEX) {
      diagnostics.reject(unit, owner,
                         "buffer.alloca element count must be size_t or index");
    }
    if (arguments.size() != 3) {
      diagnostics.reject(unit, owner,
                         "buffer.alloca requires element type, memory space, "
                         "and alignment template arguments");
    }
    element_type = cxx::template_argument_as_type(arguments[0]);
    if (!element_type ||
        !unit.typeTraits().is_same(element_type, result->elementType())) {
      diagnostics.reject(
          unit, owner,
          "buffer.alloca element type must match its result pointer");
    }
  }

  auto space_value = cxx::template_argument_value(arguments[space_argument]);
  auto* space =
      space_value ? std::get_if<cxx::ConstInt>(&*space_value) : nullptr;
  if (!space || space->isNegative() || space->toUWide() > UINT8_MAX) {
    diagnostics.reject(unit, owner,
                       "buffer.alloca memory space must be a constant enum");
  }
  auto memory_space =
      static_cast<loom_value_fact_memory_space_t>(space->toUIntMax());
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      break;
    default:
      diagnostics.reject(
          unit, owner,
          "buffer.alloca requires an allocatable memory space: global, "
          "workgroup, private, host, or generic");
  }

  auto alignment_value =
      cxx::template_argument_value(arguments[alignment_argument]);
  auto* alignment =
      alignment_value ? std::get_if<cxx::ConstInt>(&*alignment_value) : nullptr;
  if (!alignment || alignment->isNegative() ||
      alignment->toUWide() > INT64_MAX) {
    diagnostics.reject(unit, owner,
                       "buffer.alloca alignment must be a positive "
                       "power-of-two byte count");
  }
  auto alignment_bytes = static_cast<uint64_t>(alignment->toUIntMax());
  if (alignment_bytes == 0 || (alignment_bytes & (alignment_bytes - 1)) != 0) {
    diagnostics.reject(unit, owner,
                       "buffer.alloca alignment must be a positive "
                       "power-of-two byte count");
  }

  return allocates_bytes
             ? BufferIntrinsic(memory_space,
                               static_cast<int64_t>(alignment_bytes))
             : BufferIntrinsic(element_type, parameters[0], memory_space,
                               static_cast<int64_t>(alignment_bytes));
}

Value BufferIntrinsic::call(std::span<const Value> arguments, ValueArena& arena,
                            Storage& storage, cxx::AST* owner) const {
  if (operation_ == Operation::AssumeMemorySpace) {
    auto source = arguments[0];
    auto root = storage.assume_memory_space(source.components()[0],
                                            memory_space_, owner);
    if (source.is_pointer()) {
      return Value(Pointer{root, source.pointer().byte_offset});
    }
    return arena.capture(source.partition(), {&root, 1});
  }
  if (operation_ == Operation::AssumeNoalias) {
    auto source = arguments[0];
    if (source.is_pointer()) {
      return Value(storage.assume_noalias(source.pointer(), owner));
    }
    auto root = storage.assume_noalias(source.ssa(), owner);
    return arena.capture(source.partition(), {&root, 1});
  }
  if (operation_ == Operation::AssumeAlignment) {
    auto source = arguments[0];
    if (source.is_pointer()) {
      return Value(
          storage.assume_alignment(source.pointer(), alignment_, owner));
    }
    auto root = storage.assume_alignment(source.ssa(), alignment_, owner);
    return arena.capture(source.partition(), {&root, 1});
  }
  if (operation_ == Operation::AllocaBytes) {
    return Value(storage.allocate_bytes(arguments[0].ssa(), memory_space_,
                                        alignment_, owner));
  }
  return Value(storage.allocate_elements(element_type_, arguments[0].ssa(),
                                         element_count_type_, memory_space_,
                                         alignment_, owner));
}

bool BufferIntrinsic::equivalent(const BufferIntrinsic& other) const {
  return operation_ == other.operation_ &&
         element_type_ == other.element_type_ &&
         element_count_type_ == other.element_count_type_ &&
         memory_space_ == other.memory_space_ && alignment_ == other.alignment_;
}

}  // namespace loom::cxx_import
