// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/buffer.h"

#include <cxx/ast.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <cstdint>
#include <string>
#include <variant>

#include "iree/base/internal/unicode.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"

namespace loom::cxx_import {
namespace {

int64_t byte_alignment(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                       const std::optional<cxx::ConstValue>& value,
                       std::string_view operation, cxx::AST* owner) {
  auto* alignment = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!alignment || alignment->isNegative() ||
      alignment->toUWide() > INT64_MAX || alignment->toUWide() == 0 ||
      (alignment->toUWide() & (alignment->toUWide() - 1)) != 0) {
    diagnostics.reject(
        unit, owner,
        std::string(operation) +
            " alignment must be a positive power-of-two byte count");
  }
  return static_cast<int64_t>(alignment->toUIntMax());
}

loom_value_fact_memory_space_t allocation_space(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const std::optional<cxx::ConstValue>& value, std::string_view operation,
    cxx::AST* owner) {
  auto* space = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!space || space->isNegative() || space->toUWide() > UINT8_MAX) {
    diagnostics.reject(
        unit, owner,
        std::string(operation) + " memory space must be a constant enum");
  }
  auto memory_space =
      static_cast<loom_value_fact_memory_space_t>(space->toUIntMax());
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return memory_space;
    default:
      diagnostics.reject(unit, owner,
                         std::string(operation) +
                             " requires an allocatable memory space: global, "
                             "workgroup, private, host, or generic");
  }
}

}  // namespace

bool BufferIntrinsic::supports(std::string_view name) {
  return name == "buffer.constant" || name == "buffer.alloca.init" ||
         name == "buffer.alloca" || name == "buffer.assume.alignment" ||
         name == "buffer.assume.noalias" ||
         name == "buffer.assume.memory_space" || name == "buffer.cast" ||
         name == "buffer.root";
}

std::optional<BufferIntrinsic> BufferIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  auto name = attribute.arguments[0]->name();
  if (!supports(name)) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature->parameterTypes();
  auto* result = cxx::type_cast<cxx::PointerType>(
      types.unqualified(signature->returnType()));
  if (name == "buffer.cast" || name == "buffer.root") {
    if (attribute.arguments.size() != 1 || signature->isVariadic() ||
        parameters.size() != 1) {
      diagnostics.reject(
          unit, owner,
          std::string(name) + " requires one typed storage operand");
    }
    const auto& source_partition = types.partition(parameters[0], owner);
    const auto& result_partition =
        types.partition(signature->returnType(), owner);
    const auto expected_source_kind =
        name == "buffer.root" ? ValueKind::Pointer : ValueKind::Buffer;
    if (source_partition.kind != expected_source_kind ||
        result_partition.kind != ValueKind::Buffer) {
      diagnostics.reject(
          unit, owner,
          std::string(name) +
              (name == "buffer.root"
                   ? " requires a pointer source and typed buffer result"
                   : " requires typed buffer source and result"));
    }
    const auto& result_buffer =
        static_cast<const BufferPartition&>(result_partition);
    auto traits = unit.typeTraits();
    const cxx::Type* source_element_type = nullptr;
    if (name == "buffer.root") {
      auto* pointer =
          cxx::type_cast<cxx::PointerType>(types.unqualified(parameters[0]));
      source_element_type = pointer ? pointer->elementType() : nullptr;
    } else {
      source_element_type =
          static_cast<const BufferPartition&>(source_partition).element_type;
    }
    if (!source_element_type) {
      diagnostics.reject(unit, owner,
                         std::string(name) + " has no source element type");
    }
    if (name == "buffer.root") {
      if (!traits.is_same(source_element_type, result_buffer.element_type)) {
        diagnostics.reject(unit, owner,
                           "buffer.root must preserve its source element type");
      }
    } else if ((traits.is_const(source_element_type) &&
                !traits.is_const(result_buffer.element_type)) ||
               (traits.is_volatile(source_element_type) &&
                !traits.is_volatile(result_buffer.element_type))) {
      diagnostics.reject(unit, owner,
                         "buffer.cast cannot discard element qualifiers");
    }
    return BufferIntrinsic(
        name == "buffer.root" ? Operation::Root : Operation::Cast,
        &result_buffer);
  }
  if (name == "buffer.constant" || name == "buffer.alloca.init") {
    const bool immutable = name == "buffer.constant";
    if (attribute.arguments.size() != 2 ||
        (attribute.arguments[1]->name() != "bytes" &&
         attribute.arguments[1]->name() != "utf8")) {
      diagnostics.reject(unit, owner,
                         "buffer literal requires a bytes or utf8 selector");
    }
    const bool is_utf8 = attribute.arguments[1]->name() == "utf8";
    auto traits = unit.typeTraits();
    const cxx::Type* character =
        is_utf8 ? static_cast<const cxx::Type*>(unit.control()->getChar8Type())
                : static_cast<const cxx::Type*>(unit.control()->getCharType());
    auto* expected_result = unit.control()->getUnsignedCharType();
    auto* parameter =
        parameters.size() == 1 && traits.is_lvalue_reference(parameters[0])
            ? traits.remove_reference(parameters[0])
            : nullptr;
    auto* array = parameter ? cxx::type_cast<cxx::BoundedArrayType>(
                                  types.unqualified(parameter))
                            : nullptr;
    if (signature->isVariadic() || !array || !array->size() ||
        !traits.is_const(parameter) || traits.is_volatile(parameter) ||
        !traits.is_same(types.unqualified(array->elementType()), character) ||
        !result ||
        !traits.is_same(
            result->elementType(),
            immutable ? traits.add_const(expected_result) : expected_result)) {
      diagnostics.reject(unit, owner,
                         "buffer literal requires a const character array "
                         "reference and an unsigned char pointer result "
                         "(const for buffer.constant)");
    }
    auto arguments = function->templateArguments();
    const size_t alignment_index = immutable ? 0 : 1;
    auto extent_value = arguments.size() == alignment_index + 2
                            ? cxx::template_argument_value(arguments.back())
                            : std::nullopt;
    auto* extent =
        extent_value ? std::get_if<cxx::ConstInt>(&*extent_value) : nullptr;
    if (!extent || extent->isNegative() || extent->toUWide() != array->size()) {
      diagnostics.reject(unit, owner,
                         "buffer literal requires alignment and array extent "
                         "template arguments, preceded by an allocatable "
                         "memory space for buffer.alloca.init");
    }
    const auto space =
        immutable ? LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT
                  : allocation_space(unit, diagnostics,
                                     cxx::template_argument_value(arguments[0]),
                                     name, owner);
    const auto alignment = byte_alignment(
        unit, diagnostics,
        cxx::template_argument_value(arguments[alignment_index]), name, owner);
    return BufferIntrinsic(
        immutable ? Operation::Constant : Operation::AllocaInit, space,
        alignment, is_utf8);
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner, "buffer operation accepts no selectors");
  }
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
  if (signature->isVariadic() || parameters.empty() || parameters.size() > 2 ||
      !result) {
    diagnostics.reject(
        unit, owner,
        "buffer.alloca requires a count, an optional pool, and a pointer "
        "result");
  }
  const bool has_pool = parameters.size() == 2;
  if (has_pool &&
      types.partition(parameters[1], owner).kind != ValueKind::Pool) {
    diagnostics.reject(unit, owner,
                       "buffer.alloca second operand must be a Loom pool");
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

  auto memory_space = allocation_space(
      unit, diagnostics,
      cxx::template_argument_value(arguments[space_argument]), name, owner);
  auto alignment_bytes = byte_alignment(
      unit, diagnostics,
      cxx::template_argument_value(arguments[alignment_argument]), name, owner);

  return allocates_bytes
             ? BufferIntrinsic(memory_space,
                               static_cast<int64_t>(alignment_bytes), has_pool)
             : BufferIntrinsic(element_type, parameters[0], memory_space,
                               static_cast<int64_t>(alignment_bytes), has_pool);
}

Value BufferIntrinsic::call(std::span<const Value> arguments, ValueArena& arena,
                            Storage& storage, cxx::AST* owner) const {
  if (operation_ == Operation::Cast) {
    return arena.capture(*result_buffer_, arguments[0].components());
  }
  if (operation_ == Operation::Root) {
    auto root = arguments[0].pointer().root;
    return arena.capture(*result_buffer_, {&root, 1});
  }
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
    return Value(storage.allocate_bytes(
        arguments[0].ssa(), memory_space_, alignment_,
        has_pool_ ? arguments[1].ssa() : LOOM_VALUE_ID_INVALID, owner));
  }
  return Value(storage.allocate_elements(
      element_type_, arguments[0].ssa(), element_count_type_, memory_space_,
      alignment_, has_pool_ ? arguments[1].ssa() : LOOM_VALUE_ID_INVALID,
      owner));
}

Value BufferIntrinsic::call_literal(cxx::TranslationUnit& unit,
                                    Diagnostics& diagnostics,
                                    cxx::ExpressionAST* argument,
                                    Storage& storage, cxx::AST* owner) const {
  auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(
      unwrap_expression(argument));
  if (!literal) {
    diagnostics.reject(unit, argument,
                       "buffer payload requires a string literal");
  }
  auto payload = literal->literal->stringValue();
  auto text = iree_make_string_view(payload.data(), payload.size());
  if (is_utf8_ && !iree_unicode_utf8_validate(text)) {
    diagnostics.reject(unit, argument,
                       "utf8 buffer payload must be valid UTF-8");
  }
  auto root =
      storage.literal(iree_make_const_byte_span(payload.data(), payload.size()),
                      is_utf8_, memory_space_, alignment_, owner);
  return Value(storage.root(root, owner));
}

bool BufferIntrinsic::equivalent(const BufferIntrinsic& other) const {
  return operation_ == other.operation_ && is_utf8_ == other.is_utf8_ &&
         element_type_ == other.element_type_ &&
         element_count_type_ == other.element_count_type_ &&
         result_buffer_ == other.result_buffer_ &&
         memory_space_ == other.memory_space_ &&
         alignment_ == other.alignment_ && has_pool_ == other.has_pool_;
}

}  // namespace loom::cxx_import
