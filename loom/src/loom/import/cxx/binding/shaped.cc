// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/shaped.h"

#include <cxx/names.h>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

#include "loom/import/cxx/binding/combining.h"
#include "loom/import/cxx/binding/index_values.h"
#include "loom/import/cxx/binding/scalar_bindings.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/combining.h"

namespace loom::cxx_import {
namespace {

uint8_t math_permissions(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         const cxx::Attribute& attribute, size_t first,
                         cxx::AST* owner) {
  uint8_t flags = 0;
  for (size_t index = first; index < attribute.arguments.size(); ++index) {
    uint8_t flag;
    if (!loom_cxx_fastmath_flag_parse(view(attribute.arguments[index]->name()),
                                      &flag)) {
      diagnostics.reject(unit, owner,
                         "intrinsic has an unsupported fast-math flag");
    }
    flags |= flag;
  }
  return flags;
}

loom_vector_dot4i_kind_t dot4i_kind(bool lhs_unsigned, bool rhs_unsigned) {
  if (lhs_unsigned) {
    return rhs_unsigned ? LOOM_VECTOR_DOT4I_KIND_U8U8
                        : LOOM_VECTOR_DOT4I_KIND_U8S8;
  }
  return rhs_unsigned ? LOOM_VECTOR_DOT4I_KIND_S8U8
                      : LOOM_VECTOR_DOT4I_KIND_S8S8;
}

}  // namespace

std::optional<ShapedIntrinsic::Operation> ShapedIntrinsic::admit(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const cxx::Attribute& attribute, cxx::AST* owner) {
  auto name = attribute.arguments[0]->name();
  if (name == "vector.splat") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "vector splat has no semantic arguments");
    }
    return Splat{};
  }
  if (name == "vector.slice") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "vector slice has no semantic arguments");
    }
    return Slice{};
  }
  if (name == "vector.table.lookup") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "register table lookup has no semantic arguments");
    }
    return TableLookup{};
  }
  if (name == "vector.absf" || name == "vector.absi") {
    const bool floating = name == "vector.absf";
    if (!floating && attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner,
                         "integer vector absolute value has no semantic "
                         "arguments");
    }
    return Absolute{
        floating ? Absolute::Kind::Float : Absolute::Kind::Integer,
        floating ? math_permissions(unit, diagnostics, attribute, 1, owner)
                 : uint8_t{0},
    };
  }
  if (name == "vector.minimumf" || name == "vector.maximumf" ||
      name == "vector.minnumf" || name == "vector.maxnumf") {
    const FloatExtremum::Kind kind =
        name == "vector.minimumf"   ? FloatExtremum::Kind::Minimum
        : name == "vector.maximumf" ? FloatExtremum::Kind::Maximum
        : name == "vector.minnumf"  ? FloatExtremum::Kind::MinNumber
                                    : FloatExtremum::Kind::MaxNumber;
    return FloatExtremum{
        kind, math_permissions(unit, diagnostics, attribute, 1, owner)};
  }
  if (name == "vector.dot4i") {
    if (attribute.arguments.size() == 1) {
      return Dot4i{std::nullopt};
    }
    if (attribute.arguments.size() != 2) {
      diagnostics.reject(unit, owner,
                         "dot4i accepts no semantic arguments or one "
                         "signedness argument: s8s8, u8s8, s8u8, or u8u8");
    }
    auto kind = attribute.arguments[1]->name();
    if (kind == "s8s8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_S8S8};
    }
    if (kind == "u8s8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_U8S8};
    }
    if (kind == "s8u8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_S8U8};
    }
    if (kind == "u8u8") {
      return Dot4i{LOOM_VECTOR_DOT4I_KIND_U8U8};
    }
    diagnostics.reject(unit, owner, "unsupported dot4i signedness argument");
  }
  if (name == "vector.dot2f") {
    if (attribute.arguments.size() != 1) {
      diagnostics.reject(unit, owner, "dot2f has no semantic arguments");
    }
    return Dot2f{};
  }
  if (name == "vector.dotf") {
    return Dotf{math_permissions(unit, diagnostics, attribute, 1, owner)};
  }
  if (name == "vector.reduce") {
    if (attribute.arguments.size() < 2) {
      diagnostics.reject(unit, owner,
                         "vector reduction requires a combining kind");
    }
    auto kind = parse_combining_kind(attribute.arguments[1]->name());
    if (!kind) {
      diagnostics.reject(unit, owner,
                         "unsupported vector reduction combining kind");
    }
    if (loom_combining_kind_accepts_integer(*kind) &&
        attribute.arguments.size() != 2) {
      diagnostics.reject(unit, owner,
                         "integer vector reduction has no fast-math flags");
    }
    return Reduction{*kind,
                     math_permissions(unit, diagnostics, attribute, 2, owner)};
  }
  return std::nullopt;
}

ShapedIntrinsic ShapedIntrinsic::resolve(Operation operation,
                                         cxx::TranslationUnit& unit,
                                         Diagnostics& diagnostics, Types& types,
                                         const cxx::FunctionType* signature,
                                         cxx::AST* owner) {
  if (std::holds_alternative<Splat>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector.splat requires dimensions and one scalar");
    }
    const auto& admitted = types.partition(signature->returnType(), owner);
    if (admitted.kind != ValueKind::Vector) {
      diagnostics.reject(unit, owner,
                         "vector.splat result must be a semantic vector");
    }
    const auto& result = static_cast<const VectorPartition&>(admitted);
    require_logical_index_record(
        unit, diagnostics, types, signature->parameterTypes()[0],
        result.component_count - 1, owner, "vector.splat dimensions");
    if (types.unqualified(signature->parameterTypes()[1]) !=
        types.unqualified(result.element_type)) {
      diagnostics.reject(unit, owner,
                         "vector.splat value must match its result element");
    }
    return ShapedIntrinsic(operation, loom_type_none(), &result);
  }
  if (auto* slice = std::get_if<Slice>(&operation)) {
    auto parameters = signature->parameterTypes();
    if (signature->isVariadic() ||
        (parameters.size() != 2 && parameters.size() != 3)) {
      diagnostics.reject(unit, owner,
                         "vector.slice requires a source, offsets, and "
                         "optional result dimensions");
    }
    const std::array<const cxx::Type*, 2> vectors = {parameters[0],
                                                     signature->returnType()};
    std::array<const cxx::Type*, 2> elements;
    std::array<const VectorPartition*, 2> semantic = {};
    std::array<int64_t, 2> native_extents;
    std::array<std::span<const int64_t>, 2> extents;
    for (size_t index = 0; index < vectors.size(); ++index) {
      if (const auto* native = types.native_vector(vectors[index])) {
        elements[index] = types.unqualified(native->elementType());
        native_extents[index] = native->elementCount();
        extents[index] = {&native_extents[index], 1};
      } else {
        const auto& admitted = types.partition(vectors[index], owner);
        if (admitted.kind != ValueKind::Vector) {
          diagnostics.reject(unit, owner,
                             "vector.slice requires source and result vectors");
        }
        semantic[index] = &static_cast<const VectorPartition&>(admitted);
        elements[index] = semantic[index]->element_type;
        extents[index] = semantic[index]->extents;
      }
    }
    if (extents[0].size() != extents[1].size()) {
      diagnostics.reject(unit, owner,
                         "vector.slice must preserve the source rank");
    }
    bool fits = true;
    for (size_t axis = 0; axis < extents[0].size(); ++axis) {
      if (extents[0][axis] >= 0 && extents[1][axis] > extents[0][axis]) {
        fits = false;
      }
    }
    if (elements[0] != elements[1] || !fits) {
      diagnostics.reject(
          unit, owner,
          "vector.slice result must preserve the element type and fit within "
          "the source");
    }
    const auto& offsets = types.partition(parameters[1], owner);
    if (offsets.kind == ValueKind::Record) {
      slice->offset_unsigned_mask = require_index_record(
          unit, diagnostics, types, parameters[1], extents[0].size(), owner,
          "vector.slice offsets");
    } else {
      if (extents[0].size() != 1) {
        diagnostics.reject(unit, owner,
                           "vector.slice offsets require a full-rank record");
      }
      slice->offset_unsigned_mask =
          require_index_value(unit, diagnostics, types, parameters[1], owner,
                              "vector.slice offset");
    }
    slice->has_dimensions = parameters.size() == 3;
    if (semantic[1]) {
      if (semantic[1]->is_dynamic() && !slice->has_dimensions) {
        diagnostics.reject(unit, owner,
                           "dependent vector.slice requires result dimensions");
      }
      if (slice->has_dimensions) {
        require_logical_index_record(unit, diagnostics, types, parameters[2],
                                     semantic[1]->component_count - 1, owner,
                                     "vector.slice dimensions");
      }
    } else if (slice->has_dimensions) {
      diagnostics.reject(unit, owner,
                         "native vector.slice does not accept result "
                         "dimensions");
    }
    return ShapedIntrinsic(
        operation,
        semantic[1] ? loom_type_none() : types.get(vectors[1], owner),
        semantic[1], {semantic[0], nullptr, nullptr});
  }
  bool lookup = std::holds_alternative<TableLookup>(operation);
  if (auto* absolute = std::get_if<Absolute>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 1) {
      diagnostics.reject(unit, owner,
                         "vector absolute value requires one operand");
    }
    const auto* result = types.native_vector(signature->returnType());
    const auto* input = types.native_vector(signature->parameterTypes()[0]);
    if (!result || !input || result != input) {
      diagnostics.reject(
          unit, owner,
          "vector absolute value requires one vector type for its operand "
          "and result");
    }
    const auto* element = types.unqualified(result->elementType());
    const bool category_matches =
        absolute->kind == Absolute::Kind::Float
            ? types.is_float(element)
            : unit.typeTraits().is_integral(element) &&
                  element->kind() != cxx::TypeKind::kBool;
    if (!category_matches) {
      diagnostics.reject(unit, owner,
                         "vector absolute value element type does not match "
                         "the operation category");
    }
    return ShapedIntrinsic(operation, types.get(result, owner));
  }
  if (std::holds_alternative<FloatExtremum>(operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "floating vector extremum requires two operands");
    }
    const auto* result = types.native_vector(signature->returnType());
    const auto* left = types.native_vector(signature->parameterTypes()[0]);
    const auto* right = types.native_vector(signature->parameterTypes()[1]);
    if (!result || result != left || result != right ||
        !types.is_float(result->elementType())) {
      diagnostics.reject(
          unit, owner,
          "floating vector extremum requires one floating-point vector type "
          "for its operands and result");
    }
    return ShapedIntrinsic(operation, types.get(result, owner));
  }
  if (auto* reduction = std::get_if<Reduction>(&operation)) {
    if (signature->isVariadic() || signature->parameterTypes().size() != 2) {
      diagnostics.reject(unit, owner,
                         "vector reduction requires a vector and scalar seed");
    }
    const auto* result = types.unqualified(signature->returnType());
    if (!types.is_float(result) && (!unit.typeTraits().is_integral(result) ||
                                    result->kind() == cxx::TypeKind::kBool)) {
      diagnostics.reject(unit, owner,
                         "vector reduction result must be a numeric scalar");
    }
    const auto* native = types.native_vector(signature->parameterTypes()[0]);
    const VectorPartition* semantic = nullptr;
    const cxx::Type* element = nullptr;
    if (native) {
      element = native->elementType();
    } else {
      const auto& admitted =
          types.partition(signature->parameterTypes()[0], owner);
      if (admitted.kind == ValueKind::Vector) {
        semantic = &static_cast<const VectorPartition&>(admitted);
        element = semantic->element_type;
      }
    }
    if (!element || types.unqualified(element) != result ||
        types.unqualified(signature->parameterTypes()[1]) != result) {
      diagnostics.reject(
          unit, owner,
          "vector element, scalar seed, and result types must match");
    }
    if (types.is_float(result) !=
        loom_combining_kind_accepts_float(reduction->kind)) {
      diagnostics.reject(unit, owner,
                         "vector reduction combining kind does not match the "
                         "element type");
    }
    if (((reduction->kind == LOOM_COMBINING_KIND_MINSI ||
          reduction->kind == LOOM_COMBINING_KIND_MAXSI) &&
         types.is_unsigned(result)) ||
        ((reduction->kind == LOOM_COMBINING_KIND_MINUI ||
          reduction->kind == LOOM_COMBINING_KIND_MAXUI) &&
         !types.is_unsigned(result))) {
      diagnostics.reject(unit, owner,
                         "vector reduction signedness must match the source "
                         "element type");
    }
    std::array<const VectorPartition*, 3> semantic_operands = {};
    semantic_operands[0] = semantic;
    return ShapedIntrinsic(operation, types.get(result, owner), nullptr,
                           semantic_operands);
  }
  bool scalar_result = std::holds_alternative<Dotf>(operation);
  size_t operand_count = lookup ? 2 : 3;
  if (signature->isVariadic() ||
      signature->parameterTypes().size() != operand_count) {
    diagnostics.reject(unit, owner,
                       "intrinsic declaration has the wrong operand count");
  }
  const auto* result = types.unqualified(signature->returnType());
  const auto* result_vector = types.native_vector(result);
  if (!scalar_result && !result_vector) {
    diagnostics.reject(unit, owner, "shaped intrinsic result must be a vector");
  }
  if (scalar_result && !types.is_float(result) &&
      (!unit.typeTraits().is_integral(result) ||
       result->kind() == cxx::TypeKind::kBool)) {
    diagnostics.reject(
        unit, owner,
        "vector reduction or dotf result must be a numeric scalar");
  }
  auto result_type = types.get(result, owner);
  size_t vector_count = operand_count - (scalar_result ? 1 : 0);
  std::array<const cxx::VectorType*, 3> operands;
  std::array<loom_type_t, 3> operand_types;
  for (size_t index = 0; index < vector_count; ++index) {
    operands[index] = types.native_vector(signature->parameterTypes()[index]);
    if (!operands[index]) {
      diagnostics.reject(unit, owner,
                         "shaped intrinsic operands must be vectors");
    }
    operand_types[index] = types.get(operands[index], owner);
  }
  if (scalar_result &&
      (types.unqualified(operands[0]->elementType()) != result ||
       types.unqualified(signature->parameterTypes().back()) != result)) {
    diagnostics.reject(
        unit, owner,
        "vector element, scalar seed, and result types must match");
  }

  if (lookup) {
    if (types.unqualified(operands[0]->elementType()) !=
            types.unqualified(result_vector->elementType()) ||
        operands[1]->elementCount() != result_vector->elementCount() ||
        !unit.typeTraits().is_integral(operands[1]->elementType())) {
      diagnostics.reject(unit, owner,
                         "register table lookup requires the table's element "
                         "type and the integer index vector's lane count");
    }
  } else if (auto* dot = std::get_if<Dot4i>(&operation)) {
    if (loom_type_element_type(operand_types[0]) != LOOM_SCALAR_TYPE_I8 ||
        loom_type_element_type(operand_types[1]) != LOOM_SCALAR_TYPE_I8 ||
        loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_I32 ||
        operands[0]->elementCount() != operands[1]->elementCount() ||
        operands[0]->elementCount() != result_vector->elementCount() * 4 ||
        operands[2] != result_vector) {
      diagnostics.reject(unit, owner,
                         "dot4i requires equal i8 input shapes with four lanes "
                         "per i32 accumulator/result lane");
    }
    auto source_kind =
        dot4i_kind(types.is_unsigned(operands[0]->elementType()),
                   types.is_unsigned(operands[1]->elementType()));
    if (!dot->kind) {
      dot->kind = source_kind;
    } else if (*dot->kind != source_kind) {
      diagnostics.reject(unit, owner,
                         "dot4i signedness must match the source byte types");
    }
  } else if (std::holds_alternative<Dot2f>(operation)) {
    auto input_element = loom_type_element_type(operand_types[0]);
    if ((input_element != LOOM_SCALAR_TYPE_F16 &&
         input_element != LOOM_SCALAR_TYPE_BF16) ||
        operands[0] != operands[1] ||
        loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_F32 ||
        operands[0]->elementCount() != result_vector->elementCount() * 2 ||
        operands[2] != result_vector) {
      diagnostics.reject(
          unit, owner,
          "dot2f requires equal f16 or bf16 input types with two "
          "lanes per f32 accumulator/result lane");
    }
  } else if (std::holds_alternative<Dotf>(operation)) {
    if (!types.is_float(result) || operands[0] != operands[1]) {
      diagnostics.reject(unit, owner,
                         "dotf requires equal floating-point vector types");
    }
  }
  return ShapedIntrinsic(operation, result_type);
}

Value ShapedIntrinsic::call(std::span<const Value> arguments, ValueArena& arena,
                            uint8_t math_flags, loom_builder_t* builder,
                            loom_location_id_t location) const {
  if (std::holds_alternative<Splat>(operation_)) {
    auto dimensions = arguments[0].components();
    BoundTypeStorage type_storage;
    auto result_type = result_vector_->bind_type(dimensions, type_storage);
    loom_op_t* op;
    check(loom_vector_splat_build(builder, arguments[1].ssa(), result_type,
                                  location, &op));
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> components;
    std::copy(dimensions.begin(), dimensions.end(), components.begin());
    components[dimensions.size()] = loom_op_results(op)[0];
    return arena.capture(*result_vector_,
                         std::span<const loom_value_id_t>(components)
                             .first(dimensions.size() + 1));
  }
  auto vector_value = [&](size_t index) {
    const auto* semantic = semantic_operands_[index];
    return semantic ? semantic->vector_value(arguments[index].components())
                    : arguments[index].ssa();
  };
  if (auto* slice = std::get_if<Slice>(&operation_)) {
    auto offsets = arguments[1].components();
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
    std::array<int64_t, LOOM_TYPE_MAX_RANK> static_offsets;
    static_offsets.fill(std::numeric_limits<int64_t>::min());
    for (size_t axis = 0; axis < offsets.size(); ++axis) {
      indices[axis] =
          cast_index(Value(offsets[axis]),
                     is_unsigned_component(slice->offset_unsigned_mask, axis),
                     builder, location);
    }
    std::span<const loom_value_id_t> dimensions;
    if (slice->has_dimensions) {
      dimensions = arguments[2].components();
    }
    BoundTypeStorage type_storage;
    auto result_type = result_vector_
                           ? result_vector_->bind_type(dimensions, type_storage)
                           : result_type_;
    loom_op_t* op;
    check(loom_vector_slice_build(builder, vector_value(0), indices.data(),
                                  offsets.size(), static_offsets.data(),
                                  offsets.size(), result_type, location, &op));
    if (!result_vector_) {
      return Value(loom_op_results(op)[0]);
    }
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> components;
    std::copy(dimensions.begin(), dimensions.end(), components.begin());
    components[dimensions.size()] = loom_op_results(op)[0];
    return arena.capture(*result_vector_,
                         std::span<const loom_value_id_t>(components)
                             .first(dimensions.size() + 1));
  }
  loom_op_t* op;
  check(std::visit(
      [&](auto operation) -> iree_status_t {
        using T = decltype(operation);
        if constexpr (std::is_same_v<T, Splat> || std::is_same_v<T, Slice>) {
          IREE_ASSERT_UNREACHABLE(
              "shaped result handled before static dispatch");
          IREE_BUILTIN_UNREACHABLE();
        } else if constexpr (std::is_same_v<T, TableLookup>) {
          return loom_vector_table_lookup_build(builder, arguments[0].ssa(),
                                                arguments[1].ssa(),
                                                result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Absolute>) {
          if (operation.kind == Absolute::Kind::Float) {
            return loom_vector_absf_build(builder, operation.flags | math_flags,
                                          arguments[0].ssa(), result_type_,
                                          location, &op);
          }
          return loom_vector_absi_build(builder, arguments[0].ssa(),
                                        result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, FloatExtremum>) {
          auto build = operation.kind == FloatExtremum::Kind::Minimum
                           ? loom_vector_minimumf_build
                       : operation.kind == FloatExtremum::Kind::Maximum
                           ? loom_vector_maximumf_build
                       : operation.kind == FloatExtremum::Kind::MinNumber
                           ? loom_vector_minnumf_build
                           : loom_vector_maxnumf_build;
          return build(builder, operation.flags | math_flags,
                       arguments[0].ssa(), arguments[1].ssa(), result_type_,
                       location, &op);
        } else if constexpr (std::is_same_v<T, Dot4i>) {
          return loom_vector_dot4i_build(
              builder, *operation.kind, arguments[0].ssa(), arguments[1].ssa(),
              arguments[2].ssa(), result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Dot2f>) {
          return loom_vector_dot2f_build(builder, arguments[0].ssa(),
                                         arguments[1].ssa(), arguments[2].ssa(),
                                         result_type_, location, &op);
        } else if constexpr (std::is_same_v<T, Dotf>) {
          return loom_vector_dotf_build(builder, operation.flags | math_flags,
                                        arguments[0].ssa(), arguments[1].ssa(),
                                        arguments[2].ssa(), result_type_,
                                        location, &op);
        } else if constexpr (std::is_same_v<T, Reduction>) {
          uint8_t flags = loom_combining_kind_accepts_float(operation.kind)
                              ? operation.flags | math_flags
                              : 0;
          return loom_vector_reduce_build(builder, operation.kind, flags,
                                          vector_value(0), arguments[1].ssa(),
                                          result_type_, location, &op);
        }
      },
      operation_));
  return Value(loom_op_results(op)[0]);
}

bool ShapedIntrinsic::equivalent(const ShapedIntrinsic& other) const {
  return operation_ == other.operation_ &&
         loom_type_equal(result_type_, other.result_type_) &&
         result_vector_ == other.result_vector_ &&
         semantic_operands_ == other.semantic_operands_;
}

}  // namespace loom::cxx_import
