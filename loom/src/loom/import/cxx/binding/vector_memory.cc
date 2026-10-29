// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/vector_memory.h"

#include <cxx/names.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>

#include "loom/import/cxx/binding/index_values.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {

const ViewPartition& require_view(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics, Types& types,
                                  const cxx::Type* type, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::View) {
    diagnostics.reject(unit, owner,
                       "vector memory operation requires a view value");
  }
  return static_cast<const ViewPartition&>(partition);
}

struct VectorContract {
  // Native C++ vector source type, or null for a semantic vector.
  const cxx::VectorType* native = nullptr;
  // Semantic vector partition, or null for a native C++ vector.
  const VectorPartition* semantic = nullptr;
  // Unqualified scalar element source type.
  const cxx::Type* element = nullptr;
};

VectorContract require_vector(cxx::TranslationUnit& unit,
                              Diagnostics& diagnostics, Types& types,
                              const cxx::Type* type, cxx::AST* owner) {
  if (auto* native = types.native_vector(type)) {
    return {.native = native, .element = native->elementType()};
  }
  const auto& admitted = types.partition(type, owner);
  if (admitted.kind == ValueKind::Vector) {
    const auto& semantic = static_cast<const VectorPartition&>(admitted);
    return {.semantic = &semantic, .element = semantic.element_type};
  }
  diagnostics.reject(unit, owner,
                     "vector memory operation requires a vector value");
}

const cxx::VectorType& require_native_vector(cxx::TranslationUnit& unit,
                                             Diagnostics& diagnostics,
                                             Types& types,
                                             const cxx::Type* type,
                                             cxx::AST* owner) {
  auto* vector = types.native_vector(type);
  if (!vector) {
    diagnostics.reject(unit, owner,
                       "masked vector memory requires a native vector value");
  }
  return *vector;
}

void require_matching_element(cxx::TranslationUnit& unit,
                              Diagnostics& diagnostics, Types& types,
                              const cxx::Type* vector_element,
                              const ViewPartition& view, cxx::AST* owner) {
  if (types.unqualified(vector_element) !=
      types.unqualified(view.element_type)) {
    diagnostics.reject(
        unit, owner,
        "vector memory element type must match the view element type");
  }
}

loom_value_id_t view_component(Value value) {
  return value.components().back();
}

}  // namespace

std::optional<VectorMemoryIntrinsic::Operation>
VectorMemoryIntrinsic::parse_operation(std::string_view name) {
  if (name == "vector.load") {
    return Operation::Load;
  }
  if (name == "vector.store") {
    return Operation::Store;
  }
  if (name == "vector.load.mask") {
    return Operation::LoadMask;
  }
  if (name == "vector.store.mask") {
    return Operation::StoreMask;
  }
  if (name == "vector.load.expand") {
    return Operation::LoadExpand;
  }
  if (name == "vector.store.compress") {
    return Operation::StoreCompress;
  }
  return std::nullopt;
}

bool VectorMemoryIntrinsic::supports(std::string_view name) {
  return parse_operation(name).has_value();
}

bool VectorMemoryIntrinsic::is_load() const {
  return operation_ == Operation::Load || operation_ == Operation::LoadMask ||
         operation_ == Operation::LoadExpand;
}

bool VectorMemoryIntrinsic::has_mask() const {
  return operation_ == Operation::LoadMask ||
         operation_ == Operation::StoreMask ||
         operation_ == Operation::LoadExpand ||
         operation_ == Operation::StoreCompress;
}

std::optional<VectorMemoryIntrinsic> VectorMemoryIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    const cxx::FunctionType* signature, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  auto operation = parse_operation(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1 || signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "vector memory operations accept one operation name "
                       "and a fixed source signature");
  }

  auto parameters = signature->parameterTypes();
  VectorMemoryIntrinsic result(*operation);
  size_t view_index;
  size_t origin_index;
  size_t mask_index = 0;
  size_t passthrough_index = 0;
  VectorContract vector;
  switch (*operation) {
    case Operation::Load:
      if (parameters.size() != 2 && parameters.size() != 3) {
        diagnostics.reject(unit, owner,
                           "vector.load requires a view, full-rank origin, and "
                           "optional result dimensions");
      }
      view_index = 0;
      origin_index = 1;
      vector = require_vector(unit, diagnostics, types, signature->returnType(),
                              owner);
      if (vector.native && parameters.size() != 2) {
        diagnostics.reject(unit, owner,
                           "native vector.load does not accept result "
                           "dimensions");
      }
      if (vector.semantic) {
        if (vector.semantic->is_dynamic() && parameters.size() != 3) {
          diagnostics.reject(
              unit, owner, "dependent vector.load requires result dimensions");
        }
        if (parameters.size() == 3) {
          require_logical_index_record(unit, diagnostics, types, parameters[2],
                                       vector.semantic->component_count - 1,
                                       owner, "vector.load dimensions");
          result.dimensions_index_ = 2;
        }
      }
      break;
    case Operation::Store:
      if (parameters.size() != 3 ||
          !unit.typeTraits().is_void(signature->returnType())) {
        diagnostics.reject(
            unit, owner,
            "vector.store requires a value, view, and full-rank origin");
      }
      view_index = 1;
      origin_index = 2;
      vector = require_vector(unit, diagnostics, types, parameters[0], owner);
      break;
    case Operation::LoadMask:
    case Operation::LoadExpand:
      if (parameters.size() != 4) {
        diagnostics.reject(unit, owner,
                           "masked vector load requires a view, full-rank "
                           "origin, mask, and passthrough");
      }
      view_index = 0;
      origin_index = 1;
      mask_index = 2;
      passthrough_index = 3;
      vector.native = &require_native_vector(unit, diagnostics, types,
                                             signature->returnType(), owner);
      vector.element = vector.native->elementType();
      break;
    case Operation::StoreMask:
    case Operation::StoreCompress:
      if (parameters.size() != 4 ||
          !unit.typeTraits().is_void(signature->returnType())) {
        diagnostics.reject(unit, owner,
                           "masked vector store requires a value, view, "
                           "full-rank origin, and mask");
      }
      view_index = 1;
      origin_index = 2;
      mask_index = 3;
      vector.native = &require_native_vector(unit, diagnostics, types,
                                             parameters[0], owner);
      vector.element = vector.native->elementType();
      break;
  }

  result.view_ =
      &require_view(unit, diagnostics, types, parameters[view_index], owner);
  if (!result.is_load() &&
      unit.typeTraits().is_const(result.view_->element_type)) {
    diagnostics.reject(
        unit, owner,
        *operation == Operation::Store
            ? "vector.store requires a mutable destination view"
            : "masked vector store requires a mutable destination view");
  }
  require_matching_element(unit, diagnostics, types, vector.element,
                           *result.view_, owner);
  result.semantic_vector_ = vector.semantic;
  if (vector.native) {
    result.vector_type_ = types.get(vector.native, owner);
  }
  if (result.has_mask()) {
    if (result.view_->access_flags != 0) {
      diagnostics.reject(
          unit, owner, "masked vector memory does not support volatile views");
    }
    const auto& mask = require_native_vector(unit, diagnostics, types,
                                             parameters[mask_index], owner);
    if (!unit.typeTraits().is_integral(mask.elementType()) ||
        types.unqualified(mask.elementType())->kind() == cxx::TypeKind::kBool) {
      diagnostics.reject(unit, owner,
                         "vector memory mask requires integer lanes");
    }
    if (mask.elementCount() != vector.native->elementCount()) {
      diagnostics.reject(unit, owner,
                         "vector memory mask shape must match the value");
    }
    result.mask_type_ = types.get(&mask, owner);
    result.predicate_type_ = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, mask.elementCount(), 0);
    if (result.is_load() && types.unqualified(parameters[passthrough_index]) !=
                                types.unqualified(signature->returnType())) {
      diagnostics.reject(
          unit, owner,
          "masked vector load passthrough must match its result type");
    }
  }
  result.origin_unsigned_mask_ = require_index_record(
      unit, diagnostics, types, parameters[origin_index],
      result.view_->extents.size(), owner, "vector memory origin");
  return result;
}

loom_value_id_t VectorMemoryIntrinsic::build_mask(
    loom_value_id_t source, loom_builder_t* builder,
    loom_location_id_t location) const {
  loom_op_t* op;
  check(loom_vector_splati_build(builder, 0, mask_type_, location, &op));
  check(loom_vector_cmpi_build(builder, LOOM_VECTOR_CMPI_PREDICATE_NE, source,
                               loom_op_results(op)[0], mask_type_,
                               predicate_type_, location, &op));
  return loom_op_results(op)[0];
}

std::optional<Value> VectorMemoryIntrinsic::call(
    std::span<const Value> arguments, ValueArena& arena,
    loom_builder_t* builder, loom_location_id_t location) const {
  const size_t view_index = is_load() ? 0 : 1;
  const size_t origin_index = is_load() ? 1 : 2;
  auto origin = arguments[origin_index].components();
  std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
  std::array<int64_t, LOOM_TYPE_MAX_RANK> static_indices;
  static_indices.fill(std::numeric_limits<int64_t>::min());
  for (size_t axis = 0; axis < origin.size(); ++axis) {
    indices[axis] = cast_index(
        Value(origin[axis]), is_unsigned_component(origin_unsigned_mask_, axis),
        builder, location);
  }

  loom_op_t* op;
  const auto view = view_component(arguments[view_index]);
  auto vector_type = vector_type_;
  std::span<const loom_value_id_t> dimensions;
  BoundTypeStorage type_storage;
  if (semantic_vector_ && is_load()) {
    if (dimensions_index_) {
      dimensions = arguments[*dimensions_index_].components();
    }
    vector_type = semantic_vector_->bind_type(dimensions, type_storage);
  }
  auto vector_value = [&](size_t index) {
    return semantic_vector_
               ? semantic_vector_->vector_value(arguments[index].components())
               : arguments[index].ssa();
  };
  switch (operation_) {
    case Operation::Load:
      check(loom_vector_load_build(builder, 0, view_->access_flags, view,
                                   indices.data(), origin.size(),
                                   static_indices.data(), origin.size(), 0, 0,
                                   vector_type, location, &op));
      if (!semantic_vector_) {
        return Value(loom_op_results(op)[0]);
      }
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> components;
      std::copy(dimensions.begin(), dimensions.end(), components.begin());
      components[dimensions.size()] = loom_op_results(op)[0];
      return arena.capture(*semantic_vector_,
                           std::span<const loom_value_id_t>(components)
                               .first(dimensions.size() + 1));
    case Operation::Store:
      check(loom_vector_store_build(builder, 0, view_->access_flags,
                                    vector_value(0), view, indices.data(),
                                    origin.size(), static_indices.data(),
                                    origin.size(), 0, 0, location, &op));
      return std::nullopt;
    case Operation::LoadMask:
    case Operation::LoadExpand: {
      const auto mask = build_mask(arguments[2].ssa(), builder, location);
      const auto build = operation_ == Operation::LoadMask
                             ? loom_vector_load_mask_build
                             : loom_vector_load_expand_build;
      check(build(builder, 0, view, indices.data(), origin.size(),
                  static_indices.data(), origin.size(), mask,
                  arguments[3].ssa(), 0, 0, vector_type_, location, &op));
      return Value(loom_op_results(op)[0]);
    }
    case Operation::StoreMask:
    case Operation::StoreCompress: {
      const auto mask = build_mask(arguments[3].ssa(), builder, location);
      const auto build = operation_ == Operation::StoreMask
                             ? loom_vector_store_mask_build
                             : loom_vector_store_compress_build;
      check(build(builder, 0, arguments[0].ssa(), view, indices.data(),
                  origin.size(), static_indices.data(), origin.size(), mask, 0,
                  0, location, &op));
      return std::nullopt;
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown vector memory operation");
  IREE_BUILTIN_UNREACHABLE();
}

bool VectorMemoryIntrinsic::equivalent(
    const VectorMemoryIntrinsic& other) const {
  return operation_ == other.operation_ && view_ == other.view_ &&
         loom_type_equal(vector_type_, other.vector_type_) &&
         semantic_vector_ == other.semantic_vector_ &&
         dimensions_index_ == other.dimensions_index_ &&
         (!has_mask() || loom_type_equal(mask_type_, other.mask_type_)) &&
         origin_unsigned_mask_ == other.origin_unsigned_mask_;
}

}  // namespace loom::cxx_import
