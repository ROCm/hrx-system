// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/view.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/index_values.h"
#include "loom/import/cxx/binding/memory_access.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/storage.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {

static_assert(LOOM_TYPE_MAX_RANK + 2 <= std::numeric_limits<uint32_t>::digits);
const EncodingPartition& require_encoding(cxx::TranslationUnit& unit,
                                          Diagnostics& diagnostics,
                                          Types& types, const cxx::Type* type,
                                          cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::Encoding) {
    diagnostics.reject(unit, owner,
                       "view operation requires an encoding value");
  }
  return static_cast<const EncodingPartition&>(partition);
}

const EncodingPartition& require_layout(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics, Types& types,
                                        const cxx::Type* type,
                                        cxx::AST* owner) {
  const auto& encoding =
      require_encoding(unit, diagnostics, types, type, owner);
  if (encoding.role != LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    diagnostics.reject(unit, owner,
                       "layout operation requires a layout encoding value");
  }
  return encoding;
}

const ViewPartition& require_view(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics, Types& types,
                                  const cxx::Type* type, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::View) {
    diagnostics.reject(unit, owner, "view operation requires a view value");
  }
  return static_cast<const ViewPartition&>(partition);
}

size_t dynamic_extent_count(const ViewPartition& view) {
  return std::count_if(view.extents.begin(), view.extents.end(),
                       [](int64_t extent) { return extent < 0; });
}

loom_value_id_t view_component(Value value) {
  return value.components().back();
}

loom_value_id_t encoding_component(Value value, const ViewPartition& view) {
  IREE_ASSERT(view.encoding);
  auto components = value.components();
  return components[components.size() - 2];
}

bool plain_subview_preserves_dense_suffix(const ViewPartition& source,
                                          const ViewPartition& result) {
  for (size_t axis = 1; axis < source.extents.size(); ++axis) {
    if (source.extents[axis] < 0 ||
        source.extents[axis] != result.extents[axis]) {
      return false;
    }
  }
  return true;
}

bool subview_preserves_dense_suffix(
    Value source, const ViewPartition& source_view,
    const ViewPartition& result_view,
    std::span<const loom_value_id_t> result_dynamic_extents) {
  auto source_components = source.components();
  size_t source_dynamic_index = 0;
  size_t result_dynamic_index = 0;
  for (size_t axis = 0; axis < source_view.extents.size(); ++axis) {
    const bool source_dynamic = source_view.extents[axis] < 0;
    const bool result_dynamic = result_view.extents[axis] < 0;
    if (axis > 0) {
      if (source_dynamic != result_dynamic) {
        return false;
      }
      if (source_dynamic) {
        if (source_components[source_dynamic_index] !=
            result_dynamic_extents[result_dynamic_index]) {
          return false;
        }
      } else if (source_view.extents[axis] != result_view.extents[axis]) {
        return false;
      }
    }
    source_dynamic_index += source_dynamic;
    result_dynamic_index += result_dynamic;
  }
  return true;
}

uint64_t static_view_bit_count(cxx::TranslationUnit& unit,
                               Diagnostics& diagnostics,
                               const ViewPartition& view, std::string_view role,
                               cxx::AST* owner) {
  const int32_t element_bits = loom_scalar_type_bitwidth(view.element);
  if (element_bits < 8 || element_bits % 8 != 0) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " requires a byte-addressable scalar element");
  }
  uint64_t count = 1;
  for (int64_t extent : view.extents) {
    if (extent <= 0 || count > UINT64_MAX / static_cast<uint64_t>(extent)) {
      diagnostics.reject(
          unit, owner,
          std::string(role) + " requires an all-static representable shape");
    }
    count *= static_cast<uint64_t>(extent);
  }
  if (count > UINT64_MAX / static_cast<uint64_t>(element_bits)) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " has an unrepresentable total bit count");
  }
  return count * static_cast<uint64_t>(element_bits);
}

}  // namespace

bool ViewIntrinsic::supports(std::string_view name) {
  return parse_operation(name).has_value();
}

std::optional<ViewIntrinsic::Operation> ViewIntrinsic::parse_operation(
    std::string_view name) {
  if (name == "encoding.layout.dense") {
    return Operation::LayoutDense;
  }
  if (name == "encoding.layout.strided") {
    return Operation::LayoutStrided;
  }
  if (name == "buffer.view") {
    return Operation::BufferView;
  }
  if (name == "view.subview") {
    return Operation::Subview;
  }
  if (name == "view.bitcast") {
    return Operation::Bitcast;
  }
  if (name == "view.load") {
    return Operation::Load;
  }
  if (name == "view.store") {
    return Operation::Store;
  }
  return std::nullopt;
}

std::optional<ViewIntrinsic> ViewIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  auto selected = parse_operation(attribute.arguments[0]->name());
  if (!selected) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (attribute.arguments.size() != 1 || signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "view operations accept one operation name and a fixed "
                       "source signature");
  }

  auto parameters = signature->parameterTypes();
  auto require_count = [&](size_t count) {
    if (parameters.size() != count) {
      diagnostics.reject(unit, owner,
                         "view operation has the wrong operand count");
    }
  };
  auto require_void = [&] {
    if (!unit.typeTraits().is_void(signature->returnType())) {
      diagnostics.reject(unit, owner, "view.store must return void");
    }
  };

  ViewIntrinsic result(*selected);
  if (*selected == Operation::Load || *selected == Operation::Store) {
    result.memory_flags_ =
        read_memory_access_flags(unit, diagnostics, function, 0,
                                 LOOM_MEMORY_ACCESS_FLAG_VOLATILE |
                                     LOOM_MEMORY_ACCESS_FLAG_ROOT_IN_BOUNDS,
                                 owner);
  }
  switch (*selected) {
    case Operation::LayoutDense: {
      require_count(0);
      result.result_encoding_ = &require_layout(unit, diagnostics, types,
                                                signature->returnType(), owner);
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::LayoutStrided: {
      for (size_t index = 0; index < parameters.size(); ++index) {
        if (require_index_value(unit, diagnostics, types, parameters[index],
                                owner, "layout stride")) {
          result.unsigned_argument_mask_ |= uint32_t{1} << index;
        }
      }
      result.result_encoding_ = &require_layout(unit, diagnostics, types,
                                                signature->returnType(), owner);
      if (result.result_encoding_->rank != parameters.size()) {
        diagnostics.reject(unit, owner,
                           "strided layout rank must match its stride count");
      }
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::BufferView: {
      result.result_view_ = &require_view(unit, diagnostics, types,
                                          signature->returnType(), owner);
      auto* pointer =
          cxx::type_cast<cxx::PointerType>(types.unqualified(parameters[0]));
      const BufferPartition* buffer = nullptr;
      if (!pointer) {
        const auto& partition = types.partition(parameters[0], owner);
        if (partition.kind == ValueKind::Buffer) {
          buffer = &static_cast<const BufferPartition&>(partition);
        }
      }
      auto* element = pointer  ? pointer->elementType()
                      : buffer ? buffer->element_type
                               : nullptr;
      if (!element || element != result.result_view_->element_type) {
        diagnostics.reject(
            unit, owner,
            "buffer.view source element type must match its result view");
      }
      result.source_buffer_ = buffer;
      const size_t encoding_count = result.result_view_->encoding ? 1 : 0;
      const size_t base_count = 2 + encoding_count;
      if (parameters.size() != base_count &&
          (!buffer || parameters.size() != base_count + 1)) {
        diagnostics.reject(unit, owner,
                           "buffer.view has the wrong operand count");
      }
      result.has_buffer_byte_offset_ = parameters.size() == base_count + 1;
      if (result.has_buffer_byte_offset_ &&
          !types.is_offset(parameters[1], owner)) {
        diagnostics.reject(
            unit, owner, "buffer.view byte offset must use the offset domain");
      }
      const size_t dimensions_index = result.has_buffer_byte_offset_ ? 2 : 1;
      result.unsigned_dimension_mask_ = require_index_record(
          unit, diagnostics, types, parameters[dimensions_index],
          dynamic_extent_count(*result.result_view_), owner, "view dimensions");
      if (result.result_view_->encoding) {
        const auto& layout = require_encoding(
            unit, diagnostics, types, parameters[dimensions_index + 1], owner);
        if (result.result_view_->encoding != &layout) {
          diagnostics.reject(unit, owner,
                             "buffer.view encoding type must match its result "
                             "view");
        }
      }
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::Subview: {
      require_count(3);
      result.source_view_ =
          &require_view(unit, diagnostics, types, parameters[0], owner);
      result.result_view_ = &require_view(unit, diagnostics, types,
                                          signature->returnType(), owner);
      if (result.source_view_->element_type !=
          result.result_view_->element_type) {
        diagnostics.reject(
            unit, owner, "subview must preserve its source view element type");
      }
      if (result.source_view_->extents.size() !=
              result.result_view_->extents.size() ||
          result.source_view_->encoding != result.result_view_->encoding) {
        diagnostics.reject(
            unit, owner,
            "subview must preserve its source view rank and encoding type");
      }
      if (!result.source_view_->encoding &&
          !plain_subview_preserves_dense_suffix(*result.source_view_,
                                                *result.result_view_)) {
        diagnostics.reject(
            unit, owner,
            "plain subview must preserve every static dense suffix extent; "
            "use an encoded view for a remapped or dynamic suffix");
      }
      result.unsigned_origin_mask_ = require_index_record(
          unit, diagnostics, types, parameters[1],
          result.source_view_->extents.size(), owner, "subview origin");
      result.unsigned_dimension_mask_ =
          require_index_record(unit, diagnostics, types, parameters[2],
                               dynamic_extent_count(*result.result_view_),
                               owner, "subview dimensions");
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::Bitcast: {
      require_count(1);
      result.source_view_ =
          &require_view(unit, diagnostics, types, parameters[0], owner);
      result.result_view_ = &require_view(unit, diagnostics, types,
                                          signature->returnType(), owner);
      if (result.source_view_->encoding || result.result_view_->encoding) {
        diagnostics.reject(unit, owner,
                           "view.bitcast requires plain unencoded views");
      }
      if (result.source_view_->access_flags !=
          result.result_view_->access_flags) {
        diagnostics.reject(unit, owner,
                           "view.bitcast must preserve source access rights");
      }
      const uint64_t source_bits =
          static_view_bit_count(unit, diagnostics, *result.source_view_,
                                "view.bitcast source", owner);
      const uint64_t result_bits =
          static_view_bit_count(unit, diagnostics, *result.result_view_,
                                "view.bitcast result", owner);
      if (source_bits != result_bits) {
        diagnostics.reject(unit, owner,
                           "view.bitcast requires equal total bit counts");
      }
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::Load: {
      if (parameters.empty()) {
        diagnostics.reject(unit, owner,
                           "view operation has the wrong operand count");
      }
      result.source_view_ =
          &require_view(unit, diagnostics, types, parameters[0], owner);
      require_count(result.source_view_->extents.size() + 1);
      for (size_t index = 1; index < parameters.size(); ++index) {
        if (require_index_value(unit, diagnostics, types, parameters[index],
                                owner, "view index")) {
          result.unsigned_argument_mask_ |= uint32_t{1} << index;
        }
      }
      result.scalar_result_ = types.get(signature->returnType(), owner);
      if (loom_type_kind(result.scalar_result_) != LOOM_TYPE_SCALAR ||
          loom_type_element_type(result.scalar_result_) !=
              result.source_view_->element) {
        diagnostics.reject(unit, owner,
                           "view.load result must match its view element");
      }
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::Store: {
      require_void();
      if (parameters.size() < 2) {
        diagnostics.reject(unit, owner,
                           "view operation has the wrong operand count");
      }
      result.source_view_ =
          &require_view(unit, diagnostics, types, parameters[1], owner);
      require_count(result.source_view_->extents.size() + 2);
      if (unit.typeTraits().is_const(result.source_view_->element_type) ||
          types.unqualified(parameters[0]) !=
              types.unqualified(result.source_view_->element_type) ||
          loom_type_element_type(types.get(parameters[0], owner)) !=
              result.source_view_->element) {
        diagnostics.reject(
            unit, owner,
            "view.store value must match a mutable view element type");
      }
      for (size_t index = 2; index < parameters.size(); ++index) {
        if (require_index_value(unit, diagnostics, types, parameters[index],
                                owner, "view index")) {
          result.unsigned_argument_mask_ |= uint32_t{1} << index;
        }
      }
      return result;
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown view intrinsic operation");
  IREE_BUILTIN_UNREACHABLE();
}

std::optional<Value> ViewIntrinsic::call(std::span<const Value> arguments,
                                         Types& types, ValueArena& arena,
                                         Storage& storage, cxx::AST* owner,
                                         loom_builder_t* builder,
                                         loom_location_id_t location) const {
  loom_op_t* op;
  switch (operation_) {
    case Operation::LayoutDense:
      check(loom_encoding_layout_dense_build(
          builder,
          loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
          location, &op));
      return arena.capture(*result_encoding_, {loom_op_results(op), 1});
    case Operation::LayoutStrided: {
      std::vector<loom_value_id_t> strides;
      strides.reserve(arguments.size());
      for (size_t index = 0; index < arguments.size(); ++index) {
        strides.push_back(
            cast_index(arguments[index],
                       is_unsigned_component(unsigned_argument_mask_, index),
                       builder, location));
      }
      std::vector<int64_t> static_strides(arguments.size(),
                                          std::numeric_limits<int64_t>::min());
      check(loom_encoding_layout_strided_build(
          builder, strides.data(), strides.size(), static_strides.data(),
          static_strides.size(),
          loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
          location, &op));
      return arena.capture(*result_encoding_, {loom_op_results(op), 1});
    }
    case Operation::BufferView:
    case Operation::Subview: {
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 2> components;
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> offsets;
      size_t component_count = 0;
      auto dimensions = arguments[operation_ == Operation::BufferView
                                      ? (has_buffer_byte_offset_ ? 2 : 1)
                                      : 2]
                            .components();
      for (size_t index = 0; index < dimensions.size(); ++index) {
        components[component_count++] =
            cast_index(Value(dimensions[index]),
                       is_unsigned_component(unsigned_dimension_mask_, index),
                       builder, location);
      }
      if (operation_ == Operation::Subview) {
        auto origin = arguments[1].components();
        for (size_t axis = 0; axis < result_view_->extents.size(); ++axis) {
          offsets[axis] =
              cast_index(Value(origin[axis]),
                         is_unsigned_component(unsigned_origin_mask_, axis),
                         builder, location);
        }
      }

      std::optional<Pointer> pointer;
      if (operation_ == Operation::BufferView) {
        pointer = source_buffer_
                      ? has_buffer_byte_offset_
                            ? Pointer{arguments[0].ssa(), arguments[1].ssa()}
                            : storage.root(arguments[0].ssa(), owner)
                      : storage.constrain_origin(arguments[0].pointer(), owner);
      }

      const bool capture_physical_encoding =
          operation_ == Operation::Subview &&
          !subview_preserves_dense_suffix(
              arguments[0], *source_view_, *result_view_,
              std::span<const loom_value_id_t>(components)
                  .first(component_count));
      IREE_ASSERT(!capture_physical_encoding || result_view_->encoding);
      std::array<loom_value_id_t, 2> result_ids;
      const size_t result_count = capture_physical_encoding ? 2 : 1;
      check(loom_builder_reserve_values(builder, result_count,
                                        result_ids.data()));
      if (result_view_->encoding) {
        components[component_count++] =
            capture_physical_encoding ? result_ids[1]
            : operation_ == Operation::BufferView
                ? arguments[has_buffer_byte_offset_ ? 3 : 2].components()[0]
                : encoding_component(arguments[0], *source_view_);
      }
      components[component_count++] = result_ids[0];
      std::vector<loom_type_t> result_types;
      BoundTypeStorage type_storage;
      types.append_bound(result_source_type_, owner,
                         {components.data(), component_count}, type_storage,
                         result_types);
      auto result_type = result_types.back();
      if (operation_ == Operation::BufferView) {
        check(loom_buffer_view_build(builder, pointer->root,
                                     pointer->byte_offset, result_type,
                                     location, &op));
      } else {
        std::array<int64_t, LOOM_TYPE_MAX_RANK> static_offsets;
        static_offsets.fill(std::numeric_limits<int64_t>::min());
        std::array<loom_type_t, 2> subview_result_types = {result_type, {}};
        if (capture_physical_encoding) {
          subview_result_types[1] = result_types[result_types.size() - 2];
        }
        check(loom_view_subview_build(
            builder, view_component(arguments[0]), offsets.data(),
            result_view_->extents.size(), static_offsets.data(),
            result_view_->extents.size(), subview_result_types.data(),
            result_count, location, &op));
      }
      return arena.capture(*result_view_, {components.data(), component_count});
    }
    case Operation::Bitcast: {
      std::vector<loom_type_t> result_types;
      types.append(result_source_type_, owner, result_types);
      check(loom_view_bitcast_build(builder, view_component(arguments[0]),
                                    result_types.back(), location, &op));
      return arena.capture(*result_view_, {loom_op_results(op), 1});
    }
    case Operation::Load: {
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
      std::array<int64_t, LOOM_TYPE_MAX_RANK> static_indices;
      static_indices.fill(std::numeric_limits<int64_t>::min());
      for (size_t axis = 0; axis < source_view_->extents.size(); ++axis) {
        size_t argument_index = axis + 1;
        indices[axis] = cast_index(
            arguments[argument_index],
            is_unsigned_component(unsigned_argument_mask_, argument_index),
            builder, location);
      }
      check(loom_view_load_build(
          builder, 0, source_view_->access_flags | memory_flags_,
          view_component(arguments[0]), indices.data(),
          source_view_->extents.size(), static_indices.data(),
          source_view_->extents.size(), 0, 0, scalar_result_, location, &op));
      return Value(loom_op_results(op)[0]);
    }
    case Operation::Store: {
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
      std::array<int64_t, LOOM_TYPE_MAX_RANK> static_indices;
      static_indices.fill(std::numeric_limits<int64_t>::min());
      for (size_t axis = 0; axis < source_view_->extents.size(); ++axis) {
        size_t argument_index = axis + 2;
        indices[axis] = cast_index(
            arguments[argument_index],
            is_unsigned_component(unsigned_argument_mask_, argument_index),
            builder, location);
      }
      check(loom_view_store_build(
          builder, 0, source_view_->access_flags | memory_flags_,
          arguments[0].ssa(), view_component(arguments[1]), indices.data(),
          source_view_->extents.size(), static_indices.data(),
          source_view_->extents.size(), 0, 0, location, &op));
      return std::nullopt;
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown view intrinsic operation");
  IREE_BUILTIN_UNREACHABLE();
}

bool ViewIntrinsic::equivalent(const ViewIntrinsic& other) const {
  return operation_ == other.operation_ &&
         result_source_type_ == other.result_source_type_ &&
         result_encoding_ == other.result_encoding_ &&
         source_buffer_ == other.source_buffer_ &&
         has_buffer_byte_offset_ == other.has_buffer_byte_offset_ &&
         source_view_ == other.source_view_ &&
         result_view_ == other.result_view_ &&
         unsigned_argument_mask_ == other.unsigned_argument_mask_ &&
         unsigned_dimension_mask_ == other.unsigned_dimension_mask_ &&
         unsigned_origin_mask_ == other.unsigned_origin_mask_ &&
         memory_flags_ == other.memory_flags_ &&
         loom_type_equal(scalar_result_, other.scalar_result_);
}

}  // namespace loom::cxx_import
