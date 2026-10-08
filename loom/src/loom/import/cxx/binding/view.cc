// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/view.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "iree/base/api.h"
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
static_assert(LOOM_TYPE_MAX_RANK <= std::numeric_limits<uint16_t>::digits);

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

bool require_integral(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                      Types& types, const cxx::Type* type, cxx::AST* owner,
                      std::string_view role) {
  auto projected = types.get(type, owner);
  if (!unit.typeTraits().is_integral(type) ||
      loom_type_kind(projected) != LOOM_TYPE_SCALAR ||
      loom_type_element_type(projected) == LOOM_SCALAR_TYPE_I1) {
    diagnostics.reject(unit, owner,
                       std::string(role) + " requires an integer value");
  }
  return types.is_unsigned(type);
}

void require_integral_components(cxx::TranslationUnit& unit,
                                 Diagnostics& diagnostics, Types& types,
                                 const cxx::Type* type,
                                 const Partition& partition, cxx::AST* owner,
                                 std::string_view role,
                                 uint16_t& unsigned_component_mask,
                                 size_t& component_index) {
  switch (partition.kind) {
    case ValueKind::SSA: {
      if (require_integral(unit, diagnostics, types, type, owner, role)) {
        unsigned_component_mask = static_cast<uint16_t>(
            unsigned_component_mask | (uint16_t{1} << component_index));
      }
      ++component_index;
      return;
    }
    case ValueKind::Record: {
      const auto& record = static_cast<const RecordPartition&>(partition);
      for (const auto& member : record.members) {
        require_integral_components(
            unit, diagnostics, types, member.field->type(), *member.partition,
            owner, role, unsigned_component_mask, component_index);
      }
      return;
    }
    case ValueKind::Array: {
      const auto& array = static_cast<const ArrayPartition&>(partition);
      for (size_t index = 0; index < array.source->size(); ++index) {
        require_integral_components(unit, diagnostics, types,
                                    array.source->elementType(), *array.element,
                                    owner, role, unsigned_component_mask,
                                    component_index);
      }
      return;
    }
    default:
      diagnostics.reject(unit, owner,
                         std::string(role) + " requires integer fields");
  }
}

uint16_t require_integral_record(cxx::TranslationUnit& unit,
                                 Diagnostics& diagnostics, Types& types,
                                 const cxx::Type* type, size_t count,
                                 cxx::AST* owner, std::string_view role) {
  auto* record = types.record(type, owner);
  if (!record || record->component_count != count) {
    diagnostics.reject(unit, owner, std::string(role) + " has the wrong arity");
  }
  uint16_t unsigned_component_mask = 0;
  size_t component_index = 0;
  require_integral_components(unit, diagnostics, types, type, *record, owner,
                              role, unsigned_component_mask, component_index);
  return unsigned_component_mask;
}

bool is_unsigned_component(uint32_t mask, size_t index) {
  return (mask & (uint32_t{1} << index)) != 0;
}

loom_value_id_t cast_index(Value value, bool unsigned_source,
                           loom_builder_t* builder,
                           loom_location_id_t location) {
  auto input = value.ssa();
  auto input_type = loom_module_value_type(builder->module, input);
  auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(input_type, index_type)) {
    return input;
  }
  loom_op_t* op;
  auto i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  if (unsigned_source && !loom_type_equal(input_type, i64_type)) {
    check(loom_scalar_extui_build(builder, input, input_type, i64_type,
                                  location, &op));
    input = loom_op_results(op)[0];
    input_type = i64_type;
  }
  check(loom_index_cast_build(builder, input, input_type, index_type, location,
                              &op));
  return loom_op_results(op)[0];
}

size_t dynamic_extent_count(const ViewPartition& view) {
  return view.component_count - 2;
}

loom_value_id_t view_component(Value value) {
  return value.components().back();
}

loom_value_id_t encoding_component(Value value) {
  auto components = value.components();
  return components[components.size() - 2];
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
    const cxx::FunctionType* signature, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  auto selected = parse_operation(attribute.arguments[0]->name());
  if (!selected) {
    return std::nullopt;
  }
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
    if (signature->returnType()->kind() != cxx::TypeKind::kVoid) {
      diagnostics.reject(unit, owner, "view.store must return void");
    }
  };

  ViewIntrinsic result(*selected);
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
        if (require_integral(unit, diagnostics, types, parameters[index], owner,
                             "layout stride")) {
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
      require_count(3);
      result.result_view_ = &require_view(unit, diagnostics, types,
                                          signature->returnType(), owner);
      auto* pointer =
          cxx::type_cast<cxx::PointerType>(types.unqualified(parameters[0]));
      if (!pointer ||
          pointer->elementType() != result.result_view_->element_type) {
        diagnostics.reject(
            unit, owner,
            "buffer.view pointer element type must match its result view");
      }
      result.unsigned_dimension_mask_ = require_integral_record(
          unit, diagnostics, types, parameters[1],
          dynamic_extent_count(*result.result_view_), owner, "view dimensions");
      const auto& layout =
          require_encoding(unit, diagnostics, types, parameters[2], owner);
      if (layout.rank != result.result_view_->extents.size() ||
          layout.role != result.result_view_->encoding_role) {
        diagnostics.reject(unit, owner,
                           "buffer.view encoding rank and role must match its "
                           "result view");
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
          result.source_view_->encoding_role !=
              result.result_view_->encoding_role) {
        diagnostics.reject(
            unit, owner,
            "subview must preserve its source view rank and encoding role");
      }
      result.unsigned_origin_mask_ = require_integral_record(
          unit, diagnostics, types, parameters[1],
          result.source_view_->extents.size(), owner, "subview origin");
      result.unsigned_dimension_mask_ =
          require_integral_record(unit, diagnostics, types, parameters[2],
                                  dynamic_extent_count(*result.result_view_),
                                  owner, "subview dimensions");
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
        if (require_integral(unit, diagnostics, types, parameters[index], owner,
                             "view index")) {
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
        if (require_integral(unit, diagnostics, types, parameters[index], owner,
                             "view index")) {
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
      auto dimensions =
          arguments[operation_ == Operation::BufferView ? 1 : 2].components();
      for (size_t index = 0; index < dimensions.size(); ++index) {
        components[component_count++] =
            cast_index(Value(dimensions[index]),
                       is_unsigned_component(unsigned_dimension_mask_, index),
                       builder, location);
      }
      components[component_count++] = operation_ == Operation::BufferView
                                          ? arguments[2].components()[0]
                                          : encoding_component(arguments[0]);
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
        pointer = storage.constrain_origin(arguments[0].pointer(), owner);
      }

      loom_value_id_t result_id;
      check(loom_builder_reserve_values(builder, 1, &result_id));
      components[component_count++] = result_id;
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
        check(loom_view_subview_build(
            builder, view_component(arguments[0]), offsets.data(),
            result_view_->extents.size(), static_offsets.data(),
            result_view_->extents.size(), result_type, location, &op));
      }
      return arena.capture(*result_view_, {components.data(), component_count});
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
          builder, 0, source_view_->access_flags, view_component(arguments[0]),
          indices.data(), source_view_->extents.size(), static_indices.data(),
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
          builder, 0, source_view_->access_flags, arguments[0].ssa(),
          view_component(arguments[1]), indices.data(),
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
         source_view_ == other.source_view_ &&
         result_view_ == other.result_view_ &&
         unsigned_argument_mask_ == other.unsigned_argument_mask_ &&
         unsigned_dimension_mask_ == other.unsigned_dimension_mask_ &&
         unsigned_origin_mask_ == other.unsigned_origin_mask_ &&
         loom_type_equal(scalar_result_, other.scalar_result_);
}

}  // namespace loom::cxx_import
