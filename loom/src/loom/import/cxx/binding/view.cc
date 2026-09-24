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
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/storage.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {

const EncodingPartition& require_encoding(cxx::TranslationUnit& unit,
                                          Diagnostics& diagnostics,
                                          Types& types, const cxx::Type* type,
                                          cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::Encoding) {
    diagnostics.reject(unit, owner,
                       "encoding operation requires an encoding value");
  }
  return static_cast<const EncodingPartition&>(partition);
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

void require_integral(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                      Types& types, const cxx::Type* type, cxx::AST* owner,
                      std::string_view role) {
  auto projected = types.get(type, owner);
  if (!unit.typeTraits().is_integral(type) ||
      loom_type_kind(projected) != LOOM_TYPE_SCALAR ||
      loom_type_element_type(projected) == LOOM_SCALAR_TYPE_I1) {
    diagnostics.reject(unit, owner,
                       std::string(role) + " requires an integer value");
  }
}

const RecordPartition& require_record(cxx::TranslationUnit& unit,
                                      Diagnostics& diagnostics, Types& types,
                                      const cxx::Type* type, size_t count,
                                      cxx::AST* owner, std::string_view role) {
  auto* record = types.record(type, owner);
  if (!record || record->members.size() != count ||
      record->component_count != count) {
    diagnostics.reject(unit, owner, std::string(role) + " has the wrong arity");
  }
  for (const auto& member : record->members) {
    if (member.partition->kind != ValueKind::SSA) {
      diagnostics.reject(unit, owner,
                         std::string(role) + " requires integer fields");
    }
    require_integral(unit, diagnostics, types, member.field->type(), owner,
                     role);
  }
  return *record;
}

loom_value_id_t cast_index(Value value, loom_builder_t* builder,
                           loom_location_id_t location) {
  auto input = value.ssa();
  auto input_type = loom_module_value_type(builder->module, input);
  auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(input_type, index_type)) {
    return input;
  }
  loom_op_t* op;
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

loom_value_id_t layout_component(Value value) {
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
      result.result_encoding_ = &require_encoding(
          unit, diagnostics, types, signature->returnType(), owner);
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::LayoutStrided: {
      for (const auto* parameter : parameters) {
        require_integral(unit, diagnostics, types, parameter, owner,
                         "layout stride");
      }
      result.result_encoding_ = &require_encoding(
          unit, diagnostics, types, signature->returnType(), owner);
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
      require_record(unit, diagnostics, types, parameters[1],
                     dynamic_extent_count(*result.result_view_), owner,
                     "view dimensions");
      const auto& layout =
          require_encoding(unit, diagnostics, types, parameters[2], owner);
      if (layout.rank != 2) {
        diagnostics.reject(unit, owner,
                           "rank-two views require a rank-two layout");
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
      require_record(unit, diagnostics, types, parameters[1], 2, owner,
                     "subview origin");
      require_record(unit, diagnostics, types, parameters[2],
                     dynamic_extent_count(*result.result_view_), owner,
                     "subview dimensions");
      result.result_source_type_ = signature->returnType();
      return result;
    }
    case Operation::Load: {
      require_count(3);
      result.source_view_ =
          &require_view(unit, diagnostics, types, parameters[0], owner);
      require_integral(unit, diagnostics, types, parameters[1], owner,
                       "view index");
      require_integral(unit, diagnostics, types, parameters[2], owner,
                       "view index");
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
      require_count(4);
      require_void();
      result.source_view_ =
          &require_view(unit, diagnostics, types, parameters[1], owner);
      if (unit.typeTraits().is_const(result.source_view_->element_type) ||
          types.unqualified(parameters[0]) !=
              types.unqualified(result.source_view_->element_type) ||
          loom_type_element_type(types.get(parameters[0], owner)) !=
              result.source_view_->element) {
        diagnostics.reject(
            unit, owner,
            "view.store value must match a mutable view element type");
      }
      require_integral(unit, diagnostics, types, parameters[2], owner,
                       "view index");
      require_integral(unit, diagnostics, types, parameters[3], owner,
                       "view index");
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
      for (auto argument : arguments) {
        strides.push_back(cast_index(argument, builder, location));
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
      std::array<loom_value_id_t, 4> components;
      std::array<loom_value_id_t, 2> offsets;
      size_t component_count = 0;
      auto dimensions =
          arguments[operation_ == Operation::BufferView ? 1 : 2].components();
      for (auto dimension : dimensions) {
        components[component_count++] =
            cast_index(Value(dimension), builder, location);
      }
      components[component_count++] = operation_ == Operation::BufferView
                                          ? arguments[2].components()[0]
                                          : layout_component(arguments[0]);
      if (operation_ == Operation::Subview) {
        auto origin = arguments[1].components();
        for (size_t axis = 0; axis < offsets.size(); ++axis) {
          offsets[axis] = cast_index(Value(origin[axis]), builder, location);
        }
      }

      std::optional<Pointer> pointer;
      if (operation_ == Operation::BufferView) {
        pointer = storage.constrain_origin(arguments[0].pointer(), owner);
      }

      loom_value_id_t result_id;
      check(loom_builder_reserve_results(builder, 1, &result_id));
      components[component_count++] = result_id;
      std::vector<loom_type_t> result_types;
      types.append_bound(result_source_type_, owner,
                         {components.data(), component_count}, result_types);
      auto result_type = result_types.back();
      if (operation_ == Operation::BufferView) {
        check(loom_buffer_view_build(builder, pointer->root,
                                     pointer->byte_offset, result_type,
                                     location, &op));
      } else {
        const int64_t static_offsets[2] = {
            std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::min(),
        };
        check(loom_view_subview_build(builder, view_component(arguments[0]),
                                      offsets.data(), offsets.size(),
                                      static_offsets, std::size(static_offsets),
                                      result_type, location, &op));
      }
      return arena.capture(*result_view_, {components.data(), component_count});
    }
    case Operation::Load: {
      std::array<loom_value_id_t, 2> indices = {
          cast_index(arguments[1], builder, location),
          cast_index(arguments[2], builder, location),
      };
      const int64_t static_indices[2] = {
          std::numeric_limits<int64_t>::min(),
          std::numeric_limits<int64_t>::min(),
      };
      check(loom_view_load_build(
          builder, 0, source_view_->access_flags, view_component(arguments[0]),
          indices.data(), indices.size(), static_indices,
          std::size(static_indices), 0, 0, scalar_result_, location, &op));
      return Value(loom_op_results(op)[0]);
    }
    case Operation::Store: {
      std::array<loom_value_id_t, 2> indices = {
          cast_index(arguments[2], builder, location),
          cast_index(arguments[3], builder, location),
      };
      const int64_t static_indices[2] = {
          std::numeric_limits<int64_t>::min(),
          std::numeric_limits<int64_t>::min(),
      };
      check(loom_view_store_build(
          builder, 0, source_view_->access_flags, arguments[0].ssa(),
          view_component(arguments[1]), indices.data(), indices.size(),
          static_indices, std::size(static_indices), 0, 0, location, &op));
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
         loom_type_equal(scalar_result_, other.scalar_result_);
}

}  // namespace loom::cxx_import
