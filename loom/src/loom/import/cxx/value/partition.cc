// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/partition.h"

#include <algorithm>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/ir/module.h"
#include "loom/ops/type_builders.h"

namespace loom::cxx_import {
namespace {

loom_type_t make_shaped_type(loom_type_kind_t kind, loom_scalar_type_t element,
                             std::span<const uint64_t> dimensions,
                             BoundTypeStorage& storage) {
  const loom_type_t base = {
      .header = loom_type_make_header(kind, element, 0, 0),
  };
  return storage.with_dimensions(base, dimensions);
}

loom_type_t make_channel_handle_type(loom_module_t* module,
                                     loom_type_t payload_type,
                                     ChannelHandleKind handle_kind,
                                     bool mutable_read) {
  loom_type_id_t payload_id = LOOM_TYPE_ID_INVALID;
  check(loom_module_intern_type_id(module, payload_type, &payload_id));
  loom_type_t result;
  switch (handle_kind) {
    case ChannelHandleKind::Channel:
      check(loom_channel_type_make(module, payload_id, &result));
      break;
    case ChannelHandleKind::Read:
      check(loom_read_type_make(
          module, mutable_read ? LOOM_READ_TYPE_BUILD_FLAG_HAS_MODE : 0,
          payload_id,
          mutable_read ? LOOM_READ_TYPE_MODE_MUTABLE
                       : static_cast<loom_read_type_mode_t>(0),
          &result));
      break;
    case ChannelHandleKind::Write:
      check(loom_write_type_make(module, payload_id, &result));
      break;
  }
  return result;
}

// Source admission has already checked the rank and the number of dynamic
// extents. Their SSA identities precede an optional encoding or payload.
std::span<const uint64_t> bind_dimensions(
    const ShapeExtents& extents, std::span<const loom_value_id_t> dynamic,
    std::array<uint64_t, LOOM_TYPE_MAX_RANK>& dimensions) {
  size_t operand = 0;
  for (size_t axis = 0; axis < extents.size(); ++axis) {
    dimensions[axis] = extents[axis] < 0
                           ? loom_dim_pack_dynamic(dynamic[operand++])
                           : loom_dim_pack_static(extents[axis]);
  }
  return std::span<const uint64_t>(dimensions).first(extents.size());
}

loom_type_t make_view_type(const loom_type_t& element,
                           std::span<const uint64_t> dimensions,
                           std::optional<loom_value_id_t> encoding,
                           BoundTypeStorage& storage) {
  loom_type_t type =
      loom_type_is_reference(element)
          ? loom_type_reference_view(&element, dimensions.size(),
                                     storage.retain_dimensions(dimensions), 0,
                                     0)
          : storage.with_dimensions(
                {
                    .header = loom_type_make_header(
                        LOOM_TYPE_VIEW, loom_type_element_type(element), 0, 0),
                },
                dimensions);
  return BoundTypeStorage::with_encoding(type, encoding);
}

}  // namespace

loom_type_t BoundTypeStorage::with_encoding(
    loom_type_t result, std::optional<loom_value_id_t> encoding) {
  if (encoding) {
    if (*encoding > UINT16_MAX) {
      check(iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "encoding value ID %u exceeds the shaped type limit of %u", *encoding,
          UINT16_MAX));
    }
    result.encoding_id = static_cast<uint16_t>(*encoding);
    result.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  }
  return result;
}

loom_type_t BoundTypeStorage::with_dimensions(
    loom_type_t type, std::span<const uint64_t> dimensions) {
  bool all_static = true;
  for (auto dimension : dimensions) {
    all_static &= !loom_dim_is_dynamic(dimension);
  }
  uint8_t flags = all_static ? LOOM_TYPE_FLAG_ALL_STATIC : 0;
  const bool reference_element = loom_type_has_reference_element(type);
  if (reference_element) {
    flags |= LOOM_TYPE_FLAG_REFERENCE_ELEMENT;
  } else if (dimensions.size() <= 2) {
    flags |= LOOM_TYPE_FLAG_INLINE_DIMS;
  }
  type.header =
      loom_type_make_header(loom_type_kind(type), loom_type_element_type(type),
                            static_cast<uint8_t>(dimensions.size()), flags) |
      (type.header & 0xFF000000u);
  if (!reference_element) {
    type.dims[1] = 0;
  }
  type.dims[0] = 0;
  if (!reference_element && dimensions.size() <= 2) {
    std::copy(dimensions.begin(), dimensions.end(), type.dims);
  } else {
    type.dims[0] = reinterpret_cast<uintptr_t>(retain_dimensions(dimensions));
  }
  return type;
}

const loom_overflow_dim_t* BoundTypeStorage::retain_dimensions(
    std::span<const loom_overflow_dim_t> dimensions) {
  IREE_ASSERT(!dimensions.empty() && dimensions.size() <= LOOM_TYPE_MAX_RANK);
  auto& payload = dimension_payloads_.emplace_front();
  std::copy(dimensions.begin(), dimensions.end(), payload.begin());
  return payload.data();
}

void ShapeExtents::assign(std::span<const int64_t> extents) {
  IREE_ASSERT(!extents.empty() && extents.size() <= LOOM_TYPE_MAX_RANK);
  rank_ = extents.size();
  if (rank_ <= inline_extents_.size()) {
    std::copy(extents.begin(), extents.end(), inline_extents_.begin());
    overflow_extents_.reset();
    return;
  }
  overflow_extents_ = std::make_unique<int64_t[]>(rank_);
  std::copy(extents.begin(), extents.end(), overflow_extents_.get());
}

void TilePartition::bind_static_type() {
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  static_type = make_shaped_type(LOOM_TYPE_TILE, element,
                                 bind_dimensions(extents, {}, dimensions),
                                 static_type_storage);
}

loom_type_t TilePartition::bind_type(
    std::span<const loom_value_id_t> dependencies,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(dependencies.size() == component_count - 1);
  if (!requires_binding()) {
    return static_type;
  }
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  return BoundTypeStorage::with_encoding(
      make_shaped_type(LOOM_TYPE_TILE, element,
                       bind_dimensions(extents, dependencies, dimensions),
                       storage),
      encoding ? std::optional{dependencies.back()} : std::nullopt);
}

loom_value_id_t TilePartition::tile_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

void ChannelPartition::bind_static_type(loom_module_t* module) {
  static_type = make_channel_handle_type(module, payload->static_type,
                                         handle_kind, mutable_read);
}

loom_type_t ChannelPartition::bind_type(
    loom_module_t* module, std::span<const loom_value_id_t> dependencies,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(module);
  IREE_ASSERT(dependencies.size() == component_count - 1);
  if (!requires_binding()) {
    return static_type;
  }
  const auto payload_type = payload->bind_type(dependencies, storage);
  return make_channel_handle_type(module, payload_type, handle_kind,
                                  mutable_read);
}

loom_value_id_t ChannelPartition::handle_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

void VectorPartition::bind_static_type() {
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  static_type = make_shaped_type(LOOM_TYPE_VECTOR, element,
                                 bind_dimensions(extents, {}, dimensions),
                                 static_type_storage);
}

loom_type_t VectorPartition::bind_type(
    std::span<const loom_value_id_t> dynamic_extents,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(dynamic_extents.size() == component_count - 1);
  if (!is_dynamic()) {
    return static_type;
  }
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  return make_shaped_type(LOOM_TYPE_VECTOR, element,
                          bind_dimensions(extents, dynamic_extents, dimensions),
                          storage);
}

loom_value_id_t VectorPartition::vector_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

loom_type_t TensorPartition::bind_type(
    std::span<const loom_value_id_t> dynamic_extents) const {
  IREE_ASSERT(dynamic_extents.size() == (is_dynamic() ? 1u : 0u));
  auto dimension = is_dynamic() ? loom_dim_pack_dynamic(dynamic_extents[0])
                                : loom_dim_pack_static(extent);
  if (vector_lane_count) {
    return loom_type_shaped_2d(LOOM_TYPE_TENSOR, element, dimension,
                               loom_dim_pack_static(vector_lane_count), 0);
  }
  return loom_type_shaped_1d(LOOM_TYPE_TENSOR, element, dimension, 0);
}

loom_value_id_t TensorPartition::tensor_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

void ViewPartition::bind_static_type() {
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  const auto shape = bind_dimensions(extents, {}, dimensions);
  if (columns.empty()) {
    static_type = make_view_type(column_element(0), shape, std::nullopt,
                                 static_type_storage);
  } else {
    for (auto& column : columns) {
      column.static_type = make_view_type(column.element, shape, std::nullopt,
                                          static_type_storage);
    }
  }
}

loom_type_t ViewPartition::bind_type(
    std::span<const loom_value_id_t> dependencies, BoundTypeStorage& storage,
    size_t column) const {
  if (!requires_binding()) {
    return columns.empty() ? static_type : columns[column].static_type;
  }
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  return make_view_type(
      column_element(column),
      bind_dimensions(extents, dependencies, dimensions),
      encoding ? std::optional{dependencies[dynamic_count + column]}
               : std::nullopt,
      storage);
}

loom_value_id_t ViewPartition::view_value(
    std::span<const loom_value_id_t> components, size_t column) const {
  IREE_ASSERT(components.size() == component_count);
  return components[column_component(column)];
}

const cxx::Type* SubobjectPartition::type() const {
  return source_object_type(source);
}

uint64_t SubobjectPartition::byte_offset() const {
  return source_subobject_offset(source);
}

}  // namespace loom::cxx_import
