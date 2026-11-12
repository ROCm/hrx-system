// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/template_inputs.h"

#include <cxx/types.h>

#include <array>
#include <optional>

#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/tensor/ops.h"
#include "loom/ops/tile/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {

struct InputShape {
  // Native carrier kind; nominal aggregates remain outside shape matching.
  loom_type_kind_t kind;
  // Source element identity, including view access qualifiers.
  const cxx::Type* element;
  // Canonical rank-generic extents, or null for a tensor/native vector.
  const ShapeExtents* extents = nullptr;
  // Rank-one or rank-two extents for tensor and native vector carriers.
  std::array<int64_t, 2> inline_extents = {};
  // Logical native rank, including a tensor's vector-element lane axis.
  size_t rank = 0;
  // Complete encoding role, absent for native dense representation.
  const EncodingPartition* encoding = nullptr;

  int64_t extent(size_t axis) const {
    return extents ? (*extents)[axis] : inline_extents[axis];
  }
};

std::optional<InputShape> input_shape(Types& types, const cxx::Type* source,
                                      cxx::AST* owner) {
  const auto& partition = types.partition(source, owner);
  switch (partition.kind) {
    case ValueKind::Tile: {
      const auto& tile = static_cast<const TilePartition&>(partition);
      return InputShape{
          .kind = LOOM_TYPE_TILE,
          .element = tile.element_type,
          .extents = &tile.extents,
          .rank = tile.extents.size(),
          .encoding = tile.encoding,
      };
    }
    case ValueKind::View: {
      const auto& view = static_cast<const ViewPartition&>(partition);
      return InputShape{
          .kind = LOOM_TYPE_VIEW,
          .element = view.element_type,
          .extents = &view.extents,
          .rank = view.extents.size(),
          .encoding = view.encoding,
      };
    }
    case ValueKind::Vector: {
      const auto& vector = static_cast<const VectorPartition&>(partition);
      return InputShape{
          .kind = LOOM_TYPE_VECTOR,
          .element = vector.element_type,
          .extents = &vector.extents,
          .rank = vector.extents.size(),
      };
    }
    case ValueKind::Tensor: {
      const auto& tensor = static_cast<const TensorPartition&>(partition);
      return InputShape{
          .kind = LOOM_TYPE_TENSOR,
          .element = tensor.element_type,
          .inline_extents = {tensor.extent,
                             static_cast<int64_t>(tensor.vector_lane_count)},
          .rank = tensor.vector_lane_count ? 2u : 1u,
      };
    }
    case ValueKind::SSA:
      if (auto* vector = types.native_vector(source)) {
        return InputShape{
            .kind = LOOM_TYPE_VECTOR,
            .element = vector->elementType(),
            .inline_extents = {static_cast<int64_t>(vector->elementCount()), 0},
            .rank = 1,
        };
      }
      break;
    default:
      break;
  }
  return std::nullopt;
}

// Source admission already proved carrier, rank, element, and encoding-role
// compatibility. Only the destination's dynamic components need transport.
void append_dependencies(const InputShape& destination, loom_type_t actual,
                         loom_builder_t* builder, loom_location_id_t location,
                         std::vector<loom_value_id_t>& operands) {
  for (size_t axis = 0; axis < destination.rank; ++axis) {
    if (destination.extent(axis) >= 0) {
      continue;
    }
    if (loom_type_dim_is_dynamic_at(actual, axis)) {
      operands.push_back(loom_type_dim_value_id_at(actual, axis));
    } else {
      loom_op_t* constant;
      check(loom_index_constant_build(
          builder, loom_attr_i64(loom_type_dim_static_size_at(actual, axis)),
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &constant));
      operands.push_back(loom_op_results(constant)[0]);
    }
  }
  if (destination.encoding) {
    if (loom_type_has_ssa_encoding(actual)) {
      operands.push_back(loom_type_encoding_value_id(actual));
    } else {
      loom_op_t* dense;
      check(loom_encoding_layout_dense_build(
          builder,
          loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
          location, &dense));
      operands.push_back(loom_op_results(dense)[0]);
    }
  }
}

loom_value_id_t refine_input(loom_value_id_t value, loom_type_t type,
                             loom_builder_t* builder,
                             loom_location_id_t location) {
  if (loom_type_equal(type, loom_module_value_type(builder->module, value))) {
    return value;
  }
  loom_op_t* refined;
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_VIEW:
      check(loom_view_refine_build(builder, value, type, location, &refined));
      break;
    case LOOM_TYPE_VECTOR:
      check(loom_vector_refine_build(builder, value, type, location, &refined));
      break;
    case LOOM_TYPE_TILE:
      check(loom_tile_refine_build(builder, value, type, location, &refined));
      break;
    case LOOM_TYPE_TENSOR:
      check(loom_tensor_refine_build(builder, value, type, location, &refined));
      break;
    default:
      IREE_ASSERT_UNREACHABLE("admitted template input carrier");
      IREE_BUILTIN_UNREACHABLE();
  }
  return loom_op_results(refined)[0];
}

}  // namespace

bool template_input_compatible(Types& types, const cxx::Type* family,
                               const cxx::Type* source, cxx::AST* owner) {
  if (types.unqualified(family) == types.unqualified(source)) {
    return true;
  }
  const auto required = input_shape(types, family, owner);
  const auto actual = input_shape(types, source, owner);
  if (!required || !actual || required->kind != actual->kind ||
      required->element != actual->element || required->rank != actual->rank) {
    return false;
  }
  if (required->encoding && actual->encoding) {
    return required->encoding->role == actual->encoding->role;
  }
  const auto* encoding =
      required->encoding ? required->encoding : actual->encoding;
  return !encoding || encoding->role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
}

void append_template_input(Value value, const cxx::Type* family, Types& types,
                           cxx::AST* owner, loom_builder_t* builder,
                           loom_location_id_t location,
                           std::vector<loom_value_id_t>& operands) {
  if (&value.partition() == &types.partition(family, owner)) {
    value.append_to(operands);
    return;
  }
  const auto shape = input_shape(types, family, owner);
  const auto payload = value.components().back();
  append_dependencies(*shape, loom_module_value_type(builder->module, payload),
                      builder, location, operands);
  operands.push_back(payload);
}

Value bind_template_input(Value incoming, const cxx::Type* source, Types& types,
                          ValueArena& arena, cxx::AST* owner,
                          loom_builder_t* builder,
                          loom_location_id_t location) {
  const auto& partition = types.partition(source, owner);
  if (&incoming.partition() == &partition) {
    return incoming;
  }
  const auto shape = input_shape(types, source, owner);
  const auto payload = incoming.components().back();
  std::vector<loom_value_id_t> components;
  components.reserve(partition.component_count);
  append_dependencies(*shape, loom_module_value_type(builder->module, payload),
                      builder, location, components);
  components.push_back(payload);
  std::vector<loom_type_t> bound_types;
  BoundTypeStorage storage;
  types.append_bound(source, owner, components, storage, bound_types);
  components.back() =
      refine_input(payload, bound_types.back(), builder, location);
  return arena.capture(partition, components);
}

BoundTemplateSignature bind_template_signature(
    Types& types, const cxx::FunctionType* family,
    const cxx::FunctionType* provider, cxx::AST* owner, loom_builder_t* builder,
    bool has_predicates) {
  BoundTemplateSignature result;
  auto parameters = family->parameterTypes();
  std::vector<const cxx::Type*> sources(parameters.begin(), parameters.end());
  for (auto* parameter : parameters) {
    result.argument_count += types.partition(parameter, owner).component_count;
  }
  const auto* result_type = types.unqualified(family->returnType());
  if (result_type->kind() != cxx::TypeKind::kVoid) {
    sources.push_back(result_type);
  }
  result.native = bind_signature(
      types, sources, owner, builder,
      has_predicates ? SignatureIdentityRequirement::Required
                     : SignatureIdentityRequirement::DependentTypes);
  size_t offset = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto count = types.partition(parameters[i], owner).component_count;
    const auto* source = provider->parameterTypes()[i];
    if (types.unqualified(source) == types.unqualified(parameters[i])) {
      if (has_predicates) {
        result.source_identities.insert(
            result.source_identities.end(),
            result.native.identities.begin() + offset,
            result.native.identities.begin() + offset + count);
      }
    } else {
      const auto shape = input_shape(types, source, owner);
      auto& type = result.native.types[offset + count - 1];
      std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
      for (size_t axis = 0; axis < shape->rank; ++axis) {
        dimensions[axis] = shape->extent(axis) >= 0
                               ? loom_dim_pack_static(shape->extent(axis))
                               : loom_type_dim(type, axis);
        if (has_predicates && shape->extent(axis) < 0) {
          result.source_identities.push_back(
              loom_dim_is_dynamic(dimensions[axis])
                  ? loom_dim_value_id(dimensions[axis])
                  : LOOM_VALUE_ID_INVALID);
        }
      }
      if (has_predicates && shape->encoding) {
        result.source_identities.push_back(
            loom_type_has_ssa_encoding(type) ? loom_type_encoding_value_id(type)
                                             : LOOM_VALUE_ID_INVALID);
      }
      type = result.native.type_storage.with_dimensions(
          type, std::span<const uint64_t>(dimensions).first(shape->rank));
      if (!shape->encoding) {
        type.encoding_id = 0;
        type.encoding_flags = 0;
      }
      if (has_predicates) {
        result.source_identities.push_back(
            result.native.identities[offset + count - 1]);
      }
    }
    offset += count;
  }
  if (has_predicates) {
    result.source_identities.insert(
        result.source_identities.end(),
        result.native.identities.begin() + result.argument_count,
        result.native.identities.end());
  }
  return result;
}

}  // namespace loom::cxx_import
