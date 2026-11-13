// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/refinement.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>

#include "loom/import/cxx/binding/index_values.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/tensor/ops.h"
#include "loom/ops/tile/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {

std::optional<ShapedSignature> shaped_signature(Types& types,
                                                const cxx::Type* source,
                                                cxx::AST* owner) {
  const auto& partition = types.partition(source, owner);
  switch (partition.kind) {
    case ValueKind::Tile: {
      const auto& tile = static_cast<const TilePartition&>(partition);
      return ShapedSignature{
          .partition = &partition,
          .kind = LOOM_TYPE_TILE,
          .element = tile.element_type,
          .extents = &tile.extents,
          .rank = tile.extents.size(),
          .encoding = tile.encoding,
      };
    }
    case ValueKind::View: {
      const auto& view = static_cast<const ViewPartition&>(partition);
      return ShapedSignature{
          .partition = &partition,
          .kind = LOOM_TYPE_VIEW,
          .element = view.element_type,
          .extents = &view.extents,
          .rank = view.extents.size(),
          .encoding = view.encoding,
      };
    }
    case ValueKind::Vector: {
      const auto& vector = static_cast<const VectorPartition&>(partition);
      return ShapedSignature{
          .partition = &partition,
          .kind = LOOM_TYPE_VECTOR,
          .element = vector.element_type,
          .extents = &vector.extents,
          .rank = vector.extents.size(),
      };
    }
    case ValueKind::Tensor: {
      const auto& tensor = static_cast<const TensorPartition&>(partition);
      return ShapedSignature{
          .partition = &partition,
          .kind = LOOM_TYPE_TENSOR,
          .element = tensor.element_type,
          .inline_extents = {tensor.extent,
                             static_cast<int64_t>(tensor.vector_lane_count)},
          .rank = tensor.vector_lane_count ? 2u : 1u,
      };
    }
    case ValueKind::SSA:
      if (auto* vector = types.native_vector(source)) {
        return ShapedSignature{
            .partition = &partition,
            .native_type = types.get(source, owner),
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

loom_type_t ShapedSignature::bind_type(
    std::span<const loom_value_id_t> dependencies,
    BoundTypeStorage& storage) const {
  switch (partition->kind) {
    case ValueKind::View:
      return static_cast<const ViewPartition*>(partition)->bind_type(
          dependencies, storage);
    case ValueKind::Vector:
      return static_cast<const VectorPartition*>(partition)->bind_type(
          dependencies, storage);
    case ValueKind::Tile:
      return static_cast<const TilePartition*>(partition)->bind_type(
          dependencies, storage);
    case ValueKind::Tensor:
      return static_cast<const TensorPartition*>(partition)->bind_type(
          dependencies);
    case ValueKind::SSA:
      return native_type;
    default:
      IREE_ASSERT_UNREACHABLE("admitted shaped value carrier");
      IREE_BUILTIN_UNREACHABLE();
  }
}

loom_value_id_t refine_value(loom_value_id_t value, loom_type_t type,
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
      IREE_ASSERT_UNREACHABLE("admitted shaped value carrier");
      IREE_BUILTIN_UNREACHABLE();
  }
  return loom_op_results(refined)[0];
}

namespace {

std::optional<loom_type_kind_t> refinement_kind(std::string_view name) {
  if (name == "vector.refine") {
    return LOOM_TYPE_VECTOR;
  }
  if (name == "view.refine") {
    return LOOM_TYPE_VIEW;
  }
  if (name == "tile.refine") {
    return LOOM_TYPE_TILE;
  }
  return std::nullopt;
}

}  // namespace

bool RefinementIntrinsic::supports(std::string_view name) {
  return refinement_kind(name).has_value();
}

std::optional<RefinementIntrinsic> RefinementIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  const auto kind = refinement_kind(attribute.arguments[0]->name());
  if (!kind) {
    return std::nullopt;
  }
  const auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  const auto parameters = signature->parameterTypes();
  if (attribute.arguments.size() != 1 || signature->isVariadic() ||
      parameters.empty()) {
    diagnostics.reject(unit, owner,
                       "refinement requires one operation name and a fixed "
                       "source signature beginning with the shaped value");
  }
  const auto source = shaped_signature(types, parameters[0], owner);
  const auto result = shaped_signature(types, signature->returnType(), owner);
  if (!source || !result || source->kind != *kind || result->kind != *kind ||
      source->element != result->element || source->rank != result->rank) {
    diagnostics.reject(unit, owner,
                       "refinement must preserve the carrier, element type, "
                       "access qualifiers, and rank");
  }
  const size_t dynamic_count =
      result->partition->component_count - 1 - (result->encoding ? 1 : 0);
  const size_t required = 1 + (dynamic_count ? 1 : 0);
  const bool explicit_encoding =
      result->encoding && parameters.size() == required + 1;
  if (parameters.size() != required + (explicit_encoding ? 1 : 0)) {
    diagnostics.reject(unit, owner,
                       "refinement requires the source, one index record for "
                       "dependent result dimensions, and at most one result "
                       "encoding");
  }
  if (dynamic_count) {
    require_logical_index_record(unit, diagnostics, types, parameters[1],
                                 dynamic_count, owner,
                                 "refinement result dimensions");
  }
  if (explicit_encoding &&
      &types.partition(parameters.back(), owner) != result->encoding) {
    diagnostics.reject(unit, owner,
                       "refinement encoding must match the result carrier");
  }
  if (result->encoding && !explicit_encoding &&
      source->encoding != result->encoding) {
    diagnostics.reject(unit, owner,
                       "refinement must supply a result encoding when the "
                       "source does not carry the same encoding role and rank");
  }
  return RefinementIntrinsic(*result, dynamic_count, explicit_encoding);
}

Value RefinementIntrinsic::call(std::span<const Value> arguments,
                                ValueArena& arena, loom_builder_t* builder,
                                loom_location_id_t location) const {
  std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 2> components;
  if (dynamic_count_) {
    const auto dimensions = arguments[1].components();
    std::copy(dimensions.begin(), dimensions.end(), components.begin());
  }
  const size_t dependency_count = result_.partition->component_count - 1;
  const auto source = arguments[0].components();
  if (result_.encoding) {
    components[dynamic_count_] =
        explicit_encoding_ ? arguments.back().ssa() : source[source.size() - 2];
  }
  auto dependencies = std::span(components).first(dependency_count);
  BoundTypeStorage storage;
  const auto type = result_.bind_type(dependencies, storage);
  components[dependency_count] =
      refine_value(source.back(), type, builder, location);
  return arena.capture(*result_.partition,
                       std::span(components).first(dependency_count + 1));
}

}  // namespace loom::cxx_import
