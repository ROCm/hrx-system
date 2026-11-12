// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/template_inputs.h"

#include <cxx/types.h>

#include <array>
#include <optional>

#include "loom/import/cxx/binding/refinement.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"

namespace loom::cxx_import {
namespace {

// Source admission already proved carrier, rank, element, and encoding-role
// compatibility. Only the destination's dynamic components need transport.
void append_dependencies(const ShapedSignature& destination, loom_type_t actual,
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

}  // namespace

bool template_input_compatible(Types& types, const cxx::Type* family,
                               const cxx::Type* source, cxx::AST* owner) {
  if (types.unqualified(family) == types.unqualified(source)) {
    return true;
  }
  const auto required = shaped_signature(types, family, owner);
  const auto actual = shaped_signature(types, source, owner);
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
  const auto shape = shaped_signature(types, family, owner);
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
  const auto shape = shaped_signature(types, source, owner);
  const auto payload = incoming.components().back();
  std::vector<loom_value_id_t> components;
  components.reserve(partition.component_count);
  append_dependencies(*shape, loom_module_value_type(builder->module, payload),
                      builder, location, components);
  components.push_back(payload);
  BoundTypeStorage storage;
  const auto type = shape->bind_type(
      std::span<const loom_value_id_t>(components).first(components.size() - 1),
      storage);
  components.back() = refine_value(payload, type, builder, location);
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
      const auto shape = shaped_signature(types, source, owner);
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
