// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/duplication.h"

#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <array>
#include <vector>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/pool/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/tensor/ops.h"
#include "loom/ops/tile/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {

Value ValueDuplication::build(Value source, const cxx::Type* source_type,
                              cxx::AST* owner) {
  const auto& partition = types_.partition(source_type, owner);
  if (&partition != &source.partition()) {
    diagnostics_.reject(
        unit_, owner,
        "value duplication requires the source type's admitted partition");
  }
  return build_partition(source, source_type, owner);
}

Value ValueDuplication::build_leaf(Value source, const cxx::Type* source_type,
                                   cxx::AST* owner, BuildFunction build) {
  std::array<const cxx::Type*, 1> sources = {source_type};
  auto signature = bind_signature(types_, sources, owner, &builder_);
  IREE_ASSERT_EQ(signature.types.size(), source.components().size());
  loom_op_t* op;
  check(build(&builder_, source.components().data(), source.components().size(),
              signature.types.data(), signature.types.size(),
              locations_.get(owner), &op));
  return arena_.capture(source.partition(),
                        {loom_op_results(op), signature.types.size()});
}

Value ValueDuplication::build_partition(Value source,
                                        const cxx::Type* source_type,
                                        cxx::AST* owner) {
  const auto& partition = source.partition();
  switch (partition.kind) {
    case ValueKind::SSA: {
      auto type = types_.get(source_type, owner);
      if (loom_type_kind(type) == LOOM_TYPE_SCALAR) {
        return build_leaf(source, source_type, owner,
                          loom_scalar_duplicate_build);
      }
      if (loom_type_kind(type) == LOOM_TYPE_VECTOR) {
        return build_leaf(source, source_type, owner,
                          loom_vector_duplicate_build);
      }
      diagnostics_.reject(
          unit_, owner,
          "source SSA copy requires a scalar or native vector value");
    }
    case ValueKind::Pointer: {
      const auto pointer = source.pointer();
      const std::array<loom_value_id_t, 1> root_source = {pointer.root};
      const std::array<loom_type_t, 1> root_type = {loom_type_buffer()};
      loom_op_t* root;
      check(loom_buffer_duplicate_build(
          &builder_, root_source.data(), root_source.size(), root_type.data(),
          root_type.size(), locations_.get(owner), &root));
      const std::array<loom_value_id_t, 1> offset_source = {
          pointer.byte_offset};
      const std::array<loom_type_t, 1> offset_type = {
          loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)};
      loom_op_t* offset;
      check(loom_index_duplicate_build(&builder_, offset_source.data(),
                                       offset_source.size(), offset_type.data(),
                                       offset_type.size(),
                                       locations_.get(owner), &offset));
      return Value(
          Pointer{loom_op_results(root)[0], loom_op_results(offset)[0]});
    }
    case ValueKind::Record: {
      if (!partition.component_count) {
        return source;
      }
      const auto& record = static_cast<const RecordPartition&>(partition);
      std::vector<loom_value_id_t> components;
      components.reserve(partition.component_count);
      for (const auto& member : record.members) {
        auto* member_type = member.field->type();
        if (unit_.typeTraits().is_lvalue_reference(member_type)) {
          member_type = unit_.typeTraits().add_pointer(
              unit_.typeTraits().remove_reference(member_type));
        }
        build_partition(
            source.project(*member.partition, member.component_offset),
            member_type, owner)
            .append_to(components);
      }
      return arena_.capture(partition, components);
    }
    case ValueKind::Array: {
      if (!partition.component_count) {
        return source;
      }
      const auto& array = static_cast<const ArrayPartition&>(partition);
      const size_t element_count = array.element->component_count;
      std::vector<loom_value_id_t> components;
      components.reserve(partition.component_count);
      for (size_t index = 0; index < array.source->size(); ++index) {
        build_partition(source.project(*array.element, index * element_count),
                        array.source->elementType(), owner)
            .append_to(components);
      }
      return arena_.capture(partition, components);
    }
    case ValueKind::Buffer:
      return build_leaf(source, source_type, owner,
                        loom_buffer_duplicate_build);
    case ValueKind::Address:
      return build_leaf(source, source_type, owner, loom_index_duplicate_build);
    case ValueKind::OpaqueDialect:
      diagnostics_.reject(
          unit_, owner,
          "copying an opaque dialect value across a mutable call requires a "
          "dialect-owned duplication operation");
    case ValueKind::Encoding:
      return build_leaf(source, source_type, owner,
                        loom_encoding_duplicate_build);
    case ValueKind::Pool:
      return build_leaf(source, source_type, owner, loom_pool_duplicate_build);
    case ValueKind::Tile:
      return build_leaf(source, source_type, owner, loom_tile_duplicate_build);
    case ValueKind::Channel:
      diagnostics_.reject(
          unit_, owner,
          "channel capabilities cannot be structurally copied across a "
          "mutable call; use an explicit channel fanout or split operation");
    case ValueKind::Vector:
      return build_leaf(source, source_type, owner,
                        loom_vector_duplicate_build);
    case ValueKind::View:
      return build_leaf(source, source_type, owner, loom_view_duplicate_build);
    case ValueKind::Tensor:
      return build_leaf(source, source_type, owner,
                        loom_tensor_duplicate_build);
  }
  IREE_ASSERT_UNREACHABLE("unknown source value kind");
}

}  // namespace loom::cxx_import
