// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/storage.h"

#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/token.h>
#include <cxx/types.h>

#include <algorithm>

#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {

Pointer Storage::root(loom_value_id_t buffer, cxx::AST* owner) {
  return {buffer,
          scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner))};
}

Pointer Storage::advance(Pointer base, loom_value_id_t displacement,
                         const cxx::Type* base_type,
                         const cxx::Type* index_type, cxx::TokenKind operation,
                         cxx::AST* owner) {
  auto* pointer =
      cxx::type_cast<cxx::PointerType>(types_.unqualified(base_type));
  auto* array =
      cxx::type_cast<cxx::BoundedArrayType>(types_.unqualified(base_type));
  if ((!pointer && !array) || !unit_.typeTraits().is_integral(index_type) ||
      (operation != cxx::TokenKind::T_PLUS &&
       operation != cxx::TokenKind::T_MINUS)) {
    diagnostics_.reject(unit_, owner,
                        "pointer arithmetic requires an integral displacement");
  }
  auto* element_type = pointer ? pointer->elementType() : array->elementType();
  types_.get(element_type, owner);
  auto bytes = unit_.control()->memoryLayout()->sizeOf(element_type);
  if (!bytes) {
    diagnostics_.reject(unit_, owner, "unknown element size");
  }
  auto source = locations_.get(owner);
  auto wide_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  auto offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  // The fixed-width arithmetic retains the signed displacement until it is
  // combined with the existing origin. Defined C++ pointer arithmetic remains
  // within the allocation (or one-past), so that final origin is nonnegative.
  auto wide = scalars_.convert(displacement, index_type,
                               unit_.control()->getLongLongIntType(), owner);
  auto size = scalars_.integer(*bytes, LOOM_SCALAR_TYPE_I64, source);
  loom_op_t* op;
  check(
      loom_scalar_muli_build(&builder_, 0, wide, size, wide_type, source, &op));
  auto delta = loom_op_results(op)[0];
  check(loom_index_cast_build(&builder_, base.byte_offset, offset_type,
                              wide_type, source, &op));
  auto origin = loom_op_results(op)[0];
  auto build = operation == cxx::TokenKind::T_MINUS ? loom_scalar_subi_build
                                                    : loom_scalar_addi_build;
  check(build(&builder_, 0, origin, delta, wide_type, source, &op));
  auto byte_offset = loom_op_results(op)[0];
  loom_predicate_t range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {byte_offset, 0, INT64_MAX},
  };
  check(loom_scalar_assume_build(&builder_, &byte_offset, 1, &range, 1,
                                 &wide_type, 1, source, &op));
  check(loom_index_cast_build(&builder_, loom_op_results(op)[0], wide_type,
                              offset_type, source, &op));
  return {base.root, loom_op_results(op)[0]};
}

StorageAccess Storage::dereference(Pointer base, const cxx::Type* element_type,
                                   cxx::AST* owner) {
  auto element = types_.get(element_type, owner);
  auto* vector = types_.vector(element_type);
  auto view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, loom_type_element_type(element),
                          vector ? vector->elementCount() : 1, 0);
  loom_op_t* view;
  check(loom_buffer_view_build(&builder_, base.root, base.byte_offset,
                               view_type, locations_.get(owner), &view));
  return {loom_op_results(view)[0], std::nullopt};
}

loom_value_id_t Storage::load(const StorageAccess& access,
                              const cxx::Type* element_type, cxx::AST* owner) {
  int64_t selector = access.index ? INT64_MIN : 0;
  auto build = types_.vector(element_type) ? loom_vector_load_build
                                           : loom_view_load_build;
  loom_op_t* op;
  check(build(&builder_, 0, types_.memory_access_flags(element_type),
              access.view, access.index ? &*access.index : nullptr,
              access.index ? 1 : 0, &selector, 1, 0, 0,
              types_.get(element_type, owner), locations_.get(owner), &op));
  return loom_op_results(op)[0];
}

void Storage::store(const StorageAccess& access, loom_value_id_t value,
                    const cxx::Type* element_type, cxx::AST* owner) {
  int64_t selector = access.index ? INT64_MIN : 0;
  auto build = types_.vector(element_type) ? loom_vector_store_build
                                           : loom_view_store_build;
  loom_op_t* op;
  check(build(&builder_, 0, types_.memory_access_flags(element_type), value,
              access.view, access.index ? &*access.index : nullptr,
              access.index ? 1 : 0, &selector, 1, 0, 0, locations_.get(owner),
              &op));
}

StorageAccess Storage::subscript(Pointer base, loom_value_id_t index,
                                 const cxx::Type* base_type,
                                 const cxx::Type* subscript_type,
                                 cxx::AST* owner) {
  if (!unit_.typeTraits().is_integral(subscript_type)) {
    diagnostics_.reject(unit_, owner, "subscripts require integral offsets");
  }
  if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(
          types_.unqualified(base_type))) {
    auto input_type = types_.get(subscript_type, owner);
    loom_op_t* cast;
    if (types_.is_unsigned(subscript_type)) {
      if (loom_type_element_type(input_type) == LOOM_SCALAR_TYPE_I64) {
        // A defined fixed-array access is within the declared extent. Publish
        // that source precondition before entering the offset domain.
        loom_predicate_t range = {
            .kind = LOOM_PREDICATE_RANGE,
            .arg_count = 3,
            .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                         LOOM_PRED_ARG_CONST},
            .args = {index, 0, static_cast<int64_t>(array->size() - 1)},
        };
        check(loom_scalar_assume_build(&builder_, &index, 1, &range, 1,
                                       &input_type, 1, locations_.get(owner),
                                       &cast));
        index = loom_op_results(cast)[0];
      }
      auto offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
      check(loom_index_cast_build(&builder_, index, input_type, offset_type,
                                  locations_.get(owner), &cast));
      index = loom_op_results(cast)[0];
      input_type = offset_type;
    }
    check(loom_index_cast_build(&builder_, index, input_type,
                                loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                locations_.get(owner), &cast));
    return {array_views_.at(base.root), loom_op_results(cast)[0]};
  }
  auto advanced = advance(base, index, base_type, subscript_type,
                          cxx::TokenKind::T_PLUS, owner);
  auto* pointer =
      cxx::type_cast<cxx::PointerType>(types_.unqualified(base_type));
  return dereference(advanced, pointer->elementType(), owner);
}

StorageAllocation Storage::workgroup(const cxx::BoundedArrayType* array,
                                     int64_t explicit_alignment,
                                     cxx::AST* owner) {
  types_.get(array, owner);
  auto* layout = unit_.control()->memoryLayout();
  auto bytes = layout->sizeOf(array);
  auto alignment = layout->alignmentOf(array);
  if (!bytes || !alignment) {
    diagnostics_.reject(unit_, owner, "unknown shared array layout");
  }
  auto length =
      scalars_.integer(*bytes, LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner));
  loom_op_t* op;
  check(loom_buffer_alloca_build(
      &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      std::max<int64_t>(*alignment, explicit_alignment), length,
      loom_type_buffer(), locations_.get(owner), &op));
  auto root = loom_op_results(op)[0];
  auto base =
      scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner));
  auto element = types_.get(array->elementType(), owner);
  auto view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(element), array->size(), 0);
  check(loom_buffer_view_build(&builder_, root, base, view_type,
                               locations_.get(owner), &op));
  auto view = loom_op_results(op)[0];
  array_views_[root] = view;
  return {{root, base}, view};
}

}  // namespace loom::cxx_import
