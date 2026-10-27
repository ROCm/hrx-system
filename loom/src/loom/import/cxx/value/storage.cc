// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/storage.h"

#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/types.h>

#include <algorithm>
#include <numeric>

#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {

// C++ bool objects occupy a byte while their computed values are predicates.
// All other admitted scalar and vector objects share their value
// representation.
loom_type_t object_storage_type(loom_type_t value_type) {
  return loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_I1
             ? loom_type_scalar(LOOM_SCALAR_TYPE_I8)
             : value_type;
}

}  // namespace

Pointer Storage::root(loom_value_id_t buffer, cxx::AST* owner) {
  auto byte_offset =
      scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner));
  zero_origin_offsets_.insert(byte_offset);
  return {buffer, byte_offset};
}

Pointer Storage::constrain_origin(Pointer pointer, cxx::AST* owner) {
  if (unit_.control()->memoryLayout()->sizeOfPointer() != 4) {
    return pointer;
  }
  auto source = locations_.get(owner);
  auto offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_predicate_t range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {pointer.byte_offset, 0, UINT32_MAX},
  };
  loom_op_t* op;
  check(loom_index_assume_build(&builder_, &pointer.byte_offset, 1, &range, 1,
                                &offset_type, 1, source, &op));
  auto byte_offset = loom_op_results(op)[0];
  if (zero_origin_offsets_.contains(pointer.byte_offset)) {
    zero_origin_offsets_.insert(byte_offset);
  }
  return {pointer.root, byte_offset};
}

Pointer Storage::constrain_access(Pointer pointer, int64_t access_bytes,
                                  cxx::AST* owner) {
  auto root = roots_.find(pointer.root);
  if (root == roots_.end() || !root->second.allocation ||
      access_bytes > root->second.allocation->byte_length) {
    return pointer;
  }
  auto source = locations_.get(owner);
  auto offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_predicate_t range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {pointer.byte_offset, 0,
               root->second.allocation->byte_length - access_bytes},
  };
  loom_op_t* op;
  check(loom_index_assume_build(&builder_, &pointer.byte_offset, 1, &range, 1,
                                &offset_type, 1, source, &op));
  auto byte_offset = loom_op_results(op)[0];
  if (zero_origin_offsets_.contains(pointer.byte_offset)) {
    zero_origin_offsets_.insert(byte_offset);
  }
  return {pointer.root, byte_offset};
}

StorageProjection Storage::project(Pointer pointer,
                                   const cxx::Type* object_type,
                                   cxx::AST* owner) {
  types_.storage_size(object_type, owner);
  auto alignment = unit_.control()->memoryLayout()->alignmentOf(object_type);
  if (!alignment) {
    diagnostics_.reject(unit_, owner, "unknown object alignment");
  }
  return {pointer, *alignment};
}

StorageProjection Storage::advance(StorageProjection base,
                                   loom_value_id_t displacement,
                                   const cxx::Type* base_type,
                                   const cxx::Type* index_type,
                                   cxx::TokenKind operation, cxx::AST* owner) {
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
  auto bytes = types_.storage_size(element_type, owner);
  auto source = locations_.get(owner);
  auto wide_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  auto offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  // The fixed-width arithmetic retains the signed displacement until it is
  // combined with the existing origin. Defined C++ pointer arithmetic remains
  // within the allocation (or one-past), so that final origin is nonnegative.
  auto wide = scalars_.convert(displacement, index_type,
                               unit_.control()->getLongLongIntType(), owner);
  auto size = scalars_.integer(bytes, LOOM_SCALAR_TYPE_I64, source);
  loom_op_t* op;
  check(
      loom_scalar_muli_build(&builder_, 0, wide, size, wide_type, source, &op));
  auto delta = loom_op_results(op)[0];
  check(loom_index_cast_build(&builder_, base.pointer.byte_offset, offset_type,
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
      .args = {byte_offset, 0,
               unit_.control()->memoryLayout()->sizeOfPointer() == 4
                   ? UINT32_MAX
                   : INT64_MAX},
  };
  check(loom_scalar_assume_build(&builder_, &byte_offset, 1, &range, 1,
                                 &wide_type, 1, source, &op));
  check(loom_index_cast_build(&builder_, loom_op_results(op)[0], wide_type,
                              offset_type, source, &op));
  return {{base.pointer.root, loom_op_results(op)[0]},
          std::gcd(base.alignment, static_cast<uint64_t>(bytes)),
          true};
}

StorageProjection Storage::member(StorageProjection base,
                                  cxx::FieldSymbol* field, cxx::AST* owner) {
  return subobject(base, *field->offsetInClass(), owner);
}

StorageProjection Storage::element(StorageProjection base,
                                   const cxx::BoundedArrayType* type,
                                   size_t index, cxx::AST* owner) {
  return index == 0 ? base
                    : subobject(base,
                                index * types_.storage_size(type->elementType(),
                                                            owner),
                                owner);
}

StorageProjection Storage::subobject(StorageProjection base,
                                     uint64_t byte_offset, cxx::AST* owner) {
  auto source = locations_.get(owner);
  auto field_offset =
      scalars_.integer(byte_offset, LOOM_SCALAR_TYPE_OFFSET, source);
  loom_op_t* op;
  check(loom_index_add_build(&builder_, base.pointer.byte_offset, field_offset,
                             loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), source,
                             &op));
  Pointer pointer =
      constrain_origin({base.pointer.root, loom_op_results(op)[0]}, owner);
  return {pointer, std::gcd(base.alignment, byte_offset), true};
}

StorageAccess Storage::dereference(StorageProjection base,
                                   const cxx::Type* element_type,
                                   cxx::AST* owner) {
  if (!base.pointer_width_constrained) {
    base.pointer = constrain_origin(base.pointer, owner);
  }
  auto element = object_storage_type(types_.get(element_type, owner));
  base.pointer = constrain_access(
      base.pointer, types_.storage_size(element_type, owner), owner);
  auto* vector = types_.native_vector(element_type);
  auto view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, loom_type_element_type(element),
                          vector ? vector->elementCount() : 1, 0);
  view_type = loom_type_view_with_alignment(
      view_type,
      static_cast<uint8_t>(std::min<uint64_t>(
          base.alignment, loom_type_view_natural_alignment(view_type))));
  loom_op_t* view;
  check(loom_buffer_view_build(&builder_, base.pointer.root,
                               base.pointer.byte_offset, view_type,
                               locations_.get(owner), &view));
  return {loom_op_results(view)[0], std::nullopt};
}

loom_value_id_t Storage::load(const StorageAccess& access,
                              const cxx::Type* element_type, cxx::AST* owner) {
  auto value_type = types_.get(element_type, owner);
  int64_t selector = access.index ? INT64_MIN : 0;
  auto build = types_.native_vector(element_type) ? loom_vector_load_build
                                                  : loom_view_load_build;
  loom_op_t* op;
  check(build(&builder_, 0, types_.memory_access_flags(element_type),
              access.view, access.index ? &*access.index : nullptr,
              access.index ? 1 : 0, &selector, 1, 0, 0,
              object_storage_type(value_type), locations_.get(owner), &op));
  auto value = loom_op_results(op)[0];
  return loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_I1
             ? scalars_.convert(value, unit_.control()->getUnsignedCharType(),
                                element_type, owner)
             : value;
}

void Storage::store(const StorageAccess& access, loom_value_id_t value,
                    const cxx::Type* element_type, cxx::AST* owner) {
  if (loom_type_element_type(loom_module_value_type(builder_.module, value)) ==
      LOOM_SCALAR_TYPE_I1) {
    value = scalars_.convert(value, element_type,
                             unit_.control()->getUnsignedCharType(), owner);
  }
  int64_t selector = access.index ? INT64_MIN : 0;
  auto build = types_.native_vector(element_type) ? loom_vector_store_build
                                                  : loom_view_store_build;
  loom_op_t* op;
  check(build(&builder_, 0, types_.memory_access_flags(element_type), value,
              access.view, access.index ? &*access.index : nullptr,
              access.index ? 1 : 0, &selector, 1, 0, 0, locations_.get(owner),
              &op));
}

StorageAccess Storage::subscript(StorageProjection base, loom_value_id_t index,
                                 const cxx::Type* base_type,
                                 const cxx::Type* subscript_type,
                                 cxx::AST* owner) {
  if (!unit_.typeTraits().is_integral(subscript_type)) {
    diagnostics_.reject(unit_, owner, "subscripts require integral offsets");
  }
  auto* array =
      cxx::type_cast<cxx::BoundedArrayType>(types_.unqualified(base_type));
  auto root = array ? roots_.find(base.pointer.root) : roots_.end();
  const Allocation* declared = root != roots_.end() && root->second.allocation
                                   ? &*root->second.allocation
                                   : nullptr;
  if (declared && declared->array_type == array &&
      declared->array_view.has_value() &&
      declared->byte_offset == base.pointer.byte_offset) {
    // Widen before expressing the source's in-bounds precondition in the
    // signed range domain. Narrow unsigned indices retain their high bit.
    index = scalars_.convert(index, subscript_type,
                             unit_.control()->getLongLongIntType(), owner);
    auto wide_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    loom_predicate_t range = {
        .kind = LOOM_PREDICATE_RANGE,
        .arg_count = 3,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_CONST},
        .args = {index, 0, static_cast<int64_t>(array->size() - 1)},
    };
    loom_op_t* cast;
    check(loom_scalar_assume_build(&builder_, &index, 1, &range, 1, &wide_type,
                                   1, locations_.get(owner), &cast));
    check(loom_index_cast_build(&builder_, loom_op_results(cast)[0], wide_type,
                                loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                locations_.get(owner), &cast));
    return {*declared->array_view, loom_op_results(cast)[0]};
  }
  auto advanced = advance(base, index, base_type, subscript_type,
                          cxx::TokenKind::T_PLUS, owner);
  auto* pointer =
      cxx::type_cast<cxx::PointerType>(types_.unqualified(base_type));
  return dereference(
      advanced, pointer ? pointer->elementType() : array->elementType(), owner);
}

StorageAllocation Storage::allocate(const cxx::Type* type,
                                    loom_value_fact_memory_space_t memory_space,
                                    int64_t explicit_alignment,
                                    cxx::AST* owner) {
  auto bytes = types_.storage_size(type, owner);
  auto* layout = unit_.control()->memoryLayout();
  auto alignment = layout->alignmentOf(type);
  if (!alignment) {
    diagnostics_.reject(unit_, owner, "unknown object alignment");
  }
  auto length =
      scalars_.integer(bytes, LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner));
  loom_op_t* op;
  check(loom_buffer_alloca_build(
      &builder_, /*build_flags=*/0, memory_space, LOOM_VALUE_ID_INVALID,
      std::max<int64_t>(*alignment, explicit_alignment), length,
      loom_type_buffer(), locations_.get(owner), &op));
  return record_fixed(loom_op_results(op)[0], type, memory_space, bytes,
                      *alignment, owner);
}

StorageAllocation Storage::admit_fixed(
    loom_value_id_t root, const cxx::Type* type,
    loom_value_fact_memory_space_t memory_space, cxx::AST* owner) {
  auto bytes = types_.storage_size(type, owner);
  auto alignment = unit_.control()->memoryLayout()->alignmentOf(type);
  if (!alignment) {
    diagnostics_.reject(unit_, owner, "unknown object alignment");
  }
  return record_fixed(root, type, memory_space, bytes, *alignment, owner);
}

StorageAllocation Storage::record_fixed(
    loom_value_id_t root, const cxx::Type* type,
    loom_value_fact_memory_space_t memory_space, int64_t bytes,
    uint64_t natural_alignment, cxx::AST* owner) {
  loom_op_t* op;
  auto base =
      scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner));
  zero_origin_offsets_.insert(base);
  auto* array = cxx::type_cast<cxx::BoundedArrayType>(types_.unqualified(type));
  auto* view_source_type = array ? array->elementType() : type;
  auto source_kind = types_.unqualified(view_source_type)->kind();
  // A Loom view has scalar elements. Preserve the direct typed view for a
  // scalar object or scalar array; it lets dynamic scalar array indexing keep
  // the declared extent. Aggregate, nested-array and vector-array elements use
  // a byte footprint view while later object projections recover typed field
  // or vector views from the allocation root and source layout.
  bool byte_footprint = source_kind == cxx::TypeKind::kClass ||
                        source_kind == cxx::TypeKind::kBoundedArray ||
                        (array && source_kind == cxx::TypeKind::kVector);
  auto* vector =
      byte_footprint ? nullptr : types_.native_vector(view_source_type);
  auto element = byte_footprint
                     ? loom_type_scalar(LOOM_SCALAR_TYPE_I8)
                     : object_storage_type(types_.get(view_source_type, owner));
  auto count = byte_footprint ? bytes
               : array        ? array->size()
               : vector       ? vector->elementCount()
                              : 1;
  auto view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(element), count, 0);
  view_type = loom_type_view_with_alignment(
      view_type,
      static_cast<uint8_t>(std::min<uint64_t>(
          natural_alignment, loom_type_view_natural_alignment(view_type))));
  check(loom_buffer_view_build(&builder_, root, base, view_type,
                               locations_.get(owner), &op));
  auto view = loom_op_results(op)[0];
  roots_[root] = {
      .memory_space = memory_space,
      .allocation =
          Allocation{
              .byte_length = bytes,
              .array_type = array,
              .byte_offset = base,
              .array_view = array && !byte_footprint
                                ? std::optional<loom_value_id_t>(view)
                                : std::nullopt,
          },
  };
  return {{root, base}, view};
}

Pointer Storage::allocate_elements(const cxx::Type* element_type,
                                   loom_value_id_t element_count,
                                   const cxx::Type* element_count_type,
                                   loom_value_fact_memory_space_t memory_space,
                                   int64_t explicit_alignment,
                                   cxx::AST* owner) {
  auto element_bytes = types_.storage_size(element_type, owner);
  auto* layout = unit_.control()->memoryLayout();
  auto alignment = layout->alignmentOf(element_type);
  if (!alignment) {
    diagnostics_.reject(unit_, owner, "unknown object alignment");
  }

  auto source = locations_.get(owner);
  loom_op_t* op;
  auto count = element_count;
  if (types_.address_domain(element_count_type, owner) !=
      LOOM_SCALAR_TYPE_INDEX) {
    auto wide_count =
        scalars_.convert(element_count, element_count_type,
                         unit_.control()->getUnsignedLongLongIntType(), owner);
    check(loom_index_cast_build(
        &builder_, wide_count, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), source, &op));
    count = loom_op_results(op)[0];
  }
  auto stride =
      scalars_.integer(element_bytes, LOOM_SCALAR_TYPE_OFFSET, source);
  check(loom_index_scale_build(&builder_, count, stride,
                               loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                               source, &op));
  auto byte_length = loom_op_results(op)[0];
  check(loom_buffer_alloca_build(
      &builder_, /*build_flags=*/0, memory_space, LOOM_VALUE_ID_INVALID,
      std::max<int64_t>(*alignment, explicit_alignment), byte_length,
      loom_type_buffer(), source, &op));
  auto root = loom_op_results(op)[0];
  roots_[root] = {.memory_space = memory_space, .allocation = std::nullopt};
  auto base = scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET, source);
  zero_origin_offsets_.insert(base);
  return {root, base};
}

Pointer Storage::allocate_bytes(loom_value_id_t byte_length,
                                loom_value_fact_memory_space_t memory_space,
                                int64_t alignment, cxx::AST* owner) {
  auto source = locations_.get(owner);
  loom_op_t* op;
  check(loom_buffer_alloca_build(&builder_, /*build_flags=*/0, memory_space,
                                 LOOM_VALUE_ID_INVALID, alignment, byte_length,
                                 loom_type_buffer(), source, &op));
  auto root = loom_op_results(op)[0];
  roots_[root] = {.memory_space = memory_space, .allocation = std::nullopt};
  auto base = scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET, source);
  zero_origin_offsets_.insert(base);
  return {root, base};
}

Pointer Storage::assume_noalias(Pointer pointer, cxx::AST* owner) {
  return {assume_noalias(pointer.root, owner), pointer.byte_offset};
}

loom_value_id_t Storage::assume_noalias(loom_value_id_t source,
                                        cxx::AST* owner) {
  auto type = loom_type_buffer();
  loom_op_t* op;
  check(loom_buffer_assume_noalias_build(&builder_, &source, 1, &type, 1,
                                         locations_.get(owner), &op));
  auto root = loom_op_results(op)[0];
  inherit_root_facts(source, root);
  return root;
}

Pointer Storage::assume_alignment(Pointer pointer, int64_t minimum_alignment,
                                  cxx::AST* owner) {
  if (!zero_origin_offsets_.contains(pointer.byte_offset)) {
    diagnostics_.reject(
        unit_, owner,
        "buffer alignment assumption requires a zero-origin pointer");
  }
  return {assume_alignment(pointer.root, minimum_alignment, owner),
          pointer.byte_offset};
}

loom_value_id_t Storage::assume_alignment(loom_value_id_t source,
                                          int64_t minimum_alignment,
                                          cxx::AST* owner) {
  auto type = loom_type_buffer();
  loom_op_t* op;
  check(loom_buffer_assume_alignment_build(&builder_, &source, 1,
                                           minimum_alignment, &type, 1,
                                           locations_.get(owner), &op));
  auto root = loom_op_results(op)[0];
  inherit_root_facts(source, root);
  return root;
}

loom_value_id_t Storage::assume_memory_space(
    loom_value_id_t root, loom_value_fact_memory_space_t memory_space,
    cxx::AST* owner) {
  auto facts = roots_.find(root);
  if (facts != roots_.end() && facts->second.memory_space != memory_space) {
    diagnostics_.reject(unit_, owner,
                        "memory-space assumption conflicts with the "
                        "source storage domain");
  }
  loom_op_t* op;
  check(loom_buffer_assume_memory_space_build(&builder_, memory_space, root,
                                              loom_type_buffer(),
                                              locations_.get(owner), &op));
  auto result = loom_op_results(op)[0];
  roots_.emplace(result, facts != roots_.end()
                             ? facts->second
                             : RootFacts{memory_space, std::nullopt});
  return result;
}

void Storage::inherit_root_facts(loom_value_id_t source,
                                 loom_value_id_t result) {
  if (auto facts = roots_.find(source); facts != roots_.end()) {
    roots_.emplace(result, facts->second);
  }
}

}  // namespace loom::cxx_import
