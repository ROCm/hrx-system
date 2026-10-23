// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/abi/c/signature_layout.h"

#include "loom/ops/func/ops.h"

static iree_status_t loom_c_abi_layout_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  return count == 0
             ? iree_ok_status()
             : iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static uint16_t loom_c_abi_layout_type_leaf_count(
    const loom_func_c_abi_signature_t* signature, uint32_t type_index,
    const uint16_t* type_leaf_counts) {
  const loom_attribute_t type = signature->types.values[type_index];
  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY: {
      const uint32_t element_type =
          (uint32_t)loom_func_abi_c_type_attr_element_type(type);
      const uint64_t element_count =
          (uint64_t)loom_func_abi_c_type_attr_element_count(type);
      IREE_ASSERT_LT(element_type, type_index);
      IREE_ASSERT_LE(element_count * type_leaf_counts[element_type],
                     UINT16_MAX);
      return (uint16_t)(element_count * type_leaf_counts[element_type]);
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_RECORD: {
      const loom_parameterized_attr_array_t fields =
          loom_func_abi_c_type_attr_fields(type);
      uint32_t leaf_count = 0;
      for (iree_host_size_t i = 0; i < fields.count; ++i) {
        const uint32_t field_type =
            (uint32_t)loom_func_abi_c_field_attr_type(fields.values[i]);
        IREE_ASSERT_LT(field_type, type_index);
        leaf_count += type_leaf_counts[field_type];
        IREE_ASSERT_LE(leaf_count, UINT16_MAX);
      }
      return (uint16_t)leaf_count;
    }
    default:
      IREE_ASSERT_NE(loom_func_abi_c_type_attr_component_count(type), 0);
      return 1;
  }
}

static bool loom_c_abi_layout_is_aligned(uint64_t byte_offset,
                                         uint64_t byte_alignment) {
  return (byte_offset & (byte_alignment - 1u)) == 0;
}

static void loom_c_abi_layout_build_leaves(
    const loom_func_c_abi_signature_t* signature, uint32_t type_index,
    uint64_t object_byte_offset, bool naturally_aligned, uint32_t depth,
    loom_c_abi_leaf_t* leaves, uint16_t* inout_leaf_count,
    uint16_t* inout_logical_offset) {
  IREE_ASSERT_LT(type_index, signature->types.count);
  IREE_ASSERT_LT(depth, 64u);
  const loom_attribute_t type = signature->types.values[type_index];
  if (loom_func_abi_c_type_attr_component_count(type) == 0) {
    return;
  }
  const uint64_t type_alignment =
      (uint64_t)loom_func_abi_c_type_attr_byte_alignment(type);
  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY: {
      const uint32_t element_type =
          (uint32_t)loom_func_abi_c_type_attr_element_type(type);
      const uint64_t element_count =
          (uint64_t)loom_func_abi_c_type_attr_element_count(type);
      const uint64_t element_stride =
          (uint64_t)loom_func_abi_c_type_attr_element_stride(type);
      const loom_attribute_t element = signature->types.values[element_type];
      const uint64_t element_alignment =
          (uint64_t)loom_func_abi_c_type_attr_byte_alignment(element);
      const bool elements_aligned =
          naturally_aligned && type_alignment >= element_alignment &&
          loom_c_abi_layout_is_aligned(element_stride, element_alignment);
      for (uint64_t i = 0; i < element_count; ++i) {
        loom_c_abi_layout_build_leaves(signature, element_type,
                                       object_byte_offset + i * element_stride,
                                       elements_aligned, depth + 1u, leaves,
                                       inout_leaf_count, inout_logical_offset);
      }
      return;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_RECORD: {
      const loom_parameterized_attr_array_t fields =
          loom_func_abi_c_type_attr_fields(type);
      for (iree_host_size_t i = 0; i < fields.count; ++i) {
        const loom_attribute_t field = fields.values[i];
        const uint32_t field_type =
            (uint32_t)loom_func_abi_c_field_attr_type(field);
        const uint64_t field_offset =
            (uint64_t)loom_func_abi_c_field_attr_byte_offset(field);
        const loom_attribute_t field_type_attr =
            signature->types.values[field_type];
        const uint64_t field_alignment =
            (uint64_t)loom_func_abi_c_type_attr_byte_alignment(field_type_attr);
        const bool field_aligned =
            naturally_aligned && type_alignment >= field_alignment &&
            loom_c_abi_layout_is_aligned(field_offset, field_alignment);
        loom_c_abi_layout_build_leaves(signature, field_type,
                                       object_byte_offset + field_offset,
                                       field_aligned, depth + 1u, leaves,
                                       inout_leaf_count, inout_logical_offset);
      }
      return;
    }
    default: {
      const loom_func_abi_c_type_kind_t kind =
          loom_func_abi_c_type_attr_kind(type);
      const uint16_t component_count =
          (uint16_t)loom_func_abi_c_type_attr_component_count(type);
      IREE_ASSERT_LE(component_count, UINT8_MAX);
      loom_c_abi_leaf_flags_t flags =
          naturally_aligned ? LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED : 0;
      if (kind == LOOM_FUNC_ABI_C_TYPE_KIND_POINTER) {
        flags |= LOOM_C_ABI_LEAF_FLAG_POINTER;
      } else if (kind == LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR) {
        flags |= LOOM_C_ABI_LEAF_FLAG_VECTOR;
      } else if (kind == LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN) {
        flags |= LOOM_C_ABI_LEAF_FLAG_BOOLEAN;
      }
      loom_c_abi_leaf_t* leaf = &leaves[(*inout_leaf_count)++];
      *leaf = (loom_c_abi_leaf_t){
          .type_index = type_index,
          .byte_offset = object_byte_offset,
          .byte_length = (uint64_t)loom_func_abi_c_type_attr_byte_length(type),
          .logical_offset = *inout_logical_offset,
          .logical_count = (uint8_t)component_count,
          .flags = flags,
      };
      *inout_logical_offset += component_count;
      return;
    }
  }
}

static iree_status_t loom_c_abi_signature_layout_build_side(
    const loom_func_c_abi_signature_t* signature, loom_i64_array_t roots,
    const uint16_t* type_leaf_counts, iree_arena_allocator_t* result_arena,
    const loom_c_abi_object_t** out_objects,
    const loom_c_abi_leaf_t** out_leaves, uint16_t* out_leaf_count,
    uint16_t* out_logical_count) {
  *out_objects = NULL;
  *out_leaves = NULL;
  *out_leaf_count = 0;
  *out_logical_count = 0;

  uint32_t leaf_count = 0;
  uint32_t logical_count = 0;
  for (iree_host_size_t i = 0; i < roots.count; ++i) {
    const uint32_t type_index = (uint32_t)roots.values[i];
    leaf_count += type_leaf_counts[type_index];
    logical_count += (uint32_t)loom_func_abi_c_type_attr_component_count(
        signature->types.values[type_index]);
    IREE_ASSERT_LE(leaf_count, UINT16_MAX);
    IREE_ASSERT_LE(logical_count, UINT16_MAX);
  }

  loom_c_abi_object_t* objects = NULL;
  loom_c_abi_leaf_t* leaves = NULL;
  IREE_RETURN_IF_ERROR(loom_c_abi_layout_allocate_array(
      result_arena, roots.count, sizeof(*objects), (void**)&objects));
  IREE_RETURN_IF_ERROR(loom_c_abi_layout_allocate_array(
      result_arena, leaf_count, sizeof(*leaves), (void**)&leaves));

  uint16_t next_leaf = 0;
  uint16_t next_logical = 0;
  for (iree_host_size_t i = 0; i < roots.count; ++i) {
    const uint32_t type_index = (uint32_t)roots.values[i];
    const loom_attribute_t type = signature->types.values[type_index];
    loom_c_abi_object_t* object = &objects[i];
    *object = (loom_c_abi_object_t){
        .type_index = type_index,
        .byte_length = (uint64_t)loom_func_abi_c_type_attr_byte_length(type),
        .byte_alignment =
            (uint64_t)loom_func_abi_c_type_attr_byte_alignment(type),
        .logical_offset = next_logical,
        .logical_count =
            (uint16_t)loom_func_abi_c_type_attr_component_count(type),
        .leaf_offset = next_leaf,
        .leaf_count = type_leaf_counts[type_index],
    };
    loom_c_abi_layout_build_leaves(signature, type_index,
                                   /*object_byte_offset=*/0,
                                   /*naturally_aligned=*/true, /*depth=*/0,
                                   leaves, &next_leaf, &next_logical);
    IREE_ASSERT_EQ(next_leaf,
                   (uint16_t)(object->leaf_offset + object->leaf_count));
    IREE_ASSERT_EQ(next_logical,
                   (uint16_t)(object->logical_offset + object->logical_count));
  }
  IREE_ASSERT_EQ(next_leaf, leaf_count);
  IREE_ASSERT_EQ(next_logical, logical_count);
  *out_objects = objects;
  *out_leaves = leaves;
  *out_leaf_count = next_leaf;
  *out_logical_count = next_logical;
  return iree_ok_status();
}

iree_status_t loom_c_abi_signature_layout_build(
    const loom_func_c_abi_signature_t* signature,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    loom_c_abi_signature_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(signature);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(result_arena);
  IREE_ASSERT_ARGUMENT(out_layout);
  *out_layout = (loom_c_abi_signature_layout_t){.signature = *signature};

  uint16_t* type_leaf_counts = NULL;
  IREE_RETURN_IF_ERROR(loom_c_abi_layout_allocate_array(
      scratch_arena, signature->types.count, sizeof(*type_leaf_counts),
      (void**)&type_leaf_counts));
  for (uint32_t i = 0; i < signature->types.count; ++i) {
    type_leaf_counts[i] =
        loom_c_abi_layout_type_leaf_count(signature, i, type_leaf_counts);
  }

  IREE_RETURN_IF_ERROR(loom_c_abi_signature_layout_build_side(
      signature, signature->arguments, type_leaf_counts, result_arena,
      &out_layout->arguments, &out_layout->argument_leaves,
      &out_layout->argument_leaf_count, &out_layout->argument_logical_count));
  return loom_c_abi_signature_layout_build_side(
      signature, signature->results, type_leaf_counts, result_arena,
      &out_layout->results, &out_layout->result_leaves,
      &out_layout->result_leaf_count, &out_layout->result_logical_count);
}
