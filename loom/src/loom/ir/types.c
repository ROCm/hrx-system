// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/structural_hash.h"

iree_status_t loom_type_function_build(const loom_type_t* arg_types,
                                       uint16_t arg_count,
                                       const loom_type_t* result_types,
                                       uint16_t result_count,
                                       iree_allocator_t allocator,
                                       loom_type_t* out_type) {
  // Use IREE_STRUCT_LAYOUT to overflow-check the total allocation size.
  // The two STRUCT_FIELD entries validate arg_count * sizeof(loom_type_t)
  // and result_count * sizeof(loom_type_t) independently, rejecting
  // overflow from untrusted bytecode counts before any allocation.
  iree_host_size_t alloc_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_func_type_data_t), &alloc_size,
      IREE_STRUCT_FIELD_FAM(arg_count, loom_type_t),
      IREE_STRUCT_FIELD(result_count, loom_type_t, /*out_offset=*/NULL)));
  loom_func_type_data_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&data));
  data->arg_count = arg_count;
  data->result_count = result_count;
  data->reserved = 0;
  if (arg_count > 0) {
    memcpy(data->types, arg_types, arg_count * sizeof(loom_type_t));
  }
  if (result_count > 0) {
    memcpy(data->types + arg_count, result_types,
           result_count * sizeof(loom_type_t));
  }
  *out_type = loom_type_function(data);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type equality
//===----------------------------------------------------------------------===//

// Reusing a completed query for the same immutable by-value representation is
// exact even when a type or attribute is caller-owned and has not been
// interned.
static bool loom_type_representation_equal(const loom_type_t* a,
                                           const loom_type_t* b) {
  return memcmp(a, b, sizeof(*a)) == 0;
}

static bool loom_type_sequence_equal(const loom_type_t* a_types,
                                     const loom_type_t* b_types,
                                     iree_host_size_t type_count) {
  // Identical immutable sequences need no recursive child comparison.
  if (type_count == 0 || a_types == b_types) {
    return true;
  }
  if (!a_types || !b_types) {
    return false;
  }
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    // A successful comparison of the immediately preceding immutable pair is
    // still valid under the same query. Avoid expanding repeated DAG edges as
    // independent trees.
    if (i > 0 && loom_type_representation_equal(&a_types[i], &a_types[i - 1]) &&
        loom_type_representation_equal(&b_types[i], &b_types[i - 1])) {
      continue;
    }
    if (!loom_type_equal(a_types[i], b_types[i])) {
      return false;
    }
  }
  return true;
}

bool loom_type_shape_equals(loom_type_t a, loom_type_t b) {
  uint8_t rank_a = loom_type_rank(a);
  if (rank_a != loom_type_rank(b)) {
    return false;
  }
  for (uint8_t i = 0; i < rank_a; ++i) {
    if (loom_type_dim(a, i) != loom_type_dim(b, i)) {
      return false;
    }
  }
  return true;
}

bool loom_type_equal(loom_type_t a, loom_type_t b) {
  if (a.header != b.header || a.encoding_id != b.encoding_id ||
      a.encoding_flags != b.encoding_flags) {
    return false;
  }
  loom_type_kind_t kind = loom_type_kind(a);
  if (!loom_type_kind_is_valid(kind)) {
    return a.dims[0] == b.dims[0] && a.dims[1] == b.dims[1];
  }
  switch (kind) {
    case LOOM_TYPE_REFERENCE:
      return a.dims[0] == b.dims[0];
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* a_data = loom_type_func_data(a);
      const loom_func_type_data_t* b_data = loom_type_func_data(b);
      if (!a_data || !b_data) {
        return a_data == b_data;
      }
      return a_data->arg_count == b_data->arg_count &&
             a_data->result_count == b_data->result_count &&
             loom_type_sequence_equal(
                 a_data->types, b_data->types,
                 (iree_host_size_t)a_data->arg_count + a_data->result_count);
    }
    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(a);
      return loom_type_dialect_name_id(a) == loom_type_dialect_name_id(b) &&
             loom_type_sequence_equal(loom_type_dialect_params(a),
                                      loom_type_dialect_params(b), param_count);
    }
    case LOOM_TYPE_PARAMETERIZED: {
      if (loom_type_parameterized_descriptor(a) !=
          loom_type_parameterized_descriptor(b)) {
        return false;
      }
      uint8_t parameter_count = loom_type_parameterized_parameter_count(a);
      const loom_attribute_t* a_parameters =
          loom_type_parameterized_parameters(a);
      const loom_attribute_t* b_parameters =
          loom_type_parameterized_parameters(b);
      if (parameter_count == 0) {
        return true;
      }
      if (!a_parameters || !b_parameters) {
        return a_parameters == b_parameters;
      }
      for (uint8_t i = 0; i < parameter_count; ++i) {
        if (!loom_attribute_equal(&a_parameters[i], &b_parameters[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_REGISTER: {
      const loom_register_type_data_t* a_data = loom_type_register_data(a);
      const loom_register_type_data_t* b_data = loom_type_register_data(b);
      if (loom_type_register_has_value_type(a)) {
        if (!a_data || !b_data) {
          return a_data == b_data;
        }
        if (a_data->carrier_payload0 != b_data->carrier_payload0 ||
            a_data->carrier_payload1 != b_data->carrier_payload1) {
          return false;
        }
        return loom_type_equal(a_data->value_type, b_data->value_type);
      }
      if (a.dims[0] != b.dims[0] || a.dims[1] != b.dims[1]) {
        return false;
      }
      return true;
    }
    default:
      break;
  }
  if (loom_type_has_inline_dims(a)) {
    return a.dims[0] == b.dims[0] && a.dims[1] == b.dims[1];
  }
  // Overflow: compare each dim via the overflow pointer.
  uint8_t rank = loom_type_rank(a);
  const loom_overflow_dim_t* a_dims =
      (const loom_overflow_dim_t*)(uintptr_t)a.dims[0];
  const loom_overflow_dim_t* b_dims =
      (const loom_overflow_dim_t*)(uintptr_t)b.dims[0];
  if (rank == 0 || !a_dims || !b_dims) {
    return a_dims == b_dims;
  }
  for (uint8_t i = 0; i < rank; ++i) {
    if (a_dims[i] != b_dims[i]) {
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Shaped type queries
//===----------------------------------------------------------------------===//

bool loom_type_has_static_zero_extent(loom_type_t type) {
  if (!loom_type_is_shaped(type)) {
    return false;
  }
  for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) {
      continue;
    }
    if (loom_type_dim_static_size_at(type, i) == 0) {
      return true;
    }
  }
  return false;
}

bool loom_type_static_element_count(loom_type_t type,
                                    uint64_t* out_element_count) {
  *out_element_count = 0;
  if (!loom_type_is_shaped(type)) {
    return false;
  }
  if (!loom_type_is_all_static(type)) {
    return false;
  }

  uint64_t element_count = 1;
  for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
    int64_t dimension_size = loom_type_dim_static_size_at(type, i);
    if (dimension_size == 0) {
      *out_element_count = 0;
      return true;
    }
    if (dimension_size < 0 ||
        element_count > UINT64_MAX / (uint64_t)dimension_size) {
      return false;
    }
    element_count *= (uint64_t)dimension_size;
  }
  *out_element_count = element_count;
  return true;
}

//===----------------------------------------------------------------------===//
// Type hashing
//===----------------------------------------------------------------------===//

static uint32_t loom_type_hash_mix_sequence(uint32_t hash,
                                            const loom_type_t* types,
                                            iree_host_size_t type_count) {
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)type_count);
  if (!types) {
    return hash;
  }
  uint32_t element_hash = 0;
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    if (i == 0 || !loom_type_representation_equal(&types[i], &types[i - 1])) {
      element_hash = loom_type_hash(types[i]);
    }
    hash = loom_structural_hash_mix_u32(hash, element_hash);
  }
  return hash;
}

uint32_t loom_type_hash(loom_type_t type) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(hash, type.header);
  hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
  hash = loom_structural_hash_mix_u16(hash, type.encoding_flags);

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
    hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
    return loom_structural_hash_finalize(hash);
  }
  switch (kind) {
    case LOOM_TYPE_REFERENCE:
      hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
      return loom_structural_hash_finalize(hash);
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return loom_structural_hash_finalize(hash);
      }
      hash = loom_structural_hash_mix_u16(hash, data->arg_count);
      hash = loom_structural_hash_mix_u16(hash, data->result_count);
      hash = loom_type_hash_mix_sequence(
          hash, data->types,
          (iree_host_size_t)data->arg_count + data->result_count);
      return loom_structural_hash_finalize(hash);
    }
    case LOOM_TYPE_DIALECT:
      hash =
          loom_structural_hash_mix_u32(hash, loom_type_dialect_name_id(type));
      hash = loom_type_hash_mix_sequence(hash, loom_type_dialect_params(type),
                                         loom_type_dialect_param_count(type));
      return loom_structural_hash_finalize(hash);
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      hash =
          loom_structural_hash_mix_u64(hash, (uint64_t)(uintptr_t)descriptor);
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      if (!parameters) {
        return loom_structural_hash_finalize(hash);
      }
      for (uint8_t i = 0; i < parameter_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_attribute_hash(&parameters[i]));
      }
      return loom_structural_hash_finalize(hash);
    }
    case LOOM_TYPE_REGISTER: {
      const loom_register_type_data_t* data = loom_type_register_data(type);
      if (loom_type_register_has_value_type(type)) {
        if (!data) {
          return loom_structural_hash_finalize(hash);
        }
        hash = loom_structural_hash_mix_u64(hash, data->carrier_payload0);
        hash = loom_structural_hash_mix_u64(hash, data->carrier_payload1);
        hash = loom_structural_hash_mix_u32(hash,
                                            loom_type_hash(data->value_type));
        return loom_structural_hash_finalize(hash);
      }
      hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
      hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
      return loom_structural_hash_finalize(hash);
    }
    default:
      break;
  }

  if (loom_type_has_inline_dims(type)) {
    hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
    hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
    return loom_structural_hash_finalize(hash);
  }

  uint8_t rank = loom_type_rank(type);
  const loom_overflow_dim_t* dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  if (rank == 0 || !dims) {
    return loom_structural_hash_finalize(hash);
  }
  for (uint8_t i = 0; i < rank; ++i) {
    hash = loom_structural_hash_mix_u64(hash, dims[i]);
  }
  return loom_structural_hash_finalize(hash);
}
