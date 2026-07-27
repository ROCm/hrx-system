// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include <string.h>

const char* loom_group_scope_name(loom_group_scope_t scope) {
  static const char* const names[] = {
      [LOOM_GROUP_SCOPE_WORKGROUP] = "workgroup",
      [LOOM_GROUP_SCOPE_SUBGROUP] = "subgroup",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_GROUP_SCOPE_COUNT_,
                "group scope names out of sync with enum");
  if (scope < LOOM_GROUP_SCOPE_COUNT_) return names[scope];
  return NULL;
}

const char* loom_encoding_role_name(loom_encoding_role_t role) {
  static const char* const names[] = {
      [LOOM_ENCODING_ROLE_UNKNOWN] = "",
      [LOOM_ENCODING_ROLE_ADDRESS_LAYOUT] = "layout",
      [LOOM_ENCODING_ROLE_STORAGE_SCHEMA] = "schema",
      [LOOM_ENCODING_ROLE_PHYSICAL_STORAGE] = "storage",
      [LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM] = "transform",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_ENCODING_ROLE_COUNT_,
                "encoding role names out of sync with enum");
  if (loom_encoding_role_is_valid(role)) return names[role];
  return NULL;
}

bool loom_encoding_role_parse(iree_string_view_t text,
                              loom_encoding_role_t* out_role) {
  if (iree_string_view_equal(text, IREE_SV("layout"))) {
    *out_role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("schema"))) {
    *out_role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("storage"))) {
    *out_role = LOOM_ENCODING_ROLE_PHYSICAL_STORAGE;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("transform"))) {
    *out_role = LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM;
    return true;
  }
  return false;
}

const char* loom_storage_space_name(loom_storage_space_t space) {
  static const char* const names[] = {
      [LOOM_STORAGE_SPACE_STACK] = "stack",
      [LOOM_STORAGE_SPACE_SCRATCH] = "scratch",
      [LOOM_STORAGE_SPACE_PRIVATE] = "private",
      [LOOM_STORAGE_SPACE_WORKGROUP] = "workgroup",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_STORAGE_SPACE_COUNT_,
                "storage space names out of sync with enum");
  if (loom_storage_space_is_valid(space)) return names[space];
  return NULL;
}

bool loom_storage_space_parse(iree_string_view_t text,
                              loom_storage_space_t* out_space) {
  if (iree_string_view_equal(text, IREE_SV("stack"))) {
    *out_space = LOOM_STORAGE_SPACE_STACK;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("scratch"))) {
    *out_space = LOOM_STORAGE_SPACE_SCRATCH;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("private"))) {
    *out_space = LOOM_STORAGE_SPACE_PRIVATE;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("workgroup"))) {
    *out_space = LOOM_STORAGE_SPACE_WORKGROUP;
    return true;
  }
  return false;
}

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

static bool loom_type_sequence_equal(const loom_type_t* a_types,
                                     const loom_type_t* b_types,
                                     uint16_t type_count) {
  if (type_count == 0) return true;
  if (!a_types || !b_types) return a_types == b_types;
  for (uint16_t i = 0; i < type_count; ++i) {
    if (!loom_type_equal(a_types[i], b_types[i])) return false;
  }
  return true;
}

bool loom_type_shape_equals(loom_type_t a, loom_type_t b) {
  uint8_t rank_a = loom_type_rank(a);
  if (rank_a != loom_type_rank(b)) return false;
  for (uint8_t i = 0; i < rank_a; ++i) {
    if (loom_type_dim(a, i) != loom_type_dim(b, i)) return false;
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
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* a_data = loom_type_func_data(a);
      const loom_func_type_data_t* b_data = loom_type_func_data(b);
      if (!a_data || !b_data) return a_data == b_data;
      return a_data->arg_count == b_data->arg_count &&
             a_data->result_count == b_data->result_count &&
             loom_type_sequence_equal(
                 a_data->types, b_data->types,
                 (uint16_t)(a_data->arg_count + a_data->result_count));
    }
    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(a);
      return loom_type_dialect_name_id(a) == loom_type_dialect_name_id(b) &&
             loom_type_sequence_equal(loom_type_dialect_params(a),
                                      loom_type_dialect_params(b), param_count);
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
  if (rank == 0 || !a_dims || !b_dims) return a_dims == b_dims;
  for (uint8_t i = 0; i < rank; ++i) {
    if (a_dims[i] != b_dims[i]) return false;
  }
  return true;
}

static loom_value_id_t loom_type_remap_value(
    const loom_type_value_remap_t* remap, loom_value_id_t value_id) {
  if (!remap) return value_id;
  for (iree_host_size_t i = 0; i < remap->count; ++i) {
    if (remap->source_values[i] == value_id) return remap->target_values[i];
  }
  return value_id;
}

static bool loom_type_dim_equal_after_value_remap(
    uint64_t source_dim, uint64_t target_dim,
    const loom_type_value_remap_t* remap) {
  if (!loom_dim_is_dynamic(source_dim)) return source_dim == target_dim;
  loom_value_id_t remapped_value =
      loom_type_remap_value(remap, loom_dim_value_id(source_dim));
  return loom_dim_pack_dynamic(remapped_value) == target_dim;
}

static bool loom_type_encoding_equal_after_value_remap(
    loom_type_t source_type, loom_type_t target_type,
    const loom_type_value_remap_t* remap) {
  if (source_type.encoding_flags != target_type.encoding_flags) return false;
  if (!loom_type_has_ssa_encoding(source_type)) {
    return source_type.encoding_id == target_type.encoding_id;
  }
  loom_value_id_t remapped_value = loom_type_remap_value(
      remap, (loom_value_id_t)loom_type_encoding_value_id(source_type));
  if (remapped_value > UINT16_MAX) return false;
  return (uint16_t)remapped_value == target_type.encoding_id;
}

static bool loom_type_sequence_equal_after_value_remap(
    const loom_type_t* source_types, const loom_type_t* target_types,
    uint16_t type_count, const loom_type_value_remap_t* remap) {
  if (type_count == 0) return true;
  if (!source_types || !target_types) return source_types == target_types;
  for (uint16_t i = 0; i < type_count; ++i) {
    if (!loom_type_equal_after_value_remap(source_types[i], target_types[i],
                                           remap)) {
      return false;
    }
  }
  return true;
}

bool loom_type_equal_after_value_remap(loom_type_t source_type,
                                       loom_type_t target_type,
                                       const loom_type_value_remap_t* remap) {
  if (remap && remap->count > 0 &&
      (!remap->source_values || !remap->target_values)) {
    return false;
  }

  loom_type_kind_t source_kind = loom_type_kind(source_type);
  if (source_kind != loom_type_kind(target_type)) return false;
  if (!loom_type_kind_is_valid(source_kind)) {
    return source_type.dims[0] == target_type.dims[0] &&
           source_type.dims[1] == target_type.dims[1];
  }

  switch (source_kind) {
    case LOOM_TYPE_FUNCTION: {
      if (source_type.header != target_type.header ||
          source_type.encoding_id != target_type.encoding_id ||
          source_type.encoding_flags != target_type.encoding_flags) {
        return false;
      }
      const loom_func_type_data_t* source_data =
          loom_type_func_data(source_type);
      const loom_func_type_data_t* target_data =
          loom_type_func_data(target_type);
      if (!source_data || !target_data) return source_data == target_data;
      uint16_t type_count =
          (uint16_t)(source_data->arg_count + source_data->result_count);
      return source_data->arg_count == target_data->arg_count &&
             source_data->result_count == target_data->result_count &&
             loom_type_sequence_equal_after_value_remap(
                 source_data->types, target_data->types, type_count, remap);
    }

    case LOOM_TYPE_DIALECT: {
      if (source_type.header != target_type.header ||
          source_type.encoding_id != target_type.encoding_id ||
          loom_type_dialect_name_id(source_type) !=
              loom_type_dialect_name_id(target_type)) {
        return false;
      }
      uint16_t param_count = loom_type_dialect_param_count(source_type);
      return param_count == loom_type_dialect_param_count(target_type) &&
             loom_type_sequence_equal_after_value_remap(
                 loom_type_dialect_params(source_type),
                 loom_type_dialect_params(target_type), param_count, remap);
    }

    default:
      break;
  }

  if (loom_type_is_shaped(source_type) || loom_type_is_pool(source_type)) {
    if (loom_type_element_type(source_type) !=
            loom_type_element_type(target_type) ||
        loom_type_rank(source_type) != loom_type_rank(target_type) ||
        loom_type_flags(source_type) != loom_type_flags(target_type)) {
      return false;
    }
    uint8_t rank = loom_type_rank(source_type);
    for (uint8_t i = 0; i < rank; ++i) {
      if (!loom_type_dim_equal_after_value_remap(loom_type_dim(source_type, i),
                                                 loom_type_dim(target_type, i),
                                                 remap)) {
        return false;
      }
    }
    return loom_type_encoding_equal_after_value_remap(source_type, target_type,
                                                      remap);
  }

  return source_type.header == target_type.header &&
         source_type.encoding_id == target_type.encoding_id &&
         source_type.encoding_flags == target_type.encoding_flags &&
         source_type.dims[0] == target_type.dims[0] &&
         source_type.dims[1] == target_type.dims[1];
}

//===----------------------------------------------------------------------===//
// Shaped type queries
//===----------------------------------------------------------------------===//

bool loom_type_has_static_zero_extent(loom_type_t type) {
  if (!loom_type_is_shaped(type)) return false;
  for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) continue;
    if (loom_type_dim_static_size_at(type, i) == 0) return true;
  }
  return false;
}

bool loom_type_static_element_count(loom_type_t type,
                                    uint64_t* out_element_count) {
  *out_element_count = 0;
  if (!loom_type_is_shaped(type)) return false;
  if (!loom_type_is_all_static(type)) return false;

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
// Type SSA reference walking
//===----------------------------------------------------------------------===//

static bool loom_type_has_value_ref_dims(loom_type_t type) {
  return loom_type_is_shaped(type) || loom_type_is_pool(type);
}

static iree_status_t loom_type_walk_value_ref_sequence(
    const loom_type_t* types, uint16_t type_count,
    loom_type_value_ref_callback_t callback, void* user_data) {
  if (!types) return iree_ok_status();
  for (uint16_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_type_walk_value_refs(types[i], callback, user_data));
  }
  return iree_ok_status();
}

iree_status_t loom_type_walk_value_refs(loom_type_t type,
                                        loom_type_value_ref_callback_t callback,
                                        void* user_data) {
  if (!callback) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type value reference callback is NULL");
  }

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return iree_ok_status();

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return iree_ok_status();
      return loom_type_walk_value_ref_sequence(
          data->types, (uint16_t)(data->arg_count + data->result_count),
          callback, user_data);
    }

    case LOOM_TYPE_DIALECT:
      return loom_type_walk_value_ref_sequence(
          loom_type_dialect_params(type), loom_type_dialect_param_count(type),
          callback, user_data);

    default:
      break;
  }

  if (loom_type_has_value_ref_dims(type)) {
    for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
      if (!loom_type_dim_is_dynamic_at(type, i)) continue;
      IREE_RETURN_IF_ERROR(
          callback(loom_type_dim_value_id_at(type, i), user_data));
    }
  }
  if (loom_type_has_ssa_encoding(type)) {
    IREE_RETURN_IF_ERROR(
        callback(loom_type_encoding_value_id(type), user_data));
  }
  return iree_ok_status();
}

static bool loom_type_sequence_references_value(const loom_type_t* types,
                                                uint16_t type_count,
                                                loom_value_id_t value_id) {
  if (!types) return false;
  for (uint16_t i = 0; i < type_count; ++i) {
    if (loom_type_references_value(types[i], value_id)) return true;
  }
  return false;
}

bool loom_type_references_value(loom_type_t type, loom_value_id_t value_id) {
  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return false;

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return false;
      return loom_type_sequence_references_value(
          data->types, (uint16_t)(data->arg_count + data->result_count),
          value_id);
    }

    case LOOM_TYPE_DIALECT:
      return loom_type_sequence_references_value(
          loom_type_dialect_params(type), loom_type_dialect_param_count(type),
          value_id);

    default:
      break;
  }

  if (loom_type_has_value_ref_dims(type)) {
    for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
      if (!loom_type_dim_is_dynamic_at(type, i)) continue;
      if (loom_type_dim_value_id_at(type, i) == value_id) return true;
    }
  }
  return loom_type_has_ssa_encoding(type) &&
         loom_type_encoding_value_id(type) == value_id;
}

//===----------------------------------------------------------------------===//
// Type hashing
//===----------------------------------------------------------------------===//

static uint32_t loom_type_hash_mix_bytes(uint32_t hash, const void* data,
                                         iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_type_hash_mix_u16(uint32_t hash, uint16_t value) {
  return loom_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_type_hash_mix_u32(uint32_t hash, uint32_t value) {
  return loom_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_type_hash_mix_u64(uint32_t hash, uint64_t value) {
  return loom_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_type_hash_mix_sequence(uint32_t hash,
                                            const loom_type_t* types,
                                            uint16_t type_count) {
  hash = loom_type_hash_mix_u16(hash, type_count);
  if (!types) return hash;
  for (uint16_t i = 0; i < type_count; ++i) {
    uint32_t element_hash = loom_type_hash(types[i]);
    hash = loom_type_hash_mix_u32(hash, element_hash);
  }
  return hash;
}

uint32_t loom_type_hash(loom_type_t type) {
  uint32_t hash = 2166136261u;
  hash = loom_type_hash_mix_u32(hash, type.header);
  hash = loom_type_hash_mix_u16(hash, type.encoding_id);
  hash = loom_type_hash_mix_u16(hash, type.encoding_flags);

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    hash = loom_type_hash_mix_u64(hash, type.dims[0]);
    return loom_type_hash_mix_u64(hash, type.dims[1]);
  }
  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return hash;
      hash = loom_type_hash_mix_u16(hash, data->arg_count);
      hash = loom_type_hash_mix_u16(hash, data->result_count);
      return loom_type_hash_mix_sequence(
          hash, data->types, (uint16_t)(data->arg_count + data->result_count));
    }
    case LOOM_TYPE_DIALECT:
      hash = loom_type_hash_mix_u32(hash, loom_type_dialect_name_id(type));
      return loom_type_hash_mix_sequence(hash, loom_type_dialect_params(type),
                                         loom_type_dialect_param_count(type));
    default:
      break;
  }

  if (loom_type_has_inline_dims(type)) {
    hash = loom_type_hash_mix_u64(hash, type.dims[0]);
    return loom_type_hash_mix_u64(hash, type.dims[1]);
  }

  uint8_t rank = loom_type_rank(type);
  const loom_overflow_dim_t* dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  if (rank == 0 || !dims) return hash;
  for (uint8_t i = 0; i < rank; ++i) {
    hash = loom_type_hash_mix_u64(hash, dims[i]);
  }
  return hash;
}
