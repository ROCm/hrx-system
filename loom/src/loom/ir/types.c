// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
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

bool loom_type_is_module_independent(loom_type_t type) {
  const loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return false;
  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      return data && loom_type_sequence_is_module_independent(
                         data->types, (iree_host_size_t)data->arg_count +
                                          data->result_count);
    }
    case LOOM_TYPE_DIALECT:
    case LOOM_TYPE_PARAMETERIZED:
      return false;
    case LOOM_TYPE_REGISTER: {
      if (!loom_type_register_has_value_type(type)) return true;
      const loom_type_t* value_type = loom_type_register_value_type(type);
      return value_type && loom_type_is_module_independent(*value_type);
    }
    default:
      return !loom_type_has_static_encoding(type);
  }
}

bool loom_type_sequence_is_module_independent(const loom_type_t* types,
                                              iree_host_size_t type_count) {
  if (type_count > 0 && !types) return false;
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    if (!loom_type_is_module_independent(types[i])) return false;
  }
  return true;
}

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
      if (parameter_count == 0) return true;
      if (!a_parameters || !b_parameters) return a_parameters == b_parameters;
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
        if (!a_data || !b_data) return a_data == b_data;
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
  if (rank == 0 || !a_dims || !b_dims) return a_dims == b_dims;
  for (uint8_t i = 0; i < rank; ++i) {
    if (a_dims[i] != b_dims[i]) return false;
  }
  return true;
}

static loom_value_id_t loom_type_remap_value(
    const loom_type_value_remap_t* remap, loom_value_id_t value_id) {
  for (const loom_type_value_remap_t* span = remap; span; span = span->next) {
    for (iree_host_size_t i = 0; i < span->count; ++i) {
      if (span->source_values[i] == value_id) {
        return span->target_values[i];
      }
    }
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
    const loom_module_t* module, const loom_type_t* source_types,
    const loom_type_t* target_types, uint16_t type_count,
    const loom_type_value_remap_t* remap) {
  if (type_count == 0) return true;
  if (!source_types || !target_types) return source_types == target_types;
  for (uint16_t i = 0; i < type_count; ++i) {
    if (!loom_type_equal_after_value_remap(module, source_types[i],
                                           target_types[i], remap)) {
      return false;
    }
  }
  return true;
}

static bool loom_attribute_equal_after_value_remap(
    const loom_module_t* module, loom_attribute_t source_attr,
    loom_attribute_t target_attr, uint8_t depth,
    const loom_type_value_remap_t* remap) {
  if (source_attr.kind != target_attr.kind) return false;
  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_TYPE:
      if (source_attr.type_id == LOOM_TYPE_ID_INVALID ||
          target_attr.type_id == LOOM_TYPE_ID_INVALID ||
          source_attr.type_id >= module->types.count ||
          target_attr.type_id >= module->types.count) {
        return false;
      }
      return loom_type_equal_after_value_remap(
          module, module->types.entries[source_attr.type_id],
          module->types.entries[target_attr.type_id], remap);

    case LOOM_ATTR_PREDICATE_LIST:
      if (source_attr.count != target_attr.count ||
          (source_attr.count > 0 &&
           (!source_attr.predicate_list || !target_attr.predicate_list))) {
        return false;
      }
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        const loom_predicate_t* source = &source_attr.predicate_list[i];
        const loom_predicate_t* target = &target_attr.predicate_list[i];
        if (source->kind != target->kind ||
            source->arg_count != target->arg_count ||
            memcmp(source->arg_tags, target->arg_tags,
                   sizeof(source->arg_tags)) != 0) {
          return false;
        }
        for (uint8_t j = 0; j < source->arg_count; ++j) {
          if (source->arg_tags[j] == LOOM_PRED_ARG_VALUE) {
            if (source->args[j] < 0 || target->args[j] < 0 ||
                (loom_value_id_t)target->args[j] !=
                    loom_type_remap_value(remap,
                                          (loom_value_id_t)source->args[j])) {
              return false;
            }
          } else if (source->args[j] != target->args[j]) {
            return false;
          }
        }
      }
      return true;

    case LOOM_ATTR_DICT:
      if (source_attr.count != target_attr.count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (source_attr.count > 0 &&
           (!source_attr.dict_entries || !target_attr.dict_entries))) {
        return false;
      }
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        if (source_attr.dict_entries[i].name_id !=
                target_attr.dict_entries[i].name_id ||
            !loom_attribute_equal_after_value_remap(
                module, source_attr.dict_entries[i].value,
                target_attr.dict_entries[i].value, (uint8_t)(depth + 1),
                remap)) {
          return false;
        }
      }
      return true;

    case LOOM_ATTR_PARAMETERIZED:
      if (source_attr.reserved_1 != target_attr.reserved_1 ||
          source_attr.count != target_attr.count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (source_attr.count > 0 && (!source_attr.parameterized_slots ||
                                     !target_attr.parameterized_slots))) {
        return false;
      }
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        if (!loom_attribute_equal_after_value_remap(
                module, source_attr.parameterized_slots[i],
                target_attr.parameterized_slots[i], (uint8_t)(depth + 1),
                remap)) {
          return false;
        }
      }
      return true;

    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (source_attr.count != target_attr.count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (source_attr.count > 0 && (!source_attr.parameterized_array ||
                                     !target_attr.parameterized_array))) {
        return false;
      }
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        if (!loom_attribute_equal_after_value_remap(
                module, source_attr.parameterized_array[i],
                target_attr.parameterized_array[i], (uint8_t)(depth + 1),
                remap)) {
          return false;
        }
      }
      return true;

    default:
      return loom_attribute_equal(&source_attr, &target_attr);
  }
}

bool loom_type_equal_after_value_remap(const loom_module_t* module,
                                       loom_type_t source_type,
                                       loom_type_t target_type,
                                       const loom_type_value_remap_t* remap) {
  if (!module) return false;
  for (const loom_type_value_remap_t* span = remap; span; span = span->next) {
    if (span->count > 0 && (!span->source_values || !span->target_values)) {
      return false;
    }
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
                 module, source_data->types, target_data->types, type_count,
                 remap);
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
                 module, loom_type_dialect_params(source_type),
                 loom_type_dialect_params(target_type), param_count, remap);
    }

    case LOOM_TYPE_REGISTER: {
      if (source_type.header != target_type.header ||
          source_type.encoding_id != target_type.encoding_id ||
          source_type.encoding_flags != target_type.encoding_flags) {
        return false;
      }
      if (!loom_type_register_has_value_type(source_type)) {
        return source_type.dims[0] == target_type.dims[0] &&
               source_type.dims[1] == target_type.dims[1];
      }
      const loom_register_type_data_t* source_data =
          loom_type_register_data(source_type);
      const loom_register_type_data_t* target_data =
          loom_type_register_data(target_type);
      if (!source_data || !target_data) return source_data == target_data;
      return source_data->carrier_payload0 == target_data->carrier_payload0 &&
             source_data->carrier_payload1 == target_data->carrier_payload1 &&
             loom_type_equal_after_value_remap(module, source_data->value_type,
                                               target_data->value_type, remap);
    }

    case LOOM_TYPE_PARAMETERIZED: {
      if (source_type.header != target_type.header ||
          source_type.encoding_flags != target_type.encoding_flags ||
          loom_type_parameterized_descriptor(source_type) !=
              loom_type_parameterized_descriptor(target_type)) {
        return false;
      }
      uint8_t parameter_count =
          loom_type_parameterized_parameter_count(source_type);
      const loom_attribute_t* source_parameters =
          loom_type_parameterized_parameters(source_type);
      const loom_attribute_t* target_parameters =
          loom_type_parameterized_parameters(target_type);
      if (parameter_count == 0) return true;
      if (!source_parameters || !target_parameters) {
        return source_parameters == target_parameters;
      }
      for (uint8_t i = 0; i < parameter_count; ++i) {
        if (!loom_attribute_equal_after_value_remap(
                module, source_parameters[i], target_parameters[i],
                /*depth=*/1, remap)) {
          return false;
        }
      }
      return true;
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
    const loom_module_t* module, const loom_type_t* types, uint16_t type_count,
    loom_type_value_ref_callback_t callback, void* user_data) {
  if (!types) return iree_ok_status();
  for (uint16_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_type_walk_value_refs(module, types[i], callback, user_data));
  }
  return iree_ok_status();
}

iree_status_t loom_type_walk_value_refs(const loom_module_t* module,
                                        loom_type_t type,
                                        loom_type_value_ref_callback_t callback,
                                        void* user_data) {
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is NULL");
  }
  if (!callback) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type value reference callback is NULL");
  }

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return iree_ok_status();
  if (!loom_type_may_reference_values(type)) return iree_ok_status();

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return iree_ok_status();
      return loom_type_walk_value_ref_sequence(
          module, data->types, (uint16_t)(data->arg_count + data->result_count),
          callback, user_data);
    }

    case LOOM_TYPE_DIALECT:
      return loom_type_walk_value_ref_sequence(
          module, loom_type_dialect_params(type),
          loom_type_dialect_param_count(type), callback, user_data);

    case LOOM_TYPE_PARAMETERIZED: {
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      for (uint8_t i = 0; i < parameter_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_walk_attribute_value_refs(
            module, parameters[i], callback, user_data));
      }
      return iree_ok_status();
    }

    case LOOM_TYPE_REGISTER: {
      const loom_type_t* value_type = loom_type_register_value_type(type);
      return value_type ? loom_type_walk_value_refs(module, *value_type,
                                                    callback, user_data)
                        : iree_ok_status();
    }

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
                                                const loom_module_t* module,
                                                loom_value_id_t value_id) {
  if (!types) return false;
  for (uint16_t i = 0; i < type_count; ++i) {
    if (loom_type_references_value(module, types[i], value_id)) return true;
  }
  return false;
}

static bool loom_attribute_references_value(const loom_module_t* module,
                                            loom_attribute_t attr,
                                            uint8_t depth,
                                            loom_value_id_t value_id) {
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_TYPE:
      return attr.type_id != LOOM_TYPE_ID_INVALID &&
             attr.type_id < module->types.count &&
             loom_type_references_value(
                 module, module->types.entries[attr.type_id], value_id);
    case LOOM_ATTR_PREDICATE_LIST:
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        for (uint8_t j = 0; j < predicate->arg_count; ++j) {
          if (predicate->arg_tags[j] == LOOM_PRED_ARG_VALUE &&
              (loom_value_id_t)predicate->args[j] == value_id) {
            return true;
          }
        }
      }
      return false;
    case LOOM_ATTR_DICT:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) return false;
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (loom_attribute_references_value(module, attr.dict_entries[i].value,
                                            (uint8_t)(depth + 1), value_id)) {
          return true;
        }
      }
      return false;
    case LOOM_ATTR_PARAMETERIZED:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) return false;
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (loom_attribute_references_value(module, attr.parameterized_slots[i],
                                            (uint8_t)(depth + 1), value_id)) {
          return true;
        }
      }
      return false;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) return false;
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (loom_attribute_references_value(module, attr.parameterized_array[i],
                                            (uint8_t)(depth + 1), value_id)) {
          return true;
        }
      }
      return false;
    default:
      return false;
  }
}

bool loom_type_references_value(const loom_module_t* module, loom_type_t type,
                                loom_value_id_t value_id) {
  if (!module) return false;
  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return false;
  if (!loom_type_may_reference_values(type)) return false;

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return false;
      return loom_type_sequence_references_value(
          data->types, (uint16_t)(data->arg_count + data->result_count), module,
          value_id);
    }

    case LOOM_TYPE_DIALECT:
      return loom_type_sequence_references_value(
          loom_type_dialect_params(type), loom_type_dialect_param_count(type),
          module, value_id);

    case LOOM_TYPE_PARAMETERIZED: {
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      for (uint8_t i = 0; i < parameter_count; ++i) {
        if (loom_attribute_references_value(module, parameters[i], /*depth=*/1,
                                            value_id)) {
          return true;
        }
      }
      return false;
    }

    case LOOM_TYPE_REGISTER: {
      const loom_type_t* value_type = loom_type_register_value_type(type);
      return value_type &&
             loom_type_references_value(module, *value_type, value_id);
    }

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

static uint32_t loom_type_hash_mix_sequence(uint32_t hash,
                                            const loom_type_t* types,
                                            uint16_t type_count) {
  hash = loom_structural_hash_mix_u16(hash, type_count);
  if (!types) return hash;
  for (uint16_t i = 0; i < type_count; ++i) {
    uint32_t element_hash = loom_type_hash(types[i]);
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
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return loom_structural_hash_finalize(hash);
      hash = loom_structural_hash_mix_u16(hash, data->arg_count);
      hash = loom_structural_hash_mix_u16(hash, data->result_count);
      hash = loom_type_hash_mix_sequence(
          hash, data->types, (uint16_t)(data->arg_count + data->result_count));
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
      if (!parameters) return loom_structural_hash_finalize(hash);
      for (uint8_t i = 0; i < parameter_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_attribute_hash(&parameters[i]));
      }
      return loom_structural_hash_finalize(hash);
    }
    case LOOM_TYPE_REGISTER: {
      const loom_register_type_data_t* data = loom_type_register_data(type);
      if (loom_type_register_has_value_type(type)) {
        if (!data) return loom_structural_hash_finalize(hash);
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
  if (rank == 0 || !dims) return loom_structural_hash_finalize(hash);
  for (uint8_t i = 0; i < rank; ++i) {
    hash = loom_structural_hash_mix_u64(hash, dims[i]);
  }
  return loom_structural_hash_finalize(hash);
}
