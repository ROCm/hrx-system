// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_refs.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"

// Reusing a completed query for the same immutable by-value representation is
// exact even when a type or attribute is caller-owned and has not been
// interned.
static bool loom_type_representation_equal(const loom_type_t* a,
                                           const loom_type_t* b) {
  return memcmp(a, b, sizeof(*a)) == 0;
}

static bool loom_attribute_representation_equal(const loom_attribute_t* a,
                                                const loom_attribute_t* b) {
  return memcmp(a, b, sizeof(*a)) == 0;
}

//===----------------------------------------------------------------------===//
// Type reference remapping
//===----------------------------------------------------------------------===//

loom_value_id_t loom_type_remap_value(const loom_module_t* module,
                                      const loom_type_value_remap_t* remap,
                                      loom_value_id_t value_id) {
  for (const loom_type_value_remap_t* span = remap; span; span = span->next) {
    if (iree_any_bit_set(span->flags,
                         LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE)) {
      if (span->count == 0 || value_id >= module->values.count) {
        continue;
      }
      IREE_ASSERT(span->source_values[0] < module->values.count);
      const loom_value_t* first_value =
          loom_module_value(module, span->source_values[0]);
      const loom_value_t* value = loom_module_value(module, value_id);
      if (loom_value_is_block_arg(first_value) !=
          loom_value_is_block_arg(value)) {
        continue;
      }
      uint16_t first_index = loom_value_def_index(first_value);
      uint16_t value_index = loom_value_def_index(value);
      if (loom_value_is_block_arg(first_value)) {
        if (loom_value_def_block(first_value) != loom_value_def_block(value)) {
          continue;
        }
      } else {
        const loom_op_t* owner_op = loom_value_def_op(first_value);
        if (owner_op) {
          if (owner_op != loom_value_def_op(value)) {
            continue;
          }
        } else {
          // Declaration operands and results have independent index domains.
          // The declaration's ordinary operand link owns its argument index.
          IREE_ASSERT_EQ(first_value->use_count, 1);
          if (loom_value_def_op(value) || value->use_count != 1) {
            continue;
          }
          const loom_use_t first_use = loom_value_uses(first_value)[0];
          const loom_use_t value_use = loom_value_uses(value)[0];
          if (loom_use_user_op(first_use) != loom_use_user_op(value_use)) {
            continue;
          }
          first_index = loom_use_operand_index(first_use);
          value_index = loom_use_operand_index(value_use);
        }
      }
      if (value_index < first_index) {
        continue;
      }
      const uint16_t span_index = (uint16_t)(value_index - first_index);
      if (span_index >= span->count) {
        continue;
      }
      IREE_ASSERT(span->source_values[span_index] == value_id);
      return span->target_values[span_index];
    } else {
      for (uint16_t i = 0; i < span->count; ++i) {
        if (span->source_values[i] == value_id) {
          return span->target_values[i];
        }
      }
    }
  }
  return value_id;
}

static bool loom_type_dim_equal_after_value_remap(
    const loom_module_t* module, uint64_t source_dim, uint64_t target_dim,
    const loom_type_value_remap_t* remap) {
  if (!loom_dim_is_dynamic(source_dim)) {
    return source_dim == target_dim;
  }
  loom_value_id_t remapped_value =
      loom_type_remap_value(module, remap, loom_dim_value_id(source_dim));
  return loom_dim_pack_dynamic(remapped_value) == target_dim;
}

bool loom_type_shape_equal_after_value_remap(
    const loom_module_t* module, loom_type_t source_type,
    loom_type_t target_type, const loom_type_value_remap_t* remap) {
  const loom_type_kind_t source_kind = loom_type_kind(source_type);
  if (source_kind != loom_type_kind(target_type)) {
    return false;
  }
  if (source_kind == LOOM_TYPE_SCALAR) {
    return true;
  }
  if (!loom_type_is_shaped(source_type) || !loom_type_is_shaped(target_type) ||
      loom_type_rank(source_type) != loom_type_rank(target_type)) {
    return false;
  }
  const uint8_t rank = loom_type_rank(source_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (!loom_type_dim_equal_after_value_remap(
            module, loom_type_dim(source_type, i),
            loom_type_dim(target_type, i), remap)) {
      return false;
    }
  }
  return true;
}

static bool loom_type_encoding_equal_after_value_remap(
    const loom_module_t* module, loom_type_t source_type,
    loom_type_t target_type, const loom_type_value_remap_t* remap) {
  if (source_type.encoding_flags != target_type.encoding_flags) {
    return false;
  }
  if (!loom_type_has_ssa_encoding(source_type)) {
    return source_type.encoding_id == target_type.encoding_id;
  }
  loom_value_id_t remapped_value = loom_type_remap_value(
      module, remap, (loom_value_id_t)loom_type_encoding_value_id(source_type));
  if (remapped_value > UINT16_MAX) {
    return false;
  }
  return (uint16_t)remapped_value == target_type.encoding_id;
}

static bool loom_type_sequence_equal_after_value_remap(
    const loom_module_t* module, const loom_type_t* source_types,
    const loom_type_t* target_types, iree_host_size_t type_count,
    const loom_type_value_remap_t* remap) {
  if (type_count == 0) {
    return true;
  }
  if (!source_types || !target_types) {
    return source_types == target_types;
  }
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    // The remap is invariant across the sequence, so an identical adjacent
    // source/target pair has the same result as its successful predecessor.
    if (i > 0 &&
        loom_type_representation_equal(&source_types[i],
                                       &source_types[i - 1]) &&
        loom_type_representation_equal(&target_types[i],
                                       &target_types[i - 1])) {
      continue;
    }
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
    const loom_type_value_remap_t* remap);

static bool loom_attribute_sequence_equal_after_value_remap(
    const loom_module_t* module, const loom_attribute_t* source_attributes,
    const loom_attribute_t* target_attributes, iree_host_size_t attribute_count,
    uint8_t depth, const loom_type_value_remap_t* remap) {
  for (iree_host_size_t i = 0; i < attribute_count; ++i) {
    if (i > 0 &&
        loom_attribute_representation_equal(&source_attributes[i],
                                            &source_attributes[i - 1]) &&
        loom_attribute_representation_equal(&target_attributes[i],
                                            &target_attributes[i - 1])) {
      continue;
    }
    if (!loom_attribute_equal_after_value_remap(
            module, source_attributes[i], target_attributes[i], depth, remap)) {
      return false;
    }
  }
  return true;
}

static bool loom_attribute_equal_after_value_remap(
    const loom_module_t* module, loom_attribute_t source_attr,
    loom_attribute_t target_attr, uint8_t depth,
    const loom_type_value_remap_t* remap) {
  if (source_attr.kind != target_attr.kind) {
    return false;
  }
  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_TYPE:
      if (source_attr.type_id == LOOM_TYPE_ID_INVALID ||
          target_attr.type_id == LOOM_TYPE_ID_INVALID ||
          source_attr.type_id >= module->types.count ||
          target_attr.type_id >= module->types.count) {
        return false;
      }
      return loom_type_equal_after_value_remap(
          module, loom_type_table_get(&module->types, source_attr.type_id),
          loom_type_table_get(&module->types, target_attr.type_id), remap);

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
                    loom_type_remap_value(module, remap,
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
            target_attr.dict_entries[i].name_id) {
          return false;
        }
        if (i > 0 &&
            loom_attribute_representation_equal(
                &source_attr.dict_entries[i].value,
                &source_attr.dict_entries[i - 1].value) &&
            loom_attribute_representation_equal(
                &target_attr.dict_entries[i].value,
                &target_attr.dict_entries[i - 1].value)) {
          continue;
        }
        if (!loom_attribute_equal_after_value_remap(
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
      return loom_attribute_sequence_equal_after_value_remap(
          module, source_attr.parameterized_slots,
          target_attr.parameterized_slots, source_attr.count,
          (uint8_t)(depth + 1), remap);

    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (source_attr.count != target_attr.count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (source_attr.count > 0 && (!source_attr.parameterized_array ||
                                     !target_attr.parameterized_array))) {
        return false;
      }
      return loom_attribute_sequence_equal_after_value_remap(
          module, source_attr.parameterized_array,
          target_attr.parameterized_array, source_attr.count,
          (uint8_t)(depth + 1), remap);

    default:
      return loom_attribute_equal(&source_attr, &target_attr);
  }
}

bool loom_type_equal_after_value_remap(const loom_module_t* module,
                                       loom_type_t source_type,
                                       loom_type_t target_type,
                                       const loom_type_value_remap_t* remap) {
  loom_type_kind_t source_kind = loom_type_kind(source_type);
  if (source_kind != loom_type_kind(target_type)) {
    return false;
  }
  if (!loom_type_kind_is_valid(source_kind)) {
    return source_type.header == target_type.header &&
           source_type.encoding_id == target_type.encoding_id &&
           source_type.encoding_flags == target_type.encoding_flags &&
           source_type.dims[0] == target_type.dims[0] &&
           source_type.dims[1] == target_type.dims[1];
  }

  switch (source_kind) {
    case LOOM_TYPE_REFERENCE:
      return loom_type_equal(source_type, target_type);
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
      if (!source_data || !target_data) {
        return source_data == target_data;
      }
      iree_host_size_t type_count =
          (iree_host_size_t)source_data->arg_count + source_data->result_count;
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
      if (!source_data || !target_data) {
        return source_data == target_data;
      }
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
      if (parameter_count == 0) {
        return true;
      }
      if (!source_parameters || !target_parameters) {
        return source_parameters == target_parameters;
      }
      return loom_attribute_sequence_equal_after_value_remap(
          module, source_parameters, target_parameters, parameter_count,
          /*depth=*/1, remap);
    }

    default:
      break;
  }

  if (loom_type_is_shaped(source_type)) {
    if (source_type.header != target_type.header ||
        !loom_type_shape_equal_after_value_remap(module, source_type,
                                                 target_type, remap)) {
      return false;
    }
    return loom_type_encoding_equal_after_value_remap(module, source_type,
                                                      target_type, remap);
  }

  if (loom_type_has_dimensions(source_type)) {
    if (source_type.header != target_type.header) {
      return false;
    }
    const uint8_t rank = loom_type_rank(source_type);
    for (uint8_t i = 0; i < rank; ++i) {
      if (!loom_type_dim_equal_after_value_remap(
              module, loom_type_dim(source_type, i),
              loom_type_dim(target_type, i), remap)) {
        return false;
      }
    }
    return loom_type_encoding_equal_after_value_remap(module, source_type,
                                                      target_type, remap);
  }

  return source_type.header == target_type.header &&
         source_type.encoding_id == target_type.encoding_id &&
         source_type.encoding_flags == target_type.encoding_flags &&
         source_type.dims[0] == target_type.dims[0] &&
         source_type.dims[1] == target_type.dims[1];
}

//===----------------------------------------------------------------------===//
// Type SSA reference walking
//===----------------------------------------------------------------------===//

static bool loom_type_has_value_ref_dims(loom_type_t type) {
  return loom_type_has_dimensions(type);
}

static iree_status_t loom_type_walk_value_ref_sequence(
    const loom_module_t* module, const loom_type_t* types,
    iree_host_size_t type_count, loom_type_value_ref_callback_t callback,
    void* user_data) {
  if (!types) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < type_count; ++i) {
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
  if (!loom_type_kind_is_valid(kind)) {
    return iree_ok_status();
  }
  if (!loom_type_may_reference_values(type)) {
    return iree_ok_status();
  }

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return iree_ok_status();
      }
      return loom_type_walk_value_ref_sequence(
          module, data->types,
          (iree_host_size_t)data->arg_count + data->result_count, callback,
          user_data);
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
      if (!loom_type_dim_is_dynamic_at(type, i)) {
        continue;
      }
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
                                                iree_host_size_t type_count,
                                                const loom_module_t* module,
                                                loom_value_id_t value_id) {
  if (!types) {
    return false;
  }
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    if (i > 0 && loom_type_representation_equal(&types[i], &types[i - 1])) {
      continue;
    }
    if (loom_type_references_value(module, types[i], value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_attribute_references_value(const loom_module_t* module,
                                            loom_attribute_t attr,
                                            uint8_t depth,
                                            loom_value_id_t value_id);

static bool loom_attribute_sequence_references_value(
    const loom_module_t* module, const loom_attribute_t* attributes,
    iree_host_size_t attribute_count, uint8_t depth, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < attribute_count; ++i) {
    if (i > 0 && loom_attribute_representation_equal(&attributes[i],
                                                     &attributes[i - 1])) {
      continue;
    }
    if (loom_attribute_references_value(module, attributes[i], depth,
                                        value_id)) {
      return true;
    }
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
                 module, loom_type_table_get(&module->types, attr.type_id),
                 value_id);
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
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return false;
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (i > 0 &&
            loom_attribute_representation_equal(
                &attr.dict_entries[i].value, &attr.dict_entries[i - 1].value)) {
          continue;
        }
        if (loom_attribute_references_value(module, attr.dict_entries[i].value,
                                            (uint8_t)(depth + 1), value_id)) {
          return true;
        }
      }
      return false;
    case LOOM_ATTR_PARAMETERIZED:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return false;
      }
      return loom_attribute_sequence_references_value(
          module, attr.parameterized_slots, attr.count, (uint8_t)(depth + 1),
          value_id);
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return false;
      }
      return loom_attribute_sequence_references_value(
          module, attr.parameterized_array, attr.count, (uint8_t)(depth + 1),
          value_id);
    default:
      return false;
  }
}

bool loom_type_references_value(const loom_module_t* module, loom_type_t type,
                                loom_value_id_t value_id) {
  if (!module) {
    return false;
  }
  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    return false;
  }
  if (!loom_type_may_reference_values(type)) {
    return false;
  }

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return false;
      }
      return loom_type_sequence_references_value(
          data->types, (iree_host_size_t)data->arg_count + data->result_count,
          module, value_id);
    }

    case LOOM_TYPE_DIALECT:
      return loom_type_sequence_references_value(
          loom_type_dialect_params(type), loom_type_dialect_param_count(type),
          module, value_id);

    case LOOM_TYPE_PARAMETERIZED: {
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      return loom_attribute_sequence_references_value(
          module, parameters, parameter_count, /*depth=*/1, value_id);
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
      if (!loom_type_dim_is_dynamic_at(type, i)) {
        continue;
      }
      if (loom_type_dim_value_id_at(type, i) == value_id) {
        return true;
      }
    }
  }
  return loom_type_has_ssa_encoding(type) &&
         loom_type_encoding_value_id(type) == value_id;
}

//===----------------------------------------------------------------------===//
// Remapped structural hashing
//===----------------------------------------------------------------------===//

static uint32_t loom_attribute_hash_after_value_remap(
    const loom_module_t* module, const loom_attribute_t* attribute,
    uint8_t depth, const loom_type_value_remap_t* remap);

static uint32_t loom_attribute_hash_mix_sequence_after_value_remap(
    const loom_module_t* module, const loom_attribute_t* attributes,
    iree_host_size_t attribute_count, uint8_t depth,
    const loom_type_value_remap_t* remap, uint32_t hash) {
  uint32_t attribute_hash = 0;
  for (iree_host_size_t i = 0; i < attribute_count; ++i) {
    if (i == 0 || !loom_attribute_representation_equal(&attributes[i],
                                                       &attributes[i - 1])) {
      attribute_hash = loom_attribute_hash_after_value_remap(
          module, &attributes[i], depth, remap);
    }
    hash = loom_structural_hash_mix_u32(hash, attribute_hash);
  }
  return hash;
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static uint32_t loom_attribute_hash_after_value_remap(
    const loom_module_t* module, const loom_attribute_t* attribute,
    uint8_t depth, const loom_type_value_remap_t* remap) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u8(hash, attribute->kind);
  switch ((loom_attr_kind_t)attribute->kind) {
    case LOOM_ATTR_TYPE:
      if (attribute->type_id < module->types.count) {
        hash = loom_structural_hash_mix_u32(
            hash,
            loom_type_hash_after_value_remap(
                module, loom_type_table_get(&module->types, attribute->type_id),
                remap));
      } else {
        hash = loom_structural_hash_mix_u32(hash, attribute->type_id);
      }
      break;

    case LOOM_ATTR_PREDICATE_LIST:
      hash = loom_structural_hash_mix_u16(hash, attribute->count);
      if (attribute->count > 0 && attribute->predicate_list == NULL) {
        hash = loom_structural_hash_mix_u64(
            hash, (uint64_t)(uintptr_t)attribute->predicate_list);
        break;
      }
      for (uint16_t i = 0; i < attribute->count; ++i) {
        const loom_predicate_t* predicate = &attribute->predicate_list[i];
        hash = loom_structural_hash_mix_u8(hash, predicate->kind);
        hash = loom_structural_hash_mix_u8(hash, predicate->arg_count);
        hash = loom_structural_hash_mix_bytes(hash, predicate->arg_tags,
                                              sizeof(predicate->arg_tags));
        for (uint8_t j = 0; j < predicate->arg_count; ++j) {
          int64_t argument = predicate->args[j];
          if (predicate->arg_tags[j] == LOOM_PRED_ARG_VALUE && argument >= 0) {
            argument = (int64_t)loom_type_remap_value(
                module, remap, (loom_value_id_t)argument);
          }
          hash = loom_structural_hash_mix_u64(hash, (uint64_t)argument);
        }
      }
      break;

    case LOOM_ATTR_DICT: {
      hash = loom_structural_hash_mix_u16(hash, attribute->count);
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attribute->count > 0 && attribute->dict_entries == NULL)) {
        hash = loom_structural_hash_mix_u64(
            hash, (uint64_t)(uintptr_t)attribute->dict_entries);
        break;
      }
      uint32_t value_hash = 0;
      for (uint16_t i = 0; i < attribute->count; ++i) {
        hash = loom_structural_hash_mix_u32(hash,
                                            attribute->dict_entries[i].name_id);
        if (i == 0 || !loom_attribute_representation_equal(
                          &attribute->dict_entries[i].value,
                          &attribute->dict_entries[i - 1].value)) {
          value_hash = loom_attribute_hash_after_value_remap(
              module, &attribute->dict_entries[i].value, (uint8_t)(depth + 1),
              remap);
        }
        hash = loom_structural_hash_mix_u32(hash, value_hash);
      }
      break;
    }

    case LOOM_ATTR_PARAMETERIZED:
      hash = loom_structural_hash_mix_u32(hash, attribute->reserved_1);
      hash = loom_structural_hash_mix_u16(hash, attribute->count);
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attribute->count > 0 && attribute->parameterized_slots == NULL)) {
        hash = loom_structural_hash_mix_u64(
            hash, (uint64_t)(uintptr_t)attribute->parameterized_slots);
        break;
      }
      hash = loom_attribute_hash_mix_sequence_after_value_remap(
          module, attribute->parameterized_slots, attribute->count,
          (uint8_t)(depth + 1), remap, hash);
      break;

    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      hash = loom_structural_hash_mix_u16(hash, attribute->count);
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attribute->count > 0 && attribute->parameterized_array == NULL)) {
        hash = loom_structural_hash_mix_u64(
            hash, (uint64_t)(uintptr_t)attribute->parameterized_array);
        break;
      }
      hash = loom_attribute_hash_mix_sequence_after_value_remap(
          module, attribute->parameterized_array, attribute->count,
          (uint8_t)(depth + 1), remap, hash);
      break;

    default:
      hash = loom_structural_hash_mix_u32(hash, loom_attribute_hash(attribute));
      break;
  }
  return loom_structural_hash_finalize(hash);
}

static uint32_t loom_type_hash_sequence_after_value_remap(
    const loom_module_t* module, const loom_type_t* types,
    iree_host_size_t type_count, const loom_type_value_remap_t* remap,
    uint32_t hash) {
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)type_count);
  if (!types) {
    return hash;
  }
  uint32_t element_hash = 0;
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    if (i == 0 || !loom_type_representation_equal(&types[i], &types[i - 1])) {
      element_hash = loom_type_hash_after_value_remap(module, types[i], remap);
    }
    hash = loom_structural_hash_mix_u32(hash, element_hash);
  }
  return hash;
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE uint32_t
loom_type_hash_after_value_remap(const loom_module_t* module, loom_type_t type,
                                 const loom_type_value_remap_t* remap) {
  if (!loom_type_may_reference_values(type)) {
    return loom_type_hash(type);
  }
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(hash, type.header);
  hash = loom_structural_hash_mix_u16(hash, type.encoding_flags);

  const loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
    hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
    hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
    return loom_structural_hash_finalize(hash);
  }
  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return loom_structural_hash_finalize(hash);
      }
      hash = loom_structural_hash_mix_u16(hash, data->arg_count);
      hash = loom_structural_hash_mix_u16(hash, data->result_count);
      hash = loom_type_hash_sequence_after_value_remap(
          module, data->types,
          (iree_host_size_t)data->arg_count + data->result_count, remap, hash);
      return loom_structural_hash_finalize(hash);
    }

    case LOOM_TYPE_DIALECT:
      hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
      hash =
          loom_structural_hash_mix_u32(hash, loom_type_dialect_name_id(type));
      hash = loom_type_hash_sequence_after_value_remap(
          module, loom_type_dialect_params(type),
          loom_type_dialect_param_count(type), remap, hash);
      return loom_structural_hash_finalize(hash);

    case LOOM_TYPE_PARAMETERIZED: {
      hash = loom_structural_hash_mix_u64(
          hash, (uint64_t)(uintptr_t)loom_type_parameterized_descriptor(type));
      const uint8_t parameter_count =
          loom_type_parameterized_parameter_count(type);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      if (!parameters) {
        return loom_structural_hash_finalize(hash);
      }
      hash = loom_attribute_hash_mix_sequence_after_value_remap(
          module, parameters, parameter_count, /*depth=*/1, remap, hash);
      return loom_structural_hash_finalize(hash);
    }

    case LOOM_TYPE_REGISTER: {
      hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
      const loom_register_type_data_t* data = loom_type_register_data(type);
      if (!loom_type_register_has_value_type(type)) {
        hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
        hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
        return loom_structural_hash_finalize(hash);
      }
      if (!data) {
        return loom_structural_hash_finalize(hash);
      }
      hash = loom_structural_hash_mix_u64(hash, data->carrier_payload0);
      hash = loom_structural_hash_mix_u64(hash, data->carrier_payload1);
      hash = loom_structural_hash_mix_u32(
          hash,
          loom_type_hash_after_value_remap(module, data->value_type, remap));
      return loom_structural_hash_finalize(hash);
    }

    default:
      break;
  }

  if (loom_type_has_dimensions(type)) {
    uint32_t encoding = type.encoding_id;
    if (loom_type_has_ssa_encoding(type)) {
      encoding = loom_type_remap_value(module, remap,
                                       loom_type_encoding_value_id(type));
    }
    hash = loom_structural_hash_mix_u32(hash, encoding);
    const uint8_t rank = loom_type_rank(type);
    for (uint8_t i = 0; i < rank; ++i) {
      uint64_t dimension = loom_type_dim(type, i);
      if (loom_dim_is_dynamic(dimension)) {
        dimension = loom_dim_pack_dynamic(
            loom_type_remap_value(module, remap, loom_dim_value_id(dimension)));
      }
      hash = loom_structural_hash_mix_u64(hash, dimension);
    }
    return loom_structural_hash_finalize(hash);
  }

  hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
  hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
  hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
  return loom_structural_hash_finalize(hash);
}
