// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/catalog.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Numbering context
//===----------------------------------------------------------------------===//

// Sentinel for "not yet assigned" in mapping arrays.
#define LOOM_WRITER_ID_NONE UINT32_MAX

static uint32_t loom_bytecode_type_hash_mix_bytes(uint32_t hash,
                                                  const void* data,
                                                  iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_bytecode_type_hash_mix_u8(uint32_t hash, uint8_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_hash_mix_u16(uint32_t hash, uint16_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_hash_mix_u32(uint32_t hash, uint32_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_hash_mix_u64(uint32_t hash, uint64_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_wire_hash(const loom_module_t* module,
                                             loom_type_t type);
static bool loom_bytecode_type_wire_equal(const loom_module_t* module,
                                          loom_type_t a, loom_type_t b);

static uint32_t loom_bytecode_attr_wire_hash(const loom_module_t* module,
                                             const loom_attribute_t* attr,
                                             uint8_t aggregate_depth) {
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_TYPE:
      if (attr->type_id >= module->types.count) {
        return loom_attribute_hash(attr);
      }
      return loom_bytecode_type_wire_hash(module,
                                          module->types.entries[attr->type_id]);
    case LOOM_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_hash(attr);
      }
      uint32_t hash = loom_bytecode_type_hash_mix_u8(2166136261u, attr->kind);
      hash = loom_bytecode_type_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(hash,
                                               attr->dict_entries[i].name_id);
        hash = loom_bytecode_type_hash_mix_u32(
            hash,
            loom_bytecode_attr_wire_hash(module, &attr->dict_entries[i].value,
                                         aggregate_depth + 1));
      }
      return hash;
    }
    case LOOM_ATTR_PARAMETERIZED: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_hash(attr);
      }
      uint32_t hash = loom_bytecode_type_hash_mix_u8(2166136261u, attr->kind);
      hash = loom_bytecode_type_hash_mix_u32(hash, attr->reserved_1);
      hash = loom_bytecode_type_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash,
            loom_bytecode_attr_wire_hash(module, &attr->parameterized_slots[i],
                                         aggregate_depth + 1));
      }
      return hash;
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_hash(attr);
      }
      uint32_t hash = loom_bytecode_type_hash_mix_u8(2166136261u, attr->kind);
      hash = loom_bytecode_type_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash,
            loom_bytecode_attr_wire_hash(module, &attr->parameterized_array[i],
                                         aggregate_depth + 1));
      }
      return hash;
    }
    default:
      return loom_attribute_hash(attr);
  }
}

static bool loom_bytecode_attr_wire_equal(const loom_module_t* module,
                                          const loom_attribute_t* a,
                                          const loom_attribute_t* b,
                                          uint8_t aggregate_depth) {
  if (a->kind != b->kind) return false;
  switch ((loom_attr_kind_t)a->kind) {
    case LOOM_ATTR_TYPE:
      if (a->type_id >= module->types.count ||
          b->type_id >= module->types.count) {
        return a->type_id == b->type_id;
      }
      return loom_bytecode_type_wire_equal(module,
                                           module->types.entries[a->type_id],
                                           module->types.entries[b->type_id]);
    case LOOM_ATTR_DICT:
      if (a->count != b->count) return false;
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_equal(a, b);
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (a->dict_entries[i].name_id != b->dict_entries[i].name_id ||
            !loom_bytecode_attr_wire_equal(module, &a->dict_entries[i].value,
                                           &b->dict_entries[i].value,
                                           aggregate_depth + 1)) {
          return false;
        }
      }
      return true;
    case LOOM_ATTR_PARAMETERIZED:
      if (a->reserved_1 != b->reserved_1 || a->count != b->count) return false;
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_equal(a, b);
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (!loom_bytecode_attr_wire_equal(module, &a->parameterized_slots[i],
                                           &b->parameterized_slots[i],
                                           aggregate_depth + 1)) {
          return false;
        }
      }
      return true;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (a->count != b->count) return false;
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_equal(a, b);
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (!loom_bytecode_attr_wire_equal(module, &a->parameterized_array[i],
                                           &b->parameterized_array[i],
                                           aggregate_depth + 1)) {
          return false;
        }
      }
      return true;
    default:
      return loom_attribute_equal(a, b);
  }
}

static uint32_t loom_bytecode_type_wire_hash(const loom_module_t* module,
                                             loom_type_t type) {
  uint32_t hash = 2166136261u;
  loom_type_kind_t kind = loom_type_kind(type);
  hash = loom_bytecode_type_hash_mix_u8(hash, (uint8_t)kind);
  hash = loom_bytecode_type_hash_mix_u8(hash,
                                        (uint8_t)loom_type_element_type(type));
  hash = loom_bytecode_type_hash_mix_u8(hash, loom_type_rank(type));

  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      if (loom_type_has_ssa_encoding(type)) {
        hash = loom_bytecode_type_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA);
      } else if (loom_type_has_static_encoding(type)) {
        hash = loom_bytecode_type_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC);
        hash = loom_bytecode_type_hash_mix_u16(hash, type.encoding_id);
      } else {
        hash = loom_bytecode_type_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE);
      }
      for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
        uint64_t dim = loom_type_dim(type, i);
        if (loom_dim_is_dynamic(dim)) {
          hash = loom_bytecode_type_hash_mix_u8(hash, 1);
        } else {
          hash = loom_bytecode_type_hash_mix_u8(hash, 0);
          hash = loom_bytecode_type_hash_mix_u64(hash, dim);
        }
      }
      return hash;
    }
    case LOOM_TYPE_POOL: {
      uint64_t dim = loom_type_dim(type, 0);
      if (loom_dim_is_dynamic(dim)) {
        hash = loom_bytecode_type_hash_mix_u8(hash, 1);
      } else {
        hash = loom_bytecode_type_hash_mix_u8(hash, 0);
        hash = loom_bytecode_type_hash_mix_u64(hash, dim);
      }
      return hash;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return hash;
      hash = loom_bytecode_type_hash_mix_u16(hash, data->arg_count);
      hash = loom_bytecode_type_hash_mix_u16(hash, data->result_count);
      uint16_t type_count = (uint16_t)(data->arg_count + data->result_count);
      for (uint16_t i = 0; i < type_count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash, loom_bytecode_type_wire_hash(module, data->types[i]));
      }
      return hash;
    }
    case LOOM_TYPE_DIALECT: {
      hash = loom_bytecode_type_hash_mix_u32(hash,
                                             loom_type_dialect_name_id(type));
      uint16_t param_count = loom_type_dialect_param_count(type);
      hash = loom_bytecode_type_hash_mix_u16(hash, param_count);
      const loom_type_t* params = loom_type_dialect_params(type);
      for (uint16_t i = 0; params && i < param_count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash, loom_bytecode_type_wire_hash(module, params[i]));
      }
      return hash;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      if (!descriptor) return hash;
      const iree_string_view_t family_name =
          loom_bstring_view(descriptor->name);
      hash = loom_bytecode_type_hash_mix_bytes(hash, family_name.data,
                                               family_name.size);
      const uint8_t parameter_count =
          loom_type_parameterized_parameter_count(type);
      hash = loom_bytecode_type_hash_mix_u8(hash, parameter_count);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      for (uint8_t i = 0; parameters && i < parameter_count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash, loom_bytecode_attr_wire_hash(module, &parameters[i], 0));
      }
      return hash;
    }
    case LOOM_TYPE_REGISTER: {
      hash = loom_bytecode_type_hash_mix_u64(hash,
                                             loom_type_register_payload0(type));
      hash = loom_bytecode_type_hash_mix_u64(hash,
                                             loom_type_register_payload1(type));
      const loom_type_t* value_type = loom_type_register_value_type(type);
      hash = loom_bytecode_type_hash_mix_u8(hash, value_type ? 1 : 0);
      return value_type
                 ? loom_bytecode_type_hash_mix_u32(
                       hash, loom_bytecode_type_wire_hash(module, *value_type))
                 : hash;
    }
    case LOOM_TYPE_STORAGE:
      return loom_bytecode_type_hash_mix_u8(
          hash, (uint8_t)loom_type_storage_space(type));
    case LOOM_TYPE_ENCODING:
      return loom_bytecode_type_hash_mix_u8(
          hash, (uint8_t)loom_type_encoding_role(type));
    default:
      return hash;
  }
}

static bool loom_bytecode_type_dim_wire_equal(uint64_t a, uint64_t b) {
  bool a_dynamic = loom_dim_is_dynamic(a);
  bool b_dynamic = loom_dim_is_dynamic(b);
  if (a_dynamic || b_dynamic) return a_dynamic == b_dynamic;
  return a == b;
}

static bool loom_bytecode_type_encoding_wire_equal(loom_type_t a,
                                                   loom_type_t b) {
  if (loom_type_has_ssa_encoding(a) || loom_type_has_ssa_encoding(b)) {
    return loom_type_has_ssa_encoding(a) == loom_type_has_ssa_encoding(b);
  }
  if (loom_type_has_static_encoding(a) || loom_type_has_static_encoding(b)) {
    return loom_type_has_static_encoding(a) ==
               loom_type_has_static_encoding(b) &&
           a.encoding_id == b.encoding_id;
  }
  return !loom_type_has_encoding(a) && !loom_type_has_encoding(b);
}

static bool loom_bytecode_type_wire_equal(const loom_module_t* module,
                                          loom_type_t a, loom_type_t b) {
  loom_type_kind_t kind = loom_type_kind(a);
  if (kind != loom_type_kind(b)) return false;
  if (loom_type_element_type(a) != loom_type_element_type(b)) return false;
  if (loom_type_rank(a) != loom_type_rank(b)) return false;

  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
      if (!loom_bytecode_type_encoding_wire_equal(a, b)) return false;
      for (uint8_t i = 0; i < loom_type_rank(a); ++i) {
        if (!loom_bytecode_type_dim_wire_equal(loom_type_dim(a, i),
                                               loom_type_dim(b, i))) {
          return false;
        }
      }
      return true;
    case LOOM_TYPE_POOL:
      return loom_bytecode_type_dim_wire_equal(loom_type_dim(a, 0),
                                               loom_type_dim(b, 0));
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* a_data = loom_type_func_data(a);
      const loom_func_type_data_t* b_data = loom_type_func_data(b);
      if (!a_data || !b_data) return a_data == b_data;
      if (a_data->arg_count != b_data->arg_count ||
          a_data->result_count != b_data->result_count) {
        return false;
      }
      uint16_t type_count =
          (uint16_t)(a_data->arg_count + a_data->result_count);
      for (uint16_t i = 0; i < type_count; ++i) {
        if (!loom_bytecode_type_wire_equal(module, a_data->types[i],
                                           b_data->types[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_DIALECT: {
      if (loom_type_dialect_name_id(a) != loom_type_dialect_name_id(b)) {
        return false;
      }
      uint16_t param_count = loom_type_dialect_param_count(a);
      if (param_count != loom_type_dialect_param_count(b)) return false;
      const loom_type_t* a_params = loom_type_dialect_params(a);
      const loom_type_t* b_params = loom_type_dialect_params(b);
      if (!a_params || !b_params) return a_params == b_params;
      for (uint16_t i = 0; i < param_count; ++i) {
        if (!loom_bytecode_type_wire_equal(module, a_params[i], b_params[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* a_descriptor =
          loom_type_parameterized_descriptor(a);
      const loom_parameterized_type_descriptor_t* b_descriptor =
          loom_type_parameterized_descriptor(b);
      if (!a_descriptor || !b_descriptor) {
        return a_descriptor == b_descriptor;
      }
      if (!iree_string_view_equal(loom_bstring_view(a_descriptor->name),
                                  loom_bstring_view(b_descriptor->name))) {
        return false;
      }
      const uint8_t parameter_count =
          loom_type_parameterized_parameter_count(a);
      if (parameter_count != loom_type_parameterized_parameter_count(b)) {
        return false;
      }
      const loom_attribute_t* a_parameters =
          loom_type_parameterized_parameters(a);
      const loom_attribute_t* b_parameters =
          loom_type_parameterized_parameters(b);
      if (!a_parameters || !b_parameters) return a_parameters == b_parameters;
      for (uint8_t i = 0; i < parameter_count; ++i) {
        if (!loom_bytecode_attr_wire_equal(module, &a_parameters[i],
                                           &b_parameters[i], 0)) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_REGISTER: {
      if (loom_type_register_payload0(a) != loom_type_register_payload0(b) ||
          loom_type_register_payload1(a) != loom_type_register_payload1(b)) {
        return false;
      }
      const loom_type_t* a_value_type = loom_type_register_value_type(a);
      const loom_type_t* b_value_type = loom_type_register_value_type(b);
      if (!a_value_type || !b_value_type) {
        return a_value_type == b_value_type;
      }
      return loom_bytecode_type_wire_equal(module, *a_value_type,
                                           *b_value_type);
    }
    case LOOM_TYPE_STORAGE:
      return loom_type_storage_space(a) == loom_type_storage_space(b);
    case LOOM_TYPE_ENCODING:
      return loom_type_encoding_role(a) == loom_type_encoding_role(b);
    default:
      return true;
  }
}

// Appends a string_view to the ordered string list, growing if needed.
static iree_status_t loom_bytecode_numbering_append_string(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  if (numbering->strings.count >= numbering->strings.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->strings.count, /*minimum_capacity=*/16,
        sizeof(iree_string_view_t), &numbering->strings.capacity,
        (void**)&numbering->strings.values));
  }
  if (numbering->strings.count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode string count exceeds format maximum "
                            "(16M)");
  }
  uint32_t id = (uint32_t)numbering->strings.count;
  numbering->strings.values[numbering->strings.count++] = view;
  *out_writer_id = id;
  return iree_ok_status();
}

// Builds the stable module-ID to presentation-ordered wire-ordinal mapping.
static iree_status_t loom_bytecode_numbering_initialize_symbol_order(
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;
  if (module->symbols.count == 0) {
    return iree_ok_status();
  }

  // Both directions share one dense arena allocation. Symbol IDs are bounded
  // below LOOM_SYMBOL_ID_INVALID by module construction.
  loom_symbol_id_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(numbering->arena, module->symbols.count,
                                2 * sizeof(*storage), (void**)&storage));
  numbering->symbol_order.module_ids = storage;
  numbering->symbol_order.wire_ordinals = storage + module->symbols.count;
  memset(numbering->symbol_order.wire_ordinals, 0xFF,
         module->symbols.count * sizeof(*storage));

  loom_symbol_id_t wire_ordinal = 0;
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(module_block, op) {
    const loom_symbol_id_t symbol_id =
        loom_op_defining_symbol_id(module, op, loom_op_vtable(module, op));
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) continue;
    if (module->symbols.entries[symbol_id].defining_op != op) {
      continue;
    }
    IREE_ASSERT_EQ(numbering->symbol_order.wire_ordinals[symbol_id],
                   LOOM_SYMBOL_ID_INVALID);
    numbering->symbol_order.module_ids[wire_ordinal] = symbol_id;
    numbering->symbol_order.wire_ordinals[symbol_id] = wire_ordinal;
    ++wire_ordinal;
  }

  // Symbols without a live top-level defining op have no physical presentation
  // anchor. Preserve their stable module-table order after all definitions.
  for (loom_symbol_id_t module_symbol_id = 0;
       module_symbol_id < module->symbols.count; ++module_symbol_id) {
    if (numbering->symbol_order.wire_ordinals[module_symbol_id] !=
        LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    numbering->symbol_order.module_ids[wire_ordinal] = module_symbol_id;
    numbering->symbol_order.wire_ordinals[module_symbol_id] = wire_ordinal;
    ++wire_ordinal;
  }
  IREE_ASSERT_EQ(wire_ordinal, module->symbols.count);
  return iree_ok_status();
}

// Initializes the numbering context. All allocations come from |arena|,
// which the caller owns. No individual frees needed — the arena handles
// bulk deallocation.
iree_status_t loom_bytecode_numbering_initialize(
    loom_bytecode_numbering_t* numbering, const loom_module_t* module,
    iree_arena_allocator_t* arena) {
  memset(numbering, 0, sizeof(*numbering));
  numbering->module = module;
  numbering->arena = arena;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_initialize_symbol_order(numbering));

  // Module string map: parallel array for O(1) module_string_id → writer_id.
  if (module->strings.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->strings.count, sizeof(uint32_t),
        (void**)&numbering->strings.writer_ids_by_module_id));
    memset(numbering->strings.writer_ids_by_module_id, 0xFF,
           module->strings.count * sizeof(uint32_t));
  }

  // Writer string id 0 is reserved as "no SSA name" in value definitions.
  // Keep the empty string in slot 0 so named values never alias the sentinel.
  uint32_t empty_string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_append_string(
      numbering, iree_string_view_empty(), &empty_string_writer_id));
  if (numbering->strings.writer_ids_by_module_id != NULL) {
    for (iree_host_size_t i = 0; i < module->strings.count; ++i) {
      if (iree_string_view_is_empty(module->strings.entries[i])) {
        numbering->strings.writer_ids_by_module_id[i] = empty_string_writer_id;
        break;
      }
    }
  }

  // Type map: parallel array for O(1) module_type_index → writer_type_id.
  if (module->types.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->types.count, sizeof(uint32_t),
        (void**)&numbering->types.writer_ids_by_module_index));
    memset(numbering->types.writer_ids_by_module_index, 0xFF,
           module->types.count * sizeof(uint32_t));

    iree_host_size_t type_index_capacity =
        iree_host_size_next_power_of_two((module->types.count * 4 + 2) / 3);
    if (type_index_capacity < 16) type_index_capacity = 16;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, type_index_capacity, sizeof(loom_bytecode_type_index_entry_t),
        (void**)&numbering->types.index_entries));
    for (iree_host_size_t i = 0; i < type_index_capacity; ++i) {
      numbering->types.index_entries[i].hash = 0;
      numbering->types.index_entries[i].module_index = LOOM_WRITER_ID_NONE;
    }
    numbering->types.index_capacity = type_index_capacity;
    iree_host_size_t mask = type_index_capacity - 1;
    for (iree_host_size_t i = 0; i < module->types.count; ++i) {
      uint32_t hash =
          loom_bytecode_type_wire_hash(module, module->types.entries[i]);
      iree_host_size_t slot = hash & mask;
      while (numbering->types.index_entries[slot].module_index !=
             LOOM_WRITER_ID_NONE) {
        slot = (slot + 1) & mask;
      }
      numbering->types.index_entries[slot].hash = hash;
      numbering->types.index_entries[slot].module_index = (uint32_t)i;
    }
  }

  return iree_ok_status();
}

// Interns a module string by its module string_id. Returns writer string ID.
iree_status_t loom_bytecode_numbering_intern_module_string(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id,
    uint32_t* out_writer_id) {
  if (string_id == LOOM_STRING_ID_INVALID) {
    *out_writer_id = 0;
    return iree_ok_status();
  }
  if (string_id >= numbering->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string_id %u out of range (module has %" PRIhsz
                            " strings)",
                            string_id, numbering->module->strings.count);
  }
  if (numbering->strings.writer_ids_by_module_id[string_id] !=
      LOOM_WRITER_ID_NONE) {
    *out_writer_id = numbering->strings.writer_ids_by_module_id[string_id];
    return iree_ok_status();
  }
  iree_string_view_t view = numbering->module->strings.entries[string_id];
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  numbering->strings.writer_ids_by_module_id[string_id] = writer_id;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an arbitrary string_view (for vtable names not in the module).
iree_status_t loom_bytecode_numbering_intern_string_view(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  if (iree_string_view_is_empty(view)) {
    *out_writer_id = 0;
    return iree_ok_status();
  }
  // Check external strings first (linear scan, small list).
  for (iree_host_size_t i = 0; i < numbering->strings.external.count; ++i) {
    if (iree_string_view_equal(numbering->strings.external.values[i].view,
                               view)) {
      *out_writer_id = numbering->strings.external.values[i].writer_id;
      return iree_ok_status();
    }
  }
  // Also check if it happens to match a module string. Module strings are
  // assigned in first-use order, not source intern-table order, so the bytecode
  // stays canonical across text forms that intern strings differently.
  for (iree_host_size_t i = 0; i < numbering->module->strings.count; ++i) {
    if (iree_string_view_equal(numbering->module->strings.entries[i], view)) {
      return loom_bytecode_numbering_intern_module_string(
          numbering, (loom_string_id_t)i, out_writer_id);
    }
  }
  // New external string.
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  if (numbering->strings.external.count >=
      numbering->strings.external.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->strings.external.count,
        /*minimum_capacity=*/16, sizeof(loom_bytecode_external_string_t),
        &numbering->strings.external.capacity,
        (void**)&numbering->strings.external.values));
  }
  numbering->strings.external.values[numbering->strings.external.count++] =
      (loom_bytecode_external_string_t){.view = view, .writer_id = writer_id};
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Finds the representative module type table index for a given wire type.
static uint32_t loom_bytecode_find_type_index(
    const loom_bytecode_numbering_t* numbering, loom_type_t type) {
  if (numbering->types.index_capacity == 0) return LOOM_WRITER_ID_NONE;
  uint32_t hash = loom_bytecode_type_wire_hash(numbering->module, type);
  iree_host_size_t mask = numbering->types.index_capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (numbering->types.index_entries[slot].module_index !=
         LOOM_WRITER_ID_NONE) {
    const loom_bytecode_type_index_entry_t* entry =
        &numbering->types.index_entries[slot];
    if (entry->hash == hash &&
        loom_bytecode_type_wire_equal(
            numbering->module,
            numbering->module->types.entries[entry->module_index], type)) {
      return entry->module_index;
    }
    slot = (slot + 1) & mask;
  }
  return LOOM_WRITER_ID_NONE;
}

// Interns a type, recursing into sub-types first (topological order).
iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id) {
  uint32_t module_index = loom_bytecode_find_type_index(numbering, type);
  if (module_index == LOOM_WRITER_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type not found in module type table (kind=%u, "
                            "rank=%u, module types=%" PRIhsz ")",
                            (unsigned)loom_type_kind(type),
                            (unsigned)loom_type_rank(type),
                            numbering->module->types.count);
  }
  if (numbering->types.writer_ids_by_module_index[module_index] !=
      LOOM_WRITER_ID_NONE) {
    *out_writer_id = numbering->types.writer_ids_by_module_index[module_index];
    return iree_ok_status();
  }

  // Recurse into sub-types first (topological ordering).
  uint32_t unused_id = 0;
  loom_type_kind_t kind = loom_type_kind(type);
  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      // Element type is a scalar — intern it.
      loom_type_t element_type = loom_type_scalar(loom_type_element_type(type));
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, element_type, &unused_id));
      break;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* func_data = loom_type_func_data(type);
      for (uint16_t i = 0; i < func_data->arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, func_data->types[i], &unused_id));
      }
      for (uint16_t i = 0; i < func_data->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, func_data->types[func_data->arg_count + i], &unused_id));
      }
      break;
    }
    case LOOM_TYPE_DIALECT: {
      // Intern the dialect type name string.
      loom_string_id_t name_id = loom_type_dialect_name_id(type);
      if (name_id < numbering->module->strings.count) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering, numbering->module->strings.entries[name_id],
            &unused_id));
      }
      // Recurse into type parameters.
      uint16_t param_count = loom_type_dialect_param_count(type);
      const loom_type_t* params = loom_type_dialect_params(type);
      for (uint16_t i = 0; i < param_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, params[i], &unused_id));
      }
      break;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      if (!descriptor) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parameterized type has no descriptor");
      }
      const uint8_t parameter_count =
          loom_type_parameterized_parameter_count(type);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      if (parameter_count != descriptor->parameter_count ||
          (parameter_count > 0 && !parameters)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized type '%.*s' has malformed slot storage",
            (int)loom_bstring_view(descriptor->name).size,
            loom_bstring_view(descriptor->name).data);
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
          numbering, loom_bstring_view(descriptor->name), &unused_id));
      for (uint8_t i = 0; i < parameter_count; ++i) {
        const loom_attr_descriptor_t* parameter_descriptor =
            &descriptor->parameter_descriptors[i];
        if (loom_attr_is_absent(parameters[i])) {
          if (iree_any_bit_set(parameter_descriptor->flags,
                               LOOM_ATTR_OPTIONAL)) {
            continue;
          }
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "parameterized type '%.*s' has absent required parameter '%.*s'",
              (int)loom_bstring_view(descriptor->name).size,
              loom_bstring_view(descriptor->name).data,
              (int)loom_attr_descriptor_name(parameter_descriptor).size,
              loom_attr_descriptor_name(parameter_descriptor).data);
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering, loom_attr_descriptor_name(parameter_descriptor),
            &unused_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_number_attr_value(
            numbering, parameters[i], parameter_descriptor));
      }
      break;
    }
    case LOOM_TYPE_REGISTER: {
      const loom_type_t* value_type = loom_type_register_value_type(type);
      if (value_type) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, *value_type, &unused_id));
      }
      break;
    }
    default:
      break;
  }

  // Intern the parent type.
  if (numbering->types.count >= (1u << 16)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode type count exceeds format maximum (64K)");
  }
  uint32_t writer_id = numbering->types.count;
  numbering->types.writer_ids_by_module_index[module_index] = writer_id;
  if (numbering->types.count >= numbering->types.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->types.count, /*minimum_capacity=*/16,
        sizeof(iree_host_size_t), &numbering->types.capacity,
        (void**)&numbering->types.module_indices_by_writer_id));
  }
  numbering->types.module_indices_by_writer_id[numbering->types.count] =
      module_index;
  numbering->types.count++;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an op kind. Returns writer op name ID.
iree_status_t loom_bytecode_numbering_intern_op(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    uint32_t* out_writer_op_id) {
  // Check existing entries.
  for (uint32_t i = 0; i < numbering->ops.count; ++i) {
    if (numbering->ops.values[i].kind == op->kind) {
      *out_writer_op_id = numbering->ops.values[i].writer_op_id;
      return iree_ok_status();
    }
  }
  // New op kind.
  iree_string_view_t name = loom_op_name(numbering->module, op);
  uint32_t string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, name, &string_writer_id));

  if (numbering->ops.count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode op count exceeds format maximum (16M)");
  }
  uint32_t writer_op_id = (uint32_t)numbering->ops.count;
  if (numbering->ops.count >= numbering->ops.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->ops.count, /*minimum_capacity=*/16,
        sizeof(loom_bytecode_op_entry_t), &numbering->ops.capacity,
        (void**)&numbering->ops.values));
  }
  numbering->ops.values[numbering->ops.count++] = (loom_bytecode_op_entry_t){
      .kind = op->kind,
      .writer_op_id = writer_op_id,
      .string_writer_id = string_writer_id,
  };
  *out_writer_op_id = writer_op_id;
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_enum_ordinal(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t* out_ordinal) {
  if (attr.raw > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum attribute value exceeds uint8_t range");
  }
  *out_ordinal = (uint8_t)attr.raw;
  if (!descriptor ||
      iree_all_bits_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_ok_status();
  }
  if (!loom_attr_descriptor_has_enum_case(descriptor, *out_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum attribute value has no declared case");
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_enum_array(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_enum_array_t* out_array) {
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_ENUM_ARRAY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum arrays require a descriptor-backed field");
  }
  if (attr.count > 0 && !attr.enum_array) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum array has nonzero count but NULL values");
  }
  *out_array = loom_attr_as_enum_array(attr);
  if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < out_array->count; ++i) {
    if (!loom_attr_descriptor_has_enum_case(descriptor, out_array->values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "enum array value has no declared case");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_signed_enum_set(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_signed_enum_set_t* out_set) {
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_SIGNED_ENUM_SET ||
      iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "signed enum sets require a closed descriptor-backed field");
  }
  *out_set = loom_attr_as_signed_enum_set(attr);
  iree_host_size_t canonical_word_count = 0;
  IREE_RETURN_IF_ERROR(loom_signed_enum_set_canonical_word_count(
      *out_set, &canonical_word_count));
  if (canonical_word_count != out_set->word_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "signed enum set is not canonically trimmed");
  }
  for (iree_host_size_t value = 0; value < 256; ++value) {
    if (!loom_signed_enum_set_contains_positive(*out_set, (uint8_t)value) &&
        !loom_signed_enum_set_contains_negative(*out_set, (uint8_t)value)) {
      continue;
    }
    if (!loom_attr_descriptor_has_enum_case(descriptor, (uint8_t)value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signed enum-set value %u has no declared case",
                              (unsigned)value);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_symbol_collection(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_ref_array_t* out_array) {
  if (!descriptor || descriptor->attr_kind != attr.kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol collections require a descriptor-backed field");
  }
  if (attr.count > 0 && !attr.symbol_refs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol collection has nonzero count but NULL references");
  }
  *out_array = loom_make_symbol_ref_array(attr.symbol_refs, attr.count);
  for (iree_host_size_t i = 0; i < out_array->count; ++i) {
    const loom_symbol_ref_t ref = out_array->values[i];
    if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
        ref.symbol_id >= numbering->module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "symbol collection reference %" PRIhsz
                              " is not a local module symbol",
                              i);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_parameterized_attr(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind,
    const loom_parameterized_attr_descriptor_t** out_family_descriptor) {
  if (attr.kind != LOOM_ATTR_PARAMETERIZED) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized attribute payload has kind %u",
                            (unsigned)attr.kind);
  }
  const loom_parameterized_attr_kind_t family_kind =
      loom_attr_as_parameterized_kind(attr);
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_resolve_parameterized_attr(numbering->module->context,
                                              family_kind);
  if (!family_descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family kind %u is not registered",
        (unsigned)family_kind);
  }
  if (descriptor && descriptor->attr_kind != expected_descriptor_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute does not match field kind %u",
        (unsigned)descriptor->attr_kind);
  }
  if (descriptor &&
      descriptor->reference.parameterized_attr_kind !=
          LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
      descriptor->reference.parameterized_attr_kind != family_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family kind %u does not match field contract "
        "%u",
        (unsigned)family_kind,
        (unsigned)descriptor->reference.parameterized_attr_kind);
  }
  if (attr.count != family_descriptor->parameter_count ||
      (attr.count > 0 && !attr.parameterized_slots)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute '%.*s' has malformed slot storage",
        (int)loom_bstring_view(family_descriptor->name).size,
        loom_bstring_view(family_descriptor->name).data);
  }
  *out_family_descriptor = family_descriptor;
  return iree_ok_status();
}

iree_status_t loom_bytecode_parameter_is_present(
    const loom_parameterized_attr_descriptor_t* family_descriptor,
    loom_attribute_t value, uint8_t parameter_index, bool* out_present) {
  const loom_attr_descriptor_t* parameter_descriptor =
      &family_descriptor->parameter_descriptors[parameter_index];
  if (!loom_attr_is_absent(value)) {
    *out_present = true;
    return iree_ok_status();
  }
  if (iree_any_bit_set(parameter_descriptor->flags, LOOM_ATTR_OPTIONAL)) {
    *out_present = false;
    return iree_ok_status();
  }
  const iree_string_view_t family_name =
      loom_bstring_view(family_descriptor->name);
  const iree_string_view_t parameter_name =
      loom_attr_descriptor_name(parameter_descriptor);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "parameterized attribute '%.*s' has absent required parameter '%.*s'",
      (int)family_name.size, family_name.data, (int)parameter_name.size,
      parameter_name.data);
}

iree_status_t loom_bytecode_resolve_function_low_descriptor_set(
    const loom_bytecode_numbering_t* numbering, loom_func_like_t func_like,
    const loom_low_repr_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (func_like.vtable->repr_contract_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  const loom_string_id_t descriptor_set_key_id =
      loom_func_like_repr_contract(func_like);
  if (descriptor_set_key_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function representation contract is required");
  }
  if (descriptor_set_key_id >= numbering->module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function representation contract string ID is out of range");
  }
  if (!numbering->low_repr.environment.vtable) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "serializing Low functions requires a representation codec");
  }
  const iree_string_view_t descriptor_set_key =
      numbering->module->strings.entries[descriptor_set_key_id];
  *out_descriptor_set = loom_low_repr_lookup_descriptor_set(
      &numbering->low_repr.environment, descriptor_set_key);
  if (!*out_descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function representation contract '%.*s' is not available",
        (int)descriptor_set_key.size, descriptor_set_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_scoped_enum(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr) {
  if (!numbering->low_repr.active_descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scoped enum attribute is outside a representation contract");
  }
  const iree_string_view_t key =
      loom_low_repr_descriptor_key(&numbering->low_repr.environment,
                                   numbering->low_repr.active_descriptor_set,
                                   loom_attr_as_scoped_enum(attr));
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scoped enum ordinal is outside the active representation contract");
  }
  uint32_t unused_id = 0;
  return loom_bytecode_numbering_intern_string_view(numbering, key, &unused_id);
}

static iree_status_t loom_bytecode_number_attr_value_at_depth(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor, uint8_t aggregate_depth);

static iree_status_t loom_bytecode_number_parameterized_attr_payload(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind, uint8_t aggregate_depth) {
  if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aggregate attribute nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
  }
  const loom_parameterized_attr_descriptor_t* family_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_get_parameterized_attr(
      numbering, attr, descriptor, expected_descriptor_kind,
      &family_descriptor));
  uint32_t unused_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, loom_bstring_view(family_descriptor->name), &unused_id));
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_parameter_is_present(
        family_descriptor, attr.parameterized_slots[i], i, &present));
    if (!present) continue;
    const loom_attr_descriptor_t* parameter_descriptor =
        &family_descriptor->parameter_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(parameter_descriptor),
        &unused_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_number_attr_value_at_depth(
        numbering, attr.parameterized_slots[i], parameter_descriptor,
        aggregate_depth + 1));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_attr_value_at_depth(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor, uint8_t aggregate_depth) {
  uint32_t unused_id = 0;
  switch (attr.kind) {
    case LOOM_ATTR_STRING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &unused_id));
      break;
    }
    case LOOM_ATTR_ENUM: {
      break;
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      loom_enum_array_t unused_array = loom_enum_array_empty();
      return loom_bytecode_get_enum_array(attr, descriptor, &unused_array);
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      loom_signed_enum_set_t unused_set = loom_signed_enum_set_empty();
      return loom_bytecode_get_signed_enum_set(attr, descriptor, &unused_set);
    }
    case LOOM_ATTR_SCOPED_ENUM:
      return loom_bytecode_number_scoped_enum(numbering, attr);
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target_symbol =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target_symbol->name_id, &unused_id));
      }
      break;
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      loom_symbol_ref_array_t array = loom_symbol_ref_array_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_get_symbol_collection(
          numbering, attr, descriptor, &array));
      for (iree_host_size_t i = 0; i < array.count; ++i) {
        const loom_symbol_t* target_symbol =
            &numbering->module->symbols.entries[array.values[i].symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target_symbol->name_id, &unused_id));
      }
      break;
    }
    case LOOM_ATTR_TYPE: {
      if (attr.type_id >= numbering->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u out of range (module has %" PRIhsz " types)",
            (unsigned)attr.type_id, numbering->module->types.count);
      }
      loom_type_t type = numbering->module->types.entries[attr.type_id];
      IREE_RETURN_IF_ERROR(
          loom_bytecode_numbering_intern_type(numbering, type, &unused_id));
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_number_encoding(
          numbering, loom_attr_as_encoding_id(attr)));
      break;
    }
    case LOOM_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, attr.dict_entries[i].name_id, &unused_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_number_attr_value_at_depth(
            numbering, attr.dict_entries[i].value, NULL, aggregate_depth + 1));
      }
      break;
    }
    case LOOM_ATTR_PARAMETERIZED: {
      return loom_bytecode_number_parameterized_attr_payload(
          numbering, attr, descriptor, LOOM_ATTR_PARAMETERIZED,
          aggregate_depth);
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (!descriptor ||
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute arrays require a descriptor-backed "
            "field");
      }
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attr.count > 0 && !attr.parameterized_array)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array has malformed storage or nesting");
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_number_parameterized_attr_payload(
            numbering, attr.parameterized_array[i], descriptor,
            LOOM_ATTR_PARAMETERIZED_ARRAY, aggregate_depth + 1));
      }
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_number_attr_value(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor) {
  return loom_bytecode_number_attr_value_at_depth(numbering, attr, descriptor,
                                                  /*aggregate_depth=*/0);
}

iree_status_t loom_bytecode_number_encoding(
    loom_bytecode_numbering_t* numbering, uint16_t encoding_id) {
  if (encoding_id == 0 || encoding_id > numbering->module->encodings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding_id %u out of range (module has %" PRIhsz " encodings)",
        (unsigned)encoding_id, numbering->module->encodings.count);
  }
  uint32_t unused_id = 0;
  const loom_encoding_t* encoding =
      &numbering->module->encodings.entries[encoding_id - 1];
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
      numbering, encoding->name_id, &unused_id));
  if (encoding->alias_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, encoding->alias_id, &unused_id));
  }
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* attr = &encoding->attributes[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, attr->name_id, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attr->value, NULL));
  }
  return iree_ok_status();
}
