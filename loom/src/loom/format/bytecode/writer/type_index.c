// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/type_index.h"

#include <string.h>

#include "loom/format/bytecode/format.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/structural_hash.h"

// Temporary graph construction capacity; the index retains the completed edges.
typedef struct loom_bytecode_type_graph_t {
  // Index receiving exact storage and wire-equivalence facts.
  loom_bytecode_type_index_t* index;
  // Scratch arena owning graph and index allocations.
  iree_arena_allocator_t* arena;
  // Ordered immediate dependency node IDs.
  uint32_t* dependencies;
  // Number of populated dependencies.
  iree_host_size_t count;
  // Allocated dependency capacity.
  iree_host_size_t capacity;
} loom_bytecode_type_graph_t;

static uint32_t loom_bytecode_type_storage_hash(loom_type_t type) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(hash, type.header);
  hash = loom_structural_hash_mix_u32(
      hash, type.encoding_id | ((uint32_t)type.encoding_flags << 16));
  hash = loom_structural_hash_mix_u64(hash, type.dims[0]);
  hash = loom_structural_hash_mix_u64(hash, type.dims[1]);
  return loom_structural_hash_finalize(hash);
}

static bool loom_bytecode_type_storage_equal(loom_type_t a, loom_type_t b) {
  return a.header == b.header && a.encoding_id == b.encoding_id &&
         a.encoding_flags == b.encoding_flags && a.dims[0] == b.dims[0] &&
         a.dims[1] == b.dims[1];
}

static uint32_t loom_bytecode_type_storage_lookup(
    const loom_bytecode_type_index_t* index, loom_type_t type, uint32_t hash) {
  if (index->slot_capacity == 0) {
    return UINT32_MAX;
  }
  iree_host_size_t mask = index->slot_capacity - 1;
  for (iree_host_size_t slot = hash & mask; index->slots[slot] != UINT32_MAX;
       slot = (slot + 1) & mask) {
    uint32_t node = index->slots[slot];
    if (index->nodes[node].storage_hash == hash &&
        loom_bytecode_type_storage_equal(index->nodes[node].type, type)) {
      return node;
    }
  }
  return UINT32_MAX;
}

static iree_status_t loom_bytecode_type_storage_insert(
    loom_bytecode_type_graph_t* graph, loom_type_t type, uint32_t* out_node) {
  loom_bytecode_type_index_t* index = graph->index;
  const uint32_t hash = loom_bytecode_type_storage_hash(type);
  uint32_t existing = loom_bytecode_type_storage_lookup(index, type, hash);
  if (existing != UINT32_MAX) {
    *out_node = existing;
    return iree_ok_status();
  }
  if (index->count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "bytecode type storage graph exceeds 32-bit indices");
  }
  if (index->count >= index->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        graph->arena, index->count, /*minimum_capacity=*/16,
        sizeof(*index->nodes), &index->capacity, (void**)&index->nodes));
  }
  if (index->count + 1 > index->slot_capacity * 3 / 4) {
    iree_host_size_t capacity =
        index->slot_capacity ? index->slot_capacity * 2 : 16;
    uint32_t* slots = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        graph->arena, capacity, sizeof(*slots), (void**)&slots));
    memset(slots, 0xFF, capacity * sizeof(*slots));
    for (iree_host_size_t i = 0; i < index->count; ++i) {
      iree_host_size_t slot = index->nodes[i].storage_hash & (capacity - 1);
      while (slots[slot] != UINT32_MAX) {
        slot = (slot + 1) & (capacity - 1);
      }
      slots[slot] = (uint32_t)i;
    }
    index->slots = slots;
    index->slot_capacity = capacity;
  }
  uint32_t node = (uint32_t)index->count++;
  index->nodes[node] = (loom_bytecode_type_node_t){
      .type = type,
      .storage_hash = hash,
      .module_index = LOOM_TYPE_ID_INVALID,
      .representative = UINT32_MAX,
  };
  iree_host_size_t slot = hash & (index->slot_capacity - 1);
  while (index->slots[slot] != UINT32_MAX) {
    slot = (slot + 1) & (index->slot_capacity - 1);
  }
  index->slots[slot] = node;
  *out_node = node;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_graph_add_dependency(
    loom_bytecode_type_graph_t* graph, loom_type_t type) {
  uint32_t node = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_storage_insert(graph, type, &node));
  if (graph->count >= graph->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        graph->arena, graph->count, /*minimum_capacity=*/16,
        sizeof(*graph->dependencies), &graph->capacity,
        (void**)&graph->dependencies));
  }
  graph->dependencies[graph->count++] = node;
  return iree_ok_status();
}

// Aggregate attributes have bounded nesting. TYPE leaves become graph edges,
// never recursive descent into the referenced type's payload.
static iree_status_t loom_bytecode_type_graph_add_attribute(
    loom_bytecode_type_graph_t* graph, const loom_attribute_t* attr,
    uint8_t depth) {
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_TYPE:
      if (attr->type_id < graph->index->module->types.count) {
        return loom_bytecode_type_graph_add_dependency(
            graph, graph->index->module->types.entries[attr->type_id]);
      }
      return iree_ok_status();
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_PARAMETERIZED:
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      break;
    default:
      return iree_ok_status();
  }
  if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < attr->count && iree_status_is_ok(status); ++i) {
    const loom_attribute_t* child = NULL;
    switch ((loom_attr_kind_t)attr->kind) {
      case LOOM_ATTR_DICT:
        child = &attr->dict_entries[i].value;
        break;
      case LOOM_ATTR_PARAMETERIZED:
        child = &attr->parameterized_slots[i];
        break;
      default:
        child = &attr->parameterized_array[i];
        break;
    }
    status = loom_bytecode_type_graph_add_attribute(graph, child, depth + 1);
  }
  return status;
}

static iree_status_t loom_bytecode_type_graph_add_children(
    loom_bytecode_type_graph_t* graph, loom_type_t type) {
  const loom_type_t* children = NULL;
  iree_host_size_t count = 0;
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
      return loom_bytecode_type_graph_add_dependency(
          graph, loom_type_scalar(loom_type_element_type(type)));
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (data) {
        children = data->types;
        count = (iree_host_size_t)data->arg_count + data->result_count;
      }
      break;
    }
    case LOOM_TYPE_DIALECT:
      children = loom_type_dialect_params(type);
      count = children ? loom_type_dialect_param_count(type) : 0;
      break;
    case LOOM_TYPE_REGISTER:
      children = loom_type_register_value_type(type);
      count = children ? 1 : 0;
      break;
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      if (!parameters) {
        return iree_ok_status();
      }
      iree_status_t status = iree_ok_status();
      for (uint8_t i = 0; i < loom_type_parameterized_parameter_count(type) &&
                          iree_status_is_ok(status);
           ++i) {
        status =
            loom_bytecode_type_graph_add_attribute(graph, &parameters[i], 0);
      }
      return status;
    }
    default:
      break;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_bytecode_type_graph_add_dependency(graph, children[i]);
  }
  return status;
}

// Every queried child was discovered by graph construction and completed by
// postorder before its parent is classified.
static uint32_t loom_bytecode_type_class(
    const loom_bytecode_type_index_t* index, loom_type_t type) {
  uint32_t node = loom_bytecode_type_storage_lookup(
      index, type, loom_bytecode_type_storage_hash(type));
  return index->nodes[node].representative;
}

static uint32_t loom_bytecode_attr_wire_hash(
    const loom_bytecode_type_index_t* index, const loom_attribute_t* attr,
    uint8_t aggregate_depth) {
  const loom_module_t* module = index->module;
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_TYPE:
      if (attr->type_id >= module->types.count) {
        return loom_attribute_hash(attr);
      }
      return loom_bytecode_type_class(index,
                                      module->types.entries[attr->type_id]);
    case LOOM_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_hash(attr);
      }
      uint32_t hash = loom_structural_hash_mix_u8(
          loom_structural_hash_initialize(), attr->kind);
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash =
            loom_structural_hash_mix_u32(hash, attr->dict_entries[i].name_id);
        hash = loom_structural_hash_mix_u32(
            hash,
            loom_bytecode_attr_wire_hash(index, &attr->dict_entries[i].value,
                                         aggregate_depth + 1));
      }
      return hash;
    }
    case LOOM_ATTR_PARAMETERIZED: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_hash(attr);
      }
      uint32_t hash = loom_structural_hash_mix_u8(
          loom_structural_hash_initialize(), attr->kind);
      hash = loom_structural_hash_mix_u32(hash, attr->reserved_1);
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash,
            loom_bytecode_attr_wire_hash(index, &attr->parameterized_slots[i],
                                         aggregate_depth + 1));
      }
      return hash;
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_hash(attr);
      }
      uint32_t hash = loom_structural_hash_mix_u8(
          loom_structural_hash_initialize(), attr->kind);
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash,
            loom_bytecode_attr_wire_hash(index, &attr->parameterized_array[i],
                                         aggregate_depth + 1));
      }
      return hash;
    }
    default:
      return loom_attribute_hash(attr);
  }
}

static bool loom_bytecode_attr_wire_equal(
    const loom_bytecode_type_index_t* index, const loom_attribute_t* a,
    const loom_attribute_t* b, uint8_t aggregate_depth) {
  const loom_module_t* module = index->module;
  if (a->kind != b->kind) {
    return false;
  }
  switch ((loom_attr_kind_t)a->kind) {
    case LOOM_ATTR_TYPE:
      if (a->type_id >= module->types.count ||
          b->type_id >= module->types.count) {
        return a->type_id == b->type_id;
      }
      return loom_bytecode_type_class(index,
                                      module->types.entries[a->type_id]) ==
             loom_bytecode_type_class(index, module->types.entries[b->type_id]);
    case LOOM_ATTR_DICT:
      if (a->count != b->count) {
        return false;
      }
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_equal(a, b);
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (a->dict_entries[i].name_id != b->dict_entries[i].name_id ||
            !loom_bytecode_attr_wire_equal(index, &a->dict_entries[i].value,
                                           &b->dict_entries[i].value,
                                           aggregate_depth + 1)) {
          return false;
        }
      }
      return true;
    case LOOM_ATTR_PARAMETERIZED:
      if (a->reserved_1 != b->reserved_1 || a->count != b->count) {
        return false;
      }
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_equal(a, b);
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (!loom_bytecode_attr_wire_equal(index, &a->parameterized_slots[i],
                                           &b->parameterized_slots[i],
                                           aggregate_depth + 1)) {
          return false;
        }
      }
      return true;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (a->count != b->count) {
        return false;
      }
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return loom_attribute_equal(a, b);
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (!loom_bytecode_attr_wire_equal(index, &a->parameterized_array[i],
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

static uint32_t loom_bytecode_type_wire_hash_fields(
    const loom_bytecode_type_index_t* index, loom_type_t type) {
  uint32_t hash = loom_structural_hash_initialize();
  loom_type_kind_t kind = loom_type_kind(type);
  hash = loom_structural_hash_mix_u32(
      hash, (uint32_t)kind | ((uint32_t)loom_type_element_type(type) << 8) |
                ((uint32_t)loom_type_rank(type) << 16));

  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      if (loom_type_has_ssa_encoding(type)) {
        hash = loom_structural_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA);
      } else if (loom_type_has_static_encoding(type)) {
        hash = loom_structural_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC);
        hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
      } else {
        hash = loom_structural_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE);
      }
      for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
        uint64_t dim = loom_type_dim(type, i);
        if (loom_dim_is_dynamic(dim)) {
          hash = loom_structural_hash_mix_u8(hash, 1);
        } else {
          hash = loom_structural_hash_mix_u8(hash, 0);
          hash = loom_structural_hash_mix_u64(hash, dim);
        }
      }
      return hash;
    }
    case LOOM_TYPE_POOL: {
      uint64_t dim = loom_type_dim(type, 0);
      if (loom_dim_is_dynamic(dim)) {
        hash = loom_structural_hash_mix_u8(hash, 1);
      } else {
        hash = loom_structural_hash_mix_u8(hash, 0);
        hash = loom_structural_hash_mix_u64(hash, dim);
      }
      return hash;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return hash;
      }
      hash = loom_structural_hash_mix_u16(hash, data->arg_count);
      hash = loom_structural_hash_mix_u16(hash, data->result_count);
      iree_host_size_t type_count =
          (iree_host_size_t)data->arg_count + data->result_count;
      for (iree_host_size_t i = 0; i < type_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_bytecode_type_class(index, data->types[i]));
      }
      return hash;
    }
    case LOOM_TYPE_DIALECT: {
      hash =
          loom_structural_hash_mix_u32(hash, loom_type_dialect_name_id(type));
      uint16_t param_count = loom_type_dialect_param_count(type);
      hash = loom_structural_hash_mix_u16(hash, param_count);
      const loom_type_t* params = loom_type_dialect_params(type);
      for (uint16_t i = 0; params && i < param_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_bytecode_type_class(index, params[i]));
      }
      return hash;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      if (!descriptor) {
        return hash;
      }
      const iree_string_view_t family_name =
          loom_bstring_view(descriptor->name);
      hash = loom_structural_hash_mix_bytes(hash, family_name.data,
                                            family_name.size);
      const uint8_t parameter_count =
          loom_type_parameterized_parameter_count(type);
      hash = loom_structural_hash_mix_u8(hash, parameter_count);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      for (uint8_t i = 0; parameters && i < parameter_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_bytecode_attr_wire_hash(index, &parameters[i], 0));
      }
      return hash;
    }
    case LOOM_TYPE_REGISTER: {
      hash =
          loom_structural_hash_mix_u64(hash, loom_type_register_payload0(type));
      hash =
          loom_structural_hash_mix_u64(hash, loom_type_register_payload1(type));
      const loom_type_t* value_type = loom_type_register_value_type(type);
      hash = loom_structural_hash_mix_u8(hash, value_type ? 1 : 0);
      return value_type
                 ? loom_structural_hash_mix_u32(
                       hash, loom_bytecode_type_class(index, *value_type))
                 : hash;
    }
    case LOOM_TYPE_STORAGE:
      return loom_structural_hash_mix_u8(
          hash, (uint8_t)loom_type_storage_space(type));
    case LOOM_TYPE_ENCODING:
      return loom_structural_hash_mix_u8(
          hash, (uint8_t)loom_type_encoding_role(type));
    default:
      return hash;
  }
}

static uint32_t loom_bytecode_type_wire_hash(
    const loom_bytecode_type_index_t* index, loom_type_t type) {
  return loom_structural_hash_finalize(
      loom_bytecode_type_wire_hash_fields(index, type));
}

static bool loom_bytecode_type_dim_wire_equal(uint64_t a, uint64_t b) {
  bool a_dynamic = loom_dim_is_dynamic(a);
  bool b_dynamic = loom_dim_is_dynamic(b);
  if (a_dynamic || b_dynamic) {
    return a_dynamic == b_dynamic;
  }
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

static bool loom_bytecode_type_wire_equal(
    const loom_bytecode_type_index_t* index, loom_type_t a, loom_type_t b) {
  loom_type_kind_t kind = loom_type_kind(a);
  if (kind != loom_type_kind(b)) {
    return false;
  }
  if (loom_type_element_type(a) != loom_type_element_type(b)) {
    return false;
  }
  if (loom_type_rank(a) != loom_type_rank(b)) {
    return false;
  }

  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
      if (!loom_bytecode_type_encoding_wire_equal(a, b)) {
        return false;
      }
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
      if (!a_data || !b_data) {
        return a_data == b_data;
      }
      if (a_data->arg_count != b_data->arg_count ||
          a_data->result_count != b_data->result_count) {
        return false;
      }
      iree_host_size_t type_count =
          (iree_host_size_t)a_data->arg_count + a_data->result_count;
      for (iree_host_size_t i = 0; i < type_count; ++i) {
        if (loom_bytecode_type_class(index, a_data->types[i]) !=
            loom_bytecode_type_class(index, b_data->types[i])) {
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
      if (param_count != loom_type_dialect_param_count(b)) {
        return false;
      }
      const loom_type_t* a_params = loom_type_dialect_params(a);
      const loom_type_t* b_params = loom_type_dialect_params(b);
      if (!a_params || !b_params) {
        return a_params == b_params;
      }
      for (uint16_t i = 0; i < param_count; ++i) {
        if (loom_bytecode_type_class(index, a_params[i]) !=
            loom_bytecode_type_class(index, b_params[i])) {
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
      if (!a_parameters || !b_parameters) {
        return a_parameters == b_parameters;
      }
      for (uint8_t i = 0; i < parameter_count; ++i) {
        if (!loom_bytecode_attr_wire_equal(index, &a_parameters[i],
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
      return loom_bytecode_type_class(index, *a_value_type) ==
             loom_bytecode_type_class(index, *b_value_type);
    }
    case LOOM_TYPE_STORAGE:
      return loom_type_storage_space(a) == loom_type_storage_space(b);
    case LOOM_TYPE_ENCODING:
      return loom_type_encoding_role(a) == loom_type_encoding_role(b);
    default:
      return true;
  }
}

typedef struct loom_bytecode_type_frame_t {
  // Node whose remaining immediate dependencies are being completed.
  uint32_t node;
  // Next immediate dependency to visit.
  iree_host_size_t next_dependency;
} loom_bytecode_type_frame_t;

iree_status_t loom_bytecode_type_index_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_type_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  out_index->module = module;
  if (module->types.count == 0) {
    return iree_ok_status();
  }
  loom_bytecode_type_graph_t graph = {
      .index = out_index,
      .arena = arena,
  };
  // Each canonical module type needs one storage node. Start with the table
  // size instead of abandoning successively doubled arena arrays.
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->types.count,
                                                 sizeof(*out_index->nodes),
                                                 (void**)&out_index->nodes));
  out_index->capacity = module->types.count;
  out_index->slot_capacity =
      iree_host_size_next_power_of_two((module->types.count * 4 + 2) / 3);
  if (out_index->slot_capacity < 16) {
    out_index->slot_capacity = 16;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_index->slot_capacity, sizeof(*out_index->slots),
      (void**)&out_index->slots));
  memset(out_index->slots, 0xFF,
         out_index->slot_capacity * sizeof(*out_index->slots));
  for (iree_host_size_t i = 0; i < module->types.count; ++i) {
    uint32_t node = UINT32_MAX;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_storage_insert(
        &graph, module->types.entries[i], &node));
    if (out_index->nodes[node].module_index == LOOM_TYPE_ID_INVALID) {
      out_index->nodes[node].module_index = (loom_type_id_t)i;
    }
  }
  for (iree_host_size_t i = 0; i < out_index->count; ++i) {
    iree_host_size_t begin = graph.count;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_graph_add_children(
        &graph, out_index->nodes[i].type));
    out_index->nodes[i].dependencies.begin = begin;
    out_index->nodes[i].dependencies.count = graph.count - begin;
  }
  out_index->dependencies = graph.dependencies;

  iree_host_size_t class_capacity =
      iree_host_size_next_power_of_two((out_index->count * 4 + 2) / 3);
  if (class_capacity < 16) {
    class_capacity = 16;
  }
  uint32_t* classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_capacity, sizeof(*classes), (void**)&classes));
  memset(classes, 0xFF, class_capacity * sizeof(*classes));
  loom_bytecode_type_frame_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_index->count, sizeof(*stack), (void**)&stack));

  // Module-owned types are acyclic. Explicit postorder bounds the C stack and
  // classifies each shared node once regardless of incoming reference count.
  for (iree_host_size_t root = 0; root < out_index->count; ++root) {
    if (out_index->nodes[root].representative != UINT32_MAX) {
      continue;
    }
    iree_host_size_t depth = 1;
    stack[0] = (loom_bytecode_type_frame_t){.node = (uint32_t)root};
    while (depth != 0) {
      loom_bytecode_type_frame_t* frame = &stack[depth - 1];
      loom_bytecode_type_node_t* node = &out_index->nodes[frame->node];
      if (frame->next_dependency < node->dependencies.count) {
        uint32_t child = graph.dependencies[node->dependencies.begin +
                                            frame->next_dependency++];
        if (out_index->nodes[child].representative == UINT32_MAX) {
          stack[depth++] = (loom_bytecode_type_frame_t){.node = child};
        }
        continue;
      }
      node->wire_hash = loom_bytecode_type_wire_hash(out_index, node->type);
      iree_host_size_t slot = node->wire_hash & (class_capacity - 1);
      while (classes[slot] != UINT32_MAX) {
        const loom_bytecode_type_node_t* other =
            &out_index->nodes[classes[slot]];
        if (node->wire_hash == other->wire_hash &&
            loom_bytecode_type_wire_equal(out_index, node->type, other->type)) {
          break;
        }
        slot = (slot + 1) & (class_capacity - 1);
      }
      if (classes[slot] == UINT32_MAX) {
        classes[slot] = frame->node;
      }
      node->representative = classes[slot];
      // Scoped bindings may distinguish wire-equivalent module types. Keep the
      // earliest module entry regardless of postorder discovery order.
      loom_bytecode_type_node_t* representative =
          &out_index->nodes[node->representative];
      if (node->module_index < representative->module_index) {
        representative->module_index = node->module_index;
      }
      --depth;
    }
  }
  return iree_ok_status();
}

const loom_bytecode_type_node_t* loom_bytecode_type_index_lookup_node(
    const loom_bytecode_type_index_t* index, loom_type_t type) {
  uint32_t node = loom_bytecode_type_storage_lookup(
      index, type, loom_bytecode_type_storage_hash(type));
  return node == UINT32_MAX ? NULL : &index->nodes[node];
}

loom_type_id_t loom_bytecode_type_index_lookup(
    const loom_bytecode_type_index_t* index, loom_type_t type) {
  const loom_bytecode_type_node_t* node =
      loom_bytecode_type_index_lookup_node(index, type);
  return !node ? LOOM_TYPE_ID_INVALID
               : index->nodes[node->representative].module_index;
}
