// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/type_index.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/structural_hash.h"

// Temporary graph construction capacity; the index retains the completed edges.
typedef struct loom_bytecode_type_graph_t {
  // Serialization state and canonical source lookup.
  const loom_bytecode_type_index_t* index;
  // Scratch arena owning graph and index allocations.
  iree_arena_allocator_t* arena;
  // Ordered immediate canonical module type IDs.
  loom_type_id_t* dependencies;
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

static bool loom_bytecode_type_storage_equal(loom_type_t lhs, loom_type_t rhs) {
  return lhs.header == rhs.header && lhs.encoding_id == rhs.encoding_id &&
         lhs.encoding_flags == rhs.encoding_flags &&
         lhs.dims[0] == rhs.dims[0] && lhs.dims[1] == rhs.dims[1];
}

loom_type_id_t loom_bytecode_type_index_lookup(
    const loom_bytecode_type_index_t* index, loom_type_t type) {
  if (index->slot_capacity == 0) {
    return LOOM_TYPE_ID_INVALID;
  }
  const iree_host_size_t mask = index->slot_capacity - 1;
  for (iree_host_size_t slot = loom_bytecode_type_storage_hash(type) & mask;
       index->slots[slot] != LOOM_TYPE_ID_INVALID; slot = (slot + 1) & mask) {
    const loom_type_id_t type_id = index->slots[slot];
    if (loom_bytecode_type_storage_equal(
            loom_type_table_get(&index->module->types, type_id), type)) {
      return type_id;
    }
  }
  return LOOM_TYPE_ID_INVALID;
}

static iree_status_t loom_bytecode_type_graph_add_dependency(
    loom_bytecode_type_graph_t* graph, loom_type_id_t type_id) {
  if (graph->count >= graph->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        graph->arena, graph->count, /*minimum_capacity=*/16,
        sizeof(*graph->dependencies), &graph->capacity,
        (void**)&graph->dependencies));
  }
  graph->dependencies[graph->count++] = type_id;
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
        return loom_bytecode_type_graph_add_dependency(graph, attr->type_id);
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
    status = loom_bytecode_type_graph_add_dependency(
        graph, loom_bytecode_type_index_lookup(graph->index, children[i]));
  }
  return status;
}

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
  // Source construction retains the complete canonical child closure. The
  // writer stores only serialization facts alongside those stable identities.
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->types.count,
                                                 sizeof(*out_index->nodes),
                                                 (void**)&out_index->nodes));
  out_index->slot_capacity = iree_max(
      16, iree_host_size_next_power_of_two((module->types.count * 4 + 2) / 3));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_index->slot_capacity, sizeof(*out_index->slots),
      (void**)&out_index->slots));
  memset(out_index->slots, 0xFF,
         out_index->slot_capacity * sizeof(*out_index->slots));
  const iree_host_size_t mask = out_index->slot_capacity - 1;
  for (loom_type_id_t type_id = 0; type_id < module->types.count; ++type_id) {
    iree_host_size_t slot = loom_bytecode_type_storage_hash(
                                loom_type_table_get(&module->types, type_id)) &
                            mask;
    while (out_index->slots[slot] != LOOM_TYPE_ID_INVALID) {
      slot = (slot + 1) & mask;
    }
    out_index->slots[slot] = type_id;
  }
  for (iree_host_size_t i = 0; i < module->types.count; ++i) {
    const loom_type_t type = loom_type_table_get(&module->types, i);
    out_index->nodes[i] = (loom_bytecode_type_node_t){
        .has_bindings = loom_type_table_dependencies(&module->types, i) != 0};
    iree_host_size_t begin = graph.count;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_graph_add_children(&graph, type));
    out_index->nodes[i].dependencies.explicit_count = graph.count - begin;
    if (loom_type_is_shaped(type)) {
      IREE_RETURN_IF_ERROR(loom_bytecode_type_graph_add_dependency(
          &graph,
          loom_bytecode_type_index_lookup(
              out_index, loom_type_scalar(loom_type_element_type(type)))));
    }
    out_index->nodes[i].dependencies.begin = begin;
    out_index->nodes[i].dependencies.count = graph.count - begin;
  }
  out_index->dependencies = graph.dependencies;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->types.count,
                                                 sizeof(*out_index->stack),
                                                 (void**)&out_index->stack));
  return iree_ok_status();
}
