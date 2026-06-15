// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_dependencies.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

typedef struct loom_symbol_dependency_builder_t {
  // Module being scanned.
  const loom_module_t* module;
  // Arena receiving table storage.
  iree_arena_allocator_t* arena;
  // Mutable per-symbol edge heads.
  loom_symbol_dependency_symbol_edges_t* symbols;
  // Mutable edge storage.
  loom_symbol_dependency_edge_t* edges;
  // Number of live edge entries.
  iree_host_size_t edge_count;
  // Number of allocated edge slots.
  iree_host_size_t edge_capacity;
  // First module-root edge.
  loom_symbol_dependency_edge_id_t first_module_edge_id;
  // Number of module-root edges.
  uint32_t module_edge_count;
} loom_symbol_dependency_builder_t;

static void loom_symbol_dependency_initialize_symbol_edges(
    loom_symbol_dependency_symbol_edges_t* symbols,
    iree_host_size_t symbol_count) {
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    symbols[i] = (loom_symbol_dependency_symbol_edges_t){
        .first_outgoing_edge_id = LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID,
        .first_incoming_edge_id = LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID,
    };
  }
}

static iree_status_t loom_symbol_dependency_builder_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_dependency_builder_t* builder) {
  *builder = (loom_symbol_dependency_builder_t){
      .module = module,
      .arena = arena,
      .first_module_edge_id = LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID,
  };
  if (module->symbols.count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*builder->symbols),
                                                 (void**)&builder->symbols));
  loom_symbol_dependency_initialize_symbol_edges(builder->symbols,
                                                 module->symbols.count);
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_builder_append_edge(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_symbol_id_t target_symbol_id,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (source_symbol_id != LOOM_SYMBOL_ID_INVALID &&
      source_symbol_id >= builder->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol id %u is out of range for %" PRIhsz " symbols",
        (unsigned)source_symbol_id, builder->module->symbols.count);
  }
  if (target_symbol_id >= builder->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol id %u is out of range for %" PRIhsz " symbols",
        (unsigned)target_symbol_id, builder->module->symbols.count);
  }
  if (builder->edge_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol dependency table exceeds %u edges",
                            (unsigned)(UINT32_MAX - 1));
  }
  if (builder->edge_count >= builder->edge_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, builder->edge_count, builder->edge_count + 1,
        sizeof(*builder->edges), &builder->edge_capacity,
        (void**)&builder->edges));
  }

  const loom_symbol_dependency_edge_id_t edge_id =
      (loom_symbol_dependency_edge_id_t)builder->edge_count++;
  loom_symbol_dependency_edge_t* edge = &builder->edges[edge_id];
  *edge = (loom_symbol_dependency_edge_t){
      .source_symbol_id = source_symbol_id,
      .target_symbol_id = target_symbol_id,
      .kind = kind,
      .attr_index = attr_index,
      .user_op = user_op,
      .next_outgoing_edge_id = LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID,
      .next_incoming_edge_id =
          builder->symbols[target_symbol_id].first_incoming_edge_id,
  };
  builder->symbols[target_symbol_id].first_incoming_edge_id = edge_id;
  ++builder->symbols[target_symbol_id].incoming_count;

  if (source_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    edge->next_outgoing_edge_id = builder->first_module_edge_id;
    builder->first_module_edge_id = edge_id;
    ++builder->module_edge_count;
    return iree_ok_status();
  }
  edge->next_outgoing_edge_id =
      builder->symbols[source_symbol_id].first_outgoing_edge_id;
  builder->symbols[source_symbol_id].first_outgoing_edge_id = edge_id;
  ++builder->symbols[source_symbol_id].outgoing_count;
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_add_ref(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_symbol_ref_t target_ref,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0) {
    return iree_ok_status();
  }
  return loom_symbol_dependency_builder_append_edge(builder, source_symbol_id,
                                                    target_ref.symbol_id, kind,
                                                    attr_index, user_op);
}

static iree_status_t loom_symbol_dependency_visit_type(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_type_t type,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op);

static iree_status_t loom_symbol_dependency_visit_attr(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_attribute_t attr,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op, uint8_t dict_depth);

static iree_status_t loom_symbol_dependency_visit_encoding(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, const loom_encoding_t* encoding,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (!encoding || encoding->attribute_count == 0) return iree_ok_status();
  if (!encoding->attributes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty encoding attribute list has a NULL entry pointer");
  }
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_attr(
        builder, source_symbol_id, encoding->attributes[i].value, kind,
        attr_index, user_op, /*dict_depth=*/0));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_static_encoding(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, uint16_t encoding_id,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  if (encoding_id == 0) return iree_ok_status();
  const loom_encoding_t* encoding =
      loom_module_encoding(builder->module, encoding_id);
  if (!encoding) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "static encoding id %u is out of range for module with %" PRIhsz
        " encodings",
        (unsigned)encoding_id, builder->module->encodings.count);
  }
  return loom_symbol_dependency_visit_encoding(
      builder, source_symbol_id, encoding, kind, attr_index, user_op);
}

static iree_status_t loom_symbol_dependency_visit_type_sequence(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, const loom_type_t* types,
    uint16_t type_count, loom_symbol_dependency_edge_kind_t kind,
    uint16_t attr_index, const loom_op_t* user_op) {
  if (!types) return iree_ok_status();
  for (uint16_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_type(
        builder, source_symbol_id, types[i], kind, attr_index, user_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_type(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_type_t type,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op) {
  loom_type_kind_t type_kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(type_kind)) return iree_ok_status();

  if (loom_type_has_static_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_static_encoding(
        builder, source_symbol_id, type.encoding_id, kind, attr_index,
        user_op));
  }

  switch (type_kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return iree_ok_status();
      return loom_symbol_dependency_visit_type_sequence(
          builder, source_symbol_id, data->types,
          (uint16_t)(data->arg_count + data->result_count), kind, attr_index,
          user_op);
    }
    case LOOM_TYPE_DIALECT:
      return loom_symbol_dependency_visit_type_sequence(
          builder, source_symbol_id, loom_type_dialect_params(type),
          loom_type_dialect_param_count(type), kind, attr_index, user_op);
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_symbol_dependency_visit_attr(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_attribute_t attr,
    loom_symbol_dependency_edge_kind_t kind, uint16_t attr_index,
    const loom_op_t* user_op, uint8_t dict_depth) {
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_SYMBOL:
      return loom_symbol_dependency_add_ref(builder, source_symbol_id,
                                            loom_attr_as_symbol(attr), kind,
                                            attr_index, user_op);
    case LOOM_ATTR_TYPE:
      if (attr.type_id == LOOM_TYPE_ID_INVALID) return iree_ok_status();
      if (attr.type_id >= builder->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u is out of range for module with %" PRIhsz
            " types",
            (unsigned)attr.type_id, builder->module->types.count);
      }
      return loom_symbol_dependency_visit_type(
          builder, source_symbol_id,
          builder->module->types.entries[attr.type_id],
          LOOM_SYMBOL_DEPENDENCY_EDGE_TYPE_ATTR, attr_index, user_op);
    case LOOM_ATTR_ENCODING:
      return loom_symbol_dependency_visit_static_encoding(
          builder, source_symbol_id, attr.encoding_id,
          LOOM_SYMBOL_DEPENDENCY_EDGE_ENCODING_ATTR, attr_index, user_op);
    case LOOM_ATTR_DICT:
      if (dict_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute nesting exceeds max depth %u",
                                (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
      }
      if (attr.count == 0) return iree_ok_status();
      if (!attr.dict_entries) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute has a NULL entry pointer");
      }
      loom_symbol_dependency_edge_kind_t nested_kind = kind;
      if (kind == LOOM_SYMBOL_DEPENDENCY_EDGE_SYMBOL_ATTR ||
          kind == LOOM_SYMBOL_DEPENDENCY_EDGE_CALL ||
          kind == LOOM_SYMBOL_DEPENDENCY_EDGE_GLOBAL_ACCESS) {
        nested_kind = LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR;
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_attr(
            builder, source_symbol_id, attr.dict_entries[i].value, nested_kind,
            attr_index, user_op, (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static loom_symbol_dependency_edge_kind_t
loom_symbol_dependency_direct_attr_kind(const loom_op_vtable_t* vtable,
                                        uint8_t attr_index) {
  const loom_attr_descriptor_t* descriptor = NULL;
  if (vtable && vtable->attr_descriptors &&
      attr_index < vtable->attribute_count) {
    descriptor = &vtable->attr_descriptors[attr_index];
  }
  if (vtable && vtable->call_like &&
      attr_index == vtable->call_like->callee_attr_index) {
    return LOOM_SYMBOL_DEPENDENCY_EDGE_CALL;
  }
  if (descriptor && descriptor->symbol_ref &&
      iree_any_bit_set(descriptor->symbol_ref->interfaces,
                       LOOM_SYMBOL_INTERFACE_GLOBAL)) {
    return LOOM_SYMBOL_DEPENDENCY_EDGE_GLOBAL_ACCESS;
  }
  return LOOM_SYMBOL_DEPENDENCY_EDGE_SYMBOL_ATTR;
}

static iree_status_t loom_symbol_dependency_visit_value_type(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, loom_value_id_t value_id,
    const loom_op_t* user_op) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= builder->module->values.count) {
    return iree_ok_status();
  }
  return loom_symbol_dependency_visit_type(
      builder, source_symbol_id, builder->module->values.entries[value_id].type,
      LOOM_SYMBOL_DEPENDENCY_EDGE_VALUE_TYPE,
      LOOM_SYMBOL_DEPENDENCY_ATTR_INDEX_NONE, user_op);
}

static iree_status_t loom_symbol_dependency_visit_op_value_types(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_value_type(
        builder, source_symbol_id, operands[i], op));
  }
  const loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_value_type(
        builder, source_symbol_id, results[i], op));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_block_arg_types(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, const loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_value_type(
        builder, source_symbol_id, loom_block_arg_id(block, i),
        /*user_op=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_op_attrs(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (vtable && vtable->symbol_def &&
        i == vtable->symbol_def->name_attr_index) {
      continue;
    }
    loom_symbol_dependency_edge_kind_t kind =
        loom_symbol_dependency_direct_attr_kind(vtable, i);
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_attr(
        builder, source_symbol_id, attrs[i], kind, i, op, /*dict_depth=*/0));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_region(
    loom_symbol_dependency_builder_t* builder,
    loom_symbol_id_t source_symbol_id, const loom_region_t* region) {
  if (!region) return iree_ok_status();
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_block_arg_types(
        builder, source_symbol_id, block));
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
      loom_symbol_id_t nested_source_symbol_id = source_symbol_id;
      loom_symbol_ref_t op_symbol_ref = loom_symbol_ref_null();
      if (loom_op_defining_symbol_ref(builder->module, op, &op_symbol_ref)) {
        nested_source_symbol_id = op_symbol_ref.symbol_id;
      }
      IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_op_value_types(
          builder, nested_source_symbol_id, op));
      IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_op_attrs(
          builder, nested_source_symbol_id, op, vtable));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_region(
            builder, nested_source_symbol_id, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_module_encodings(
    loom_symbol_dependency_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->module->encodings.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_encoding(
        builder, LOOM_SYMBOL_ID_INVALID, &builder->module->encodings.entries[i],
        LOOM_SYMBOL_DEPENDENCY_EDGE_MODULE_ENCODING,
        LOOM_SYMBOL_DEPENDENCY_ATTR_INDEX_NONE, /*user_op=*/NULL));
  }
  return iree_ok_status();
}

iree_status_t loom_symbol_dependency_table_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_dependency_table_t* out_table) {
  if (!out_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol dependency output table is NULL");
  }
  *out_table = (loom_symbol_dependency_table_t){0};
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol dependency module is NULL");
  }
  if (!arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol dependency arena is NULL");
  }

  loom_symbol_dependency_builder_t builder = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_dependency_builder_initialize(module, arena, &builder));
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_region(
      &builder, LOOM_SYMBOL_ID_INVALID, module->body));
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_visit_module_encodings(&builder));

  *out_table = (loom_symbol_dependency_table_t){
      .module = module,
      .symbols = builder.symbols,
      .symbol_count = module->symbols.count,
      .edges = builder.edges,
      .edge_count = builder.edge_count,
      .first_module_edge_id = builder.first_module_edge_id,
      .module_edge_count = builder.module_edge_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_symbol_dependency_visit_scc_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_symbol_dependency_table_t* table =
      (const loom_symbol_dependency_table_t*)user_data;
  if (node >= table->symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol dependency SCC node %" PRIhsz
                            " out of range for %" PRIhsz " symbols",
                            node, table->symbol_count);
  }
  loom_symbol_dependency_edge_id_t edge_id =
      table->symbols[node].first_outgoing_edge_id;
  while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
    const loom_symbol_dependency_edge_t* edge = &table->edges[edge_id];
    IREE_RETURN_IF_ERROR(
        successor.fn(successor.user_data, edge->target_symbol_id));
    edge_id = edge->next_outgoing_edge_id;
  }
  return iree_ok_status();
}

loom_scc_graph_t loom_symbol_dependency_scc_graph(
    const loom_symbol_dependency_table_t* table) {
  return (loom_scc_graph_t){
      .node_count = table ? table->symbol_count : 0,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_symbol_dependency_visit_scc_successors, (void*)table),
  };
}
