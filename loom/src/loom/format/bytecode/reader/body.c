// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/body.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/ops/op_defs.h"

typedef loom_bytecode_body_materializer_t
    loom_bytecode_body_policy_materializer_t;
typedef loom_bytecode_value_scope_t loom_bytecode_body_policy_value_scope_t;

static void loom_bytecode_body_policy_initialize_value_scope(
    loom_bytecode_body_policy_materializer_t* materializer,
    iree_arena_allocator_t* arena, iree_string_view_t symbol_name,
    uint64_t payload_offset, loom_value_id_t* value_map,
    iree_host_size_t value_count,
    loom_bytecode_body_policy_value_scope_t* out_scope) {
  *out_scope = (loom_bytecode_body_policy_value_scope_t){
      .decoder = materializer->attributes.decoder,
      .module_view = materializer->attributes.module_view,
      .output_module = materializer->attributes.output_module,
      .arena = arena,
      .symbol_name = symbol_name,
      .payload_offset = payload_offset,
      .value_map = value_map,
      .value_capacity = value_count,
  };
}

static loom_bytecode_reader_decoder_t* loom_bytecode_body_policy_decoder(
    loom_bytecode_body_policy_materializer_t* materializer) {
  return materializer->attributes.decoder;
}

static iree_arena_block_pool_t* loom_bytecode_body_policy_block_pool(
    loom_bytecode_body_policy_materializer_t* materializer) {
  return materializer->block_pool;
}

static const loom_low_repr_environment_t*
loom_bytecode_body_policy_low_repr_environment(
    loom_bytecode_body_policy_materializer_t* materializer) {
  return materializer->low_repr_environment;
}

static iree_status_t loom_bytecode_body_policy_resolve_source_string(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= value_scope->module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(value_scope->module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(value_scope->decoder,
                                           LOOM_ERR_BYTECODE_010, params,
                                           IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_string = value_scope->module_view->strings.values[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_policy_project_string(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    loom_string_id_t* out_string_id) {
  iree_string_view_t unused_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_resolve_source_string(
      value_scope, string_id, field_name, offset, &unused_string));
  *out_string_id = (loom_string_id_t)string_id;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_policy_materialize_type(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t type_id,
    uint64_t offset, loom_type_id_t* out_type_id, loom_type_t* out_type) {
  if (type_id >= value_scope->output_module->types.count) {
    return loom_bytecode_reader_emit_table_ref(
        value_scope->decoder, IREE_SV("TYPES"), type_id,
        value_scope->output_module->types.count, offset);
  }
  *out_type_id = (loom_type_id_t)type_id;
  *out_type = value_scope->output_module->types.entries[type_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_policy_materialize_location(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t location_id,
    uint64_t offset, loom_location_id_t* out_location_id) {
  if (location_id >= value_scope->module_view->locations.count) {
    return loom_bytecode_reader_emit_table_ref(
        value_scope->decoder, IREE_SV("LOCATIONS"), location_id,
        value_scope->module_view->locations.count, offset);
  }
  *out_location_id = (loom_location_id_t)location_id;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_policy_resolve_op(
    loom_bytecode_body_policy_value_scope_t* value_scope,
    uint64_t op_table_index_plus1, uint64_t offset,
    const loom_op_vtable_t** out_vtable, loom_op_kind_t* out_kind) {
  if (op_table_index_plus1 == 0 ||
      op_table_index_plus1 > value_scope->module_view->ops.count) {
    return loom_bytecode_reader_emit_table_ref(
        value_scope->decoder, IREE_SV("OPS"), op_table_index_plus1,
        value_scope->module_view->ops.count, offset);
  }
  const iree_host_size_t op_ordinal =
      (iree_host_size_t)op_table_index_plus1 - 1;
  *out_vtable = value_scope->module_view->ops.values[op_ordinal];
  *out_kind = value_scope->module_view->ops.kinds[op_ordinal];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_policy_materialize_attribute_ssa(
    loom_bytecode_body_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope) {
  return loom_bytecode_attribute_materialize_ssa(
      &materializer->attributes, cursor, descriptor, kind, out_attr,
      materializer->attributes.module_view->types.count, ssa_scope);
}

#define LOOM_BYTECODE_BODY_VALUE_SCOPE_INITIALIZE_FRESH \
  loom_bytecode_value_scope_initialize_fresh
#define LOOM_BYTECODE_BODY_VALUE_SCOPE_MATERIALIZE_DEFINITION \
  loom_bytecode_value_scope_materialize_definition
#define LOOM_BYTECODE_BODY_MATERIALIZE_SYMBOL_REGIONS \
  loom_bytecode_body_materialize_symbol_regions
#define LOOM_BYTECODE_BODY_DEFINE_SUMMARY 1
#include "loom/format/bytecode/reader/body_materializer_impl.inl"
#undef LOOM_BYTECODE_BODY_DEFINE_SUMMARY
#undef LOOM_BYTECODE_BODY_MATERIALIZE_SYMBOL_REGIONS
#undef LOOM_BYTECODE_BODY_VALUE_SCOPE_MATERIALIZE_DEFINITION
#undef LOOM_BYTECODE_BODY_VALUE_SCOPE_INITIALIZE_FRESH
