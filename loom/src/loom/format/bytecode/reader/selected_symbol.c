// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_symbol.h"

#include <string.h>

#include "loom/format/bytecode/function_header.h"
#include "loom/format/bytecode/reader/selected_attribute.h"
#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/format/bytecode/reader/symbol_schema.h"
#include "loom/ops/op_defs.h"

typedef loom_bytecode_selected_symbol_materializer_t
    loom_bytecode_symbol_policy_materializer_t;
typedef loom_bytecode_selected_value_scope_t
    loom_bytecode_symbol_policy_value_scope_t;
typedef loom_bytecode_selected_symbol_t
    loom_bytecode_symbol_policy_body_source_t;

static const loom_bytecode_symbol_metadata_t*
loom_bytecode_selected_symbol_metadata(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  IREE_ASSERT(symbol_ordinal < materializer->tables->metadata->symbol_count);
  return &materializer->tables->metadata->symbols[symbol_ordinal];
}

static uint64_t loom_bytecode_symbol_policy_source_name_id(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  return loom_bytecode_selected_symbol_metadata(materializer, symbol_ordinal)
      ->name_string_index;
}

static uint8_t loom_bytecode_symbol_policy_kind(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  return (uint8_t)loom_bytecode_selected_symbol_metadata(materializer,
                                                         symbol_ordinal)
      ->kind;
}

static uint16_t loom_bytecode_symbol_policy_flags(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  return (uint16_t)loom_bytecode_selected_symbol_metadata(materializer,
                                                          symbol_ordinal)
      ->flags;
}

static iree_status_t loom_bytecode_symbol_policy_resolve_source_string(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    uint64_t source_string_id, iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  (void)field_name;
  (void)offset;
  IREE_ASSERT(source_string_id < materializer->tables->metadata->strings.count);
  *out_string =
      materializer->tables->metadata->strings.values[source_string_id];
  return iree_ok_status();
}

static iree_string_view_t loom_bytecode_symbol_policy_source_string(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    uint32_t source_string_id) {
  IREE_ASSERT(source_string_id < materializer->tables->metadata->strings.count);
  return materializer->tables->metadata->strings.values[source_string_id];
}

static iree_status_t loom_bytecode_symbol_policy_project_string(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    uint64_t source_string_id, iree_string_view_t field_name, uint64_t offset,
    loom_string_id_t* out_string_id) {
  (void)field_name;
  (void)offset;
  IREE_ASSERT(source_string_id < materializer->tables->metadata->strings.count);
  return loom_bytecode_selected_table_intern_string(
      materializer->tables, (uint32_t)source_string_id, out_string_id);
}

static uint16_t loom_bytecode_symbol_policy_lookup_symbol(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    uint32_t source_name_id) {
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  const bool found = loom_bytecode_selected_table_lookup_symbol(
      materializer->tables, source_name_id, &target_ref);
  IREE_ASSERT(found);
  return target_ref.symbol_id;
}

static iree_status_t loom_bytecode_symbol_policy_project_symbol_ordinal(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    uint32_t source_symbol_ordinal, loom_symbol_ref_t* out_target_ref) {
  IREE_ASSERT(source_symbol_ordinal <
              materializer->tables->metadata->symbol_count);
  const uint32_t source_name_ordinal =
      materializer->tables->metadata->symbols[source_symbol_ordinal]
          .name_string_index;
  bool found = false;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_resolve_symbol(
      materializer->tables, source_name_ordinal, out_target_ref, &found));
  if (!found) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected function header references an unprojected symbol");
  }
  return iree_ok_status();
}

static void loom_bytecode_symbol_policy_resolve_defining_op(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal, uint64_t op_table_index_plus1,
    const loom_op_vtable_t** out_vtable, loom_op_kind_t* out_kind) {
  const loom_bytecode_symbol_metadata_t* metadata =
      loom_bytecode_selected_symbol_metadata(materializer, symbol_ordinal);
  IREE_ASSERT(op_table_index_plus1 != 0);
  IREE_ASSERT(metadata->defining_op_ordinal == op_table_index_plus1 - 1);
  IREE_ASSERT(metadata->defining_op_ordinal <
              materializer->tables->metadata->ops.count);
  const loom_bytecode_op_metadata_t* op_metadata =
      &materializer->tables->metadata->ops
           .entries[metadata->defining_op_ordinal];
  *out_vtable = op_metadata->vtable;
  *out_kind = op_metadata->kind;
}

static iree_status_t loom_bytecode_symbol_policy_value_scope_initialize_fresh(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_arena_allocator_t* arena, iree_string_view_t symbol_name,
    uint64_t payload_offset, loom_value_id_t* value_map,
    iree_host_size_t value_count,
    loom_bytecode_symbol_policy_value_scope_t* out_scope) {
  return loom_bytecode_selected_value_scope_initialize_fresh(
      &materializer->body_materializer, arena, symbol_name, payload_offset,
      value_map, value_count, out_scope);
}

static iree_status_t
loom_bytecode_symbol_policy_value_scope_materialize_definition(
    loom_bytecode_symbol_policy_value_scope_t* value_scope,
    loom_bytecode_reader_cursor_t* cursor, loom_value_id_t* out_value_id) {
  return loom_bytecode_selected_value_scope_materialize_definition(
      value_scope, cursor, out_value_id);
}

static iree_status_t loom_bytecode_symbol_policy_materialize_attribute_named(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr) {
  return loom_bytecode_selected_attribute_materialize_named(
      materializer->tables, cursor, descriptor, kind, out_attr,
      materializer->tables->metadata->types.count);
}

static iree_status_t loom_bytecode_symbol_policy_materialize_attribute_ssa(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope) {
  return loom_bytecode_selected_attribute_materialize_ssa(
      materializer->tables, cursor, descriptor, kind, out_attr,
      materializer->tables->metadata->types.count, ssa_scope);
}

static iree_status_t loom_bytecode_symbol_policy_materialize_region(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_string_view_t symbol_name,
    const loom_bytecode_symbol_policy_body_source_t* body_source,
    iree_host_size_t symbol_ordinal, uint8_t region_payload_ordinal,
    const loom_bytecode_region_payload_metadata_t* header_payload,
    loom_builder_t* builder, loom_op_t* parent_op,
    loom_bytecode_region_materialization_flags_t flags,
    const loom_value_id_t* predefined_values, uint16_t predefined_value_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set) {
  IREE_ASSERT(body_source->source_ordinal == symbol_ordinal);
  const loom_bytecode_symbol_metadata_t* symbol_metadata =
      loom_bytecode_selected_symbol_metadata(materializer, symbol_ordinal);
  IREE_ASSERT(region_payload_ordinal < symbol_metadata->region_payload_count);
  const loom_bytecode_region_payload_metadata_t* payload =
      &materializer->tables->metadata
           ->region_payloads[symbol_metadata->first_region_payload_index +
                             region_payload_ordinal];
  IREE_ASSERT(payload->region_index == header_payload->region_index);
  IREE_ASSERT(payload->offset == header_payload->offset);
  IREE_ASSERT(payload->length == header_payload->length);
  IREE_ASSERT(region_payload_ordinal < body_source->region_summary_count);
  IREE_ASSERT(payload->absolute_offset <=
              materializer->tables->bytecode.data_length);
  IREE_ASSERT(payload->length <= materializer->tables->bytecode.data_length -
                                     payload->absolute_offset);
  const iree_const_byte_span_t payload_bytes =
      iree_make_const_byte_span(materializer->tables->bytecode.data +
                                    (iree_host_size_t)payload->absolute_offset,
                                payload->length);
  return loom_bytecode_selected_body_materialize_region(
      &materializer->body_materializer, symbol_name, payload_bytes,
      payload->absolute_offset,
      &body_source->region_summaries[region_payload_ordinal], builder,
      parent_op, payload->region_index, flags, predefined_values,
      predefined_value_count, low_descriptor_set);
}

void loom_bytecode_selected_symbol_materializer_initialize(
    const loom_bytecode_reader_decoder_t* decoder,
    iree_arena_block_pool_t* block_pool,
    loom_bytecode_selected_table_materializer_t* tables,
    const loom_low_repr_environment_t* low_repr_environment,
    loom_bytecode_selected_symbol_materializer_t* out_materializer) {
  *out_materializer = (loom_bytecode_selected_symbol_materializer_t){
      .decoder = *decoder,
      .arena = tables->scratch_arena,
      .output_module = tables->output_module,
      .low_repr_environment = *low_repr_environment,
      .tables = tables,
  };
  out_materializer->body_materializer =
      (loom_bytecode_selected_body_materializer_t){
          .tables = tables,
          .block_pool = block_pool,
          .low_repr_environment = &out_materializer->low_repr_environment,
      };
}

static loom_symbol_kind_t loom_bytecode_selected_symbol_kind(
    loom_bytecode_symbol_kind_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SYMBOL_FUNC_DEF:
      return LOOM_SYMBOL_FUNC_DEF;
    case LOOM_BYTECODE_SYMBOL_FUNC_DECL:
      return LOOM_SYMBOL_FUNC_DECL;
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL:
      return LOOM_SYMBOL_TEMPLATE_DECL;
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF:
      return LOOM_SYMBOL_TEMPLATE_DEF;
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL:
      return LOOM_SYMBOL_TEMPLATE_UKERNEL;
    case LOOM_BYTECODE_SYMBOL_GLOBAL:
      return LOOM_SYMBOL_GLOBAL;
    case LOOM_BYTECODE_SYMBOL_EXECUTABLE:
      return LOOM_SYMBOL_EXECUTABLE;
    case LOOM_BYTECODE_SYMBOL_RECORD:
      return LOOM_SYMBOL_RECORD;
    case LOOM_BYTECODE_SYMBOL_COUNT_:
      break;
  }
  IREE_ASSERT_UNREACHABLE("validated bytecode symbol kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_symbols_predeclare(
    loom_bytecode_selected_symbol_materializer_t* materializer,
    const loom_bytecode_selected_symbol_t* selected_symbols,
    iree_host_size_t selected_symbol_count) {
  uint32_t previous_source_ordinal = 0;
  for (iree_host_size_t i = 0; i < selected_symbol_count; ++i) {
    const uint32_t source_ordinal = selected_symbols[i].source_ordinal;
    IREE_ASSERT(source_ordinal < materializer->tables->metadata->symbol_count);
    IREE_ASSERT(i == 0 || source_ordinal > previous_source_ordinal);
    previous_source_ordinal = source_ordinal;
    const loom_bytecode_symbol_metadata_t* source_symbol =
        &materializer->tables->metadata->symbols[source_ordinal];
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    bool found = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_resolve_symbol(
        materializer->tables, source_symbol->name_string_index, &target_ref,
        &found));
    if (!found) {
      loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
          materializer->tables, source_symbol->name_string_index,
          &target_name_id));
      IREE_RETURN_IF_ERROR(loom_module_add_symbol(
          materializer->output_module, target_name_id, &target_ref.symbol_id));
      target_ref.module_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_bind_symbol(
          materializer->tables, source_symbol->name_string_index,
          target_ref.symbol_id));
    }
    IREE_ASSERT(target_ref.module_id == 0);
    IREE_ASSERT(target_ref.symbol_id <
                materializer->output_module->symbols.count);
    loom_symbol_t* target_symbol =
        &materializer->output_module->symbols.entries[target_ref.symbol_id];
    IREE_ASSERT(target_symbol->defining_op == NULL);
    target_symbol->kind =
        loom_bytecode_selected_symbol_kind(source_symbol->kind);
    if (iree_any_bit_set(source_symbol->flags,
                         LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC)) {
      target_symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
    }
    if (iree_any_bit_set(source_symbol->flags,
                         LOOM_BYTECODE_SYMBOL_FLAG_RETAIN)) {
      target_symbol->flags |= LOOM_SYMBOL_FLAG_RETAIN;
    }
  }
  return iree_ok_status();
}

#include "loom/format/bytecode/reader/symbol_materializer_impl.inl"

iree_status_t loom_bytecode_selected_function_header_materialize(
    loom_bytecode_selected_symbol_materializer_t* materializer,
    uint32_t source_symbol_ordinal,
    loom_bytecode_function_header_t* out_header) {
  IREE_ASSERT(source_symbol_ordinal <
              materializer->tables->metadata->symbol_count);
  const loom_bytecode_symbol_metadata_t* metadata =
      loom_bytecode_selected_symbol_metadata(materializer,
                                             source_symbol_ordinal);
  loom_symbol_ref_t target_symbol = loom_symbol_ref_null();
  bool found = false;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_resolve_symbol(
      materializer->tables, metadata->name_string_index, &target_symbol,
      &found));
  if (!found) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected function header source symbol is not projected");
  }

  IREE_ASSERT(metadata->entry_offset <=
              materializer->tables->bytecode.data_length);
  IREE_ASSERT(metadata->entry_length <=
              materializer->tables->bytecode.data_length -
                  metadata->entry_offset);
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      materializer->tables->bytecode.data +
          (iree_host_size_t)metadata->entry_offset,
      (iree_host_size_t)metadata->entry_length, metadata->entry_offset,
      IREE_SV("SYMBOLS"), &cursor);

  loom_bytecode_symbol_entry_header_t entry_header;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_symbol_entry_header(
      materializer, &cursor, source_symbol_ordinal, &entry_header));
  if (entry_header.kind > LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected symbol is not function-like");
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_function_header(
      materializer, &cursor, source_symbol_ordinal,
      entry_header.name_string_ordinal, entry_header.flags,
      entry_header.import_module_id, entry_header.import_symbol_id,
      out_header));
  return loom_bytecode_reader_expect_empty(&materializer->decoder, &cursor,
                                           IREE_SV("symbol_entry"));
}

iree_status_t loom_bytecode_selected_symbols_materialize(
    loom_bytecode_selected_symbol_materializer_t* materializer,
    const loom_bytecode_selected_symbol_t* selected_symbols,
    iree_host_size_t selected_symbol_count) {
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_symbols_predeclare(
      materializer, selected_symbols, selected_symbol_count));

  loom_builder_t builder;
  loom_builder_initialize(
      materializer->output_module, &materializer->output_module->arena,
      loom_module_block(materializer->output_module), &builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < selected_symbol_count && iree_status_is_ok(status); ++i) {
    const loom_bytecode_selected_symbol_t* selected = &selected_symbols[i];
    const loom_bytecode_symbol_metadata_t* metadata =
        loom_bytecode_selected_symbol_metadata(materializer,
                                               selected->source_ordinal);
    IREE_ASSERT(metadata->entry_offset <=
                materializer->tables->bytecode.data_length);
    IREE_ASSERT(metadata->entry_length <=
                materializer->tables->bytecode.data_length -
                    metadata->entry_offset);
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(
        materializer->tables->bytecode.data +
            (iree_host_size_t)metadata->entry_offset,
        (iree_host_size_t)metadata->entry_length, metadata->entry_offset,
        IREE_SV("SYMBOLS"), &cursor);
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(materializer->arena);
    status = loom_bytecode_symbol_materialize_entry(
        materializer, &cursor, selected, selected->source_ordinal, &builder);
    if (iree_status_is_ok(status)) {
      status = loom_bytecode_reader_expect_empty(
          &materializer->decoder, &cursor, IREE_SV("symbol_entry"));
    }
    iree_arena_checkpoint_restore(&checkpoint);
  }
  return status;
}
