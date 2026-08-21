// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/symbol_materializer.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/function_header.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/format/bytecode/reader/symbol_schema.h"
#include "loom/ops/op_defs.h"

typedef loom_bytecode_symbol_materializer_t
    loom_bytecode_symbol_policy_materializer_t;
typedef loom_bytecode_value_scope_t loom_bytecode_symbol_policy_value_scope_t;
typedef loom_bytecode_reader_section_t
    loom_bytecode_symbol_policy_body_source_t;

static loom_bytecode_attribute_materializer_t
loom_bytecode_symbol_attribute_materializer(
    loom_bytecode_symbol_policy_materializer_t* materializer) {
  return (loom_bytecode_attribute_materializer_t){
      .decoder = &materializer->decoder,
      .context = materializer->context,
      .module_view = &materializer->view,
      .scratch_arena = materializer->arena,
      .output_module = materializer->output_module,
  };
}

static uint64_t loom_bytecode_symbol_policy_source_name_id(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  return materializer->view.symbols.name_ids[symbol_ordinal];
}

static uint8_t loom_bytecode_symbol_policy_kind(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  return materializer->view.symbols.kinds[symbol_ordinal];
}

static uint16_t loom_bytecode_symbol_policy_flags(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal) {
  return materializer->view.symbols.flags[symbol_ordinal];
}

static iree_status_t loom_bytecode_symbol_policy_resolve_source_string(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    uint64_t source_string_id, iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  return loom_bytecode_symbol_validate_string_ref(
      &materializer->decoder, &materializer->view, source_string_id, field_name,
      offset, out_string);
}

static iree_string_view_t loom_bytecode_symbol_policy_source_string(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    uint32_t source_string_id) {
  return materializer->view.strings.values[source_string_id];
}

static iree_status_t loom_bytecode_symbol_policy_project_string(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    uint64_t source_string_id, iree_string_view_t field_name, uint64_t offset,
    loom_string_id_t* out_string_id) {
  (void)materializer;
  (void)field_name;
  (void)offset;
  *out_string_id = (loom_string_id_t)source_string_id;
  return iree_ok_status();
}

static uint16_t loom_bytecode_symbol_policy_lookup_symbol(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    uint32_t source_name_id) {
  return loom_symbol_map_find(&materializer->view.symbols.map,
                              (loom_string_id_t)source_name_id);
}

static iree_status_t loom_bytecode_symbol_policy_project_symbol_ordinal(
    const loom_bytecode_symbol_policy_materializer_t* materializer,
    uint32_t source_symbol_ordinal, loom_symbol_ref_t* out_target_ref) {
  IREE_ASSERT(source_symbol_ordinal < materializer->view.symbols.count);
  *out_target_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = loom_bytecode_symbol_policy_lookup_symbol(
          materializer,
          materializer->view.symbols.name_ids[source_symbol_ordinal]),
  };
  return iree_ok_status();
}

static void loom_bytecode_symbol_policy_resolve_defining_op(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_host_size_t symbol_ordinal, uint64_t op_table_index_plus1,
    const loom_op_vtable_t** out_vtable, loom_op_kind_t* out_kind) {
  (void)op_table_index_plus1;
  const uint32_t op_ordinal =
      materializer->view.symbols.defining_op_ordinals[symbol_ordinal];
  *out_vtable = materializer->view.ops.values[op_ordinal];
  *out_kind = materializer->view.ops.kinds[op_ordinal];
}

static iree_status_t loom_bytecode_symbol_policy_value_scope_initialize_fresh(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_arena_allocator_t* arena, iree_string_view_t symbol_name,
    uint64_t payload_offset, loom_value_id_t* value_map,
    iree_host_size_t value_count,
    loom_bytecode_symbol_policy_value_scope_t* out_scope) {
  return loom_bytecode_value_scope_initialize_fresh(
      &materializer->body_materializer, arena, symbol_name, payload_offset,
      value_map, value_count, out_scope);
}

static iree_status_t
loom_bytecode_symbol_policy_value_scope_materialize_definition(
    loom_bytecode_symbol_policy_value_scope_t* value_scope,
    loom_bytecode_reader_cursor_t* cursor, loom_value_id_t* out_value_id) {
  return loom_bytecode_value_scope_materialize_definition(value_scope, cursor,
                                                          out_value_id);
}

static iree_status_t loom_bytecode_symbol_policy_materialize_attribute_named(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr) {
  loom_bytecode_attribute_materializer_t attributes =
      loom_bytecode_symbol_attribute_materializer(materializer);
  return loom_bytecode_attribute_materialize_named(
      &attributes, cursor, descriptor, kind, out_attr,
      materializer->view.types.count);
}

static iree_status_t loom_bytecode_symbol_policy_materialize_attribute_ssa(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope) {
  loom_bytecode_attribute_materializer_t attributes =
      loom_bytecode_symbol_attribute_materializer(materializer);
  return loom_bytecode_attribute_materialize_ssa(
      &attributes, cursor, descriptor, kind, out_attr,
      materializer->view.types.count, ssa_scope);
}

static iree_status_t loom_bytecode_symbol_policy_materialize_body(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    iree_string_view_t symbol_name,
    const loom_bytecode_symbol_policy_body_source_t* body_source,
    iree_host_size_t symbol_ordinal, uint64_t ir_offset, uint32_t ir_length,
    loom_builder_t* builder, loom_op_t* parent_op, uint8_t first_region_index,
    const loom_bytecode_predefined_region_values_t* predefined_regions,
    uint8_t predefined_region_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set) {
  (void)symbol_ordinal;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
      &materializer->decoder, IREE_SV("IR body"), ir_offset, ir_length,
      body_source ? body_source->length : 0));
  const iree_const_byte_span_t body_bytes = iree_make_const_byte_span(
      body_source->bytes.data + (iree_host_size_t)ir_offset, ir_length);
  const uint64_t body_absolute_offset =
      body_source->absolute_offset + ir_offset;
  loom_bytecode_body_summary_t summary;
  IREE_RETURN_IF_ERROR(loom_bytecode_body_summary_read(
      &materializer->decoder, symbol_name, body_bytes, body_absolute_offset,
      &summary));
  return loom_bytecode_body_materialize_symbol_regions(
      &materializer->body_materializer, symbol_name, body_bytes,
      body_absolute_offset, &summary, builder, parent_op, first_region_index,
      predefined_regions, predefined_region_count, low_descriptor_set);
}

void loom_bytecode_symbol_materializer_initialize(
    const loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    iree_arena_allocator_t* arena, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_reader_module_view_t* module_view,
    loom_module_t* output_module,
    const loom_low_repr_environment_t* low_repr_environment,
    loom_bytecode_symbol_materializer_t* out_materializer) {
  *out_materializer = (loom_bytecode_symbol_materializer_t){
      .decoder = *decoder,
      .context = context,
      .arena = arena,
      .view = *module_view,
      .output_module = output_module,
      .low_repr_environment = *low_repr_environment,
  };
  out_materializer->body_materializer = (loom_bytecode_body_materializer_t){
      .attributes =
          {
              .decoder = &out_materializer->decoder,
              .context = context,
              .module_view = &out_materializer->view,
              .scratch_arena = arena,
              .output_module = output_module,
          },
      .block_pool = block_pool,
      .low_repr_environment = &out_materializer->low_repr_environment,
  };
}

static loom_symbol_kind_t loom_bytecode_reader_decode_symbol_kind(
    uint8_t kind) {
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
    case LOOM_BYTECODE_SYMBOL_ANCHOR:
      return LOOM_SYMBOL_NONE;
    default:
      return LOOM_SYMBOL_NONE;
  }
}

static iree_status_t loom_bytecode_reader_symbol_cursor_to_entries(
    loom_bytecode_symbol_policy_materializer_t* materializer,
    const loom_bytecode_reader_section_t* symbols_section,
    loom_bytecode_reader_cursor_t* cursor) {
  loom_bytecode_reader_cursor_initialize(
      symbols_section->bytes.data, symbols_section->bytes.data_length,
      symbols_section->absolute_offset, IREE_SV("SYMBOLS"), cursor);
  uint64_t count = 0;
  uint64_t import_count = 0;
  uint64_t export_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(&materializer->decoder,
                                                         cursor, &count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &materializer->decoder, cursor, &import_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &materializer->decoder, cursor, &export_count));
  for (uint64_t i = 0; i < import_count + export_count; ++i) {
    uint64_t unused_offset = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &materializer->decoder, cursor, &unused_offset));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbols_predeclare(
    loom_bytecode_symbol_materializer_t* materializer) {
  for (iree_host_size_t i = 0; i < materializer->view.symbols.count; ++i) {
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(
        materializer->output_module, materializer->view.symbols.name_ids[i],
        &symbol_id));
    loom_symbol_t* symbol =
        &materializer->output_module->symbols.entries[symbol_id];
    symbol->kind = loom_bytecode_reader_decode_symbol_kind(
        materializer->view.symbols.kinds[i]);
    symbol->flags = 0;
    const loom_bytecode_symbol_flags_t flags =
        materializer->view.symbols.flags[i];
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC)) {
      symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
    }
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_RETAIN)) {
      symbol->flags |= LOOM_SYMBOL_FLAG_RETAIN;
    }
  }
  return iree_ok_status();
}

#include "loom/format/bytecode/reader/symbol_materializer_impl.inl"

iree_status_t loom_bytecode_symbols_materialize(
    loom_bytecode_symbol_materializer_t* materializer,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section) {
  loom_bytecode_reader_cursor_t cursor;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_symbol_cursor_to_entries(
      materializer, symbols_section, &cursor));

  loom_builder_t builder;
  loom_builder_initialize(
      materializer->output_module, &materializer->output_module->arena,
      loom_module_block(materializer->output_module), &builder);
  for (iree_host_size_t i = 0; i < materializer->view.symbols.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_materialize_entry(
        materializer, &cursor, ir_section, i, &builder));
  }
  return loom_bytecode_reader_expect_empty(&materializer->decoder, &cursor,
                                           IREE_SV("SYMBOLS"));
}
