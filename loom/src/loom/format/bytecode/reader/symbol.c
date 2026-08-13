// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/symbol.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/format/bytecode/reader/symbol_schema.h"
#include "loom/ops/op_defs.h"

#define LOOM_BYTECODE_MAX_SYMBOL_COUNT LOOM_SYMBOL_ID_INVALID

void loom_bytecode_symbol_validator_initialize(
    const loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* metadata_arena,
    loom_bytecode_reader_module_view_t* module_view,
    loom_bytecode_symbol_validator_t* out_validator) {
  *out_validator = (loom_bytecode_symbol_validator_t){
      .decoder = *decoder,
      .context = context,
      .arena = arena,
      .metadata_arena = metadata_arena,
      .view = *module_view,
      .output_view = module_view,
  };
}

static loom_bytecode_attribute_validator_t
loom_bytecode_symbol_attribute_validator(
    loom_bytecode_symbol_validator_t* validator) {
  return (loom_bytecode_attribute_validator_t){
      .decoder = &validator->decoder,
      .context = validator->context,
      .module_view = &validator->view,
  };
}

static iree_status_t loom_bytecode_reader_skip_func_payload_attrs(
    loom_bytecode_symbol_validator_t* reader,
    loom_bytecode_reader_cursor_t* cursor, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    const loom_bytecode_attribute_ssa_validation_scope_t* ssa_scope) {
  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_function_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  loom_bytecode_attribute_validator_t attribute_validator =
      loom_bytecode_symbol_attribute_validator(reader);
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
        &reader->decoder, &reader->view, key_id,
        IREE_SV("function_attribute_key"), key_offset, &key));
    uint8_t attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_symbol_func_metadata_attr_is_shared(vtable, func_like,
                                                          attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_is_reconstructed_from_shared_metadata"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        &reader->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_ssa(
        &attribute_validator, cursor, &vtable->attr_descriptors[attr_index],
        value_kind, reader->view.types.count, ssa_scope));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_symbol_definition_flags(
    loom_bytecode_symbol_validator_t* reader, uint64_t symbol_index,
    uint16_t symbol_flags, const loom_op_vtable_t* vtable,
    uint64_t op_ref_offset) {
  if (!vtable || !vtable->symbol_def) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("symbol_defining_op_must_declare_a_symbol_contract"));
  }
  const loom_bytecode_symbol_flags_t definition_flag_mask =
      LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION |
      LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY;
  loom_bytecode_symbol_flags_t expected_flags = 0;
  if (loom_symbol_definition_is_declaration(vtable->symbol_def)) {
    expected_flags |= LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION;
  }
  if (loom_symbol_definition_is_test_only(vtable->symbol_def)) {
    expected_flags |= LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY;
  }
  if ((symbol_flags & definition_flag_mask) == expected_flags) {
    return iree_ok_status();
  }
  return loom_bytecode_reader_emit_invalid_field(
      &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
      IREE_SV("flags"), op_ref_offset,
      IREE_SV("symbol_definition_flags_must_match_defining_op_contract"));
}

static iree_status_t loom_bytecode_reader_skip_value_def(
    loom_bytecode_symbol_validator_t* reader,
    loom_bytecode_reader_cursor_t* cursor) {
  uint64_t name_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t name_id = 0;
  uint64_t type_id = 0;
  uint64_t dim_binding_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &name_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &type_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &dim_binding_count));
  if (name_id != 0) {
    iree_string_view_t unused_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
        &reader->decoder, &reader->view, name_id, IREE_SV("value_name"),
        name_offset, &unused_name));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_type_ref(
      &reader->decoder, &reader->view, type_id, name_offset));
  for (uint64_t i = 0; i < dim_binding_count; ++i) {
    int64_t unused_value_ref = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
        &reader->decoder, cursor, &unused_value_ref));
  }
  uint64_t unused_encoding_ref = 0;
  return loom_bytecode_reader_read_uvarint(&reader->decoder, cursor,
                                           &unused_encoding_ref);
}

static iree_status_t loom_bytecode_reader_skip_global_payload(
    loom_bytecode_symbol_validator_t* reader,
    loom_bytecode_reader_cursor_t* cursor, uint64_t symbol_index,
    uint16_t symbol_flags, loom_bytecode_symbol_metadata_t* symbol_metadata) {
  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_resolve_op_ref(
      &reader->decoder, &reader->view, op_table_index_plus1, op_ref_offset,
      &vtable));
  reader->view.symbols.defining_op_ordinals[symbol_index] =
      (uint32_t)(op_table_index_plus1 - 1);
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_global_vtable(
      &reader->decoder, symbol_index, vtable, op_ref_offset));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_symbol_definition_flags(
      reader, symbol_index, symbol_flags, vtable, op_ref_offset));
  if (symbol_metadata) {
    symbol_metadata->defining_op_name = loom_op_vtable_name(vtable);
    symbol_metadata->interfaces = vtable->symbol_def->interfaces;
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_source_trivia_validate(&reader->decoder, cursor));

  uint64_t result_count = 0;
  uint64_t result_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &result_count));
  if (result_count == 0 || result_count > UINT16_MAX ||
      result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("result_count"), result_count_offset,
        IREE_SV("global_result_count_must_be_non_zero_and_fit_in_uint16_t"));
  }
  if (symbol_metadata) {
    symbol_metadata->result_count = (uint16_t)result_count;
  }
  uint64_t local_value_count = 0;
  uint64_t local_value_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &local_value_count));
  if (local_value_count < result_count ||
      local_value_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("local_value_count"), local_value_count_offset,
        IREE_SV("global_local_value_count_must_cover_all_op_results"));
  }
  if (symbol_metadata) {
    symbol_metadata->local_value_count = local_value_count;
  }
  for (uint64_t i = 0; i < local_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_value_def(reader, cursor));
  }

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_global_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  loom_bytecode_attribute_validator_t attribute_validator =
      loom_bytecode_symbol_attribute_validator(reader);
  const loom_bytecode_attribute_ssa_validation_scope_t attribute_ssa_scope = {
      .symbol_name = reader->view.strings
                         .values[reader->view.symbols.name_ids[symbol_index]],
      .value_count = local_value_count,
  };
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
        &reader->decoder, &reader->view, key_id,
        IREE_SV("global_attribute_key"), key_offset, &key));
    uint8_t attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_symbol_attr_is_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        &reader->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_ssa(
        &attribute_validator, cursor, &vtable->attr_descriptors[attr_index],
        value_kind, reader->view.types.count, &attribute_ssa_scope));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_record_payload(
    loom_bytecode_symbol_validator_t* reader,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_reader_section_t* ir_section, uint64_t symbol_index,
    uint16_t symbol_flags, loom_bytecode_symbol_metadata_t* symbol_metadata) {
  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_resolve_op_ref(
      &reader->decoder, &reader->view, op_table_index_plus1, op_ref_offset,
      &vtable));
  reader->view.symbols.defining_op_ordinals[symbol_index] =
      (uint32_t)(op_table_index_plus1 - 1);
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_record_vtable(
      &reader->decoder, symbol_index, vtable, op_ref_offset));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_symbol_definition_flags(
      reader, symbol_index, symbol_flags, vtable, op_ref_offset));
  if (symbol_metadata) {
    symbol_metadata->defining_op_name = loom_op_vtable_name(vtable);
    symbol_metadata->interfaces = vtable->symbol_def->interfaces;
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_source_trivia_validate(&reader->decoder, cursor));

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_record_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  loom_bytecode_attribute_validator_t attribute_validator =
      loom_bytecode_symbol_attribute_validator(reader);
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
        &reader->decoder, &reader->view, key_id,
        IREE_SV("record_attribute_key"), key_offset, &key));
    uint8_t attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_symbol_attr_is_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        &reader->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_named(
        &attribute_validator, cursor, &vtable->attr_descriptors[attr_index],
        value_kind, reader->view.types.count));
  }

  uint8_t has_body = 0;
  uint64_t has_body_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &has_body));
  if (has_body > 1) {
    return loom_bytecode_reader_emit_enum_value(
        &reader->decoder, IREE_SV("has_body"), has_body, 2, has_body_offset);
  }
  if ((vtable->region_count == 1) != (has_body != 0)) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("has_body"), has_body_offset,
        IREE_SV("record_body_presence_must_match_the_defining_op_region"));
  }
  if (has_body) {
    uint64_t ir_offset = 0;
    uint32_t ir_length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(&reader->decoder, cursor, &ir_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(&reader->decoder, cursor, &ir_length));
    if (ir_section) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
          &reader->decoder, IREE_SV("IR body"), ir_offset, ir_length,
          ir_section->length));
    }
    if (symbol_metadata) {
      symbol_metadata->has_body = true;
      symbol_metadata->body_offset = ir_offset;
      symbol_metadata->body_absolute_offset =
          ir_section ? ir_section->absolute_offset + ir_offset : 0;
      symbol_metadata->body_length = ir_length;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbols_validate(
    loom_bytecode_symbol_validator_t* reader,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      symbols_section->bytes.data, symbols_section->bytes.data_length,
      symbols_section->absolute_offset, IREE_SV("SYMBOLS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &count));
  if (count > LOOM_BYTECODE_MAX_SYMBOL_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("SYMBOLS"), count,
        LOOM_BYTECODE_MAX_SYMBOL_COUNT, count_offset);
  }
  reader->view.symbols.count = (iree_host_size_t)count;
  if (count > 0) {
    const iree_host_size_t symbol_fact_size =
        sizeof(*reader->view.symbols.name_ids) +
        sizeof(*reader->view.symbols.defining_op_ordinals) +
        sizeof(*reader->view.symbols.flags) +
        sizeof(*reader->view.symbols.kinds);
    uint8_t* symbol_facts = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, (iree_host_size_t)count,
                                  symbol_fact_size, (void**)&symbol_facts));
    reader->view.symbols.name_ids = (loom_string_id_t*)symbol_facts;
    symbol_facts += count * sizeof(*reader->view.symbols.name_ids);
    reader->view.symbols.defining_op_ordinals = (uint32_t*)symbol_facts;
    symbol_facts += count * sizeof(*reader->view.symbols.defining_op_ordinals);
    reader->view.symbols.flags = (loom_bytecode_symbol_flags_t*)symbol_facts;
    symbol_facts += count * sizeof(*reader->view.symbols.flags);
    reader->view.symbols.kinds = symbol_facts;
  }

  uint64_t import_count = 0;
  uint64_t export_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &import_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &export_count));
  if (import_count > count || export_count > count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("header"), 0,
        IREE_SV("import_export_count"),
        loom_bytecode_reader_cursor_absolute_position(&cursor),
        IREE_SV("import_export_counts_must_not_exceed_symbol_count"));
  }
  loom_bytecode_module_metadata_t* module_metadata =
      reader->view.output_metadata;
  if (module_metadata) {
    module_metadata->symbol_count = (iree_host_size_t)count;
    module_metadata->import_count = (iree_host_size_t)import_count;
    module_metadata->export_count = (iree_host_size_t)export_count;
    if (count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)count,
          sizeof(loom_bytecode_symbol_metadata_t),
          (void**)&module_metadata->symbols));
      memset(module_metadata->symbols, 0,
             (iree_host_size_t)count * sizeof(*module_metadata->symbols));
    }
    if (import_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)import_count,
          sizeof(uint32_t), (void**)&module_metadata->import_symbol_indices));
    }
    if (export_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)export_count,
          sizeof(uint32_t), (void**)&module_metadata->export_symbol_indices));
    }
  }
  uint64_t offset_table_count = import_count + export_count;
  uint64_t* offset_table = NULL;
  if (offset_table_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)offset_table_count, sizeof(uint64_t),
        (void**)&offset_table));
  }
  for (uint64_t i = 0; i < offset_table_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &reader->decoder, &cursor, &offset_table[i]));
  }
  uint64_t entries_base = cursor.absolute_offset + cursor.cursor.position;
  uint64_t entries_payload_length =
      symbols_section->absolute_offset + symbols_section->length - entries_base;
  for (uint64_t i = 0; i < offset_table_count; ++i) {
    if (offset_table[i] >= entries_payload_length) {
      return loom_bytecode_reader_emit_range_error(
          &reader->decoder, IREE_SV("symbol_offset_table"), offset_table[i], 1,
          entries_payload_length);
    }
  }

  const uint64_t* import_offsets = offset_table;
  const uint64_t* export_offsets =
      offset_table ? offset_table + import_count : NULL;
  uint64_t import_index = 0;
  uint64_t export_index = 0;
  for (uint64_t symbol_index = 0; symbol_index < count; ++symbol_index) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t entry_relative_offset = entry_offset - entries_base;
    loom_bytecode_symbol_metadata_t* symbol_metadata =
        module_metadata ? &module_metadata->symbols[symbol_index] : NULL;
    if (symbol_metadata) {
      symbol_metadata->entry_offset = entry_offset;
    }
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &name_id));
    iree_string_view_t unused_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
        &reader->decoder, &reader->view, name_id, IREE_SV("symbol_name"),
        name_offset, &unused_name));
    uint16_t indexed_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_symbol_map_find_or_insert(
        &reader->view.symbols.map, reader->arena, (loom_string_id_t)name_id,
        (uint16_t)symbol_index, &indexed_symbol_id));
    if (indexed_symbol_id != symbol_index) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("name_id"), name_offset,
          IREE_SV("symbol_name_appears_more_than_once"));
    }
    reader->view.symbols.name_ids[symbol_index] = (loom_string_id_t)name_id;
    if (symbol_metadata) {
      symbol_metadata->name = unused_name;
      symbol_metadata->name_string_index = (uint32_t)name_id;
    }

    uint8_t kind = 0;
    uint8_t visibility = 0;
    uint16_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &visibility));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(&reader->decoder, &cursor, &flags));
    if (kind >= LOOM_BYTECODE_SYMBOL_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          &reader->decoder, IREE_SV("symbol_kind"), kind,
          LOOM_BYTECODE_SYMBOL_COUNT_, kind_offset);
    }
    if (visibility >= LOOM_BYTECODE_SYMBOL_VISIBILITY_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          &reader->decoder, IREE_SV("symbol_visibility"), visibility,
          LOOM_BYTECODE_SYMBOL_VISIBILITY_COUNT_, kind_offset + 1);
    }
    if (flags &
        ~(LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC | LOOM_BYTECODE_SYMBOL_FLAG_IMPORT |
          LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL |
          LOOM_BYTECODE_SYMBOL_FLAG_RETAIN |
          LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION |
          LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY |
          LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("flags"), kind_offset + 2,
          IREE_SV("symbol_has_unsupported_flag_bits"));
    }
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL) &&
        !iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("flags"), kind_offset + 2,
          IREE_SV("explicit_import_symbol_flag_requires_import_flag"));
    }
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES) &&
        kind > LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("flags"), kind_offset + 2,
          IREE_SV("predicates_flag_requires_function_symbol_kind"));
    }
    if (kind == LOOM_BYTECODE_SYMBOL_ANCHOR &&
        (visibility != LOOM_BYTECODE_SYMBOL_VISIBILITY_PRIVATE || flags != 0)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("anchor_header"), kind_offset,
          IREE_SV("provider_anchor_must_be_private_and_unflagged"));
    }
    reader->view.symbols.flags[symbol_index] = flags;
    reader->view.symbols.kinds[symbol_index] = kind;
    reader->view.symbols.defining_op_ordinals[symbol_index] = UINT32_MAX;
    if (kind == LOOM_BYTECODE_SYMBOL_ANCHOR) {
      ++reader->view.symbols.unresolved_anchor_count;
    }
    if (symbol_metadata) {
      symbol_metadata->kind = (loom_bytecode_symbol_kind_t)kind;
      symbol_metadata->visibility =
          (loom_bytecode_symbol_visibility_t)visibility;
      symbol_metadata->flags = flags;
    }
    const bool is_import =
        iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT);
    const bool is_export =
        !is_import && iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC);
    if (is_import) {
      if (import_index >= import_count ||
          import_offsets[import_index] != entry_relative_offset) {
        uint64_t diagnostic_offset = import_index < import_count
                                         ? symbols_section->absolute_offset +
                                               import_offsets[import_index]
                                         : entry_offset;
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"),
            IREE_SV("import_offset_table"), import_index,
            IREE_SV("symbol_entry_offset"), diagnostic_offset,
            IREE_SV("import_offset_table_must_match_import_symbol_order"));
      }
      if (module_metadata) {
        module_metadata->import_symbol_indices[import_index] =
            (uint32_t)symbol_index;
      }
      ++import_index;
    } else if (is_export) {
      if (export_index >= export_count ||
          export_offsets[export_index] != entry_relative_offset) {
        uint64_t diagnostic_offset = export_index < export_count
                                         ? symbols_section->absolute_offset +
                                               export_offsets[export_index]
                                         : entry_offset;
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"),
            IREE_SV("export_offset_table"), export_index,
            IREE_SV("symbol_entry_offset"), diagnostic_offset,
            IREE_SV("export_offset_table_must_match_export_symbol_order"));
      }
      if (module_metadata) {
        module_metadata->export_symbol_indices[export_index] =
            (uint32_t)symbol_index;
      }
      ++export_index;
    }
    if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
      uint64_t module_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t module_ref_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &module_ref_id));
      iree_string_view_t import_module = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
          &reader->decoder, &reader->view, module_ref_id,
          IREE_SV("source_module_id"), module_ref_offset, &import_module));
      uint64_t symbol_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t symbol_ref_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &symbol_ref_id));
      iree_string_view_t import_symbol = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
          &reader->decoder, &reader->view, symbol_ref_id,
          IREE_SV("source_symbol_id"), symbol_ref_offset, &import_symbol));
      if (symbol_metadata) {
        symbol_metadata->import_module = import_module;
        symbol_metadata->import_symbol = import_symbol;
      }
    }

    if (kind <= LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
      uint64_t op_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t op_table_index_plus1 = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &op_table_index_plus1));
      const loom_op_vtable_t* unused_vtable = NULL;
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_resolve_op_ref(
          &reader->decoder, &reader->view, op_table_index_plus1, op_ref_offset,
          &unused_vtable));
      reader->view.symbols.defining_op_ordinals[symbol_index] =
          (uint32_t)(op_table_index_plus1 - 1);
      if (!unused_vtable->func_like) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
            symbol_index, IREE_SV("def_op_table_index_plus1"), op_ref_offset,
            IREE_SV("function_symbol_defining_op_must_implement_funclike"));
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_validate_symbol_definition_flags(
              reader, symbol_index, flags, unused_vtable, op_ref_offset));
      if (symbol_metadata) {
        symbol_metadata->defining_op_name = loom_op_vtable_name(unused_vtable);
        symbol_metadata->interfaces = unused_vtable->symbol_def->interfaces;
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_source_trivia_validate(&reader->decoder, &cursor));

      uint8_t calling_convention = 0;
      uint64_t cc_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(
          &reader->decoder, &cursor, &calling_convention));
      uint8_t purity = 0;
      uint64_t purity_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &purity));
      const loom_func_like_vtable_t* func_like = unused_vtable->func_like;
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_func_enum(
          &reader->decoder, symbol_index, unused_vtable,
          func_like->cc_attr_index, IREE_SV("calling_convention"),
          calling_convention, cc_offset));
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_func_enum(
          &reader->decoder, symbol_index, unused_vtable,
          func_like->purity_attr_index, IREE_SV("purity"), purity,
          purity_offset));
      uint64_t workload_count_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t workload_arg_count = 0;
      uint64_t arg_count = 0;
      uint64_t result_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &workload_arg_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &arg_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &result_count));
      if (workload_arg_count > UINT16_MAX || arg_count > UINT16_MAX ||
          result_count > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
            symbol_index, IREE_SV("signature_count"), cc_offset,
            IREE_SV("function_signature_exceeds_runtime_field_width"));
      }
      if (!loom_symbol_definition_implements(unused_vtable->symbol_def,
                                             LOOM_SYMBOL_INTERFACE_KERNEL) &&
          workload_arg_count != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
            symbol_index, IREE_SV("kernel_workload_arg_count"),
            workload_count_offset,
            IREE_SV("ordinary_function_has_kernel_workload_arguments"));
      }
      if (symbol_metadata) {
        symbol_metadata->calling_convention = calling_convention;
        symbol_metadata->purity = purity;
        symbol_metadata->kernel_workload_argument_count =
            (uint16_t)workload_arg_count;
        symbol_metadata->argument_count = (uint16_t)arg_count;
        symbol_metadata->result_count = (uint16_t)result_count;
      }
      for (uint64_t i = 0; i < workload_arg_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_skip_value_def(reader, &cursor));
      }
      for (uint64_t i = 0; i < arg_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_skip_value_def(reader, &cursor));
      }
      uint64_t actual_tied_result_count = 0;
      for (uint64_t i = 0; i < result_count; ++i) {
        uint8_t is_tied = 0;
        uint64_t tie_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &is_tied));
        if (is_tied > 1) {
          return loom_bytecode_reader_emit_enum_value(
              &reader->decoder, IREE_SV("is_tied"), is_tied, 2, tie_offset);
        }
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_skip_value_def(reader, &cursor));
        if (is_tied) {
          uint64_t operand_index = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              &reader->decoder, &cursor, &operand_index));
          if (operand_index >= arg_count) {
            return loom_bytecode_reader_emit_invalid_field(
                &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
                symbol_index, IREE_SV("tied_operand_index"), tie_offset,
                IREE_SV("tied_result_references_an_argument_out_of_range"));
          }
          ++actual_tied_result_count;
        }
      }
      uint64_t tied_result_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &tied_result_count));
      if (tied_result_count != actual_tied_result_count) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
            symbol_index, IREE_SV("tied_result_count"),
            loom_bytecode_reader_cursor_absolute_position(&cursor),
            IREE_SV("tied_result_summary_does_not_match_tied_result_records"));
      }
      if (symbol_metadata) {
        symbol_metadata->tied_result_count = (uint16_t)tied_result_count;
      }
      loom_bytecode_attribute_validator_t attribute_validator =
          loom_bytecode_symbol_attribute_validator(reader);
      const loom_bytecode_attribute_ssa_validation_scope_t attribute_ssa_scope =
          {
              .symbol_name = unused_name,
              .value_count = workload_arg_count + arg_count + result_count,
          };
      uint16_t predicate_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_predicate_list_ssa(
          &attribute_validator, &cursor, &attribute_ssa_scope,
          &predicate_count));
      const bool has_predicates =
          iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES);
      if (!has_predicates && predicate_count != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
            symbol_index, IREE_SV("predicates"),
            loom_bytecode_reader_cursor_absolute_position(&cursor),
            IREE_SV("nonempty_predicate_list_requires_predicates_flag"));
      }
      if (has_predicates && unused_vtable->func_like->predicates_attr_index ==
                                LOOM_ATTR_INDEX_NONE) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"),
            symbol_index, IREE_SV("predicates"),
            loom_bytecode_reader_cursor_absolute_position(&cursor),
            IREE_SV("predicate_flag_requires_funclike_predicates_attr"));
      }
      if (kind == LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE ||
          kind == LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
        uint64_t implements_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t implements_string_id = 0;
        uint64_t priority = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &implements_string_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &priority));
        iree_string_view_t implements_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_symbol_validate_string_ref(
            &reader->decoder, &reader->view, implements_string_id,
            IREE_SV("implements_op_name"), implements_offset,
            &implements_name));
        if (symbol_metadata) {
          symbol_metadata->implements_op_name = implements_name;
          symbol_metadata->priority = priority;
        }
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_func_payload_attrs(
          reader, &cursor, symbol_index, unused_vtable, func_like,
          &attribute_ssa_scope));
      uint8_t has_body = 0;
      uint64_t has_body_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &has_body));
      if (has_body > 1) {
        return loom_bytecode_reader_emit_enum_value(
            &reader->decoder, IREE_SV("has_body"), has_body, 2,
            has_body_offset);
      }
      if (has_body) {
        uint64_t ir_offset = 0;
        uint32_t ir_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
            &reader->decoder, &cursor, &ir_offset));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u32_le(
            &reader->decoder, &cursor, &ir_length));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
            &reader->decoder, IREE_SV("IR body"), ir_offset, ir_length,
            ir_section ? ir_section->length : 0));
        if (symbol_metadata) {
          symbol_metadata->has_body = true;
          symbol_metadata->body_offset = ir_offset;
          symbol_metadata->body_absolute_offset =
              ir_section ? ir_section->absolute_offset + ir_offset : 0;
          symbol_metadata->body_length = ir_length;
        }
      }
    } else if (kind == LOOM_BYTECODE_SYMBOL_GLOBAL) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_global_payload(
          reader, &cursor, symbol_index, flags, symbol_metadata));
    } else if (kind == LOOM_BYTECODE_SYMBOL_RECORD) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_record_payload(
          reader, &cursor, ir_section, symbol_index, flags, symbol_metadata));
    }

    uint64_t next_entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor) - entries_base;
    if (next_entry_offset > entries_payload_length) {
      return loom_bytecode_reader_emit_range_error(
          &reader->decoder, IREE_SV("symbol_entry"), next_entry_offset, 0,
          entries_payload_length);
    }
    if (symbol_metadata) {
      symbol_metadata->entry_length =
          loom_bytecode_reader_cursor_absolute_position(&cursor) -
          symbol_metadata->entry_offset;
    }
  }
  if (import_index != import_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("import_offset_table"),
        import_index, IREE_SV("symbol_entry_offset"),
        symbols_section->absolute_offset,
        IREE_SV("import_offset_table_contains_entries_without_import_symbols"));
  }
  if (export_index != export_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("export_offset_table"),
        export_index, IREE_SV("symbol_entry_offset"),
        symbols_section->absolute_offset,
        IREE_SV("export_offset_table_contains_entries_without_export_symbols"));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      &reader->decoder, &cursor, IREE_SV("SYMBOLS")));
  *reader->output_view = reader->view;
  return iree_ok_status();
}
