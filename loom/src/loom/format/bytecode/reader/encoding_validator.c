// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/encoding_validator.h"

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/attribute.h"

#define LOOM_BYTECODE_MAX_ENCODING_COUNT (UINT64_C(1) << 16)

typedef struct loom_bytecode_encoding_validator_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Finalized encoding and attribute registry context.
  loom_context_t* context;
  // Scratch storage for family identities used by later tables.
  iree_arena_allocator_t* scratch_arena;
  // Module facts populated by encoding validation.
  loom_bytecode_reader_module_view_t* module_view;
} loom_bytecode_encoding_validator_t;

typedef struct loom_bytecode_encoding_table_t {
  // Cursor positioned at the next encoding instance.
  loom_bytecode_reader_cursor_t cursor;
  // Attribute validator for instance parameters.
  loom_bytecode_attribute_validator_t attributes;
  // Number of instance entries following the family table.
  iree_host_size_t count;
} loom_bytecode_encoding_table_t;

static iree_status_t loom_bytecode_encoding_validate_string_ref(
    loom_bytecode_encoding_validator_t* validator, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= validator->module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(validator->module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(validator->decoder,
                                           LOOM_ERR_BYTECODE_010, params,
                                           IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_string = validator->module_view->strings.values[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_encoding_table_begin(
    loom_bytecode_encoding_validator_t* validator,
    const loom_bytecode_reader_section_t* section,
    loom_bytecode_encoding_table_t* out_table) {
  loom_bytecode_encoding_table_t table = {0};
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("ENCODINGS"), &table.cursor);

  uint64_t family_count = 0;
  const uint64_t family_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&table.cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table.cursor, &family_count));
  if (family_count > LOOM_BYTECODE_MAX_ENCODING_COUNT ||
      family_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        validator->decoder, IREE_SV("encoding_families"), family_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, family_count_offset);
  }
  if (family_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        validator->scratch_arena, (iree_host_size_t)family_count,
        sizeof(loom_string_id_t),
        (void**)&validator->module_view->encodings.family_name_ids));
  }
  validator->module_view->encodings.family_count =
      (iree_host_size_t)family_count;
  for (uint64_t i = 0; i < family_count; ++i) {
    const uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&table.cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        validator->decoder, &table.cursor, &name_id));
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_encoding_validate_string_ref(
        validator, name_id, IREE_SV("encoding_family_name"), name_offset,
        &name));
    if (loom_context_resolve_encoding_name(validator->context, name)
            .family_id == LOOM_ENCODING_FAMILY_ID_INVALID) {
      return loom_bytecode_reader_emit_invalid_field(
          validator->decoder, IREE_SV("ENCODINGS"), IREE_SV("family"), i,
          IREE_SV("name_id"), name_offset,
          IREE_SV("encoding_family_is_not_registered_in_the_context"));
    }
    validator->module_view->encodings.family_name_ids[i] =
        (loom_string_id_t)name_id;
  }

  uint64_t instance_count = 0;
  const uint64_t instance_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&table.cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table.cursor, &instance_count));
  if (instance_count > LOOM_BYTECODE_MAX_ENCODING_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        validator->decoder, IREE_SV("encoding_instances"), instance_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, instance_count_offset);
  }
  validator->module_view->encodings.count = (iree_host_size_t)instance_count;
  table.count = (iree_host_size_t)instance_count;
  table.attributes = (loom_bytecode_attribute_validator_t){
      .decoder = validator->decoder,
      .context = validator->context,
      .module_view = validator->module_view,
  };
  *out_table = table;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_encoding_decode_entry(
    loom_bytecode_encoding_validator_t* validator,
    loom_bytecode_encoding_table_t* table, iree_host_size_t index,
    loom_bytecode_encoding_metadata_t* out_metadata) {
  const uint64_t entry_offset =
      loom_bytecode_reader_cursor_absolute_position(&table->cursor);
  const uint64_t family_offset = entry_offset;
  uint64_t family_index = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table->cursor, &family_index));
  if (family_index >= validator->module_view->encodings.family_count) {
    return loom_bytecode_reader_emit_table_ref(
        validator->decoder, IREE_SV("encoding_families"), family_index,
        validator->module_view->encodings.family_count, family_offset);
  }
  const uint64_t alias_offset =
      loom_bytecode_reader_cursor_absolute_position(&table->cursor);
  uint64_t alias_plus_one = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table->cursor, &alias_plus_one));
  if (alias_plus_one > validator->module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("alias_string_id_plus1")),
        loom_param_u64(alias_plus_one - 1),
        loom_param_u64(validator->module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(
        validator->decoder, LOOM_ERR_BYTECODE_010, params,
        IREE_ARRAYSIZE(params), alias_offset, 0);
  }
  uint64_t parameter_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table->cursor, &parameter_count));
  iree_string_view_t previous_parameter = iree_string_view_empty();
  for (uint64_t parameter_index = 0; parameter_index < parameter_count;
       ++parameter_index) {
    const uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&table->cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        validator->decoder, &table->cursor, &name_id));
    iree_string_view_t parameter_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_encoding_validate_string_ref(
        validator, name_id, IREE_SV("encoding_param_name"), name_offset,
        &parameter_name));
    if (parameter_index > 0 &&
        iree_string_view_compare(parameter_name, previous_parameter) <= 0) {
      return loom_bytecode_reader_emit_invalid_field(
          validator->decoder, IREE_SV("ENCODINGS"), IREE_SV("instance"), index,
          IREE_SV("param_name"), name_offset,
          IREE_SV("encoding_parameters_are_not_in_canonical_order"));
    }
    previous_parameter = parameter_name;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        validator->decoder, &table->cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_named(
        &table->attributes, &table->cursor, /*descriptor=*/NULL, value_kind,
        validator->module_view->types.count));
  }
  *out_metadata = (loom_bytecode_encoding_metadata_t){
      .entry_offset = entry_offset,
      .entry_length =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor) -
          entry_offset,
      .name_string_index = (uint32_t)validator->module_view->encodings
                               .family_name_ids[family_index],
  };
  return iree_ok_status();
}

static iree_status_t loom_bytecode_encoding_table_finish(
    loom_bytecode_encoding_validator_t* validator,
    loom_bytecode_encoding_table_t* table) {
  return loom_bytecode_reader_expect_empty(validator->decoder, &table->cursor,
                                           IREE_SV("ENCODINGS"));
}

iree_status_t loom_bytecode_encoding_table_validate(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_encoding_validator_t validator = {
      .decoder = decoder,
      .context = context,
      .scratch_arena = scratch_arena,
      .module_view = module_view,
  };
  loom_bytecode_encoding_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_encoding_table_begin(&validator, section, &table));
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    loom_bytecode_encoding_metadata_t discarded_metadata;
    IREE_RETURN_IF_ERROR(loom_bytecode_encoding_decode_entry(
        &validator, &table, i, &discarded_metadata));
  }
  return loom_bytecode_encoding_table_finish(&validator, &table);
}

iree_status_t loom_bytecode_encoding_table_index(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_encoding_metadata_t** out_entries,
    iree_host_size_t* out_count) {
  *out_entries = NULL;
  *out_count = 0;
  loom_bytecode_encoding_validator_t validator = {
      .decoder = decoder,
      .context = context,
      .scratch_arena = scratch_arena,
      .module_view = module_view,
  };
  loom_bytecode_encoding_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_encoding_table_begin(&validator, section, &table));

  loom_bytecode_encoding_metadata_t* entries = NULL;
  if (table.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        retained_arena, table.count, sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_encoding_decode_entry(&validator, &table,
                                                             i, &entries[i]));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_encoding_table_finish(&validator, &table));
  *out_entries = entries;
  *out_count = table.count;
  return iree_ok_status();
}
