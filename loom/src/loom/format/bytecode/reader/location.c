// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/location.h"

#include <string.h>

#include "loom/error/error_catalog.h"

#define LOOM_BYTECODE_MAX_LOCATION_COUNT (UINT64_C(1) << 24)

typedef struct loom_bytecode_location_validator_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Module facts populated by location validation.
  loom_bytecode_reader_module_view_t* module_view;
  // Optional tentative output; NULL validates/indexes without native storage.
  loom_module_t* output_module;
} loom_bytecode_location_validator_t;

typedef struct loom_bytecode_location_table_t {
  // Cursor positioned at the next location entry.
  loom_bytecode_reader_cursor_t cursor;
  // Number of location entries.
  iree_host_size_t count;
} loom_bytecode_location_table_t;

static iree_status_t loom_bytecode_location_table_begin(
    loom_bytecode_location_validator_t* validator,
    const loom_bytecode_reader_section_t* section,
    loom_bytecode_location_table_t* out_table) {
  loom_bytecode_location_table_t table = {0};
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &table.cursor);
  uint64_t count = 0;
  const uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&table.cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table.cursor, &count));
  if (count > LOOM_BYTECODE_MAX_LOCATION_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        validator->decoder, IREE_SV("LOCATIONS"), count,
        LOOM_BYTECODE_MAX_LOCATION_COUNT, count_offset);
  }
  table.count = (iree_host_size_t)count;
  validator->module_view->locations.count = table.count;
  *out_table = table;
  return iree_ok_status();
}

// Inline the grammar so index/validation callers erase native construction when
// their output is absent, instead of paying for a runtime materialization mode.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_status_t
loom_bytecode_location_decode_entry(
    loom_bytecode_location_validator_t* validator,
    loom_bytecode_location_table_t* table, iree_host_size_t index,
    loom_bytecode_table_entry_metadata_t* out_metadata) {
  uint8_t kind = 0;
  uint8_t flags = 0;
  const uint64_t kind_offset =
      loom_bytecode_reader_cursor_absolute_position(&table->cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(validator->decoder, &table->cursor, &kind));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(validator->decoder, &table->cursor, &flags));
  if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
    return loom_bytecode_reader_emit_invalid_field(
        validator->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), index,
        IREE_SV("flags"), kind_offset + 1,
        IREE_SV("location_has_unsupported_flag_bits"));
  }
  if (index == 0 && kind != LOOM_LOCATION_NONE) {
    return loom_bytecode_reader_emit_invalid_field(
        validator->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), index,
        IREE_SV("kind"), kind_offset,
        IREE_SV("location_0_must_be_the_unflagged_none_location"));
  }
  loom_location_entry_t entry = {
      .kind = (loom_location_kind_t)kind,
      .flags = flags,
  };
  switch ((loom_location_kind_t)kind) {
    case LOOM_LOCATION_NONE:
      if (index != 0 || flags != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
            index, IREE_SV("kind"), kind_offset,
            IREE_SV("only_location_0_may_be_the_unflagged_none_location"));
      }
      break;
    case LOOM_LOCATION_FILE: {
      uint64_t source_id = 0;
      const uint64_t source_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &source_id));
      if (source_id >= validator->module_view->sources.count) {
        return loom_bytecode_reader_emit_table_ref(
            validator->decoder, IREE_SV("SOURCES"), source_id,
            validator->module_view->sources.count, source_offset);
      }
      uint32_t coordinates[4];
      for (int field = 0; field < 4; ++field) {
        const uint64_t coordinate_offset =
            loom_bytecode_reader_cursor_absolute_position(&table->cursor);
        uint64_t value = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            validator->decoder, &table->cursor, &value));
        if (value > LOOM_LOCATION_COORDINATE_MAX) {
          const iree_string_view_t field_names[] = {
              IREE_SV("start_line"), IREE_SV("start_col"), IREE_SV("end_line"),
              IREE_SV("end_col")};
          return loom_bytecode_reader_emit_invalid_field(
              validator->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
              index, field_names[field], coordinate_offset,
              IREE_SV("file_location_coordinate_exceeds_supported_range"));
        }
        coordinates[field] = (uint32_t)value;
      }
      entry.file.source_id = (loom_source_id_t)source_id;
      entry.file.start_line = coordinates[0];
      entry.file.start_col = coordinates[1];
      entry.file.end_line = coordinates[2];
      entry.file.end_col = coordinates[3];
      break;
    }
    case LOOM_LOCATION_FUSED: {
      const uint64_t child_count_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      uint64_t child_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &child_count));
      if (child_count > LOOM_LOCATION_FUSED_CHILD_COUNT_MAX ||
          child_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("location_children"), child_count,
            LOOM_LOCATION_FUSED_CHILD_COUNT_MAX, child_count_offset);
      }
      loom_location_id_t* children = NULL;
      const iree_host_size_t remaining =
          loom_bytecode_cursor_remaining(&table->cursor.cursor);
      if (child_count > remaining) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("location_children"), child_count,
            remaining, child_count_offset);
      }
      if (validator->output_module && child_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &validator->output_module->arena, (iree_host_size_t)child_count,
            sizeof(*children), (void**)&children));
      }
      for (uint64_t child_index = 0; child_index < child_count; ++child_index) {
        const uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&table->cursor);
        uint64_t child = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            validator->decoder, &table->cursor, &child));
        if (child >= index) {
          return loom_bytecode_reader_emit_table_ref(
              validator->decoder, IREE_SV("LOCATIONS"), child, index,
              child_offset);
        }
        if (children) {
          children[child_index] = (loom_location_id_t)child;
        }
      }
      entry.fused.count = (uint32_t)child_count;
      entry.fused.children = children;
      break;
    }
    case LOOM_LOCATION_OPAQUE: {
      uint64_t source_id = 0;
      const uint64_t source_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &source_id));
      if (source_id >= validator->module_view->sources.count) {
        return loom_bytecode_reader_emit_table_ref(
            validator->decoder, IREE_SV("SOURCES"), source_id,
            validator->module_view->sources.count, source_offset);
      }
      const uint64_t data_length_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      uint64_t data_length = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &data_length));
      if (data_length > LOOM_LOCATION_DATA_LENGTH_MAX ||
          data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("opaque_location_data"), data_length,
            LOOM_LOCATION_DATA_LENGTH_MAX, data_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          validator->decoder, &table->cursor, data_length, &span));
      uint8_t* data = NULL;
      if (validator->output_module && span.data_length > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            &validator->output_module->arena, span.data_length, (void**)&data));
        memcpy(data, span.data, span.data_length);
      }
      entry.opaque.source_id = (loom_source_id_t)source_id;
      entry.opaque.data_length = (uint32_t)span.data_length;
      entry.opaque.data = data;
      break;
    }
    case LOOM_LOCATION_TAGGED: {
      uint64_t tag = 0;
      const uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &tag));
      if (tag == LOOM_LOCATION_TAG_INVALID || tag > LOOM_LOCATION_TAG_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            validator->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
            index, IREE_SV("tag"), tag_offset,
            IREE_SV("tagged location tag must be in [1, 65535]"));
      }
      uint64_t child = 0;
      const uint64_t child_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &child));
      if (child >= index) {
        return loom_bytecode_reader_emit_table_ref(validator->decoder,
                                                   IREE_SV("LOCATIONS"), child,
                                                   index, child_offset);
      }
      const uint64_t data_length_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      uint64_t data_length = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &data_length));
      if (data_length > LOOM_LOCATION_DATA_LENGTH_MAX ||
          data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("tagged_location_data"), data_length,
            LOOM_LOCATION_DATA_LENGTH_MAX, data_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          validator->decoder, &table->cursor, data_length, &span));
      uint8_t* data = NULL;
      if (validator->output_module && span.data_length > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            &validator->output_module->arena, span.data_length, (void**)&data));
        memcpy(data, span.data, span.data_length);
      }
      entry.tagged.tag = (loom_location_tag_t)tag;
      entry.tagged.child = (loom_location_id_t)child;
      entry.tagged.data_length = (uint32_t)span.data_length;
      entry.tagged.data = data;
      break;
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          validator->decoder, IREE_SV("location_kind"), kind,
          LOOM_LOCATION_COUNT_, kind_offset);
  }
  if (validator->output_module && kind != LOOM_LOCATION_NONE) {
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_module_add_location(validator->output_module,
                                                  entry, &location_id));
    IREE_ASSERT(location_id == index);
  }
  *out_metadata = (loom_bytecode_table_entry_metadata_t){
      .entry_offset = kind_offset,
      .entry_length =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor) -
          kind_offset,
  };
  return iree_ok_status();
}

static iree_status_t loom_bytecode_location_table_finish(
    loom_bytecode_location_validator_t* validator,
    loom_bytecode_location_table_t* table) {
  return loom_bytecode_reader_expect_empty(validator->decoder, &table->cursor,
                                           IREE_SV("LOCATIONS"));
}

iree_status_t loom_bytecode_location_table_validate(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_module_view_t* module_view,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_location_validator_t validator = {
      .decoder = decoder,
      .module_view = module_view,
  };
  loom_bytecode_location_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_location_table_begin(&validator, section, &table));
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    loom_bytecode_table_entry_metadata_t discarded_metadata;
    IREE_RETURN_IF_ERROR(loom_bytecode_location_decode_entry(
        &validator, &table, i, &discarded_metadata));
  }
  return loom_bytecode_location_table_finish(&validator, &table);
}

iree_status_t loom_bytecode_location_table_index(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_module_view_t* module_view,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_table_entry_metadata_t** out_entries,
    iree_host_size_t* out_count) {
  *out_entries = NULL;
  *out_count = 0;
  loom_bytecode_location_validator_t validator = {
      .decoder = decoder,
      .module_view = module_view,
  };
  loom_bytecode_location_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_location_table_begin(&validator, section, &table));

  loom_bytecode_table_entry_metadata_t* entries = NULL;
  if (table.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        retained_arena, table.count, sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_location_decode_entry(&validator, &table,
                                                             i, &entries[i]));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_location_table_finish(&validator, &table));
  *out_entries = entries;
  *out_count = table.count;
  return iree_ok_status();
}

iree_status_t loom_bytecode_location_table_read_count(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_module_view_t* module_view,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_location_validator_t validator = {
      .decoder = decoder,
      .module_view = module_view,
  };
  loom_bytecode_location_table_t table;
  return loom_bytecode_location_table_begin(&validator, section, &table);
}

iree_status_t loom_bytecode_location_table_materialize(
    loom_bytecode_location_materializer_t* materializer,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_location_validator_t validator = {
      .decoder = materializer->decoder,
      .module_view = materializer->module_view,
      .output_module = materializer->output_module,
  };
  loom_bytecode_location_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_location_table_begin(&validator, section, &table));
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    loom_bytecode_table_entry_metadata_t discarded_metadata;
    IREE_RETURN_IF_ERROR(loom_bytecode_location_decode_entry(
        &validator, &table, i, &discarded_metadata));
  }
  return loom_bytecode_location_table_finish(&validator, &table);
}
