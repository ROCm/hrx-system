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

static iree_status_t loom_bytecode_location_decode_entry(
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
      for (int field = 0; field < 4; ++field) {
        const uint64_t coordinate_offset =
            loom_bytecode_reader_cursor_absolute_position(&table->cursor);
        uint64_t value = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            validator->decoder, &table->cursor, &value));
        if (value > UINT16_MAX) {
          const iree_string_view_t field_names[] = {
              IREE_SV("start_line"), IREE_SV("start_col"), IREE_SV("end_line"),
              IREE_SV("end_col")};
          return loom_bytecode_reader_emit_invalid_field(
              validator->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
              index, field_names[field], coordinate_offset,
              IREE_SV("file_location_coordinate_exceeds_runtime_field_width"));
        }
      }
      break;
    }
    case LOOM_LOCATION_FUSED: {
      const uint64_t child_count_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      uint64_t child_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &child_count));
      if (child_count > UINT32_MAX || child_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("location_children"), child_count,
            UINT32_MAX, child_count_offset);
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
      }
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
      if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("opaque_location_data"), data_length,
            UINT32_MAX, data_length_offset);
      }
      iree_const_byte_span_t unused = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          validator->decoder, &table->cursor, data_length, &unused));
      break;
    }
    case LOOM_LOCATION_TAGGED: {
      uint64_t tag = 0;
      const uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(&table->cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          validator->decoder, &table->cursor, &tag));
      if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
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
      if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            validator->decoder, IREE_SV("tagged_location_data"), data_length,
            UINT32_MAX, data_length_offset);
      }
      iree_const_byte_span_t unused = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          validator->decoder, &table->cursor, data_length, &unused));
      break;
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          validator->decoder, IREE_SV("location_kind"), kind,
          LOOM_LOCATION_COUNT_, kind_offset);
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

static iree_status_t loom_bytecode_location_read_coordinate(
    loom_bytecode_location_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, uint64_t location_index,
    iree_string_view_t field_name, uint16_t* out_value) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(materializer->decoder, cursor, &value));
  if (value > UINT16_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
        location_index, field_name, offset,
        IREE_SV("file_location_coordinate_exceeds_runtime_field_width"));
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_location_read_source_ref(
    loom_bytecode_location_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, loom_source_id_t* out_source_id) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_index = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, cursor, &source_index));
  if (source_index >= materializer->module_view->sources.count) {
    return loom_bytecode_reader_emit_table_ref(
        materializer->decoder, IREE_SV("SOURCES"), source_index,
        materializer->module_view->sources.count, offset);
  }
  *out_source_id = materializer->source_ids[source_index];
  return iree_ok_status();
}

iree_status_t loom_bytecode_location_table_materialize(
    loom_bytecode_location_materializer_t* materializer,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &cursor);

  uint64_t count = 0;
  const uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(materializer->decoder,
                                                         &cursor, &count));
  if (count != materializer->module_view->locations.count) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), 0,
        IREE_SV("count"), count_offset,
        IREE_SV("location_count_changed_between_validation_and_materialize"));
  }

  for (uint64_t i = 0; i < count; ++i) {
    uint8_t kind = 0;
    uint8_t flags = 0;
    const uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(materializer->decoder, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(materializer->decoder, &cursor, &flags));
    if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
      return loom_bytecode_reader_emit_invalid_field(
          materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("flags"), kind_offset + 1,
          IREE_SV("location_has_unsupported_flag_bits"));
    }

    loom_location_entry_t entry = {
        .kind = (loom_location_kind_t)kind,
        .flags = flags,
    };
    switch ((loom_location_kind_t)kind) {
      case LOOM_LOCATION_NONE:
        if (i != 0 || flags != 0) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
              i, IREE_SV("kind"), kind_offset,
              IREE_SV("only_location_0_may_be_the_unflagged_none_location"));
        }
        continue;
      case LOOM_LOCATION_FILE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_location_read_source_ref(
            materializer, &cursor, &entry.file.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_location_read_coordinate(
            materializer, &cursor, i, IREE_SV("start_line"),
            &entry.file.start_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_location_read_coordinate(
            materializer, &cursor, i, IREE_SV("start_col"),
            &entry.file.start_col));
        IREE_RETURN_IF_ERROR(loom_bytecode_location_read_coordinate(
            materializer, &cursor, i, IREE_SV("end_line"),
            &entry.file.end_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_location_read_coordinate(
            materializer, &cursor, i, IREE_SV("end_col"), &entry.file.end_col));
        break;
      }
      case LOOM_LOCATION_FUSED: {
        const uint64_t child_count_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t child_count = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &child_count));
        if (child_count > UINT32_MAX || child_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              materializer->decoder, IREE_SV("location_children"), child_count,
              UINT32_MAX, child_count_offset);
        }
        loom_location_id_t* children = NULL;
        if (child_count > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate_array(&materializer->output_module->arena,
                                        (iree_host_size_t)child_count,
                                        sizeof(*children), (void**)&children));
        }
        for (uint64_t child_index = 0; child_index < child_count;
             ++child_index) {
          const uint64_t child_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t child = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              materializer->decoder, &cursor, &child));
          if (child >= i) {
            return loom_bytecode_reader_emit_table_ref(materializer->decoder,
                                                       IREE_SV("LOCATIONS"),
                                                       child, i, child_offset);
          }
          children[child_index] = (loom_location_id_t)child;
        }
        entry.fused.count = (uint32_t)child_count;
        entry.fused.children = children;
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_location_read_source_ref(
            materializer, &cursor, &entry.opaque.source_id));
        const uint64_t data_length_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &data_length));
        if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              materializer->decoder, IREE_SV("opaque_location_data"),
              data_length, UINT32_MAX, data_length_offset);
        }
        iree_const_byte_span_t data_span = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            materializer->decoder, &cursor, data_length, &data_span));
        uint8_t* data = NULL;
        if (data_span.data_length > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(&materializer->output_module->arena,
                                  data_span.data_length, (void**)&data));
          memcpy(data, data_span.data, data_span.data_length);
        }
        entry.opaque.data_length = (uint32_t)data_span.data_length;
        entry.opaque.data = data;
        break;
      }
      case LOOM_LOCATION_TAGGED: {
        const uint64_t tag_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t tag = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &tag));
        if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
              i, IREE_SV("tag"), tag_offset,
              IREE_SV("tagged location tag must be in [1, 65535]"));
        }
        const uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t child = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &child));
        if (child >= i) {
          return loom_bytecode_reader_emit_table_ref(materializer->decoder,
                                                     IREE_SV("LOCATIONS"),
                                                     child, i, child_offset);
        }
        const uint64_t data_length_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &data_length));
        if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              materializer->decoder, IREE_SV("tagged_location_data"),
              data_length, UINT32_MAX, data_length_offset);
        }
        iree_const_byte_span_t data_span = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            materializer->decoder, &cursor, data_length, &data_span));
        uint8_t* data = NULL;
        if (data_span.data_length > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(&materializer->output_module->arena,
                                  data_span.data_length, (void**)&data));
          memcpy(data, data_span.data, data_span.data_length);
        }
        entry.tagged.tag = (loom_location_tag_t)tag;
        entry.tagged.child = (loom_location_id_t)child;
        entry.tagged.data_length = (uint32_t)data_span.data_length;
        entry.tagged.data = data;
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            materializer->decoder, IREE_SV("location_kind"), kind,
            LOOM_LOCATION_COUNT_, kind_offset);
    }

    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_module_add_location(materializer->output_module,
                                                  entry, &location_id));
    if (location_id != i) {
      return loom_bytecode_reader_emit_invalid_field(
          materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("location_id"), kind_offset,
          IREE_SV("location_table_must_preserve_bytecode_location_ids"));
    }
  }
  return loom_bytecode_reader_expect_empty(materializer->decoder, &cursor,
                                           IREE_SV("LOCATIONS"));
}
