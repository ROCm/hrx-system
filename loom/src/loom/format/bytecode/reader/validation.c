// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/validation.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/format/bytecode/reader/symbol_validator.h"
#include "loom/format/bytecode/reader/type_validator.h"
#include "loom/ops/op_defs.h"

// Keep reader allocation guards aligned with the bytecode format comment.
#define LOOM_BYTECODE_MAX_SECTION_COUNT 256
#define LOOM_BYTECODE_MAX_STRING_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_STRING_LENGTH (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_OP_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_LOCATION_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_ENCODING_COUNT (UINT64_C(1) << 16)
#define LOOM_BYTECODE_MAX_PROVIDER_IMPORT_COUNT (UINT64_C(1) << 24)

// State required to validate exactly one module.
typedef struct loom_bytecode_module_reader_t {
  // Full bytecode file bytes.
  iree_const_byte_span_t bytecode;
  // Dialect and encoding registry context.
  loom_context_t* context;
  // Transient metadata arena.
  iree_arena_allocator_t* arena;
  // Persistent arena for public metadata index output.
  iree_arena_allocator_t* metadata_arena;
  // Bounded decoder sharing the file reader's diagnostic counters.
  loom_bytecode_reader_decoder_t decoder;
  // File-level source-location mode constraining this module.
  loom_bytecode_location_mode_t location_mode;
  // Scratch-lived facts established by module validation.
  loom_bytecode_reader_module_view_t view;
} loom_bytecode_module_reader_t;

static loom_bytecode_attribute_validator_t
loom_bytecode_reader_attribute_validator(
    loom_bytecode_module_reader_t* reader) {
  return (loom_bytecode_attribute_validator_t){
      .decoder = &reader->decoder,
      .context = reader->context,
      .module_view = &reader->view,
  };
}

static const char* loom_bytecode_section_name(uint16_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SECTION_STRINGS:
      return "STRINGS";
    case LOOM_BYTECODE_SECTION_SOURCES:
      return "SOURCES";
    case LOOM_BYTECODE_SECTION_TYPES:
      return "TYPES";
    case LOOM_BYTECODE_SECTION_ENCODINGS:
      return "ENCODINGS";
    case LOOM_BYTECODE_SECTION_OPS:
      return "OPS";
    case LOOM_BYTECODE_SECTION_LOCATIONS:
      return "LOCATIONS";
    case LOOM_BYTECODE_SECTION_SOURCE_TRIVIA:
      return "SOURCE_TRIVIA";
    case LOOM_BYTECODE_SECTION_SYMBOLS:
      return "SYMBOLS";
    case LOOM_BYTECODE_SECTION_IR:
      return "IR";
    case LOOM_BYTECODE_SECTION_RESOURCES:
      return "RESOURCES";
    case LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS:
      return "PROVIDER_IMPORTS";
    case LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES:
      return "SYMBOL_REFERENCES";
    default:
      return "UNKNOWN";
  }
}

static bool loom_bytecode_reader_string_is_valid_utf8(iree_string_view_t text) {
  return iree_unicode_utf8_validate(text);
}

static iree_status_t loom_bytecode_reader_read_file_header(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* section,
    loom_bytecode_source_trivia_t* out_file_header) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("SOURCE_TRIVIA"), &cursor);
  loom_bytecode_source_trivia_t source_trivia;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
      &reader->decoder, &cursor, reader->arena, &source_trivia));
  if (source_trivia.leading_blank_line) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SOURCE_TRIVIA"), IREE_SV("file_header"), 0,
        IREE_SV("leading_blank_line"), section->absolute_offset,
        IREE_SV("file_header_must_not_have_a_leading_blank_line"));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      &reader->decoder, &cursor, IREE_SV("file_header")));
  *out_file_header = source_trivia;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_string_ref(
    loom_bytecode_module_reader_t* reader, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= reader->view.strings.count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(reader->view.strings.count),
    };
    return loom_bytecode_reader_emit_error(&reader->decoder,
                                           LOOM_ERR_BYTECODE_010, params,
                                           IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_string = reader->view.strings.values[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_string_table(
    loom_bytecode_module_reader_t* reader,
    loom_bytecode_reader_section_t section, iree_string_view_t table_name,
    uint64_t count_limit, iree_string_view_t** out_strings,
    iree_host_size_t* out_count) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section.bytes.data, section.bytes.data_length, section.absolute_offset,
      table_name, &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &count));
  if (count > count_limit || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, table_name, count, count_limit, count_offset);
  }

  iree_string_view_t* strings = NULL;
  if (count > 0) {
    iree_arena_allocator_t* storage_arena =
        reader->view.output_metadata ? reader->metadata_arena : reader->arena;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        storage_arena, (iree_host_size_t)count, sizeof(iree_string_view_t),
        (void**)&strings));
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t string_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &length));
    if (length > LOOM_BYTECODE_MAX_STRING_LENGTH) {
      return loom_bytecode_reader_emit_count_exceeds(
          &reader->decoder, IREE_SV("string_length"), length,
          LOOM_BYTECODE_MAX_STRING_LENGTH, string_offset);
    }
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
        &reader->decoder, &cursor, length, &bytes));
    iree_string_view_t text =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!loom_bytecode_reader_string_is_valid_utf8(text)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, table_name, IREE_SV("string"), i,
          IREE_SV("utf8_data"), string_offset,
          IREE_SV("string_payload_is_not_valid_utf_8"));
    }
    strings[i] = text;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_expect_empty(&reader->decoder, &cursor, table_name));
  *out_strings = strings;
  *out_count = (iree_host_size_t)count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_encodings(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("ENCODINGS"), &cursor);

  uint64_t family_count = 0;
  uint64_t family_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &family_count));
  if (family_count > LOOM_BYTECODE_MAX_ENCODING_COUNT ||
      family_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("encoding_families"), family_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, family_count_offset);
  }
  if (family_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)family_count, sizeof(loom_string_id_t),
        (void**)&reader->view.encodings.family_name_ids));
  }
  reader->view.encodings.family_count = (iree_host_size_t)family_count;
  for (uint64_t i = 0; i < family_count; ++i) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &name_id));
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("encoding_family_name"), name_offset, &name));
    if (loom_context_resolve_encoding_name(reader->context, name).family_id ==
        LOOM_ENCODING_FAMILY_ID_INVALID) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("ENCODINGS"), IREE_SV("family"), i,
          IREE_SV("name_id"), name_offset,
          IREE_SV("encoding_family_is_not_registered_in_the_context"));
    }
    reader->view.encodings.family_name_ids[i] = (loom_string_id_t)name_id;
  }

  uint64_t instance_count = 0;
  uint64_t instance_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &instance_count));
  if (instance_count > LOOM_BYTECODE_MAX_ENCODING_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("encoding_instances"), instance_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, instance_count_offset);
  }
  reader->view.encodings.count = (iree_host_size_t)instance_count;
  loom_bytecode_encoding_metadata_t* encoding_metadata = NULL;
  if (reader->view.output_metadata) {
    reader->view.output_metadata->encodings.count =
        (iree_host_size_t)instance_count;
    if (instance_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)instance_count,
          sizeof(*encoding_metadata), (void**)&encoding_metadata));
      reader->view.output_metadata->encodings.entries = encoding_metadata;
    }
  }
  loom_bytecode_attribute_validator_t attribute_validator =
      loom_bytecode_reader_attribute_validator(reader);
  for (uint64_t i = 0; i < instance_count; ++i) {
    const uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t family_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t family_index = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &family_index));
    if (family_index >= family_count) {
      return loom_bytecode_reader_emit_table_ref(
          &reader->decoder, IREE_SV("encoding_families"), family_index,
          family_count, family_offset);
    }
    uint64_t alias_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t alias_plus_one = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &alias_plus_one));
    if (alias_plus_one > reader->view.strings.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("alias_string_id_plus1")),
          loom_param_u64(alias_plus_one - 1),
          loom_param_u64(reader->view.strings.count),
      };
      return loom_bytecode_reader_emit_error(
          &reader->decoder, LOOM_ERR_BYTECODE_010, params,
          IREE_ARRAYSIZE(params), alias_offset, 0);
    }
    uint64_t param_count = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &param_count));
    iree_string_view_t previous_param = iree_string_view_empty();
    for (uint64_t param_index = 0; param_index < param_count; ++param_index) {
      uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t name_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &name_id));
      iree_string_view_t param_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, name_id, IREE_SV("encoding_param_name"), name_offset,
          &param_name));
      if (param_index > 0 &&
          iree_string_view_compare(param_name, previous_param) <= 0) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("ENCODINGS"), IREE_SV("instance"), i,
            IREE_SV("param_name"), name_offset,
            IREE_SV("encoding_parameters_are_not_in_canonical_order"));
      }
      previous_param = param_name;
      loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
          &reader->decoder, &cursor, &value_kind));
      IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_named(
          &attribute_validator, &cursor, /*descriptor=*/NULL, value_kind,
          reader->view.types.count));
    }
    if (encoding_metadata) {
      encoding_metadata[i] = (loom_bytecode_encoding_metadata_t){
          .entry_offset = entry_offset,
          .entry_length =
              loom_bytecode_reader_cursor_absolute_position(&cursor) -
              entry_offset,
          .name_string_index =
              (uint32_t)reader->view.encodings.family_name_ids[family_index],
      };
    }
  }
  return loom_bytecode_reader_expect_empty(&reader->decoder, &cursor,
                                           IREE_SV("ENCODINGS"));
}

static iree_status_t loom_bytecode_reader_read_ops(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("OPS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &count));
  if (count > LOOM_BYTECODE_MAX_OP_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("OPS"), count, LOOM_BYTECODE_MAX_OP_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(const loom_op_vtable_t*),
        (void**)&reader->view.ops.values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(loom_op_kind_t),
        (void**)&reader->view.ops.kinds));
  }
  loom_bytecode_op_metadata_t* op_metadata = NULL;
  if (reader->view.output_metadata) {
    reader->view.output_metadata->ops.count = (iree_host_size_t)count;
    if (count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)count, sizeof(*op_metadata),
          (void**)&op_metadata));
      reader->view.output_metadata->ops.entries = op_metadata;
    }
  }
  reader->view.ops.count = (iree_host_size_t)count;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &name_id));
    iree_string_view_t op_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("op_name"), name_offset, &op_name));
    loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
    const loom_op_vtable_t* vtable =
        loom_context_lookup_op_by_name(reader->context, op_name, &kind);
    if (!vtable) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("OPS"), IREE_SV("op"), i,
          IREE_SV("name_id"), name_offset,
          IREE_SV("op_name_is_not_registered_in_the_context"));
    }
    reader->view.ops.values[i] = vtable;
    reader->view.ops.kinds[i] = kind;
    if (op_metadata) {
      op_metadata[i].name = op_name;
    }
  }
  return loom_bytecode_reader_expect_empty(&reader->decoder, &cursor,
                                           IREE_SV("OPS"));
}

static iree_status_t loom_bytecode_reader_read_locations(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &count));
  if (count > LOOM_BYTECODE_MAX_LOCATION_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("LOCATIONS"), count,
        LOOM_BYTECODE_MAX_LOCATION_COUNT, count_offset);
  }
  reader->view.locations.count = (iree_host_size_t)count;
  loom_bytecode_table_entry_metadata_t* location_metadata = NULL;
  if (reader->view.output_metadata) {
    reader->view.output_metadata->locations.count = (iree_host_size_t)count;
    if (count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)count,
          sizeof(*location_metadata), (void**)&location_metadata));
      reader->view.output_metadata->locations.entries = location_metadata;
    }
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint8_t kind = 0;
    uint8_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &flags));
    if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("flags"), kind_offset + 1,
          IREE_SV("location_has_unsupported_flag_bits"));
    }
    switch ((loom_location_kind_t)kind) {
      case LOOM_LOCATION_NONE:
        break;
      case LOOM_LOCATION_FILE: {
        uint64_t source_id = 0;
        uint64_t source_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &source_id));
        if (source_id >= reader->view.sources.count) {
          return loom_bytecode_reader_emit_table_ref(
              &reader->decoder, IREE_SV("SOURCES"), source_id,
              reader->view.sources.count, source_offset);
        }
        for (int field = 0; field < 4; ++field) {
          uint64_t value = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              &reader->decoder, &cursor, &value));
        }
        break;
      }
      case LOOM_LOCATION_FUSED: {
        uint64_t child_count = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &child_count));
        for (uint64_t child_index = 0; child_index < child_count;
             ++child_index) {
          uint64_t child_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t child = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              &reader->decoder, &cursor, &child));
          if (child >= i) {
            return loom_bytecode_reader_emit_table_ref(
                &reader->decoder, IREE_SV("LOCATIONS"), child, i, child_offset);
          }
        }
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        uint64_t source_id = 0;
        uint64_t source_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &source_id));
        if (source_id >= reader->view.sources.count) {
          return loom_bytecode_reader_emit_table_ref(
              &reader->decoder, IREE_SV("SOURCES"), source_id,
              reader->view.sources.count, source_offset);
        }
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &data_length));
        iree_const_byte_span_t unused = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            &reader->decoder, &cursor, data_length, &unused));
        break;
      }
      case LOOM_LOCATION_TAGGED: {
        uint64_t tag = 0;
        uint64_t tag_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &tag));
        if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
              IREE_SV("tag"), tag_offset,
              IREE_SV("tagged location tag must be in [1, 65535]"));
        }
        uint64_t child = 0;
        uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &child));
        if (child >= i) {
          return loom_bytecode_reader_emit_table_ref(
              &reader->decoder, IREE_SV("LOCATIONS"), child, i, child_offset);
        }
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &data_length));
        iree_const_byte_span_t unused = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            &reader->decoder, &cursor, data_length, &unused));
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            &reader->decoder, IREE_SV("location_kind"), kind,
            LOOM_LOCATION_COUNT_, kind_offset);
    }
    if (location_metadata) {
      location_metadata[i] = (loom_bytecode_table_entry_metadata_t){
          .entry_offset = kind_offset,
          .entry_length =
              loom_bytecode_reader_cursor_absolute_position(&cursor) -
              kind_offset,
      };
    }
  }
  return loom_bytecode_reader_expect_empty(&reader->decoder, &cursor,
                                           IREE_SV("LOCATIONS"));
}

static iree_status_t loom_bytecode_reader_read_provider_imports(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* provider_imports_section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      provider_imports_section->bytes.data,
      provider_imports_section->bytes.data_length,
      provider_imports_section->absolute_offset, IREE_SV("PROVIDER_IMPORTS"),
      &cursor);

  uint64_t provider_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  uint64_t provider_count = 0;
  uint64_t total_anchor_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &provider_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &total_anchor_count));
  if (provider_count > LOOM_BYTECODE_MAX_PROVIDER_IMPORT_COUNT ||
      provider_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), provider_count,
        LOOM_BYTECODE_MAX_PROVIDER_IMPORT_COUNT, provider_count_offset);
  }
  if (total_anchor_count > UINT32_MAX ||
      total_anchor_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("header"), 0,
        IREE_SV("total_anchor_count"), provider_count_offset,
        IREE_SV("provider_anchor_count_exceeds_index_width"));
  }

  reader->view.provider_imports.count = (iree_host_size_t)provider_count;
  reader->view.provider_imports.anchor_count =
      (iree_host_size_t)total_anchor_count;
  if (provider_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)provider_count,
        sizeof(*reader->view.provider_imports.values),
        (void**)&reader->view.provider_imports.values));
  }
  if (total_anchor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)total_anchor_count,
        sizeof(*reader->view.provider_imports.anchors),
        (void**)&reader->view.provider_imports.anchors));
  }

  iree_host_size_t anchor_index = 0;
  iree_string_view_t previous_provider = iree_string_view_empty();
  const iree_host_size_t anchor_usage_word_count =
      reader->view.symbols.unresolved_anchor_count > 0
          ? (reader->view.symbols.count + 63) / 64
          : 0;
  uint64_t* anchor_usage_bits = NULL;
  if (anchor_usage_word_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, anchor_usage_word_count, sizeof(*anchor_usage_bits),
        (void**)&anchor_usage_bits));
    memset(anchor_usage_bits, 0,
           anchor_usage_word_count * sizeof(*anchor_usage_bits));
  }
  for (iree_host_size_t provider_index = 0;
       provider_index < (iree_host_size_t)provider_count; ++provider_index) {
    uint64_t provider_id_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t provider_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &provider_id));
    iree_string_view_t provider = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, provider_id, IREE_SV("provider_string_id"), provider_id_offset,
        &provider));
    if (provider_index > 0 &&
        iree_string_view_compare(previous_provider, provider) >= 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("provider"),
          provider_index, IREE_SV("provider_string_id"), provider_id_offset,
          IREE_SV("providers_must_be_strictly_sorted_by_exact_key"));
    }
    previous_provider = provider;

    uint64_t anchor_count_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t anchor_count = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &anchor_count));
    if (anchor_count > UINT16_MAX ||
        anchor_count > total_anchor_count - anchor_index) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("provider"),
          provider_index, IREE_SV("anchor_count"), anchor_count_offset,
          IREE_SV("provider_anchor_count_exceeds_declared_total"));
    }

    loom_bytecode_reader_provider_import_t* provider_import =
        &reader->view.provider_imports.values[provider_index];
    provider_import->provider_id = (loom_string_id_t)provider_id;
    provider_import->first_anchor_index = (uint32_t)anchor_index;
    provider_import->anchor_count = (uint32_t)anchor_count;

    iree_string_view_t previous_anchor = iree_string_view_empty();
    for (iree_host_size_t local_anchor_index = 0;
         local_anchor_index < (iree_host_size_t)anchor_count;
         ++local_anchor_index, ++anchor_index) {
      uint64_t symbol_index_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t symbol_index = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &symbol_index));
      if (symbol_index >= reader->view.symbols.count) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("anchor"),
            anchor_index, IREE_SV("symbol_index"), symbol_index_offset,
            IREE_SV("provider_anchor_symbol_index_is_out_of_range"));
      }
      iree_string_view_t anchor_name =
          reader->view.strings
              .values[reader->view.symbols.name_ids[symbol_index]];
      if (local_anchor_index > 0 &&
          iree_string_view_compare(previous_anchor, anchor_name) >= 0) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("anchor"),
            anchor_index, IREE_SV("symbol_index"), symbol_index_offset,
            IREE_SV("provider_anchors_must_be_strictly_sorted_by_name"));
      }
      previous_anchor = anchor_name;
      reader->view.provider_imports.anchors[anchor_index] = (loom_symbol_ref_t){
          .module_id = 0, .symbol_id = (uint16_t)symbol_index};
      if (reader->view.symbols.kinds[symbol_index] ==
          LOOM_BYTECODE_SYMBOL_ANCHOR) {
        anchor_usage_bits[symbol_index / 64] |= UINT64_C(1)
                                                << (symbol_index % 64);
      }
    }

    IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
        &reader->decoder, &cursor, reader->arena,
        &provider_import->source_trivia));
  }

  if (anchor_index != (iree_host_size_t)total_anchor_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("header"), 0,
        IREE_SV("total_anchor_count"), provider_count_offset,
        IREE_SV("provider_anchor_records_do_not_match_declared_total"));
  }
  for (iree_host_size_t symbol_index = 0;
       symbol_index < reader->view.symbols.count; ++symbol_index) {
    if (reader->view.symbols.kinds[symbol_index] !=
        LOOM_BYTECODE_SYMBOL_ANCHOR) {
      continue;
    }
    if ((anchor_usage_bits[symbol_index / 64] &
         (UINT64_C(1) << (symbol_index % 64))) == 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("PROVIDER_IMPORTS"), IREE_SV("anchor"),
          symbol_index, IREE_SV("symbol_index"),
          provider_imports_section->absolute_offset,
          IREE_SV("provider_anchor_symbol_is_not_used_by_any_provider"));
    }
  }

  return loom_bytecode_reader_expect_empty(&reader->decoder, &cursor,
                                           IREE_SV("PROVIDER_IMPORTS"));
}

// Keep index-only retained allocation out of shared module validation so full
// reads do not pay its instruction-cache cost.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_reader_project_provider_imports(
    const loom_bytecode_reader_module_view_t* view,
    loom_bytecode_module_metadata_t* output_metadata,
    iree_arena_allocator_t* retained_arena) {
  output_metadata->provider_import_count = view->provider_imports.count;
  if (view->provider_imports.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(retained_arena, view->provider_imports.count,
                                  sizeof(*output_metadata->provider_imports),
                                  (void**)&output_metadata->provider_imports));
  }
  output_metadata->provider_import_anchor_count =
      view->provider_imports.anchor_count;
  if (view->provider_imports.anchor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        retained_arena, view->provider_imports.anchor_count,
        sizeof(*output_metadata->provider_import_anchor_symbol_indices),
        (void**)&output_metadata->provider_import_anchor_symbol_indices));
  }

  for (iree_host_size_t provider_index = 0;
       provider_index < view->provider_imports.count; ++provider_index) {
    const loom_bytecode_reader_provider_import_t* provider_import =
        &view->provider_imports.values[provider_index];
    iree_string_view_t* retained_comments = NULL;
    if (provider_import->source_trivia.comment_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          retained_arena, provider_import->source_trivia.comment_count,
          sizeof(*retained_comments), (void**)&retained_comments));
      memcpy(retained_comments, provider_import->source_trivia.comments,
             provider_import->source_trivia.comment_count *
                 sizeof(*retained_comments));
    }
    output_metadata->provider_imports[provider_index] =
        (loom_bytecode_provider_import_metadata_t){
            .provider = view->strings.values[provider_import->provider_id],
            .first_anchor_index = provider_import->first_anchor_index,
            .anchor_count = provider_import->anchor_count,
            .leading_blank_line =
                provider_import->source_trivia.leading_blank_line,
            .comments = retained_comments,
            .comment_count = provider_import->source_trivia.comment_count,
        };
  }
  for (iree_host_size_t anchor_index = 0;
       anchor_index < view->provider_imports.anchor_count; ++anchor_index) {
    output_metadata->provider_import_anchor_symbol_indices[anchor_index] =
        view->provider_imports.anchors[anchor_index].symbol_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_symbol_references(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* symbol_references_section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      symbol_references_section->bytes.data,
      symbol_references_section->bytes.data_length,
      symbol_references_section->absolute_offset, IREE_SV("SYMBOL_REFERENCES"),
      &cursor);

  const uint64_t symbol_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  uint64_t symbol_count = 0;
  uint64_t total_dependency_count = 0;
  uint64_t total_contract_demand_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &symbol_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &total_dependency_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &total_contract_demand_count));
  if (symbol_count != reader->view.symbols.count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("header"), 0,
        IREE_SV("symbol_count"), symbol_count_offset,
        IREE_SV("symbol_reference_row_count_must_match_symbols"));
  }
  if (total_dependency_count > UINT32_MAX ||
      total_dependency_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("header"), 0,
        IREE_SV("total_dependency_count"), symbol_count_offset,
        IREE_SV("dependency_count_exceeds_index_width"));
  }
  if (total_contract_demand_count > UINT32_MAX ||
      total_contract_demand_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("header"), 0,
        IREE_SV("total_contract_demand_count"), symbol_count_offset,
        IREE_SV("contract_demand_count_exceeds_index_width"));
  }

  // Every row count and referenced ordinal consumes at least one byte. Bound
  // retained allocations by the section payload before allocating them.
  const uint64_t minimum_encoded_bytes = 1 + symbol_count * 2 +
                                         total_dependency_count +
                                         total_contract_demand_count;
  if (minimum_encoded_bytes > loom_bytecode_cursor_remaining(&cursor.cursor)) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("header"), 0,
        IREE_SV("record_counts"), symbol_count_offset,
        IREE_SV("declared_reference_records_exceed_section_length"));
  }

  loom_bytecode_module_metadata_t* module_metadata =
      reader->view.output_metadata;
  if (module_metadata) {
    module_metadata->dependency_count =
        (iree_host_size_t)total_dependency_count;
    module_metadata->contract_demand_count =
        (iree_host_size_t)total_contract_demand_count;
    if (symbol_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)symbol_count,
          sizeof(*module_metadata->symbol_references),
          (void**)&module_metadata->symbol_references));
      memset(module_metadata->symbol_references, 0,
             (iree_host_size_t)symbol_count *
                 sizeof(*module_metadata->symbol_references));
    }
    if (total_dependency_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)total_dependency_count,
          sizeof(*module_metadata->dependency_symbol_indices),
          (void**)&module_metadata->dependency_symbol_indices));
    }
    if (total_contract_demand_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)total_contract_demand_count,
          sizeof(*module_metadata->contract_demands),
          (void**)&module_metadata->contract_demands));
    }
  }

  uint64_t module_dependency_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  uint64_t module_dependency_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &module_dependency_count));
  if (module_dependency_count > total_dependency_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("module"), 0,
        IREE_SV("dependency_count"), module_dependency_count_offset,
        IREE_SV("row_dependency_count_exceeds_declared_total"));
  }
  if (module_metadata) {
    module_metadata->module_dependency_count =
        (uint32_t)module_dependency_count;
  }

  iree_host_size_t dependency_index = 0;
  for (uint64_t i = 0; i < module_dependency_count; ++i, ++dependency_index) {
    const uint64_t dependency_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t dependency = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &dependency));
    if (dependency >= symbol_count) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOL_REFERENCES"),
          IREE_SV("module_dependency"), i, IREE_SV("target_symbol_index"),
          dependency_offset,
          IREE_SV("dependency_symbol_index_is_out_of_range"));
    }
    if (module_metadata) {
      module_metadata->dependency_symbol_indices[dependency_index] =
          (uint32_t)dependency;
    }
  }

  iree_host_size_t contract_demand_index = 0;
  for (iree_host_size_t symbol_index = 0;
       symbol_index < reader->view.symbols.count; ++symbol_index) {
    const uint64_t dependency_count_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t dependency_count = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &dependency_count));
    if (dependency_count > total_dependency_count - dependency_index) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("symbol"),
          symbol_index, IREE_SV("dependency_count"), dependency_count_offset,
          IREE_SV("row_dependency_count_exceeds_declared_total"));
    }
    loom_bytecode_symbol_reference_metadata_t* symbol_metadata =
        module_metadata ? &module_metadata->symbol_references[symbol_index]
                        : NULL;
    if (symbol_metadata) {
      symbol_metadata->first_dependency_index = (uint32_t)dependency_index;
      symbol_metadata->dependency_count = (uint32_t)dependency_count;
    }

    for (uint64_t i = 0; i < dependency_count; ++i, ++dependency_index) {
      const uint64_t dependency_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t dependency = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &dependency));
      if (dependency >= symbol_count) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOL_REFERENCES"),
            IREE_SV("dependency"), i, IREE_SV("target_symbol_index"),
            dependency_offset,
            IREE_SV("dependency_symbol_index_is_out_of_range"));
      }
      if (module_metadata) {
        module_metadata->dependency_symbol_indices[dependency_index] =
            (uint32_t)dependency;
      }
    }

    const uint64_t contract_count_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t contract_count = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, &contract_count));
    if (contract_count > total_contract_demand_count - contract_demand_index) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("symbol"),
          symbol_index, IREE_SV("contract_demand_count"), contract_count_offset,
          IREE_SV("row_contract_count_exceeds_declared_total"));
    }
    if (symbol_metadata) {
      symbol_metadata->first_contract_demand_index =
          (uint32_t)contract_demand_index;
      symbol_metadata->contract_demand_count = (uint32_t)contract_count;
    }

    for (uint64_t i = 0; i < contract_count; ++i, ++contract_demand_index) {
      const uint64_t contract_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t contract_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, &cursor, &contract_string_id));
      iree_string_view_t contract = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, contract_string_id, IREE_SV("contract_string_id"),
          contract_offset, &contract));
      if (module_metadata) {
        module_metadata->contract_demands[contract_demand_index] = contract;
      }
    }
  }

  if (dependency_index != (iree_host_size_t)total_dependency_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("header"), 0,
        IREE_SV("total_dependency_count"), symbol_count_offset,
        IREE_SV("dependency_records_do_not_match_declared_total"));
  }
  if (contract_demand_index != (iree_host_size_t)total_contract_demand_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOL_REFERENCES"), IREE_SV("header"), 0,
        IREE_SV("total_contract_demand_count"), symbol_count_offset,
        IREE_SV("contract_records_do_not_match_declared_total"));
  }

  reader->view.summary.dependency_count = total_dependency_count;
  reader->view.summary.contract_demand_count = total_contract_demand_count;
  return loom_bytecode_reader_expect_empty(&reader->decoder, &cursor,
                                           IREE_SV("SYMBOL_REFERENCES"));
}

static iree_status_t loom_bytecode_reader_validate_file_header(
    loom_bytecode_file_reader_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_string_pool_length) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 16)) {
    return loom_bytecode_reader_emit_unexpected_end(
        &reader->decoder, 0, 16,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }

  uint8_t magic[LOOM_BYTECODE_MAGIC_LENGTH] = {0};
  iree_const_byte_span_t magic_span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
      &reader->decoder, cursor, LOOM_BYTECODE_MAGIC_LENGTH, &magic_span));
  memcpy(magic, magic_span.data, sizeof(magic));
  if (memcmp(magic, LOOM_BYTECODE_MAGIC, LOOM_BYTECODE_MAGIC_LENGTH) != 0) {
    char actual_magic[9] = {0};
    static const char kHex[] = "0123456789ABCDEF";
    for (int i = 0; i < LOOM_BYTECODE_MAGIC_LENGTH; ++i) {
      actual_magic[i * 2 + 0] = kHex[magic[i] >> 4];
      actual_magic[i * 2 + 1] = kHex[magic[i] & 0xF];
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV(LOOM_BYTECODE_MAGIC)),
        loom_param_string(iree_make_cstring_view(actual_magic)),
    };
    return loom_bytecode_reader_emit_error(
        &reader->decoder, LOOM_ERR_BYTECODE_001, params, IREE_ARRAYSIZE(params),
        0, LOOM_BYTECODE_MAGIC_LENGTH);
  }

  uint8_t version = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &version));
  if (version != LOOM_BYTECODE_FORMAT_VERSION) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(version),
        loom_param_u32(LOOM_BYTECODE_FORMAT_VERSION),
    };
    return loom_bytecode_reader_emit_error(&reader->decoder,
                                           LOOM_ERR_BYTECODE_002, params,
                                           IREE_ARRAYSIZE(params), 4, 1);
  }
  reader->format_version = version;
  uint8_t location_mode = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &location_mode));
  if (location_mode > LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    return loom_bytecode_reader_emit_enum_value(
        &reader->decoder, IREE_SV("location_mode"), location_mode,
        LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS + 1, 5);
  }
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("FILE"), IREE_SV("header"), 0,
        IREE_SV("location_mode"), 5,
        IREE_SV("full_locations_bytecode_requires_field_span_support"));
  }
  reader->result.location_mode = (loom_bytecode_location_mode_t)location_mode;

  uint16_t module_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u16_le(&reader->decoder,
                                                        cursor, &module_count));
  if (module_count == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("FILE"), IREE_SV("header"), 0,
        IREE_SV("module_count"), 6,
        IREE_SV("bytecode_files_must_contain_at_least_one_module"));
  }
  reader->module_count = module_count;
  reader->result.module_count = module_count;

  uint32_t string_pool_length = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u32_le(
      &reader->decoder, cursor, &string_pool_length));
  *out_string_pool_length = string_pool_length;
  uint32_t reserved = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u32_le(&reader->decoder, cursor, &reserved));
  if (reserved != 0) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("FILE"), IREE_SV("header"), 0,
        IREE_SV("reserved"), 12, IREE_SV("reserved_header_field_must_be_zero"));
  }

  uint64_t producer_start = cursor->cursor.position;
  while (cursor->cursor.position < cursor->cursor.length &&
         cursor->cursor.data[cursor->cursor.position] != 0) {
    ++cursor->cursor.position;
  }
  if (cursor->cursor.position >= cursor->cursor.length) {
    return loom_bytecode_reader_emit_unexpected_end(
        &reader->decoder, producer_start, 1,
        cursor->cursor.length >= producer_start
            ? cursor->cursor.length - producer_start
            : 0);
  }
  iree_string_view_t producer =
      iree_make_string_view((const char*)cursor->cursor.data + producer_start,
                            cursor->cursor.position - producer_start);
  if (!loom_bytecode_reader_string_is_valid_utf8(producer)) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("FILE"), IREE_SV("header"), 0,
        IREE_SV("producer"), producer_start,
        IREE_SV("producer_string_is_not_valid_utf_8"));
  }
  reader->producer = producer;
  ++cursor->cursor.position;
  while ((cursor->cursor.position & 7) != 0) {
    uint64_t padding_offset = cursor->cursor.position;
    uint8_t padding = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, cursor, &padding));
    if (padding != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("FILE"), IREE_SV("header"), 0,
          IREE_SV("producer_padding"), padding_offset,
          IREE_SV("header_alignment_padding_must_be_zero"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_module_directory(
    loom_bytecode_file_reader_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t string_pool_length) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, reader->module_count,
      sizeof(loom_bytecode_reader_module_t), (void**)&reader->modules));
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    loom_bytecode_reader_module_t* module = &reader->modules[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u32_le(
        &reader->decoder, cursor, &module->name_offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u16_le(
        &reader->decoder, cursor, &module->name_length));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u16_le(
        &reader->decoder, cursor, &module->flags));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &reader->decoder, cursor, &module->offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &reader->decoder, cursor, &module->length));
    if (module->flags & ~LOOM_BYTECODE_MODULE_FLAG_DECLARATIONS_ONLY) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("FILE"), IREE_SV("module_directory"), i,
          IREE_SV("flags"), entry_offset + 6,
          IREE_SV("module_has_unsupported_flag_bits"));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        &reader->decoder, IREE_SV("file_string_pool"), module->name_offset,
        module->name_length, string_pool_length));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        &reader->decoder, IREE_SV("module"), module->offset, module->length,
        reader->bytecode.data_length));
  }

  uint64_t string_pool_end = 0;
  if (string_pool_length > UINT64_MAX - cursor->cursor.position) {
    return loom_bytecode_reader_emit_range_error(
        &reader->decoder, IREE_SV("file_string_pool"), cursor->cursor.position,
        string_pool_length, reader->bytecode.data_length);
  }
  string_pool_end = cursor->cursor.position + string_pool_length;
  if (string_pool_end > UINT64_MAX - 7) {
    return loom_bytecode_reader_emit_range_error(
        &reader->decoder, IREE_SV("file_string_pool"), cursor->cursor.position,
        string_pool_length, reader->bytecode.data_length);
  }
  uint64_t previous_end = (string_pool_end + 7) & ~UINT64_C(7);
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    const loom_bytecode_reader_module_t* module = &reader->modules[i];
    if (module->offset < previous_end) {
      return loom_bytecode_reader_emit_range_error(
          &reader->decoder, IREE_SV("module"), module->offset, module->length,
          reader->bytecode.data_length);
    }
    previous_end = module->offset + module->length;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_file_string_pool(
    loom_bytecode_file_reader_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t string_pool_length) {
  iree_const_byte_span_t pool = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
      &reader->decoder, cursor, string_pool_length, &pool));
  reader->file_string_pool =
      iree_make_string_view((const char*)pool.data, pool.data_length);
  if (!loom_bytecode_reader_string_is_valid_utf8(reader->file_string_pool)) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("FILE"), IREE_SV("string_pool"), 0,
        IREE_SV("utf8_data"), cursor->absolute_offset,
        IREE_SV("file_string_pool_is_not_valid_utf_8"));
  }
  while ((cursor->cursor.position & 7) != 0) {
    uint64_t padding_offset = cursor->cursor.position;
    uint8_t padding = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, cursor, &padding));
    if (padding != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("FILE"), IREE_SV("string_pool"), 0,
          IREE_SV("padding"), padding_offset,
          IREE_SV("file_string_pool_alignment_padding_must_be_zero"));
    }
  }
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    loom_bytecode_reader_module_t* module = &reader->modules[i];
    module->name = iree_make_string_view(
        reader->file_string_pool.data + module->name_offset,
        module->name_length);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_file_reader_project_index(
    const loom_bytecode_file_reader_t* reader,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_file_metadata_t* metadata) {
  metadata->format_version = reader->format_version;
  metadata->location_mode = reader->result.location_mode;
  metadata->producer = reader->producer;
  metadata->module_count = reader->module_count;
  if (reader->module_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        retained_arena, reader->module_count,
        sizeof(loom_bytecode_module_metadata_t), (void**)&metadata->modules));
    memset(metadata->modules, 0,
           reader->module_count * sizeof(loom_bytecode_module_metadata_t));
  }
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    const loom_bytecode_reader_module_t* source = &reader->modules[i];
    loom_bytecode_module_metadata_t* target = &metadata->modules[i];
    target->name = source->name;
    target->flags = source->flags;
    target->offset = source->offset;
    target->length = source->length;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_section_directory(
    loom_bytecode_module_reader_t* reader) {
  const loom_bytecode_reader_module_t* module = reader->view.directory_entry;
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      reader->bytecode.data + module->offset, (iree_host_size_t)module->length,
      module->offset, IREE_SV("MODULE"), &cursor);

  uint64_t section_count = 0;
  uint64_t section_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, &cursor, &section_count));
  if (section_count > LOOM_BYTECODE_MAX_SECTION_COUNT ||
      section_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("SECTIONS"), section_count,
        LOOM_BYTECODE_MAX_SECTION_COUNT, section_count_offset);
  }

  memset(&reader->view.summary, 0, sizeof(reader->view.summary));
  uint64_t* summary_fields[] = {
      &reader->view.summary.value_count,
      &reader->view.summary.region_count,
      &reader->view.summary.block_count,
      &reader->view.summary.op_count,
  };
  for (int i = 0; i < 4; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, &cursor, summary_fields[i]));
  }

  loom_bytecode_reader_section_t* sections = NULL;
  if (section_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)section_count,
        sizeof(loom_bytecode_reader_section_t), (void**)&sections));
  }
  uint64_t minimum_section_offset = cursor.cursor.position + section_count * 32;
  uint64_t previous_end = minimum_section_offset;
  for (uint64_t i = 0; i < section_count; ++i) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    loom_bytecode_reader_section_t* section = &sections[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u16_le(
        &reader->decoder, &cursor, &section->kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u16_le(
        &reader->decoder, &cursor, &section->flags));
    uint32_t reserved = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(&reader->decoder, &cursor, &reserved));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &reader->decoder, &cursor, &section->offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &reader->decoder, &cursor, &section->length));
    uint64_t uncompressed_length = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        &reader->decoder, &cursor, &uncompressed_length));
    if (section->flags != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("flags"), entry_offset + 2,
          IREE_SV("section_has_unsupported_flag_bits"));
    }
    if (reserved != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("reserved"), entry_offset + 4,
          IREE_SV("section_reserved_field_must_be_zero"));
    }
    if (uncompressed_length != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("uncompressed_length"), entry_offset + 24,
          IREE_SV(
              "uncompressed length must be zero for uncompressed sections"));
    }
    for (uint64_t j = 0; j < i; ++j) {
      if (sections[j].kind == section->kind) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
            IREE_SV("kind"), entry_offset,
            IREE_SV("section_kind_appears_more_than_once"));
      }
    }
    if (section->offset < previous_end) {
      return loom_bytecode_reader_emit_range_error(
          &reader->decoder, IREE_SV("section"), section->offset,
          section->length, module->length);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        &reader->decoder, IREE_SV("section"), section->offset, section->length,
        module->length));
    section->absolute_offset = module->offset + section->offset;
    section->bytes = iree_make_const_byte_span(
        reader->bytecode.data + section->absolute_offset,
        (iree_host_size_t)section->length);
    previous_end = section->offset + section->length;
  }
  if (reader->view.output_metadata) {
    loom_bytecode_module_metadata_t* metadata = reader->view.output_metadata;
    metadata->summary = reader->view.summary;
    metadata->section_count = (iree_host_size_t)section_count;
    if (section_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)section_count,
          sizeof(loom_bytecode_section_metadata_t),
          (void**)&metadata->sections));
      for (uint64_t i = 0; i < section_count; ++i) {
        metadata->sections[i] = (loom_bytecode_section_metadata_t){
            .kind = sections[i].kind,
            .flags = sections[i].flags,
            .offset = sections[i].offset,
            .length = sections[i].length,
            .absolute_offset = sections[i].absolute_offset,
            .bytes = sections[i].bytes,
        };
      }
    }
  }
  reader->view.sections.values = sections;
  reader->view.sections.count = (iree_host_size_t)section_count;
  return iree_ok_status();
}

static const loom_bytecode_reader_section_t* loom_bytecode_reader_find_section(
    const loom_bytecode_reader_section_t* sections, iree_host_size_t count,
    uint16_t kind) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (sections[i].kind == kind) {
      return &sections[i];
    }
  }
  return NULL;
}

static iree_status_t loom_bytecode_reader_require_section(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_section_t* sections, iree_host_size_t count,
    uint16_t kind, const loom_bytecode_reader_section_t** out_section) {
  const loom_bytecode_reader_section_t* section =
      loom_bytecode_reader_find_section(sections, count, kind);
  if (!section) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
        IREE_SV("required_section"), 0,
        iree_make_cstring_view(loom_bytecode_section_name(kind)));
  }
  *out_section = section;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_module(
    loom_bytecode_module_reader_t* reader,
    const loom_bytecode_reader_module_t* module,
    loom_bytecode_module_metadata_t* output_metadata) {
  reader->view = (loom_bytecode_reader_module_view_t){
      .directory_entry = module,
      .output_metadata = output_metadata,
  };

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_section_directory(reader));

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_STRINGS, &reader->view.sections.strings));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_SOURCES, &reader->view.sections.sources));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_ENCODINGS, &reader->view.sections.encodings));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_TYPES, &reader->view.sections.types));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_OPS, &reader->view.sections.ops));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_SYMBOLS, &reader->view.sections.symbols));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS,
      &reader->view.sections.provider_imports));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES,
      &reader->view.sections.symbol_references));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_IR, &reader->view.sections.ir));

  reader->view.sections.locations = loom_bytecode_reader_find_section(
      reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_LOCATIONS);
  reader->view.sections.source_trivia = loom_bytecode_reader_find_section(
      reader->view.sections.values, reader->view.sections.count,
      LOOM_BYTECODE_SECTION_SOURCE_TRIVIA);
  if (reader->location_mode == LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    if (reader->view.sections.locations) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
          IREE_SV("LOCATIONS"),
          reader->view.sections.locations->absolute_offset,
          IREE_SV("no_locations_bytecode_must_not_contain_locations"));
    }
  } else if (!reader->view.sections.locations) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
        IREE_SV("LOCATIONS"), 0,
        IREE_SV("source_location_bytecode_must_contain_locations"));
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_string_table(
      reader, *reader->view.sections.strings, IREE_SV("STRINGS"),
      LOOM_BYTECODE_MAX_STRING_COUNT, &reader->view.strings.values,
      &reader->view.strings.count));
  if (reader->view.output_metadata) {
    reader->view.output_metadata->strings.count = reader->view.strings.count;
    reader->view.output_metadata->strings.values = reader->view.strings.values;
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_string_table(
      reader, *reader->view.sections.sources, IREE_SV("SOURCES"),
      LOOM_BYTECODE_MAX_STRING_COUNT, &reader->view.sources.values,
      &reader->view.sources.count));
  if (reader->view.output_metadata) {
    reader->view.output_metadata->sources.count = reader->view.sources.count;
    reader->view.output_metadata->sources.values = reader->view.sources.values;
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_encodings(
      reader, reader->view.sections.encodings));
  IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_build(
      &reader->decoder, reader->context, &reader->view, reader->arena,
      reader->view.sections.types->bytes,
      reader->view.sections.types->absolute_offset));
  if (reader->view.output_metadata) {
    loom_bytecode_module_metadata_t* metadata = reader->view.output_metadata;
    metadata->types.count = reader->view.types.count;
    if (reader->view.types.count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, reader->view.types.count,
          sizeof(*metadata->types.entries), (void**)&metadata->types.entries));
      for (iree_host_size_t i = 0; i < reader->view.types.count; ++i) {
        const uint64_t entry_offset =
            reader->view.types.entries[i].bytecode_offset;
        const uint64_t next_offset =
            i + 1 < reader->view.types.count
                ? reader->view.types.entries[i + 1].bytecode_offset
                : reader->view.sections.types->absolute_offset +
                      reader->view.sections.types->bytes.data_length;
        metadata->types.entries[i] = (loom_bytecode_table_entry_metadata_t){
            .entry_offset = entry_offset,
            .entry_length = next_offset - entry_offset,
        };
      }
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_ops(reader, reader->view.sections.ops));
  if (reader->view.sections.locations) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_locations(
        reader, reader->view.sections.locations));
  }
  if (reader->view.sections.source_trivia) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_file_header(
        reader, reader->view.sections.source_trivia,
        &reader->view.file_header));
  }
  loom_bytecode_symbol_validator_t symbol_validator;
  loom_bytecode_symbol_validator_initialize(
      &reader->decoder, reader->context, reader->arena, reader->metadata_arena,
      &reader->view, &symbol_validator);
  IREE_RETURN_IF_ERROR(loom_bytecode_symbols_validate(
      &symbol_validator, reader->view.sections.symbols,
      reader->view.sections.ir));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_provider_imports(
      reader, reader->view.sections.provider_imports));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_symbol_references(
      reader, reader->view.sections.symbol_references));

  reader->view.summary.string_count = reader->view.strings.count;
  reader->view.summary.source_count = reader->view.sources.count;
  reader->view.summary.type_count = reader->view.types.count;
  reader->view.summary.encoding_count = reader->view.encodings.count;
  reader->view.summary.op_name_count = reader->view.ops.count;
  reader->view.summary.location_count = reader->view.locations.count;
  reader->view.summary.symbol_count = reader->view.symbols.count;
  reader->view.summary.provider_import_count =
      reader->view.provider_imports.count;
  reader->view.summary.provider_import_anchor_count =
      reader->view.provider_imports.anchor_count;
  if (reader->view.output_metadata) {
    reader->view.output_metadata->summary = reader->view.summary;
  }
  return iree_ok_status();
}

static void loom_bytecode_reader_initialize_module(
    const loom_bytecode_file_reader_t* file_reader,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_module_reader_t* out_reader) {
  *out_reader = (loom_bytecode_module_reader_t){
      .bytecode = file_reader->bytecode,
      .context = file_reader->context,
      .arena = file_reader->arena,
      .metadata_arena = retained_arena,
      .decoder = file_reader->decoder,
      .location_mode = file_reader->result.location_mode,
  };
}

// Consumes the private code-only marker paired with a diagnosed user error.
// All other statuses retain their ownership and propagate to the caller.
iree_status_t loom_bytecode_reader_normalize_diagnosed_error(
    iree_status_t status, uint32_t error_count) {
  if (error_count == 0 ||
      status != iree_status_from_code(IREE_STATUS_DEFERRED)) {
    return status;
  }
  iree_status_ignore(status);
  return iree_ok_status();
}

void loom_bytecode_file_reader_initialize(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_allocator_t* arena,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_bytecode_file_reader_t* out_reader) {
  *out_reader = (loom_bytecode_file_reader_t){
      .bytecode = bytecode,
      .context = context,
      .arena = arena,
  };
  loom_bytecode_reader_decoder_initialize(diagnostic_sink, filename,
                                          &out_reader->result.error_count,
                                          &out_reader->decoder);
}

iree_status_t loom_bytecode_file_reader_validate(
    loom_bytecode_file_reader_t* reader) {
  loom_bytecode_reader_cursor_t file_cursor;
  loom_bytecode_reader_cursor_initialize(reader->bytecode.data,
                                         reader->bytecode.data_length, 0,
                                         IREE_SV("FILE"), &file_cursor);
  uint64_t string_pool_length = 0;
  iree_status_t status = loom_bytecode_reader_validate_file_header(
      reader, &file_cursor, &string_pool_length);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_reader_read_module_directory(reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_reader_read_file_string_pool(reader, &file_cursor,
                                                        string_pool_length);
  }
  return status;
}

iree_status_t loom_bytecode_module_validate(
    const loom_bytecode_file_reader_t* file_reader,
    const loom_bytecode_reader_module_t* module,
    loom_bytecode_reader_module_view_t* out_view) {
  loom_bytecode_module_reader_t module_reader = {0};
  loom_bytecode_reader_initialize_module(file_reader, /*retained_arena=*/NULL,
                                         &module_reader);
  iree_status_t status = loom_bytecode_reader_validate_module(
      &module_reader, module, /*output_metadata=*/NULL);
  *out_view = module_reader.view;
  return status;
}

iree_status_t loom_bytecode_module_index(
    const loom_bytecode_file_reader_t* file_reader,
    const loom_bytecode_reader_module_t* module,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_module_metadata_t* out_metadata,
    loom_bytecode_reader_module_view_t* out_view) {
  loom_bytecode_module_reader_t module_reader = {0};
  loom_bytecode_reader_initialize_module(file_reader, retained_arena,
                                         &module_reader);
  iree_status_t status = loom_bytecode_reader_validate_module(
      &module_reader, module, out_metadata);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_reader_project_provider_imports(
        &module_reader.view, out_metadata, retained_arena);
  }
  *out_view = module_reader.view;
  return status;
}
