// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/string_table.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/error/error_catalog.h"

#define LOOM_BYTECODE_MAX_STRING_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_STRING_LENGTH (UINT64_C(1) << 24)

static uint64_t loom_bytecode_string_hash(iree_string_view_t value) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    hash ^= (uint8_t)value.data[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static uint64_t loom_bytecode_string_absolute_offset(
    const loom_bytecode_reader_section_t* section, iree_string_view_t value) {
  return section->absolute_offset +
         (uint64_t)((const uint8_t*)value.data - section->bytes.data);
}

static iree_status_t loom_bytecode_string_table_read_values(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_string_view_t table_name, uint64_t count_limit,
    iree_arena_allocator_t* storage_arena, iree_string_view_t** out_values,
    iree_host_size_t* out_count) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      table_name, &cursor);
  const uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  uint64_t count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &count));
  if (count > count_limit || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(decoder, table_name, count,
                                                   count_limit, count_offset);
  }

  iree_string_view_t* values = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(storage_arena, (iree_host_size_t)count,
                                  sizeof(*values), (void**)&values));
  }
  for (uint64_t i = 0; i < count; ++i) {
    const uint64_t string_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, &cursor, &length));
    if (length > LOOM_BYTECODE_MAX_STRING_LENGTH) {
      return loom_bytecode_reader_emit_count_exceeds(
          decoder, IREE_SV("string_length"), length,
          LOOM_BYTECODE_MAX_STRING_LENGTH, string_offset);
    }
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_span(decoder, &cursor, length, &bytes));
    const iree_string_view_t value =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!iree_unicode_utf8_validate(value)) {
      return loom_bytecode_reader_emit_invalid_field(
          decoder, table_name, IREE_SV("string"), i, IREE_SV("utf8_data"),
          string_offset, IREE_SV("string_payload_is_not_valid_utf_8"));
    }
    values[i] = value;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_expect_empty(decoder, &cursor, table_name));
  *out_values = values;
  *out_count = (iree_host_size_t)count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_string_table_validate_unique(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_string_view_t table_name, iree_string_view_t duplicate_reason,
    const iree_string_view_t* values, iree_host_size_t count,
    iree_arena_allocator_t* scratch_arena) {
  if (count < 2) {
    return iree_ok_status();
  }

  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_mul(count, 2, &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "string uniqueness table capacity overflow");
  }
  const iree_host_size_t capacity =
      iree_host_size_next_power_of_two(minimum_capacity);
  if (capacity < minimum_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "string uniqueness table capacity overflow");
  }

  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  uint32_t* ordinals = NULL;
  iree_status_t status = iree_arena_allocate_array(
      scratch_arena, capacity, sizeof(*ordinals), (void**)&ordinals);
  if (iree_status_is_ok(status)) {
    memset(ordinals, 0xFF, capacity * sizeof(*ordinals));
  }
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_string_view_t value = values[i];
    const iree_host_size_t mask = capacity - 1;
    iree_host_size_t slot =
        (iree_host_size_t)loom_bytecode_string_hash(value) & mask;
    while (ordinals[slot] != UINT32_MAX &&
           !iree_string_view_equal(values[ordinals[slot]], value)) {
      slot = (slot + 1) & mask;
    }
    if (ordinals[slot] != UINT32_MAX) {
      status = loom_bytecode_reader_emit_invalid_field(
          decoder, table_name, IREE_SV("string"), i, IREE_SV("string"),
          loom_bytecode_string_absolute_offset(section, value),
          duplicate_reason);
    } else {
      ordinals[slot] = (uint32_t)i;
    }
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

static iree_status_t loom_bytecode_string_table_validate_canonical(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_module_view_t* module_view) {
  if (module_view->strings.count == 0 ||
      !iree_string_view_is_empty(module_view->strings.values[0])) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("STRINGS"), IREE_SV("string"), 0, IREE_SV("string"),
        section->absolute_offset, IREE_SV("string_zero_must_be_empty"));
  }
  return loom_bytecode_string_table_validate_unique(
      decoder, section, IREE_SV("STRINGS"),
      IREE_SV("string_table_must_be_deduplicated"), module_view->strings.values,
      module_view->strings.count, scratch_arena);
}

iree_status_t loom_bytecode_string_table_read(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* storage_arena,
    loom_bytecode_reader_module_view_t* module_view) {
  IREE_RETURN_IF_ERROR(loom_bytecode_string_table_read_values(
      decoder, section, IREE_SV("STRINGS"), LOOM_BYTECODE_MAX_STRING_COUNT,
      storage_arena, &module_view->strings.values,
      &module_view->strings.count));
  return loom_bytecode_string_table_validate_canonical(
      decoder, section, scratch_arena, module_view);
}

iree_status_t loom_bytecode_source_table_read(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* storage_arena,
    loom_bytecode_reader_module_view_t* module_view) {
  IREE_RETURN_IF_ERROR(loom_bytecode_string_table_read_values(
      decoder, section, IREE_SV("SOURCES"), LOOM_SOURCE_ID_INVALID,
      storage_arena, &module_view->sources.values,
      &module_view->sources.count));
  return loom_bytecode_string_table_validate_unique(
      decoder, section, IREE_SV("SOURCES"),
      IREE_SV("source_table_must_be_deduplicated"), module_view->sources.values,
      module_view->sources.count, scratch_arena);
}

iree_status_t loom_bytecode_string_table_materialize(
    const loom_bytecode_reader_module_view_t* module_view,
    loom_module_t* output_module) {
  for (iree_host_size_t i = 0; i < module_view->strings.count; ++i) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        output_module, module_view->strings.values[i], &string_id));
    IREE_ASSERT(string_id == i);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_source_table_materialize(
    const loom_bytecode_reader_module_view_t* module_view,
    loom_module_t* output_module) {
  for (iree_host_size_t i = 0; i < module_view->sources.count; ++i) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_append_source(
        output_module, module_view->sources.values[i], &source_id));
    IREE_ASSERT(source_id == i);
  }
  return iree_ok_status();
}
