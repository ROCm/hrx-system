// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/location.h"

#include "loom/ir/location_tag_table.inc"

loom_location_kind_t loom_location_get_kind(loom_location_entry_t entry) {
  return entry.kind;
}

iree_host_size_t loom_location_field_span_data_length(
    loom_location_field_span_format_t format, iree_host_size_t count) {
  switch (format) {
    case LOOM_LOCATION_FIELD_SPAN_FORMAT_NONE:
      return 0;
    case LOOM_LOCATION_FIELD_SPAN_FORMAT_COMPACT:
      return count * sizeof(loom_location_compact_field_span_t);
    case LOOM_LOCATION_FIELD_SPAN_FORMAT_WIDE:
      return count * sizeof(loom_location_field_span_t);
    default:
      IREE_ASSERT_UNREACHABLE("invalid location field span format");
      return 0;
  }
}

loom_location_field_span_t loom_location_file_field_span_at(
    const loom_location_entry_t* entry, uint16_t index) {
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT(entry->kind == LOOM_LOCATION_FILE);
  IREE_ASSERT(index < entry->file.field_span_count);
  IREE_ASSERT(entry->file.field_span_data);
  switch (entry->file.field_span_format) {
    case LOOM_LOCATION_FIELD_SPAN_FORMAT_COMPACT: {
      const loom_location_compact_field_span_t* spans =
          (const loom_location_compact_field_span_t*)
              entry->file.field_span_data;
      return (loom_location_field_span_t){
          .kind = spans[index].kind,
          .index = spans[index].index,
          .start_line = spans[index].start_line,
          .start_col = spans[index].start_col,
          .end_line = spans[index].end_line,
          .end_col = spans[index].end_col,
      };
    }
    case LOOM_LOCATION_FIELD_SPAN_FORMAT_WIDE:
      return ((const loom_location_field_span_t*)
                  entry->file.field_span_data)[index];
    default:
      IREE_ASSERT_UNREACHABLE("file location has invalid field span format");
      return (loom_location_field_span_t){0};
  }
}

loom_location_entry_t loom_location_file_range(loom_source_id_t source_id,
                                               uint32_t start_line,
                                               uint32_t start_col,
                                               uint32_t end_line,
                                               uint32_t end_col) {
  loom_location_entry_t entry = {.kind = LOOM_LOCATION_FILE};
  entry.file.source_id = source_id;
  entry.file.field_span_count = 0;
  entry.file.field_span_format = LOOM_LOCATION_FIELD_SPAN_FORMAT_NONE;
  entry.file.start_line = start_line;
  entry.file.start_col = start_col;
  entry.file.end_line = end_line;
  entry.file.end_col = end_col;
  entry.file.field_span_data = NULL;
  return entry;
}

loom_location_entry_t loom_location_tagged(loom_location_tag_t tag,
                                           loom_location_id_t child,
                                           const uint8_t* data,
                                           uint32_t data_length) {
  loom_location_entry_t entry = {.kind = LOOM_LOCATION_TAGGED};
  entry.tagged.tag = tag;
  entry.tagged.data_length = data_length;
  entry.tagged.child = child;
  entry.tagged.data = data;
  return entry;
}

iree_string_view_t loom_location_tag_name(loom_location_tag_t tag) {
  return loom_location_tag_builtin_name(tag);
}

bool loom_location_tag_parse(iree_string_view_t name,
                             loom_location_tag_t* out_tag) {
  *out_tag = loom_location_tag_classify_name(name);
  return *out_tag != LOOM_LOCATION_TAG_INVALID;
}
