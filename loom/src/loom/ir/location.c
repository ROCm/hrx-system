// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/location.h"

loom_location_kind_t loom_location_get_kind(loom_location_entry_t entry) {
  return (loom_location_kind_t)entry.kind;
}

loom_location_entry_t loom_location_file_range(loom_source_id_t source_id,
                                               uint16_t start_line,
                                               uint16_t start_col,
                                               uint16_t end_line,
                                               uint16_t end_col) {
  loom_location_entry_t entry = {.kind = LOOM_LOCATION_FILE};
  entry.file.source_id = source_id;
  entry.file.field_span_count = 0;
  entry.file.start_line = start_line;
  entry.file.start_col = start_col;
  entry.file.end_line = end_line;
  entry.file.end_col = end_col;
  entry.file.field_spans = NULL;
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
  switch (tag) {
    case LOOM_LOCATION_TAG_SANITIZER_SITE:
      return IREE_SV("sanitizer_site");
    case LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION:
      return IREE_SV("template_instantiation");
    case LOOM_LOCATION_TAG_TILE_LOWERING:
      return IREE_SV("tile_lowering");
    case LOOM_LOCATION_TAG_UKERNEL_SELECTION:
      return IREE_SV("ukernel_selection");
    default:
      return iree_string_view_empty();
  }
}

bool loom_location_tag_parse(iree_string_view_t name,
                             loom_location_tag_t* out_tag) {
  *out_tag = LOOM_LOCATION_TAG_INVALID;
  if (iree_string_view_equal(name, IREE_SV("sanitizer_site"))) {
    *out_tag = LOOM_LOCATION_TAG_SANITIZER_SITE;
    return true;
  }
  if (iree_string_view_equal(name, IREE_SV("template_instantiation"))) {
    *out_tag = LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION;
    return true;
  }
  if (iree_string_view_equal(name, IREE_SV("tile_lowering"))) {
    *out_tag = LOOM_LOCATION_TAG_TILE_LOWERING;
    return true;
  }
  if (iree_string_view_equal(name, IREE_SV("ukernel_selection"))) {
    *out_tag = LOOM_LOCATION_TAG_UKERNEL_SELECTION;
    return true;
  }
  return false;
}
