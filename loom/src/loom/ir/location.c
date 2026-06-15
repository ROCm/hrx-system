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
