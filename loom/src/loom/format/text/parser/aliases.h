// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_ALIASES_H_
#define LOOM_FORMAT_TEXT_PARSER_ALIASES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_alias_entry_t {
  // Bare alias name without '#', e.g. "q6_k".
  iree_string_view_t name;
  // 1-based index into the module encoding table.
  uint16_t encoding_id;
} loom_alias_entry_t;

typedef struct loom_alias_table_t {
  loom_alias_entry_t* entries;
  iree_host_size_t capacity;
  iree_host_size_t count;
} loom_alias_table_t;

// Registers an encoding alias. Grows the table via the arena.
iree_status_t loom_alias_table_add(loom_alias_table_t* table,
                                   iree_arena_allocator_t* arena,
                                   iree_string_view_t name,
                                   uint16_t encoding_id);

// Returns the encoding_id for |name|, or 0 if not found.
uint16_t loom_alias_table_lookup(const loom_alias_table_t* table,
                                 iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_ALIASES_H_
