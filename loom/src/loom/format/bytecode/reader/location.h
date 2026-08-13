// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Location-table validation, retained indexing, and IR materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_LOCATION_H_
#define LOOM_FORMAT_BYTECODE_READER_LOCATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates one complete LOCATIONS section without retaining entry ranges.
iree_status_t loom_bytecode_location_table_validate(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_module_view_t* module_view,
    const loom_bytecode_reader_section_t* section);

// Validates one complete LOCATIONS section and retains the exact byte range of
// every entry.
iree_status_t loom_bytecode_location_table_index(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_module_view_t* module_view,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_table_entry_metadata_t** out_entries,
    iree_host_size_t* out_count);

// State required to materialize a validated location table into a module.
typedef struct loom_bytecode_location_materializer_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Immutable validated module facts referenced by location entries.
  const loom_bytecode_reader_module_view_t* module_view;
  // Module receiving canonical location-table entries.
  loom_module_t* output_module;
} loom_bytecode_location_materializer_t;

// Materializes every entry in one validated LOCATIONS section.
iree_status_t loom_bytecode_location_table_materialize(
    loom_bytecode_location_materializer_t* materializer,
    const loom_bytecode_reader_section_t* section);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_LOCATION_H_
