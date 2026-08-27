// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// String/source table validation and IR projection.

#ifndef LOOM_FORMAT_BYTECODE_READER_STRING_TABLE_H_
#define LOOM_FORMAT_BYTECODE_READER_STRING_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates the canonical STRINGS section and retains borrowed UTF-8 views in
// |storage_arena|. Temporary uniqueness-validation storage is reclaimed from
// |scratch_arena| before returning.
iree_status_t loom_bytecode_string_table_read(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* storage_arena,
    loom_bytecode_reader_module_view_t* module_view);

// Validates the SOURCES section and retains borrowed UTF-8 views in
// |storage_arena|. The source count is bounded by the runtime source-ID space
// and source spellings are unique in wire order.
iree_status_t loom_bytecode_source_table_read(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* storage_arena,
    loom_bytecode_reader_module_view_t* module_view);

// Interns every validated string into an otherwise-empty output module while
// preserving bytecode string IDs exactly.
iree_status_t loom_bytecode_string_table_materialize(
    const loom_bytecode_reader_module_view_t* module_view,
    loom_module_t* output_module);

// Appends every canonical validated source to an otherwise-empty output module
// while preserving bytecode source IDs exactly.
iree_status_t loom_bytecode_source_table_materialize(
    const loom_bytecode_reader_module_view_t* module_view,
    loom_module_t* output_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_STRING_TABLE_H_
