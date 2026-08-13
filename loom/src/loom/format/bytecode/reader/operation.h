// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Registered operation-table validation and retained indexing.

#ifndef LOOM_FORMAT_BYTECODE_READER_OPERATION_H_
#define LOOM_FORMAT_BYTECODE_READER_OPERATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates one complete OPS section and resolves its dense operation table.
iree_status_t loom_bytecode_operation_table_validate(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_section_t* section);

// Validates one complete OPS section, resolves its dense operation table, and
// retains the registered operation names in wire order.
iree_status_t loom_bytecode_operation_table_index(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_op_metadata_t** out_entries, iree_host_size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_OPERATION_H_
