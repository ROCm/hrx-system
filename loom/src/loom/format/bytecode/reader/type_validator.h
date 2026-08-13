// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Type-table validation and immutable topological plan construction.

#ifndef LOOM_FORMAT_BYTECODE_READER_TYPE_VALIDATOR_H_
#define LOOM_FORMAT_BYTECODE_READER_TYPE_VALIDATOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_reader_module_view_t
    loom_bytecode_reader_module_view_t;

// Validates one TYPES section and builds its immutable topological plan.
iree_status_t loom_bytecode_type_plan_build(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, iree_const_byte_span_t section_bytes,
    uint64_t section_absolute_offset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_VALIDATOR_H_
