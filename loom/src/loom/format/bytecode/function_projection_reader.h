// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Selective function-header and root-region projection from bytecode.

#ifndef LOOM_FORMAT_BYTECODE_FUNCTION_PROJECTION_READER_H_
#define LOOM_FORMAT_BYTECODE_FUNCTION_PROJECTION_READER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/function_header.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/low_repr.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_function_projection_reader_t
    loom_bytecode_function_projection_reader_t;

// Resolves one reached source symbol into the caller-owned output module.
typedef iree_status_t (*loom_bytecode_function_projection_symbol_resolver_fn_t)(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_symbol_ref);

// Options controlling selected function projection.
typedef struct loom_bytecode_function_projection_reader_options_t {
  // Sink for structured malformed-bytecode diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Stable-key codec used when a projected root contains Low IR.
  loom_low_repr_environment_t low_repr_environment;
  // Optional resolver for source symbols reached by the selected projection.
  loom_bytecode_function_projection_symbol_resolver_fn_t symbol_resolver;
  // Caller-owned payload passed to symbol_resolver for the reader lifetime.
  void* symbol_resolver_user_data;
} loom_bytecode_function_projection_reader_options_t;

// Allocates a reader that projects selected source identities into
// |output_module|.
//
// The reader borrows every input and owns only transient projections. Source
// symbols must be bound explicitly before a decoded header or root region
// references them. The caller owns |output_module| and may continue building
// it after the reader is released.
iree_status_t loom_bytecode_function_projection_reader_allocate(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    iree_arena_block_pool_t* block_pool,
    const loom_bytecode_module_metadata_t* metadata,
    loom_module_t* output_module,
    const loom_bytecode_function_projection_reader_options_t* options,
    iree_allocator_t allocator,
    loom_bytecode_function_projection_reader_t** out_reader);

// Frees |reader| without modifying its caller-owned output module.
void loom_bytecode_function_projection_reader_free(
    loom_bytecode_function_projection_reader_t* reader);

// Binds one module-local source symbol to an existing output-module symbol.
iree_status_t loom_bytecode_function_projection_reader_bind_symbol(
    loom_bytecode_function_projection_reader_t* reader,
    uint32_t source_symbol_ordinal, loom_symbol_ref_t target_symbol_ref);

// Decodes one function-like symbol header without reading root-region bytes.
//
// Arrays and projected IR identities in |out_header| remain valid until the
// reader is released. Repeated reads of the same header are intentionally not
// cached; callers retain the returned header through their projection.
iree_status_t loom_bytecode_function_projection_reader_read_header(
    loom_bytecode_function_projection_reader_t* reader,
    uint32_t source_symbol_ordinal,
    loom_bytecode_function_header_t* out_header);

// Materializes one independently bounded source root region into a target op.
//
// |source_region_payload_ordinal| indexes |header|'s payload array.
// |predefined_values| bind the target entry block arguments in source order.
// No other root-region payload is read.
iree_status_t loom_bytecode_function_projection_reader_materialize_region(
    loom_bytecode_function_projection_reader_t* reader,
    const loom_bytecode_function_header_t* header,
    uint8_t source_region_payload_ordinal, loom_builder_t* builder,
    loom_op_t* target_parent_op, uint8_t target_region_index,
    const loom_value_id_t* predefined_values, uint16_t predefined_value_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_FUNCTION_PROJECTION_READER_H_
