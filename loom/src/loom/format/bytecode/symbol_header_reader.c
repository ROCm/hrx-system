// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/symbol_header_reader.h"

#include <string.h>

#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/selected_symbol.h"
#include "loom/format/bytecode/reader/selected_tables.h"
#include "loom/ir/module.h"

struct loom_bytecode_symbol_header_reader_t {
  // Host allocator owning this reader and transient selected-table storage.
  iree_allocator_t allocator;
  // Reset-free storage retaining decoded header payloads for the reader life.
  iree_arena_allocator_t arena;
  // Validated source module metadata borrowed from the owning bytecode index.
  const loom_bytecode_module_metadata_t* metadata;
  // Reader-owned source-ordinal metadata projection.
  loom_module_t* module;
  // Number of structured bytecode errors emitted while reading headers.
  uint32_t error_count;
  // Bounded decoder sharing error_count and the caller's diagnostic sink.
  loom_bytecode_reader_decoder_t decoder;
  // Reached-only shared-table projection retained across header reads.
  loom_bytecode_selected_table_materializer_t tables;
  // Function and bodyless-symbol materializer sharing tables.
  loom_bytecode_selected_symbol_materializer_t symbols;
};

static iree_status_t loom_bytecode_symbol_header_reader_resolve_symbol(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_symbol_ref) {
  loom_bytecode_symbol_header_reader_t* reader =
      (loom_bytecode_symbol_header_reader_t*)user_data;
  IREE_ASSERT(source_symbol_ordinal < reader->metadata->symbol_count);
  *out_target_symbol_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = (loom_symbol_id_t)source_symbol_ordinal,
  };
  return iree_ok_status();
}

static iree_status_t loom_bytecode_symbol_header_reader_allocate_module(
    loom_bytecode_symbol_header_reader_t* reader, loom_context_t* context,
    iree_arena_block_pool_t* block_pool) {
  iree_host_size_t string_count = 0;
  if (!iree_host_size_checked_add(reader->metadata->symbol_count, 1,
                                  &string_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode metadata string count overflow");
  }
  const loom_module_size_hints_t hints = {
      .string_count = string_count,
      .type_count = reader->metadata->types.count,
      .encoding_count = reader->metadata->encodings.count,
      .symbol_count = reader->metadata->symbol_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_module_allocate(context, reader->metadata->name, block_pool, &hints,
                           reader->allocator, &reader->module));

  for (iree_host_size_t i = 0; i < reader->metadata->symbol_count; ++i) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        reader->module, reader->metadata->symbols[i].name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(reader->module, name_id, &symbol_id));
    IREE_ASSERT(symbol_id == i);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbol_header_reader_create(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_module_metadata_t* metadata,
    const loom_bytecode_symbol_header_reader_options_t* options,
    iree_allocator_t allocator,
    loom_bytecode_symbol_header_reader_t** out_reader) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(out_reader);
  *out_reader = NULL;

  loom_bytecode_symbol_header_reader_t* reader = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*reader), (void**)&reader));
  reader->allocator = allocator;
  reader->metadata = metadata;
  iree_arena_initialize(block_pool, &reader->arena);

  iree_status_t status = loom_bytecode_symbol_header_reader_allocate_module(
      reader, context, block_pool);
  if (iree_status_is_ok(status)) {
    loom_bytecode_reader_decoder_initialize(
        options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
        filename, &reader->error_count, &reader->decoder);
    loom_bytecode_selected_table_materializer_initialize(
        &reader->decoder, bytecode, context, metadata, &reader->arena,
        reader->module,
        loom_bytecode_selected_symbol_resolver_make(
            loom_bytecode_symbol_header_reader_resolve_symbol, reader),
        allocator, &reader->tables);
    const loom_low_repr_environment_t low_repr_environment =
        options ? options->low_repr_environment
                : (loom_low_repr_environment_t){0};
    loom_bytecode_selected_symbol_materializer_initialize(
        &reader->decoder, block_pool, &reader->tables, &low_repr_environment,
        &reader->symbols);
  }
  if (iree_status_is_ok(status)) {
    *out_reader = reader;
  } else {
    loom_module_free(reader->module);
    iree_arena_deinitialize(&reader->arena);
    iree_allocator_free(allocator, reader);
  }
  return status;
}

void loom_bytecode_symbol_header_reader_free(
    loom_bytecode_symbol_header_reader_t* reader) {
  if (reader == NULL) {
    return;
  }
  iree_allocator_t allocator = reader->allocator;
  loom_bytecode_selected_table_materializer_deinitialize(&reader->tables);
  loom_module_free(reader->module);
  iree_arena_deinitialize(&reader->arena);
  iree_allocator_free(allocator, reader);
}

loom_module_t* loom_bytecode_symbol_header_reader_module(
    const loom_bytecode_symbol_header_reader_t* reader) {
  IREE_ASSERT_ARGUMENT(reader);
  return reader->module;
}

iree_status_t loom_bytecode_symbol_header_reader_read_function(
    loom_bytecode_symbol_header_reader_t* reader,
    uint32_t source_symbol_ordinal,
    loom_bytecode_function_header_t* out_header) {
  IREE_ASSERT_ARGUMENT(reader);
  IREE_ASSERT_ARGUMENT(out_header);
  if (source_symbol_ordinal >= reader->metadata->symbol_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source symbol ordinal %u is out of range [0, "
                            "%" PRIhsz ")",
                            source_symbol_ordinal,
                            reader->metadata->symbol_count);
  }
  return loom_bytecode_selected_function_header_materialize(
      &reader->symbols, source_symbol_ordinal, out_header);
}

iree_status_t loom_bytecode_symbol_header_reader_materialize_bodyless_symbol(
    loom_bytecode_symbol_header_reader_t* reader,
    uint32_t source_symbol_ordinal) {
  IREE_ASSERT_ARGUMENT(reader);
  if (source_symbol_ordinal >= reader->metadata->symbol_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source symbol ordinal %u is out of range [0, "
                            "%" PRIhsz ")",
                            source_symbol_ordinal,
                            reader->metadata->symbol_count);
  }
  const loom_bytecode_symbol_metadata_t* metadata =
      &reader->metadata->symbols[source_symbol_ordinal];
  if (metadata->has_body) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source symbol '@%.*s' has a body unavailable to the metadata reader",
        (int)metadata->name.size, metadata->name.data);
  }
  if (reader->module->symbols.entries[source_symbol_ordinal].defining_op !=
      NULL) {
    return iree_ok_status();
  }
  const loom_bytecode_selected_symbol_t selected = {
      .source_ordinal = source_symbol_ordinal,
  };
  return loom_bytecode_selected_symbols_materialize(&reader->symbols, &selected,
                                                    1);
}
