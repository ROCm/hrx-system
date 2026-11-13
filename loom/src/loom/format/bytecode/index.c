// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/index.h"

#include "loom/format/bytecode/reader/validation.h"

iree_status_t loom_bytecode_read_metadata(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_index_options_t* options,
    loom_bytecode_read_result_t* out_result) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_file_reader_t reader;
  loom_bytecode_file_reader_initialize(
      bytecode, filename, context, &arena,
      options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      &reader);

  bool accepted = loom_bytecode_file_reader_validate(&reader);
  if (accepted) {
    for (iree_host_size_t i = 0; i < reader.module_count; ++i) {
      loom_bytecode_reader_module_view_t module_view;
      accepted = loom_bytecode_module_validate(
          &reader, &reader.modules[i], LOOM_BYTECODE_MODULE_VALIDATION_NONE,
          &module_view);
      if (i == 0) {
        reader.result.first_module = module_view.summary;
      }
      if (!accepted) {
        break;
      }
    }
  }

  iree_status_t status = loom_bytecode_reader_consume_status(&reader.decoder);
  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_bytecode_read_index(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* metadata_arena,
    const loom_bytecode_index_options_t* options,
    loom_bytecode_read_result_t* out_result,
    loom_bytecode_file_metadata_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(metadata_arena);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_ASSERT_ARGUMENT(out_metadata);
  *out_metadata = (loom_bytecode_file_metadata_t){0};

  loom_bytecode_file_metadata_t metadata = {0};
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_file_reader_t reader;
  loom_bytecode_file_reader_initialize(
      bytecode, filename, context, &arena,
      options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      &reader);

  bool accepted = loom_bytecode_file_reader_validate(&reader);
  if (accepted) {
    accepted = loom_bytecode_reader_check_status(
        &reader.decoder, loom_bytecode_file_reader_project_index(
                             &reader, metadata_arena, &metadata));
  }
  if (accepted) {
    for (iree_host_size_t i = 0; i < reader.module_count; ++i) {
      loom_bytecode_reader_module_view_t module_view;
      accepted = loom_bytecode_module_index(&reader, &reader.modules[i],
                                            metadata_arena,
                                            &metadata.modules[i], &module_view);
      if (i == 0) {
        reader.result.first_module = module_view.summary;
      }
      if (!accepted) {
        break;
      }
    }
  }

  if (accepted) {
    *out_metadata = metadata;
  }
  iree_status_t status = loom_bytecode_reader_consume_status(&reader.decoder);
  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
  }
  iree_arena_deinitialize(&arena);
  return status;
}
