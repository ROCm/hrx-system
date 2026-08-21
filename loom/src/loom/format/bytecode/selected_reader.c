// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/selected_reader.h"

#include "loom/format/bytecode/reader/selected_materializer.h"
#include "loom/format/bytecode/reader/validation.h"
#include "loom/verify/verify.h"

iree_status_t loom_bytecode_materialize_module_symbols(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_file_metadata_t* metadata, uint16_t module_ordinal,
    loom_bytecode_symbol_ordinal_list_t selection,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_result = (loom_bytecode_read_result_t){0};
  *out_module = NULL;
  if (module_ordinal >= metadata->module_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "module ordinal %u is out of range [0, %" PRIhsz
                            ")",
                            module_ordinal, metadata->module_count);
  }
  if (selection.count > 0 && selection.ordinals == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected symbol count is non-zero but ordinals is NULL");
  }
  const loom_bytecode_module_metadata_t* module_metadata =
      &metadata->modules[module_ordinal];
  for (iree_host_size_t i = 0; i < selection.count; ++i) {
    if (selection.ordinals[i] >= module_metadata->symbol_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "selected symbol ordinal %" PRIhsz
                              " is out of range [0, %" PRIhsz ")",
                              selection.ordinals[i],
                              module_metadata->symbol_count);
    }
    if (i > 0 && selection.ordinals[i - 1] >= selection.ordinals[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected symbol ordinals must be strictly increasing");
    }
  }

  out_result->module_count = (uint16_t)metadata->module_count;
  out_result->location_mode = metadata->location_mode;
  if (module_ordinal == 0) {
    out_result->first_module = module_metadata->summary;
  }
  loom_bytecode_reader_decoder_t decoder;
  loom_bytecode_reader_decoder_initialize(
      options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      filename, &out_result->error_count, &decoder);
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  const loom_bytecode_selected_module_materializer_t materializer = {
      .decoder = &decoder,
      .bytecode = bytecode,
      .context = context,
      .scratch_arena = &arena,
      .block_pool = block_pool,
      .metadata = module_metadata,
      .low_repr_environment = options ? options->low_repr_environment
                                      : (loom_low_repr_environment_t){0},
      .host_allocator = host_allocator,
  };
  loom_module_t* output_module = NULL;
  iree_status_t status = loom_bytecode_selected_module_materialize(
      &materializer, selection.ordinals, selection.count, &output_module);
  status = loom_bytecode_reader_normalize_diagnosed_error(
      status, out_result->error_count);
  if (iree_status_is_ok(status) && out_result->error_count == 0 && options &&
      options->verify_module) {
    loom_verify_result_t verify_result = {0};
    const loom_verify_options_t verify_options = {
        .sink = options->diagnostic_sink,
        .max_errors = options->verify_max_errors,
    };
    status = loom_verify_module(output_module, &verify_options, &verify_result);
    out_result->error_count += verify_result.error_count;
    out_result->warning_count += verify_result.warning_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    *out_module = output_module;
    output_module = NULL;
  }
  loom_module_free(output_module);
  iree_arena_deinitialize(&arena);
  return status;
}
