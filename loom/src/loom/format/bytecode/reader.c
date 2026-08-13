// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader.h"

#include "loom/format/bytecode/reader/materializer.h"
#include "loom/format/bytecode/reader/validation.h"
#include "loom/verify/verify.h"

static iree_status_t loom_bytecode_read_module_impl(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    uint16_t module_ordinal, bool require_single_module,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  *out_module = NULL;

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_file_reader_t reader;
  loom_bytecode_file_reader_initialize(
      bytecode, filename, context, &arena,
      options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      &reader);

  iree_status_t status = loom_bytecode_file_reader_validate(&reader);
  if (iree_status_is_ok(status) && require_single_module &&
      reader.module_count != 1) {
    status = loom_bytecode_reader_emit_invalid_field(
        &reader.decoder, IREE_SV("FILE"), IREE_SV("header"), 0,
        IREE_SV("module_count"), 0,
        IREE_SV("module_materialization_requires_exactly_one_module"));
  }
  if (iree_status_is_ok(status) && module_ordinal >= reader.module_count) {
    status = loom_bytecode_reader_emit_invalid_field(
        &reader.decoder, IREE_SV("FILE"), IREE_SV("module_directory"),
        module_ordinal, IREE_SV("module_ordinal"), 0,
        IREE_SV("requested_module_ordinal_is_out_of_range"));
  }

  loom_bytecode_reader_module_view_t module_view = {0};
  loom_module_t* output_module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_module_validate(
        &reader, &reader.modules[module_ordinal], &module_view);
    if (module_ordinal == 0) {
      reader.result.first_module = module_view.summary;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_bytecode_module_materializer_t materializer = {
        .decoder = &reader.decoder,
        .bytecode = bytecode,
        .context = context,
        .scratch_arena = &arena,
        .block_pool = block_pool,
        .module_view = &module_view,
        .low_repr_environment = options ? options->low_repr_environment
                                        : (loom_low_repr_environment_t){0},
        .host_allocator = host_allocator,
    };
    status = loom_bytecode_module_materialize(&materializer, &output_module);
  }
  status = loom_bytecode_reader_normalize_diagnosed_error(
      status, reader.result.error_count);
  if (iree_status_is_ok(status) && reader.result.error_count == 0 && options &&
      options->verify_module) {
    loom_verify_result_t verify_result = {0};
    loom_verify_options_t verify_options = {
        .sink = options->diagnostic_sink,
        .max_errors = options->verify_max_errors,
    };
    status = loom_verify_module(output_module, &verify_options, &verify_result);
    reader.result.error_count += verify_result.error_count;
    reader.result.warning_count += verify_result.warning_count;
  }

  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
    if (reader.result.error_count == 0) {
      *out_module = output_module;
      output_module = NULL;
    }
  }
  if (output_module) {
    loom_module_free(output_module);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_bytecode_read_module(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  return loom_bytecode_read_module_impl(bytecode, filename, context, block_pool,
                                        /*module_ordinal=*/0,
                                        /*require_single_module=*/true, options,
                                        out_result, out_module, host_allocator);
}

iree_status_t loom_bytecode_read_module_ordinal(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    uint16_t module_ordinal, const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  return loom_bytecode_read_module_impl(
      bytecode, filename, context, block_pool, module_ordinal,
      /*require_single_module=*/false, options, out_result, out_module,
      host_allocator);
}
