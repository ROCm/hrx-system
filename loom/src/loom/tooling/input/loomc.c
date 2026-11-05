// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/input/loomc.h"

#include "loom/tooling/io/file.h"
#include "loom/tooling/io/source_path.h"
#include "loomc/iree.h"

iree_status_t loom_tooling_input_admit_loomc_module(
    const loom_tooling_loomc_input_options_t* options, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  *out_module = NULL;
  *out_result = NULL;
  const iree_string_view_t physical_identifier =
      loom_tooling_file_path_is_stdio(options->path) ? IREE_SV("<stdin>")
                                                     : options->path;
  const iree_string_view_t format = options->input.format;
  iree_string_view_t provider_options = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_input_options_for_provider(
      options->input.provider_options, format, &provider_options));

  loomc_source_format_t source_format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (iree_string_view_equal(format, loom_input_text_format.name)) {
    source_format = LOOMC_SOURCE_FORMAT_TEXT;
  } else if (iree_string_view_equal(format, loom_input_bytecode_format.name)) {
    source_format = LOOMC_SOURCE_FORMAT_BYTECODE;
  }
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = source_format,
      .identifier = loomc_string_view_from_iree(physical_identifier),
      .contents = loomc_byte_span_from_iree(iree_make_const_byte_span(
          options->source.data, options->source.size)),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_source_create(
      &source_options, loomc_allocator_from_iree(host_allocator), &source)));

  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(format, loom_input_text_format.name) ||
      iree_string_view_equal(format, loom_input_bytecode_format.name)) {
    if (!iree_string_view_is_empty(provider_options)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "%.*s input does not accept input options",
                                (int)format.size, format.data);
    }
    iree_string_view_t logical_identifier = physical_identifier;
    char* logical_identifier_storage = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_tooling_source_path_remap(
          physical_identifier, &options->input.source_path_options,
          host_allocator, &logical_identifier, &logical_identifier_storage);
    }
    const loomc_module_deserialize_options_t deserialize_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS,
        .structure_size = sizeof(deserialize_options),
        .format = source_format,
        .identifier = loomc_string_view_from_iree(logical_identifier),
    };
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_module_deserialize_from_source(
          context, workspace, source, &deserialize_options,
          loomc_allocator_from_iree(host_allocator), out_module, out_result));
    }
    iree_allocator_free(host_allocator, logical_identifier_storage);
  } else if (options->import != NULL) {
    status = options->import(
        options->import_user_data, format, provider_options,
        &options->input.source_path_options, context, workspace, source,
        block_pool, host_allocator, out_module, out_result);
  } else {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "input format '%.*s' has no LoomC importer linked into this runner",
        (int)format.size, format.data);
  }
  loomc_source_release(source);
  return status;
}
