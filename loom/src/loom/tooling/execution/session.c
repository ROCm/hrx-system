// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/session.h"

#include <string.h>

#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"

enum {
  LOOM_RUN_DEFAULT_BLOCK_POOL_BLOCK_SIZE = 32 * 1024,
  LOOM_RUN_DEFAULT_MAX_PARSE_ERRORS = 20,
};

void loom_run_session_options_initialize(
    loom_run_session_options_t* out_options) {
  *out_options = (loom_run_session_options_t){
      .host_allocator = iree_allocator_system(),
      .block_pool_block_size = LOOM_RUN_DEFAULT_BLOCK_POOL_BLOCK_SIZE,
  };
}

iree_status_t loom_run_session_initialize(
    const loom_run_session_options_t* options,
    loom_run_session_t* out_session) {
  *out_session = (loom_run_session_t){
      .host_allocator = options->host_allocator,
  };

  const iree_host_size_t block_pool_block_size =
      options->block_pool_block_size == 0
          ? LOOM_RUN_DEFAULT_BLOCK_POOL_BLOCK_SIZE
          : options->block_pool_block_size;
  iree_arena_block_pool_initialize(
      block_pool_block_size, options->host_allocator, &out_session->block_pool);
  out_session->block_pool_initialized = true;

  iree_status_t status = options->initialize_low_descriptor_registry.fn(
      options->initialize_low_descriptor_registry.user_data,
      &out_session->low_descriptor_registry);
  if (iree_status_is_ok(status)) {
    loom_context_initialize(options->host_allocator, &out_session->context);
    out_session->context_initialized = true;
    status = options->register_context.fn(options->register_context.user_data,
                                          &out_session->context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&out_session->context);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_session_deinitialize(out_session);
  }
  return status;
}

void loom_run_session_deinitialize(loom_run_session_t* session) {
  if (session == NULL) {
    return;
  }
  if (session->context_initialized) {
    loom_context_deinitialize(&session->context);
  }
  if (session->block_pool_initialized) {
    iree_arena_block_pool_deinitialize(&session->block_pool);
  }
  *session = (loom_run_session_t){0};
}

loom_context_t* loom_run_session_context(loom_run_session_t* session) {
  return &session->context;
}

iree_arena_block_pool_t* loom_run_session_block_pool(
    loom_run_session_t* session) {
  return &session->block_pool;
}

const loom_target_low_descriptor_registry_t*
loom_run_session_low_descriptor_registry(const loom_run_session_t* session) {
  return &session->low_descriptor_registry;
}

void loom_run_module_parse_options_initialize(
    loom_run_module_parse_options_t* out_options) {
  *out_options = (loom_run_module_parse_options_t){
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = LOOM_RUN_DEFAULT_MAX_PARSE_ERRORS,
  };
}

static bool loom_run_module_input_is_bytecode(iree_string_view_t source) {
  return source.size >= LOOM_BYTECODE_MAGIC_LENGTH &&
         memcmp(source.data, LOOM_BYTECODE_MAGIC, LOOM_BYTECODE_MAGIC_LENGTH) ==
             0;
}

static iree_status_t loom_run_module_parse_text(
    loom_run_session_t* session, const loom_run_module_parse_options_t* options,
    loom_run_module_t* out_module) {
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = options->diagnostic_sink,
      .max_errors = options->max_errors,
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &session->low_descriptor_registry.registry,
      &parse_options.low_asm_environment);

  iree_status_t status = loom_text_parse(
      options->source, options->filename, &session->context,
      &session->block_pool, &parse_options, &out_module->module);
  if (iree_status_is_ok(status) && out_module->module == NULL) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input module has parse errors");
  }
  if (iree_status_is_ok(status)) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    status = loom_module_register_source(out_module->module, options->filename,
                                         &source_id);
    if (iree_status_is_ok(status)) {
      out_module->source_entry = (loom_source_entry_t){
          .source_id = source_id,
          .source = options->source,
          .filename = options->filename,
      };
      out_module->source_table_resolver = (loom_source_table_resolver_t){
          .entries = &out_module->source_entry,
          .count = 1,
      };
      out_module->has_source_entry = true;
    }
  }
  return status;
}

static iree_status_t loom_run_module_read_bytecode(
    loom_run_session_t* session, const loom_run_module_parse_options_t* options,
    loom_run_module_t* out_module) {
  const iree_const_byte_span_t bytecode = iree_make_const_byte_span(
      options->source.data, (iree_host_size_t)options->source.size);
  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink = options->diagnostic_sink,
      .verify_module = false,
      .verify_max_errors = options->max_errors,
  };
  loom_bytecode_read_result_t read_result = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_read_module(
      bytecode, options->filename, &session->context, &session->block_pool,
      &read_options, &read_result, &out_module->module,
      session->host_allocator));
  if (read_result.error_count > 0 || out_module->module == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "failed to read bytecode input '%.*s'",
        (int)options->filename.size, options->filename.data);
  }
  return iree_ok_status();
}

iree_status_t loom_run_module_parse(
    loom_run_session_t* session, const loom_run_module_parse_options_t* options,
    loom_run_module_t* out_module) {
  *out_module = (loom_run_module_t){
      .filename = options->filename,
      .source = options->source,
  };

  iree_status_t status =
      loom_run_module_input_is_bytecode(options->source)
          ? loom_run_module_read_bytecode(session, options, out_module)
          : loom_run_module_parse_text(session, options, out_module);
  if (!iree_status_is_ok(status)) {
    loom_run_module_deinitialize(out_module);
  }
  return status;
}

void loom_run_module_deinitialize(loom_run_module_t* run_module) {
  if (run_module == NULL) {
    return;
  }
  loom_module_free(run_module->module);
  *run_module = (loom_run_module_t){0};
}

loom_source_resolver_t loom_run_module_source_resolver(
    loom_run_module_t* run_module) {
  if (run_module == NULL || !run_module->has_source_entry) {
    return (loom_source_resolver_t){0};
  }
  return (loom_source_resolver_t){
      .fn = loom_source_table_resolve,
      .user_data = &run_module->source_table_resolver,
  };
}
