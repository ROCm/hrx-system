// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "context.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loomc/iree.h"
#include "module.h"
#include "source.h"
#include "target.h"

static loomc_status_t loomc_module_decode_text_source(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loom_diagnostic_sink_t diagnostic_sink, loomc_allocator_t allocator,
    loomc_module_t* module, loom_module_t** out_internal_module) {
  (void)allocator;
  *out_internal_module = NULL;
  const loomc_byte_span_t contents = loomc_source_contents(source);
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = diagnostic_sink,
  };
  loomc_target_pass_environment_initialize_text_asm_environment(
      loomc_context_target_pass_environment(context),
      &parse_options.low_asm_environment);
  return loomc_status_from_iree(loom_text_parse(
      iree_make_string_view((const char*)contents.data, contents.data_length),
      iree_string_view_from_loomc(options->identifier),
      loomc_context_loom_context(context), loomc_module_block_pool(module),
      &parse_options, out_internal_module));
}

static loomc_status_t loomc_module_text_print_options(
    const loomc_module_t* module,
    const loomc_module_resolved_serialize_options_t* options,
    loom_text_print_options_t* out_options) {
  *out_options = (loom_text_print_options_t){
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  loomc_target_pass_environment_initialize_text_asm_environment(
      loomc_context_target_pass_environment(loomc_module_context(module)),
      &out_options->low_asm_environment);
  if (options->text_presentation == LOOMC_MODULE_TEXT_PRESENTATION_GENERIC) {
    return loomc_ok_status();
  }
  out_options->flags |= LOOM_TEXT_PRINT_PREFER_LOW_ASM;
  if (options->text_presentation == LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM) {
    out_options->flags |= LOOM_TEXT_PRINT_REQUIRE_LOW_ASM;
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_module_encode_text_source(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_text_print_options_t print_options;
  loomc_status_t status =
      loomc_module_text_print_options(module, options, &print_options);
  if (loomc_status_is_ok(status)) {
    status =
        loomc_status_from_iree(loom_text_print_module_to_builder_with_options(
            internal_module, &builder, &print_options));
  }
  const iree_host_size_t length = iree_string_builder_size(&builder);
  char* storage = NULL;
  if (loomc_status_is_ok(status)) {
    storage = iree_string_builder_take_storage(&builder);
    status = loomc_source_create_take_contents(
        LOOMC_SOURCE_FORMAT_TEXT, options->identifier,
        loomc_make_byte_span(storage, length), allocator, out_source);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, storage);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

loomc_status_t loomc_module_deserialize_text_from_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  return loomc_module_deserialize_explicit_source(
      context, workspace, source, options, LOOMC_SOURCE_FORMAT_TEXT,
      loomc_module_decode_text_source, allocator, out_module, out_result);
}

loomc_status_t loomc_module_serialize_text_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  return loomc_module_serialize_explicit_source(
      module, options, LOOMC_SOURCE_FORMAT_TEXT,
      loomc_module_encode_text_source, allocator, out_source);
}
