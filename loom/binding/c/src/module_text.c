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
#include "pass_program.h"
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
      .max_errors = options->max_errors,
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

static void loomc_module_text_print_options(
    const loomc_context_t* context,
    const loomc_module_resolved_serialize_options_t* options,
    loom_text_print_options_t* out_options) {
  *out_options = (loom_text_print_options_t){
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  if (options->print_locations) {
    out_options->flags |= LOOM_TEXT_PRINT_LOCATIONS;
  }
  loomc_target_pass_environment_initialize_text_asm_environment(
      loomc_context_target_pass_environment(context),
      &out_options->low_asm_environment);
  switch (options->text_presentation) {
    case LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT:
      out_options->flags |= LOOM_TEXT_PRINT_PREFER_LOW_ASM;
      break;
    case LOOMC_MODULE_TEXT_PRESENTATION_GENERIC:
      break;
    case LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM:
      out_options->flags |=
          LOOM_TEXT_PRINT_PREFER_LOW_ASM | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM;
      break;
    case LOOMC_MODULE_TEXT_PRESENTATION_PRESERVE:
      out_options->flags |= LOOM_TEXT_PRINT_PRESERVE_LOW_ASM;
      break;
  }
}

static bool loomc_module_op_matches_dialect(const loom_op_vtable_t* vtable,
                                            loomc_string_view_t dialect) {
  if (vtable == NULL || dialect.size == 0) {
    return false;
  }
  const iree_string_view_t op_name = loom_op_vtable_name(vtable);
  const iree_string_view_t dialect_name = iree_string_view_from_loomc(dialect);
  return op_name.size > dialect_name.size &&
         op_name.data[dialect_name.size] == '.' &&
         iree_string_view_starts_with(op_name, dialect_name);
}

static loomc_status_t loomc_module_print_internal_text_to_source(
    const loomc_context_t* context, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_text_print_options_t print_options;
  loomc_module_text_print_options(context, options, &print_options);
  loomc_status_t status = loomc_ok_status();
  if (loomc_string_view_is_empty(options->top_level_dialect)) {
    status =
        loomc_status_from_iree(loom_text_print_module_to_builder_with_options(
            internal_module, &builder, &print_options));
  } else {
    // A filtered view omits module-level encoding aliases. Inline every
    // encoding so each selected operation remains independently parseable.
    print_options.flags &= ~LOOM_TEXT_PRINT_USE_ALIASES;
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    loom_text_printer_t printer;
    loom_text_printer_initialize(internal_module, &printer);
    bool has_output = false;
    const loom_block_t* block =
        loom_region_const_entry_block(internal_module->body);
    iree_status_t print_status = iree_ok_status();
    for (const loom_op_t* op = block ? block->first_op : NULL;
         op != NULL && iree_status_is_ok(print_status); op = op->next_op) {
      const loom_op_vtable_t* vtable = loom_op_vtable(internal_module, op);
      if (!loomc_module_op_matches_dialect(vtable,
                                           options->top_level_dialect)) {
        continue;
      }
      if (has_output) {
        print_status = loom_output_stream_write_char(&stream, '\n');
      }
      if (iree_status_is_ok(print_status)) {
        print_status = loom_text_printer_write_operation(&printer, op, &stream,
                                                         &print_options);
      }
      has_output = true;
    }
    loom_text_printer_deinitialize(&printer);
    status = loomc_status_from_iree(print_status);
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

static loomc_status_t loomc_module_encode_text_source(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  return loomc_module_print_internal_text_to_source(
      loomc_module_context(module), internal_module, options, allocator,
      out_source);
}

loomc_status_t loomc_module_deserialize_text_from_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loomc_status_t return_status = loomc_module_deserialize_explicit_source(
      context, workspace, source, options, LOOMC_SOURCE_FORMAT_TEXT,
      loomc_module_decode_text_source, allocator, out_module, out_result);
  IREE_TRACE_ZONE_END(z0);
  return return_status;
}

loomc_status_t loomc_module_print_text_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  loomc_module_resolved_serialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_module_resolve_serialize_options(
      options, LOOMC_SOURCE_FORMAT_TEXT, &resolved_options));
  const loom_module_t* internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(module, &internal_module));
  IREE_TRACE_ZONE_BEGIN(z0);
  loomc_status_t status = loomc_module_encode_text_source(
      module, internal_module, &resolved_options, allocator, out_source);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

loomc_status_t loomc_module_serialize_text_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loomc_status_t return_status = loomc_module_serialize_explicit_source(
      module, options, LOOMC_SOURCE_FORMAT_TEXT,
      loomc_module_encode_text_source, allocator, out_source);
  IREE_TRACE_ZONE_END(z0);
  return return_status;
}

loomc_status_t loomc_pass_program_print_text_to_source(
    const loomc_pass_program_t* pass_program,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  if (pass_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass_program must not be NULL");
  }
  loomc_module_resolved_serialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_module_resolve_serialize_options(
      options, LOOMC_SOURCE_FORMAT_TEXT, &resolved_options));
  IREE_TRACE_ZONE_BEGIN(z0);
  loomc_status_t status = loomc_module_print_internal_text_to_source(
      loomc_pass_program_context(pass_program),
      loomc_pass_program_loom_pass_program(pass_program)->source_module,
      &resolved_options, allocator, out_source);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
