// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/printer.h"

#include "loom/format/text/printer/atoms.h"
#include "loom/format/text/printer/context.h"
#include "loom/format/text/printer/regions.h"

static loom_print_context_t loom_print_context_make(
    const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options) {
  loom_text_print_flags_t flags =
      options ? options->flags : LOOM_TEXT_PRINT_DEFAULT;
  return (loom_print_context_t){
      .stream = stream,
      .module = module,
      .flags = flags,
      .low_asm_environment = options ? options->low_asm_environment
                                     : (loom_text_low_asm_environment_t){0},
      .low_asm_descriptor_set_key = options
                                        ? options->low_asm_descriptor_set_key
                                        : iree_string_view_empty(),
  };
}

iree_status_t loom_text_print_module(const loom_module_t* module,
                                     loom_output_stream_t* stream,
                                     loom_text_print_flags_t flags) {
  loom_text_print_options_t options = {
      .flags = flags,
  };
  return loom_text_print_module_with_options(module, stream, &options);
}

iree_status_t loom_text_print_module_with_options(
    const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options) {
  if (!module || !module->body) {
    return iree_ok_status();
  }
  loom_print_context_t ctx = loom_print_context_make(module, stream, options);
  IREE_RETURN_IF_ERROR(loom_print_encoding_aliases(&ctx, module));
  return loom_print_module_body(&ctx, module->body);
}

iree_status_t loom_text_print_operation(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_output_stream_t* stream,
                                        loom_text_print_flags_t flags) {
  loom_text_print_options_t options = {
      .flags = flags,
  };
  return loom_text_print_operation_with_options(module, op, stream, &options);
}

iree_status_t loom_text_print_operation_with_options(
    const loom_module_t* module, const loom_op_t* op,
    loom_output_stream_t* stream, const loom_text_print_options_t* options) {
  if (!module || !op) {
    return iree_ok_status();
  }
  loom_print_context_t ctx = loom_print_context_make(module, stream, options);
  IREE_RETURN_IF_ERROR(loom_print_op_comments(&ctx, op));
  return loom_print_op(&ctx, op);
}

iree_status_t loom_text_print_module_to_builder(const loom_module_t* module,
                                                iree_string_builder_t* builder,
                                                loom_text_print_flags_t flags) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_module(module, &stream, flags);
}

iree_status_t loom_text_print_module_to_builder_with_options(
    const loom_module_t* module, iree_string_builder_t* builder,
    const loom_text_print_options_t* options) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_module_with_options(module, &stream, options);
}

iree_status_t loom_text_print_operation_to_builder(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_operation(module, op, &stream, flags);
}

iree_status_t loom_text_print_operation_to_builder_with_options(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, const loom_text_print_options_t* options) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_operation_with_options(module, op, &stream, options);
}

iree_status_t loom_text_print_operation_with_field_callback(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags,
    loom_print_field_callback_t callback) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_print_context_t ctx = {
      .stream = &stream,
      .module = module,
      .flags = flags,
      .field_callback = callback,
  };
  IREE_RETURN_IF_ERROR(loom_print_op_comments(&ctx, op));
  return loom_print_op(&ctx, op);
}
