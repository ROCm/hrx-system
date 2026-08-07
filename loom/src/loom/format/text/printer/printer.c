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
    const loom_text_print_options_t* options,
    const loom_print_name_plan_t* name_plan) {
  loom_print_context_t context = {0};
  context.stream = stream;
  context.module = module;
  context.name_plan = name_plan;
  context.flags = options ? options->flags : LOOM_TEXT_PRINT_DEFAULT;
  if (options) context.low_asm_environment = options->low_asm_environment;
  return context;
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
  loom_print_name_plan_t name_plan;
  IREE_RETURN_IF_ERROR(loom_print_name_plan_initialize(module, &name_plan));
  loom_print_context_t ctx =
      loom_print_context_make(module, stream, options, &name_plan);
  iree_status_t status = loom_print_encoding_aliases(&ctx, module);
  if (iree_status_is_ok(status)) {
    status = loom_print_module_body(&ctx, module->body);
  }
  loom_print_name_plan_deinitialize(&name_plan);
  return status;
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
  loom_print_name_plan_t name_plan;
  IREE_RETURN_IF_ERROR(loom_print_name_plan_initialize(module, &name_plan));
  loom_print_context_t ctx =
      loom_print_context_make(module, stream, options, &name_plan);
  iree_status_t status = loom_print_op_comments(&ctx, op);
  if (iree_status_is_ok(status)) status = loom_print_op(&ctx, op);
  loom_print_name_plan_deinitialize(&name_plan);
  return status;
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
  loom_print_name_plan_t name_plan;
  IREE_RETURN_IF_ERROR(loom_print_name_plan_initialize(module, &name_plan));
  loom_print_context_t ctx = {
      .stream = &stream,
      .module = module,
      .name_plan = &name_plan,
      .flags = flags,
      .field_callback = callback,
  };
  iree_status_t status = loom_print_op_comments(&ctx, op);
  if (iree_status_is_ok(status)) status = loom_print_op(&ctx, op);
  loom_print_name_plan_deinitialize(&name_plan);
  return status;
}
