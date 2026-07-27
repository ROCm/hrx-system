// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-format/convert.h"

#include <string.h>

#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Format parsing and detection
//===----------------------------------------------------------------------===//

const char* loom_module_format_name(loom_module_format_t format) {
  switch (format) {
    case LOOM_MODULE_FORMAT_AUTO:
      return "auto";
    case LOOM_MODULE_FORMAT_TEXT:
      return "text";
    case LOOM_MODULE_FORMAT_BYTECODE:
      return "bc";
  }
  return "unknown";
}

iree_status_t loom_module_format_parse(iree_string_view_t value,
                                       bool allow_auto,
                                       loom_module_format_t* out_format) {
  if (allow_auto &&
      iree_string_view_equal(value, iree_make_cstring_view("auto"))) {
    *out_format = LOOM_MODULE_FORMAT_AUTO;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, iree_make_cstring_view("text"))) {
    *out_format = LOOM_MODULE_FORMAT_TEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, iree_make_cstring_view("bc")) ||
      iree_string_view_equal(value, iree_make_cstring_view("bytecode"))) {
    *out_format = LOOM_MODULE_FORMAT_BYTECODE;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported format '%.*s'; expected %stext, bc, or bytecode",
      (int)value.size, value.data, allow_auto ? "auto, " : "");
}

loom_module_format_t loom_module_format_detect_input(
    iree_const_byte_span_t input) {
  if (input.data_length >= LOOM_BYTECODE_MAGIC_LENGTH &&
      memcmp(input.data, LOOM_BYTECODE_MAGIC, LOOM_BYTECODE_MAGIC_LENGTH) ==
          0) {
    return LOOM_MODULE_FORMAT_BYTECODE;
  }
  return LOOM_MODULE_FORMAT_TEXT;
}

//===----------------------------------------------------------------------===//
// Module read/write
//===----------------------------------------------------------------------===//

static iree_status_t loom_format_read_text_module(
    iree_const_byte_span_t input, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_text_low_asm_environment_t low_asm_environment,
    loom_module_t** out_module) {
  iree_string_view_t source =
      iree_make_string_view((const char*)input.data, input.data_length);
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = diagnostic_sink,
      .low_asm_environment = low_asm_environment,
  };
  IREE_RETURN_IF_ERROR(loom_text_parse(source, filename, context, block_pool,
                                       &parse_options, out_module));
  if (*out_module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse text input '%.*s'",
                            (int)filename.size, filename.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_format_read_bytecode_module(
    iree_const_byte_span_t input, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_diagnostic_sink_t diagnostic_sink, loom_module_t** out_module,
    iree_allocator_t allocator) {
  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink = diagnostic_sink,
      .verify_module = false,
      .verify_max_errors = 0,
  };
  loom_bytecode_read_result_t read_result = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_read_module(
      input, filename, context, block_pool, &read_options, &read_result,
      out_module, allocator));
  if (read_result.error_count > 0 || *out_module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to read bytecode input '%.*s'",
                            (int)filename.size, filename.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_format_read_module(
    iree_const_byte_span_t input, iree_string_view_t filename,
    loom_module_format_t input_format, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, loom_diagnostic_sink_t diagnostic_sink,
    loom_text_low_asm_environment_t low_asm_environment,
    loom_module_t** out_module, iree_allocator_t allocator) {
  if (input_format == LOOM_MODULE_FORMAT_AUTO) {
    input_format = loom_module_format_detect_input(input);
  }

  switch (input_format) {
    case LOOM_MODULE_FORMAT_TEXT:
      return loom_format_read_text_module(input, filename, context, block_pool,
                                          diagnostic_sink, low_asm_environment,
                                          out_module);
    case LOOM_MODULE_FORMAT_BYTECODE:
      return loom_format_read_bytecode_module(input, filename, context,
                                              block_pool, diagnostic_sink,
                                              out_module, allocator);
    case LOOM_MODULE_FORMAT_AUTO:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported input format '%s'",
                          loom_module_format_name(input_format));
}

static iree_status_t loom_format_write_text_output(
    const loom_module_t* module,
    loom_text_low_asm_environment_t low_asm_environment,
    loom_format_output_t* out_output, iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = low_asm_environment,
  };
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &builder, &print_options);
  if (iree_status_is_ok(status)) {
    out_output->length = iree_string_builder_size(&builder);
    out_output->data = (uint8_t*)iree_string_builder_take_storage(&builder);
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_format_write_bytecode_output(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    loom_format_output_t* out_output, iree_allocator_t allocator) {
  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, allocator, &stream);

  loom_bytecode_write_options_t write_options = {
      .producer = IREE_SV("loom-format"),
      .location_mode = LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS,
  };
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_write_module(module, stream, &write_options, block_pool);
  }

  iree_io_stream_pos_t stream_length = 0;
  if (iree_status_is_ok(status)) {
    stream_length = iree_io_stream_length(stream);
    if (stream_length < 0 || stream_length > IREE_HOST_SIZE_MAX) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "bytecode output length is not host-addressable");
    }
  }

  if (iree_status_is_ok(status)) {
    out_output->length = (iree_host_size_t)stream_length;
    if (out_output->length > 0) {
      status = iree_allocator_malloc(allocator, out_output->length,
                                     (void**)&out_output->data);
    }
  }
  if (iree_status_is_ok(status) && out_output->length > 0) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status) && out_output->length > 0) {
    status =
        iree_io_stream_read(stream, out_output->length, out_output->data, NULL);
  }

  iree_io_stream_release(stream);
  if (!iree_status_is_ok(status)) {
    loom_format_output_deinitialize(out_output, allocator);
  }
  return status;
}

static iree_status_t loom_format_write_output(
    const loom_module_t* module, loom_module_format_t output_format,
    iree_arena_block_pool_t* block_pool,
    loom_text_low_asm_environment_t low_asm_environment,
    loom_format_output_t* out_output, iree_allocator_t allocator) {
  switch (output_format) {
    case LOOM_MODULE_FORMAT_TEXT:
      return loom_format_write_text_output(module, low_asm_environment,
                                           out_output, allocator);
    case LOOM_MODULE_FORMAT_BYTECODE:
      return loom_format_write_bytecode_output(module, block_pool, out_output,
                                               allocator);
    case LOOM_MODULE_FORMAT_AUTO:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "output format must be explicit");
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported output format '%s'",
                          loom_module_format_name(output_format));
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

iree_status_t loom_format_convert(iree_const_byte_span_t input,
                                  iree_string_view_t filename,
                                  loom_context_t* context,
                                  iree_arena_block_pool_t* block_pool,
                                  const loom_format_convert_options_t* options,
                                  loom_format_output_t* out_output,
                                  iree_allocator_t allocator) {
  *out_output = (loom_format_output_t){0};

  loom_format_convert_options_t resolved_options = {
      .input_format = LOOM_MODULE_FORMAT_AUTO,
      .output_format = LOOM_MODULE_FORMAT_TEXT,
      .diagnostic_sink = {0},
      .low_asm_environment = {0},
  };
  if (options != NULL) {
    resolved_options = *options;
  }

  if (resolved_options.output_format == LOOM_MODULE_FORMAT_AUTO) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output format must be explicit");
  }

  loom_module_t* module = NULL;
  iree_status_t status = loom_format_read_module(
      input, filename, resolved_options.input_format, context, block_pool,
      resolved_options.diagnostic_sink, resolved_options.low_asm_environment,
      &module, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_format_write_output(
        module, resolved_options.output_format, block_pool,
        resolved_options.low_asm_environment, out_output, allocator);
  }

  loom_module_free(module);
  if (!iree_status_is_ok(status)) {
    loom_format_output_deinitialize(out_output, allocator);
  }
  return status;
}

void loom_format_output_deinitialize(loom_format_output_t* output,
                                     iree_allocator_t allocator) {
  if (output == NULL) return;
  iree_allocator_free(allocator, output->data);
  *output = (loom_format_output_t){0};
}
