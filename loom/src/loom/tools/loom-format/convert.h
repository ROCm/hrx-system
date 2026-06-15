// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Format conversion support for loom-format.
//
// The converter always materializes through loom_module_t so every conversion
// path exercises the same parser/reader and printer/writer contracts as the
// rest of the compiler. Bytecode-to-bytecode and text-to-text are therefore
// canonicalization paths, not byte copying paths.

#ifndef LOOM_TOOLS_LOOM_FORMAT_CONVERT_H_
#define LOOM_TOOLS_LOOM_FORMAT_CONVERT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/low_asm.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supported external module formats.
typedef enum loom_module_format_e {
  LOOM_MODULE_FORMAT_AUTO = 0,
  LOOM_MODULE_FORMAT_TEXT = 1,
  LOOM_MODULE_FORMAT_BYTECODE = 2,
} loom_module_format_t;

// Options controlling a single format conversion.
typedef struct loom_format_convert_options_t {
  // Input format. AUTO detects bytecode by the `LOOM` file magic and treats all
  // other input as text.
  loom_module_format_t input_format;

  // Output format. AUTO is not valid because output must be explicit.
  loom_module_format_t output_format;

  // Sink for parser and bytecode-reader diagnostics. If fn is NULL, diagnostics
  // are dropped and conversion failure is reported through the returned status.
  loom_diagnostic_sink_t diagnostic_sink;

  // Optional environment used to parse and print target-low register types.
  loom_text_low_asm_environment_t low_asm_environment;
} loom_format_convert_options_t;

// Allocator-owned output bytes from a conversion.
typedef struct loom_format_output_t {
  // Output byte buffer allocated with the caller-provided allocator.
  uint8_t* data;

  // Number of valid bytes in data.
  iree_host_size_t length;
} loom_format_output_t;

// Returns the stable flag spelling for |format|.
const char* loom_module_format_name(loom_module_format_t format);

// Parses a format flag value: auto, text, bc, or bytecode.
iree_status_t loom_module_format_parse(iree_string_view_t value,
                                       bool allow_auto,
                                       loom_module_format_t* out_format);

// Detects the input format from bytes using the bytecode file magic.
loom_module_format_t loom_module_format_detect_input(
    iree_const_byte_span_t input);

// Converts |input| from options.input_format to options.output_format.
//
// The context must be finalized with every dialect/encoding that may appear in
// the input. |block_pool| supplies transient parser, reader, and writer
// storage. The returned output must be released with
// loom_format_output_deinitialize().
iree_status_t loom_format_convert(iree_const_byte_span_t input,
                                  iree_string_view_t filename,
                                  loom_context_t* context,
                                  iree_arena_block_pool_t* block_pool,
                                  const loom_format_convert_options_t* options,
                                  loom_format_output_t* out_output,
                                  iree_allocator_t allocator);

// Releases output storage. Safe to call with zero-initialized output.
void loom_format_output_deinitialize(loom_format_output_t* output,
                                     iree_allocator_t allocator);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_FORMAT_CONVERT_H_
