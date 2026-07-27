// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_CONTEXT_H_
#define LOOM_FORMAT_TEXT_PRINTER_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/printer.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_print_context_t {
  // Output stream receiving textual IR bytes.
  loom_output_stream_t* stream;
  // Module whose string, type, value, and location tables are printed.
  const loom_module_t* module;
  // Flag bitset controlling layout and optional annotations.
  loom_text_print_flags_t flags;
  // Optional descriptor-backed environment for low asm region syntax.
  loom_text_low_asm_environment_t low_asm_environment;
  // Descriptor-set context used for printing target-low register types.
  const loom_text_low_asm_descriptor_set_t* low_register_descriptor_set;
  // Descriptor-set key selected for low asm region syntax.
  iree_string_view_t low_asm_descriptor_set_key;
  // Nesting depth of active descriptor-backed low asm region bodies.
  uint16_t low_asm_region_depth;
  // True when the current logical line already contains a printed token.
  bool has_previous_token;
  // True when the next token should be glued to the previous token.
  bool glue_next;
  // Last byte emitted through the token spacing model.
  char last_char;
  // Current indentation depth in logical two-space levels.
  uint16_t indent;
  // Optional callback receiving semantic field byte ranges.
  loom_print_field_callback_t field_callback;
} loom_print_context_t;

// Emits text through the token spacing model.
iree_status_t loom_print_emit(loom_print_context_t* ctx,
                              iree_string_view_t text, bool glue);

// Emits a C string through the token spacing model.
iree_status_t loom_print_emit_cstr(loom_print_context_t* ctx, const char* text,
                                   bool glue);

// Emits a space if the token spacing model requires one before direct output.
iree_status_t loom_print_space_if_needed(loom_print_context_t* ctx);

// Updates token spacing state after direct output.
void loom_print_did_write(loom_print_context_t* ctx);

// Glues the next token to the previously printed token.
void loom_print_set_glue(loom_print_context_t* ctx);

// Reports a semantic field byte range to the optional field callback.
void loom_print_report_field(loom_print_context_t* ctx,
                             loom_print_field_ref_t field_ref,
                             iree_host_size_t start, iree_host_size_t end);

// Returns the byte offset where a token would start after spacing is applied.
iree_host_size_t loom_print_next_token_start_offset(
    const loom_print_context_t* ctx, bool glue, char first_char);

// Writes indentation for a specific logical indentation depth.
iree_status_t loom_print_indent_at(loom_print_context_t* ctx, uint16_t indent);

// Writes indentation for the current logical line.
iree_status_t loom_print_indent(loom_print_context_t* ctx);

// Prints comments attached to |op| at the current indentation.
iree_status_t loom_print_op_comments(loom_print_context_t* ctx,
                                     const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_CONTEXT_H_
