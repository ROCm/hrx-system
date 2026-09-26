// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_REGIONS_H_
#define LOOM_FORMAT_TEXT_PARSER_REGIONS_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

loom_token_kind_t loom_keyword_token_kind(uint16_t keyword_id);
iree_status_t loom_parse_keyword(loom_parser_t* parser, uint16_t keyword_id);

iree_status_t loom_parser_seed_region_entry_block(loom_parser_t* parser,
                                                  loom_region_t* region);
iree_status_t loom_parser_parse_optional_block_label(loom_parser_t* parser,
                                                     loom_region_t* region,
                                                     loom_block_t* block,
                                                     bool* out_present);
iree_status_t loom_parser_append_implicit_terminator(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_block_t* block);

// Parses one operation using an already-resolved assembly spelling, or NULL
// to resolve the canonical dotted name at the current token position.
iree_status_t loom_parse_op(loom_parser_t* parser,
                            const loom_op_assembly_format_t* assembly);

typedef iree_status_t (*loom_parse_region_body_fn_t)(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t* region, const void* user_data);

typedef struct loom_parse_region_body_callback_t {
  // Parses the already-opened region body and consumes the closing brace.
  // Errors leave remaining input for recovery by the owning operation.
  loom_parse_region_body_fn_t fn;
  // Opaque parser-internal state passed to |fn|.
  const void* user_data;
} loom_parse_region_body_callback_t;

iree_status_t loom_parse_braced_region_with_body(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_parse_region_body_callback_t body, loom_region_t** out_region);
iree_status_t loom_parse_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region);
iree_status_t loom_parse_region_with_syntax(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_syntax_t syntax, loom_region_t** out_region);

iree_status_t loom_parser_emit_result_count_mismatch(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, uint16_t expected_count, uint16_t actual_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_REGIONS_H_
