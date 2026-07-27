// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Declarations shared by low-asm parser leaves.

#ifndef LOOM_FORMAT_TEXT_PARSER_LOW_ASM_H_
#define LOOM_FORMAT_TEXT_PARSER_LOW_ASM_H_

#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/attrs.h"
#include "loom/format/text/parser/context.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/locations.h"
#include "loom/format/text/parser/regions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_asm_result_names_t {
  // Result-name tokens in parsed source order.
  loom_token_t* tokens;
  // Number of result-name tokens currently stored.
  iree_host_size_t count;
  // Number of result-name tokens that fit in |tokens|.
  iree_host_size_t capacity;
  // Inline storage for the common single-result asm packet.
  loom_token_t inline_tokens[4];
} loom_low_asm_result_names_t;

typedef struct loom_low_asm_value_list_t {
  // SSA value IDs in parsed source order.
  loom_value_id_t* values;
  // Number of value IDs currently stored.
  iree_host_size_t count;
  // Number of value IDs that fit in |values|.
  iree_host_size_t capacity;
  // Inline storage for the common small return packet.
  loom_value_id_t inline_values[4];
} loom_low_asm_value_list_t;

// Initializes an empty low-asm result-name list.
void loom_low_asm_result_names_initialize(loom_low_asm_result_names_t* names);

// Appends a result name token.
iree_status_t loom_low_asm_result_names_append(
    loom_parser_t* parser, loom_low_asm_result_names_t* names,
    loom_token_t token);

// Initializes an empty low-asm value list.
void loom_low_asm_value_list_initialize(loom_low_asm_value_list_t* values);

// Appends a value ID.
iree_status_t loom_low_asm_value_list_append(loom_parser_t* parser,
                                             loom_low_asm_value_list_t* values,
                                             loom_value_id_t value_id);

// Emits a low-asm parse error with a detail string.
iree_status_t loom_parser_emit_low_asm_error(loom_parser_t* parser,
                                             loom_token_t token,
                                             iree_string_view_t detail);

// Emits a low-asm result-count mismatch diagnostic.
iree_status_t loom_parser_emit_low_asm_result_count_mismatch(
    loom_parser_t* parser, loom_token_t mnemonic_token, uint32_t expected_count,
    uint32_t actual_count);

// Emits a low-asm operand-count mismatch diagnostic.
iree_status_t loom_parser_emit_low_asm_operand_count_mismatch(
    loom_parser_t* parser, loom_token_t mnemonic_token, uint32_t expected_count,
    uint32_t actual_count);

// Parses an optional packet location and rejects trailing same-line tokens.
iree_status_t loom_parse_low_asm_packet_location(
    loom_parser_t* parser, loom_token_t start_token,
    loom_token_t mnemonic_token, loom_parsed_op_t* parsed_spans,
    loom_location_id_t* out_location);

// Parses the target-low asm prefixed region syntax:
// `asm<descriptor-set> { packet... }`.
iree_status_t loom_parse_low_asm_prefixed_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region);

// Parses a braced low-asm region inheriting the active descriptor set.
iree_status_t loom_parse_low_asm_inherited_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region);

// Returns true when |token| names a low-asm structural intrinsic.
bool loom_low_asm_structural_kind_from_token(
    loom_token_t token, loom_text_low_asm_structural_kind_t* out_kind);

// Parses and builds a low-asm structural intrinsic packet.
iree_status_t loom_parse_low_asm_structural(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_LOW_ASM_H_
