// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_FORMAT_SIGNATURES_H_
#define LOOM_FORMAT_TEXT_PARSER_FORMAT_SIGNATURES_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses one body result annotation: a type or `%operand as type`. Records
// authored ties and their operand source spans in |parsed| without defining
// result values. Canonical operations and Low assembly share this syntax.
iree_status_t loom_parse_body_result_type(loom_parser_t* parser,
                                          const loom_value_id_t* operands,
                                          iree_host_size_t operand_count,
                                          uint16_t result_index,
                                          loom_parsed_op_t* parsed,
                                          loom_type_t* out_type);

iree_status_t loom_parse_format_result_type(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, bool defines_signature_results);
iree_status_t loom_parse_format_result_type_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, bool defines_signature_results);
iree_status_t loom_parse_format_binding_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_block_args(loom_parser_t* parser,
                                           const loom_format_element_t* element,
                                           uint16_t pending_block_arg_base,
                                           loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_func_args(loom_parser_t* parser,
                                          const loom_op_vtable_t* vtable,
                                          const loom_format_element_t* element,
                                          uint16_t pending_func_arg_base,
                                          loom_parsed_op_t* parsed);

// Completes the argument-then-result flags after the whole signature is parsed.
// FuncArgs accumulates argument bytes in parser storage; this publishes the
// complete immutable payload in the module arena.
iree_status_t loom_parse_finalize_signature_flags(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_parsed_op_t* parsed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_FORMAT_SIGNATURES_H_
