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

iree_status_t loom_parse_format_result_type(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, bool is_symbol_definition);
iree_status_t loom_parse_format_result_type_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed, bool is_symbol_definition);
iree_status_t loom_parse_format_binding_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_block_args(loom_parser_t* parser);
iree_status_t loom_parse_format_func_args(loom_parser_t* parser,
                                          loom_parsed_op_t* parsed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_FORMAT_SIGNATURES_H_
