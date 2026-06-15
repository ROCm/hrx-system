// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_FORMAT_TABLES_H_
#define LOOM_FORMAT_TEXT_PARSER_FORMAT_TABLES_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_parse_format_inline_attr_dict(loom_parser_t* parser,
                                                 const loom_op_vtable_t* vtable,
                                                 loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_apply_elided_attr_defaults(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element, loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_operand_dict(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_attr_table(loom_parser_t* parser,
                                           const loom_op_vtable_t* vtable,
                                           const loom_format_element_t* element,
                                           loom_parsed_op_t* parsed);
iree_status_t loom_parse_format_region_table(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, loom_parsed_op_t* parsed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_FORMAT_TABLES_H_
