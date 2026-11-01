// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_FORMAT_H_
#define LOOM_FORMAT_TEXT_PARSER_FORMAT_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses an optional <flag|flag> list using the operation's declared
// vocabulary.
iree_status_t loom_parse_format_flags(loom_parser_t* parser,
                                      const loom_op_vtable_t* vtable,
                                      loom_parsed_op_t* parsed);

// Parses the contents of an instance-flag list, without its angle brackets.
iree_status_t loom_parse_format_instance_flag_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable, uint8_t* out_flags);

// Transfers signature arguments to the next region entry, optionally cloning
// their identities for regions that project a distinct copy of the signature.
iree_status_t loom_parse_format_project_func_args(
    loom_parser_t* parser, uint16_t pending_func_arg_start, bool clone_values);

iree_status_t loom_parser_walk_format(
    loom_parser_t* parser, const loom_op_vtable_t* vtable, loom_format_t format,
    loom_token_t op_name_token, loom_parsed_op_t* parsed,
    uint16_t pending_func_arg_start, bool* out_func_args_consumed_by_region);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_FORMAT_H_
