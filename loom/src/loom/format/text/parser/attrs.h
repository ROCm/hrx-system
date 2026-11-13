// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_ATTRS_H_
#define LOOM_FORMAT_TEXT_PARSER_ATTRS_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/context.h"
#include "loom/format/text/parser/types.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
                                    loom_attribute_t* out_attr);
iree_status_t loom_parse_attr_value_with_type_mode(
    loom_parser_t* parser, const loom_attr_descriptor_t* descriptor,
    loom_type_parse_mode_t type_mode, loom_attribute_t* out_attr);
// Parses the angle-bracket parameter payload for one known attribute family.
iree_status_t loom_parse_parameterized_attr_parameters(
    loom_parser_t* parser, loom_parameterized_attr_kind_t family_kind,
    loom_attribute_t* out_attr);
// Parses a symbol reference, admitting definition names before their bodies are
// constructed. Duplicate definitions retain the first definition's source
// token.
iree_status_t loom_parse_symbol_ref_attr(loom_parser_t* parser,
                                         bool is_definition,
                                         loom_attribute_t* out_attr);
iree_status_t loom_parse_generic_attr_value(loom_parser_t* parser,
                                            uint16_t nesting_depth,
                                            loom_attribute_t* out_attr);
// Parses a bracketed predicate list using the enclosing SSA binding mode.
iree_status_t loom_parse_predicate_list(loom_parser_t* parser,
                                        loom_type_parse_mode_t type_mode,
                                        loom_attribute_t* out_attr);
iree_status_t loom_parse_attr_dict(loom_parser_t* parser,
                                   loom_attribute_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_ATTRS_H_
