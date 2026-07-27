// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_ATTRS_H_
#define LOOM_FORMAT_TEXT_PARSER_ATTRS_H_

#include "iree/base/api.h"
#include "loom/format/text/parser/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
                                    loom_attribute_t* out_attr);
iree_status_t loom_parse_symbol_ref_attr(loom_parser_t* parser,
                                         loom_attribute_t* out_attr);
iree_status_t loom_parse_generic_attr_value(loom_parser_t* parser,
                                            uint16_t nesting_depth,
                                            loom_attribute_t* out_attr);
iree_status_t loom_parse_predicate_list(loom_parser_t* parser,
                                        loom_attribute_t* out_attr);
iree_status_t loom_parse_attr_dict(loom_parser_t* parser,
                                   loom_attribute_t* out_attr);

typedef struct loom_parsed_attr_dict_entry_t {
  // Parsed attribute payload in canonical module storage form.
  loom_named_attr_t attr;
  // Source token for duplicate-key diagnostics.
  loom_token_t key_token;
} loom_parsed_attr_dict_entry_t;

iree_status_t loom_parser_emit_duplicate_attr_dict_key(
    loom_parser_t* parser, loom_token_t key_token,
    loom_token_t previous_key_token);
void loom_parser_sort_attr_dict_entries(const loom_module_t* module,
                                        loom_parsed_attr_dict_entry_t* entries,
                                        uint16_t count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_ATTRS_H_
