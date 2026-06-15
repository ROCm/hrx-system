// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_TYPES_H_
#define LOOM_FORMAT_TEXT_PARSER_TYPES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_parser_t loom_parser_t;

// Controls how dynamic dim names are resolved during type parsing.
typedef enum loom_type_parse_mode_e {
  // Function arg context: [%M] creates a new index value if not already defined
  // in scope.
  LOOM_TYPE_PARSE_ARG = 0,
  // Op body context: [%M] must already be defined in scope.
  LOOM_TYPE_PARSE_BODY = 1,
} loom_type_parse_mode_t;

#define LOOM_PARSER_TYPE_LIST_MIN_CAPACITY 8
typedef struct loom_parser_type_list_t {
  struct loom_parser_type_list_t* next_free;
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_type_t types[];
} loom_parser_type_list_t;

#define LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS 8
typedef struct loom_parser_encoding_params_t {
  struct loom_parser_encoding_params_t* next_free;
  loom_named_attr_t* attrs;
  iree_host_size_t capacity;
  uint8_t count;

  loom_named_attr_t inline_attrs[LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS];
} loom_parser_encoding_params_t;

// Parses a type from the token stream according to |mode|.
iree_status_t loom_parse_type(loom_parser_t* parser,
                              loom_type_parse_mode_t mode,
                              loom_type_t* out_type);

// Parses a static encoding reference from a HASH_ATTR token.
iree_status_t loom_parse_static_encoding(loom_parser_t* parser,
                                         loom_string_id_t alias_id,
                                         uint16_t* out_encoding_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_TYPES_H_
