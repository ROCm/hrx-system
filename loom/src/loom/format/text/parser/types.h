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
  // Next reusable list owned by the parser, or NULL when in use.
  struct loom_parser_type_list_t* next_free;
  // Number of populated entries in |types|.
  iree_host_size_t count;
  // Maximum number of entries available in |types|.
  iree_host_size_t capacity;
  // Parsed types retained for the duration of one composite type parse.
  loom_type_t types[];
} loom_parser_type_list_t;

#define LOOM_PARSER_TYPE_PARAMETER_SLOTS_MIN_CAPACITY 4
typedef struct loom_parser_type_parameter_slots_t {
  // Next reusable slot frame owned by the parser, or NULL when in use.
  struct loom_parser_type_parameter_slots_t* next_free;
  // Maximum number of entries available in |slots|.
  iree_host_size_t capacity;
  // Parsed attributes retained for one descriptor-backed type parse.
  loom_attribute_t slots[];
} loom_parser_type_parameter_slots_t;

#define LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS 8
typedef struct loom_parser_encoding_params_t {
  // Next reusable parameter frame owned by the parser, or NULL when in use.
  struct loom_parser_encoding_params_t* next_free;
  // Active parameter storage, either inline or parser-arena allocated.
  loom_named_attr_t* attrs;
  // Maximum number of entries available in |attrs|.
  iree_host_size_t capacity;
  // Number of populated entries in |attrs|.
  uint8_t count;

  // Inline storage covering the common encoding parameter count.
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
