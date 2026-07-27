// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/locations.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/attrs.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/regions.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Location parsing
//===----------------------------------------------------------------------===//

enum {
  LOOM_PARSER_LOCATION_MAX_NESTING_DEPTH = 16,
};

static iree_status_t loom_parser_emit_location_error(loom_parser_t* parser,
                                                     iree_string_view_t detail,
                                                     loom_token_t token) {
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(detail),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_011, params,
                          IREE_ARRAYSIZE(params), token);
}

static iree_status_t loom_parser_resolve_location_source(
    loom_parser_t* parser, iree_string_view_t source_name,
    loom_source_id_t* out_source_id) {
  if (parser->cached_location.source_id != LOOM_SOURCE_ID_INVALID &&
      iree_string_view_equal(parser->cached_location.source_name,
                             source_name)) {
    *out_source_id = parser->cached_location.source_id;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_module_register_source(parser->module, source_name, out_source_id));
  parser->cached_location.source_name = source_name;
  parser->cached_location.source_id = *out_source_id;
  return iree_ok_status();
}

static iree_status_t loom_parse_location_integer_u16(loom_parser_t* parser,
                                                     iree_string_view_t detail,
                                                     uint16_t* out_value) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);

  int64_t value = 0;
  if (!iree_string_view_atoi_int64(token.text, &value) || value < 0 ||
      value > UINT16_MAX) {
    return loom_parser_emit_location_error(parser, detail, token);
  }

  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_parse_location_u16(loom_parser_t* parser,
                                             uint16_t* out_value) {
  return loom_parse_location_integer_u16(
      parser, IREE_SV("line/column must be an integer in [0, 65535]"),
      out_value);
}

static iree_status_t loom_parse_location_tag(loom_parser_t* parser,
                                             loom_location_tag_t* out_tag) {
  *out_tag = LOOM_LOCATION_TAG_INVALID;
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_BARE_IDENT) {
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &token);
    if (!loom_location_tag_parse(token.text, out_tag)) {
      return loom_parser_emit_location_error(
          parser, IREE_SV("unknown tagged location tag"), token);
    }
    return iree_ok_status();
  }

  if (token.kind == LOOM_TOKEN_INTEGER) {
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(loom_parse_location_integer_u16(
        parser, IREE_SV("tag must be an integer in [1, 65535]"), &value));
    if (value == LOOM_LOCATION_TAG_INVALID) {
      return loom_parser_emit_location_error(
          parser, IREE_SV("tagged location tag 0 is invalid"), token);
    }
    *out_tag = (loom_location_tag_t)value;
    return iree_ok_status();
  }

  return loom_parser_emit_location_error(
      parser, IREE_SV("expected a tagged location tag name or integer"), token);
}

static int loom_parse_location_hex_digit(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return 10 + value - 'a';
  if (value >= 'A' && value <= 'F') return 10 + value - 'A';
  return -1;
}

static iree_status_t loom_parse_location_hex_payload(
    loom_parser_t* parser, loom_token_t token, uint8_t** out_data,
    uint32_t* out_data_length) {
  *out_data = NULL;
  *out_data_length = 0;
  if (token.text.size % 2 != 0) {
    return loom_parser_emit_location_error(
        parser, IREE_SV("tagged location payload hex length must be even"),
        token);
  }
  iree_host_size_t data_length = token.text.size / 2;
  if (data_length > UINT32_MAX) {
    return loom_parser_emit_location_error(
        parser, IREE_SV("tagged location payload is too large"), token);
  }
  if (data_length == 0) {
    return iree_ok_status();
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&parser->module->arena, data_length, (void**)&data));
  for (iree_host_size_t i = 0; i < data_length; ++i) {
    int high = loom_parse_location_hex_digit(token.text.data[2 * i]);
    int low = loom_parse_location_hex_digit(token.text.data[2 * i + 1]);
    if (high < 0 || low < 0) {
      return loom_parser_emit_location_error(
          parser,
          IREE_SV("tagged location payload must contain only hex digits"),
          token);
    }
    data[i] = (uint8_t)((high << 4) | low);
  }
  *out_data = data;
  *out_data_length = (uint32_t)data_length;
  return iree_ok_status();
}

static iree_status_t loom_parse_location_coordinate_pair(loom_parser_t* parser,
                                                         uint16_t* out_line,
                                                         uint16_t* out_column) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  IREE_RETURN_IF_ERROR(loom_parse_location_u16(parser, out_line));
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  return loom_parse_location_u16(parser, out_column);
}

static iree_status_t loom_parse_location_body(loom_parser_t* parser,
                                              uint16_t nesting_depth,
                                              loom_location_id_t* out_location);

static iree_status_t loom_parse_file_location_body(
    loom_parser_t* parser, loom_location_id_t* out_location) {
  loom_token_t source_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &source_token);

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parser_resolve_location_source(
      parser, source_token.text, &source_id));

  uint16_t start_line = 0;
  uint16_t start_column = 0;
  IREE_RETURN_IF_ERROR(
      loom_parse_location_coordinate_pair(parser, &start_line, &start_column));

  uint16_t end_line = start_line;
  uint16_t end_column = start_column;
  if (loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("to"))) {
    IREE_RETURN_IF_ERROR(loom_parse_location_u16(parser, &end_line));
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
    IREE_RETURN_IF_ERROR(loom_parse_location_u16(parser, &end_column));
  }

  loom_location_entry_t entry = loom_location_file_range(
      source_id, start_line, start_column, end_line, end_column);
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_fused_location_body(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_location_id_t* out_location) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);

  uint32_t errors_before = parser->error_count;
  loom_location_id_t* child_locations = NULL;
  iree_host_size_t child_count = 0;
  iree_host_size_t child_capacity = 0;
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    for (;;) {
      if (child_count >= child_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            &parser->parser_arena, child_count, /*minimum_capacity=*/8,
            sizeof(*child_locations), &child_capacity,
            (void**)&child_locations));
      }
      IREE_RETURN_IF_ERROR(
          loom_parse_location_body(parser, (uint16_t)(nesting_depth + 1),
                                   &child_locations[child_count]));
      if (parser->error_count > errors_before) {
        return iree_ok_status();
      }
      ++child_count;
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  if (child_count > UINT32_MAX) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_location_error(
        parser, IREE_SV("fused location has too many children"), token);
  }

  loom_location_id_t* module_children = NULL;
  if (child_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, child_count, sizeof(*module_children),
        (void**)&module_children));
    memcpy(module_children, child_locations,
           child_count * sizeof(*module_children));
  }

  loom_location_entry_t entry = {
      .kind = LOOM_LOCATION_FUSED,
  };
  entry.fused.count = (uint32_t)child_count;
  entry.fused.children = module_children;
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_opaque_location_body(
    loom_parser_t* parser, loom_location_id_t* out_location) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);

  loom_token_t source_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &source_token);
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parser_resolve_location_source(
      parser, source_token.text, &source_id));

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);

  loom_token_t data_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &data_token);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  if (data_token.text.size > UINT32_MAX) {
    return loom_parser_emit_location_error(
        parser, IREE_SV("opaque location data is too large"), data_token);
  }

  uint8_t* module_data = NULL;
  if (!iree_string_view_is_empty(data_token.text)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &parser->module->arena, data_token.text.size, (void**)&module_data));
    memcpy(module_data, data_token.text.data, data_token.text.size);
  }

  loom_location_entry_t entry = {
      .kind = LOOM_LOCATION_OPAQUE,
  };
  entry.opaque.source_id = source_id;
  entry.opaque.data_length = (uint32_t)data_token.text.size;
  entry.opaque.data = module_data;
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_tagged_location_body(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_location_id_t* out_location) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);

  loom_location_tag_t tag = LOOM_LOCATION_TAG_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_location_tag(parser, &tag));
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);

  loom_token_t data_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &data_token);
  uint8_t* data = NULL;
  uint32_t data_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_parse_location_hex_payload(parser, data_token, &data, &data_length));

  loom_location_id_t child = LOOM_LOCATION_UNKNOWN;
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
    IREE_RETURN_IF_ERROR(loom_parse_location_body(
        parser, (uint16_t)(nesting_depth + 1), &child));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  loom_location_entry_t entry =
      loom_location_tagged(tag, child, data, data_length);
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_location_body(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_location_id_t* out_location) {
  if (nesting_depth >= LOOM_PARSER_LOCATION_MAX_NESTING_DEPTH) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_location_error(
        parser, IREE_SV("location nesting exceeds the maximum supported depth"),
        token);
  }

  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_STRING) {
    return loom_parse_file_location_body(parser, out_location);
  }
  if (token.kind == LOOM_TOKEN_BARE_IDENT &&
      iree_string_view_equal(token.text, IREE_SV("fused"))) {
    return loom_parse_fused_location_body(parser, nesting_depth, out_location);
  }
  if (token.kind == LOOM_TOKEN_BARE_IDENT &&
      iree_string_view_equal(token.text, IREE_SV("opaque"))) {
    return loom_parse_opaque_location_body(parser, out_location);
  }
  if (token.kind == LOOM_TOKEN_BARE_IDENT &&
      iree_string_view_equal(token.text, IREE_SV("tagged"))) {
    return loom_parse_tagged_location_body(parser, nesting_depth, out_location);
  }
  return loom_parser_emit_location_error(
      parser,
      IREE_SV(
          "expected a file location string, 'fused', 'opaque', or 'tagged'"),
      token);
}

iree_status_t loom_parse_optional_op_location(
    loom_parser_t* parser, loom_location_id_t fallback_location,
    loom_location_id_t* out_location) {
  *out_location = fallback_location;
  if (!loom_tokenizer_at_keyword(&parser->tokenizer, IREE_SV("loc"))) {
    return iree_ok_status();
  }

  uint32_t errors_before = parser->error_count;
  if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("loc"))) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, token, IREE_SV("'loc'"));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  IREE_RETURN_IF_ERROR(
      loom_parse_location_body(parser, /*nesting_depth=*/0, out_location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return iree_ok_status();
}
