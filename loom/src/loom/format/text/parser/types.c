// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/types.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/aliases.h"
#include "loom/format/text/parser/attrs.h"
#include "loom/format/text/parser/context.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/scope.h"
#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"

//===----------------------------------------------------------------------===//
// Encoding parameter parsing
//===----------------------------------------------------------------------===//

static void loom_parser_encoding_params_initialize(
    loom_parser_encoding_params_t* params) {
  memset(params, 0, sizeof(*params));
  params->attrs = params->inline_attrs;
  params->capacity = LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS;
}

static void loom_parser_encoding_params_reset(
    loom_parser_encoding_params_t* params) {
  params->count = 0;
}

static iree_status_t loom_parser_acquire_encoding_params(
    loom_parser_t* parser, loom_parser_encoding_params_t** out_params) {
  loom_parser_encoding_params_t* params = parser->encoding_params_free_list;
  if (params) {
    parser->encoding_params_free_list = params->next_free;
    params->next_free = NULL;
    loom_parser_encoding_params_reset(params);
    *out_params = params;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate(&parser->parser_arena,
                                           sizeof(*params), (void**)&params));
  loom_parser_encoding_params_initialize(params);
  *out_params = params;
  return iree_ok_status();
}

static void loom_parser_release_encoding_params(
    loom_parser_t* parser, loom_parser_encoding_params_t* params) {
  if (!params) return;
  params->next_free = parser->encoding_params_free_list;
  parser->encoding_params_free_list = params;
}

static iree_status_t loom_parse_encoding_params_emit_static_ssa_value(
    loom_parser_t* parser, loom_token_t name_token, loom_token_t value_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_028, params,
                          IREE_ARRAYSIZE(params), value_token);
}

static iree_status_t loom_parse_encoding_params_emit_duplicate_name(
    loom_parser_t* parser, loom_token_t name_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_029, params,
                          IREE_ARRAYSIZE(params), name_token);
}

// Parses encoding parameters from the token stream. Called after LANGLE
// has been consumed. Consumes tokens through the closing RANGLE
// (inclusive). Grammar: key=value [, key=value]* >.
static iree_status_t loom_parse_encoding_params(
    loom_parser_t* parser, loom_parser_encoding_params_t* params) {
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (params->count > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }

    if (params->count == UINT8_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }
    if (params->count >= params->capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, params->count,
          /*minimum_capacity=*/LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS,
          sizeof(loom_named_attr_t), &params->capacity,
          (void**)&params->attrs));
    }

    // Parameter name.
    loom_token_t name_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &name_token);

    // Separator.
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_string_id_t param_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        parser->module, name_token.text, &param_name_id));
    for (uint8_t i = 0; i < params->count; ++i) {
      if (params->attrs[i].name_id == param_name_id) {
        return loom_parse_encoding_params_emit_duplicate_name(parser,
                                                              name_token);
      }
    }
    params->attrs[params->count].name_id = param_name_id;
    params->attrs[params->count].reserved = 0;

    uint32_t errors_before = parser->error_count;
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      loom_token_t value_token = loom_tokenizer_peek(&parser->tokenizer);
      IREE_RETURN_IF_ERROR(loom_parse_encoding_params_emit_static_ssa_value(
          parser, name_token, value_token));
    } else {
      IREE_RETURN_IF_ERROR(loom_parse_generic_attr_value(
          parser, /*nesting_depth=*/0, &params->attrs[params->count].value));
    }
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    ++params->count;
  }

  // Consume closing '>'.
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
  return iree_ok_status();
}

iree_status_t loom_parse_static_encoding(loom_parser_t* parser,
                                         loom_string_id_t alias_id,
                                         uint16_t* out_encoding_id) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_HASH_ATTR, &token);

  uint16_t aliased_id = loom_alias_table_lookup(&parser->aliases, token.text);
  if (aliased_id != 0) {
    *out_encoding_id = aliased_id;
    return iree_ok_status();
  }

  loom_encoding_t encoding = {
      .name_id = LOOM_STRING_ID_INVALID,
      .alias_id = alias_id,
  };
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, token.text, &encoding.name_id));

  loom_parser_encoding_params_t* params_scratch = NULL;
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    IREE_RETURN_IF_ERROR(
        loom_parser_acquire_encoding_params(parser, &params_scratch));

    uint32_t errors_before = parser->error_count;
    iree_status_t status = loom_parse_encoding_params(parser, params_scratch);
    if (iree_status_is_ok(status) && parser->error_count > errors_before) {
      loom_parser_release_encoding_params(parser, params_scratch);
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) {
      loom_parser_release_encoding_params(parser, params_scratch);
      return status;
    }
    encoding.attribute_count = params_scratch->count;
    encoding.attributes = params_scratch->attrs;
  }

  if (parser->context->encoding_vtables.count > 0 &&
      !loom_context_lookup_encoding_vtable(parser->context, token.text)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    loom_parser_release_encoding_params(parser, params_scratch);
    return loom_parser_emit(parser, LOOM_ERR_PARSE_008, params,
                            IREE_ARRAYSIZE(params), token);
  }

  iree_status_t status =
      loom_module_add_encoding(parser->module, &encoding, out_encoding_id);
  loom_parser_release_encoding_params(parser, params_scratch);
  return status;
}

//===----------------------------------------------------------------------===//
// Composite type list parsing
//===----------------------------------------------------------------------===//

// Allocates a fresh parser-owned FAM type list with at least
// |minimum_capacity| elements.
static iree_status_t loom_parser_allocate_type_list(
    loom_parser_t* parser, iree_host_size_t minimum_capacity,
    loom_parser_type_list_t** out_list) {
  iree_host_size_t capacity = LOOM_PARSER_TYPE_LIST_MIN_CAPACITY;
  if (capacity < minimum_capacity) capacity = minimum_capacity;

  iree_host_size_t alloc_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(loom_parser_type_list_t), &alloc_size,
                         IREE_STRUCT_FIELD_FAM(capacity, loom_type_t)));

  loom_parser_type_list_t* list = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&parser->parser_arena, alloc_size, (void**)&list));
  list->next_free = NULL;
  list->count = 0;
  list->capacity = capacity;
  *out_list = list;
  return iree_ok_status();
}

// Leases one parser-owned type-list frame for an active composite type parse.
// Reused frames keep their previous FAM capacity and only reset count.
static iree_status_t loom_parser_acquire_type_list(
    loom_parser_t* parser, loom_parser_type_list_t** out_list) {
  loom_parser_type_list_t* list = parser->type_list_free_list;
  if (list) {
    parser->type_list_free_list = list->next_free;
    list->next_free = NULL;
    list->count = 0;
    *out_list = list;
    return iree_ok_status();
  }
  return loom_parser_allocate_type_list(
      parser, LOOM_PARSER_TYPE_LIST_MIN_CAPACITY, out_list);
}

// Returns a type-list frame to parser->type_list_free_list while preserving
// its retained FAM capacity for later sibling parses.
static void loom_parser_release_type_list(loom_parser_t* parser,
                                          loom_parser_type_list_t* list) {
  if (!list) return;
  list->next_free = parser->type_list_free_list;
  parser->type_list_free_list = list;
}

// Appends one type to a leased type-list frame, growing to a larger retained
// FAM allocation if this parse exceeds the current high-water capacity.
static iree_status_t loom_parser_type_list_append(
    loom_parser_t* parser, loom_parser_type_list_t** inout_list,
    loom_type_t type) {
  loom_parser_type_list_t* list = *inout_list;
  if (list->count >= list->capacity) {
    iree_host_size_t new_capacity = list->capacity * 2;
    if (new_capacity < list->count + 1) new_capacity = list->count + 1;

    loom_parser_type_list_t* new_list = NULL;
    IREE_RETURN_IF_ERROR(
        loom_parser_allocate_type_list(parser, new_capacity, &new_list));
    new_list->count = list->count;
    memcpy(new_list->types, list->types, list->count * sizeof(loom_type_t));
    loom_parser_release_type_list(parser, list);
    *inout_list = new_list;
    list = new_list;
  }

  list->types[list->count++] = type;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type reference resolution
//===----------------------------------------------------------------------===//

// Resolves an SSA value reference in a type context. |name_token| is
// an SSA_VALUE token whose text is the bare name (e.g., "M" for %M)
// and whose line/column point to the '%' sigil for diagnostics.
//
// If the name is already in scope, returns its value_id.
//
// In definition context (ARG mode), creates a new NONE-typed placeholder value
// and defines it in scope. The surrounding Scope(...) format element controls
// that mode for signatures/globals, so BODY-mode tied result annotations still
// reject unknown names even inside a signature scope. Whether an inferred use
// resolves that placeholder immediately or waits for a later explicit
// `%name: type` binder is a property of the active declaration scope.
//
// In body context (BODY mode), emits PARSE/001 for undefined names.
static iree_status_t loom_resolve_type_reference(
    loom_parser_t* parser, loom_token_t name_token, loom_type_parse_mode_t mode,
    loom_value_id_t* out_value_id) {
  loom_string_id_t name_id =
      loom_module_lookup_string(parser->module, name_token.text);
  loom_value_id_t value_id = loom_parser_lookup_value(parser, name_id);
  if (value_id != LOOM_VALUE_ID_INVALID) {
    *out_value_id = value_id;
    return iree_ok_status();
  }

  if (mode == LOOM_TYPE_PARSE_ARG) {
    // Forward reference in definition context — create placeholder.
    IREE_RETURN_IF_ERROR(
        loom_module_define_value(parser->module, loom_type_none(), &value_id));
    IREE_RETURN_IF_ERROR(
        loom_parser_define_value_name(parser, name_token, value_id));
    IREE_RETURN_IF_ERROR(
        loom_parser_add_unresolved_placeholder(parser, value_id, name_token));
    *out_value_id = value_id;
    return iree_ok_status();
  }

  // Body mode: undefined name is an error.
  return loom_parser_resolve_value(parser, name_token, out_value_id);
}

// Resolves a type-binding SSA reference and bails out of the enclosing parse
// function if BODY-mode resolution emits a parser diagnostic.
#define LOOM_PARSE_RESOLVE_TYPE_REFERENCE(parser, name_token, mode,            \
                                          out_value_id)                        \
  do {                                                                         \
    uint32_t _resolve_errors = (parser)->error_count;                          \
    IREE_RETURN_IF_ERROR(loom_resolve_type_reference((parser), (name_token),   \
                                                     (mode), (out_value_id))); \
    if ((parser)->error_count > _resolve_errors) return iree_ok_status();      \
  } while (0)

// Returns the unresolved-placeholder record for a declaration-local binding
// value, or NULL if `value_id` was not created by ARG-mode parsing in the
// active Scope(...).
static loom_parser_unresolved_placeholder_t*
loom_type_binding_lookup_placeholder(loom_parser_t* parser,
                                     loom_value_id_t value_id) {
  if (!loom_parser_in_definition_scope(parser)) return NULL;
  for (iree_host_size_t i = parser->definition_scope.placeholder_start;
       i < parser->unresolved_placeholders.count; ++i) {
    loom_parser_unresolved_placeholder_t* placeholder =
        &parser->unresolved_placeholders.entries[i];
    if (placeholder->value_id == value_id) {
      return placeholder;
    }
  }
  return NULL;
}

// Assigns `binding_type` to one dim/encoding binding value when that value is
// still NONE-typed. Declaration-local placeholders follow the active
// Scope(...) policy: globals resolve by first inferred use, while func-like
// signatures wait for a later explicit binder.
static iree_status_t loom_assign_type_binding_value_type(
    loom_parser_t* parser, loom_value_id_t value_id, loom_type_t binding_type) {
  loom_parser_unresolved_placeholder_t* placeholder =
      loom_type_binding_lookup_placeholder(parser, value_id);
  loom_value_t* value = &parser->module->values.entries[value_id];

  if (placeholder) {
    if (!placeholder->resolved &&
        !iree_any_bit_set(
            parser->definition_scope.flags,
            LOOM_PARSER_DEFINITION_SCOPE_FLAG_RESOLVE_PLACEHOLDERS_FROM_USE)) {
      return iree_ok_status();
    }
    if (loom_type_kind(value->type) != LOOM_TYPE_NONE &&
        !loom_type_equal(value->type, binding_type)) {
      return loom_parser_emit_duplicate_value_name(parser,
                                                   placeholder->name_token);
    }
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(parser->module, value_id, binding_type));
    placeholder->resolved = true;
    return iree_ok_status();
  }

  if (loom_type_kind(value->type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(parser->module, value_id, binding_type));
  }
  return iree_ok_status();
}

// Assigns types to binding values referenced by a parsed type. Dim binding
// values get index type, and SSA encoding bindings get encoding type.
static iree_status_t loom_assign_type_binding_types(loom_parser_t* parser,
                                                    loom_type_t type) {
  // Dim bindings.
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) {
      loom_value_id_t value_id = loom_type_dim_value_id_at(type, i);
      IREE_RETURN_IF_ERROR(loom_assign_type_binding_value_type(
          parser, value_id, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
    }
  }

  // Encoding binding (SSA encoding).
  if (loom_type_has_ssa_encoding(type)) {
    loom_value_id_t enc_id = loom_type_encoding_value_id(type);
    IREE_RETURN_IF_ERROR(loom_assign_type_binding_value_type(
        parser, enc_id, loom_type_encoding()));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dimension parsing
//===----------------------------------------------------------------------===//

// Parses a single dimension: INTEGER (static) or [ SSA_VALUE ] (dynamic).
// Returns the packed dim value. The caller has already peeked the token
// and confirmed it is INTEGER or LBRACKET.
static iree_status_t loom_parse_dim(loom_parser_t* parser,
                                    loom_type_parse_mode_t mode,
                                    uint64_t* out_dim) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_INTEGER) {
    loom_tokenizer_next(&parser->tokenizer);
    int64_t size = 0;
    if (!iree_string_view_atoi_int64(token.text, &size) || size < 0) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(token.text),
      };
      return loom_parser_emit(parser, LOOM_ERR_PARSE_015, params,
                              IREE_ARRAYSIZE(params), token);
    }
    *out_dim = loom_dim_pack_static(size);
    return iree_ok_status();
  }
  // [SSA_VALUE]
  loom_tokenizer_next(&parser->tokenizer);  // consume [
  loom_token_t name_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &name_token);
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_TYPE_REFERENCE(parser, name_token, mode, &value_id);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACKET, NULL);
  *out_dim = loom_dim_pack_dynamic(value_id);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type encoding parsing
//===----------------------------------------------------------------------===//

// Parses a type encoding after the comma in a shaped type interior.
// Called after COMMA has been consumed, with in_dim_list already false.
// Handles SSA encodings (`%enc`), static encoding aliases (`#enc`), and
// canonical family spellings (`#q8_0` or `#q8_0<block=32>`).
static iree_status_t loom_parse_type_encoding(
    loom_parser_t* parser, loom_type_parse_mode_t mode,
    uint16_t* out_encoding_id, loom_encoding_flags_t* out_encoding_flags) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_SSA_VALUE) {
    // SSA encoding: %enc.
    loom_tokenizer_next(&parser->tokenizer);
    loom_value_id_t enc_value_id = LOOM_VALUE_ID_INVALID;
    LOOM_PARSE_RESOLVE_TYPE_REFERENCE(parser, token, mode, &enc_value_id);
    *out_encoding_id = (uint16_t)enc_value_id;
    *out_encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return iree_ok_status();
  }

  if (token.kind == LOOM_TOKEN_HASH_ATTR) {
    IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
        parser, LOOM_STRING_ID_INVALID, out_encoding_id));
    *out_encoding_flags = 0;
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_008, params,
                          IREE_ARRAYSIZE(params), token);
}

//===----------------------------------------------------------------------===//
// Shaped type construction
//===----------------------------------------------------------------------===//

// Constructs, interns, and assigns binding types for a shaped type from
// its parsed components. Handles inline dims (rank 0-2) and overflow
// allocation (rank > 2).
static iree_status_t loom_intern_shaped_type(
    loom_parser_t* parser, loom_type_kind_t kind,
    loom_scalar_type_t element_type, const uint64_t* dims, uint8_t rank,
    uint16_t encoding_id, loom_encoding_flags_t encoding_flags,
    loom_type_t* out_type) {
  loom_type_t type = {0};
  if (rank == 0) {
    type = loom_type_shaped_0d(kind, element_type, encoding_id);
  } else if (rank == 1) {
    type = loom_type_shaped_1d(kind, element_type, dims[0], encoding_id);
  } else if (rank == 2) {
    type =
        loom_type_shaped_2d(kind, element_type, dims[0], dims[1], encoding_id);
  } else {
    loom_overflow_dim_t* overflow = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow));
    memcpy(overflow, dims, rank * sizeof(uint64_t));

    uint8_t flags = 0;
    bool all_static = true;
    for (uint8_t i = 0; i < rank; ++i) {
      if (loom_dim_is_dynamic(dims[i])) {
        all_static = false;
        break;
      }
    }
    if (all_static) flags |= LOOM_TYPE_FLAG_ALL_STATIC;

    type.header = loom_type_make_header(kind, element_type, rank, flags);
    type.encoding_id = encoding_id;
    type.encoding_flags = encoding_flags;
    type.dims[0] = (uint64_t)(uintptr_t)overflow;
    // Precompute hash for fast inequality rejection.
    uint64_t hash = 0;
    for (uint8_t i = 0; i < rank; ++i) hash = hash * 31 + dims[i];
    type.dims[1] = hash;
  }

  if (rank <= 2) {
    type.encoding_flags = encoding_flags;
  }

  IREE_RETURN_IF_ERROR(loom_module_intern_type(parser->module, type, out_type));
  return loom_assign_type_binding_types(parser, *out_type);
}

//===----------------------------------------------------------------------===//
// Shaped type parsing
//===----------------------------------------------------------------------===//

// Parses a shaped type (tile, tensor, vector, or view) from the token stream.
// Called after LANGLE has been consumed. Consumes tokens through RANGLE.
//
// Grammar: dim (x dim)* x element_type [, encoding] >
//
// The tokenizer's in_dim_list flag is managed here: set after the
// first dim is consumed, cleared before scanning what follows each
// DIM_X separator. This ensures element types and encoding params
// are always scanned in normal (non-dim) mode.
static iree_status_t loom_parse_shaped_type(loom_parser_t* parser,
                                            loom_type_kind_t kind,
                                            loom_type_parse_mode_t mode,
                                            loom_type_t* out_type) {
  uint64_t dims[16];
  uint8_t rank = 0;

  // Parse dimensions. in_dim_list must be true from the start so
  // that '0x' in '0xf32' is scanned as INTEGER(0) + DIM_X(x) +
  // BARE_IDENT(f32), not as hex INTEGER(0xf32).
  parser->tokenizer.in_dim_list = true;
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_INTEGER || token.kind == LOOM_TOKEN_LBRACKET) {
    IREE_RETURN_IF_ERROR(loom_parse_dim(parser, mode, &dims[rank++]));

    while (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_DIM_X)) {
      loom_tokenizer_next(&parser->tokenizer);  // consume 'x'
      parser->tokenizer.in_dim_list = false;

      token = loom_tokenizer_peek(&parser->tokenizer);
      if (token.kind != LOOM_TOKEN_INTEGER &&
          token.kind != LOOM_TOKEN_LBRACKET) {
        break;  // element type follows
      }
      if (rank >= 16) {
        return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                                 token);
      }
      parser->tokenizer.in_dim_list = true;
      IREE_RETURN_IF_ERROR(loom_parse_dim(parser, mode, &dims[rank++]));
    }
  } else {
    // Rank 0 — no dims. Clear in_dim_list before element type.
    parser->tokenizer.in_dim_list = false;
  }

  if (kind == LOOM_TYPE_VECTOR && rank == 0) {
    token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(
        parser, token, IREE_SV("vector types must have rank >= 1"));
  }

  // Parse element type.
  loom_token_t element_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &element_token);
  loom_scalar_type_t element_type = 0;
  if (!loom_scalar_type_parse(element_token.text, &element_type)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(element_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_007, params,
                            IREE_ARRAYSIZE(params), element_token);
  }

  // Parse optional encoding.
  uint16_t encoding_id = 0;
  loom_encoding_flags_t encoding_flags = 0;
  if (kind == LOOM_TYPE_VECTOR &&
      loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
    loom_token_t comma_token = loom_tokenizer_peek(&parser->tokenizer);
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV(
            "vector types must not carry encoding or layout attachments")),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_004, params,
                            IREE_ARRAYSIZE(params), comma_token);
  }
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
    IREE_RETURN_IF_ERROR(
        loom_parse_type_encoding(parser, mode, &encoding_id, &encoding_flags));
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  return loom_intern_shaped_type(parser, kind, element_type, dims, rank,
                                 encoding_id, encoding_flags, out_type);
}

//===----------------------------------------------------------------------===//
// Pool type parsing
//===----------------------------------------------------------------------===//

// Parses a pool type from the token stream. Called after LANGLE has
// been consumed. Consumes tokens through RANGLE. Pool types have a
// single dimension and no element type or encoding.
static iree_status_t loom_parse_pool_type(loom_parser_t* parser,
                                          loom_type_parse_mode_t mode,
                                          loom_type_t* out_type) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind != LOOM_TOKEN_INTEGER && token.kind != LOOM_TOKEN_LBRACKET) {
    return loom_parser_emit_unexpected_token(parser, token,
                                             IREE_SV("integer or '['"));
  }
  uint64_t dim = 0;
  IREE_RETURN_IF_ERROR(loom_parse_dim(parser, mode, &dim));
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  *out_type = loom_type_pool(dim);
  return loom_assign_type_binding_types(parser, *out_type);
}

//===----------------------------------------------------------------------===//
// Group type parsing
//===----------------------------------------------------------------------===//

// Parses a group type from the token stream. Called after LANGLE has
// been consumed. Consumes tokens through RANGLE.
static iree_status_t loom_parse_group_type(loom_parser_t* parser,
                                           loom_type_t* out_type) {
  loom_token_t scope_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &scope_token);
  loom_type_t type = {0};
  if (iree_string_view_equal(scope_token.text, IREE_SV("workgroup"))) {
    type.header = loom_type_make_raw_header(LOOM_TYPE_GROUP,
                                            LOOM_GROUP_SCOPE_WORKGROUP, 0, 0);
  } else if (iree_string_view_equal(scope_token.text, IREE_SV("subgroup"))) {
    type.header = loom_type_make_raw_header(LOOM_TYPE_GROUP,
                                            LOOM_GROUP_SCOPE_SUBGROUP, 0, 0);
  } else {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("group scope")),
        loom_param_string(scope_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                            IREE_ARRAYSIZE(params), scope_token);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
  *out_type = type;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Storage type parsing
//===----------------------------------------------------------------------===//

// Parses a storage type from the token stream. Called after LANGLE has
// been consumed. Consumes tokens through RANGLE.
static iree_status_t loom_parse_storage_type(loom_parser_t* parser,
                                             loom_type_t* out_type) {
  loom_token_t space_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &space_token);
  loom_storage_space_t space = LOOM_STORAGE_SPACE_COUNT_;
  if (!loom_storage_space_parse(space_token.text, &space)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("storage space")),
        loom_param_string(space_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                            IREE_ARRAYSIZE(params), space_token);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
  *out_type = loom_type_storage(space);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dialect type parsing
//===----------------------------------------------------------------------===//

// Parses the comma-separated parameter list in a parameterized dialect type.
// Called after '<' has been consumed and consumes through the closing '>'.
static iree_status_t loom_parse_dialect_type_params(
    loom_parser_t* parser, loom_type_parse_mode_t mode,
    loom_parser_type_list_t** inout_param_types) {
  uint32_t errors_before = parser->error_count;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if ((*inout_param_types)->count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if ((*inout_param_types)->count == UINT16_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }

    loom_type_t param_type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, mode, &param_type));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(
        loom_parser_type_list_append(parser, inout_param_types, param_type));
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  return iree_ok_status();
}

// Parses a dialect type (e.g., hal.buffer, vm.ref<i32>) from the
// token stream. Called after the OP_NAME token has been consumed.
static iree_status_t loom_parse_dialect_type(loom_parser_t* parser,
                                             loom_token_t name_token,
                                             loom_type_parse_mode_t mode,
                                             loom_type_t* out_type) {
  loom_string_id_t name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));

  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    *out_type = loom_type_dialect_opaque(name_id);
    return iree_ok_status();
  }

  // Parameterized: name<type, type, ...>.
  loom_tokenizer_next(&parser->tokenizer);  // consume '<'
  uint32_t errors_before = parser->error_count;
  loom_parser_type_list_t* param_types = NULL;
  IREE_RETURN_IF_ERROR(loom_parser_acquire_type_list(parser, &param_types));
  iree_status_t status =
      loom_parse_dialect_type_params(parser, mode, &param_types);
  if (parser->error_count > errors_before || !iree_status_is_ok(status)) {
    loom_parser_release_type_list(parser, param_types);
    return status;
  }

  if (param_types->count > 0) {
    loom_type_t type = loom_type_dialect(name_id, (uint16_t)param_types->count,
                                         param_types->types);
    status = loom_module_intern_type(parser->module, type, out_type);
  } else {
    loom_type_t type = loom_type_dialect_opaque(name_id);
    status = loom_module_intern_type(parser->module, type, out_type);
  }
  loom_parser_release_type_list(parser, param_types);
  return status;
}

//===----------------------------------------------------------------------===//
// Function type parsing
//===----------------------------------------------------------------------===//

// Parses a comma-separated type list terminated by RPAREN. Used for
// both argument and result type lists in function types. Caller has
// consumed the opening '('. This function consumes through ')'.
static iree_status_t loom_parse_type_list(loom_parser_t* parser,
                                          loom_type_parse_mode_t mode,
                                          loom_parser_type_list_t** inout_types,
                                          uint16_t* out_count) {
  iree_host_size_t start_count = (*inout_types)->count;
  uint32_t errors_before = parser->error_count;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if ((*inout_types)->count > start_count) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if ((*inout_types)->count - start_count == UINT16_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, LOOM_ERR_PARSE_004,
                                               peek);
    }
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, mode, &type));
    if (parser->error_count > errors_before) return iree_ok_status();
    IREE_RETURN_IF_ERROR(
        loom_parser_type_list_append(parser, inout_types, type));
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  *out_count = (uint16_t)((*inout_types)->count - start_count);
  return iree_ok_status();
}

// Parses a function type into |*inout_types| and interns the resulting
// signature. Called after the opening LPAREN has been consumed.
static iree_status_t loom_parse_function_type_impl(
    loom_parser_t* parser, loom_type_parse_mode_t mode,
    loom_parser_type_list_t** inout_types, loom_type_t* out_type) {
  uint32_t errors_before = parser->error_count;
  uint16_t arg_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_parse_type_list(parser, mode, inout_types, &arg_count));
  if (parser->error_count > errors_before) return iree_ok_status();

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_ARROW)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'->'"));
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_parse_type_list(parser, mode, inout_types, &result_count));
  if (parser->error_count > errors_before) return iree_ok_status();

  loom_type_t* arg_types = (*inout_types)->types;
  loom_type_t* result_types = arg_types + arg_count;
  return loom_module_intern_function_type(parser->module, arg_types, arg_count,
                                          result_types, result_count, out_type);
}

// Parses a function type: (arg_types) -> (result_types). Called after the
// opening LPAREN has been consumed.
static iree_status_t loom_parse_function_type(loom_parser_t* parser,
                                              loom_type_parse_mode_t mode,
                                              loom_type_t* out_type) {
  loom_parser_type_list_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_parser_acquire_type_list(parser, &types));
  iree_status_t status =
      loom_parse_function_type_impl(parser, mode, &types, out_type);
  loom_parser_release_type_list(parser, types);
  return status;
}

//===----------------------------------------------------------------------===//
// Type dispatch
//===----------------------------------------------------------------------===//

// Parses the optional role parameter on an encoding type. Called after the
// `encoding` keyword has been consumed.
static iree_status_t loom_parse_encoding_type(loom_parser_t* parser,
                                              loom_type_t* out_type) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    *out_type = loom_type_encoding();
    return iree_ok_status();
  }

  loom_token_t role_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &role_token);
  loom_encoding_role_t role = LOOM_ENCODING_ROLE_UNKNOWN;
  if (!loom_encoding_role_parse(role_token.text, &role)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("encoding role")),
        loom_param_string(role_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_018, params,
                            IREE_ARRAYSIZE(params), role_token);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
  *out_type = loom_type_encoding_with_role(role);
  return iree_ok_status();
}

static iree_status_t loom_parse_register_unit_suffix(loom_parser_t* parser,
                                                     loom_token_t suffix_token,
                                                     uint32_t* out_unit_count) {
  iree_string_view_t count_text = iree_string_view_empty();
  if (iree_string_view_equal(suffix_token.text, IREE_SV("x"))) {
    loom_token_t count_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &count_token);
    count_text = count_token.text;
    suffix_token = count_token;
  } else if (suffix_token.text.size >= 2 && suffix_token.text.data[0] == 'x') {
    count_text = iree_make_string_view(suffix_token.text.data + 1,
                                       suffix_token.text.size - 1);
  } else {
    return loom_parser_emit_unexpected_token(
        parser, suffix_token, IREE_SV("register unit suffix 'xN'"));
  }

  uint32_t unit_count = 0;
  if (!iree_string_view_atoi_uint32(count_text, &unit_count) ||
      unit_count == 0) {
    return loom_parser_emit_unexpected_token(
        parser, suffix_token, IREE_SV("positive register unit suffix 'xN'"));
  }
  *out_unit_count = unit_count;
  return iree_ok_status();
}

// Parses a target-low register type. Called after the `reg` keyword has been
// consumed. Grammar: reg<namespace.class> | reg<namespace.class xN>.
static iree_status_t loom_parse_register_type(loom_parser_t* parser,
                                              loom_type_t* out_type) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }

  loom_token_t class_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_OP_NAME, &class_token);
  if (!loom_register_class_name_is_qualified(class_token.text)) {
    return loom_parser_emit_unexpected_token(
        parser, class_token, IREE_SV("namespace-qualified register class"));
  }
  uint32_t unit_count = 1;
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t suffix_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &suffix_token);
    IREE_RETURN_IF_ERROR(
        loom_parse_register_unit_suffix(parser, suffix_token, &unit_count));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  if (!parser->low_register_descriptor_set ||
      !parser->low_asm_environment.vtable ||
      !parser->low_asm_environment.vtable->resolve_register_type) {
    return loom_parser_emit_unexpected_token(
        parser, class_token,
        IREE_SV("register class in a target-low descriptor context"));
  }
  loom_type_t type = loom_type_none();
  bool found = false;
  IREE_RETURN_IF_ERROR(
      parser->low_asm_environment.vtable->resolve_register_type(
          parser->low_asm_environment.state,
          parser->low_register_descriptor_set, class_token.text, unit_count,
          &type, &found));
  if (!found) {
    return loom_parser_emit_unexpected_token(
        parser, class_token,
        IREE_SV("register class defined by the selected descriptor set"));
  }
  return loom_module_intern_type(parser->module, type, out_type);
}

// Consumes keyword, expects LANGLE, dispatches to the type-specific
// parser. Shared entry for tile, tensor, vector, view, pool, and group.
static iree_status_t loom_parse_angle_bracketed_type(
    loom_parser_t* parser, loom_type_kind_t kind, loom_type_parse_mode_t mode,
    loom_type_t* out_type) {
  loom_tokenizer_next(&parser->tokenizer);  // consume type keyword
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  iree_status_t status;
  if (kind == LOOM_TYPE_POOL) {
    status = loom_parse_pool_type(parser, mode, out_type);
  } else if (kind == LOOM_TYPE_GROUP) {
    status = loom_parse_group_type(parser, out_type);
  } else if (kind == LOOM_TYPE_STORAGE) {
    status = loom_parse_storage_type(parser, out_type);
  } else {
    status = loom_parse_shaped_type(parser, kind, mode, out_type);
  }
  parser->tokenizer.in_dim_list = false;  // cleanup on error paths
  return status;
}

static iree_status_t loom_parse_registered_type(loom_parser_t* parser,
                                                loom_token_t token,
                                                loom_type_parse_mode_t mode,
                                                loom_type_t* out_type,
                                                bool* out_matched) {
  *out_matched = false;
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_lookup(token.text);
  if (!descriptor) return iree_ok_status();
  *out_matched = true;

  switch (descriptor->ir_kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
    case LOOM_TYPE_POOL:
    case LOOM_TYPE_GROUP:
    case LOOM_TYPE_STORAGE:
      return loom_parse_angle_bracketed_type(parser, descriptor->ir_kind, mode,
                                             out_type);
    case LOOM_TYPE_BUFFER:
      loom_tokenizer_next(&parser->tokenizer);
      *out_type = loom_type_buffer();
      return iree_ok_status();
    case LOOM_TYPE_DIALECT:
      loom_tokenizer_next(&parser->tokenizer);
      return loom_parse_dialect_type(parser, token, mode, out_type);
    default:
      break;
  }
  return loom_parser_emit_unexpected_token(parser, token, IREE_SV("a type"));
}

iree_status_t loom_parse_type(loom_parser_t* parser,
                              loom_type_parse_mode_t mode,
                              loom_type_t* out_type) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);

  if (token.kind == LOOM_TOKEN_BARE_IDENT) {
    // Scalar type keyword (f32, i8, index, ...).
    loom_scalar_type_t scalar_type = 0;
    if (loom_scalar_type_parse(token.text, &scalar_type)) {
      loom_tokenizer_next(&parser->tokenizer);
      *out_type = loom_type_scalar(scalar_type);
      return iree_ok_status();
    }

    // "encoding" keyword.
    if (iree_string_view_equal(token.text, IREE_SV("encoding"))) {
      loom_tokenizer_next(&parser->tokenizer);
      return loom_parse_encoding_type(parser, out_type);
    }

    // "reg" keyword.
    if (iree_string_view_equal(token.text, IREE_SV("reg"))) {
      loom_tokenizer_next(&parser->tokenizer);
      return loom_parse_register_type(parser, out_type);
    }

    bool matched_registered_type = false;
    IREE_RETURN_IF_ERROR(loom_parse_registered_type(
        parser, token, mode, out_type, &matched_registered_type));
    if (matched_registered_type) {
      return iree_ok_status();
    }

    // Unknown bare ident.
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_007, params,
                            IREE_ARRAYSIZE(params), token);
  }

  // Dialect type: dotted name (e.g., hal.buffer).
  if (token.kind == LOOM_TOKEN_OP_NAME) {
    bool matched_registered_type = false;
    IREE_RETURN_IF_ERROR(loom_parse_registered_type(
        parser, token, mode, out_type, &matched_registered_type));
    if (matched_registered_type) {
      return iree_ok_status();
    }

    loom_tokenizer_next(&parser->tokenizer);
    return loom_parse_dialect_type(parser, token, mode, out_type);
  }

  // Function type: (types) -> (types).
  if (token.kind == LOOM_TOKEN_LPAREN) {
    loom_tokenizer_next(&parser->tokenizer);  // consume '('
    return loom_parse_function_type(parser, mode, out_type);
  }

  return loom_parser_emit_unexpected_token(parser, token, IREE_SV("a type"));
}
