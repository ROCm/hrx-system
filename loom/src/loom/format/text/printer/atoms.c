// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/atoms.h"

#include <inttypes.h>
#include <math.h>

#include "iree/base/internal/unicode.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"

static const loom_op_vtable_t* loom_print_value_defining_op_vtable(
    const loom_module_t* module, const loom_op_t* op) {
  if (!module || !module->context || !op) {
    return NULL;
  }
  return loom_context_resolve_op(module->context, op->kind);
}

static bool loom_print_value_has_name(const loom_module_t* module,
                                      loom_value_id_t value_id,
                                      loom_string_id_t* out_name_id) {
  *out_name_id = LOOM_STRING_ID_INVALID;
  if (!module || value_id >= module->values.count) {
    return false;
  }
  loom_string_id_t name_id = module->values.entries[value_id].name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return false;
  }
  *out_name_id = name_id;
  return true;
}

// Returns the parser scope that will receive |value_id|'s printed definition.
// Sibling regions and symbol signatures have independent SSA namespaces, while
// values defined inside one region body share that region's namespace.
static const void* loom_print_value_parse_scope(const loom_module_t* module,
                                                loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) {
    return NULL;
  }
  const loom_value_t* value = &module->values.entries[value_id];
  if (loom_value_is_block_arg(value)) {
    loom_block_t* block = loom_value_def_block(value);
    return block ? (const void*)block->parent_region : NULL;
  }

  loom_op_t* def_op = loom_value_def_op(value);
  if (def_op) {
    const loom_op_vtable_t* vtable =
        loom_print_value_defining_op_vtable(module, def_op);
    if (vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
      return (const void*)def_op;
    }
    loom_block_t* block = def_op->parent_block;
    return block ? (const void*)block->parent_region : NULL;
  }

  const loom_use_t* uses = loom_value_uses(value);
  if (!uses || value->use_count == 0) {
    return NULL;
  }
  loom_op_t* user_op = loom_use_user_op(uses[0]);
  const loom_op_vtable_t* user_vtable =
      loom_print_value_defining_op_vtable(module, user_op);
  if (user_vtable &&
      iree_any_bit_set(user_vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return (const void*)user_op;
  }
  loom_block_t* block = user_op ? user_op->parent_block : NULL;
  return block ? (const void*)block->parent_region : NULL;
}

static bool loom_print_value_is_printable(const loom_module_t* module,
                                          loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = &module->values.entries[value_id];
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value) != NULL;
  }
  loom_op_t* def_op = loom_value_def_op(value);
  if (def_op) {
    return !iree_any_bit_set(def_op->flags, LOOM_OP_FLAG_DEAD);
  }
  return value->use_count > 0;
}

static bool loom_print_value_name_is_duplicated(const loom_module_t* module,
                                                loom_value_id_t value_id,
                                                loom_string_id_t name_id) {
  const void* scope = loom_print_value_parse_scope(module, value_id);
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    if (i == value_id) {
      continue;
    }
    if (!loom_print_value_is_printable(module, (loom_value_id_t)i)) {
      continue;
    }
    if (loom_print_value_parse_scope(module, (loom_value_id_t)i) != scope) {
      continue;
    }
    if (module->values.entries[i].name_id == name_id) {
      return true;
    }
  }
  return false;
}

static bool loom_print_explicit_value_name_exists(const loom_module_t* module,
                                                  const void* scope,
                                                  iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    if (!loom_print_value_is_printable(module, (loom_value_id_t)i)) {
      continue;
    }
    if (loom_print_value_parse_scope(module, (loom_value_id_t)i) != scope) {
      continue;
    }
    loom_string_id_t name_id = module->values.entries[i].name_id;
    if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
      continue;
    }
    if (iree_string_view_equal(module->strings.entries[name_id], name)) {
      return true;
    }
  }
  return false;
}

// Emits a canonical JSON-compatible string literal. Stored strings are expected
// to contain decoded UTF-8 payload bytes; this helper validates that invariant
// before writing so malformed IR never serializes as malformed text.
static iree_status_t loom_print_string_literal(loom_output_stream_t* stream,
                                               iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  iree_host_size_t position = 0;
  while (position < text.size) {
    iree_host_size_t codepoint_start = position;
    uint32_t codepoint = iree_unicode_utf8_decode(text, &position);
    if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid UTF-8 string literal");
    }
    switch (codepoint) {
      case '"': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\\""));
        break;
      }
      case '\\': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\\\"));
        break;
      }
      case '\b': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\b"));
        break;
      }
      case '\f': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\f"));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\t"));
        break;
      }
      default: {
        if (codepoint < 0x20) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              stream, "\\u%04" PRIX32, codepoint));
        } else {
          IREE_RETURN_IF_ERROR(loom_output_stream_write(
              stream, iree_make_string_view(text.data + codepoint_start,
                                            position - codepoint_start)));
        }
        break;
      }
    }
  }
  return loom_output_stream_write_char(stream, '"');
}

static iree_string_view_t loom_print_resolve_generated_value_name(
    const loom_module_t* module, loom_value_id_t value_id, char* buffer,
    iree_host_size_t buffer_size) {
  const void* scope = loom_print_value_parse_scope(module, value_id);
  int length = iree_snprintf(buffer, buffer_size, "%%%" PRIhsz,
                             (iree_host_size_t)value_id);
  if (length > 1 && (iree_host_size_t)length < buffer_size) {
    iree_string_view_t candidate =
        iree_make_string_view(buffer + 1, (iree_host_size_t)length - 1);
    if (!loom_print_explicit_value_name_exists(module, scope, candidate)) {
      return iree_make_string_view(buffer, (iree_host_size_t)length);
    }
  }
  for (iree_host_size_t attempt = 1; attempt < 8; ++attempt) {
    length = iree_snprintf(buffer, buffer_size, "%%%" PRIhsz "$%" PRIhsz,
                           (iree_host_size_t)value_id, attempt);
    if (length <= 1 || (iree_host_size_t)length >= buffer_size) {
      break;
    }
    iree_string_view_t candidate =
        iree_make_string_view(buffer + 1, (iree_host_size_t)length - 1);
    if (!loom_print_explicit_value_name_exists(module, scope, candidate)) {
      return iree_make_string_view(buffer, (iree_host_size_t)length);
    }
  }
  length = iree_snprintf(buffer, buffer_size, "%%%" PRIhsz,
                         (iree_host_size_t)value_id);
  IREE_ASSERT(length > 1 && (iree_host_size_t)length < buffer_size,
              "generated value names must fit LOOM_VALUE_NAME_BUFFER_SIZE");
  return iree_make_string_view(buffer, length);
}

static bool loom_print_explicit_value_name_with_suffix_exists(
    const loom_module_t* module, const void* scope, iree_string_view_t base,
    iree_string_view_t suffix) {
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    if (!loom_print_value_is_printable(module, (loom_value_id_t)i)) {
      continue;
    }
    if (loom_print_value_parse_scope(module, (loom_value_id_t)i) != scope) {
      continue;
    }
    loom_string_id_t name_id = module->values.entries[i].name_id;
    if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
      continue;
    }
    iree_string_view_t name = module->strings.entries[name_id];
    if (name.size != base.size + suffix.size) {
      continue;
    }
    if (!iree_string_view_starts_with(name, base)) {
      continue;
    }
    iree_string_view_t name_suffix =
        iree_string_view_substr(name, base.size, suffix.size);
    if (iree_string_view_equal(name_suffix, suffix)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_print_duplicate_value_ref(
    loom_output_stream_t* stream, const loom_module_t* module,
    loom_value_id_t value_id, iree_string_view_t bare_name) {
  const void* scope = loom_print_value_parse_scope(module, value_id);
  for (iree_host_size_t attempt = 0; attempt < 8; ++attempt) {
    char suffix_buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
    int suffix_length =
        attempt == 0 ? iree_snprintf(suffix_buffer, sizeof(suffix_buffer),
                                     "$%" PRIhsz, (iree_host_size_t)value_id)
                     : iree_snprintf(suffix_buffer, sizeof(suffix_buffer),
                                     "$%" PRIhsz "$%" PRIhsz,
                                     (iree_host_size_t)value_id, attempt);
    if (suffix_length <= 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix_buffer)) {
      break;
    }
    iree_string_view_t suffix =
        iree_make_string_view(suffix_buffer, (iree_host_size_t)suffix_length);
    if (loom_print_explicit_value_name_with_suffix_exists(module, scope,
                                                          bare_name, suffix)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, bare_name));
    return loom_output_stream_write(stream, suffix);
  }
  char buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
  return loom_output_stream_write(
      stream, loom_print_resolve_generated_value_name(module, value_id, buffer,
                                                      sizeof(buffer)));
}

iree_status_t loom_print_value_ref(loom_output_stream_t* stream,
                                   const loom_module_t* module,
                                   loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) {
    return loom_output_stream_write_cstring(stream, "%?");
  }
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  if (loom_print_value_has_name(module, value_id, &name_id)) {
    iree_string_view_t bare_name = module->strings.entries[name_id];
    if (!loom_print_value_name_is_duplicated(module, value_id, name_id)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
      return loom_output_stream_write(stream, bare_name);
    }
    return loom_print_duplicate_value_ref(stream, module, value_id, bare_name);
  }
  char buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
  return loom_output_stream_write(
      stream, loom_print_resolve_generated_value_name(module, value_id, buffer,
                                                      sizeof(buffer)));
}

iree_status_t loom_print_value_name(loom_print_context_t* ctx,
                                    loom_value_id_t value_id) {
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_value_ref(ctx->stream, ctx->module, value_id));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

iree_status_t loom_print_value_name_with_field(
    loom_print_context_t* ctx, loom_value_id_t value_id,
    loom_print_field_ref_t field_ref) {
  iree_host_size_t start = loom_print_next_token_start_offset(ctx, false, '%');
  IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, value_id));
  iree_host_size_t end = ctx->stream->offset;
  loom_print_report_field(ctx, field_ref, start, end);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type printing.
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_scalar_type(loom_output_stream_t* stream,
                                            loom_scalar_type_t scalar) {
  const char* name = loom_scalar_type_name(scalar);
  if (!name) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown scalar type %d", (int)scalar);
  }
  return loom_output_stream_write_cstring(stream, name);
}

static iree_status_t loom_print_encoding_type(loom_output_stream_t* stream,
                                              loom_type_t type) {
  loom_encoding_role_t role = loom_type_encoding_role(type);
  if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
    return loom_output_stream_write_cstring(stream, "encoding");
  }
  const char* role_name = loom_encoding_role_name(role);
  if (!role_name) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown encoding role %d", (int)role);
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "encoding<"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, role_name));
  return loom_output_stream_write_char(stream, '>');
}

static bool loom_print_is_bare_identifier_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         c == '$';
}

static bool loom_print_is_bare_identifier_continue(char c) {
  return loom_print_is_bare_identifier_start(c) || (c >= '0' && c <= '9') ||
         c == '-';
}

static bool loom_print_is_bare_string_attr(iree_string_view_t text) {
  if (text.size == 0 || !loom_print_is_bare_identifier_start(text.data[0])) {
    return false;
  }
  for (iree_host_size_t i = 1; i < text.size; ++i) {
    if (!loom_print_is_bare_identifier_continue(text.data[i])) return false;
  }
  return true;
}

static bool loom_print_matrix_operand_symbol_param_name(
    iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV("element_format")) ||
         iree_string_view_equal(name, IREE_SV("payload_packing")) ||
         iree_string_view_equal(name, IREE_SV("scale_topology")) ||
         iree_string_view_equal(name, IREE_SV("scale_format")) ||
         iree_string_view_equal(name, IREE_SV("secondary_scale_format")) ||
         iree_string_view_equal(name, IREE_SV("affine")) ||
         iree_string_view_equal(name, IREE_SV("rounding")) ||
         iree_string_view_equal(name, IREE_SV("codebook")) ||
         iree_string_view_equal(name, IREE_SV("sparsity"));
}

static bool loom_print_static_encoding_param_as_bare_symbol(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_named_attr_t* param, iree_string_view_t* out_symbol) {
  *out_symbol = iree_string_view_empty();
  if (param->value.kind != LOOM_ATTR_STRING ||
      param->value.string_id == LOOM_STRING_ID_INVALID ||
      param->value.string_id >= module->strings.count ||
      encoding->name_id >= module->strings.count ||
      param->name_id >= module->strings.count) {
    return false;
  }
  if (!iree_string_view_equal(module->strings.entries[encoding->name_id],
                              IREE_SV("matrix_operand")) ||
      !loom_print_matrix_operand_symbol_param_name(
          module->strings.entries[param->name_id])) {
    return false;
  }
  iree_string_view_t symbol = module->strings.entries[param->value.string_id];
  if (!loom_print_is_bare_string_attr(symbol)) return false;
  *out_symbol = symbol;
  return true;
}

static iree_status_t loom_print_canonical_encoding(
    loom_output_stream_t* stream, const loom_module_t* module,
    const loom_encoding_t* encoding) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
  if (encoding->name_id < module->strings.count) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, module->strings.entries[encoding->name_id]));
  }
  if (encoding->attribute_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    const loom_named_attr_t* param = &encoding->attributes[i];
    if (param->name_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, module->strings.entries[param->name_id]));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '='));
    iree_string_view_t bare_symbol = iree_string_view_empty();
    if (loom_print_static_encoding_param_as_bare_symbol(module, encoding, param,
                                                        &bare_symbol)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, bare_symbol));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_print_attr(stream, &param->value, module, NULL));
    }
  }
  return loom_output_stream_write_char(stream, '>');
}

static iree_status_t loom_print_static_encoding(loom_output_stream_t* stream,
                                                const loom_module_t* module,
                                                uint16_t encoding_id) {
  if (module && encoding_id > 0 && encoding_id <= module->encodings.count) {
    const loom_encoding_t* encoding =
        &module->encodings.entries[encoding_id - 1];
    if (encoding->alias_id != LOOM_STRING_ID_INVALID &&
        encoding->alias_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
      return loom_output_stream_write(
          stream, module->strings.entries[encoding->alias_id]);
    }
    return loom_print_canonical_encoding(stream, module, encoding);
  }

  return loom_output_stream_write_format(stream, "#encoding_%" PRIu16,
                                         encoding_id);
}

static iree_status_t loom_print_dim(loom_output_stream_t* stream,
                                    loom_type_t type,
                                    iree_host_size_t dim_index,
                                    const loom_module_t* module) {
  uint64_t packed = loom_type_dim(type, dim_index);
  if (loom_dim_is_dynamic(packed)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
    IREE_RETURN_IF_ERROR(
        loom_print_value_ref(stream, module, loom_dim_value_id(packed)));
    return loom_output_stream_write_char(stream, ']');
  }
  return loom_output_stream_write_format(stream, "%" PRId64,
                                         loom_dim_static_size(packed));
}

static iree_status_t loom_print_shaped_interior(loom_output_stream_t* stream,
                                                loom_type_t type,
                                                const loom_module_t* module) {
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
    }
    IREE_RETURN_IF_ERROR(loom_print_dim(stream, type, i, module));
  }
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
  }
  IREE_RETURN_IF_ERROR(
      loom_print_scalar_type(stream, loom_type_element_type(type)));
  if (loom_type_has_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    if (loom_type_has_ssa_encoding(type)) {
      IREE_RETURN_IF_ERROR(loom_print_value_ref(
          stream, module, loom_type_encoding_value_id(type)));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_print_static_encoding(stream, module, type.encoding_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_shaped_type_prefix(loom_output_stream_t* stream,
                                                   loom_type_kind_t kind) {
  switch (kind) {
    case LOOM_TYPE_TILE:
      return loom_output_stream_write_cstring(stream, "tile<");
    case LOOM_TYPE_TENSOR:
      return loom_output_stream_write_cstring(stream, "tensor<");
    case LOOM_TYPE_VECTOR:
      return loom_output_stream_write_cstring(stream, "vector<");
    case LOOM_TYPE_VIEW:
      return loom_output_stream_write_cstring(stream, "view<");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "type kind %d is not shaped", (int)kind);
  }
}

static iree_status_t loom_text_print_type_impl(
    loom_type_t type, const loom_module_t* module, loom_output_stream_t* stream,
    const loom_print_context_t* ctx) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_NONE:
      return loom_output_stream_write_cstring(stream, "none");
    case LOOM_TYPE_SCALAR:
      return loom_print_scalar_type(stream, loom_type_element_type(type));
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      IREE_RETURN_IF_ERROR(
          loom_print_shaped_type_prefix(stream, loom_type_kind(type)));
      IREE_RETURN_IF_ERROR(loom_print_shaped_interior(stream, type, module));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_GROUP: {
      const char* scope_name =
          loom_group_scope_name(loom_type_group_scope(type));
      if (!scope_name) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown group scope %d",
                                (int)loom_type_group_scope(type));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "group<"));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, scope_name));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_DIALECT: {
      loom_string_id_t name_id = loom_type_dialect_name_id(type);
      if (module && name_id < module->strings.count) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write(stream, module->strings.entries[name_id]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "?dialect"));
      }
      uint16_t param_count = loom_type_dialect_param_count(type);
      if (param_count > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));
        const loom_type_t* params = loom_type_dialect_params(type);
        for (uint16_t i = 0; i < param_count; ++i) {
          if (i > 0) {
            IREE_RETURN_IF_ERROR(
                loom_output_stream_write_cstring(stream, ", "));
          }
          IREE_RETURN_IF_ERROR(
              loom_text_print_type_impl(params[i], module, stream, ctx));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '>'));
      }
      return iree_ok_status();
    }
    case LOOM_TYPE_REGISTER: {
      if (!ctx) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "reg<"));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, "0x%" PRIx64 ":%" PRIu16,
            loom_low_register_type_descriptor_set_stable_id(type),
            loom_low_register_type_class_id(type)));
        uint32_t unit_count = loom_low_register_type_unit_count(type);
        if (unit_count != 1) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " x"));
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_format(stream, "%u", unit_count));
        }
        return loom_output_stream_write_cstring(stream, ">");
      }
      const loom_text_low_asm_descriptor_set_t* descriptor_set =
          ctx->low_register_descriptor_set;
      if (!descriptor_set && ctx->low_asm_environment.vtable &&
          ctx->low_asm_environment.vtable->lookup_register_descriptor_set) {
        IREE_RETURN_IF_ERROR(
            ctx->low_asm_environment.vtable->lookup_register_descriptor_set(
                ctx->low_asm_environment.state, type, &descriptor_set));
      }
      if (!descriptor_set || !ctx->low_asm_environment.vtable ||
          !ctx->low_asm_environment.vtable->describe_register_type) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "printing reg<...> requires a target-low descriptor context");
      }
      iree_string_view_t register_class_name = iree_string_view_empty();
      uint32_t unit_count = 0;
      bool found = false;
      IREE_RETURN_IF_ERROR(
          ctx->low_asm_environment.vtable->describe_register_type(
              ctx->low_asm_environment.state, descriptor_set, type,
              &register_class_name, &unit_count, &found));
      if (!found) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "reg<...> type does not belong to the selected descriptor set");
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "reg<"));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write(stream, register_class_name));
      if (unit_count != 1) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " x"));
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_format(stream, "%u", unit_count));
      }
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_STORAGE: {
      const char* space_name =
          loom_storage_space_name(loom_type_storage_space(type));
      if (!space_name) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown storage space %d",
                                (int)loom_type_storage_space(type));
      }
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "low.storage<"));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, space_name));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_ENCODING:
      return loom_print_encoding_type(stream, type);
    case LOOM_TYPE_BUFFER:
      return loom_output_stream_write_cstring(stream, "buffer");
    case LOOM_TYPE_POOL: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "pool<"));
      IREE_RETURN_IF_ERROR(loom_print_dim(stream, type, 0, module));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* func_data = loom_type_func_data(type);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '('));
      for (uint16_t i = 0; i < func_data->arg_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_text_print_type_impl(func_data->types[i],
                                                       module, stream, ctx));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ") -> ("));
      for (uint16_t i = 0; i < func_data->result_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_text_print_type_impl(
            func_data->types[func_data->arg_count + i], module, stream, ctx));
      }
      return loom_output_stream_write_char(stream, ')');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown type kind %d",
                              (int)loom_type_kind(type));
  }
}

iree_status_t loom_text_print_type(loom_type_t type,
                                   const loom_module_t* module,
                                   loom_output_stream_t* stream) {
  return loom_text_print_type_impl(type, module, stream, NULL);
}

iree_status_t loom_text_print_type_with_options(
    loom_type_t type, const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options) {
  loom_print_context_t ctx = {
      .stream = stream,
      .module = module,
      .flags = options ? options->flags : LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = options ? options->low_asm_environment
                                     : (loom_text_low_asm_environment_t){0},
      .low_asm_descriptor_set_key = options
                                        ? options->low_asm_descriptor_set_key
                                        : iree_string_view_empty(),
  };
  if (!iree_string_view_is_empty(ctx.low_asm_descriptor_set_key) &&
      ctx.low_asm_environment.vtable &&
      ctx.low_asm_environment.vtable->lookup_descriptor_set) {
    IREE_RETURN_IF_ERROR(ctx.low_asm_environment.vtable->lookup_descriptor_set(
        ctx.low_asm_environment.state, ctx.low_asm_descriptor_set_key,
        &ctx.low_register_descriptor_set));
  }
  return loom_text_print_type_impl(type, module, stream, &ctx);
}

iree_status_t loom_print_type(loom_print_context_t* ctx, loom_type_t type) {
  return loom_text_print_type_impl(type, ctx->module, ctx->stream, ctx);
}

static iree_status_t loom_print_result_type_dim(loom_output_stream_t* stream,
                                                loom_type_t type,
                                                iree_host_size_t dim_index,
                                                const loom_module_t* module) {
  return loom_print_dim(stream, type, dim_index, module);
}

static iree_status_t loom_text_print_result_type(loom_type_t type,
                                                 const loom_module_t* module,
                                                 loom_output_stream_t* stream,
                                                 loom_print_context_t* ctx) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      IREE_RETURN_IF_ERROR(
          loom_print_shaped_type_prefix(stream, loom_type_kind(type)));
      uint8_t rank = loom_type_rank(type);
      for (uint8_t i = 0; i < rank; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
        }
        IREE_RETURN_IF_ERROR(
            loom_print_result_type_dim(stream, type, i, module));
      }
      if (rank > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
      }
      IREE_RETURN_IF_ERROR(
          loom_print_scalar_type(stream, loom_type_element_type(type)));
      if (loom_type_has_encoding(type)) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        if (loom_type_has_ssa_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_print_value_ref(
              stream, module, loom_type_encoding_value_id(type)));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_print_static_encoding(stream, module, type.encoding_id));
        }
      }
      return loom_output_stream_write_cstring(stream, ">");
    }
    default:
      return loom_text_print_type_impl(type, module, stream, ctx);
  }
}

iree_status_t loom_print_value_type(loom_print_context_t* ctx,
                                    loom_value_id_t value_id) {
  if (value_id < ctx->module->values.count) {
    return loom_print_type(ctx, ctx->module->values.entries[value_id].type);
  }
  return loom_output_stream_write_cstring(ctx->stream, "<unknown>");
}

iree_status_t loom_print_result_value_type(loom_print_context_t* ctx,
                                           loom_value_id_t value_id) {
  if (value_id < ctx->module->values.count) {
    return loom_text_print_result_type(
        ctx->module->values.entries[value_id].type, ctx->module, ctx->stream,
        ctx);
  }
  return loom_output_stream_write_cstring(ctx->stream, "<unknown>");
}

//===----------------------------------------------------------------------===//
// Location printing.
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_location_source(
    const loom_module_t* module, loom_source_id_t source_id,
    iree_string_view_t* out_source) {
  if (source_id >= module->sources.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location source_id %u out of range", source_id);
  }
  *out_source = module->sources.entries[source_id];
  return iree_ok_status();
}

static iree_status_t loom_print_location_body(loom_output_stream_t* stream,
                                              const loom_module_t* module,
                                              loom_location_id_t location_id) {
  if (location_id >= module->locations.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location_id %u out of range (module has %" PRIhsz
                            " locations)",
                            location_id, module->locations.count);
  }
  const loom_location_entry_t* entry = &module->locations.entries[location_id];
  switch (entry->kind) {
    case LOOM_LOCATION_NONE:
      return iree_ok_status();
    case LOOM_LOCATION_FILE: {
      iree_string_view_t source = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_print_location_source(module, entry->file.source_id, &source));
      IREE_RETURN_IF_ERROR(loom_print_string_literal(stream, source));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ":%u:%u", entry->file.start_line, entry->file.start_col));
      if (entry->file.end_line != entry->file.start_line ||
          entry->file.end_col != entry->file.start_col) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, " to %u:%u", entry->file.end_line, entry->file.end_col));
      }
      return iree_ok_status();
    }
    case LOOM_LOCATION_FUSED: {
      if (entry->fused.count > 0 && !entry->fused.children) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "fused location has count %u but NULL children",
                                entry->fused.count);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "fused<"));
      for (uint32_t i = 0; i < entry->fused.count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_print_location_body(stream, module, entry->fused.children[i]));
      }
      return loom_output_stream_write_char(stream, '>');
    }
    case LOOM_LOCATION_OPAQUE: {
      iree_string_view_t tag = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_print_location_source(module, entry->opaque.source_id, &tag));
      if (entry->opaque.data_length > 0 && !entry->opaque.data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "opaque location has data_length %u but NULL data",
            entry->opaque.data_length);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "opaque<"));
      IREE_RETURN_IF_ERROR(loom_print_string_literal(stream, tag));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_print_string_literal(
          stream, iree_make_string_view((const char*)entry->opaque.data,
                                        entry->opaque.data_length)));
      return loom_output_stream_write_char(stream, '>');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown location kind %d", (int)entry->kind);
  }
}

iree_status_t loom_print_location(loom_output_stream_t* stream,
                                  const loom_module_t* module,
                                  loom_location_id_t location_id) {
  if (location_id == LOOM_LOCATION_UNKNOWN) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " loc("));
  IREE_RETURN_IF_ERROR(loom_print_location_body(stream, module, location_id));
  return loom_output_stream_write_char(stream, ')');
}

//===----------------------------------------------------------------------===//
// Attribute printing.
//===----------------------------------------------------------------------===//

iree_status_t loom_print_attr(loom_output_stream_t* stream,
                              const loom_attribute_t* attr,
                              const loom_module_t* module,
                              const loom_attr_descriptor_t* descriptor) {
  switch (attr->kind) {
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, attr->i64);
    case LOOM_ATTR_F64: {
      if (isnan(attr->f64)) {
        return loom_output_stream_write_cstring(stream, "nan");
      }
      if (isinf(attr->f64)) {
        return loom_output_stream_write_cstring(
            stream, attr->f64 < 0.0 ? "-inf" : "inf");
      }
      char buffer[32];
      int length = iree_snprintf(buffer, sizeof(buffer), "%.17g", attr->f64);
      bool has_dot = false;
      bool has_exp = false;
      for (int i = 0; i < length; ++i) {
        if (buffer[i] == '.') {
          has_dot = true;
        }
        if (buffer[i] == 'e' || buffer[i] == 'E') {
          has_exp = true;
        }
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, iree_make_string_view(buffer, length)));
      if (!has_dot && !has_exp) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ".0"));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_STRING: {
      loom_string_id_t id = attr->string_id;
      iree_string_view_t attr_string = iree_string_view_empty();
      if (module && id < module->strings.count) {
        attr_string = module->strings.entries[id];
      }
      return loom_print_string_literal(stream, attr_string);
    }
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(stream,
                                              attr->raw ? "true" : "false");
    case LOOM_ATTR_ENUM: {
      uint8_t case_index = (uint8_t)attr->raw;
      if (descriptor && descriptor->enum_case_names &&
          case_index < descriptor->enum_case_count &&
          descriptor->enum_case_names[case_index]) {
        return loom_output_stream_write(
            stream, loom_bstring_view(descriptor->enum_case_names[case_index]));
      }
      char fallback[16];
      iree_snprintf(fallback, sizeof(fallback), "<%u>", case_index);
      return loom_output_stream_write_cstring(stream, fallback);
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr->symbol;
      if (module && ref.symbol_id < module->symbols.count) {
        loom_string_id_t name_id =
            module->symbols.entries[ref.symbol_id].name_id;
        if (name_id < module->strings.count) {
          iree_string_view_t name = module->strings.entries[name_id];
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '@'));
          return loom_output_stream_write(stream, name);
        }
      }
      return loom_output_stream_write_format(stream, "@<symbol:%" PRIu16 ">",
                                             ref.symbol_id);
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, "%" PRId64, attr->i64_array[i]));
      }
      return loom_output_stream_write_char(stream, ']');
    }
    case LOOM_ATTR_TYPE:
      if (module && attr->type_id < module->types.count) {
        return loom_text_print_type(module->types.entries[attr->type_id],
                                    module, stream);
      }
      return loom_output_stream_write_format(stream, "type<%" PRIu32 ">",
                                             attr->type_id);
    case LOOM_ATTR_ENCODING:
      return loom_print_static_encoding(stream, module,
                                        loom_attr_as_encoding_id(*attr));
    case LOOM_ATTR_DICT: {
      if (attr->count > 0 && !attr->dict_entries) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "DICT attr has count %u but NULL entries",
                                attr->count);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        if (module && entry->name_id < module->strings.count) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write(
              stream, module->strings.entries[entry->name_id]));
        } else {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              stream, "<name:%" PRIu16 ">", entry->name_id));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " = "));
        IREE_RETURN_IF_ERROR(
            loom_print_attr(stream, &entry->value, module, NULL));
      }
      return loom_output_stream_write_char(stream, '}');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown attribute kind %d", (int)attr->kind);
  }
}

iree_status_t loom_text_print_attribute(const loom_attribute_t* attr,
                                        const loom_module_t* module,
                                        loom_output_stream_t* stream) {
  return loom_print_attr(stream, attr, module, NULL);
}

iree_status_t loom_print_encoding_aliases(loom_print_context_t* ctx,
                                          const loom_module_t* module) {
  for (uint16_t i = 0; i < module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &module->encodings.entries[i];
    if (encoding->alias_id == LOOM_STRING_ID_INVALID ||
        encoding->alias_id >= module->strings.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '#'));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, module->strings.entries[encoding->alias_id]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(
        loom_print_canonical_encoding(ctx->stream, module, encoding));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}
