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
#include "loom/ops/type_registry.h"
#include "loom/target/registers.h"

iree_status_t loom_print_value_ref(const loom_print_context_t* ctx,
                                   loom_value_id_t value_id) {
  return loom_print_name_plan_write_value_ref(ctx->name_plan, ctx->stream,
                                              ctx->module, value_id);
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

static bool loom_print_is_bare_identifier(iree_string_view_t value) {
  if (value.size == 0) return false;
  const char first = value.data[0];
  if (!((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') ||
        first == '_' || first == '$')) {
    return false;
  }
  for (iree_host_size_t i = 1; i < value.size; ++i) {
    const char c = value.data[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '$' || c == '-')) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_print_value_name(loom_print_context_t* ctx,
                                    loom_value_id_t value_id) {
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_value_ref(ctx, value_id));
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

// Parameterized types, static encodings, and type-valued attributes are
// mutually recursive.
static iree_status_t loom_print_attr_impl(
    loom_output_stream_t* stream, const loom_attribute_t* attr,
    const loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    const loom_print_context_t* type_context);

static const loom_encoding_alias_descriptor_t*
loom_select_canonical_encoding_alias(const loom_module_t* module,
                                     const loom_encoding_t* encoding) {
  const loom_encoding_vtable_t* vtable =
      loom_module_encoding_vtable(module, encoding);
  if (!vtable || !loom_encoding_static_parameters_are_valid(encoding) ||
      vtable->descriptor->alias_count == 0) {
    return NULL;
  }
  const uint8_t discriminator_index =
      vtable->descriptor->alias_discriminator_parameter_index;
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* parameter = &encoding->attributes[i];
    const uint8_t parameter_index =
        loom_encoding_parameter_descriptor_index(parameter);
    if (parameter_index < discriminator_index) continue;
    if (parameter_index > discriminator_index ||
        parameter->value.kind != LOOM_ATTR_ENUM) {
      return NULL;
    }
    const uint8_t discriminator_value = loom_attr_as_enum(parameter->value);
    const uint8_t alias_ordinal =
        vtable->descriptor
            ->alias_ordinals_by_discriminator[discriminator_value];
    if (alias_ordinal == 0) return NULL;
    const loom_encoding_alias_descriptor_t* alias =
        &vtable->descriptor->aliases[alias_ordinal - 1];

    // An alias must reproduce the complete structural encoding when reparsed.
    // Require every contributed parameter to be present and every fixed value
    // to match. Default parameters may differ and are printed as overrides.
    uint8_t encoding_parameter_index = 0;
    for (uint8_t alias_parameter_index = 0;
         alias_parameter_index < alias->parameter_count;
         ++alias_parameter_index) {
      const loom_encoding_alias_parameter_t* alias_parameter =
          &alias->parameters[alias_parameter_index];
      while (encoding_parameter_index < encoding->attribute_count &&
             loom_encoding_parameter_descriptor_index(
                 &encoding->attributes[encoding_parameter_index]) <
                 alias_parameter->parameter_index) {
        ++encoding_parameter_index;
      }
      if (encoding_parameter_index == encoding->attribute_count) return NULL;
      const loom_named_attr_t* encoding_parameter =
          &encoding->attributes[encoding_parameter_index];
      if (loom_encoding_parameter_descriptor_index(encoding_parameter) !=
          alias_parameter->parameter_index) {
        return NULL;
      }
      if (iree_any_bit_set(alias_parameter->flags,
                           LOOM_ENCODING_ALIAS_PARAMETER_FIXED) &&
          !loom_attribute_equal(&encoding_parameter->value,
                                &alias_parameter->value)) {
        return NULL;
      }
      ++encoding_parameter_index;
    }
    return alias;
  }
  return NULL;
}

static iree_status_t loom_print_canonical_encoding(
    loom_output_stream_t* stream, const loom_module_t* module,
    const loom_encoding_t* encoding, const loom_print_context_t* ctx) {
  const loom_encoding_alias_descriptor_t* alias =
      loom_select_canonical_encoding_alias(module, encoding);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
  if (alias) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write(stream, loom_bstring_view(alias->name)));
  } else if (encoding->name_id < module->strings.count) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, module->strings.entries[encoding->name_id]));
  }

  const loom_encoding_vtable_t* vtable =
      loom_module_encoding_vtable(module, encoding);

  bool parameter_list_open = false;
  uint8_t alias_parameter_index = 0;
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* param = &encoding->attributes[i];
    const uint8_t parameter_descriptor_index =
        loom_encoding_parameter_descriptor_index(param);
    const loom_encoding_alias_parameter_t* alias_parameter = NULL;
    if (alias) {
      while (alias_parameter_index < alias->parameter_count &&
             alias->parameters[alias_parameter_index].parameter_index <
                 parameter_descriptor_index) {
        ++alias_parameter_index;
      }
      if (alias_parameter_index < alias->parameter_count &&
          alias->parameters[alias_parameter_index].parameter_index ==
              parameter_descriptor_index) {
        alias_parameter = &alias->parameters[alias_parameter_index];
      }
    }
    if (alias_parameter &&
        (iree_any_bit_set(alias_parameter->flags,
                          LOOM_ENCODING_ALIAS_PARAMETER_FIXED) ||
         loom_attribute_equal(&param->value, &alias_parameter->value))) {
      continue;
    }
    if (!parameter_list_open) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));
      parameter_list_open = true;
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    if (param->name_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, module->strings.entries[param->name_id]));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '='));
    const loom_attr_descriptor_t* descriptor = NULL;
    if (vtable) {
      if (parameter_descriptor_index != LOOM_ENCODING_PARAMETER_INDEX_INVALID) {
        descriptor = &vtable->descriptor
                          ->parameter_descriptors[parameter_descriptor_index];
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_print_attr_impl(stream, &param->value, module, descriptor, ctx));
  }
  return parameter_list_open ? loom_output_stream_write_char(stream, '>')
                             : iree_ok_status();
}

static iree_status_t loom_print_static_encoding(
    loom_output_stream_t* stream, const loom_module_t* module,
    uint16_t encoding_id, const loom_print_context_t* ctx) {
  if (module && encoding_id > 0 && encoding_id <= module->encodings.count) {
    const loom_encoding_t* encoding =
        &module->encodings.entries[encoding_id - 1];
    if (encoding->alias_id != LOOM_STRING_ID_INVALID &&
        encoding->alias_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
      return loom_output_stream_write(
          stream, module->strings.entries[encoding->alias_id]);
    }
    return loom_print_canonical_encoding(stream, module, encoding, ctx);
  }

  return loom_output_stream_write_format(stream, "#encoding_%" PRIu16,
                                         encoding_id);
}

static iree_status_t loom_print_dim(loom_output_stream_t* stream,
                                    loom_type_t type,
                                    iree_host_size_t dim_index,
                                    const loom_module_t* module,
                                    const loom_print_context_t* ctx) {
  uint64_t packed = loom_type_dim(type, dim_index);
  if (loom_dim_is_dynamic(packed)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
    IREE_RETURN_IF_ERROR(loom_print_name_plan_write_value_ref(
        ctx ? ctx->name_plan : NULL, stream, module,
        loom_dim_value_id(packed)));
    return loom_output_stream_write_char(stream, ']');
  }
  return loom_output_stream_write_format(stream, "%" PRId64,
                                         loom_dim_static_size(packed));
}

// Shared in source but kept inline in both complete-type and result-type hot
// paths so declaration-driven spelling does not add a call per printed type.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_status_t
loom_print_shaped_interior(loom_output_stream_t* stream, loom_type_t type,
                           const loom_module_t* module,
                           const loom_print_context_t* ctx) {
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
    }
    IREE_RETURN_IF_ERROR(loom_print_dim(stream, type, i, module, ctx));
  }
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
  }
  IREE_RETURN_IF_ERROR(
      loom_print_scalar_type(stream, loom_type_element_type(type)));
  if (loom_type_has_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    if (loom_type_has_ssa_encoding(type)) {
      IREE_RETURN_IF_ERROR(loom_print_name_plan_write_value_ref(
          ctx ? ctx->name_plan : NULL, stream, module,
          loom_type_encoding_value_id(type)));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_print_static_encoding(stream, module, type.encoding_id, ctx));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_compact_shape_prefix(
    loom_output_stream_t* stream, loom_type_kind_t type_kind) {
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_lookup_builtin(type_kind);
  const iree_string_view_t name = loom_bstring_view(descriptor->name);
  return loom_output_stream_write(
      stream, iree_make_string_view(name.data, name.size + 1));
}

static iree_status_t loom_print_descriptor_backed_type(
    loom_type_t type, const loom_module_t* module, loom_output_stream_t* stream,
    const loom_print_context_t* type_context) {
  const loom_type_descriptor_t* descriptor = NULL;
  const loom_parameterized_type_descriptor_t* parameterized = NULL;
  uint8_t parameter_count = 0;
  const loom_attribute_t* parameters = NULL;
  loom_attribute_t inline_parameter = loom_attr_absent();
  if (loom_type_is_parameterized(type)) {
    parameterized = loom_type_parameterized_descriptor(type);
    if (!parameterized) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameterized type has no family descriptor");
    }
    descriptor = loom_type_registry_lookup(
        module->context, loom_bstring_view(parameterized->name));
    parameter_count = loom_type_parameterized_parameter_count(type);
    parameters = loom_type_parameterized_parameters(type);
  } else {
    descriptor = loom_type_registry_lookup_builtin(loom_type_kind(type));
    if (descriptor) parameterized = descriptor->parameterized;
    if (!parameterized || parameterized->parameter_count != 1 ||
        parameterized->parameter_descriptors[0].attr_kind != LOOM_ATTR_ENUM) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "compact descriptor-backed type has no enum parameter descriptor");
    }
    const uint8_t payload = loom_type_payload(type);
    inline_parameter =
        payload == 0 &&
                iree_any_bit_set(parameterized->parameter_descriptors[0].flags,
                                 LOOM_ATTR_OPTIONAL)
            ? loom_attr_absent()
            : loom_attr_enum(payload);
    parameter_count = 1;
    parameters = &inline_parameter;
  }
  if (!descriptor || descriptor->parameterized != parameterized) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized type family is not registered");
  }
  if (parameter_count != parameterized->parameter_count ||
      (parameter_count > 0 && !parameters)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized type has malformed slot storage");
  }

  bool omit_parameter_list = iree_any_bit_set(
      parameterized->flags, LOOM_PARAMETERIZED_TYPE_OMIT_EMPTY_PARAMETER_LIST);
  for (uint8_t i = 0; i < parameter_count && omit_parameter_list; ++i) {
    omit_parameter_list = loom_attr_is_absent(parameters[i]);
  }
  const iree_string_view_t name = loom_bstring_view(descriptor->name);
  if (omit_parameter_list) {
    return loom_output_stream_write(stream, name);
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write(
      stream, iree_make_string_view(name.data, name.size + 1)));

  loom_print_context_t parameter_context = {0};
  if (type_context) parameter_context = *type_context;
  parameter_context.stream = stream;
  parameter_context.module = module;
  parameter_context.has_previous_token = false;
  parameter_context.glue_next = false;
  parameter_context.last_char = 0;
  for (uint16_t i = 0; i < descriptor->format_element_count; ++i) {
    const loom_type_format_element_t* element = &descriptor->format_elements[i];
    switch (element->kind) {
      case LOOM_TYPE_FMT_PARAM: {
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(&parameter_context));
        uint8_t parameter_index = element->field_index;
        IREE_RETURN_IF_ERROR(loom_print_attr_impl(
            stream, &parameters[parameter_index], module,
            &parameterized->parameter_descriptors[parameter_index],
            &parameter_context));
        loom_print_did_write(&parameter_context);
        break;
      }
      case LOOM_TYPE_FMT_PARAM_KEY: {
        iree_string_view_t parameter_name = loom_attr_descriptor_name(
            &parameterized->parameter_descriptors[element->field_index]);
        IREE_RETURN_IF_ERROR(
            loom_print_emit(&parameter_context, parameter_name, false));
        break;
      }
      case LOOM_TYPE_FMT_KEYWORD: {
        IREE_RETURN_IF_ERROR(loom_print_emit(
            &parameter_context,
            loom_bstring_view(
                loom_keyword_bstring((loom_keyword_id_t)element->data)),
            false));
        break;
      }
      case LOOM_TYPE_FMT_OPTIONAL:
        if (loom_attr_is_absent(parameters[element->field_index])) {
          i = (uint16_t)(i + (element->data >> 8));
        }
        break;
      case LOOM_TYPE_FMT_GLUE:
        loom_print_set_glue(&parameter_context);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("unsupported parameterized type format kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return loom_output_stream_write_char(stream, '>');
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
          loom_print_compact_shape_prefix(stream, loom_type_kind(type)));
      IREE_RETURN_IF_ERROR(
          loom_print_shaped_interior(stream, type, module, ctx));
      return loom_output_stream_write_char(stream, '>');
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
    case LOOM_TYPE_PARAMETERIZED:
      return loom_print_descriptor_backed_type(type, module, stream, ctx);
    case LOOM_TYPE_REGISTER: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "reg<"));
      if (!ctx) {
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
      } else {
        const loom_text_low_asm_descriptor_set_t* descriptor_set =
            ctx->low_repr.descriptor_set;
        if (!descriptor_set &&
            iree_string_view_is_empty(ctx->low_repr.contract_key) &&
            ctx->low_asm_environment.vtable &&
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
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write(stream, register_class_name));
        if (unit_count != 1) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " x"));
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_format(stream, "%u", unit_count));
        }
      }
      const loom_type_t* value_type = loom_type_register_value_type(type);
      if (value_type) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " : "));
        IREE_RETURN_IF_ERROR(
            loom_text_print_type_impl(*value_type, module, stream, ctx));
      }
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_STORAGE:
    case LOOM_TYPE_ENCODING:
      return loom_print_descriptor_backed_type(type, module, stream, ctx);
    case LOOM_TYPE_BUFFER:
      return loom_output_stream_write_cstring(stream, "buffer");
    case LOOM_TYPE_POOL: {
      IREE_RETURN_IF_ERROR(
          loom_print_compact_shape_prefix(stream, loom_type_kind(type)));
      IREE_RETURN_IF_ERROR(loom_print_dim(stream, type, 0, module, ctx));
      return loom_output_stream_write_char(stream, '>');
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
  loom_print_context_t ctx = {0};
  ctx.stream = stream;
  ctx.module = module;
  ctx.flags = options ? options->flags : LOOM_TEXT_PRINT_DEFAULT;
  if (options) ctx.low_asm_environment = options->low_asm_environment;
  return loom_text_print_type_impl(type, module, stream, &ctx);
}

iree_status_t loom_print_type(loom_print_context_t* ctx, loom_type_t type) {
  return loom_text_print_type_impl(type, ctx->module, ctx->stream, ctx);
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
          loom_print_compact_shape_prefix(stream, loom_type_kind(type)));
      IREE_RETURN_IF_ERROR(
          loom_print_shaped_interior(stream, type, module, ctx));
      return loom_output_stream_write_char(stream, '>');
    }
    default:
      return loom_text_print_type_impl(type, module, stream, ctx);
  }
}

iree_status_t loom_print_value_type(loom_print_context_t* ctx,
                                    loom_value_id_t value_id) {
  if (value_id < ctx->module->values.count) {
    return loom_print_type(ctx, loom_module_value_type(ctx->module, value_id));
  }
  return loom_output_stream_write_cstring(ctx->stream, "<unknown>");
}

iree_status_t loom_print_result_value_type(loom_print_context_t* ctx,
                                           loom_value_id_t value_id) {
  if (value_id < ctx->module->values.count) {
    return loom_text_print_result_type(
        loom_module_value_type(ctx->module, value_id), ctx->module, ctx->stream,
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

static iree_status_t loom_print_byte_hex_string_literal(
    loom_output_stream_t* stream, const uint8_t* data, uint32_t data_length) {
  static const char kHexDigits[] = "0123456789abcdef";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  for (uint32_t i = 0; i < data_length; ++i) {
    char encoded[] = {
        kHexDigits[data[i] >> 4],
        kHexDigits[data[i] & 0x0F],
    };
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, iree_make_string_view(encoded, IREE_ARRAYSIZE(encoded))));
  }
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_print_location_tag(loom_output_stream_t* stream,
                                             loom_location_tag_t tag) {
  if (tag == LOOM_LOCATION_TAG_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tagged location has invalid tag 0");
  }
  iree_string_view_t name = loom_location_tag_name(tag);
  if (!iree_string_view_is_empty(name)) {
    return loom_output_stream_write(stream, name);
  }
  return loom_output_stream_write_format(stream, "%u", (unsigned)tag);
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
    case LOOM_LOCATION_TAGGED: {
      if (entry->tagged.data_length > 0 && !entry->tagged.data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tagged location has data_length %u but NULL data",
            entry->tagged.data_length);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "tagged<"));
      IREE_RETURN_IF_ERROR(loom_print_location_tag(stream, entry->tagged.tag));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_print_byte_hex_string_literal(
          stream, entry->tagged.data, entry->tagged.data_length));
      if (entry->tagged.child != LOOM_LOCATION_UNKNOWN) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        IREE_RETURN_IF_ERROR(
            loom_print_location_body(stream, module, entry->tagged.child));
      }
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

typedef enum loom_parameterized_attr_print_form_e {
  LOOM_PARAMETERIZED_ATTR_PRINT_FORM_COMPLETE = 0,
  LOOM_PARAMETERIZED_ATTR_PRINT_FORM_PARAMETERS = 1,
} loom_parameterized_attr_print_form_t;

static iree_status_t loom_print_parameterized_attr_impl(
    loom_output_stream_t* stream, const loom_attribute_t* attr,
    const loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    const loom_print_context_t* type_context,
    loom_parameterized_attr_print_form_t form) {
  if (!module || !module->context) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "printing a parameterized attribute requires a module context");
  }
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_resolve_parameterized_attr(
          module->context, loom_attr_as_parameterized_kind(*attr));
  if (!family_descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute has unknown family kind %u",
        (unsigned)loom_attr_as_parameterized_kind(*attr));
  }
  if (descriptor && descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute does not match field kind %u",
        (unsigned)descriptor->attr_kind);
  }
  if (descriptor &&
      descriptor->reference.parameterized_attr_kind !=
          LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
      descriptor->reference.parameterized_attr_kind !=
          family_descriptor->kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family kind %u does not match field contract "
        "%u",
        (unsigned)family_descriptor->kind,
        (unsigned)descriptor->reference.parameterized_attr_kind);
  }
  iree_string_view_t family_name = loom_bstring_view(family_descriptor->name);
  if (attr->count != family_descriptor->parameter_count ||
      (attr->count > 0 && !attr->parameterized_slots)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute '%.*s' has malformed slot storage",
        (int)family_name.size, family_name.data);
  }
  if (form == LOOM_PARAMETERIZED_ATTR_PRINT_FORM_PARAMETERS &&
      (!descriptor || descriptor->reference.parameterized_attr_kind ==
                          LOOM_PARAMETERIZED_ATTR_KIND_ANY)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "printing parameterized attribute parameters requires an exact-family "
        "field descriptor");
  }

  if (form == LOOM_PARAMETERIZED_ATTR_PRINT_FORM_COMPLETE) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, family_name));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));

  bool has_previous_parameter = false;
  const uint8_t primary_parameter_index =
      family_descriptor->primary_parameter_index;
  if (primary_parameter_index != LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER) {
    const loom_attribute_t* primary_parameter =
        &attr->parameterized_slots[primary_parameter_index];
    const loom_attr_descriptor_t* primary_parameter_descriptor =
        &family_descriptor->parameter_descriptors[primary_parameter_index];
    if (loom_attr_is_absent(*primary_parameter)) {
      iree_string_view_t parameter_name =
          loom_attr_descriptor_name(primary_parameter_descriptor);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "required parameter '%.*s' of parameterized attribute '%.*s' is "
          "absent",
          (int)parameter_name.size, parameter_name.data, (int)family_name.size,
          family_name.data);
    }
    IREE_RETURN_IF_ERROR(loom_print_attr_impl(stream, primary_parameter, module,
                                              primary_parameter_descriptor,
                                              type_context));
    has_previous_parameter = true;
  }
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    if (i == primary_parameter_index) continue;
    const loom_attr_descriptor_t* parameter_descriptor =
        &family_descriptor->parameter_descriptors[i];
    const loom_attribute_t* parameter = &attr->parameterized_slots[i];
    if (loom_attr_is_absent(*parameter)) {
      if (iree_any_bit_set(parameter_descriptor->flags, LOOM_ATTR_OPTIONAL)) {
        continue;
      }
      iree_string_view_t parameter_name =
          loom_attr_descriptor_name(parameter_descriptor);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "required parameter '%.*s' of parameterized attribute '%.*s' is "
          "absent",
          (int)parameter_name.size, parameter_name.data, (int)family_name.size,
          family_name.data);
    }
    if (has_previous_parameter) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, loom_attr_descriptor_name(parameter_descriptor)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " = "));
    IREE_RETURN_IF_ERROR(loom_print_attr_impl(
        stream, parameter, module, parameter_descriptor, type_context));
    has_previous_parameter = true;
  }
  return loom_output_stream_write_char(stream, '>');
}

static iree_status_t loom_print_attr_impl(
    loom_output_stream_t* stream, const loom_attribute_t* attr,
    const loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    const loom_print_context_t* type_context) {
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
      if (descriptor &&
          iree_any_bit_set(descriptor->flags, LOOM_ATTR_BARE_IDENTIFIER) &&
          loom_print_is_bare_identifier(attr_string)) {
        return loom_output_stream_write(stream, attr_string);
      }
      return loom_print_string_literal(stream, attr_string);
    }
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(stream,
                                              attr->raw ? "true" : "false");
    case LOOM_ATTR_ENUM: {
      uint8_t case_index = (uint8_t)attr->raw;
      loom_bstring_t case_name =
          loom_attr_descriptor_enum_case_name(descriptor, case_index);
      if (case_name) {
        return loom_output_stream_write(stream, loom_bstring_view(case_name));
      }
      char fallback[16];
      iree_snprintf(fallback, sizeof(fallback), "<%u>", case_index);
      return loom_output_stream_write_cstring(stream, fallback);
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      if (!descriptor || descriptor->attr_kind != LOOM_ATTR_ENUM_ARRAY ||
          !descriptor->enum_case_names) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "printing an enum array requires a descriptor-backed operation "
            "field");
      }
      if (attr->count > 0 && !attr->enum_array) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ENUM_ARRAY attr has count %u but NULL values",
                                attr->count);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        uint8_t value = attr->enum_array[i];
        loom_bstring_t case_name =
            loom_attr_descriptor_enum_case_name(descriptor, value);
        if (case_name) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write(stream, loom_bstring_view(case_name)));
        } else if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_format(stream, "<%u>", (unsigned)value));
        } else {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "closed enum-array field '%.*s' has no value %u",
              (int)loom_attr_descriptor_name(descriptor).size,
              loom_attr_descriptor_name(descriptor).data, (unsigned)value);
        }
      }
      return loom_output_stream_write_char(stream, ']');
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      if (!descriptor || descriptor->attr_kind != LOOM_ATTR_SIGNED_ENUM_SET ||
          !descriptor->enum_case_names ||
          iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "printing a signed enum set requires a closed descriptor-backed "
            "field");
      }
      iree_host_size_t canonical_word_count = 0;
      IREE_RETURN_IF_ERROR(loom_signed_enum_set_canonical_word_count(
          loom_attr_as_signed_enum_set(*attr), &canonical_word_count));
      if (canonical_word_count != attr->count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "signed enum-set field '%.*s' is not canonically trimmed",
            (int)loom_attr_descriptor_name(descriptor).size,
            loom_attr_descriptor_name(descriptor).data);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      bool wrote_value = false;
      loom_signed_enum_set_t set = loom_attr_as_signed_enum_set(*attr);
      for (iree_host_size_t value = 0; value < 256; ++value) {
        bool positive =
            loom_signed_enum_set_contains_positive(set, (uint8_t)value);
        bool negative =
            loom_signed_enum_set_contains_negative(set, (uint8_t)value);
        if (!positive && !negative) continue;
        loom_bstring_t case_name =
            loom_attr_descriptor_enum_case_name(descriptor, (uint8_t)value);
        if (!case_name) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "closed signed enum-set field '%.*s' has no value %u",
              (int)loom_attr_descriptor_name(descriptor).size,
              loom_attr_descriptor_name(descriptor).data, (unsigned)value);
        }
        if (wrote_value) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        if (negative) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '-'));
        }
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write(stream, loom_bstring_view(case_name)));
        wrote_value = true;
      }
      return loom_output_stream_write_char(stream, ']');
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
    case LOOM_ATTR_BYTES: {
      static const char kHexDigits[] = "0123456789abcdef";
      iree_const_byte_span_t bytes = loom_attr_as_bytes(*attr);
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "bytes(\""));
      for (iree_host_size_t i = 0; i < bytes.data_length; ++i) {
        char hex[2] = {
            kHexDigits[bytes.data[i] >> 4],
            kHexDigits[bytes.data[i] & 0x0F],
        };
        IREE_RETURN_IF_ERROR(loom_output_stream_write(
            stream, iree_make_string_view(hex, IREE_ARRAYSIZE(hex))));
      }
      return loom_output_stream_write_cstring(stream, "\")");
    }
    case LOOM_ATTR_TYPE:
      if (module && attr->type_id < module->types.count) {
        return loom_text_print_type_impl(module->types.entries[attr->type_id],
                                         module, stream, type_context);
      }
      return loom_output_stream_write_format(stream, "type<%" PRIu32 ">",
                                             attr->type_id);
    case LOOM_ATTR_ENCODING:
      return loom_print_static_encoding(
          stream, module, loom_attr_as_encoding_id(*attr), type_context);
    case LOOM_ATTR_PARAMETERIZED: {
      return loom_print_parameterized_attr_impl(
          stream, attr, module, descriptor, type_context,
          LOOM_PARAMETERIZED_ATTR_PRINT_FORM_COMPLETE);
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (!descriptor ||
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "printing a parameterized attribute array requires a "
            "descriptor-backed field");
      }
      if (attr->count > 0 && !attr->parameterized_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "PARAMETERIZED_ARRAY attr has count %u but NULL values",
            attr->count);
      }
      loom_attr_descriptor_t element_descriptor = *descriptor;
      element_descriptor.attr_kind = LOOM_ATTR_PARAMETERIZED;
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_print_attr_impl(stream, &attr->parameterized_array[i], module,
                                 &element_descriptor, type_context));
      }
      return loom_output_stream_write_char(stream, ']');
    }
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
        IREE_RETURN_IF_ERROR(loom_print_attr_impl(stream, &entry->value, module,
                                                  NULL, type_context));
      }
      return loom_output_stream_write_char(stream, '}');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown attribute kind %d", (int)attr->kind);
  }
}

iree_status_t loom_print_attr(const loom_print_context_t* ctx,
                              const loom_attribute_t* attr,
                              const loom_attr_descriptor_t* descriptor) {
  return loom_print_attr_impl(ctx->stream, attr, ctx->module, descriptor, ctx);
}

iree_status_t loom_print_parameterized_attr_parameters(
    const loom_print_context_t* ctx, const loom_attribute_t* attr,
    const loom_attr_descriptor_t* descriptor) {
  if (attr->kind != LOOM_ATTR_PARAMETERIZED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute parameters require a PARAMETERIZED value");
  }
  return loom_print_parameterized_attr_impl(
      ctx->stream, attr, ctx->module, descriptor, ctx,
      LOOM_PARAMETERIZED_ATTR_PRINT_FORM_PARAMETERS);
}

iree_status_t loom_text_print_attribute(const loom_attribute_t* attr,
                                        const loom_module_t* module,
                                        loom_output_stream_t* stream) {
  return loom_print_attr_impl(stream, attr, module, /*descriptor=*/NULL,
                              /*type_context=*/NULL);
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
        loom_print_canonical_encoding(ctx->stream, module, encoding, ctx));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}
