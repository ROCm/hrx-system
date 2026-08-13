// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/type_validator.h"

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ops/type_registry.h"

#define LOOM_BYTECODE_MAX_TYPE_COUNT (UINT64_C(1) << 16)

static loom_bytecode_attribute_validator_t
loom_bytecode_type_plan_attribute_validator(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view) {
  return (loom_bytecode_attribute_validator_t){
      .decoder = decoder,
      .context = context,
      .module_view = module_view,
  };
}

static iree_status_t loom_bytecode_type_plan_validate_string_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_010,
                                           params, IREE_ARRAYSIZE(params),
                                           offset, 0);
  }
  *out_string = module_view->strings.values[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_validate_type_ref(
    loom_bytecode_reader_decoder_t* decoder, uint64_t type_id,
    uint64_t available_type_count, uint64_t offset) {
  if (type_id >= available_type_count) {
    return loom_bytecode_reader_emit_table_ref(
        decoder, IREE_SV("TYPES"), type_id, available_type_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_validate_encoding_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view, uint64_t encoding_id,
    uint64_t offset) {
  const iree_host_size_t encoding_count = module_view->encodings.count;
  if (encoding_id == 0 || encoding_id > encoding_count) {
    return loom_bytecode_reader_emit_table_ref(
        decoder, IREE_SV("ENCODINGS"), encoding_id, encoding_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_validate_parameterized(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    loom_bytecode_reader_cursor_t* cursor, loom_type_id_t type_id,
    loom_bytecode_parameterized_type_fact_t** out_fact) {
  const uint64_t family_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t family_name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &family_name_id));
  iree_string_view_t family_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_string_ref(
      decoder, module_view, family_name_id,
      IREE_SV("parameterized_type_family"), family_offset, &family_name));
  const loom_type_descriptor_t* type_descriptor =
      loom_type_registry_lookup(context, family_name);
  if (!type_descriptor || !type_descriptor->parameterized) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), type_id,
        IREE_SV("family_name"), family_offset,
        IREE_SV("parameterized_type_family_is_not_registered"));
  }
  const loom_parameterized_type_descriptor_t* descriptor =
      type_descriptor->parameterized;

  const uint64_t present_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t present_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &present_count));
  if (present_count > descriptor->parameter_count) {
    return loom_bytecode_reader_emit_count_exceeds(
        decoder, IREE_SV("parameterized_type_parameters"), present_count,
        descriptor->parameter_count, present_count_offset);
  }

  const iree_host_size_t parameters_position = cursor->cursor.position;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_bytecode_parameterized_type_fact_t), &allocation_size,
      IREE_STRUCT_FIELD_FAM((iree_host_size_t)present_count, uint8_t)));
  loom_bytecode_parameterized_type_fact_t* fact = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, allocation_size, (void**)&fact));
  *fact = (loom_bytecode_parameterized_type_fact_t){
      .base =
          {
              .type_id = type_id,
              .kind = LOOM_TYPE_PARAMETERIZED,
          },
      .descriptor = descriptor,
      .parameters_offset = cursor->absolute_offset + parameters_position,
      .present_count = (uint8_t)present_count,
  };

  loom_bytecode_attribute_validator_t attribute_validator =
      loom_bytecode_type_plan_attribute_validator(decoder, context,
                                                  module_view);
  uint8_t next_parameter_index = 0;
  for (iree_host_size_t i = 0; i < (iree_host_size_t)present_count; ++i) {
    const uint64_t parameter_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t parameter_name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &parameter_name_id));
    iree_string_view_t parameter_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_string_ref(
        decoder, module_view, parameter_name_id,
        IREE_SV("parameterized_type_parameter"), parameter_offset,
        &parameter_name));
    uint8_t relative_parameter_index = LOOM_ATTR_INDEX_NONE;
    const loom_attr_descriptor_t* parameter_descriptor =
        loom_attr_descriptor_find_by_name(
            descriptor->parameter_descriptors + next_parameter_index,
            (uint8_t)(descriptor->parameter_count - next_parameter_index),
            parameter_name, &relative_parameter_index);
    const uint8_t parameter_index =
        parameter_descriptor
            ? (uint8_t)(next_parameter_index + relative_parameter_index)
            : LOOM_ATTR_INDEX_NONE;
    if (parameter_index == LOOM_ATTR_INDEX_NONE) {
      uint8_t declared_parameter_index = LOOM_ATTR_INDEX_NONE;
      const bool declared_before_cursor =
          loom_attr_descriptor_find_by_name(
              descriptor->parameter_descriptors, descriptor->parameter_count,
              parameter_name, &declared_parameter_index) != NULL;
      return loom_bytecode_reader_emit_invalid_field(
          decoder, IREE_SV("TYPES"), IREE_SV("type"), type_id,
          IREE_SV("parameter_name"), parameter_offset,
          declared_before_cursor
              ? IREE_SV("parameterized_type_parameters_are_not_in_declaration_"
                        "order")
              : IREE_SV("parameterized_type_parameter_is_not_declared"));
    }
    fact->parameter_indices[i] = parameter_index;
    next_parameter_index = parameter_index + 1;

    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_attribute_read_kind(decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_named(
        &attribute_validator, cursor, parameter_descriptor, value_kind,
        type_id));
  }

  iree_host_size_t present_ordinal = 0;
  for (uint8_t parameter_index = 0;
       parameter_index < descriptor->parameter_count; ++parameter_index) {
    const bool is_present =
        present_ordinal < (iree_host_size_t)present_count &&
        fact->parameter_indices[present_ordinal] == parameter_index;
    if (is_present) {
      ++present_ordinal;
    } else if (!iree_any_bit_set(
                   descriptor->parameter_descriptors[parameter_index].flags,
                   LOOM_ATTR_OPTIONAL)) {
      return loom_bytecode_reader_emit_invalid_field(
          decoder, IREE_SV("TYPES"), IREE_SV("type"), type_id,
          IREE_SV("parameter_name"), present_count_offset,
          IREE_SV("parameterized_type_required_parameter_is_absent"));
    }
  }

  fact->parameters_length = cursor->cursor.position - parameters_position;
  *out_fact = fact;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_decode_kind(
    loom_bytecode_reader_decoder_t* decoder, uint8_t kind_byte, uint64_t offset,
    loom_type_kind_t* out_kind) {
  switch (kind_byte) {
    case LOOM_BYTECODE_TYPE_NONE:
      *out_kind = LOOM_TYPE_NONE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_SCALAR:
      *out_kind = LOOM_TYPE_SCALAR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_TILE:
      *out_kind = LOOM_TYPE_TILE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_TENSOR:
      *out_kind = LOOM_TYPE_TENSOR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_FUNCTION:
      *out_kind = LOOM_TYPE_FUNCTION;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_DIALECT:
      *out_kind = LOOM_TYPE_DIALECT;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_ENCODING:
      *out_kind = LOOM_TYPE_ENCODING;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_POOL:
      *out_kind = LOOM_TYPE_POOL;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_VECTOR:
      *out_kind = LOOM_TYPE_VECTOR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_VIEW:
      *out_kind = LOOM_TYPE_VIEW;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_BUFFER:
      *out_kind = LOOM_TYPE_BUFFER;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_REGISTER:
      *out_kind = LOOM_TYPE_REGISTER;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_STORAGE:
      *out_kind = LOOM_TYPE_STORAGE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_PARAMETERIZED:
      *out_kind = LOOM_TYPE_PARAMETERIZED;
      return iree_ok_status();
    default: {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(kind_byte),
          loom_param_u64(offset),
      };
      return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_004,
                                             params, IREE_ARRAYSIZE(params),
                                             offset, 1);
    }
  }
}

static iree_status_t loom_bytecode_type_plan_build_shaped(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, loom_type_kind_t kind,
    loom_scalar_type_t element_type, uint8_t rank, uint8_t attachment,
    uint64_t encoding_instance, const uint64_t* dims, loom_type_t* out_type,
    uint64_t offset) {
  if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
    return loom_bytecode_reader_emit_enum_value(
        decoder, IREE_SV("element_type"), element_type, LOOM_SCALAR_TYPE_COUNT_,
        offset);
  }
  if (rank > LOOM_TYPE_MAX_RANK) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("rank_exceeds_loom_type_maximum"));
  }
  if (kind == LOOM_TYPE_VECTOR && rank == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("vector_rank_zero"));
  }

  uint16_t encoding_id = 0;
  loom_encoding_flags_t encoding_flags = 0;
  switch (attachment) {
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE:
      if (encoding_instance != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("none_encoding_attachment_must_have_id_0"));
      }
      break;
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC: {
      IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_encoding_ref(
          decoder, module_view, encoding_instance, offset));
      if (encoding_instance > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("encoding_id_exceeds_runtime_type_field_width"));
      }
      encoding_id = (uint16_t)encoding_instance;
      break;
    }
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA:
      if (encoding_instance != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("dynamic_encoding_attachment_must_have_id_0"));
      }
      encoding_flags = LOOM_ENCODING_FLAG_SSA;
      break;
    default:
      return loom_bytecode_reader_emit_enum_value(
          decoder, IREE_SV("encoding_attachment"), attachment,
          LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA + 1, offset);
  }
  if (kind == LOOM_TYPE_VECTOR && (encoding_id != 0 || encoding_flags != 0)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
        IREE_SV("encoding_attachment"), offset,
        IREE_SV("vector_types_must_not_carry_encoding_or_layout_attachments"));
  }

  loom_type_t type = {0};
  if (rank == 0) {
    type = loom_type_shaped_0d(kind, element_type, encoding_id);
  } else if (rank == 1) {
    type = loom_type_shaped_1d(kind, element_type, dims[0], encoding_id);
  } else if (rank == 2) {
    type =
        loom_type_shaped_2d(kind, element_type, dims[0], dims[1], encoding_id);
  } else {
    loom_overflow_dim_t* overflow_dims = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow_dims));
    bool all_static = true;
    for (uint8_t i = 0; i < rank; ++i) {
      overflow_dims[i] = dims[i];
      if (loom_dim_is_dynamic(dims[i])) {
        all_static = false;
      }
    }
    uint8_t flags = all_static ? LOOM_TYPE_FLAG_ALL_STATIC : 0;
    type.header = loom_type_make_header(kind, element_type, rank, flags);
    type.encoding_id = encoding_id;
    type.dims[0] = (uint64_t)(uintptr_t)overflow_dims;
  }
  type.encoding_flags = encoding_flags;
  *out_type = type;
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_plan_build(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, iree_const_byte_span_t section_bytes,
    uint64_t section_absolute_offset) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section_bytes.data, section_bytes.data_length, section_absolute_offset,
      IREE_SV("TYPES"), &cursor);

  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &count));
  if (count > LOOM_BYTECODE_MAX_TYPE_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        decoder, IREE_SV("TYPES"), count, LOOM_BYTECODE_MAX_TYPE_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, (iree_host_size_t)count,
                                  sizeof(loom_bytecode_type_plan_entry_t),
                                  (void**)&module_view->types.entries));
  }
  module_view->types.count = (iree_host_size_t)count;
  loom_bytecode_type_fact_t* first_fact = NULL;
  loom_bytecode_type_fact_t* last_fact = NULL;
  for (uint64_t type_index = 0; type_index < count; ++type_index) {
    uint64_t type_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    loom_bytecode_type_plan_entry_t* plan_entry =
        &module_view->types.entries[type_index];
    plan_entry->bytecode_offset = type_offset;
    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(decoder, &cursor, &kind_byte));
    loom_type_kind_t kind = LOOM_TYPE_NONE;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_kind(
        decoder, kind_byte, type_offset, &kind));

    loom_type_t direct_type = {0};
    loom_bytecode_type_fact_t* type_fact = NULL;
    switch (kind) {
      case LOOM_TYPE_NONE:
        direct_type = loom_type_none();
        break;
      case LOOM_TYPE_SCALAR: {
        uint8_t element_type = 0;
        uint64_t element_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &element_type));
        if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("scalar_type"), element_type,
              LOOM_SCALAR_TYPE_COUNT_, element_offset);
        }
        direct_type = loom_type_scalar((loom_scalar_type_t)element_type);
        break;
      }
      case LOOM_TYPE_TILE:
      case LOOM_TYPE_TENSOR:
      case LOOM_TYPE_VECTOR:
      case LOOM_TYPE_VIEW: {
        uint8_t element_type = 0;
        uint8_t rank = 0;
        uint8_t attachment = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &element_type));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &rank));
        uint64_t attachment_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &attachment));
        uint64_t encoding_instance = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            decoder, &cursor, &encoding_instance));
        uint64_t dims[LOOM_TYPE_MAX_RANK] = {0};
        if (rank > LOOM_TYPE_MAX_RANK) {
          return loom_bytecode_reader_emit_invalid_field(
              decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("rank"), type_offset,
              IREE_SV("rank_exceeds_loom_type_maximum"));
        }
        for (uint8_t i = 0; i < rank; ++i) {
          uint8_t is_dynamic = 0;
          uint64_t dim_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_u8(decoder, &cursor, &is_dynamic));
          if (is_dynamic == 0) {
            uint64_t size = 0;
            IREE_RETURN_IF_ERROR(
                loom_bytecode_reader_read_uvarint(decoder, &cursor, &size));
            if (size > LOOM_DIM_MAX_STATIC_SIZE) {
              return loom_bytecode_reader_emit_invalid_field(
                  decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                  IREE_SV("dim_size"), dim_offset,
                  IREE_SV("static_dimension_exceeds_loom_maximum"));
            }
            dims[i] = loom_dim_pack_static((int64_t)size);
          } else if (is_dynamic == 1) {
            dims[i] = loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID);
          } else {
            return loom_bytecode_reader_emit_enum_value(
                decoder, IREE_SV("is_dynamic"), is_dynamic, 2, dim_offset);
          }
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_build_shaped(
            decoder, module_view, scratch_arena, kind,
            (loom_scalar_type_t)element_type, rank, attachment,
            encoding_instance, dims, &direct_type, attachment_offset));
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        uint64_t arg_count = 0;
        uint64_t result_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, &cursor, &arg_count));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, &cursor, &result_count));
        if (arg_count > UINT16_MAX || result_count > UINT16_MAX ||
            arg_count + result_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("signature_count"), type_offset,
              IREE_SV("function_signature_exceeds_runtime_field_width"));
        }
        iree_host_size_t total_count =
            (iree_host_size_t)(arg_count + result_count);
        iree_host_size_t allocation_size = 0;
        IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
            sizeof(loom_bytecode_function_type_fact_t), &allocation_size,
            IREE_STRUCT_FIELD_FAM(total_count, loom_type_id_t)));
        loom_bytecode_function_type_fact_t* fact = NULL;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(scratch_arena, allocation_size, (void**)&fact));
        *fact = (loom_bytecode_function_type_fact_t){
            .base =
                {
                    .type_id = (loom_type_id_t)type_index,
                    .kind = LOOM_TYPE_FUNCTION,
                },
            .argument_count = (uint16_t)arg_count,
            .result_count = (uint16_t)result_count,
        };
        for (iree_host_size_t i = 0; i < total_count; ++i) {
          uint64_t ref_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t type_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(decoder, &cursor, &type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_type_ref(
              decoder, type_id, type_index, ref_offset));
          fact->type_ids[i] = (loom_type_id_t)type_id;
        }
        type_fact = &fact->base;
        break;
      }
      case LOOM_TYPE_DIALECT: {
        uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t name_id = 0;
        uint64_t param_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, &cursor, &name_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, &cursor, &param_count));
        iree_string_view_t unused_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_string_ref(
            decoder, module_view, name_id, IREE_SV("dialect_type_name"),
            name_offset, &unused_name));
        if (param_count > UINT16_MAX || param_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              decoder, IREE_SV("dialect_type_params"), param_count, UINT16_MAX,
              type_offset);
        }
        if (param_count == 0) {
          direct_type = loom_type_dialect_opaque((loom_string_id_t)name_id);
          break;
        }
        iree_host_size_t allocation_size = 0;
        IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
            sizeof(loom_bytecode_dialect_type_fact_t), &allocation_size,
            IREE_STRUCT_FIELD_FAM((iree_host_size_t)param_count,
                                  loom_type_id_t)));
        loom_bytecode_dialect_type_fact_t* fact = NULL;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(scratch_arena, allocation_size, (void**)&fact));
        *fact = (loom_bytecode_dialect_type_fact_t){
            .base =
                {
                    .type_id = (loom_type_id_t)type_index,
                    .kind = LOOM_TYPE_DIALECT,
                },
            .name_id = (loom_string_id_t)name_id,
            .parameter_count = (uint16_t)param_count,
        };
        for (uint64_t i = 0; i < param_count; ++i) {
          uint64_t ref_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t type_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(decoder, &cursor, &type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_type_ref(
              decoder, type_id, type_index, ref_offset));
          fact->type_ids[i] = (loom_type_id_t)type_id;
        }
        type_fact = &fact->base;
        break;
      }
      case LOOM_TYPE_PARAMETERIZED: {
        loom_bytecode_parameterized_type_fact_t* fact = NULL;
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_parameterized(
            decoder, context, module_view, scratch_arena, &cursor,
            (loom_type_id_t)type_index, &fact));
        type_fact = &fact->base;
        break;
      }
      case LOOM_TYPE_REGISTER: {
        uint64_t payload0_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t payload0 = 0;
        uint64_t payload1 = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, &cursor, &payload0));
        uint64_t payload1_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, &cursor, &payload1));
        uint64_t has_value_type_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint8_t has_value_type = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &has_value_type));
        if (payload0 == 0) {
          return loom_bytecode_reader_emit_invalid_field(
              decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("register_payload0"), payload0_offset,
              IREE_SV("register_descriptor_set_stable_id_must_be_non_zero"));
        }
        if (((payload1 >> 16) & UINT32_MAX) == 0) {
          return loom_bytecode_reader_emit_invalid_field(
              decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("register_payload1"), payload1_offset,
              IREE_SV("register_unit_count_must_be_non_zero"));
        }
        if ((payload1 >> 48) != 0) {
          return loom_bytecode_reader_emit_invalid_field(
              decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("register_payload1"), payload1_offset,
              IREE_SV("register_payload_reserved_bits_must_be_zero"));
        }
        if (has_value_type > 1) {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("register_has_value_type"), has_value_type, 2,
              has_value_type_offset);
        }
        if (has_value_type) {
          uint64_t value_type_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t value_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              decoder, &cursor, &value_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_type_ref(
              decoder, value_type_id, type_index, value_type_offset));
          loom_bytecode_typed_register_fact_t* fact = NULL;
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(scratch_arena, sizeof(*fact), (void**)&fact));
          *fact = (loom_bytecode_typed_register_fact_t){
              .base =
                  {
                      .type_id = (loom_type_id_t)type_index,
                      .kind = LOOM_TYPE_REGISTER,
                  },
              .carrier_payload0 = payload0,
              .carrier_payload1 = payload1,
              .value_type_id = (loom_type_id_t)value_type_id,
          };
          type_fact = &fact->base;
        } else {
          direct_type = loom_type_register_payload(payload0, payload1);
        }
        break;
      }
      case LOOM_TYPE_STORAGE: {
        uint8_t space = 0;
        uint64_t space_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &space));
        if (space >= LOOM_STORAGE_SPACE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("storage_space"), space,
              LOOM_STORAGE_SPACE_COUNT_, space_offset);
        }
        direct_type = loom_type_storage((loom_storage_space_t)space);
        break;
      }
      case LOOM_TYPE_ENCODING: {
        uint8_t role = 0;
        uint64_t role_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &role));
        if (role >= LOOM_ENCODING_ROLE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("encoding_role"), role,
              LOOM_ENCODING_ROLE_COUNT_, role_offset);
        }
        direct_type = loom_type_encoding_with_role((loom_encoding_role_t)role);
        break;
      }
      case LOOM_TYPE_POOL: {
        uint8_t is_dynamic = 0;
        uint64_t dim_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, &cursor, &is_dynamic));
        if (is_dynamic == 0) {
          uint64_t size = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(decoder, &cursor, &size));
          if (size > LOOM_DIM_MAX_STATIC_SIZE) {
            return loom_bytecode_reader_emit_invalid_field(
                decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                IREE_SV("block_size"), dim_offset,
                IREE_SV("static_pool_block_size_exceeds_loom_maximum"));
          }
          direct_type = loom_type_pool(loom_dim_pack_static((int64_t)size));
        } else if (is_dynamic == 1) {
          direct_type =
              loom_type_pool(loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID));
        } else {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("is_dynamic"), is_dynamic, 2, dim_offset);
        }
        break;
      }
      case LOOM_TYPE_BUFFER:
        direct_type = loom_type_buffer();
        break;
      default:
        break;
    }
    if (type_fact) {
      type_fact->next = NULL;
      if (last_fact) {
        last_fact->next = type_fact;
      } else {
        first_fact = type_fact;
      }
      last_fact = type_fact;
    } else {
      plan_entry->direct_type = direct_type;
    }
  }
  module_view->types.facts = first_fact;
  return loom_bytecode_reader_expect_empty(decoder, &cursor, IREE_SV("TYPES"));
}
