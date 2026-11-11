// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/type_validator.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/reference_binding.h"
#include "loom/format/bytecode/reader/type.h"
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
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_count_bytes(
      decoder, cursor, IREE_SV("parameterized_type_parameters"), present_count,
      /*minimum_entry_length=*/2,
      /*minimum_trailing_length=*/0, present_count_offset));

  const iree_host_size_t parameters_position = cursor->cursor.position;
  loom_bytecode_parameterized_type_fact_t* fact = NULL;
  if (out_fact) {
    iree_host_size_t allocation_size = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        sizeof(loom_bytecode_parameterized_type_fact_t), &allocation_size,
        IREE_STRUCT_FIELD_FAM((iree_host_size_t)present_count, uint8_t)));
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
  }

  loom_bytecode_attribute_validator_t attribute_validator =
      loom_bytecode_type_plan_attribute_validator(decoder, context,
                                                  module_view);
  uint8_t next_parameter_index = 0;
  bool missing_required_parameter = false;
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
    // Names advance monotonically through the descriptor. Record required
    // gaps during lookup rather than recovering presence in a later scan.
    const loom_attr_descriptor_t* parameter_descriptor = NULL;
    for (; next_parameter_index < descriptor->parameter_count;
         ++next_parameter_index) {
      const loom_attr_descriptor_t* candidate =
          &descriptor->parameter_descriptors[next_parameter_index];
      if (iree_string_view_equal(loom_attr_descriptor_name(candidate),
                                 parameter_name)) {
        parameter_descriptor = candidate;
        break;
      }
      missing_required_parameter |=
          !iree_any_bit_set(candidate->flags, LOOM_ATTR_OPTIONAL);
    }
    if (!parameter_descriptor) {
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
    if (fact) {
      fact->parameter_indices[i] = next_parameter_index;
    }
    ++next_parameter_index;

    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_attribute_read_kind(decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_validate_type_parameter(
        &attribute_validator, cursor, parameter_descriptor, value_kind,
        type_id));
  }

  for (; next_parameter_index < descriptor->parameter_count;
       ++next_parameter_index) {
    missing_required_parameter |= !iree_any_bit_set(
        descriptor->parameter_descriptors[next_parameter_index].flags,
        LOOM_ATTR_OPTIONAL);
  }
  // Report absence only after validating every present value, preserving the
  // wire-error precedence independently of whether construction is retained.
  if (missing_required_parameter) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), type_id,
        IREE_SV("parameter_name"), present_count_offset,
        IREE_SV("parameterized_type_required_parameter_is_absent"));
  }

  if (out_fact) {
    fact->parameters_length = cursor->cursor.position - parameters_position;
    *out_fact = fact;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_build_shaped(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, loom_type_kind_t kind,
    loom_scalar_type_t element_type, uint8_t rank, uint8_t attachment,
    uint64_t encoding_instance, uint8_t alignment, const uint64_t* dims,
    uint64_t offset, loom_type_t* out_type) {
  if (!loom_scalar_type_is_valid(element_type)) {
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
      return loom_bytecode_reader_emit_invalid_field(
          decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
          IREE_SV("encoding_attachment"), offset,
          IREE_SV("ssa_encoding_requires_a_scoped_type"));
    default:
      return loom_bytecode_reader_emit_enum_value(
          decoder, IREE_SV("encoding_attachment"), attachment,
          LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA + 1, offset);
  }
  if (kind == LOOM_TYPE_VECTOR && encoding_id != 0) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
        IREE_SV("encoding_attachment"), offset,
        IREE_SV("vector_types_must_not_carry_encoding_or_layout_attachments"));
  }
  if (alignment &&
      !loom_type_view_alignment_is_valid(element_type, alignment)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), 0,
        IREE_SV("access_alignment"), offset,
        IREE_SV(
            "view_alignment_must_be_a_power_of_two_no_greater_than_natural"));
  }
  if (!out_type) {
    return iree_ok_status();
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
  if (kind == LOOM_TYPE_VIEW) {
    type = loom_type_view_with_alignment(type, alignment);
  }
  *out_type = type;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_build_group(
    loom_bytecode_reader_decoder_t* decoder,
    iree_arena_allocator_t* scratch_arena, uint8_t rank,
    const uint64_t* dimensions, uint64_t offset, loom_type_t* out_type) {
  if (rank == 0 || rank > LOOM_TYPE_MAX_RANK) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("group_rank_must_be_between_one_and_loom_type_maximum"));
  }
  if (!out_type) {
    return iree_ok_status();
  }

  bool all_static = true;
  for (uint8_t i = 0; i < rank; ++i) {
    all_static &= !loom_dim_is_dynamic(dimensions[i]);
  }
  uint8_t flags = rank <= 2 ? LOOM_TYPE_FLAG_INLINE_DIMS : 0;
  if (all_static) {
    flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  }
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(LOOM_TYPE_GROUP, 0, rank, flags);
  if (rank <= 2) {
    for (uint8_t i = 0; i < rank; ++i) {
      type.dims[i] = dimensions[i];
    }
  } else {
    loom_overflow_dim_t* retained_dimensions = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, rank, sizeof(*retained_dimensions),
        (void**)&retained_dimensions));
    memcpy(retained_dimensions, dimensions,
           (iree_host_size_t)rank * sizeof(*dimensions));
    type.dims[0] = (uint64_t)(uintptr_t)retained_dimensions;
  }
  *out_type = type;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_type_plan_decode_entry(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    loom_bytecode_reader_cursor_t* cursor, loom_type_id_t type_index,
    loom_bytecode_reference_bindings_t* reference_bindings,
    loom_bytecode_type_plan_entry_t* out_plan_entry,
    loom_bytecode_type_fact_t** out_fact) {
  const uint64_t type_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  if (out_plan_entry) {
    *out_plan_entry = (loom_bytecode_type_plan_entry_t){
        .bytecode_offset = type_offset,
    };
    *out_fact = NULL;
  }
  uint8_t kind_byte = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(decoder, cursor, &kind_byte));
  loom_type_kind_t kind = LOOM_TYPE_NONE;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_type_decode_kind(decoder, kind_byte, type_offset, &kind));

  loom_type_t direct_type = {0};
  loom_bytecode_type_fact_t* type_fact = NULL;
  switch (kind) {
    case LOOM_TYPE_NONE:
      direct_type = loom_type_none();
      break;
    case LOOM_TYPE_SCALAR: {
      uint8_t element_type = 0;
      uint64_t element_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &element_type));
      if (!loom_scalar_type_is_valid(element_type)) {
        return loom_bytecode_reader_emit_enum_value(
            decoder, IREE_SV("scalar_type"), element_type,
            LOOM_SCALAR_TYPE_COUNT_, element_offset);
      }
      direct_type = loom_type_scalar((loom_scalar_type_t)element_type);
      break;
    }
    case LOOM_TYPE_REFERENCE: {
      uint8_t ownership = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &ownership));
      if (ownership > LOOM_REFERENCE_OWNERSHIP_BORROWED) {
        return loom_bytecode_reader_emit_enum_value(
            decoder, IREE_SV("reference_ownership"), ownership, 2,
            type_offset + 1);
      }
      uint64_t string_ids[3] = {0};
      iree_status_t status = iree_ok_status();
      for (uint8_t i = 0;
           i < IREE_ARRAYSIZE(string_ids) && iree_status_is_ok(status); ++i) {
        const uint64_t offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        status =
            loom_bytecode_reader_read_uvarint(decoder, cursor, &string_ids[i]);
        iree_string_view_t unused = iree_string_view_empty();
        if (iree_status_is_ok(status)) {
          status = loom_bytecode_type_plan_validate_string_ref(
              decoder, module_view, string_ids[i], IREE_SV("reference_key"),
              offset, &unused);
        }
      }
      IREE_RETURN_IF_ERROR(status);
      if (reference_bindings) {
        IREE_RETURN_IF_ERROR(loom_bytecode_reference_binding_expect(
            decoder, scratch_arena, reference_bindings,
            (loom_string_id_t)string_ids[2], (loom_string_id_t)string_ids[0],
            (loom_string_id_t)string_ids[1], type_offset));
      }
      if (out_plan_entry) {
        loom_bytecode_reference_type_fact_t* reference = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            scratch_arena, sizeof(*reference), (void**)&reference));
        *reference = (loom_bytecode_reference_type_fact_t){
            .base = {.type_id = type_index, .kind = LOOM_TYPE_REFERENCE},
            .namespace_id = (loom_string_id_t)string_ids[0],
            .name_id = (loom_string_id_t)string_ids[1],
            .declaration_name_id = (loom_string_id_t)string_ids[2],
            .ownership = ownership,
        };
        type_fact = &reference->base;
      }
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
          loom_bytecode_reader_read_u8(decoder, cursor, &element_type));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &rank));
      uint64_t attachment_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &attachment));
      uint64_t encoding_instance = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          decoder, cursor, &encoding_instance));
      uint8_t alignment = 0;
      if (kind == LOOM_TYPE_VIEW) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, cursor, &alignment));
      }
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
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, cursor, &is_dynamic));
        if (is_dynamic == 0) {
          uint64_t size = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(decoder, cursor, &size));
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
          (loom_scalar_type_t)element_type, rank, attachment, encoding_instance,
          alignment, dims, attachment_offset,
          out_plan_entry ? &direct_type : NULL));
      break;
    }
    case LOOM_TYPE_GROUP: {
      uint8_t rank = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &rank));
      if (rank == 0 || rank > LOOM_TYPE_MAX_RANK) {
        return loom_bytecode_reader_emit_invalid_field(
            decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
            IREE_SV("rank"), type_offset,
            IREE_SV("group_rank_must_be_between_one_and_loom_type_maximum"));
      }
      uint64_t dimensions[LOOM_TYPE_MAX_RANK] = {0};
      for (uint8_t i = 0; i < rank; ++i) {
        uint8_t is_dynamic = 0;
        const uint64_t dimension_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, cursor, &is_dynamic));
        if (is_dynamic == 0) {
          uint64_t size = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(decoder, cursor, &size));
          if (size > LOOM_DIM_MAX_STATIC_SIZE) {
            return loom_bytecode_reader_emit_invalid_field(
                decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                IREE_SV("dim_size"), dimension_offset,
                IREE_SV("static_dimension_exceeds_loom_maximum"));
          }
          dimensions[i] = loom_dim_pack_static((int64_t)size);
        } else if (is_dynamic == 1) {
          dimensions[i] = loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID);
        } else {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("is_dynamic"), is_dynamic, 2, dimension_offset);
        }
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_build_group(
          decoder, scratch_arena, rank, dimensions, type_offset,
          out_plan_entry ? &direct_type : NULL));
      break;
    }
    case LOOM_TYPE_FUNCTION:
    case LOOM_TYPE_DIALECT:
    case LOOM_TYPE_REGISTER: {
      loom_bytecode_structural_type_plan_t plan;
      uint64_t payload[2];
      IREE_RETURN_IF_ERROR(loom_bytecode_type_read_structural_header(
          decoder, cursor, kind, type_index, module_view->strings.count, &plan,
          payload));
      if (kind == LOOM_TYPE_REGISTER) {
        const uint64_t presence_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint8_t has_value_type = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(decoder, cursor, &has_value_type));
        if (has_value_type > 1) {
          return loom_bytecode_reader_emit_enum_value(
              decoder, IREE_SV("register_has_value_type"), has_value_type, 2,
              presence_offset);
        }
        if (!has_value_type) {
          direct_type = loom_type_register_payload(payload[0], payload[1]);
          break;
        }
      } else if (kind == LOOM_TYPE_DIALECT && !plan.parameter_count) {
        direct_type = loom_type_dialect_opaque(plan.name_id);
        break;
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_count_bytes(
          decoder, cursor, IREE_SV("structural_type_children"),
          plan.dependency_count,
          /*minimum_entry_length=*/1, /*minimum_trailing_length=*/0,
          type_offset));
      loom_bytecode_structural_type_fact_t* fact = NULL;
      if (out_plan_entry) {
        iree_host_size_t allocation_size = 0;
        IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
            sizeof(loom_bytecode_structural_type_fact_t), &allocation_size,
            IREE_STRUCT_FIELD_FAM(plan.dependency_count, loom_type_id_t)));
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(scratch_arena, allocation_size, (void**)&fact));
        *fact = (loom_bytecode_structural_type_fact_t){
            .base = {.type_id = type_index, .kind = kind},
            .payload_prefix = {payload[0], payload[1]},
        };
        out_plan_entry->structural = plan;
      }
      for (uint32_t i = 0; i < plan.dependency_count; ++i) {
        const uint64_t reference_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t type_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(decoder, cursor, &type_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_type_ref(
            decoder, type_id, type_index, reference_offset));
        if (fact) {
          fact->type_ids[i] = (loom_type_id_t)type_id;
        }
      }
      if (fact) {
        type_fact = &fact->base;
      }
      break;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      loom_bytecode_parameterized_type_fact_t* fact = NULL;
      IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_validate_parameterized(
          decoder, context, module_view, scratch_arena, cursor, type_index,
          out_plan_entry ? &fact : NULL));
      if (fact) {
        type_fact = &fact->base;
      }
      break;
    }
    case LOOM_TYPE_STORAGE: {
      uint8_t space = 0;
      uint64_t space_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &space));
      if (space >= LOOM_STORAGE_SPACE_COUNT_) {
        return loom_bytecode_reader_emit_enum_value(
            decoder, IREE_SV("storage_space"), space, LOOM_STORAGE_SPACE_COUNT_,
            space_offset);
      }
      direct_type = loom_type_storage((loom_storage_space_t)space);
      break;
    }
    case LOOM_TYPE_ENCODING: {
      uint8_t role = 0;
      uint64_t role_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(decoder, cursor, &role));
      if (role >= LOOM_ENCODING_ROLE_COUNT_) {
        return loom_bytecode_reader_emit_enum_value(
            decoder, IREE_SV("encoding_role"), role, LOOM_ENCODING_ROLE_COUNT_,
            role_offset);
      }
      direct_type = loom_type_encoding_with_role((loom_encoding_role_t)role);
      break;
    }
    case LOOM_TYPE_POOL:
      direct_type = loom_type_pool();
      break;
    case LOOM_TYPE_BUFFER:
      direct_type = loom_type_buffer();
      break;
    default:
      break;
  }
  if (type_fact) {
    type_fact->next = NULL;
    *out_fact = type_fact;
  } else if (out_plan_entry) {
    out_plan_entry->direct_type = direct_type;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_plan_decode_indexed_entry(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, loom_type_id_t type_index,
    iree_const_byte_span_t entry_bytes, uint64_t entry_absolute_offset,
    loom_bytecode_type_plan_entry_t* out_plan_entry,
    loom_bytecode_type_fact_t** out_fact) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      entry_bytes.data, entry_bytes.data_length, entry_absolute_offset,
      IREE_SV("TYPES"), &cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_entry(
      decoder, context, module_view, scratch_arena, &cursor, type_index,
      /*reference_bindings=*/NULL, out_plan_entry, out_fact));
  return loom_bytecode_reader_expect_empty(decoder, &cursor,
                                           IREE_SV("type_entry"));
}

static iree_status_t loom_bytecode_type_table_begin(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_module_view_t* module_view,
    iree_const_byte_span_t section_bytes, uint64_t section_absolute_offset,
    loom_bytecode_reader_cursor_t* out_cursor) {
  loom_bytecode_reader_cursor_initialize(
      section_bytes.data, section_bytes.data_length, section_absolute_offset,
      IREE_SV("TYPES"), out_cursor);

  const uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(out_cursor);
  uint64_t count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, out_cursor, &count));
  if (count > LOOM_BYTECODE_MAX_TYPE_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        decoder, IREE_SV("TYPES"), count, LOOM_BYTECODE_MAX_TYPE_COUNT,
        count_offset);
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_count_bytes(
      decoder, out_cursor, IREE_SV("TYPES"), count,
      /*minimum_entry_length=*/1, /*minimum_trailing_length=*/0, count_offset));
  module_view->types.count = (iree_host_size_t)count;
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_validation_begin(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_const_byte_span_t section_bytes, uint64_t section_absolute_offset,
    loom_bytecode_type_retention_t retention, iree_arena_allocator_t* arena,
    loom_bytecode_type_validation_t* out_validation) {
  *out_validation = (loom_bytecode_type_validation_t){
      .decoder = decoder,
      .context = context,
      .module_view = module_view,
      .arena = arena,
      .retention = retention,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_type_table_begin(
      decoder, module_view, section_bytes, section_absolute_offset,
      &out_validation->cursor));
  if (module_view->types.count > 0) {
    if (retention == LOOM_BYTECODE_TYPE_RETAIN_PLAN) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, module_view->types.count,
                                    sizeof(loom_bytecode_type_plan_entry_t),
                                    (void**)&module_view->types.entries));
    } else if (retention == LOOM_BYTECODE_TYPE_RETAIN_RANGES) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, module_view->types.count,
          sizeof(loom_bytecode_table_entry_metadata_t),
          (void**)&out_validation->entries));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_validation_advance(
    loom_bytecode_type_validation_t* validation, iree_host_size_t type_count) {
  // Specialize retention outside the entry loop. Ordinary validation and
  // indexing never allocate construction payloads or branch on that policy.
  switch (validation->retention) {
    case LOOM_BYTECODE_TYPE_RETAIN_NONE:
      for (loom_type_id_t i = validation->position; i < type_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_entry(
            validation->decoder, validation->context, validation->module_view,
            validation->arena, &validation->cursor, i,
            &validation->module_view->reference_bindings,
            /*out_plan_entry=*/NULL, /*out_fact=*/NULL));
      }
      break;
    case LOOM_BYTECODE_TYPE_RETAIN_RANGES:
      for (loom_type_id_t i = validation->position; i < type_count; ++i) {
        const uint64_t offset =
            loom_bytecode_reader_cursor_absolute_position(&validation->cursor);
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_entry(
            validation->decoder, validation->context, validation->module_view,
            validation->arena, &validation->cursor, i,
            &validation->module_view->reference_bindings,
            /*out_plan_entry=*/NULL, /*out_fact=*/NULL));
        validation->entries[i] = (loom_bytecode_table_entry_metadata_t){
            .entry_offset = offset,
            .entry_length = loom_bytecode_reader_cursor_absolute_position(
                                &validation->cursor) -
                            offset,
        };
      }
      break;
    case LOOM_BYTECODE_TYPE_RETAIN_PLAN:
      for (loom_type_id_t i = validation->position; i < type_count; ++i) {
        loom_bytecode_type_fact_t* fact = NULL;
        IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_entry(
            validation->decoder, validation->context, validation->module_view,
            validation->arena, &validation->cursor, i,
            &validation->module_view->reference_bindings,
            &validation->module_view->types.entries[i], &fact));
        if (fact) {
          if (validation->last_fact) {
            validation->last_fact->next = fact;
          } else {
            validation->module_view->types.facts = fact;
          }
          validation->last_fact = fact;
        }
      }
      break;
  }
  validation->position = (loom_type_id_t)type_count;
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_validation_finish(
    loom_bytecode_type_validation_t* validation) {
  IREE_RETURN_IF_ERROR(loom_bytecode_type_validation_advance(
      validation, validation->module_view->types.count));
  return loom_bytecode_reader_expect_empty(
      validation->decoder, &validation->cursor, IREE_SV("TYPES"));
}
