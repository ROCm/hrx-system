// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/type.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/module_view.h"

iree_status_t loom_bytecode_type_decode_kind(
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
    case LOOM_BYTECODE_TYPE_GROUP:
      *out_kind = LOOM_TYPE_GROUP;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_REFERENCE:
      *out_kind = LOOM_TYPE_REFERENCE;
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

iree_status_t loom_bytecode_type_read_register_carrier(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t type_index,
    uint64_t out_payload[2]) {
  const uint64_t payload0_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &out_payload[0]));
  const uint64_t payload1_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &out_payload[1]));
  if (out_payload[0] == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("type"), type_index,
        IREE_SV("register_payload0"), payload0_offset,
        IREE_SV("register_descriptor_set_stable_id_must_be_non_zero"));
  }
  if (((out_payload[1] >> 16) & UINT32_MAX) == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("type"), type_index,
        IREE_SV("register_payload1"), payload1_offset,
        IREE_SV("register_unit_count_must_be_non_zero"));
  }
  if ((out_payload[1] >> 48) != 0) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("type"), type_index,
        IREE_SV("register_payload1"), payload1_offset,
        IREE_SV("register_payload_reserved_bits_must_be_zero"));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_read_structural_header(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, loom_type_kind_t kind,
    uint64_t type_index, iree_host_size_t string_count,
    loom_bytecode_structural_type_plan_t* out_plan, uint64_t out_payload[2]) {
  *out_plan = (loom_bytecode_structural_type_plan_t){0};
  out_payload[0] = out_payload[1] = 0;
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  if (kind == LOOM_TYPE_REGISTER) {
    IREE_RETURN_IF_ERROR(loom_bytecode_type_read_register_carrier(
        decoder, cursor, type_index, out_payload));
    out_plan->dependency_count = 1;
    return iree_ok_status();
  }

  if (kind == LOOM_TYPE_FUNCTION) {
    uint64_t argument_count = 0;
    uint64_t result_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &argument_count));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &result_count));
    if (argument_count > UINT16_MAX || result_count > UINT16_MAX) {
      return loom_bytecode_reader_emit_invalid_field(
          decoder, cursor->range_name, IREE_SV("type"), type_index,
          IREE_SV("signature_count"), offset,
          IREE_SV("function_signature_exceeds_runtime_field_width"));
    }
    const loom_func_type_data_t header = {
        .arg_count = (uint16_t)argument_count,
        .result_count = (uint16_t)result_count,
    };
    memcpy(out_payload, &header, sizeof(header));
    out_plan->dependency_count = (uint32_t)(argument_count + result_count);
  } else {
    uint64_t name_id = 0;
    uint64_t parameter_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &name_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &parameter_count));
    if (name_id >= string_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("dialect_type_name")),
          loom_param_u64(name_id),
          loom_param_u64(string_count),
      };
      return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_010,
                                             params, IREE_ARRAYSIZE(params),
                                             offset, 0);
    }
    if (parameter_count > UINT16_MAX) {
      return loom_bytecode_reader_emit_count_exceeds(
          decoder, IREE_SV("dialect_type_params"), parameter_count, UINT16_MAX,
          offset);
    }
    out_plan->name_id = (loom_string_id_t)name_id;
    out_plan->parameter_count = (uint16_t)parameter_count;
    out_plan->dependency_count = (uint32_t)parameter_count;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_type_emit_invalid_scoped_record(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, iree_string_view_t reason) {
  return loom_bytecode_reader_emit_invalid_field(
      decoder, cursor->range_name, IREE_SV("scoped_type"), 0, IREE_SV("record"),
      loom_bytecode_reader_cursor_absolute_position(cursor), reason);
}

iree_status_t loom_bytecode_type_read_scoped_shape(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, loom_type_kind_t kind,
    uint64_t value_count, iree_host_size_t encoding_count,
    loom_bytecode_scoped_shape_t* out_shape) {
  uint64_t element = 0, rank = 0, attachment = 0, encoding = 0, alignment = 0;
  if (kind != LOOM_TYPE_GROUP) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &element));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &rank));
  if (kind == LOOM_TYPE_GROUP) {
    if (rank == 0 || rank > LOOM_TYPE_MAX_RANK) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          decoder, cursor, IREE_SV("group_rank"));
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &attachment));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &encoding));
    if (kind == LOOM_TYPE_VIEW) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(decoder, cursor, &alignment));
    }
    if (element >= LOOM_SCALAR_TYPE_COUNT_ ||
        !loom_scalar_type_is_valid(element) || rank > LOOM_TYPE_MAX_RANK ||
        attachment > LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA ||
        (kind == LOOM_TYPE_VECTOR && (rank == 0 || attachment != 0))) {
      return loom_bytecode_type_emit_invalid_scoped_record(decoder, cursor,
                                                           IREE_SV("shape"));
    }
    if (alignment && !loom_type_view_alignment_is_valid(element, alignment)) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          decoder, cursor, IREE_SV("access_alignment"));
    }
    if (attachment == LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC) {
      if (!encoding || encoding > encoding_count) {
        return loom_bytecode_type_emit_invalid_scoped_record(
            decoder, cursor, IREE_SV("encoding"));
      }
    } else if (attachment == LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA) {
      if (encoding >= value_count) {
        return loom_bytecode_type_emit_invalid_scoped_record(
            decoder, cursor, IREE_SV("encoding_value"));
      }
    } else if (encoding) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          decoder, cursor, IREE_SV("unexpected_encoding"));
    }
  }
  uint16_t dynamic_axes = 0;
  for (uint64_t i = 0; i < rank; ++i) {
    uint64_t dynamic = 0, payload = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &dynamic));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(decoder, cursor, &payload));
    if (dynamic > 1 || (dynamic && payload > value_count) ||
        (!dynamic && loom_dim_is_dynamic(payload))) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          decoder, cursor,
          kind == LOOM_TYPE_GROUP ? IREE_SV("group_dimension")
                                  : IREE_SV("dimension"));
    }
    out_shape->dimensions[i] = payload;
    dynamic_axes |= (uint16_t)(dynamic << i);
  }
  out_shape->encoding_ordinal = encoding;
  out_shape->dynamic_axes = dynamic_axes;
  out_shape->rank = (uint8_t)rank;
  out_shape->element_type = (uint8_t)element;
  out_shape->attachment = (uint8_t)attachment;
  out_shape->access_alignment = (uint8_t)alignment;
  return iree_ok_status();
}

static loom_bytecode_attribute_materializer_t
loom_bytecode_type_attribute_materializer(
    loom_bytecode_type_materializer_t* materializer) {
  return (loom_bytecode_attribute_materializer_t){
      .decoder = materializer->decoder,
      .context = materializer->context,
      .module_view = materializer->module_view,
      .scratch_arena = materializer->scratch_arena,
      .output_module = materializer->output_module,
  };
}

static iree_status_t loom_bytecode_type_materialize_parameterized(
    loom_bytecode_type_materializer_t* materializer,
    const loom_bytecode_parameterized_type_fact_t* fact, loom_type_t* out_type,
    loom_type_id_t* out_type_id) {
  const loom_parameterized_type_descriptor_t* descriptor = fact->descriptor;
  loom_attribute_t* parameters = NULL;
  if (descriptor->parameter_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        materializer->scratch_arena, descriptor->parameter_count,
        sizeof(*parameters), (void**)&parameters));
    memset(parameters, 0, descriptor->parameter_count * sizeof(*parameters));
  }

  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      materializer->bytecode.data + (iree_host_size_t)fact->parameters_offset,
      fact->parameters_length, fact->parameters_offset, IREE_SV("TYPES"),
      &cursor);
  loom_bytecode_attribute_materializer_t attribute_materializer =
      loom_bytecode_type_attribute_materializer(materializer);
  for (uint8_t i = 0; i < fact->present_count; ++i) {
    uint64_t unused_parameter_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        materializer->decoder, &cursor, &unused_parameter_name_id));
    const uint8_t parameter_index = fact->parameter_indices[i];
    IREE_ASSERT(parameter_index < descriptor->parameter_count);

    uint8_t encoded_value_kind = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(
        materializer->decoder, &cursor, &encoded_value_kind));
    IREE_ASSERT(encoded_value_kind < LOOM_BYTECODE_ATTR_COUNT);
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_decode_named(
        &attribute_materializer, &cursor,
        &descriptor->parameter_descriptors[parameter_index],
        (loom_bytecode_attr_kind_t)encoded_value_kind, fact->base.type_id,
        &parameters[parameter_index]));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      materializer->decoder, &cursor,
      IREE_SV("parameterized_type_parameters")));

  return loom_module_make_parameterized_type(
      materializer->output_module, descriptor, parameters,
      descriptor->parameter_count, out_type, out_type_id);
}

iree_status_t loom_bytecode_type_materialize_structural(
    const loom_bytecode_structural_type_plan_t* plan,
    const loom_bytecode_structural_type_fact_t* fact,
    const loom_type_id_t* dependency_ids, loom_module_t* module,
    loom_type_id_t* out_type_id) {
  if (fact->base.kind == LOOM_TYPE_FUNCTION) {
    loom_func_type_data_t metadata;
    memcpy(&metadata, fact->payload_prefix, sizeof(metadata));
    return loom_module_intern_topological_type_id(
        module, loom_type_function(&metadata), dependency_ids,
        plan->dependency_count, out_type_id);
  }
  if (fact->base.kind == LOOM_TYPE_REGISTER) {
    const loom_register_type_data_t metadata = {
        .carrier_payload0 = fact->payload_prefix[0],
        .carrier_payload1 = fact->payload_prefix[1],
    };
    return loom_module_intern_topological_type_id(
        module, loom_type_register_payload_with_value_type(&metadata),
        dependency_ids, plan->dependency_count, out_type_id);
  }
  return loom_module_intern_topological_type_id(
      module, loom_type_dialect(plan->name_id, plan->parameter_count, NULL),
      dependency_ids, plan->dependency_count, out_type_id);
}

iree_status_t loom_bytecode_type_materialize_prefix(
    loom_bytecode_type_materializer_t* materializer,
    iree_host_size_t type_count) {
  loom_bytecode_type_fact_t* fact = materializer->next_fact;
  for (iree_host_size_t type_index = materializer->position;
       type_index < type_count; ++type_index) {
    IREE_ASSERT(!fact || fact->type_id >= type_index);
    loom_type_t type = {0};
    loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
    iree_status_t status = iree_ok_status();
    const iree_host_size_t previous_type_count =
        materializer->output_module->types.count;
    loom_bytecode_type_plan_entry_t* entry =
        &materializer->module_view->types.entries[type_index];
    if (fact && fact->type_id == type_index) {
      if (fact->kind == LOOM_TYPE_PARAMETERIZED) {
        const iree_arena_checkpoint_t checkpoint =
            iree_arena_checkpoint_save(materializer->scratch_arena);
        status = loom_bytecode_type_materialize_parameterized(
            materializer, (const loom_bytecode_parameterized_type_fact_t*)fact,
            &type, &type_id);
        iree_arena_checkpoint_restore(&checkpoint);
      } else if (fact->kind == LOOM_TYPE_REFERENCE) {
        const loom_bytecode_reference_type_fact_t* reference =
            (const loom_bytecode_reference_type_fact_t*)fact;
        const loom_symbol_id_t declaration_id =
            loom_symbol_map_find(&materializer->module_view->symbols.map,
                                 reference->declaration_name_id);
        type =
            loom_type_reference(reference->ownership, reference->namespace_id,
                                reference->name_id, declaration_id);
      } else {
        loom_bytecode_structural_type_fact_t* structural_fact =
            (loom_bytecode_structural_type_fact_t*)fact;
        for (uint32_t i = 0; i < entry->structural.dependency_count; ++i) {
          structural_fact->type_ids[i] =
              materializer->module_view->types
                  .entries[structural_fact->type_ids[i]]
                  .completed_type;
        }
        status = loom_bytecode_type_materialize_structural(
            &entry->structural, structural_fact, structural_fact->type_ids,
            materializer->output_module, &type_id);
      }
      fact = fact->next;
    } else {
      type = entry->direct_type;
    }
    if (iree_status_is_ok(status) && type_id == LOOM_TYPE_ID_INVALID) {
      status = loom_module_intern_topological_type_id(
          materializer->output_module, type, NULL, 0, &type_id);
    }
    if (iree_status_is_ok(status) && type_id < previous_type_count) {
      status = loom_bytecode_reader_emit_invalid_field(
          materializer->decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
          IREE_SV("type"),
          materializer->module_view->types.entries[type_index].bytecode_offset,
          IREE_SV("type_table_must_be_deduplicated_and_topologically_ordered"));
    }
    IREE_RETURN_IF_ERROR(status);
    entry->completed_type = type_id;
  }
  materializer->position = (loom_type_id_t)type_count;
  materializer->next_fact = fact;
  return iree_ok_status();
}
