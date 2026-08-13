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
    const loom_bytecode_parameterized_type_fact_t* fact,
    loom_type_t* out_type) {
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
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_materialize_named(
        &attribute_materializer, &cursor,
        &descriptor->parameter_descriptors[parameter_index],
        (loom_bytecode_attr_kind_t)encoded_value_kind,
        &parameters[parameter_index], fact->base.type_id));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      materializer->decoder, &cursor,
      IREE_SV("parameterized_type_parameters")));

  return loom_module_make_parameterized_type(
      materializer->output_module, descriptor, parameters,
      descriptor->parameter_count, out_type);
}

static iree_status_t loom_bytecode_type_materialize_fact(
    loom_bytecode_type_materializer_t* materializer,
    const loom_bytecode_type_fact_t* fact, loom_type_t* out_type) {
  switch (fact->kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_bytecode_function_type_fact_t* function_fact =
          (const loom_bytecode_function_type_fact_t*)fact;
      const iree_host_size_t type_count =
          (iree_host_size_t)function_fact->argument_count +
          function_fact->result_count;
      iree_host_size_t allocation_size = 0;
      IREE_RETURN_IF_ERROR(
          IREE_STRUCT_LAYOUT(sizeof(loom_func_type_data_t), &allocation_size,
                             IREE_STRUCT_FIELD_FAM(type_count, loom_type_t)));
      loom_func_type_data_t* data = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(materializer->scratch_arena,
                                               allocation_size, (void**)&data));
      data->arg_count = function_fact->argument_count;
      data->result_count = function_fact->result_count;
      data->reserved = 0;
      for (iree_host_size_t i = 0; i < type_count; ++i) {
        data->types[i] = materializer->output_module->types
                             .entries[function_fact->type_ids[i]];
      }
      *out_type = loom_type_function(data);
      return iree_ok_status();
    }
    case LOOM_TYPE_DIALECT: {
      const loom_bytecode_dialect_type_fact_t* dialect_fact =
          (const loom_bytecode_dialect_type_fact_t*)fact;
      loom_type_t* parameters = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          materializer->scratch_arena, dialect_fact->parameter_count,
          sizeof(*parameters), (void**)&parameters));
      for (uint16_t i = 0; i < dialect_fact->parameter_count; ++i) {
        parameters[i] = materializer->output_module->types
                            .entries[dialect_fact->type_ids[i]];
      }
      *out_type = loom_type_dialect(dialect_fact->name_id,
                                    dialect_fact->parameter_count, parameters);
      return iree_ok_status();
    }
    case LOOM_TYPE_PARAMETERIZED:
      return loom_bytecode_type_materialize_parameterized(
          materializer, (const loom_bytecode_parameterized_type_fact_t*)fact,
          out_type);
    case LOOM_TYPE_REGISTER: {
      const loom_bytecode_typed_register_fact_t* register_fact =
          (const loom_bytecode_typed_register_fact_t*)fact;
      loom_register_type_data_t* data = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(materializer->scratch_arena,
                                               sizeof(*data), (void**)&data));
      *data = (loom_register_type_data_t){
          .carrier_payload0 = register_fact->carrier_payload0,
          .carrier_payload1 = register_fact->carrier_payload1,
          .value_type = materializer->output_module->types
                            .entries[register_fact->value_type_id],
      };
      *out_type = loom_type_register_payload_with_value_type(data);
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("validated type fact kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_bytecode_type_materialize(
    loom_bytecode_type_materializer_t* materializer) {
  const loom_bytecode_type_fact_t* fact =
      materializer->module_view->types.facts;
  for (iree_host_size_t type_index = 0;
       type_index < materializer->module_view->types.count; ++type_index) {
    IREE_ASSERT(!fact || fact->type_id >= type_index);
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(materializer->scratch_arena);
    loom_type_t type = {0};
    iree_status_t status = iree_ok_status();
    if (fact && fact->type_id == type_index) {
      status = loom_bytecode_type_materialize_fact(materializer, fact, &type);
      fact = fact->next;
    } else {
      type = materializer->module_view->types.entries[type_index].direct_type;
    }
    loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
    if (iree_status_is_ok(status)) {
      status = loom_module_intern_type_id(materializer->output_module, type,
                                          &type_id);
    }
    if (iree_status_is_ok(status) && type_id != type_index) {
      status = loom_bytecode_reader_emit_invalid_field(
          materializer->decoder, IREE_SV("TYPES"), IREE_SV("type"), type_index,
          IREE_SV("type"),
          materializer->module_view->types.entries[type_index].bytecode_offset,
          IREE_SV("type_table_must_be_deduplicated_and_topologically_ordered"));
    }
    iree_arena_checkpoint_restore(&checkpoint);
    IREE_RETURN_IF_ERROR(status);
  }
  IREE_ASSERT(!fact);
  return iree_ok_status();
}
