// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_tables.h"

#include <string.h>

#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/selected_attribute.h"
#include "loom/format/bytecode/reader/type_plan.h"
#include "loom/format/bytecode/reader/type_validator.h"

static loom_bytecode_selected_projection_domain_t
loom_bytecode_selected_table_projection_domain(
    loom_bytecode_selected_table_kind_t table_kind) {
  switch (table_kind) {
    case LOOM_BYTECODE_SELECTED_TABLE_ENCODING:
      return LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING;
    case LOOM_BYTECODE_SELECTED_TABLE_TYPE:
      return LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE;
    case LOOM_BYTECODE_SELECTED_TABLE_LOCATION:
      return LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION;
  }
  IREE_ASSERT_UNREACHABLE("selected table kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_table_push(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal) {
  if (materializer->worklist.count == materializer->worklist.capacity) {
    iree_host_size_t minimum_capacity = 0;
    if (!iree_host_size_checked_add(materializer->worklist.count, 1,
                                    &minimum_capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "selected table worklist capacity overflow");
    }
    IREE_RETURN_IF_ERROR(
        iree_allocator_grow_array(materializer->allocator, minimum_capacity,
                                  sizeof(*materializer->worklist.values),
                                  &materializer->worklist.capacity,
                                  (void**)&materializer->worklist.values));
  }
  materializer->worklist.values[materializer->worklist.count++] =
      (loom_bytecode_selected_table_frame_t){
          .table_kind = table_kind,
          .source_ordinal = source_ordinal,
      };
  return iree_ok_status();
}

static iree_const_byte_span_t loom_bytecode_selected_table_entry_span(
    const loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_table_entry_metadata_t* entry) {
  IREE_ASSERT(entry->entry_offset <= materializer->bytecode.data_length);
  IREE_ASSERT(entry->entry_length <=
              materializer->bytecode.data_length - entry->entry_offset);
  return iree_make_const_byte_span(
      materializer->bytecode.data + (iree_host_size_t)entry->entry_offset,
      (iree_host_size_t)entry->entry_length);
}

static iree_status_t loom_bytecode_selected_table_project_source(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, loom_source_id_t* out_target_source_id) {
  IREE_ASSERT(source_ordinal < materializer->metadata->sources.count);
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE, source_ordinal,
          &target_id)) {
    *out_target_source_id = (loom_source_id_t)target_id;
    return iree_ok_status();
  }
  loom_source_id_t target_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_append_source(
      materializer->output_module,
      materializer->metadata->sources.values[source_ordinal],
      &target_source_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_projection_insert(
      &materializer->projection,
      LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE, source_ordinal,
      target_source_id));
  *out_target_source_id = target_source_id;
  return iree_ok_status();
}

void loom_bytecode_selected_table_materializer_initialize(
    loom_bytecode_reader_decoder_t* decoder, iree_const_byte_span_t bytecode,
    loom_context_t* context, const loom_bytecode_module_metadata_t* metadata,
    iree_arena_allocator_t* scratch_arena, loom_module_t* output_module,
    iree_allocator_t allocator,
    loom_bytecode_selected_table_materializer_t* out_materializer) {
  *out_materializer = (loom_bytecode_selected_table_materializer_t){
      .decoder = decoder,
      .bytecode = bytecode,
      .context = context,
      .metadata = metadata,
      .scratch_arena = scratch_arena,
      .output_module = output_module,
      .allocator = allocator,
  };
  loom_bytecode_selected_projection_initialize(allocator,
                                               &out_materializer->projection);
}

void loom_bytecode_selected_table_materializer_deinitialize(
    loom_bytecode_selected_table_materializer_t* materializer) {
  loom_bytecode_selected_projection_deinitialize(&materializer->projection);
  iree_allocator_free(materializer->allocator, materializer->worklist.values);
  memset(materializer, 0, sizeof(*materializer));
}

iree_status_t loom_bytecode_selected_table_bind_symbol(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, uint16_t target_symbol_id) {
  IREE_ASSERT(source_name_ordinal < materializer->metadata->strings.count);
  return loom_bytecode_selected_projection_insert(
      &materializer->projection,
      LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME, source_name_ordinal,
      target_symbol_id);
}

iree_status_t loom_bytecode_selected_table_intern_string(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_string_ordinal, loom_string_id_t* out_target_string_id) {
  IREE_ASSERT(source_string_ordinal < materializer->metadata->strings.count);
  return loom_module_intern_string(
      materializer->output_module,
      materializer->metadata->strings.values[source_string_ordinal],
      out_target_string_id);
}

bool loom_bytecode_selected_table_lookup_symbol(
    const loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, loom_symbol_ref_t* out_target_symbol_ref) {
  uint32_t target_symbol_id = 0;
  if (!loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME,
          source_name_ordinal, &target_symbol_id)) {
    return false;
  }
  *out_target_symbol_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = (uint16_t)target_symbol_id,
  };
  return true;
}

iree_status_t loom_bytecode_selected_table_project_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint16_t source_encoding_id, uint16_t* out_target_encoding_id,
    loom_bytecode_selected_reference_state_t* out_state) {
  IREE_ASSERT(source_encoding_id > 0);
  const uint32_t source_ordinal = source_encoding_id - 1;
  IREE_ASSERT(source_ordinal < materializer->metadata->encodings.count);
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING, source_ordinal,
          &target_id)) {
    *out_target_encoding_id = (uint16_t)target_id;
    *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
    return iree_ok_status();
  }
  *out_target_encoding_id = 0;
  *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED;
  return loom_bytecode_selected_table_push(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_ENCODING, source_ordinal);
}

iree_status_t loom_bytecode_selected_table_project_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_type_id_t source_type_id, loom_type_id_t* out_target_type_id,
    loom_bytecode_selected_reference_state_t* out_state) {
  IREE_ASSERT(source_type_id < materializer->metadata->types.count);
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE, source_type_id,
          &target_id)) {
    *out_target_type_id = (loom_type_id_t)target_id;
    *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
    return iree_ok_status();
  }
  *out_target_type_id = LOOM_TYPE_ID_INVALID;
  *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED;
  return loom_bytecode_selected_table_push(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_TYPE, source_type_id);
}

iree_status_t loom_bytecode_selected_table_project_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id,
    loom_bytecode_selected_reference_state_t* out_state) {
  IREE_ASSERT(source_location_id < materializer->metadata->locations.count);
  if (source_location_id == LOOM_LOCATION_UNKNOWN) {
    *out_target_location_id = LOOM_LOCATION_UNKNOWN;
    *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
    return iree_ok_status();
  }
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, source_location_id,
          &target_id)) {
    *out_target_location_id = (loom_location_id_t)target_id;
    *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
    return iree_ok_status();
  }
  *out_target_location_id = LOOM_LOCATION_UNKNOWN;
  *out_state = LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED;
  return loom_bytecode_selected_table_push(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_LOCATION, source_location_id);
}

static iree_status_t loom_bytecode_selected_table_try_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, bool* out_ready) {
  *out_ready = false;
  const loom_bytecode_encoding_metadata_t* metadata =
      &materializer->metadata->encodings.entries[source_ordinal];
  const loom_bytecode_table_entry_metadata_t entry_metadata = {
      .entry_offset = metadata->entry_offset,
      .entry_length = metadata->entry_length,
  };
  const iree_const_byte_span_t entry_bytes =
      loom_bytecode_selected_table_entry_span(materializer, &entry_metadata);
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      entry_bytes.data, entry_bytes.data_length, metadata->entry_offset,
      IREE_SV("ENCODINGS"), &cursor);

  uint64_t unused_family_index = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, &cursor, &unused_family_index));
  uint64_t alias_plus_one = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, &cursor, &alias_plus_one));
  uint64_t parameter_count = 0;
  const uint64_t parameter_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, &cursor, &parameter_count));
  if (parameter_count > UINT8_MAX || parameter_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        materializer->decoder, IREE_SV("encoding_params"), parameter_count,
        UINT8_MAX, parameter_count_offset);
  }

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
      materializer, metadata->name_string_index, &target_name_id));
  loom_string_id_t target_alias_id = LOOM_STRING_ID_INVALID;
  if (alias_plus_one > 0) {
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, (uint32_t)(alias_plus_one - 1), &target_alias_id));
  }

  loom_named_attr_t* parameters = NULL;
  if (parameter_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        materializer->scratch_arena, (iree_host_size_t)parameter_count,
        sizeof(*parameters), (void**)&parameters));
  }
  bool dependencies_ready = true;
  for (uint64_t parameter_index = 0; parameter_index < parameter_count;
       ++parameter_index) {
    uint64_t source_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        materializer->decoder, &cursor, &source_name_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, (uint32_t)source_name_id,
        &parameters[parameter_index].name_id));
    parameters[parameter_index].reserved = 0;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        materializer->decoder, &cursor, &value_kind));
    loom_bytecode_selected_attribute_state_t attribute_state =
        LOOM_BYTECODE_SELECTED_ATTRIBUTE_READY;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_decode_named(
        materializer, &cursor, /*descriptor=*/NULL, value_kind,
        &parameters[parameter_index].value,
        /*available_type_count=*/0, &attribute_state));
    if (attribute_state == LOOM_BYTECODE_SELECTED_ATTRIBUTE_WAITING) {
      dependencies_ready = false;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      materializer->decoder, &cursor, IREE_SV("encoding_entry")));
  if (!dependencies_ready) {
    return iree_ok_status();
  }

  const loom_encoding_t encoding = {
      .name_id = target_name_id,
      .alias_id = target_alias_id,
      .attribute_count = (uint8_t)parameter_count,
      .attributes = parameters,
  };
  uint16_t target_encoding_id = 0;
  IREE_RETURN_IF_ERROR(loom_module_add_encoding(
      materializer->output_module, &encoding, &target_encoding_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_projection_insert(
      &materializer->projection,
      LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING, source_ordinal,
      target_encoding_id));
  *out_ready = true;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_project_type_dependencies(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_type_id_t* source_type_ids, iree_host_size_t type_count,
    loom_type_id_t** out_target_type_ids, loom_type_t** out_target_types,
    bool* out_ready) {
  *out_target_type_ids = NULL;
  *out_target_types = NULL;
  *out_ready = false;
  if (type_count == 0) {
    *out_ready = true;
    return iree_ok_status();
  }
  loom_type_id_t* target_type_ids = NULL;
  loom_type_t* target_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      materializer->scratch_arena, type_count, sizeof(*target_type_ids),
      (void**)&target_type_ids));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(materializer->scratch_arena, type_count,
                                sizeof(*target_types), (void**)&target_types));
  bool dependencies_ready = true;
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    loom_bytecode_selected_reference_state_t reference_state =
        LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_type(
        materializer, source_type_ids[i], &target_type_ids[i],
        &reference_state));
    if (reference_state == LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED) {
      dependencies_ready = false;
    } else {
      target_types[i] =
          materializer->output_module->types.entries[target_type_ids[i]];
    }
  }
  if (!dependencies_ready) {
    return iree_ok_status();
  }
  *out_target_type_ids = target_type_ids;
  *out_target_types = target_types;
  *out_ready = true;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_try_parameterized_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_parameterized_type_fact_t* fact, loom_type_t* out_type,
    bool* out_ready) {
  *out_ready = false;
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
  bool dependencies_ready = true;
  for (uint8_t i = 0; i < fact->present_count; ++i) {
    uint64_t unused_parameter_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        materializer->decoder, &cursor, &unused_parameter_name_id));
    const uint8_t parameter_index = fact->parameter_indices[i];
    IREE_ASSERT(parameter_index < descriptor->parameter_count);
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        materializer->decoder, &cursor, &value_kind));
    loom_bytecode_selected_attribute_state_t attribute_state =
        LOOM_BYTECODE_SELECTED_ATTRIBUTE_READY;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_decode_named(
        materializer, &cursor,
        &descriptor->parameter_descriptors[parameter_index], value_kind,
        &parameters[parameter_index], fact->base.type_id, &attribute_state));
    if (attribute_state == LOOM_BYTECODE_SELECTED_ATTRIBUTE_WAITING) {
      dependencies_ready = false;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      materializer->decoder, &cursor,
      IREE_SV("parameterized_type_parameters")));
  if (!dependencies_ready) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_module_make_parameterized_type(
      materializer->output_module, descriptor, parameters,
      descriptor->parameter_count, out_type));
  *out_ready = true;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_construct_type_fact(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_type_fact_t* fact, loom_type_t* out_type,
    loom_type_id_t** out_dependency_ids, iree_host_size_t* out_dependency_count,
    bool* out_ready) {
  *out_dependency_ids = NULL;
  *out_dependency_count = 0;
  *out_ready = false;
  switch (fact->kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_bytecode_function_type_fact_t* function_fact =
          (const loom_bytecode_function_type_fact_t*)fact;
      const iree_host_size_t type_count =
          (iree_host_size_t)function_fact->argument_count +
          function_fact->result_count;
      loom_type_id_t* target_type_ids = NULL;
      loom_type_t* target_types = NULL;
      bool dependencies_ready = false;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_project_type_dependencies(
              materializer, function_fact->type_ids, type_count,
              &target_type_ids, &target_types, &dependencies_ready));
      if (!dependencies_ready) {
        return iree_ok_status();
      }
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
      if (type_count > 0) {
        memcpy(data->types, target_types, type_count * sizeof(*target_types));
      }
      *out_type = loom_type_function(data);
      *out_dependency_ids = target_type_ids;
      *out_dependency_count = type_count;
      *out_ready = true;
      return iree_ok_status();
    }
    case LOOM_TYPE_DIALECT: {
      const loom_bytecode_dialect_type_fact_t* dialect_fact =
          (const loom_bytecode_dialect_type_fact_t*)fact;
      loom_type_id_t* target_type_ids = NULL;
      loom_type_t* target_types = NULL;
      bool dependencies_ready = false;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_project_type_dependencies(
              materializer, dialect_fact->type_ids,
              dialect_fact->parameter_count, &target_type_ids, &target_types,
              &dependencies_ready));
      if (!dependencies_ready) {
        return iree_ok_status();
      }
      loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
          materializer, dialect_fact->name_id, &target_name_id));
      *out_type = loom_type_dialect(
          target_name_id, dialect_fact->parameter_count, target_types);
      *out_dependency_ids = target_type_ids;
      *out_dependency_count = dialect_fact->parameter_count;
      *out_ready = true;
      return iree_ok_status();
    }
    case LOOM_TYPE_PARAMETERIZED:
      return loom_bytecode_selected_table_try_parameterized_type(
          materializer, (const loom_bytecode_parameterized_type_fact_t*)fact,
          out_type, out_ready);
    case LOOM_TYPE_REGISTER: {
      const loom_bytecode_typed_register_fact_t* register_fact =
          (const loom_bytecode_typed_register_fact_t*)fact;
      loom_type_id_t* target_type_ids = NULL;
      loom_type_t* target_types = NULL;
      bool dependencies_ready = false;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_project_type_dependencies(
              materializer, &register_fact->value_type_id, 1, &target_type_ids,
              &target_types, &dependencies_ready));
      if (!dependencies_ready) {
        return iree_ok_status();
      }
      loom_register_type_data_t* data = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(materializer->scratch_arena,
                                               sizeof(*data), (void**)&data));
      *data = (loom_register_type_data_t){
          .carrier_payload0 = register_fact->carrier_payload0,
          .carrier_payload1 = register_fact->carrier_payload1,
          .value_type = target_types[0],
      };
      *out_type = loom_type_register_payload_with_value_type(data);
      *out_dependency_ids = target_type_ids;
      *out_dependency_count = 1;
      *out_ready = true;
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("validated type fact kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_bytecode_selected_table_try_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, bool* out_ready) {
  *out_ready = false;
  const loom_bytecode_table_entry_metadata_t* metadata =
      &materializer->metadata->types.entries[source_ordinal];
  const iree_const_byte_span_t entry_bytes =
      loom_bytecode_selected_table_entry_span(materializer, metadata);
  loom_bytecode_reader_module_view_t module_view = {
      .strings =
          {
              .values = materializer->metadata->strings.values,
              .count = materializer->metadata->strings.count,
          },
      .types =
          {
              .count = materializer->metadata->types.count,
          },
      .encodings =
          {
              .count = materializer->metadata->encodings.count,
          },
  };
  loom_bytecode_type_plan_entry_t plan_entry;
  loom_bytecode_type_fact_t* fact = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_indexed_entry(
      materializer->decoder, materializer->context, &module_view,
      materializer->scratch_arena, (loom_type_id_t)source_ordinal, entry_bytes,
      metadata->entry_offset, &plan_entry, &fact));

  loom_type_t type = plan_entry.direct_type;
  loom_type_id_t* dependency_ids = NULL;
  iree_host_size_t dependency_count = 0;
  if (fact != NULL) {
    bool fact_ready = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_construct_type_fact(
        materializer, fact, &type, &dependency_ids, &dependency_count,
        &fact_ready));
    if (!fact_ready) {
      return iree_ok_status();
    }
  } else {
    if (loom_type_kind(type) == LOOM_TYPE_DIALECT) {
      loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
          materializer, loom_type_dialect_name_id(type), &target_name_id));
      type = loom_type_dialect_opaque(target_name_id);
    }
    if (loom_type_has_static_encoding(type)) {
      uint16_t target_encoding_id = 0;
      loom_bytecode_selected_reference_state_t reference_state =
          LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_encoding(
          materializer, type.encoding_id, &target_encoding_id,
          &reference_state));
      if (reference_state == LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED) {
        return iree_ok_status();
      }
      type.encoding_id = target_encoding_id;
    }
  }

  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_topological_type_id(
      materializer->output_module, type, dependency_ids, dependency_count,
      &target_type_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_projection_insert(
      &materializer->projection, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE,
      source_ordinal, target_type_id));
  *out_ready = true;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_read_location_coordinate(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, uint32_t source_ordinal,
    iree_string_view_t field_name, uint16_t* out_value) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(materializer->decoder, cursor, &value));
  if (value > UINT16_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
        source_ordinal, field_name, offset,
        IREE_SV("file_location_coordinate_exceeds_runtime_field_width"));
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_read_source(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, loom_source_id_t* out_source_id) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, cursor, &source_ordinal));
  if (source_ordinal >= materializer->metadata->sources.count) {
    return loom_bytecode_reader_emit_table_ref(
        materializer->decoder, IREE_SV("SOURCES"), source_ordinal,
        materializer->metadata->sources.count, offset);
  }
  return loom_bytecode_selected_table_project_source(
      materializer, (uint32_t)source_ordinal, out_source_id);
}

static iree_status_t loom_bytecode_selected_table_copy_location_payload(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_entry_t* entry) {
  switch (entry->kind) {
    case LOOM_LOCATION_FUSED: {
      loom_location_id_t* children = NULL;
      if (entry->fused.count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &materializer->output_module->arena, entry->fused.count,
            sizeof(*children), (void**)&children));
        memcpy(children, entry->fused.children,
               (iree_host_size_t)entry->fused.count * sizeof(*children));
      }
      entry->fused.children = children;
      return iree_ok_status();
    }
    case LOOM_LOCATION_OPAQUE: {
      uint8_t* data = NULL;
      if (entry->opaque.data_length > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(&materializer->output_module->arena,
                                entry->opaque.data_length, (void**)&data));
        memcpy(data, entry->opaque.data, entry->opaque.data_length);
      }
      entry->opaque.data = data;
      return iree_ok_status();
    }
    case LOOM_LOCATION_TAGGED: {
      uint8_t* data = NULL;
      if (entry->tagged.data_length > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(&materializer->output_module->arena,
                                entry->tagged.data_length, (void**)&data));
        memcpy(data, entry->tagged.data, entry->tagged.data_length);
      }
      entry->tagged.data = data;
      return iree_ok_status();
    }
    case LOOM_LOCATION_NONE:
    case LOOM_LOCATION_FILE:
      return iree_ok_status();
    case LOOM_LOCATION_COUNT_:
      break;
  }
  IREE_ASSERT_UNREACHABLE("validated location kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_table_try_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, bool* out_ready) {
  *out_ready = false;
  const loom_bytecode_table_entry_metadata_t* metadata =
      &materializer->metadata->locations.entries[source_ordinal];
  const iree_const_byte_span_t entry_bytes =
      loom_bytecode_selected_table_entry_span(materializer, metadata);
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      entry_bytes.data, entry_bytes.data_length, metadata->entry_offset,
      IREE_SV("LOCATIONS"), &cursor);
  uint8_t kind = 0;
  uint8_t flags = 0;
  const uint64_t kind_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(materializer->decoder, &cursor, &kind));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(materializer->decoder, &cursor, &flags));
  if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
        source_ordinal, IREE_SV("flags"), kind_offset + 1,
        IREE_SV("location_has_unsupported_flag_bits"));
  }
  loom_location_entry_t entry = {
      .kind = (loom_location_kind_t)kind,
      .flags = flags,
  };
  switch ((loom_location_kind_t)kind) {
    case LOOM_LOCATION_NONE:
      IREE_ASSERT(source_ordinal == 0 && flags == 0);
      *out_ready = true;
      return iree_ok_status();
    case LOOM_LOCATION_FILE: {
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_read_source(
          materializer, &cursor, &entry.file.source_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("start_line"),
              &entry.file.start_line));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("start_col"),
              &entry.file.start_col));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("end_line"),
              &entry.file.end_line));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("end_col"),
              &entry.file.end_col));
      break;
    }
    case LOOM_LOCATION_FUSED: {
      const uint64_t child_count_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t child_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &child_count));
      if (child_count > UINT32_MAX || child_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("location_children"), child_count,
            UINT32_MAX, child_count_offset);
      }
      loom_location_id_t* children = NULL;
      if (child_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)child_count,
            sizeof(*children), (void**)&children));
      }
      bool dependencies_ready = true;
      for (uint64_t child_index = 0; child_index < child_count; ++child_index) {
        const uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t source_child = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &source_child));
        if (source_child >= source_ordinal) {
          return loom_bytecode_reader_emit_table_ref(
              materializer->decoder, IREE_SV("LOCATIONS"), source_child,
              source_ordinal, child_offset);
        }
        loom_bytecode_selected_reference_state_t reference_state =
            LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
        IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_location(
            materializer, (loom_location_id_t)source_child,
            &children[child_index], &reference_state));
        if (reference_state == LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED) {
          dependencies_ready = false;
        }
      }
      if (!dependencies_ready) {
        return iree_ok_status();
      }
      entry.fused.count = (uint32_t)child_count;
      entry.fused.children = children;
      break;
    }
    case LOOM_LOCATION_OPAQUE: {
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_read_source(
          materializer, &cursor, &entry.opaque.source_id));
      uint64_t data_length = 0;
      const uint64_t data_length_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &data_length));
      if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("opaque_location_data"), data_length,
            UINT32_MAX, data_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, &cursor, data_length, &span));
      entry.opaque.data_length = (uint32_t)span.data_length;
      entry.opaque.data = span.data;
      break;
    }
    case LOOM_LOCATION_TAGGED: {
      uint64_t tag = 0;
      const uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &tag));
      if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
            source_ordinal, IREE_SV("tag"), tag_offset,
            IREE_SV("tagged location tag must be in [1, 65535]"));
      }
      uint64_t source_child = 0;
      const uint64_t child_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &source_child));
      if (source_child >= source_ordinal) {
        return loom_bytecode_reader_emit_table_ref(
            materializer->decoder, IREE_SV("LOCATIONS"), source_child,
            source_ordinal, child_offset);
      }
      loom_bytecode_selected_reference_state_t reference_state =
          LOOM_BYTECODE_SELECTED_REFERENCE_RESOLVED;
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_location(
          materializer, (loom_location_id_t)source_child, &entry.tagged.child,
          &reference_state));
      if (reference_state == LOOM_BYTECODE_SELECTED_REFERENCE_SCHEDULED) {
        return iree_ok_status();
      }
      uint64_t data_length = 0;
      const uint64_t data_length_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &data_length));
      if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("tagged_location_data"), data_length,
            UINT32_MAX, data_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, &cursor, data_length, &span));
      entry.tagged.tag = (loom_location_tag_t)tag;
      entry.tagged.data_length = (uint32_t)span.data_length;
      entry.tagged.data = span.data;
      break;
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          materializer->decoder, IREE_SV("location_kind"), kind,
          LOOM_LOCATION_COUNT_, kind_offset);
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      materializer->decoder, &cursor, IREE_SV("location_entry")));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_selected_table_copy_location_payload(materializer, &entry));
  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_module_add_location(materializer->output_module,
                                                entry, &target_location_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_projection_insert(
      &materializer->projection,
      LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, source_ordinal,
      target_location_id));
  *out_ready = true;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_try_frame(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_frame_t frame, bool* out_ready) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(materializer->scratch_arena);
  iree_status_t status = iree_ok_status();
  switch (frame.table_kind) {
    case LOOM_BYTECODE_SELECTED_TABLE_ENCODING:
      status = loom_bytecode_selected_table_try_encoding(
          materializer, frame.source_ordinal, out_ready);
      break;
    case LOOM_BYTECODE_SELECTED_TABLE_TYPE:
      status = loom_bytecode_selected_table_try_type(
          materializer, frame.source_ordinal, out_ready);
      break;
    case LOOM_BYTECODE_SELECTED_TABLE_LOCATION:
      status = loom_bytecode_selected_table_try_location(
          materializer, frame.source_ordinal, out_ready);
      break;
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

static iree_status_t loom_bytecode_selected_table_materialize_root(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal,
    uint32_t* out_target_id) {
  const loom_bytecode_selected_projection_domain_t projection_domain =
      loom_bytecode_selected_table_projection_domain(table_kind);
  if (loom_bytecode_selected_projection_lookup(&materializer->projection,
                                               projection_domain,
                                               source_ordinal, out_target_id)) {
    return iree_ok_status();
  }
  IREE_ASSERT(materializer->worklist.count == 0);
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_push(
      materializer, table_kind, source_ordinal));
  iree_status_t status = iree_ok_status();
  while (materializer->worklist.count > 0 && iree_status_is_ok(status)) {
    const iree_host_size_t frame_index = materializer->worklist.count - 1;
    const loom_bytecode_selected_table_frame_t frame =
        materializer->worklist.values[frame_index];
    uint32_t existing_target_id = 0;
    if (loom_bytecode_selected_projection_lookup(
            &materializer->projection,
            loom_bytecode_selected_table_projection_domain(frame.table_kind),
            frame.source_ordinal, &existing_target_id)) {
      --materializer->worklist.count;
      continue;
    }
    bool ready = false;
    status =
        loom_bytecode_selected_table_try_frame(materializer, frame, &ready);
    if (iree_status_is_ok(status) && ready) {
      IREE_ASSERT(materializer->worklist.count == frame_index + 1);
      --materializer->worklist.count;
    } else if (iree_status_is_ok(status)) {
      IREE_ASSERT(materializer->worklist.count > frame_index + 1);
    }
  }
  materializer->worklist.count = 0;
  if (iree_status_is_ok(status)) {
    const bool found = loom_bytecode_selected_projection_lookup(
        &materializer->projection, projection_domain, source_ordinal,
        out_target_id);
    IREE_ASSERT(found);
  }
  return status;
}

iree_status_t loom_bytecode_selected_table_materialize_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint16_t source_encoding_id, uint16_t* out_target_encoding_id) {
  IREE_ASSERT(source_encoding_id > 0);
  IREE_ASSERT(source_encoding_id <= materializer->metadata->encodings.count);
  uint32_t target_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_root(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_ENCODING,
      source_encoding_id - 1, &target_id));
  *out_target_encoding_id = (uint16_t)target_id;
  return iree_ok_status();
}

iree_status_t loom_bytecode_selected_table_materialize_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_type_id_t source_type_id, loom_type_id_t* out_target_type_id) {
  IREE_ASSERT(source_type_id < materializer->metadata->types.count);
  uint32_t target_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_root(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_TYPE, source_type_id,
      &target_id));
  *out_target_type_id = (loom_type_id_t)target_id;
  return iree_ok_status();
}

iree_status_t loom_bytecode_selected_table_materialize_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id) {
  IREE_ASSERT(source_location_id < materializer->metadata->locations.count);
  if (source_location_id == LOOM_LOCATION_UNKNOWN) {
    *out_target_location_id = LOOM_LOCATION_UNKNOWN;
    return iree_ok_status();
  }
  uint32_t target_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_root(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_LOCATION, source_location_id,
      &target_id));
  *out_target_location_id = (loom_location_id_t)target_id;
  return iree_ok_status();
}
