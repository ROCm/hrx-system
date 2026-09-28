// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete scope-local type records refer to completed children. Global TYPES
// contain only SSA-independent types; no unbound template is retained for a
// scoped record.
static iree_string_builder_t* loom_bytecode_record_buffer_reset(
    loom_bytecode_numbering_t* numbering) {
  iree_string_builder_t* buffer = &numbering->type_record_buffer.builder;
  if (!buffer->allocator.ctl) {
    loom_bytecode_buffer_initialize(numbering->arena,
                                    &numbering->type_record_buffer);
  }
  iree_string_builder_reset(buffer);
  return buffer;
}

static iree_status_t loom_bytecode_emit_complete_type(
    iree_string_builder_t* sink, loom_bytecode_numbering_t* numbering,
    loom_bytecode_value_numbering_t* values, loom_type_id_t type_id) {
  const loom_bytecode_type_index_t* index = &numbering->types.index;
  const loom_bytecode_type_node_t* node = &index->nodes[type_id];
  const loom_type_t type =
      loom_type_table_get(&numbering->module->types, type_id);
  const loom_type_kind_t kind = loom_type_kind(type);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_u8(sink, loom_bytecode_type_kind_byte(kind)));
  switch (kind) {
    case LOOM_TYPE_POOL: {
      const uint64_t dimension = loom_type_dim(type, 0);
      uint64_t reference = 0;
      if (loom_dim_value_id(dimension) != LOOM_VALUE_ID_INVALID) {
        uint32_t number = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
            values, loom_dim_value_id(dimension), &number));
        reference = (uint64_t)number + 1;
      }
      return loom_bytecode_emit_uvarint(sink, reference);
    }
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(sink, loom_type_element_type(type)));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(sink, loom_type_rank(type)));
      const uint32_t attachment = loom_type_has_ssa_encoding(type)      ? 2
                                  : loom_type_has_static_encoding(type) ? 1
                                                                        : 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, attachment));
      uint32_t encoding = type.encoding_id;
      if (attachment == 2) {
        IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
            values, loom_type_encoding_value_id(type), &encoding));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, encoding));
      if (loom_type_kind(type) == LOOM_TYPE_VIEW) {
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(
            sink, loom_type_view_alignment_override(type)));
      }
      for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
        const uint64_t dimension = loom_type_dim(type, i);
        const bool dynamic = loom_dim_is_dynamic(dimension);
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, dynamic));
        uint64_t payload = dimension;
        if (dynamic) {
          payload = 0;
          if (loom_dim_value_id(dimension) != LOOM_VALUE_ID_INVALID) {
            uint32_t number = 0;
            IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
                values, loom_dim_value_id(dimension), &number));
            payload = (uint64_t)number + 1;
          }
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, payload));
      }
      return iree_ok_status();
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, data->arg_count));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(sink, data->result_count));
      break;
    }
    case LOOM_TYPE_DIALECT: {
      uint32_t name = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, loom_type_dialect_name_id(type), &name));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, name));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(sink, node->dependencies.explicit_count));
      break;
    }
    case LOOM_TYPE_REGISTER: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(sink, loom_type_register_payload0(type)));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(sink, loom_type_register_payload1(type)));
      break;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      uint32_t name = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
          numbering, loom_bstring_view(descriptor->name), &name));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, name));
      uint8_t count = 0;
      for (uint8_t i = 0; i < descriptor->parameter_count; ++i) {
        count += !loom_attr_is_absent(parameters[i]);
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, count));
      for (uint8_t i = 0; i < descriptor->parameter_count; ++i) {
        if (loom_attr_is_absent(parameters[i])) {
          continue;
        }
        const loom_attr_descriptor_t* parameter =
            &descriptor->parameter_descriptors[i];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering, loom_attr_descriptor_name(parameter), &name));
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(sink, name));
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value_at_depth(
            sink, numbering, values, parameters[i], parameter, 0));
      }
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE(
          "SSA-dependent type from the completed type index");
      IREE_BUILTIN_UNREACHABLE();
  }
  for (iree_host_size_t i = 0; i < node->dependencies.explicit_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(
        sink,
        loom_bytecode_scoped_type_reference(
            numbering, index->dependencies[node->dependencies.begin + i])));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_complete_bindings(
    iree_string_builder_t* payload, loom_bytecode_numbering_t* numbering,
    loom_bytecode_value_numbering_t* values, loom_type_id_t type_id) {
  uint32_t count = 0;
  uint32_t binding = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_prepare_type_bindings(
      numbering, values, type_id, &count, &binding));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(payload, count));
  for (uint32_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_complete_type(
        payload, numbering, values, numbering->types.index.pending[i]));
  }
  return loom_bytecode_emit_uvarint(payload, binding);
}

iree_status_t loom_bytecode_write_type_bindings(
    loom_bytecode_page_writer_t* sink, loom_bytecode_numbering_t* numbering,
    loom_bytecode_value_numbering_t* values, loom_type_id_t type_id) {
  if (!numbering->types.index.nodes[type_id].has_bindings) {
    return loom_bytecode_page_writer_write_uvarint(sink, 0);
  }
  iree_string_builder_t* payload = loom_bytecode_record_buffer_reset(numbering);
  iree_status_t status =
      loom_bytecode_emit_complete_bindings(payload, numbering, values, type_id);
  const iree_string_view_t bytes = iree_string_builder_view(payload);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(sink, bytes.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write(sink, bytes.data, bytes.size);
  }
  return status;
}
