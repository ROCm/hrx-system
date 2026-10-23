// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Policy-specialized symbol-body materialization implementation.
//
// This file intentionally has no include guard. The including translation unit
// supplies concrete policy types, direct reference operations, and names for
// the three externally visible entry points. Full and selective readers thereby
// share one wire grammar without runtime modes or indirect dispatch.

#if !defined(LOOM_BYTECODE_BODY_VALUE_SCOPE_INITIALIZE_FRESH)
#error "body value-scope initializer name is required"
#endif
#if !defined(LOOM_BYTECODE_BODY_VALUE_SCOPE_MATERIALIZE_DEFINITION)
#error "body value-definition materializer name is required"
#endif
#if !defined(LOOM_BYTECODE_BODY_MATERIALIZE_REGION)
#error "body root-region materializer name is required"
#endif

#define LOOM_BYTECODE_MAX_REGION_DEPTH 256

typedef struct loom_bytecode_body_counts_t {
  // SSA values defined while decoding a body.
  uint64_t value_count;
  // Regions decoded, including nested regions.
  uint64_t region_count;
  // Blocks decoded, including nested regions.
  uint64_t block_count;
  // Operations decoded, including nested regions.
  uint64_t op_count;
} loom_bytecode_body_counts_t;

typedef struct loom_bytecode_body_reader_t {
  // Shared module, attribute, and representation materialization state.
  loom_bytecode_body_policy_materializer_t* materializer;
  // Root-region-local SSA value namespace and type-binding state.
  loom_bytecode_body_policy_value_scope_t values;
  // True while decoding the entry block owning the predefined values.
  bool expect_predefined_entry_args;
  // Representation contract selected once for the containing function.
  const loom_low_repr_descriptor_set_t* low_descriptor_set;
  // Actual region counts used to verify the encoded allocation summary.
  loom_bytecode_body_counts_t counts;
} loom_bytecode_body_reader_t;

static iree_status_t loom_bytecode_body_emit_invalid(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t symbol_name,
    uint64_t offset, iree_string_view_t failure_code) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
      loom_param_u64(offset),
      loom_param_string(failure_code),
  };
  return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_016, params,
                                         IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_value_scope_emit_invalid(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t offset,
    iree_string_view_t failure_code) {
  return loom_bytecode_body_emit_invalid(
      value_scope->decoder, value_scope->symbol_name, offset, failure_code);
}

static iree_status_t loom_bytecode_value_scope_lookup_value(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t value_number,
    uint64_t value_limit, uint64_t offset, iree_string_view_t failure_code,
    loom_value_id_t* out_value_id) {
  if (value_number >= value_limit) {
    return loom_bytecode_value_scope_emit_invalid(value_scope, offset,
                                                  failure_code);
  }
  *out_value_id = value_scope->value_map[value_number];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_read_region(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    loom_op_t* parent_op, uint32_t depth, loom_region_t** out_region);

iree_status_t LOOM_BYTECODE_BODY_VALUE_SCOPE_MATERIALIZE_DEFINITION(
    loom_bytecode_body_policy_value_scope_t* value_scope,
    loom_bytecode_reader_cursor_t* cursor, loom_value_id_t* out_value_id) {
  uint64_t name_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t name_id = 0;
  uint64_t type_id = 0;

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(value_scope->decoder,
                                                         cursor, &name_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(value_scope->decoder,
                                                         cursor, &type_id));

  if (value_scope->next_value_number >= value_scope->available_value_count) {
    return loom_bytecode_value_scope_emit_invalid(
        value_scope, name_offset,
        IREE_SV("value_definition_was_not_reserved_before_decoding"));
  }
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  if (name_id != 0) {
    IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_project_string(
        value_scope, name_id, IREE_SV("value_name"), name_offset,
        &value_name_id));
  }
  const loom_bytecode_attribute_ssa_materialization_scope_t bindings = {
      .symbol_name = value_scope->symbol_name,
      .values = value_scope->value_map,
      .value_count = value_scope->available_value_count,
      .bindings = &value_scope->bindings,
  };
  loom_type_id_t canonical_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_materialize_type_bindings(
      value_scope, cursor, &bindings, type_id, &canonical_type_id));
  const loom_type_t type = loom_type_table_get(
      &value_scope->output_module->types, canonical_type_id);
  loom_value_id_t value_id =
      value_scope->value_map[value_scope->next_value_number];
  const bool is_predefined =
      value_scope->next_value_number >= value_scope->predefined_value_start &&
      value_scope->next_value_number - value_scope->predefined_value_start <
          value_scope->predefined_value_count;
  if (is_predefined) {
    if (value_id >= value_scope->output_module->values.count) {
      return loom_bytecode_value_scope_emit_invalid(
          value_scope, name_offset,
          IREE_SV("predefined_value_is_not_present_in_the_module"));
    }
    const loom_value_t* value =
        loom_module_value(value_scope->output_module, value_id);
    if (!loom_type_equal(value->type, type)) {
      return loom_bytecode_value_scope_emit_invalid(
          value_scope, name_offset,
          IREE_SV("predefined_value_type_does_not_match_body_value_type"));
    }
    if (value->name_id != value_name_id) {
      return loom_bytecode_value_scope_emit_invalid(
          value_scope, name_offset,
          IREE_SV("predefined_value_name_does_not_match_body_value_name"));
    }
    value_scope->value_map[value_scope->next_value_number++] = value_id;
    *out_value_id = value_id;
    return iree_ok_status();
  }

  if (loom_type_table_dependencies(&value_scope->output_module->types,
                                   canonical_type_id) == 0) {
    // This exact type-table entry has no region-local bindings and the reserved
    // value is fresh, so installing it cannot invalidate type-use state.
    loom_module_value(value_scope->output_module, value_id)->type =
        loom_type_table_get(&value_scope->output_module->types,
                            canonical_type_id);
  } else {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(value_scope->output_module, value_id, type));
  }
  if (value_name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_set_value_name(value_scope->output_module,
                                                    value_id, value_name_id));
  }
  ++value_scope->next_value_number;
  *out_value_id = value_id;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_value_scope_prepare_fresh_values(
    loom_bytecode_body_policy_value_scope_t* value_scope,
    iree_host_size_t count) {
  loom_value_id_t base_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_define_untyped_values(
      value_scope->output_module, count, &base_value_id));
  value_scope->next_fresh_value_id = base_value_id;
  value_scope->remaining_fresh_value_count = count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_value_scope_reserve_definitions(
    loom_bytecode_body_policy_value_scope_t* value_scope, uint64_t count,
    loom_value_id_t* out_value_ids) {
  if (value_scope->next_value_number > value_scope->value_capacity ||
      count > value_scope->value_capacity - value_scope->next_value_number) {
    return loom_bytecode_value_scope_emit_invalid(
        value_scope, value_scope->payload_offset,
        IREE_SV("root_region_defines_more_values_than_its_summary"));
  }
  uint64_t start_value_number = value_scope->next_value_number;
  uint64_t end_value_number = start_value_number + count;
  for (uint64_t value_number = start_value_number;
       value_number < end_value_number; ++value_number) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    if (value_number >= value_scope->predefined_value_start &&
        value_number - value_scope->predefined_value_start <
            value_scope->predefined_value_count) {
      value_id =
          value_scope->predefined_values[value_number -
                                         value_scope->predefined_value_start];
      if (value_id >= value_scope->output_module->values.count) {
        return loom_bytecode_value_scope_emit_invalid(
            value_scope, value_scope->payload_offset,
            IREE_SV("predefined_value_is_not_present_in_the_module"));
      }
    } else {
      if (value_scope->remaining_fresh_value_count == 0) {
        return loom_bytecode_value_scope_emit_invalid(
            value_scope, value_scope->payload_offset,
            IREE_SV("fresh_value_definition_exceeds_region_summary"));
      }
      value_id = value_scope->next_fresh_value_id++;
      --value_scope->remaining_fresh_value_count;
    }
    value_scope->value_map[value_number] = value_id;
    if (out_value_ids) {
      out_value_ids[value_number - start_value_number] = value_id;
    }
  }
  value_scope->available_value_count = end_value_number;
  return iree_ok_status();
}

iree_status_t LOOM_BYTECODE_BODY_VALUE_SCOPE_INITIALIZE_FRESH(
    loom_bytecode_body_policy_materializer_t* materializer,
    iree_arena_allocator_t* arena, iree_string_view_t symbol_name,
    uint64_t payload_offset, loom_value_id_t* value_map,
    iree_host_size_t value_count,
    loom_bytecode_body_policy_value_scope_t* out_scope) {
  loom_bytecode_body_policy_value_scope_t value_scope;
  loom_bytecode_body_policy_initialize_value_scope(
      materializer, arena, symbol_name, payload_offset, value_map, value_count,
      &value_scope);
  IREE_RETURN_IF_ERROR(loom_bytecode_value_scope_prepare_fresh_values(
      &value_scope, value_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_value_scope_reserve_definitions(
      &value_scope, value_count, value_map));
  *out_scope = value_scope;
  return iree_ok_status();
}

typedef struct loom_bytecode_body_op_record_t {
  // Materialized operation awaiting nested regions and finalization.
  loom_op_t* op;
  // Source trivia retained until the operation is finalized.
  loom_bytecode_source_trivia_t source_trivia;
} loom_bytecode_body_op_record_t;

// Decodes and materializes an operation record without descending into its
// nested regions. Keeping the large record-decoding frame out of the recursive
// region walk bounds stack consumption at the bytecode nesting limit.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_body_reader_read_op_record(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    loom_bytecode_body_op_record_t* out_record) {
  uint64_t op_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  loom_op_kind_t op_kind = LOOM_OP_KIND_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_resolve_op(
      &body_reader->values, op_table_index_plus1, op_offset, &vtable,
      &op_kind));
  if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, op_offset,
        IREE_SV("symbol_definition_is_not_permitted_in_a_symbol_body"));
  }

  uint8_t instance_flags = 0;
  uint64_t instance_flags_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(body_reader->values.decoder,
                                                    cursor, &instance_flags));
  uint8_t instance_flags_mask = vtable->instance_flags_mask;
  if (iree_any_bit_set(instance_flags, (uint8_t)~instance_flags_mask)) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, instance_flags_offset,
        IREE_SV("operation_instance_flags_contain_undeclared_bits"));
  }
  uint64_t location_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &location_id));
  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  if (location_id != 0) {
    IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_materialize_location(
        &body_reader->values, location_id,
        loom_bytecode_reader_cursor_absolute_position(cursor),
        &target_location_id));
  }

  loom_bytecode_source_trivia_t source_trivia;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
      body_reader->values.decoder, cursor, body_reader->values.arena,
      &source_trivia));

  uint64_t operand_count = 0;
  uint64_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &operand_count));
  if (operand_count > UINT16_MAX || operand_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, op_offset,
        IREE_SV("operand_count_exceeds_field_width"));
  }
  loom_value_id_t* operands = NULL;
  if (operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->values.arena, (iree_host_size_t)operand_count,
        sizeof(loom_value_id_t), (void**)&operands));
  }
  for (uint64_t i = 0; i < operand_count; ++i) {
    uint64_t ref_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->values.decoder, cursor, &value_number));
    IREE_RETURN_IF_ERROR(loom_bytecode_value_scope_lookup_value(
        &body_reader->values, value_number,
        body_reader->values.next_value_number, ref_offset,
        IREE_SV("operand value reference must target a previously defined "
                "value"),
        &operands[i]));
  }
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  uint16_t* operand_segment_counts = NULL;
  if (operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->values.arena, operand_segment_count, sizeof(uint16_t),
        (void**)&operand_segment_counts));
    uint32_t total_segment_count = 0;
    for (uint8_t i = 0; i < operand_segment_count; ++i) {
      uint64_t segment_count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t segment_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          body_reader->values.decoder, cursor, &segment_count));
      if (segment_count > UINT16_MAX) {
        return loom_bytecode_value_scope_emit_invalid(
            &body_reader->values, segment_count_offset,
            IREE_SV("operand_segment_count_exceeds_field_width"));
      }
      const loom_operand_descriptor_t* descriptor =
          &vtable->operand_descriptors[i];
      if (!iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC)) {
        const bool optional =
            iree_any_bit_set(descriptor->flags, LOOM_OPERAND_OPTIONAL);
        if ((!optional && segment_count != 1) ||
            (optional && segment_count > 1)) {
          return loom_bytecode_value_scope_emit_invalid(
              &body_reader->values, segment_count_offset,
              IREE_SV("operand_segment_count_violates_field_arity"));
        }
      }
      operand_segment_counts[i] = (uint16_t)segment_count;
      total_segment_count += (uint32_t)segment_count;
    }
    if (total_segment_count != operand_count) {
      return loom_bytecode_value_scope_emit_invalid(
          &body_reader->values, op_offset,
          IREE_SV("operand_segment_counts_do_not_sum_to_operand_count"));
    }
  }

  uint64_t successor_count = 0;
  uint64_t successor_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &successor_count));
  if (successor_count > UINT8_MAX || successor_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, successor_count_offset,
        IREE_SV("successor_count_exceeds_field_width"));
  }
  loom_block_t** successors = NULL;
  if (successor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->values.arena, (iree_host_size_t)successor_count,
        sizeof(loom_block_t*), (void**)&successors));
    loom_region_t* successor_region =
        builder->ip.block ? builder->ip.block->parent_region : NULL;
    if (!successor_region) {
      return loom_bytecode_value_scope_emit_invalid(
          &body_reader->values, successor_count_offset,
          IREE_SV("operation_successors_require_an_enclosing_region"));
    }
    for (uint64_t i = 0; i < successor_count; ++i) {
      uint64_t ref_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t block_index = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          body_reader->values.decoder, cursor, &block_index));
      if (block_index >= successor_region->block_count) {
        return loom_bytecode_value_scope_emit_invalid(
            &body_reader->values, ref_offset,
            IREE_SV("successor_block_index_is_out_of_range"));
      }
      successors[i] =
          loom_region_block(successor_region, (uint16_t)block_index);
    }
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &result_count));
  if (result_count > UINT16_MAX || result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, op_offset,
        IREE_SV("result_count_exceeds_field_width"));
  }
  loom_value_id_t* results = NULL;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->values.arena, (iree_host_size_t)result_count,
        sizeof(loom_value_id_t), (void**)&results));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_value_scope_reserve_definitions(
      &body_reader->values, result_count, results));
  for (uint64_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(LOOM_BYTECODE_BODY_VALUE_SCOPE_MATERIALIZE_DEFINITION(
        &body_reader->values, cursor, &results[i]));
  }
  body_reader->counts.value_count += result_count;

  uint64_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &tied_result_count));
  if (tied_result_count > UINT16_MAX ||
      tied_result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, op_offset,
        IREE_SV("tied_result_count_exceeds_field_width"));
  }
  loom_tied_result_t* tied_results = NULL;
  if (tied_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->values.arena, (iree_host_size_t)tied_result_count,
        sizeof(loom_tied_result_t), (void**)&tied_results));
  }
  for (uint64_t i = 0; i < tied_result_count; ++i) {
    uint64_t tie_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t result_index = 0;
    uint64_t operand_index = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->values.decoder, cursor, &result_index));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->values.decoder, cursor, &operand_index));
    if (result_index >= result_count || operand_index >= operand_count) {
      return loom_bytecode_value_scope_emit_invalid(
          &body_reader->values, tie_offset,
          IREE_SV("tied_result_references_an_out_of_range_operand_or_result"));
    }
    tied_results[i] = (loom_tied_result_t){
        .result_index = (uint16_t)result_index,
        .operand_index = (uint16_t)operand_index,
    };
  }

  uint64_t present_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &present_attr_count));
  if (present_attr_count > vtable->attribute_count) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, op_offset,
        IREE_SV("present_attribute_count_exceeds_op_attribute_slots"));
  }
  loom_attribute_t* attrs = NULL;
  loom_trait_flags_t effective_traits = 0;
  loom_memory_space_mask_t intrinsic_write_spaces = 0;
  bool has_effective_traits = false;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->values.arena, vtable->attribute_count,
        sizeof(loom_attribute_t), (void**)&attrs));
    memset(attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  const loom_bytecode_attribute_ssa_materialization_scope_t
      attribute_ssa_scope = {
          .symbol_name = body_reader->values.symbol_name,
          .values = body_reader->values.value_map,
          .value_count = body_reader->values.next_value_number,
          .bindings = &body_reader->values.bindings,
  };
  for (uint64_t i = 0; i < present_attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->values.decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_resolve_source_string(
        &body_reader->values, key_id, IREE_SV("attribute_key"), key_offset,
        &key));
    const loom_attr_descriptor_t* descriptor = NULL;
    uint8_t attr_index = 0;
    for (; attr_index < vtable->attribute_count; ++attr_index) {
      const loom_attr_descriptor_t* candidate =
          &vtable->attr_descriptors[attr_index];
      if (iree_string_view_equal(key, loom_attr_descriptor_name(candidate))) {
        descriptor = candidate;
        break;
      }
    }
    if (!descriptor) {
      return loom_bytecode_value_scope_emit_invalid(
          &body_reader->values, key_offset,
          IREE_SV("attribute_key_is_not_declared_by_the_op"));
    }
    if (!loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_value_scope_emit_invalid(
          &body_reader->values, key_offset,
          IREE_SV("attribute_key_appears_more_than_once"));
    }
    const uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        body_reader->values.decoder, cursor, &value_kind));
    if (descriptor->attr_kind == LOOM_ATTR_SCOPED_ENUM) {
      if (value_kind != LOOM_BYTECODE_ATTR_SCOPED_ENUM) {
        return loom_bytecode_value_scope_emit_invalid(
            &body_reader->values, value_kind_offset,
            IREE_SV("scoped_enum_attribute_has_wrong_wire_kind"));
      }
      if (!body_reader->low_descriptor_set) {
        return loom_bytecode_value_scope_emit_invalid(
            &body_reader->values, value_kind_offset,
            IREE_SV("scoped_enum_attribute_has_no_function_contract"));
      }
      uint64_t descriptor_key_id = 0;
      uint64_t descriptor_key_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          body_reader->values.decoder, cursor, &descriptor_key_id));
      iree_string_view_t descriptor_key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_resolve_source_string(
          &body_reader->values, descriptor_key_id, IREE_SV("descriptor_key"),
          descriptor_key_offset, &descriptor_key));
      loom_low_repr_descriptor_value_t descriptor_value = {0};
      if (!loom_low_repr_resolve_descriptor(
              loom_bytecode_body_policy_low_repr_environment(
                  body_reader->materializer),
              body_reader->low_descriptor_set, descriptor_key,
              &descriptor_value)) {
        return loom_bytecode_value_scope_emit_invalid(
            &body_reader->values, descriptor_key_offset,
            IREE_SV("descriptor_key_is_not_in_the_function_contract"));
      }
      attrs[attr_index] = loom_attr_scoped_enum(descriptor_value.ordinal);
      effective_traits = descriptor_value.effective_traits;
      intrinsic_write_spaces = descriptor_value.intrinsic_write_spaces;
      has_effective_traits = true;
    } else {
      IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_materialize_attribute_ssa(
          body_reader->materializer, cursor, descriptor, value_kind,
          &attrs[attr_index], &attribute_ssa_scope));
    }
  }

  uint64_t region_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &region_count));
  if (region_count > UINT8_MAX || region_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, op_offset,
        IREE_SV("region_count_exceeds_field_width"));
  }

  loom_op_t* op = NULL;
  if (operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op_with_successors(
        builder, op_kind, (uint16_t)operand_count, operand_segment_counts,
        operand_segment_count, (uint16_t)result_count, (uint8_t)successor_count,
        (uint8_t)region_count, (uint16_t)tied_result_count,
        vtable->attribute_count, target_location_id, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
        builder, op_kind, (uint16_t)operand_count, (uint16_t)result_count,
        (uint8_t)successor_count, (uint8_t)region_count,
        (uint16_t)tied_result_count, vtable->attribute_count,
        target_location_id, &op));
  }
  if (operand_count > 0) {
    memcpy(loom_op_operands(op), operands,
           (iree_host_size_t)operand_count * sizeof(loom_value_id_t));
  }
  if (successor_count > 0) {
    memcpy(loom_op_successors(op), successors,
           (iree_host_size_t)successor_count * sizeof(loom_block_t*));
  }
  if (result_count > 0) {
    memcpy(loom_op_results(op), results,
           (iree_host_size_t)result_count * sizeof(loom_value_id_t));
  }
  if (tied_result_count > 0) {
    memcpy(loom_op_tied_results(op), tied_results,
           (iree_host_size_t)tied_result_count * sizeof(loom_tied_result_t));
  }
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }
  if (has_effective_traits) {
    op->traits = effective_traits;
    op->intrinsic_write_spaces = intrinsic_write_spaces;
    IREE_RETURN_IF_ERROR(loom_low_repr_resolve_packet_attributes(
        loom_bytecode_body_policy_low_repr_environment(
            body_reader->materializer),
        body_reader->low_descriptor_set, builder->module, op));
  }
  if (source_trivia.leading_blank_line) {
    op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  }
  op->instance_flags = instance_flags;
  *out_record = (loom_bytecode_body_op_record_t){
      .op = op,
      .source_trivia = source_trivia,
  };
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_read_op(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    uint32_t depth) {
  loom_bytecode_body_op_record_t record = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_read_op_record(
      body_reader, cursor, builder, &record));
  for (uint64_t i = 0; i < record.op->region_count; ++i) {
    loom_region_t* region = NULL;
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_read_region(
        body_reader, cursor, builder, record.op, depth + 1, &region));
    loom_op_regions(record.op)[i] = region;
  }
  ++body_reader->counts.op_count;
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, record.op));
  if (record.source_trivia.comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        body_reader->values.output_module, record.op,
        record.source_trivia.comments, record.source_trivia.comment_count));
  }
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_body_reader_read_block(loom_bytecode_body_reader_t* body_reader,
                                     loom_bytecode_reader_cursor_t* cursor,
                                     loom_builder_t* builder,
                                     loom_block_t* block, uint32_t depth) {
  uint8_t has_label = 0;
  uint64_t label_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(body_reader->values.decoder,
                                                    cursor, &has_label));
  if (has_label > 1) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, label_offset,
        IREE_SV("block_has_label_byte_must_be_0_or_1"));
  }
  if (has_label) {
    uint64_t label_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->values.decoder, cursor, &label_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_body_policy_project_string(
        &body_reader->values, label_id, IREE_SV("block_label"), label_offset,
        &block->label_id));
  }

  loom_bytecode_source_trivia_t source_trivia;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
      body_reader->values.decoder, cursor, body_reader->values.arena,
      &source_trivia));
  if (source_trivia.leading_blank_line) {
    block->flags |= LOOM_BLOCK_FLAG_LEADING_BLANK_LINE;
  }
  if (source_trivia.comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_block_comments(
        body_reader->values.output_module, block, source_trivia.comments,
        source_trivia.comment_count));
  }

  uint64_t arg_count = 0;
  uint64_t arg_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &arg_count));
  if (arg_count > UINT16_MAX || arg_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, arg_count_offset,
        IREE_SV("block_argument_count_exceeds_field_width"));
  }
  if (body_reader->expect_predefined_entry_args) {
    body_reader->expect_predefined_entry_args = false;
    if (arg_count != body_reader->values.predefined_value_count) {
      return loom_bytecode_value_scope_emit_invalid(
          &body_reader->values, arg_count_offset,
          IREE_SV("signature_argument_count_does_not_match_region_entry"));
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_value_scope_reserve_definitions(
      &body_reader->values, arg_count, NULL));
  for (uint64_t i = 0; i < arg_count; ++i) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(LOOM_BYTECODE_BODY_VALUE_SCOPE_MATERIALIZE_DEFINITION(
        &body_reader->values, cursor, &value_id));
    IREE_RETURN_IF_ERROR(
        loom_block_add_arg(body_reader->values.output_module, block, value_id));
  }
  body_reader->counts.value_count += arg_count;
  loom_builder_set_block(builder, block);
  uint64_t op_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &op_count));
  if (op_count > UINT16_MAX || op_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, arg_count_offset,
        IREE_SV("block_operation_count_exceeds_field_width"));
  }
  for (uint64_t i = 0; i < op_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_body_reader_read_op(body_reader, cursor, builder, depth));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_read_region(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    loom_op_t* parent_op, uint32_t depth, loom_region_t** out_region) {
  if (depth >= LOOM_BYTECODE_MAX_REGION_DEPTH) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values,
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("region_nesting_exceeds_bytecode_maximum_depth"));
  }
  uint64_t source_flags = 0;
  uint64_t source_flags_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &source_flags));
  if (source_flags > UINT16_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, source_flags_offset,
        IREE_SV("region_source_flags_exceed_field_width"));
  }
  if ((source_flags & ~((uint64_t)LOOM_REGION_SOURCE_FLAG_MASK)) != 0) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, source_flags_offset,
        IREE_SV("region_source_flags_have_unsupported_bits"));
  }
  uint64_t block_count = 0;
  uint64_t block_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->values.decoder, cursor, &block_count));
  if (block_count > UINT16_MAX || block_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_value_scope_emit_invalid(
        &body_reader->values, block_count_offset,
        IREE_SV("region_block_count_exceeds_field_width"));
  }
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(
      body_reader->values.output_module, (uint16_t)block_count, &region));
  region->source_flags = (loom_region_source_flags_t)source_flags;
  ++body_reader->counts.region_count;
  body_reader->counts.block_count += block_count;

  loom_builder_ip_t saved =
      loom_builder_enter_region(builder, parent_op, region);
  iree_status_t status = iree_ok_status();
  for (uint64_t i = 0; i < block_count && iree_status_is_ok(status); ++i) {
    status = loom_bytecode_body_reader_read_block(
        body_reader, cursor, builder, loom_region_block(region, (uint16_t)i),
        depth);
  }
  loom_builder_restore(builder, saved);
  if (iree_status_is_ok(status)) {
    *out_region = region;
  }
  return status;
}

#if defined(LOOM_BYTECODE_BODY_DEFINE_SUMMARY)
iree_status_t loom_bytecode_region_summary_read(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t symbol_name,
    iree_const_byte_span_t payload_bytes, uint64_t payload_absolute_offset,
    loom_bytecode_region_summary_t* out_summary) {
  *out_summary = (loom_bytecode_region_summary_t){0};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      payload_bytes.data, payload_bytes.data_length, payload_absolute_offset,
      IREE_SV("IR"), &cursor);
  uint64_t value_count = 0;
  uint64_t region_count = 0;
  uint64_t block_count = 0;
  uint64_t op_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &value_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &region_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &block_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, &cursor, &op_count));
  if (value_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_body_emit_invalid(
        decoder, symbol_name, payload_absolute_offset,
        IREE_SV("region_payload_value_count_exceeds_host_size"));
  }
  const iree_host_size_t payload_length = cursor.cursor.length;
  if (value_count > UINT32_MAX || region_count > UINT32_MAX ||
      block_count > UINT32_MAX || op_count > UINT32_MAX ||
      value_count > payload_length || region_count > payload_length ||
      block_count > payload_length || op_count > payload_length) {
    return loom_bytecode_body_emit_invalid(
        decoder, symbol_name, payload_absolute_offset,
        IREE_SV("region_allocation_summary_exceeds_payload_length"));
  }

  *out_summary = (loom_bytecode_region_summary_t){
      .value_count = (uint32_t)value_count,
      .region_count = (uint32_t)region_count,
      .block_count = (uint32_t)block_count,
      .op_count = (uint32_t)op_count,
      .payload_offset = (uint8_t)cursor.cursor.position,
  };
  return iree_ok_status();
}
#endif  // LOOM_BYTECODE_BODY_DEFINE_SUMMARY

iree_status_t LOOM_BYTECODE_BODY_MATERIALIZE_REGION(
    loom_bytecode_body_policy_materializer_t* materializer,
    iree_string_view_t symbol_name, iree_const_byte_span_t payload_bytes,
    uint64_t payload_absolute_offset,
    const loom_bytecode_region_summary_t* summary, loom_builder_t* builder,
    loom_op_t* parent_op, uint8_t region_index,
    loom_bytecode_region_materialization_flags_t flags,
    const loom_value_id_t* predefined_values, uint16_t predefined_value_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set) {
  iree_arena_allocator_t body_arena;
  iree_arena_initialize(loom_bytecode_body_policy_block_pool(materializer),
                        &body_arena);

  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      payload_bytes.data + summary->payload_offset,
      payload_bytes.data_length - summary->payload_offset,
      payload_absolute_offset + summary->payload_offset, IREE_SV("IR"),
      &cursor);
  loom_bytecode_body_reader_t body_reader = {
      .materializer = materializer,
      .low_descriptor_set = low_descriptor_set,
  };
  loom_bytecode_body_policy_initialize_value_scope(
      materializer, &body_arena, symbol_name, payload_absolute_offset,
      /*value_map=*/NULL, summary->value_count, &body_reader.values);
  iree_status_t status = iree_ok_status();
  if (predefined_value_count > summary->value_count) {
    status = loom_bytecode_value_scope_emit_invalid(
        &body_reader.values, body_reader.values.payload_offset,
        IREE_SV("predefined_value_count_exceeds_region_summary"));
  }
  if (iree_status_is_ok(status) && region_index >= parent_op->region_count) {
    status = loom_bytecode_value_scope_emit_invalid(
        &body_reader.values, body_reader.values.payload_offset,
        IREE_SV("root_region_index_is_out_of_range_for_symbol_op"));
  }
  if (iree_status_is_ok(status) && summary->value_count > 0) {
    status = iree_arena_allocate_array(&body_arena, summary->value_count,
                                       sizeof(loom_value_id_t),
                                       (void**)&body_reader.values.value_map);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_value_scope_prepare_fresh_values(
        &body_reader.values,
        summary->value_count - (iree_host_size_t)predefined_value_count);
  }
  if (iree_status_is_ok(status)) {
    body_reader.values.predefined_values = predefined_values;
    body_reader.values.predefined_value_start = 0;
    body_reader.values.predefined_value_count = predefined_value_count;
    body_reader.expect_predefined_entry_args = iree_any_bit_set(
        flags, LOOM_BYTECODE_REGION_MATERIALIZATION_FLAG_BIND_ENTRY_ARGUMENTS);
    loom_region_t* region = NULL;
    status = loom_bytecode_body_reader_read_region(
        &body_reader, &cursor, builder, parent_op, 0, &region);
    if (iree_status_is_ok(status)) {
      loom_op_regions(parent_op)[region_index] = region;
    }
  }
  if (iree_status_is_ok(status)) {
    if (body_reader.counts.value_count != summary->value_count ||
        body_reader.counts.region_count != summary->region_count ||
        body_reader.counts.block_count != summary->block_count ||
        body_reader.counts.op_count != summary->op_count) {
      status = loom_bytecode_value_scope_emit_invalid(
          &body_reader.values, body_reader.values.payload_offset,
          IREE_SV("root_region_allocation_summary_does_not_match_ir"));
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_reader_expect_empty(
        loom_bytecode_body_policy_decoder(materializer), &cursor,
        IREE_SV("IR"));
  }
  iree_arena_deinitialize(&body_arena);
  return status;
}
