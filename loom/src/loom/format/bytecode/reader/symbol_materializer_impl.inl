// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Policy-specialized symbol payload materialization implementation.
//
// This file intentionally has no include guard. The including translation unit
// supplies concrete materializer, value-scope, and body-source types plus
// direct source-to-output identity operations. Full and selective readers
// thereby share one symbol wire grammar without runtime modes or dispatch.

static iree_status_t loom_bytecode_reader_read_func_payload_attrs(
    loom_bytecode_symbol_policy_materializer_t* reader,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope,
    uint64_t symbol_index, const loom_op_vtable_t* vtable,
    const loom_func_like_vtable_t* func_like, loom_attribute_t* attrs) {
  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_function_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_resolve_source_string(
        reader, key_id, IREE_SV("function_attribute_key"), key_offset, &key));
    uint8_t attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_symbol_func_metadata_attr_is_shared(vtable, func_like,
                                                          attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_is_reconstructed_from_shared_metadata"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit) ||
        !loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        &reader->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_materialize_attribute_ssa(
        reader, cursor, descriptor, value_kind, &attrs[attr_index], ssa_scope));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_resolve_function_low_descriptor_set(
    loom_bytecode_symbol_policy_materializer_t* reader, uint64_t symbol_index,
    const loom_func_like_vtable_t* func_like, const loom_attribute_t* attrs,
    const loom_low_repr_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (func_like->repr_contract_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }

  const loom_attribute_t repr_contract_attr =
      attrs[func_like->repr_contract_attr_index];
  if (repr_contract_attr.kind != LOOM_ATTR_STRING) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("descriptor_set"), 0,
        IREE_SV("function_representation_contract_must_be_a_string"));
  }
  if (!reader->low_repr_environment.vtable) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "materializing Low functions requires a representation codec");
  }

  const iree_string_view_t repr_contract =
      reader->output_module->strings
          .entries[loom_attr_as_string_id(repr_contract_attr)];
  *out_descriptor_set = loom_low_repr_lookup_descriptor_set(
      &reader->low_repr_environment, repr_contract);
  if (!*out_descriptor_set) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("descriptor_set"), 0,
        IREE_SV("function_representation_contract_is_not_available"));
  }
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_reader_materialize_function_symbol(
    loom_bytecode_symbol_policy_materializer_t* reader,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_symbol_policy_body_source_t* body_source,
    iree_host_size_t symbol_ordinal, uint64_t name_id, uint16_t flags,
    loom_string_id_t import_module_id, loom_string_id_t import_symbol_id,
    loom_builder_t* builder) {
  uint16_t symbol_id =
      loom_bytecode_symbol_policy_lookup_symbol(reader, (uint32_t)name_id);
  loom_symbol_ref_t callee_ref = {0, symbol_id};
  const iree_string_view_t symbol_name =
      loom_bytecode_symbol_policy_source_string(reader, (uint32_t)name_id);

  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  loom_op_kind_t op_kind = LOOM_OP_KIND_UNKNOWN;
  loom_bytecode_symbol_policy_resolve_defining_op(
      reader, symbol_ordinal, op_table_index_plus1, &vtable, &op_kind);
  const loom_func_like_vtable_t* func_like = vtable->func_like;

  loom_bytecode_source_trivia_t source_trivia;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
      &reader->decoder, cursor, reader->arena, &source_trivia));

  uint8_t calling_convention = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(&reader->decoder, cursor,
                                                    &calling_convention));
  uint8_t purity = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &purity));
  uint64_t workload_arg_count = 0;
  uint64_t arg_count = 0;
  uint64_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &workload_arg_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &arg_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &result_count));
  uint64_t signature_value_count =
      workload_arg_count + arg_count + result_count;
  loom_value_id_t* signature_values = NULL;
  if (signature_value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)signature_value_count,
        sizeof(loom_value_id_t), (void**)&signature_values));
  }
  loom_bytecode_symbol_policy_value_scope_t signature_scope;
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_value_scope_initialize_fresh(
      reader, reader->arena, symbol_name,
      loom_bytecode_reader_cursor_absolute_position(cursor), signature_values,
      (iree_host_size_t)signature_value_count, &signature_scope));
  for (uint64_t i = 0; i < workload_arg_count; ++i) {
    loom_value_id_t workload_arg_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_policy_value_scope_materialize_definition(
            &signature_scope, cursor, &workload_arg_id));
  }
  for (uint64_t i = 0; i < arg_count; ++i) {
    loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_policy_value_scope_materialize_definition(
            &signature_scope, cursor, &arg_id));
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)result_count,
        sizeof(loom_tied_result_t), (void**)&tied_results));
  }
  for (uint64_t i = 0; i < result_count; ++i) {
    uint8_t is_tied = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, cursor, &is_tied));
    loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_policy_value_scope_materialize_definition(
            &signature_scope, cursor, &result_id));
    if (is_tied) {
      uint64_t operand_index = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          &reader->decoder, cursor, &operand_index));
      tied_results[tied_result_count++] = (loom_tied_result_t){
          .result_index = (uint16_t)i,
          .operand_index = (uint16_t)operand_index,
      };
    }
  }
  uint64_t encoded_tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &encoded_tied_result_count));
  (void)encoded_tied_result_count;

  const loom_bytecode_attribute_ssa_materialization_scope_t
      attribute_ssa_scope = {
          .symbol_name = symbol_name,
          .values = signature_values,
          .value_count = signature_scope.next_value_number,
      };
  loom_attribute_t predicates_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_materialize_attribute_ssa(
      reader, cursor, /*descriptor=*/NULL, LOOM_BYTECODE_ATTR_PREDICATE_LIST,
      &predicates_attr, &attribute_ssa_scope));
  const bool has_predicates =
      iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES);

  loom_string_id_t implements_id = LOOM_STRING_ID_INVALID;
  int64_t priority = 0;
  if (func_like->implements_attr_index != LOOM_ATTR_INDEX_NONE) {
    const uint64_t implements_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t implements_string_id = 0;
    uint64_t priority_value = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, cursor, &implements_string_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, cursor, &priority_value));
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_project_string(
        reader, implements_string_id, IREE_SV("implements"), implements_offset,
        &implements_id));
    priority = (int64_t)priority_value;
  }

  loom_attribute_t* func_attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, vtable->attribute_count, sizeof(loom_attribute_t),
        (void**)&func_attrs));
    memset(func_attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_func_payload_attrs(
      reader, cursor, &attribute_ssa_scope, symbol_id, vtable, func_like,
      func_attrs));

  func_attrs[func_like->callee_attr_index] = loom_attr_symbol(callee_ref);
  if ((flags & LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC) &&
      func_like->visibility_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->visibility_attr_index] = loom_attr_enum(1);
  }
  if (flags & LOOM_BYTECODE_SYMBOL_FLAG_RETAIN) {
    if (!vtable->symbol_def ||
        !vtable->symbol_def->retain_attr_index_plus_one) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
          IREE_SV("flags"), 0,
          IREE_SV("retained_symbol_op_has_no_retain_attr"));
    }
    const uint8_t retain_attr_index =
        vtable->symbol_def->retain_attr_index_plus_one - 1;
    func_attrs[retain_attr_index] = loom_attr_enum(1);
  }
  if (calling_convention != 0 &&
      func_like->cc_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->cc_attr_index] = loom_attr_enum(calling_convention);
  }
  if (purity != 0 && func_like->purity_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->purity_attr_index] = loom_attr_enum(purity);
  }
  if (import_module_id != LOOM_STRING_ID_INVALID) {
    uint8_t import_module_attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(
            vtable, IREE_SV("import_module"));
    if (import_module_attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
          IREE_SV("import_module"), 0,
          IREE_SV("imported_symbol_op_has_no_import_module_attr"));
    }
    func_attrs[import_module_attr_index] = loom_attr_string(import_module_id);
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL)) {
      uint8_t import_symbol_attr_index =
          loom_bytecode_symbol_find_op_attr_index_by_name(
              vtable, IREE_SV("import_symbol"));
      if (import_symbol_attr_index == LOOM_ATTR_INDEX_NONE) {
        return loom_bytecode_reader_emit_invalid_field(
            &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
            IREE_SV("import_symbol"), 0,
            IREE_SV("imported_symbol_op_has_no_import_symbol_attr"));
      }
      func_attrs[import_symbol_attr_index] = loom_attr_string(import_symbol_id);
    }
  }
  if (has_predicates) {
    func_attrs[func_like->predicates_attr_index] = predicates_attr;
  }
  if (implements_id != LOOM_STRING_ID_INVALID &&
      func_like->implements_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->implements_attr_index] =
        loom_attr_string(implements_id);
  }
  if (priority != 0 && func_like->priority_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->priority_attr_index] = loom_attr_i64(priority);
  }

  uint8_t has_body = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &has_body));
  const loom_low_repr_descriptor_set_t* low_descriptor_set = NULL;
  if (has_body) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_resolve_function_low_descriptor_set(
            reader, symbol_id, func_like, func_attrs, &low_descriptor_set));
  }
  uint64_t ir_offset = 0;
  uint32_t ir_length = 0;
  if (has_body) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(&reader->decoder, cursor, &ir_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(&reader->decoder, cursor, &ir_length));
  }

  uint64_t operand_count = 0;
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  uint16_t* operand_segment_counts = NULL;
  if (operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, operand_segment_count, sizeof(uint16_t),
        (void**)&operand_segment_counts));
    memset(operand_segment_counts, 0,
           (iree_host_size_t)operand_segment_count * sizeof(uint16_t));
  }
  const uint8_t workload_operand_field_index =
      vtable->symbol_def->kernel_workload_operand_field_index_plus_one
          ? vtable->symbol_def->kernel_workload_operand_field_index_plus_one - 1
          : LOOM_OPERAND_INDEX_NONE;
  if (workload_operand_field_index != LOOM_OPERAND_INDEX_NONE) {
    operand_count += workload_arg_count;
    if (operand_segment_counts) {
      operand_segment_counts[workload_operand_field_index] =
          (uint16_t)workload_arg_count;
    }
  }
  if (func_like->args_operand_field_index != LOOM_OPERAND_INDEX_NONE) {
    operand_count += arg_count;
    if (operand_segment_counts) {
      operand_segment_counts[func_like->args_operand_field_index] =
          (uint16_t)arg_count;
    }
  }
  if (operand_count > UINT16_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
        IREE_SV("signature_count"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("declaration_signature_exceeds_operand_count_field_width"));
  }
  uint8_t region_count = has_body ? vtable->region_count : 0;
  loom_op_t* op = NULL;
  if (operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
        builder, op_kind, (uint16_t)operand_count, operand_segment_counts,
        operand_segment_count, (uint16_t)result_count, region_count,
        tied_result_count, vtable->attribute_count, LOOM_LOCATION_NONE, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, op_kind, (uint16_t)operand_count, (uint16_t)result_count,
        region_count, tied_result_count, vtable->attribute_count,
        LOOM_LOCATION_NONE, &op));
  }
  if (source_trivia.leading_blank_line) {
    op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  }
  if (workload_operand_field_index != LOOM_OPERAND_INDEX_NONE) {
    loom_value_slice_t workload_operands =
        loom_op_operand_field_span(vtable, op, workload_operand_field_index);
    if (workload_operands.count > 0) {
      memcpy(workload_operands.values, signature_values,
             workload_operands.count * sizeof(loom_value_id_t));
    }
  }
  if (func_like->args_operand_field_index != LOOM_OPERAND_INDEX_NONE) {
    loom_value_slice_t args = loom_op_operand_field_span(
        vtable, op, func_like->args_operand_field_index);
    if (args.count > 0) {
      memcpy(args.values, signature_values + workload_arg_count,
             args.count * sizeof(loom_value_id_t));
    }
  }
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), func_attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }

  for (uint64_t i = 0; i < result_count; ++i) {
    loom_op_results(op)[i] =
        signature_values[workload_arg_count + arg_count + i];
  }
  if (tied_result_count > 0) {
    memcpy(loom_op_tied_results(op), tied_results,
           tied_result_count * sizeof(loom_tied_result_t));
  }
  if (has_body) {
    loom_bytecode_predefined_region_values_t predefined_regions[2];
    uint8_t predefined_region_count = 0;
    if (func_like->body_region_index != LOOM_REGION_INDEX_NONE) {
      predefined_regions[predefined_region_count++] =
          (loom_bytecode_predefined_region_values_t){
              .region_index = func_like->body_region_index,
              .values =
                  arg_count ? signature_values + workload_arg_count : NULL,
              .count = (uint16_t)arg_count,
          };
    }
    if (vtable->symbol_def->kernel_workload_region_index_plus_one) {
      predefined_regions[predefined_region_count++] =
          (loom_bytecode_predefined_region_values_t){
              .region_index =
                  vtable->symbol_def->kernel_workload_region_index_plus_one - 1,
              .values = workload_arg_count ? signature_values : NULL,
              .count = (uint16_t)workload_arg_count,
          };
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_materialize_body(
        reader, symbol_name, body_source, symbol_ordinal, ir_offset, ir_length,
        builder, op, func_like->body_region_index, predefined_regions,
        predefined_region_count, low_descriptor_set));
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (source_trivia.comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        reader->output_module, op, source_trivia.comments,
        source_trivia.comment_count));
  }
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_reader_materialize_global_symbol(
    loom_bytecode_symbol_policy_materializer_t* reader,
    loom_bytecode_reader_cursor_t* cursor, uint64_t name_id,
    uint64_t symbol_index, loom_builder_t* builder) {
  uint16_t symbol_id =
      loom_bytecode_symbol_policy_lookup_symbol(reader, (uint32_t)name_id);
  loom_symbol_ref_t symbol_ref = {0, symbol_id};
  const iree_string_view_t symbol_name =
      loom_bytecode_symbol_policy_source_string(reader, (uint32_t)name_id);

  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  loom_op_kind_t op_kind = LOOM_OP_KIND_UNKNOWN;
  loom_bytecode_symbol_policy_resolve_defining_op(
      reader, symbol_index, op_table_index_plus1, &vtable, &op_kind);

  loom_bytecode_source_trivia_t source_trivia;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
      &reader->decoder, cursor, reader->arena, &source_trivia));

  uint64_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &result_count));
  uint64_t local_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &local_value_count));

  loom_value_id_t* local_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, (iree_host_size_t)local_value_count,
      sizeof(loom_value_id_t), (void**)&local_values));
  loom_bytecode_symbol_policy_value_scope_t global_scope;
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_value_scope_initialize_fresh(
      reader, reader->arena, symbol_name,
      loom_bytecode_reader_cursor_absolute_position(cursor), local_values,
      (iree_host_size_t)local_value_count, &global_scope));
  for (uint64_t i = 0; i < local_value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_policy_value_scope_materialize_definition(
            &global_scope, cursor, &local_values[i]));
  }

  loom_attribute_t* attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, vtable->attribute_count,
                                  sizeof(loom_attribute_t), (void**)&attrs));
    memset(attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  uint8_t symbol_attr_index =
      loom_bytecode_symbol_find_identity_attr_index(vtable);
  attrs[symbol_attr_index] = loom_attr_symbol(symbol_ref);

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_global_attribute_count_exceeds_op_attribute_slots"));
  }
  const loom_bytecode_attribute_ssa_materialization_scope_t
      attribute_ssa_scope = {
          .symbol_name = symbol_name,
          .values = local_values,
          .value_count = global_scope.next_value_number,
      };
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_resolve_source_string(
        reader, key_id, IREE_SV("global_attribute_key"), key_offset, &key));
    uint8_t attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_is_not_declared_by_the_op"));
    }
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    if (loom_bytecode_symbol_attr_is_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    if (!loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_appears_more_than_once"));
    }
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        &reader->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_materialize_attribute_ssa(
        reader, cursor, descriptor, value_kind, &attrs[attr_index],
        &attribute_ssa_scope));
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op_kind, 0, (uint16_t)result_count, 0, 0,
      vtable->attribute_count, LOOM_LOCATION_NONE, &op));
  if (source_trivia.leading_blank_line) {
    op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  }
  memcpy(loom_op_results(op), local_values,
         (iree_host_size_t)result_count * sizeof(loom_value_id_t));
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (source_trivia.comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        reader->output_module, op, source_trivia.comments,
        source_trivia.comment_count));
  }
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_reader_materialize_record_symbol(
    loom_bytecode_symbol_policy_materializer_t* reader,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_symbol_policy_body_source_t* body_source,
    uint64_t name_id, uint64_t symbol_index, loom_builder_t* builder) {
  uint16_t symbol_id =
      loom_bytecode_symbol_policy_lookup_symbol(reader, (uint32_t)name_id);
  loom_symbol_ref_t symbol_ref = {0, symbol_id};

  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  loom_op_kind_t op_kind = LOOM_OP_KIND_UNKNOWN;
  loom_bytecode_symbol_policy_resolve_defining_op(
      reader, symbol_index, op_table_index_plus1, &vtable, &op_kind);

  loom_bytecode_source_trivia_t source_trivia;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_materialize(
      &reader->decoder, cursor, reader->arena, &source_trivia));

  loom_attribute_t* attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, vtable->attribute_count,
                                  sizeof(loom_attribute_t), (void**)&attrs));
    memset(attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  uint8_t symbol_attr_index =
      loom_bytecode_symbol_find_identity_attr_index(vtable);
  attrs[symbol_attr_index] = loom_attr_symbol(symbol_ref);

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_record_attribute_count_exceeds_op_attribute_slots"));
  }
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &key_id));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_resolve_source_string(
        reader, key_id, IREE_SV("record_attribute_key"), key_offset, &key));
    uint8_t attr_index =
        loom_bytecode_symbol_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_is_not_declared_by_the_op"));
    }
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    if (loom_bytecode_symbol_attr_is_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    if (!loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_appears_more_than_once"));
    }
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        &reader->decoder, cursor, &value_kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_policy_materialize_attribute_named(
            reader, cursor, descriptor, value_kind, &attrs[attr_index]));
  }

  uint8_t has_body = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &has_body));

  uint64_t ir_offset = 0;
  uint32_t ir_length = 0;
  if (has_body) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(&reader->decoder, cursor, &ir_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(&reader->decoder, cursor, &ir_length));
  }

  loom_op_t* op = NULL;
  uint8_t region_count = has_body ? vtable->region_count : 0;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op_kind, 0, 0, region_count, 0, vtable->attribute_count,
      LOOM_LOCATION_NONE, &op));
  if (source_trivia.leading_blank_line) {
    op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  }
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }
  if (has_body) {
    const iree_string_view_t symbol_name =
        loom_bytecode_symbol_policy_source_string(reader, (uint32_t)name_id);
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_materialize_body(
        reader, symbol_name, body_source, symbol_index, ir_offset, ir_length,
        builder, op, LOOM_REGION_INDEX_NONE, /*predefined_regions=*/NULL,
        /*predefined_region_count=*/0, /*low_descriptor_set=*/NULL));
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (source_trivia.comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        reader->output_module, op, source_trivia.comments,
        source_trivia.comment_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_symbol_materialize_entry(
    loom_bytecode_symbol_policy_materializer_t* reader,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_symbol_policy_body_source_t* body_source,
    iree_host_size_t symbol_ordinal, loom_builder_t* builder) {
  uint64_t unused_name_id = 0;
  const uint64_t name_id =
      loom_bytecode_symbol_policy_source_name_id(reader, symbol_ordinal);
  uint8_t unused_kind = 0;
  const uint8_t kind = loom_bytecode_symbol_policy_kind(reader, symbol_ordinal);
  uint8_t visibility = 0;
  uint16_t unused_flags = 0;
  const uint16_t flags =
      loom_bytecode_symbol_policy_flags(reader, symbol_ordinal);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &unused_name_id));
  uint64_t kind_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &unused_kind));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(&reader->decoder, cursor, &visibility));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u16_le(&reader->decoder,
                                                        cursor, &unused_flags));
  (void)visibility;
  loom_string_id_t import_module_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t import_symbol_id = LOOM_STRING_ID_INVALID;
  if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
    const uint64_t import_module_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t encoded_import_module_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, cursor, &encoded_import_module_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_project_string(
        reader, encoded_import_module_id, IREE_SV("import_module"),
        import_module_offset, &import_module_id));
    const uint64_t import_symbol_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t encoded_import_symbol_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        &reader->decoder, cursor, &encoded_import_symbol_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_policy_project_string(
        reader, encoded_import_symbol_id, IREE_SV("import_symbol"),
        import_symbol_offset, &import_symbol_id));
  }
  if (kind <= LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
    return loom_bytecode_reader_materialize_function_symbol(
        reader, cursor, body_source, symbol_ordinal, name_id, flags,
        import_module_id, import_symbol_id, builder);
  } else if (kind == LOOM_BYTECODE_SYMBOL_GLOBAL) {
    return loom_bytecode_reader_materialize_global_symbol(
        reader, cursor, name_id, symbol_ordinal, builder);
  } else if (kind == LOOM_BYTECODE_SYMBOL_RECORD) {
    return loom_bytecode_reader_materialize_record_symbol(
        reader, cursor, body_source, name_id, symbol_ordinal, builder);
  } else if (kind == LOOM_BYTECODE_SYMBOL_ANCHOR) {
    return iree_ok_status();
  } else {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_ordinal,
        IREE_SV("kind"), kind_offset,
        IREE_SV("module_materialization_does_not_support_this_symbol_kind"));
  }
}
