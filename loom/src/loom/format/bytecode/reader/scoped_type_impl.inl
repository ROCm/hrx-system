// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete scoped-type grammar shared with attribute construction. Each node is
// canonicalized once after all referenced nodes have completed. Only final type
// IDs survive the per-record scratch checkpoint.

iree_status_t LOOM_BYTECODE_TYPE_PROJECT_COMPLETED(
    loom_bytecode_attribute_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* scope,
    uint64_t reference, loom_type_id_t* out_type) {
  const uint64_t ordinal = reference >> 1;
  if (reference & 1) {
    if (ordinal >= scope->bindings->count) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          materializer->decoder, cursor, IREE_SV("incomplete_child"));
    }
    *out_type = scope->bindings->entries[ordinal].type;
    return iree_ok_status();
  }
  if (ordinal >= LOOM_BYTECODE_ATTRIBUTE_TABLES(materializer)->types.count) {
    return loom_bytecode_type_emit_invalid_scoped_record(
        materializer->decoder, cursor, IREE_SV("unknown_static_type"));
  }
#if LOOM_BYTECODE_ATTRIBUTE_SELECTED
  return loom_bytecode_selected_table_materialize_type(materializer, ordinal,
                                                       out_type);
#else
  *out_type = materializer->module_view->types.entries[ordinal].completed_type;
  return iree_ok_status();
#endif
}

static iree_status_t loom_bytecode_read_complete_type(
    loom_bytecode_attribute_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* scope,
    loom_type_id_t* out_type) {
  const uint64_t kind_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint8_t kind_byte = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(materializer->decoder, cursor, &kind_byte));
  loom_type_kind_t kind = LOOM_TYPE_NONE;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_decode_kind(
      materializer->decoder, kind_byte, kind_offset, &kind));
  if (kind == LOOM_TYPE_PARAMETERIZED) {
    return LOOM_BYTECODE_ATTRIBUTE_DECODE_COMPLETE_TYPE(materializer, cursor,
                                                        scope, out_type);
  }
  if (kind == LOOM_TYPE_GROUP || kind == LOOM_TYPE_TILE ||
      kind == LOOM_TYPE_TENSOR || kind == LOOM_TYPE_VECTOR ||
      kind == LOOM_TYPE_VIEW) {
    loom_bytecode_scoped_shape_t shape;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_read_scoped_shape(
        materializer->decoder, cursor, kind, scope->value_count,
        LOOM_BYTECODE_ATTRIBUTE_TABLES(materializer)->encodings.count, &shape));
    uint16_t target_encoding = 0;
    if (shape.attachment == LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC) {
#if LOOM_BYTECODE_ATTRIBUTE_SELECTED
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_encoding(
          materializer, shape.encoding_ordinal, &target_encoding));
#else
      target_encoding = (uint16_t)shape.encoding_ordinal;
#endif
    } else if (shape.attachment == LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA) {
      const loom_value_id_t value_id = scope->values[shape.encoding_ordinal];
      if (value_id > UINT16_MAX) {
        return loom_bytecode_type_emit_invalid_scoped_record(
            materializer->decoder, cursor, IREE_SV("encoding_value"));
      }
      target_encoding = (uint16_t)value_id;
    }
    for (uint8_t i = 0; i < shape.rank; ++i) {
      if (shape.dynamic_axes & (1u << i)) {
        const uint64_t source_value = shape.dimensions[i];
        shape.dimensions[i] =
            loom_dim_pack_dynamic(source_value ? scope->values[source_value - 1]
                                               : LOOM_VALUE_ID_INVALID);
      }
    }
    loom_type_t type = {0};
    type.header = loom_type_make_raw_header(
        kind, shape.element_type, shape.rank,
        (shape.rank <= 2 ? LOOM_TYPE_FLAG_INLINE_DIMS : 0) |
            (!shape.dynamic_axes ? LOOM_TYPE_FLAG_ALL_STATIC : 0));
    type.encoding_id = target_encoding;
    type.encoding_flags =
        shape.attachment == LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA
            ? LOOM_ENCODING_FLAG_SSA
            : 0;
    if (kind == LOOM_TYPE_VIEW) {
      type = loom_type_view_with_alignment(type, shape.access_alignment);
    }
    if (shape.rank > 2) {
      type.dims[0] = (uint64_t)(uintptr_t)shape.dimensions;
    } else {
      for (uint8_t i = 0; i < shape.rank; ++i) {
        type.dims[i] = shape.dimensions[i];
      }
    }
    return loom_module_intern_topological_type_id(materializer->output_module,
                                                  type, NULL, 0, out_type);
  }
  loom_bytecode_structural_type_plan_t plan = {0};
  loom_bytecode_structural_type_fact_t fact = {0};
  if (kind != LOOM_TYPE_FUNCTION && kind != LOOM_TYPE_DIALECT &&
      kind != LOOM_TYPE_REGISTER) {
    return loom_bytecode_type_emit_invalid_scoped_record(
        materializer->decoder, cursor, IREE_SV("invalid_scoped_type_kind"));
  }
  fact.base.kind = kind;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_read_structural_header(
      materializer->decoder, cursor, kind, 0,
      LOOM_BYTECODE_ATTRIBUTE_TABLES(materializer)->strings.count, &plan,
      fact.payload_prefix));
#if LOOM_BYTECODE_ATTRIBUTE_SELECTED
  if (kind == LOOM_TYPE_DIALECT) {
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, plan.name_id, &plan.name_id));
  }
#endif
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_count_bytes(
      materializer->decoder, cursor, IREE_SV("scoped_type_children"),
      plan.dependency_count,
      /*minimum_entry_length=*/1, /*minimum_trailing_length=*/0, kind_offset));
  loom_type_id_t* children = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      materializer->scratch_arena, plan.dependency_count, sizeof(*children),
      (void**)&children));
  for (uint32_t i = 0; i < plan.dependency_count; ++i) {
    uint64_t reference = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        materializer->decoder, cursor, &reference));
    IREE_RETURN_IF_ERROR(LOOM_BYTECODE_TYPE_PROJECT_COMPLETED(
        materializer, cursor, scope, reference, &children[i]));
  }
  return loom_bytecode_type_materialize_structural(
      &plan, &fact, children, materializer->output_module, out_type);
}

iree_status_t LOOM_BYTECODE_TYPE_MATERIALIZE_BINDINGS(
    loom_bytecode_attribute_policy_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* scope,
    uint64_t source_type_id, loom_type_id_t* out_type_id) {
  loom_bytecode_reader_cursor_t records;
  uint64_t count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_open_scoped_records(
      materializer->decoder, cursor, source_type_id, &records, &count));
  if (records.cursor.length == 0) {
    return LOOM_BYTECODE_TYPE_PROJECT_COMPLETED(materializer, cursor, scope,
                                                source_type_id, out_type_id);
  }
  iree_status_t status = iree_ok_status();
  for (uint64_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(materializer->scratch_arena);
    loom_type_id_t type = LOOM_TYPE_ID_INVALID;
    status =
        loom_bytecode_read_complete_type(materializer, &records, scope, &type);
    iree_arena_checkpoint_restore(&checkpoint);
    if (iree_status_is_ok(status) &&
        scope->bindings->count == scope->bindings->capacity) {
      status = iree_arena_grow_array(
          scope->bindings->arena, scope->bindings->count,
          scope->bindings->count + 1, sizeof(*scope->bindings->entries),
          &scope->bindings->capacity, (void**)&scope->bindings->entries);
    }
    if (iree_status_is_ok(status)) {
      scope->bindings->entries[scope->bindings->count++] =
          (loom_bytecode_type_binding_t){.type = type};
    }
  }
  IREE_RETURN_IF_ERROR(status);
  uint64_t root = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_close_scoped_records(
      materializer->decoder, &records, scope->bindings->count, &root));
  *out_type_id = scope->bindings->entries[root].type;
  return iree_ok_status();
}
