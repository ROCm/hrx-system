// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/operation.h"

#include "loom/error/error_catalog.h"

#define LOOM_BYTECODE_MAX_OP_COUNT (UINT64_C(1) << 24)

typedef struct loom_bytecode_operation_validator_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Finalized dialect registry context.
  loom_context_t* context;
  // Scratch storage for dense resolved operation facts.
  iree_arena_allocator_t* scratch_arena;
  // Module facts populated by operation validation.
  loom_bytecode_reader_module_view_t* module_view;
} loom_bytecode_operation_validator_t;

typedef struct loom_bytecode_operation_table_t {
  // Cursor positioned at the next operation entry.
  loom_bytecode_reader_cursor_t cursor;
  // Number of operation entries.
  iree_host_size_t count;
} loom_bytecode_operation_table_t;

static iree_status_t loom_bytecode_operation_table_begin(
    loom_bytecode_operation_validator_t* validator,
    const loom_bytecode_reader_section_t* section,
    loom_bytecode_operation_table_t* out_table) {
  loom_bytecode_operation_table_t table = {0};
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("OPS"), &table.cursor);
  uint64_t count = 0;
  const uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&table.cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table.cursor, &count));
  if (count > LOOM_BYTECODE_MAX_OP_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        validator->decoder, IREE_SV("OPS"), count, LOOM_BYTECODE_MAX_OP_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        validator->scratch_arena, (iree_host_size_t)count,
        sizeof(*validator->module_view->ops.values),
        (void**)&validator->module_view->ops.values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        validator->scratch_arena, (iree_host_size_t)count,
        sizeof(*validator->module_view->ops.kinds),
        (void**)&validator->module_view->ops.kinds));
  }
  table.count = (iree_host_size_t)count;
  validator->module_view->ops.count = table.count;
  *out_table = table;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_operation_decode_entry(
    loom_bytecode_operation_validator_t* validator,
    loom_bytecode_operation_table_t* table, iree_host_size_t index,
    loom_bytecode_op_metadata_t* out_metadata) {
  const uint64_t name_offset =
      loom_bytecode_reader_cursor_absolute_position(&table->cursor);
  uint64_t name_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      validator->decoder, &table->cursor, &name_id));
  if (name_id >= validator->module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("op_name")),
        loom_param_u64(name_id),
        loom_param_u64(validator->module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(
        validator->decoder, LOOM_ERR_BYTECODE_010, params,
        IREE_ARRAYSIZE(params), name_offset, 0);
  }
  const iree_string_view_t op_name =
      validator->module_view->strings.values[name_id];
  loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
  const loom_op_vtable_t* vtable =
      loom_context_lookup_op_by_name(validator->context, op_name, &kind);
  if (!vtable) {
    return loom_bytecode_reader_emit_invalid_field(
        validator->decoder, IREE_SV("OPS"), IREE_SV("op"), index,
        IREE_SV("name_id"), name_offset,
        IREE_SV("op_name_is_not_registered_in_the_context"));
  }
  validator->module_view->ops.values[index] = vtable;
  validator->module_view->ops.kinds[index] = kind;
  *out_metadata = (loom_bytecode_op_metadata_t){.name = op_name};
  return iree_ok_status();
}

static iree_status_t loom_bytecode_operation_table_finish(
    loom_bytecode_operation_validator_t* validator,
    loom_bytecode_operation_table_t* table) {
  return loom_bytecode_reader_expect_empty(validator->decoder, &table->cursor,
                                           IREE_SV("OPS"));
}

iree_status_t loom_bytecode_operation_table_validate(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_operation_validator_t validator = {
      .decoder = decoder,
      .context = context,
      .scratch_arena = scratch_arena,
      .module_view = module_view,
  };
  loom_bytecode_operation_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_operation_table_begin(&validator, section, &table));
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    loom_bytecode_op_metadata_t discarded_metadata;
    IREE_RETURN_IF_ERROR(loom_bytecode_operation_decode_entry(
        &validator, &table, i, &discarded_metadata));
  }
  return loom_bytecode_operation_table_finish(&validator, &table);
}

iree_status_t loom_bytecode_operation_table_index(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_op_metadata_t** out_entries, iree_host_size_t* out_count) {
  *out_entries = NULL;
  *out_count = 0;
  loom_bytecode_operation_validator_t validator = {
      .decoder = decoder,
      .context = context,
      .scratch_arena = scratch_arena,
      .module_view = module_view,
  };
  loom_bytecode_operation_table_t table;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_operation_table_begin(&validator, section, &table));

  loom_bytecode_op_metadata_t* entries = NULL;
  if (table.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        retained_arena, table.count, sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_operation_decode_entry(
        &validator, &table, i, &entries[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_operation_table_finish(&validator, &table));
  *out_entries = entries;
  *out_count = table.count;
  return iree_ok_status();
}
