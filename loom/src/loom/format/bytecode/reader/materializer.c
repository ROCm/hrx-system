// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/materializer.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/encoding.h"
#include "loom/format/bytecode/reader/symbol_materializer.h"
#include "loom/format/bytecode/reader/type.h"
#include "loom/ops/module/ops.h"

typedef struct loom_bytecode_module_materialization_t {
  // Complete source bytecode containing retained table spans.
  iree_const_byte_span_t bytecode;
  // Finalized dialect and registry context.
  loom_context_t* context;
  // Resettable storage for transient materialization state.
  iree_arena_allocator_t* arena;
  // Bounded decoder sharing the public diagnostic result state.
  loom_bytecode_reader_decoder_t decoder;
  // Immutable validated module facts consumed by every table materializer.
  loom_bytecode_reader_module_view_t view;
  // Block source for the output module and body-local arenas.
  iree_arena_block_pool_t* block_pool;
  // Host allocator owning the output module.
  iree_allocator_t host_allocator;
  // Module under construction.
  loom_module_t* output_module;
  // Materialized module source IDs in validated source order.
  loom_source_id_t* source_ids;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
} loom_bytecode_module_materialization_t;

static iree_status_t loom_bytecode_module_materialize_strings(
    loom_bytecode_module_materialization_t* reader) {
  for (iree_host_size_t i = 0; i < reader->view.strings.count; ++i) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        reader->output_module, reader->view.strings.values[i], &string_id));
    if (string_id != i) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("STRINGS"), IREE_SV("string"), i,
          IREE_SV("string"), 0,
          IREE_SV("string_table_must_be_deduplicated_and_preserve_intern_ids"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_module_materialize_sources(
    loom_bytecode_module_materialization_t* reader) {
  if (reader->view.sources.count > LOOM_SOURCE_ID_INVALID) {
    return loom_bytecode_reader_emit_count_exceeds(
        &reader->decoder, IREE_SV("SOURCES"), reader->view.sources.count,
        LOOM_SOURCE_ID_INVALID, 0);
  }
  if (reader->view.sources.count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, reader->view.sources.count, sizeof(loom_source_id_t),
      (void**)&reader->source_ids));
  for (iree_host_size_t i = 0; i < reader->view.sources.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_register_source(
        reader->output_module, reader->view.sources.values[i],
        &reader->source_ids[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_module_read_location_coordinate(
    loom_bytecode_module_materialization_t* reader,
    loom_bytecode_reader_cursor_t* cursor, uint64_t location_index,
    iree_string_view_t field_name, uint16_t* out_value) {
  uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, cursor, &value));
  if (value > UINT16_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
        location_index, field_name, offset,
        IREE_SV("file_location_coordinate_exceeds_runtime_field_width"));
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_module_read_source_ref(
    loom_bytecode_module_materialization_t* reader,
    loom_bytecode_reader_cursor_t* cursor, loom_source_id_t* out_source_id) {
  uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_index = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      &reader->decoder, cursor, &source_index));
  if (source_index >= reader->view.sources.count || !reader->source_ids) {
    return loom_bytecode_reader_emit_table_ref(
        &reader->decoder, IREE_SV("SOURCES"), source_index,
        reader->view.sources.count, offset);
  }
  *out_source_id = reader->source_ids[source_index];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_module_materialize_locations(
    loom_bytecode_module_materialization_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &cursor);

  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &count));
  if (count != reader->view.locations.count) {
    return loom_bytecode_reader_emit_invalid_field(
        &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), 0,
        IREE_SV("count"), count_offset,
        IREE_SV("location_count_changed_between_validation_and_materialize"));
  }

  for (uint64_t i = 0; i < count; ++i) {
    uint8_t kind = 0;
    uint8_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(&reader->decoder, &cursor, &flags));
    if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("flags"), kind_offset + 1,
          IREE_SV("location_has_unsupported_flag_bits"));
    }

    loom_location_entry_t entry = {
        .kind = (loom_location_kind_t)kind,
        .flags = flags,
    };
    switch (kind) {
      case LOOM_LOCATION_NONE:
        if (i != 0 || flags != 0) {
          return loom_bytecode_reader_emit_invalid_field(
              &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
              IREE_SV("kind"), kind_offset,
              IREE_SV("only_location_0_may_be_the_unflagged_none_location"));
        }
        continue;
      case LOOM_LOCATION_FILE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_module_read_source_ref(
            reader, &cursor, &entry.file.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_module_read_location_coordinate(
            reader, &cursor, i, IREE_SV("start_line"), &entry.file.start_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_module_read_location_coordinate(
            reader, &cursor, i, IREE_SV("start_col"), &entry.file.start_col));
        IREE_RETURN_IF_ERROR(loom_bytecode_module_read_location_coordinate(
            reader, &cursor, i, IREE_SV("end_line"), &entry.file.end_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_module_read_location_coordinate(
            reader, &cursor, i, IREE_SV("end_col"), &entry.file.end_col));
        break;
      }
      case LOOM_LOCATION_FUSED: {
        uint64_t child_count_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t child_count = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &child_count));
        if (child_count > UINT32_MAX || child_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              &reader->decoder, IREE_SV("location_children"), child_count,
              UINT32_MAX, child_count_offset);
        }
        loom_location_id_t* children = NULL;
        if (child_count > 0) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &reader->output_module->arena, (iree_host_size_t)child_count,
              sizeof(loom_location_id_t), (void**)&children));
        }
        for (uint64_t child_index = 0; child_index < child_count;
             ++child_index) {
          uint64_t child_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t child = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              &reader->decoder, &cursor, &child));
          if (child >= i) {
            return loom_bytecode_reader_emit_table_ref(
                &reader->decoder, IREE_SV("LOCATIONS"), child, i, child_offset);
          }
          children[child_index] = (loom_location_id_t)child;
        }
        entry.fused.count = (uint32_t)child_count;
        entry.fused.children = children;
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_module_read_source_ref(
            reader, &cursor, &entry.opaque.source_id));
        uint64_t data_length_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &data_length));
        if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              &reader->decoder, IREE_SV("opaque_location_data"), data_length,
              UINT32_MAX, data_length_offset);
        }
        iree_const_byte_span_t data_span = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            &reader->decoder, &cursor, data_length, &data_span));
        uint8_t* data = NULL;
        if (data_span.data_length > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(&reader->output_module->arena,
                                  data_span.data_length, (void**)&data));
          memcpy(data, data_span.data, data_span.data_length);
        }
        entry.opaque.data_length = (uint32_t)data_span.data_length;
        entry.opaque.data = data;
        break;
      }
      case LOOM_LOCATION_TAGGED: {
        uint64_t tag_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t tag = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(&reader->decoder, &cursor, &tag));
        if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
              IREE_SV("tag"), tag_offset,
              IREE_SV("tagged location tag must be in [1, 65535]"));
        }
        uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t child = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &child));
        if (child >= i) {
          return loom_bytecode_reader_emit_table_ref(
              &reader->decoder, IREE_SV("LOCATIONS"), child, i, child_offset);
        }
        uint64_t data_length_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            &reader->decoder, &cursor, &data_length));
        if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              &reader->decoder, IREE_SV("tagged_location_data"), data_length,
              UINT32_MAX, data_length_offset);
        }
        iree_const_byte_span_t data_span = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            &reader->decoder, &cursor, data_length, &data_span));
        uint8_t* data = NULL;
        if (data_span.data_length > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(&reader->output_module->arena,
                                  data_span.data_length, (void**)&data));
          memcpy(data, data_span.data, data_span.data_length);
        }
        entry.tagged.tag = (loom_location_tag_t)tag;
        entry.tagged.child = (loom_location_id_t)child;
        entry.tagged.data_length = (uint32_t)data_span.data_length;
        entry.tagged.data = data;
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            &reader->decoder, IREE_SV("location_kind"), kind,
            LOOM_LOCATION_COUNT_, kind_offset);
    }

    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(
        loom_module_add_location(reader->output_module, entry, &location_id));
    if (location_id != i) {
      return loom_bytecode_reader_emit_invalid_field(
          &reader->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("location_id"), kind_offset,
          IREE_SV("location_table_must_preserve_bytecode_location_ids"));
    }
  }
  return loom_bytecode_reader_expect_empty(&reader->decoder, &cursor,
                                           IREE_SV("LOCATIONS"));
}

static iree_status_t loom_bytecode_module_materialize_provider_imports(
    loom_bytecode_module_materialization_t* reader) {
  loom_symbol_ref_t* anchors = NULL;
  if (reader->view.provider_imports.anchor_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&reader->output_module->arena,
                                  reader->view.provider_imports.anchor_count,
                                  sizeof(*anchors), (void**)&anchors));
    memcpy(anchors, reader->view.provider_imports.anchors,
           reader->view.provider_imports.anchor_count * sizeof(*anchors));
  }

  loom_builder_t builder;
  loom_builder_initialize(reader->output_module, &reader->output_module->arena,
                          loom_module_block(reader->output_module), &builder);
  for (iree_host_size_t provider_index = 0;
       provider_index < reader->view.provider_imports.count; ++provider_index) {
    const loom_bytecode_reader_provider_import_t* provider_import =
        &reader->view.provider_imports.values[provider_index];
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        &builder, LOOM_OP_MODULE_IMPORT, /*operand_count=*/0,
        /*result_count=*/0, /*region_count=*/0, /*tied_result_count=*/0,
        /*attribute_count=*/2, LOOM_LOCATION_NONE, &op));
    loom_attribute_t* attrs = loom_op_attrs(op);
    attrs[loom_module_import_provider_ATTR_INDEX] =
        loom_attr_string(provider_import->provider_id);
    loom_symbol_ref_t* provider_anchors =
        provider_import->anchor_count > 0
            ? anchors + provider_import->first_anchor_index
            : NULL;
    attrs[loom_module_import_symbols_ATTR_INDEX] = loom_attr_symbol_set(
        provider_anchors, (uint16_t)provider_import->anchor_count);
    if (provider_import->source_trivia.leading_blank_line) {
      op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
    }
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(&builder, op));
    if (provider_import->source_trivia.comment_count > 0) {
      IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
          reader->output_module, op, provider_import->source_trivia.comments,
          provider_import->source_trivia.comment_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_module_allocate_output(
    loom_bytecode_module_materialization_t* reader) {
  loom_module_size_hints_t hints = {
      .value_count = (iree_host_size_t)reader->view.summary.value_count,
      .string_count = reader->view.strings.count,
      .type_count = reader->view.types.count,
      .encoding_count = reader->view.encodings.count,
      .symbol_count = reader->view.symbols.count,
  };
  // Bytecode string IDs must materialize 1:1 into module string IDs. Allocate
  // with an empty module name so STRINGS[0] can remain the value-name sentinel.
  return loom_module_allocate(reader->context, iree_string_view_empty(),
                              reader->block_pool, &hints,
                              reader->host_allocator, &reader->output_module);
}

static iree_status_t loom_bytecode_module_materialize_tables(
    loom_bytecode_module_materialization_t* reader) {
  loom_bytecode_symbol_materializer_t symbol_materializer;
  loom_bytecode_symbol_materializer_initialize(
      &reader->decoder, reader->context, reader->arena, reader->block_pool,
      &reader->view, reader->output_module, &reader->low_repr_environment,
      &symbol_materializer);
  IREE_RETURN_IF_ERROR(loom_bytecode_module_materialize_strings(reader));
  loom_string_id_t module_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      reader->output_module, reader->view.directory_entry->name,
      &module_name_id));
  reader->output_module->name_id = module_name_id;
  if (reader->view.file_header.comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_file_header(
        reader->output_module, reader->view.file_header.comments,
        reader->view.file_header.comment_count));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_module_materialize_sources(reader));
  loom_bytecode_encoding_materializer_t encoding_materializer = {
      .decoder = &reader->decoder,
      .context = reader->context,
      .module_view = &reader->view,
      .scratch_arena = reader->arena,
      .output_module = reader->output_module,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_encoding_table_materialize(
      &encoding_materializer, reader->view.sections.encodings));
  IREE_RETURN_IF_ERROR(loom_bytecode_symbols_predeclare(&symbol_materializer));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_module_materialize_provider_imports(reader));
  loom_bytecode_type_materializer_t type_materializer = {
      .decoder = &reader->decoder,
      .bytecode = reader->bytecode,
      .context = reader->context,
      .module_view = &reader->view,
      .scratch_arena = reader->arena,
      .output_module = reader->output_module,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_type_materialize(&type_materializer));
  if (reader->view.sections.locations) {
    IREE_RETURN_IF_ERROR(loom_bytecode_module_materialize_locations(
        reader, reader->view.sections.locations));
  }
  return loom_bytecode_symbols_materialize(&symbol_materializer,
                                           reader->view.sections.symbols,
                                           reader->view.sections.ir);
}

iree_status_t loom_bytecode_module_materialize(
    const loom_bytecode_module_materializer_t* materializer,
    loom_module_t** out_module) {
  *out_module = NULL;
  loom_bytecode_module_materialization_t state = {
      .bytecode = materializer->bytecode,
      .context = materializer->context,
      .arena = materializer->scratch_arena,
      .decoder = *materializer->decoder,
      .view = *materializer->module_view,
      .block_pool = materializer->block_pool,
      .host_allocator = materializer->host_allocator,
      .low_repr_environment = materializer->low_repr_environment,
  };

  iree_status_t status = loom_bytecode_module_allocate_output(&state);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_module_materialize_tables(&state);
  }
  if (iree_status_is_ok(status)) {
    *out_module = state.output_module;
    state.output_module = NULL;
  }
  if (state.output_module) {
    loom_module_free(state.output_module);
  }
  return status;
}
