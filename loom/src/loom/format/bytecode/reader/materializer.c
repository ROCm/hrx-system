// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/materializer.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/reader/encoding.h"
#include "loom/format/bytecode/reader/location.h"
#include "loom/format/bytecode/reader/string_table.h"
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
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
} loom_bytecode_module_materialization_t;

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
      .source_count = reader->view.sources.count,
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
  IREE_RETURN_IF_ERROR(loom_bytecode_string_table_materialize(
      &reader->view, reader->output_module));
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
  IREE_RETURN_IF_ERROR(loom_bytecode_source_table_materialize(
      &reader->view, reader->output_module));
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
    loom_bytecode_location_materializer_t location_materializer = {
        .decoder = &reader->decoder,
        .module_view = &reader->view,
        .output_module = reader->output_module,
    };
    IREE_RETURN_IF_ERROR(loom_bytecode_location_table_materialize(
        &location_materializer, reader->view.sections.locations));
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
