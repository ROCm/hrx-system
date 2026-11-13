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

typedef struct loom_bytecode_module_materialization_t {
  // Complete source bytecode containing retained table spans.
  iree_const_byte_span_t bytecode;
  // Finalized dialect and registry context.
  loom_context_t* context;
  // Resettable storage for transient materialization state.
  iree_arena_allocator_t* arena;
  // Checkpointed construction payloads never own completed scope bindings.
  iree_arena_allocator_t construction_arena;
  // Bounded decoder sharing the public diagnostic result state.
  loom_bytecode_reader_decoder_t* decoder;
  // Validated module facts whose type-plan slots become completed identities.
  loom_bytecode_reader_module_view_t view;
  // Block source for the output module and root-region-local arenas.
  iree_arena_block_pool_t* block_pool;
  // Host allocator owning the output module.
  iree_allocator_t host_allocator;
  // Module under construction.
  loom_module_t* output_module;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
} loom_bytecode_module_materialization_t;

static iree_status_t loom_bytecode_module_allocate_output(
    loom_bytecode_module_materialization_t* reader) {
  loom_module_size_hints_t hints = {
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

static bool loom_bytecode_module_materialize_tables(
    loom_bytecode_module_materialization_t* reader) {
  loom_bytecode_symbol_materializer_t symbol_materializer;
  loom_bytecode_symbol_materializer_initialize(
      reader->decoder, reader->context, reader->arena,
      &reader->construction_arena, reader->block_pool, &reader->view,
      reader->output_module, &reader->low_repr_environment,
      &symbol_materializer);
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
      reader->decoder, loom_bytecode_string_table_materialize(
                           &reader->view, reader->output_module)));
  loom_string_id_t module_name_id = LOOM_STRING_ID_INVALID;
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
      reader->decoder,
      loom_module_intern_string(reader->output_module,
                                reader->view.directory_entry->name,
                                &module_name_id)));
  reader->output_module->name_id = module_name_id;
  if (reader->view.file_header.comment_count > 0) {
    LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
        reader->decoder,
        loom_module_attach_file_header(
            reader->output_module, reader->view.file_header.comments,
            reader->view.file_header.comment_count)));
  }
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
      reader->decoder, loom_bytecode_source_table_materialize(
                           &reader->view, reader->output_module)));
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
      reader->decoder, loom_bytecode_symbols_predeclare(&symbol_materializer)));
  loom_bytecode_type_materializer_t type_materializer = {
      .decoder = reader->decoder,
      .bytecode = reader->bytecode,
      .context = reader->context,
      .module_view = &reader->view,
      .scratch_arena = &reader->construction_arena,
      .output_module = reader->output_module,
      .next_fact = reader->view.types.facts,
  };
  loom_bytecode_encoding_materializer_t encoding_materializer = {
      .decoder = reader->decoder,
      .context = reader->context,
      .module_view = &reader->view,
      .scratch_arena = &reader->construction_arena,
      .output_module = reader->output_module,
      .types = &type_materializer,
  };
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_encoding_table_materialize(
      &encoding_materializer, reader->view.sections.encodings));
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_type_materialize_prefix(
      &type_materializer, reader->view.types.count));
  if (reader->view.sections.locations) {
    loom_bytecode_location_materializer_t location_materializer = {
        .decoder = reader->decoder,
        .module_view = &reader->view,
        .output_module = reader->output_module,
    };
    LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_location_table_materialize(
        &location_materializer, reader->view.sections.locations));
  }
  return loom_bytecode_symbols_materialize(&symbol_materializer,
                                           reader->view.sections.symbols,
                                           reader->view.sections.ir);
}

bool loom_bytecode_module_materialize(
    const loom_bytecode_module_materializer_t* materializer,
    loom_module_t** out_module) {
  *out_module = NULL;
  loom_bytecode_module_materialization_t state = {
      .bytecode = materializer->bytecode,
      .context = materializer->context,
      .arena = materializer->scratch_arena,
      .decoder = materializer->decoder,
      .view = *materializer->module_view,
      .block_pool = materializer->block_pool,
      .host_allocator = materializer->host_allocator,
      .low_repr_environment = materializer->low_repr_environment,
  };

  iree_arena_initialize(materializer->block_pool, &state.construction_arena);
  bool accepted = loom_bytecode_reader_check_status(
      materializer->decoder, loom_bytecode_module_allocate_output(&state));
  if (accepted) {
    accepted = loom_bytecode_module_materialize_tables(&state);
  }
  if (accepted) {
    *out_module = state.output_module;
    state.output_module = NULL;
  }
  if (state.output_module) {
    loom_module_free(state.output_module);
  }
  iree_arena_deinitialize(&state.construction_arena);
  return accepted;
}
