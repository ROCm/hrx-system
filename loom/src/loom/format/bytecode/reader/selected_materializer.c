// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_materializer.h"

static iree_const_byte_span_t loom_bytecode_selected_region_span(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const loom_bytecode_region_payload_metadata_t* payload) {
  IREE_ASSERT(payload->absolute_offset <= materializer->bytecode.data_length);
  IREE_ASSERT(payload->length <=
              materializer->bytecode.data_length - payload->absolute_offset);
  return iree_make_const_byte_span(
      materializer->bytecode.data + (iree_host_size_t)payload->absolute_offset,
      payload->length);
}

static bool loom_bytecode_selected_module_prepare(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_bytecode_selected_symbol_t** out_symbols) {
  *out_symbols = NULL;
  loom_bytecode_selected_symbol_t* selected_symbols = NULL;
  if (ordinal_count > 0) {
    LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
        materializer->decoder,
        iree_arena_allocate_array(materializer->scratch_arena, ordinal_count,
                                  sizeof(*selected_symbols),
                                  (void**)&selected_symbols)));
  }
  for (iree_host_size_t i = 0; i < ordinal_count; ++i) {
    const iree_host_size_t source_ordinal = ordinals[i];
    IREE_ASSERT(source_ordinal < materializer->metadata->symbol_count);
    IREE_ASSERT(i == 0 || source_ordinal > ordinals[i - 1]);
    const loom_bytecode_symbol_metadata_t* symbol =
        &materializer->metadata->symbols[source_ordinal];
    loom_bytecode_selected_symbol_t* selected = &selected_symbols[i];
    selected->source_ordinal = (uint32_t)source_ordinal;
    selected->region_summary_count = symbol->region_payload_count;
    if (selected->region_summary_count > 0) {
      LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_check_status(
          materializer->decoder,
          iree_arena_allocate_array(materializer->scratch_arena,
                                    selected->region_summary_count,
                                    sizeof(*selected->region_summaries),
                                    (void**)&selected->region_summaries)));
      for (uint8_t region_ordinal = 0;
           region_ordinal < selected->region_summary_count; ++region_ordinal) {
        const loom_bytecode_region_payload_metadata_t* payload =
            &materializer->metadata
                 ->region_payloads[symbol->first_region_payload_index +
                                   region_ordinal];
        const iree_const_byte_span_t payload_bytes =
            loom_bytecode_selected_region_span(materializer, payload);
        LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_region_summary_read(
            materializer->decoder, symbol->name, payload_bytes,
            payload->absolute_offset,
            &selected->region_summaries[region_ordinal]));
      }
    }
  }
  *out_symbols = selected_symbols;
  return true;
}

static iree_status_t loom_bytecode_selected_module_allocate(
    const loom_bytecode_selected_module_materializer_t* materializer,
    iree_host_size_t selected_symbol_count, loom_module_t** out_module) {
  // The selection is bounded by the indexed module's 16-bit symbol domain.
  const iree_host_size_t string_count = selected_symbol_count + 1;
  loom_module_size_hints_t hints = {
      .string_count = string_count,
      .type_count = selected_symbol_count,
      .encoding_count = 0,
      .source_count = 0,
      .symbol_count = selected_symbol_count,
  };
  return loom_module_allocate(materializer->context,
                              materializer->metadata->name,
                              materializer->block_pool, &hints,
                              materializer->host_allocator, out_module);
}

static bool loom_bytecode_selected_module_materialize_prepared_into(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const loom_bytecode_selected_symbol_t* selected_symbols,
    iree_host_size_t selected_symbol_count,
    loom_bytecode_selected_symbol_resolver_t symbol_resolver,
    loom_module_t* output_module) {
  loom_bytecode_selected_table_materializer_t tables;
  loom_bytecode_selected_table_materializer_initialize(
      materializer->decoder, materializer->bytecode, materializer->context,
      materializer->metadata, materializer->scratch_arena, output_module,
      symbol_resolver, &tables);
  loom_bytecode_selected_symbol_materializer_t symbols;
  loom_bytecode_selected_symbol_materializer_initialize(
      materializer->decoder, materializer->block_pool, &tables,
      &materializer->low_repr_environment, &symbols);
  bool accepted = loom_bytecode_reader_check_status(
      materializer->decoder,
      loom_bytecode_selected_symbols_predeclare(&symbols, selected_symbols,
                                                selected_symbol_count));
  if (accepted) {
    accepted = loom_bytecode_selected_symbols_materialize(
        &symbols, selected_symbols, selected_symbol_count);
  }
  loom_bytecode_selected_table_materializer_deinitialize(&tables);
  return accepted;
}

bool loom_bytecode_selected_module_materialize(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_module_t** out_module) {
  *out_module = NULL;
  loom_bytecode_selected_symbol_t* selected_symbols = NULL;
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_selected_module_prepare(
      materializer, ordinals, ordinal_count, &selected_symbols));

  loom_module_t* output_module = NULL;
  bool accepted = loom_bytecode_reader_check_status(
      materializer->decoder, loom_bytecode_selected_module_allocate(
                                 materializer, ordinal_count, &output_module));
  if (accepted) {
    accepted = loom_bytecode_selected_module_materialize_prepared_into(
        materializer, selected_symbols, ordinal_count,
        loom_bytecode_selected_symbol_resolver_empty(), output_module);
  }
  if (accepted) {
    *out_module = output_module;
    output_module = NULL;
  }
  loom_module_free(output_module);
  return accepted;
}

bool loom_bytecode_selected_module_materialize_into(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_bytecode_selected_symbol_resolver_t symbol_resolver,
    loom_module_t* output_module) {
  IREE_ASSERT_ARGUMENT(materializer);
  IREE_ASSERT_ARGUMENT(output_module);
  IREE_ASSERT(output_module->context == materializer->context);
  loom_bytecode_selected_symbol_t* selected_symbols = NULL;
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_selected_module_prepare(
      materializer, ordinals, ordinal_count, &selected_symbols));
  return loom_bytecode_selected_module_materialize_prepared_into(
      materializer, selected_symbols, ordinal_count, symbol_resolver,
      output_module);
}
