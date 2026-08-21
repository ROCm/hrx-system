// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_materializer.h"

typedef struct loom_bytecode_selected_module_preparation_t {
  // Prepared selected symbols in canonical source order.
  loom_bytecode_selected_symbol_t* symbols;
  // Exact number of module values materialized by the selection.
  iree_host_size_t value_count;
} loom_bytecode_selected_module_preparation_t;

static iree_const_byte_span_t loom_bytecode_selected_body_span(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const loom_bytecode_symbol_metadata_t* symbol) {
  IREE_ASSERT(symbol->has_body);
  IREE_ASSERT(symbol->body_absolute_offset <=
              materializer->bytecode.data_length);
  IREE_ASSERT(symbol->body_length <= materializer->bytecode.data_length -
                                         symbol->body_absolute_offset);
  return iree_make_const_byte_span(
      materializer->bytecode.data +
          (iree_host_size_t)symbol->body_absolute_offset,
      symbol->body_length);
}

static iree_status_t loom_bytecode_selected_module_add_value_count(
    iree_host_size_t additional_count,
    loom_bytecode_selected_module_preparation_t* preparation) {
  iree_host_size_t value_count = 0;
  if (!iree_host_size_checked_add(preparation->value_count, additional_count,
                                  &value_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "selected module value count overflow");
  }
  preparation->value_count = value_count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_module_prepare(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_bytecode_selected_module_preparation_t* out_preparation) {
  *out_preparation = (loom_bytecode_selected_module_preparation_t){0};
  loom_bytecode_selected_symbol_t* selected_symbols = NULL;
  if (ordinal_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        materializer->scratch_arena, ordinal_count, sizeof(*selected_symbols),
        (void**)&selected_symbols));
  }
  loom_bytecode_selected_module_preparation_t preparation = {
      .symbols = selected_symbols,
  };
  for (iree_host_size_t i = 0; i < ordinal_count; ++i) {
    const iree_host_size_t source_ordinal = ordinals[i];
    IREE_ASSERT(source_ordinal < materializer->metadata->symbol_count);
    IREE_ASSERT(i == 0 || source_ordinal > ordinals[i - 1]);
    const loom_bytecode_symbol_metadata_t* symbol =
        &materializer->metadata->symbols[source_ordinal];
    loom_bytecode_selected_symbol_t* selected = &selected_symbols[i];
    selected->source_ordinal = (uint32_t)source_ordinal;
    if (symbol->has_body) {
      const iree_const_byte_span_t body_bytes =
          loom_bytecode_selected_body_span(materializer, symbol);
      IREE_RETURN_IF_ERROR(loom_bytecode_body_summary_read(
          materializer->decoder, symbol->name, body_bytes,
          symbol->body_absolute_offset, &selected->body_summary));
    }

    iree_host_size_t value_count = 0;
    switch (symbol->kind) {
      case LOOM_BYTECODE_SYMBOL_FUNC_DEF:
      case LOOM_BYTECODE_SYMBOL_FUNC_DECL:
      case LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL:
      case LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF:
      case LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL: {
        const iree_host_size_t prefix_count =
            symbol->has_body
                ? (iree_host_size_t)symbol->result_count
                : (iree_host_size_t)symbol->kernel_workload_argument_count +
                      symbol->argument_count + symbol->result_count;
        const iree_host_size_t body_value_count =
            symbol->has_body ? selected->body_summary.value_count : 0;
        if (!iree_host_size_checked_add(prefix_count, body_value_count,
                                        &value_count)) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "selected function value count overflow");
        }
        break;
      }
      case LOOM_BYTECODE_SYMBOL_GLOBAL:
        value_count = (iree_host_size_t)symbol->local_value_count;
        break;
      case LOOM_BYTECODE_SYMBOL_RECORD:
        value_count = selected->body_summary.value_count;
        break;
      case LOOM_BYTECODE_SYMBOL_EXECUTABLE:
      case LOOM_BYTECODE_SYMBOL_ANCHOR:
        break;
      case LOOM_BYTECODE_SYMBOL_COUNT_:
        IREE_ASSERT_UNREACHABLE("validated bytecode symbol kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_module_add_value_count(
        value_count, &preparation));
  }
  *out_preparation = preparation;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_module_allocate(
    const loom_bytecode_selected_module_materializer_t* materializer,
    iree_host_size_t selected_symbol_count,
    const loom_bytecode_selected_module_preparation_t* preparation,
    loom_module_t** out_module) {
  iree_host_size_t string_count = 0;
  if (!iree_host_size_checked_add(selected_symbol_count, 1, &string_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "selected module string count overflow");
  }
  const loom_module_size_hints_t hints = {
      .value_count = preparation->value_count,
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

iree_status_t loom_bytecode_selected_module_materialize(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_module_t** out_module) {
  *out_module = NULL;
  loom_bytecode_selected_module_preparation_t preparation;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_module_prepare(
      materializer, ordinals, ordinal_count, &preparation));

  loom_module_t* output_module = NULL;
  iree_status_t status = loom_bytecode_selected_module_allocate(
      materializer, ordinal_count, &preparation, &output_module);
  loom_bytecode_selected_table_materializer_t tables;
  if (iree_status_is_ok(status)) {
    loom_bytecode_selected_table_materializer_initialize(
        materializer->decoder, materializer->bytecode, materializer->context,
        materializer->metadata, materializer->scratch_arena, output_module,
        loom_bytecode_selected_symbol_resolver_empty(),
        materializer->host_allocator, &tables);
    loom_bytecode_selected_symbol_materializer_t symbols;
    loom_bytecode_selected_symbol_materializer_initialize(
        materializer->decoder, materializer->block_pool, &tables,
        &materializer->low_repr_environment, &symbols);
    status = loom_bytecode_selected_symbols_materialize(
        &symbols, preparation.symbols, ordinal_count);
    loom_bytecode_selected_table_materializer_deinitialize(&tables);
  }
  if (iree_status_is_ok(status)) {
    *out_module = output_module;
    output_module = NULL;
  }
  loom_module_free(output_module);
  return status;
}
