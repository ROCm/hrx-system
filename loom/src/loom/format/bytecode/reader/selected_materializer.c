// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_materializer.h"

#include "loom/format/bytecode/reader/module_ops.h"

typedef struct loom_bytecode_selected_module_preparation_t {
  // Prepared selected symbols in canonical source order.
  loom_bytecode_selected_symbol_t* symbols;
  // Exact number of module values materialized by the selection.
  iree_host_size_t value_count;
  // Optional non-symbol module operation payload and validated summary.
  struct {
    // Source MODULE_OPS section retained by the bytecode index.
    const loom_bytecode_section_metadata_t* section;
    // Validated allocation summary for section.
    loom_bytecode_module_ops_summary_t summary;
  } module_ops;
} loom_bytecode_selected_module_preparation_t;

static const loom_bytecode_section_metadata_t*
loom_bytecode_selected_module_find_section(
    const loom_bytecode_module_metadata_t* metadata, uint16_t kind) {
  for (iree_host_size_t i = 0; i < metadata->section_count; ++i) {
    if (metadata->sections[i].kind == kind) return &metadata->sections[i];
  }
  return NULL;
}

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
  preparation.module_ops.section = loom_bytecode_selected_module_find_section(
      materializer->metadata, LOOM_BYTECODE_SECTION_MODULE_OPS);
  if (preparation.module_ops.section) {
    IREE_RETURN_IF_ERROR(loom_bytecode_module_ops_summary_read(
        materializer->decoder, preparation.module_ops.section->bytes,
        preparation.module_ops.section->absolute_offset,
        &preparation.module_ops.summary));
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_module_add_value_count(
        preparation.module_ops.summary.value_count, &preparation));
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
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          materializer->scratch_arena, selected->region_summary_count,
          sizeof(*selected->region_summaries),
          (void**)&selected->region_summaries));
      for (uint8_t region_ordinal = 0;
           region_ordinal < selected->region_summary_count; ++region_ordinal) {
        const loom_bytecode_region_payload_metadata_t* payload =
            &materializer->metadata
                 ->region_payloads[symbol->first_region_payload_index +
                                   region_ordinal];
        const iree_const_byte_span_t payload_bytes =
            loom_bytecode_selected_region_span(materializer, payload);
        IREE_RETURN_IF_ERROR(loom_bytecode_region_summary_read(
            materializer->decoder, symbol->name, payload_bytes,
            payload->absolute_offset,
            &selected->region_summaries[region_ordinal]));
      }
    }

    iree_host_size_t value_count = 0;
    switch (symbol->kind) {
      case LOOM_BYTECODE_SYMBOL_FUNC_DEF:
      case LOOM_BYTECODE_SYMBOL_FUNC_DECL:
      case LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL:
      case LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF:
      case LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL: {
        const iree_host_size_t prefix_count =
            symbol->region_payload_count > 0
                ? (iree_host_size_t)symbol->result_count
                : (iree_host_size_t)symbol->kernel_workload_argument_count +
                      symbol->argument_count + symbol->result_count;
        value_count = prefix_count;
        for (uint8_t region_ordinal = 0;
             region_ordinal < selected->region_summary_count;
             ++region_ordinal) {
          if (!iree_host_size_checked_add(
                  value_count,
                  selected->region_summaries[region_ordinal].value_count,
                  &value_count)) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "selected function value count overflow");
          }
        }
        break;
      }
      case LOOM_BYTECODE_SYMBOL_GLOBAL:
        value_count = (iree_host_size_t)symbol->local_value_count;
        break;
      case LOOM_BYTECODE_SYMBOL_RECORD:
        for (uint8_t region_ordinal = 0;
             region_ordinal < selected->region_summary_count;
             ++region_ordinal) {
          if (!iree_host_size_checked_add(
                  value_count,
                  selected->region_summaries[region_ordinal].value_count,
                  &value_count)) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "selected record value count overflow");
          }
        }
        break;
      case LOOM_BYTECODE_SYMBOL_EXECUTABLE:
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

static iree_status_t loom_bytecode_selected_module_materialize_prepared_into(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const loom_bytecode_selected_module_preparation_t* preparation,
    iree_host_size_t selected_symbol_count,
    loom_bytecode_selected_symbol_resolver_t symbol_resolver,
    loom_module_t* output_module) {
  loom_bytecode_selected_table_materializer_t tables;
  loom_bytecode_selected_table_materializer_initialize(
      materializer->decoder, materializer->bytecode, materializer->context,
      materializer->metadata, materializer->scratch_arena, output_module,
      symbol_resolver, materializer->host_allocator, &tables);
  loom_bytecode_selected_symbol_materializer_t symbols;
  loom_bytecode_selected_symbol_materializer_initialize(
      materializer->decoder, materializer->block_pool, &tables,
      &materializer->low_repr_environment, &symbols);
  iree_status_t status = loom_bytecode_selected_symbols_materialize(
      &symbols, preparation->symbols, selected_symbol_count);
  if (iree_status_is_ok(status) && preparation->module_ops.section) {
    status = loom_bytecode_selected_body_materialize_module_ops(
        &symbols.body_materializer, materializer->metadata->name,
        preparation->module_ops.section->bytes,
        preparation->module_ops.section->absolute_offset,
        &preparation->module_ops.summary);
  }
  loom_bytecode_selected_table_materializer_deinitialize(&tables);
  return status;
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
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_selected_module_materialize_prepared_into(
        materializer, &preparation, ordinal_count,
        loom_bytecode_selected_symbol_resolver_empty(), output_module);
  }
  if (iree_status_is_ok(status)) {
    *out_module = output_module;
    output_module = NULL;
  }
  loom_module_free(output_module);
  return status;
}

iree_status_t loom_bytecode_selected_module_materialize_into(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_bytecode_selected_symbol_resolver_t symbol_resolver,
    loom_module_t* output_module) {
  IREE_ASSERT_ARGUMENT(materializer);
  IREE_ASSERT_ARGUMENT(output_module);
  IREE_ASSERT(output_module->context == materializer->context);
  loom_bytecode_selected_module_preparation_t preparation;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_module_prepare(
      materializer, ordinals, ordinal_count, &preparation));
  return loom_bytecode_selected_module_materialize_prepared_into(
      materializer, &preparation, ordinal_count, symbol_resolver,
      output_module);
}
