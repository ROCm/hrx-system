// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/function_projection_reader.h"

#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/selected_body.h"
#include "loom/format/bytecode/reader/selected_symbol.h"
#include "loom/format/bytecode/reader/selected_tables.h"

struct loom_bytecode_function_projection_reader_t {
  // Host allocator owning this reader and transient selected-table storage.
  iree_allocator_t allocator;
  // Reset-free storage retaining decoded header payloads for the reader life.
  iree_arena_allocator_t arena;
  // Complete source bytecode retaining every indexed payload.
  iree_const_byte_span_t bytecode;
  // Validated source module metadata.
  const loom_bytecode_module_metadata_t* metadata;
  // Caller-owned module receiving selected identities and root IR.
  loom_module_t* output_module;
  // Number of structured bytecode errors emitted while projecting.
  uint32_t error_count;
  // Bounded decoder sharing error_count and the caller's diagnostic sink.
  loom_bytecode_reader_decoder_t decoder;
  // Reached-only shared-table projection retained across selected reads.
  loom_bytecode_selected_table_materializer_t tables;
  // Function header and root-region materializer sharing tables.
  loom_bytecode_selected_symbol_materializer_t symbols;
};

iree_status_t loom_bytecode_function_projection_reader_allocate(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    iree_arena_block_pool_t* block_pool,
    const loom_bytecode_module_metadata_t* metadata,
    loom_module_t* output_module,
    const loom_bytecode_function_projection_reader_options_t* options,
    iree_allocator_t allocator,
    loom_bytecode_function_projection_reader_t** out_reader) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(output_module);
  IREE_ASSERT_ARGUMENT(out_reader);
  *out_reader = NULL;

  loom_bytecode_function_projection_reader_t* reader = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*reader), (void**)&reader));
  reader->allocator = allocator;
  reader->bytecode = bytecode;
  reader->metadata = metadata;
  reader->output_module = output_module;
  reader->error_count = 0;
  iree_arena_initialize(block_pool, &reader->arena);
  loom_bytecode_reader_decoder_initialize(
      options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      filename, &reader->error_count, &reader->decoder);
  loom_bytecode_selected_table_materializer_initialize(
      &reader->decoder, bytecode, output_module->context, metadata,
      &reader->arena, output_module,
      options && options->symbol_resolver
          ? loom_bytecode_selected_symbol_resolver_make(
                options->symbol_resolver, options->symbol_resolver_user_data)
          : loom_bytecode_selected_symbol_resolver_empty(),
      allocator, &reader->tables);
  const loom_low_repr_environment_t low_repr_environment =
      options ? options->low_repr_environment
              : (loom_low_repr_environment_t){0};
  loom_bytecode_selected_symbol_materializer_initialize(
      &reader->decoder, block_pool, &reader->tables, &low_repr_environment,
      &reader->symbols);
  *out_reader = reader;
  return iree_ok_status();
}

void loom_bytecode_function_projection_reader_free(
    loom_bytecode_function_projection_reader_t* reader) {
  if (reader == NULL) return;
  const iree_allocator_t allocator = reader->allocator;
  loom_bytecode_selected_table_materializer_deinitialize(&reader->tables);
  iree_arena_deinitialize(&reader->arena);
  iree_allocator_free(allocator, reader);
}

iree_status_t loom_bytecode_function_projection_reader_bind_symbol(
    loom_bytecode_function_projection_reader_t* reader,
    uint32_t source_symbol_ordinal, loom_symbol_ref_t target_symbol_ref) {
  IREE_ASSERT_ARGUMENT(reader);
  if (source_symbol_ordinal >= reader->metadata->symbol_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source symbol ordinal is out of range");
  }
  if (target_symbol_ref.module_id != 0 ||
      target_symbol_ref.symbol_id >= reader->output_module->symbols.count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "target symbol reference is out of range");
  }
  const uint32_t source_name_string_ordinal =
      reader->metadata->symbols[source_symbol_ordinal].name_string_index;
  return loom_bytecode_selected_table_bind_symbol(
      &reader->tables, source_name_string_ordinal, target_symbol_ref.symbol_id);
}

iree_status_t loom_bytecode_function_projection_reader_read_header(
    loom_bytecode_function_projection_reader_t* reader,
    uint32_t source_symbol_ordinal,
    loom_bytecode_function_header_t* out_header) {
  IREE_ASSERT_ARGUMENT(reader);
  IREE_ASSERT_ARGUMENT(out_header);
  if (source_symbol_ordinal >= reader->metadata->symbol_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source symbol ordinal is out of range");
  }
  return loom_bytecode_selected_function_header_materialize(
      &reader->symbols, source_symbol_ordinal, out_header);
}

iree_status_t loom_bytecode_function_projection_reader_materialize_region(
    loom_bytecode_function_projection_reader_t* reader,
    const loom_bytecode_function_header_t* header,
    uint8_t source_region_payload_ordinal, loom_builder_t* builder,
    loom_op_t* target_parent_op, uint8_t target_region_index,
    const loom_value_id_t* predefined_values, uint16_t predefined_value_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set) {
  IREE_ASSERT_ARGUMENT(reader);
  IREE_ASSERT_ARGUMENT(header);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(target_parent_op);
  IREE_ASSERT(header->source_symbol_ordinal < reader->metadata->symbol_count);
  if (source_region_payload_ordinal >= header->region_payload_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source region payload ordinal is out of range");
  }
  const loom_bytecode_symbol_metadata_t* source_symbol =
      &reader->metadata->symbols[header->source_symbol_ordinal];
  IREE_ASSERT(source_region_payload_ordinal <
              source_symbol->region_payload_count);
  const loom_bytecode_region_payload_metadata_t* payload =
      &reader->metadata
           ->region_payloads[source_symbol->first_region_payload_index +
                             source_region_payload_ordinal];
  const loom_bytecode_region_payload_metadata_t* header_payload =
      &header->region_payloads[source_region_payload_ordinal];
  IREE_ASSERT(payload->region_index == header_payload->region_index);
  IREE_ASSERT(payload->offset == header_payload->offset);
  IREE_ASSERT(payload->length == header_payload->length);
  IREE_ASSERT(payload->absolute_offset <= reader->bytecode.data_length);
  IREE_ASSERT(payload->length <=
              reader->bytecode.data_length - payload->absolute_offset);
  const iree_const_byte_span_t payload_bytes = iree_make_const_byte_span(
      reader->bytecode.data + (iree_host_size_t)payload->absolute_offset,
      payload->length);
  loom_bytecode_region_summary_t summary = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_region_summary_read(
      &reader->decoder, header->name, payload_bytes, payload->absolute_offset,
      &summary));
  return loom_bytecode_selected_body_materialize_region(
      &reader->symbols.body_materializer, header->name, payload_bytes,
      payload->absolute_offset, &summary, builder, target_parent_op,
      target_region_index,
      LOOM_BYTECODE_REGION_MATERIALIZATION_FLAG_BIND_ENTRY_ARGUMENTS,
      predefined_values, predefined_value_count, low_descriptor_set);
}
