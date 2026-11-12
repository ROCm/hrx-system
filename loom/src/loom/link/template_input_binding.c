// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/template_input_binding.h"

#include "loom/format/bytecode/function_projection_reader.h"
#include "loom/rewrite/remap.h"

struct loom_link_template_input_module_t {
  // Native source identity correspondence, initialized only when needed.
  loom_ir_remap_t native;
  // Bytecode correspondence, NULL for native sources or before first import.
  loom_bytecode_function_projection_reader_t* reader;
  // Caller-bound contracts indexed by source contract ordinal.
  const loom_input_contract_t** contracts;
  // True after native correspondence initialization succeeds.
  bool native_initialized;
};

static iree_status_t loom_link_template_input_import_encoding(
    loom_link_template_input_binding_t* binding,
    const loom_link_module_index_module_t* source,
    loom_link_template_input_module_t* query, uint16_t encoding_id,
    uint16_t* out_encoding_id) {
  if (source->materialized_module) {
    if (!query->native_initialized) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_initialize(source->materialized_module, binding->module,
                                   binding->arena, NULL, &query->native));
      query->native_initialized = true;
    }
    return loom_ir_remap_encoding_id(&query->native, encoding_id,
                                     out_encoding_id);
  }
  if (!query->reader) {
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(binding->index,
                                           source->provider_ordinal);
    const loom_bytecode_module_metadata_t* metadata =
        &provider->bytecode.metadata.modules[source->provider_module_ordinal];
    IREE_RETURN_IF_ERROR(loom_bytecode_function_projection_reader_allocate(
        provider->bytecode.contents, provider->bytecode.filename, metadata,
        binding->module, NULL, binding->arena, &query->reader));
  }
  return loom_bytecode_function_projection_reader_import_encoding(
      query->reader, encoding_id, out_encoding_id);
}

iree_status_t loom_link_template_input_bind(
    loom_link_template_input_binding_t* binding,
    const loom_link_module_index_symbol_t* source_symbol,
    const loom_input_contract_t** out_contract) {
  *out_contract = NULL;
  if (!source_symbol->input_contract_ordinal_plus_one) {
    return iree_ok_status();
  }
  const loom_link_module_index_module_t* source =
      loom_link_module_index_symbol_module(binding->index, source_symbol);
  const uint32_t ordinal = source_symbol->input_contract_ordinal_plus_one - 1;
  const loom_input_contract_t* contract = source->input_contracts[ordinal];
  if (!binding->modules) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        iree_arena_allocator(binding->arena),
        loom_link_module_index_module_count(binding->index),
        sizeof(*binding->modules), (void**)&binding->modules));
  }
  loom_link_template_input_module_t* query = binding->modules[source->ordinal];
  if (!query) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_arena_allocator(binding->arena), sizeof(*query), (void**)&query));
    binding->modules[source->ordinal] = query;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        iree_arena_allocator(binding->arena), source->input_contract_count,
        sizeof(*query->contracts), (void**)&query->contracts));
  }
  if (query->contracts[ordinal]) {
    *out_contract = query->contracts[ordinal];
    return iree_ok_status();
  }
  loom_input_contract_t* bound = NULL;
  for (uint32_t i = 0; i < contract->count; ++i) {
    const loom_input_requirement_t* requirement = &contract->values[i];
    if (requirement->axis != UINT8_MAX || requirement->expected_is_argument ||
        !requirement->expected) {
      continue;
    }
    uint16_t encoding_id;
    IREE_RETURN_IF_ERROR(loom_link_template_input_import_encoding(
        binding, source, query, (uint16_t)requirement->expected, &encoding_id));
    if (encoding_id == requirement->expected) {
      continue;
    }
    if (!bound) {
      const iree_host_size_t length =
          sizeof(*bound) + contract->count * sizeof(*bound->values);
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(binding->arena, length, (void**)&bound));
      memcpy(bound, contract, length);
    }
    bound->values[i].expected = encoding_id;
  }
  *out_contract = query->contracts[ordinal] = bound ? bound : contract;
  return iree_ok_status();
}

void loom_link_template_input_binding_deinitialize(
    loom_link_template_input_binding_t* binding) {
  if (!binding->modules) {
    return;
  }
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(binding->index);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    if (binding->modules[i]) {
      loom_bytecode_function_projection_reader_deinitialize(
          binding->modules[i]->reader);
    }
  }
}
