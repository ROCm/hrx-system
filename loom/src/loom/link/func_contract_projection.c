// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/func_contract_projection.h"

#include <string.h>

#include "loom/format/bytecode/function_projection_reader.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/remap.h"

typedef struct loom_link_func_contract_projection_module_t {
  // Owning projection.
  loom_link_func_contract_projection_t* projection;
  // Indexed source module.
  const loom_link_module_index_module_t* indexed_module;
  // Reusable materialized-source remap, initialized on first use.
  loom_ir_remap_t materialized_remap;
  // Lazy bytecode header reader, or NULL before first use.
  loom_bytecode_function_projection_reader_t* bytecode_reader;
} loom_link_func_contract_projection_module_t;

struct loom_link_func_contract_projection_t {
  // Borrowed immutable provider index.
  const loom_link_module_index_t* index;
  // Host allocator owning this object and bytecode readers.
  iree_allocator_t allocator;
  // Block pool backing the scratch arena, identity module, and readers.
  iree_arena_block_pool_t* block_pool;
  // Reset-free storage for projection tables and contract arrays.
  iree_arena_allocator_t arena;
  // Private module owning all projected values, types, and attributes.
  loom_module_t* identity_module;
  // Identity-module symbol refs indexed by index-wide symbol ordinal.
  loom_symbol_ref_t* symbol_refs;
  // Lazy per-index-module projection state.
  loom_link_func_contract_projection_module_t** modules;
  // Lazy cached contract views indexed by index-wide symbol ordinal.
  loom_link_func_contract_t** contracts;
};

static iree_status_t loom_link_func_contract_projection_add_symbol(
    loom_link_func_contract_projection_t* projection,
    const loom_link_module_index_symbol_t* symbol, loom_symbol_ref_t* out_ref) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(projection->identity_module,
                                                 symbol->name, &name_id));
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(projection->identity_module,
                                              name_id, &out_ref->symbol_id));
  out_ref->module_id = 0;
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_symbol_ref(
    loom_link_func_contract_projection_t* projection,
    const loom_link_module_index_symbol_t* symbol, loom_symbol_ref_t* out_ref) {
  const loom_link_module_index_symbol_t* identity = symbol;
  if (symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    identity =
        loom_link_module_index_lookup_global(projection->index, symbol->name);
    IREE_ASSERT(identity);
  }
  loom_symbol_ref_t ref = projection->symbol_refs[identity->ordinal];
  if (!loom_symbol_ref_is_valid(ref)) {
    IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_add_symbol(
        projection, identity, &ref));
    projection->symbol_refs[identity->ordinal] = ref;
  }
  projection->symbol_refs[symbol->ordinal] = ref;
  *out_ref = ref;
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_get_module_state(
    loom_link_func_contract_projection_t* projection,
    iree_host_size_t module_ordinal,
    loom_link_func_contract_projection_module_t** out_module) {
  loom_link_func_contract_projection_module_t* module =
      projection->modules[module_ordinal];
  if (module == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&projection->arena,
                                             sizeof(*module), (void**)&module));
    *module = (loom_link_func_contract_projection_module_t){
        .projection = projection,
        .indexed_module =
            loom_link_module_index_module_at(projection->index, module_ordinal),
    };
    IREE_ASSERT(module->indexed_module);
    projection->modules[module_ordinal] = module;
  }
  *out_module = module;
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_link_func_contract_projection_module_t* module =
      (loom_link_func_contract_projection_module_t*)user_data;
  IREE_ASSERT_EQ(source_module, module->indexed_module->materialized_module);
  IREE_ASSERT_EQ(target_module, module->projection->identity_module);
  IREE_ASSERT_EQ(source_ref.module_id, 0);
  IREE_ASSERT_LT(source_ref.symbol_id, module->indexed_module->symbol_count);
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(
          module->projection->index,
          module->indexed_module->symbol_start_ordinal + source_ref.symbol_id);
  IREE_ASSERT(symbol);
  return loom_link_func_contract_projection_symbol_ref(module->projection,
                                                       symbol, out_target_ref);
}

static iree_status_t loom_link_func_contract_projection_initialize_remap(
    loom_link_func_contract_projection_module_t* module) {
  if (module->materialized_remap.source_module != NULL) {
    return iree_ok_status();
  }
  const loom_ir_remap_options_t options = {
      .remap_symbol = loom_ir_remap_symbol_callback_make(
          loom_link_func_contract_projection_remap_symbol, module),
      .value_map_kind = LOOM_IR_REMAP_VALUE_MAP_SOURCE_INDEXED,
  };
  return loom_ir_remap_initialize(module->indexed_module->materialized_module,
                                  module->projection->identity_module,
                                  &module->projection->arena, &options,
                                  &module->materialized_remap);
}

static iree_status_t loom_link_func_contract_projection_values(
    loom_link_func_contract_projection_t* projection, loom_ir_remap_t* remap,
    const loom_link_func_contract_t* source, loom_value_id_t** out_values) {
  *out_values = NULL;
  const iree_host_size_t value_count = source->workload_arguments.count +
                                       source->arguments.count +
                                       source->results.count;
  if (value_count == 0) {
    return iree_ok_status();
  }

  loom_value_id_t* source_values = NULL;
  loom_value_id_t* target_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &projection->arena, value_count, sizeof(*source_values),
      (void**)&source_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &projection->arena, value_count, sizeof(*target_values),
      (void**)&target_values));
  iree_host_size_t position = 0;
  if (source->workload_arguments.count != 0) {
    memcpy(source_values + position, source->workload_arguments.values,
           source->workload_arguments.count * sizeof(*source_values));
  }
  position += source->workload_arguments.count;
  if (source->arguments.count != 0) {
    memcpy(source_values + position, source->arguments.values,
           source->arguments.count * sizeof(*source_values));
  }
  position += source->arguments.count;
  if (source->results.count != 0) {
    memcpy(source_values + position, source->results.values,
           source->results.count * sizeof(*source_values));
  }

  loom_value_id_t target_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_define_untyped_values(
      projection->identity_module, value_count, &target_base));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    target_values[i] = target_base + (loom_value_id_t)i;
  }
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(remap, source_values,
                                                target_values, value_count));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_type_t target_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(source->module, source_values[i]),
        &target_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        projection->identity_module, target_values[i], target_type));
  }
  *out_values = target_values;
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_attributes(
    loom_link_func_contract_projection_t* projection, loom_ir_remap_t* remap,
    const loom_link_func_contract_t* source, uint8_t attribute_count,
    loom_attribute_t** out_attributes) {
  *out_attributes = NULL;
  if (attribute_count == 0) {
    return iree_ok_status();
  }
  loom_attribute_t* attributes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&projection->arena, attribute_count,
                                sizeof(*attributes), (void**)&attributes));
  for (uint8_t i = 0; i < attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_attribute(remap, source->attributes[i], &attributes[i]));
  }
  *out_attributes = attributes;
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_tied_results(
    loom_link_func_contract_projection_t* projection,
    const loom_link_func_contract_t* source,
    const loom_tied_result_t** out_tied_results) {
  *out_tied_results = NULL;
  if (source->tied_result_count == 0) {
    return iree_ok_status();
  }
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&projection->arena, source->tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));
  memcpy(tied_results, source->tied_results,
         source->tied_result_count * sizeof(*tied_results));
  *out_tied_results = tied_results;
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_materialized_load(
    loom_link_func_contract_projection_t* projection,
    loom_link_func_contract_projection_module_t* module,
    const loom_link_module_index_symbol_t* symbol,
    loom_link_func_contract_t* out_contract) {
  IREE_RETURN_IF_ERROR(
      loom_link_func_contract_projection_initialize_remap(module));
  const loom_symbol_t* source_symbol =
      &module->indexed_module->materialized_module->symbols
           .entries[symbol->module_symbol_ordinal];
  IREE_ASSERT(source_symbol->defining_op);
  const loom_op_vtable_t* vtable = loom_op_vtable(
      module->indexed_module->materialized_module, source_symbol->defining_op);
  IREE_ASSERT(vtable);
  IREE_ASSERT(vtable->func_like);
  const loom_link_func_contract_t source = loom_link_func_contract_from_op(
      module->indexed_module->materialized_module, source_symbol->defining_op);

  loom_value_id_t* values = NULL;
  loom_attribute_t* attributes = NULL;
  const loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_values(
      projection, &module->materialized_remap, &source, &values));
  IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_attributes(
      projection, &module->materialized_remap, &source, vtable->attribute_count,
      &attributes));
  IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_tied_results(
      projection, &source, &tied_results));

  const iree_host_size_t workload_count = source.workload_arguments.count;
  const iree_host_size_t argument_count = source.arguments.count;
  *out_contract = (loom_link_func_contract_t){
      .module = projection->identity_module,
      .symbol_definition = source.symbol_definition,
      .function = source.function,
      .workload_arguments =
          {
              .values = values,
              .count = workload_count,
          },
      .arguments =
          {
              .values = values ? values + workload_count : NULL,
              .count = argument_count,
          },
      .results =
          {
              .values =
                  values ? values + workload_count + argument_count : NULL,
              .count = source.results.count,
          },
      .tied_results = tied_results,
      .tied_result_count = source.tied_result_count,
      .attributes = attributes,
  };
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_resolve_bytecode_symbol(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_ref) {
  loom_link_func_contract_projection_module_t* module =
      (loom_link_func_contract_projection_module_t*)user_data;
  IREE_ASSERT_LT(source_symbol_ordinal, module->indexed_module->symbol_count);
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(
          module->projection->index,
          module->indexed_module->symbol_start_ordinal + source_symbol_ordinal);
  IREE_ASSERT(symbol);
  return loom_link_func_contract_projection_symbol_ref(module->projection,
                                                       symbol, out_target_ref);
}

static iree_status_t loom_link_func_contract_projection_initialize_reader(
    loom_link_func_contract_projection_module_t* module) {
  if (module->bytecode_reader != NULL) {
    return iree_ok_status();
  }
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(
          module->projection->index, module->indexed_module->provider_ordinal);
  IREE_ASSERT(provider);
  IREE_ASSERT_EQ(provider->kind, LOOM_LINK_PROVIDER_BYTECODE);
  const loom_bytecode_module_metadata_t* metadata =
      &provider->bytecode.metadata
           .modules[module->indexed_module->provider_module_ordinal];
  const loom_bytecode_function_projection_reader_options_t options = {
      .symbol_resolver =
          loom_link_func_contract_projection_resolve_bytecode_symbol,
      .symbol_resolver_user_data = module,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_function_projection_reader_allocate(
      provider->bytecode.contents, provider->bytecode.filename,
      module->projection->block_pool, metadata,
      module->projection->identity_module, &options,
      module->projection->allocator, &module->bytecode_reader));
  return iree_ok_status();
}

static iree_status_t loom_link_func_contract_projection_bytecode_load(
    loom_link_func_contract_projection_t* projection,
    loom_link_func_contract_projection_module_t* module,
    const loom_link_module_index_symbol_t* symbol,
    loom_link_func_contract_t* out_contract) {
  IREE_RETURN_IF_ERROR(
      loom_link_func_contract_projection_initialize_reader(module));
  loom_symbol_ref_t projected_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_symbol_ref(
      projection, symbol, &projected_symbol));
  IREE_RETURN_IF_ERROR(loom_bytecode_function_projection_reader_bind_symbol(
      module->bytecode_reader, (uint32_t)symbol->module_symbol_ordinal,
      projected_symbol));
  loom_bytecode_function_header_t header = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_function_projection_reader_read_header(
      module->bytecode_reader, (uint32_t)symbol->module_symbol_ordinal,
      &header));
  *out_contract = (loom_link_func_contract_t){
      .module = projection->identity_module,
      .symbol_definition = header.vtable->symbol_def,
      .function = header.func_like,
      .workload_arguments =
          {
              .values = header.signature_values,
              .count = header.workload_argument_count,
          },
      .arguments =
          {
              .values =
                  header.signature_values
                      ? header.signature_values + header.workload_argument_count
                      : NULL,
              .count = header.argument_count,
          },
      .results =
          {
              .values = header.signature_values
                            ? header.signature_values +
                                  header.workload_argument_count +
                                  header.argument_count
                            : NULL,
              .count = header.result_count,
          },
      .tied_results = header.tied_results,
      .tied_result_count = header.tied_result_count,
      .attributes = header.attributes,
  };
  return iree_ok_status();
}

iree_status_t loom_link_func_contract_projection_allocate(
    const loom_link_module_index_t* index, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator,
    loom_link_func_contract_projection_t** out_projection) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_projection);
  *out_projection = NULL;

  loom_link_func_contract_projection_t* projection = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*projection),
                                             (void**)&projection));
  memset(projection, 0, sizeof(*projection));
  projection->index = index;
  projection->allocator = allocator;
  projection->block_pool = block_pool;
  iree_arena_initialize(block_pool, &projection->arena);

  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(index);
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(index);
  iree_status_t status = loom_module_allocate(
      loom_link_module_index_context(index), IREE_SV("link.contracts"),
      block_pool, /*hints=*/NULL, allocator, &projection->identity_module);
  if (iree_status_is_ok(status) && symbol_count != 0) {
    status = iree_arena_allocate_array(&projection->arena, symbol_count,
                                       sizeof(*projection->symbol_refs),
                                       (void**)&projection->symbol_refs);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < symbol_count; ++i) {
        projection->symbol_refs[i] = loom_symbol_ref_null();
      }
    }
  }
  if (iree_status_is_ok(status) && symbol_count != 0) {
    status = iree_arena_allocate_array(&projection->arena, symbol_count,
                                       sizeof(*projection->contracts),
                                       (void**)&projection->contracts);
    if (iree_status_is_ok(status)) {
      memset(projection->contracts, 0,
             symbol_count * sizeof(*projection->contracts));
    }
  }
  if (iree_status_is_ok(status) && module_count != 0) {
    status = iree_arena_allocate_array(&projection->arena, module_count,
                                       sizeof(*projection->modules),
                                       (void**)&projection->modules);
    if (iree_status_is_ok(status)) {
      memset(projection->modules, 0,
             module_count * sizeof(*projection->modules));
    }
  }
  if (iree_status_is_ok(status)) {
    *out_projection = projection;
  } else {
    loom_link_func_contract_projection_free(projection);
  }
  return status;
}

void loom_link_func_contract_projection_free(
    loom_link_func_contract_projection_t* projection) {
  if (projection == NULL) {
    return;
  }
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(projection->index);
  for (iree_host_size_t i = 0; projection->modules && i < module_count; ++i) {
    loom_link_func_contract_projection_module_t* module =
        projection->modules[i];
    if (module != NULL) {
      loom_bytecode_function_projection_reader_free(module->bytecode_reader);
    }
  }
  loom_module_free(projection->identity_module);
  iree_arena_deinitialize(&projection->arena);
  iree_allocator_free(projection->allocator, projection);
}

loom_module_t* loom_link_func_contract_projection_module(
    const loom_link_func_contract_projection_t* projection) {
  return projection ? projection->identity_module : NULL;
}

iree_status_t loom_link_func_contract_projection_load(
    loom_link_func_contract_projection_t* projection,
    const loom_link_module_index_symbol_t* symbol,
    const loom_link_func_contract_t** out_contract) {
  IREE_ASSERT_ARGUMENT(projection);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_contract);
  IREE_ASSERT_LT(symbol->ordinal,
                 loom_link_module_index_symbol_count(projection->index));
  IREE_ASSERT_EQ(symbol, loom_link_module_index_symbol_at(projection->index,
                                                          symbol->ordinal));
  IREE_ASSERT(iree_any_bit_set(symbol->facets.schema.interfaces,
                               LOOM_SYMBOL_INTERFACE_FUNC_LIKE));
  loom_link_func_contract_t* contract = projection->contracts[symbol->ordinal];
  if (contract == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &projection->arena, sizeof(*contract), (void**)&contract));
    memset(contract, 0, sizeof(*contract));
    loom_link_func_contract_projection_module_t* module = NULL;
    IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_get_module_state(
        projection, symbol->module_ordinal, &module));
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(
            projection->index, module->indexed_module->provider_ordinal);
    IREE_ASSERT(provider);
    if (provider->kind == LOOM_LINK_PROVIDER_BYTECODE) {
      IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_bytecode_load(
          projection, module, symbol, contract));
    } else {
      IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_materialized_load(
          projection, module, symbol, contract));
    }
    projection->contracts[symbol->ordinal] = contract;
  }
  *out_contract = contract;
  return iree_ok_status();
}
