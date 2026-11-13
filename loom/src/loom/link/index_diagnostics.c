// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/index_diagnostics.h"

#include "loom/format/bytecode/indexed_source.h"
#include "loom/ir/module.h"

typedef struct loom_link_index_source_t {
  // Embedding supplying exact source bytes.
  loom_link_diagnostic_environment_t environment;
  // Indexed identity of the materialized source module.
  const loom_link_module_index_module_t* input;
} loom_link_index_source_t;

static bool loom_link_index_resolve_exact_source(
    void* user_data, const loom_module_t* module, loom_location_id_t location,
    loom_source_range_t* out_range) {
  loom_link_index_source_t* source = user_data;
  const loom_location_entry_t* entry =
      loom_source_file_location(module, location);
  if (!entry) {
    return false;
  }
  const loom_source_entry_t snapshot = source->environment.source_entry(
      source->environment.user_data, source->input, entry->file.source_id);
  if (snapshot.source_id == LOOM_SOURCE_ID_INVALID) {
    return false;
  }
  // The entry's source identity can be sparse; resolve it without allocating a
  // table sized by that identity. Coordinate interpretation belongs to
  // source.c.
  return loom_source_resolve_snapshot(&snapshot, entry, out_range);
}

bool loom_link_index_resolve_source(
    const loom_link_module_index_t* index,
    loom_link_diagnostic_environment_t environment,
    const loom_link_module_index_symbol_t* symbol,
    loom_source_range_t* out_range) {
  if (!symbol) {
    return false;
  }
  const loom_link_module_index_module_t* input =
      loom_link_module_index_symbol_module(index, symbol);
  if (input->materialized_module) {
    const loom_module_t* module = input->materialized_module;
    const loom_op_t* op =
        module->symbols.entries[symbol->module_symbol_ordinal].defining_op;
    if (!op) {
      return false;
    }
    loom_link_index_source_t source = {environment, input};
    const loom_source_resolver_t resolver = {
        .fn = environment.source_entry ? loom_link_index_resolve_exact_source
                                       : NULL,
        .user_data = &source,
    };
    return loom_source_resolve(resolver, module, op->location, out_range);
  }
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(index, symbol);
  const loom_bytecode_module_metadata_t* metadata =
      &provider->bytecode.metadata.modules[input->provider_module_ordinal];
  return loom_bytecode_index_resolve_source(
      provider->bytecode.contents, metadata,
      metadata->symbols[symbol->module_symbol_ordinal].location, out_range);
}

iree_status_t loom_link_index_emit_diagnostic(
    const loom_link_module_index_t* index,
    loom_link_diagnostic_environment_t environment,
    const loom_link_module_index_symbol_t* symbol,
    const loom_link_module_index_symbol_t* related,
    const loom_error_def_t* error, const loom_diagnostic_param_t* parameters,
    iree_host_size_t parameter_count) {
  loom_source_range_t origin;
  loom_source_range_t related_origin;
  const bool has_origin =
      loom_link_index_resolve_source(index, environment, symbol, &origin);
  const bool has_related = loom_link_index_resolve_source(
      index, environment, related, &related_origin);
  const loom_diagnostic_related_op_t related_location = {
      .label = IREE_SV("conflicting declaration or definition"),
      .origin = &related_origin,
  };
  const loom_diagnostic_emission_t emission = {
      .origin = has_origin ? &origin : NULL,
      .error = error,
      .params = parameters,
      .param_count = parameter_count,
      .related_ops = has_related ? &related_location : NULL,
      .related_op_count = has_related ? 1 : 0,
  };
  return iree_diagnostic_emit(environment.emitter, &emission);
}
