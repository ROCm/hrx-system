// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/program_source.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/read_only_data.h"
#include "loom/error/parameter.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

static iree_string_view_t loom_aie2p_program_source_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  return loom_string_table_get(&module->strings,
                               module->symbols.entries[symbol_id].name_id);
}

iree_status_t loom_aie2p_program_source_table_initialize(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_target_function_version_snapshot_t* function_versions,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_program_source_table_t* out_table) {
  *out_table = (loom_aie2p_program_source_table_t){
      .module = module,
      .symbol_facts = symbol_facts,
      .function_versions = function_versions,
      .descriptor_registry = descriptor_registry,
      .diagnostic_emitter = diagnostic_emitter,
      .arena = arena,
  };
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*out_table->functions_by_symbol),
        (void**)&out_table->functions_by_symbol));
    memset(out_table->functions_by_symbol, 0,
           module->symbols.count * sizeof(*out_table->functions_by_symbol));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_source_table_resolve(
    loom_aie2p_program_source_table_t* table, loom_symbol_ref_t symbol,
    bool* out_resolved, loom_aie2p_program_source_function_t** out_function) {
  *out_resolved = false;
  *out_function = NULL;
  IREE_ASSERT_EQ(symbol.module_id, 0u);
  IREE_ASSERT_LT(symbol.symbol_id, table->module->symbols.count);
  loom_aie2p_program_source_function_t* function =
      table->functions_by_symbol[symbol.symbol_id];
  if (function != NULL) {
    *out_function = function;
    *out_resolved = true;
    return iree_ok_status();
  }

  const loom_op_t* function_op =
      table->module->symbols.entries[symbol.symbol_id].defining_op;
  IREE_ASSERT(loom_low_func_def_isa(function_op));
  const loom_target_function_version_t* version =
      table->function_versions != NULL
          ? loom_target_function_version_snapshot_at(table->function_versions,
                                                     symbol.symbol_id)
          : NULL;
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      table->module, table->symbol_facts, function_op,
      version != NULL ? version->function_target_facts : NULL,
      table->descriptor_registry, table->diagnostic_emitter, &target));
  if (target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(target.descriptor_set->stable_id,
                 AIE2P_CORE_DESCRIPTOR_SET_ID);

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(table->arena, sizeof(*function), (void**)&function));
  *function = (loom_aie2p_program_source_function_t){
      .symbol = symbol,
      .function_op = function_op,
      .target = target,
      .memory_accesses = version != NULL ? version->memory_accesses : NULL,
      .loop_execution = version != NULL ? version->loop_execution : NULL,
  };
  IREE_RETURN_IF_ERROR(loom_low_function_requirements_build(
      table->module, target.descriptor_set, loom_low_func_def_body(function_op),
      table->arena, &function->requirements));
  function->program_data = &function->requirements.read_only_data;
  table->functions_by_symbol[symbol.symbol_id] = function;
  *out_function = function;
  *out_resolved = true;
  return iree_ok_status();
}

const loom_aie2p_program_source_function_t*
loom_aie2p_program_source_table_lookup(
    const loom_aie2p_program_source_table_t* table, loom_symbol_ref_t symbol) {
  IREE_ASSERT_EQ(symbol.module_id, 0u);
  IREE_ASSERT_LT(symbol.symbol_id, table->module->symbols.count);
  const loom_aie2p_program_source_function_t* function =
      table->functions_by_symbol[symbol.symbol_id];
  IREE_ASSERT_NE(function, NULL);
  return function;
}

// The shared graph engine owns traversal and cycle detection. This adapter
// supplies the source inventory once and stops at already admitted subtrees.
typedef struct loom_aie2p_program_source_graph_t {
  // Lazy source index owning immutable requirements and completed closures.
  loom_aie2p_program_source_table_t* table;
  // Target-binding rejection encountered while resolving a reached function.
  bool rejected;
} loom_aie2p_program_source_graph_t;

static iree_status_t loom_aie2p_program_source_visit_callees(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  loom_aie2p_program_source_graph_t* graph = user_data;
  loom_aie2p_program_source_function_t* source = NULL;
  bool resolved = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_source_table_resolve(
      graph->table, (loom_symbol_ref_t){.symbol_id = (loom_symbol_id_t)node},
      &resolved, &source));
  if (!resolved) {
    graph->rejected = true;
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (!source->helpers_admitted) {
    const loom_low_direct_callee_requirement_list_t callees =
        source->requirements.direct_callees;
    for (iree_host_size_t i = 0; i < callees.count && iree_status_is_ok(status);
         ++i) {
      status =
          successor.fn(successor.user_data, callees.values[i].symbol.symbol_id);
    }
  }
  return status;
}

static iree_status_t loom_aie2p_program_source_append_helper(
    loom_symbol_ref_t symbol, uint8_t* membership,
    iree_arena_allocator_t* arena, loom_symbol_ref_t** helpers,
    iree_host_size_t* count, iree_host_size_t* capacity) {
  if (membership[symbol.symbol_id]) {
    return iree_ok_status();
  }
  if (*count == *capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, *count, iree_max(*count + 1, 4u),
                              sizeof(**helpers), capacity, (void**)helpers));
  }
  (*helpers)[(*count)++] = symbol;
  membership[symbol.symbol_id] = 1;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_source_build_closure(
    loom_aie2p_program_source_table_t* table,
    loom_aie2p_program_source_function_t* source, uint8_t* membership,
    bool* out_admitted) {
  loom_symbol_ref_t* helpers = NULL;
  iree_host_size_t helper_count = 0;
  iree_host_size_t helper_capacity = 0;
  const loom_low_direct_callee_requirement_list_t calls =
      source->requirements.direct_callees;
  iree_status_t status = iree_ok_status();
  bool rejected = false;
  for (iree_host_size_t i = 0;
       i < calls.count && iree_status_is_ok(status) && !rejected; ++i) {
    const loom_aie2p_program_source_function_t* helper =
        loom_aie2p_program_source_table_lookup(table, calls.values[i].symbol);
    if (helper->requirements.resource_count != 0 ||
        helper->requirements.storage_layout.record_count != 0) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(loom_aie2p_program_source_symbol_name(
              table->module, helper->symbol.symbol_id)),
      };
      const loom_diagnostic_emission_t emission = {
          .module = table->module,
          .op = helper->function_op,
          .error = LOOM_ERR_XDNA_065,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      status = iree_diagnostic_emit(table->diagnostic_emitter, &emission);
      rejected = true;
    }
    for (iree_host_size_t j = 0;
         j < helper->helpers.count && iree_status_is_ok(status) && !rejected;
         ++j) {
      status = loom_aie2p_program_source_append_helper(
          helper->helpers.values[j], membership, table->arena, &helpers,
          &helper_count, &helper_capacity);
    }
    if (iree_status_is_ok(status) && !rejected) {
      status = loom_aie2p_program_source_append_helper(
          helper->symbol, membership, table->arena, &helpers, &helper_count,
          &helper_capacity);
    }
  }
  loom_low_read_only_data_builder_t data_builder = {0};
  bool has_helper_data = false;
  for (iree_host_size_t i = 0;
       i < helper_count && iree_status_is_ok(status) && !rejected; ++i) {
    const loom_aie2p_program_source_function_t* helper =
        loom_aie2p_program_source_table_lookup(table, helpers[i]);
    const loom_low_read_only_data_set_t* data =
        &helper->requirements.read_only_data;
    if (data->count != 0) {
      if (!has_helper_data) {
        status = loom_low_read_only_data_builder_extend(
            &data_builder, &source->requirements.read_only_data, table->arena);
        has_helper_data = true;
      }
      if (iree_status_is_ok(status)) {
        status = loom_low_read_only_data_builder_extend(&data_builder, data,
                                                        table->arena);
      }
    }
  }
  if (iree_status_is_ok(status) && !rejected && has_helper_data) {
    loom_low_read_only_data_set_t* data = NULL;
    status = iree_arena_allocate(table->arena, sizeof(*data), (void**)&data);
    if (iree_status_is_ok(status)) {
      *data = loom_low_read_only_data_builder_finish(&data_builder);
      source->program_data = data;
    }
  }
  // Clear only visited members; many small closures do not repeatedly clear
  // the whole source symbol domain.
  for (iree_host_size_t i = 0; i < helper_count; ++i) {
    membership[helpers[i].symbol_id] = 0;
  }
  if (iree_status_is_ok(status) && !rejected) {
    source->helpers = loom_make_symbol_ref_array(helpers, helper_count);
    source->helpers_admitted = true;
    *out_admitted = true;
  }
  return status;
}

static iree_status_t loom_aie2p_program_source_admit_helpers_impl(
    loom_aie2p_program_source_table_t* table,
    loom_aie2p_program_source_function_t* entry,
    iree_arena_allocator_t* scratch_arena, bool* out_admitted) {
  loom_aie2p_program_source_graph_t source_graph = {.table = table};
  const loom_scc_graph_t graph = {
      .node_count = table->module->symbols.count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_aie2p_program_source_visit_callees, &source_graph),
  };
  const iree_host_size_t root = entry->symbol.symbol_id;
  const loom_scc_options_t options = {.root_nodes = &root, .root_count = 1};
  loom_scc_list_t sccs = {0};
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, &options, scratch_arena, &sccs));
  if (source_graph.rejected) {
    return iree_ok_status();
  }
  uint8_t* membership = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, graph.node_count,
                                sizeof(*membership), (void**)&membership));
  memset(membership, 0, graph.node_count * sizeof(*membership));
  bool admitted = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < sccs.count && admitted && iree_status_is_ok(status); ++i) {
    const loom_scc_t* scc = &sccs.values[i];
    loom_aie2p_program_source_function_t* source =
        table->functions_by_symbol[scc->nodes[0]];
    if (scc->is_cycle) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(loom_aie2p_program_source_symbol_name(
              table->module, source->symbol.symbol_id)),
      };
      const loom_diagnostic_emission_t emission = {
          .module = table->module,
          .op = source->function_op,
          .error = LOOM_ERR_XDNA_063,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      status = iree_diagnostic_emit(table->diagnostic_emitter, &emission);
      admitted = false;
    } else if (!source->helpers_admitted) {
      admitted = false;
      status = loom_aie2p_program_source_build_closure(table, source,
                                                       membership, &admitted);
    }
  }
  *out_admitted = admitted && iree_status_is_ok(status);
  return status;
}

iree_status_t loom_aie2p_program_source_admit_helpers(
    loom_aie2p_program_source_table_t* table,
    loom_aie2p_program_source_function_t* entry, bool* out_admitted) {
  *out_admitted = entry->helpers_admitted;
  if (*out_admitted) {
    return iree_ok_status();
  }
  if (entry->requirements.direct_callees.count == 0) {
    entry->helpers_admitted = true;
    *out_admitted = true;
    return iree_ok_status();
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(table->arena->block_pool, &scratch_arena);
  iree_status_t status = loom_aie2p_program_source_admit_helpers_impl(
      table, entry, &scratch_arena, out_admitted);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
