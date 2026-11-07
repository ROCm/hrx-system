// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/program_source.h"

#include <string.h>

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

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_aie2p_program_source_emit_nested_call(
    const loom_aie2p_program_source_table_t* table,
    const loom_aie2p_program_source_function_t* entry,
    const loom_aie2p_program_source_function_t* helper,
    const loom_low_direct_callee_requirement_t* nested_call) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_aie2p_program_source_symbol_name(
          table->module, helper->symbol.symbol_id)),
      loom_param_string(
          loom_low_diagnostic_symbol_name(table->module, nested_call->symbol)),
  };
  const loom_diagnostic_related_op_t related_ops[] = {
      {
          .label = IREE_SV("entry defined here"),
          .op = entry->function_op,
      },
      {
          .label = IREE_SV("helper defined here"),
          .op = helper->function_op,
      },
  };
  const loom_diagnostic_emission_t emission = {
      .module = table->module,
      .op = nested_call->first_call_op,
      .error = LOOM_ERR_XDNA_063,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(table->diagnostic_emitter, &emission);
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

iree_status_t loom_aie2p_program_source_admit_direct_helpers(
    loom_aie2p_program_source_table_t* table,
    loom_aie2p_program_source_function_t* entry, bool* out_admitted) {
  *out_admitted = false;
  if (entry->direct_helpers_admitted) {
    *out_admitted = true;
    return iree_ok_status();
  }
  loom_low_read_only_data_builder_t data_builder = {0};
  bool has_helper_data = false;
  for (iree_host_size_t i = 0; i < entry->requirements.direct_callees.count;
       ++i) {
    const loom_low_direct_callee_requirement_t* call =
        &entry->requirements.direct_callees.values[i];
    const loom_symbol_ref_t helper_ref = call->symbol;
    bool helper_resolved = false;
    loom_aie2p_program_source_function_t* helper = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_program_source_table_resolve(
        table, helper_ref, &helper_resolved, &helper));
    if (!helper_resolved) {
      return iree_ok_status();
    }
    if (helper->requirements.direct_callees.count != 0) {
      IREE_RETURN_IF_ERROR(loom_aie2p_program_source_emit_nested_call(
          table, entry, helper,
          &helper->requirements.direct_callees.values[0]));
      return iree_ok_status();
    }
    if (helper->requirements.read_only_data.count != 0) {
      if (!has_helper_data) {
        IREE_RETURN_IF_ERROR(loom_low_read_only_data_builder_extend(
            &data_builder, entry->program_data, table->arena));
        has_helper_data = true;
      }
      IREE_RETURN_IF_ERROR(loom_low_read_only_data_builder_extend(
          &data_builder, &helper->requirements.read_only_data, table->arena));
    }
  }
  if (has_helper_data) {
    loom_low_read_only_data_set_t* program_data = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        table->arena, sizeof(*program_data), (void**)&program_data));
    *program_data = loom_low_read_only_data_builder_finish(&data_builder);
    entry->program_data = program_data;
  }
  entry->direct_helpers_admitted = true;
  *out_admitted = true;
  return iree_ok_status();
}
