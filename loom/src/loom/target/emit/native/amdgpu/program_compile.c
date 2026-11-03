// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_compile.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/parameter.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/target_identity.h"
#include "loom/target/registers.h"

static iree_string_view_t loom_amdgpu_program_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  if (symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static bool loom_amdgpu_program_is_direct_low_call(
    const loom_symbol_reference_occurrence_t* occurrence) {
  return occurrence->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL &&
         occurrence->user_op != NULL &&
         loom_low_func_call_isa(occurrence->user_op);
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_program_emit_invalid_helper(const loom_module_t* module,
                                        const loom_op_t* call_op,
                                        iree_string_view_t function_name,
                                        iree_string_view_t callee_name,
                                        iree_string_view_t reason,
                                        const loom_op_t* helper_op,
                                        iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(function_name),
      loom_param_string(callee_name),
      loom_param_string(reason),
  };
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("callee defined here"),
      .op = helper_op,
  }};
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = call_op,
      .error = LOOM_ERR_AMDGPU_054,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = helper_op != NULL ? related_ops : NULL,
      .related_op_count = helper_op != NULL ? IREE_ARRAYSIZE(related_ops) : 0,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_program_emit_nested_call(const loom_module_t* module,
                                     const loom_op_t* call_op,
                                     iree_string_view_t helper_name,
                                     iree_string_view_t callee_name,
                                     iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(helper_name),
      loom_param_string(callee_name),
  };
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = call_op,
      .error = LOOM_ERR_AMDGPU_057,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_program_emit_target_mismatch(
    const loom_module_t* module, loom_op_t* entry_op,
    const loom_low_resolved_target_t* entry_target,
    iree_string_view_t helper_name, loom_op_t* helper_op,
    const loom_low_resolved_target_t* helper_target,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter) {
  const loom_amdgpu_target_facts_t* entry_facts =
      loom_amdgpu_target_facts_cast(entry_target->target_facts);
  const loom_amdgpu_target_facts_t* helper_facts =
      loom_amdgpu_target_facts_cast(helper_target->target_facts);
  iree_string_view_t entry_target_key = IREE_SV("<none>");
  iree_string_view_t helper_target_key = IREE_SV("<none>");
  if (entry_facts != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
        &entry_facts->identity, arena, &entry_target_key));
  }
  if (helper_facts != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
        &helper_facts->identity, arena, &helper_target_key));
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, entry_op)),
      loom_param_string(entry_target_key),
      loom_param_string(entry_target->descriptor_set_key),
      loom_param_string(helper_name),
      loom_param_string(helper_target_key),
      loom_param_string(helper_target->descriptor_set_key),
  };
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("kernel entry defined here"),
      .op = entry_op,
  }};
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = helper_op,
      .error = LOOM_ERR_AMDGPU_056,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_amdgpu_program_discover_helpers(
    const loom_module_t* module, loom_op_t* entry_function_op,
    const loom_symbol_reference_table_t* references,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* scratch_arena,
    bool* out_accepted, uint8_t** out_helper_symbols,
    iree_host_size_t* out_helper_count) {
  *out_accepted = false;
  *out_helper_symbols = NULL;
  *out_helper_count = 0;
  IREE_ASSERT_EQ(references->module, module);
  const loom_func_like_t entry_function =
      loom_func_like_cast(module, entry_function_op);
  const loom_symbol_ref_t entry_ref = loom_func_like_callee(entry_function);
  IREE_ASSERT_EQ(entry_ref.module_id, 0u);
  IREE_ASSERT_LT(entry_ref.symbol_id, module->symbols.count);

  uint8_t* helper_symbols = NULL;
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, module->symbols.count, sizeof(*helper_symbols),
        (void**)&helper_symbols));
    memset(helper_symbols, 0, module->symbols.count * sizeof(*helper_symbols));
  }
  iree_host_size_t helper_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id =
      loom_symbol_reference_table_symbol(references, entry_ref.symbol_id)
          .first_outgoing_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(references, occurrence_id);
    occurrence_id = occurrence->next_outgoing_occurrence_id;
    if (!loom_amdgpu_program_is_direct_low_call(occurrence)) {
      continue;
    }
    IREE_ASSERT_LT(occurrence->target_symbol_id, module->symbols.count);
    const loom_symbol_t* helper_symbol =
        &module->symbols.entries[occurrence->target_symbol_id];
    loom_op_t* helper_op = helper_symbol->defining_op;
    iree_string_view_t reason = iree_string_view_empty();
    if (helper_op == NULL) {
      reason = IREE_SV("missing-definition");
    } else if (!loom_low_func_def_isa(helper_op)) {
      reason = IREE_SV("not-low-function-definition");
    } else if (!loom_func_like_is_module_internal(
                   loom_func_like_cast(module, helper_op))) {
      reason = IREE_SV("not-module-internal");
    }
    if (!iree_string_view_is_empty(reason)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_emit_invalid_helper(
          module, occurrence->user_op,
          loom_low_diagnostic_function_name(module, entry_function_op),
          loom_amdgpu_program_symbol_name(module, occurrence->target_symbol_id),
          reason, helper_op, emitter));
      return iree_ok_status();
    }
    if (!helper_symbols[occurrence->target_symbol_id]) {
      helper_symbols[occurrence->target_symbol_id] = 1;
      ++helper_count;
    }
  }

  for (loom_symbol_id_t helper_id = 0; helper_id < module->symbols.count;
       ++helper_id) {
    if (!helper_symbols[helper_id]) {
      continue;
    }
    occurrence_id = loom_symbol_reference_table_symbol(references, helper_id)
                        .first_outgoing_occurrence_id;
    while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* occurrence =
          loom_symbol_reference_table_occurrence(references, occurrence_id);
      occurrence_id = occurrence->next_outgoing_occurrence_id;
      if (!loom_amdgpu_program_is_direct_low_call(occurrence)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_emit_nested_call(
          module, occurrence->user_op,
          loom_amdgpu_program_symbol_name(module, helper_id),
          loom_amdgpu_program_symbol_name(module, occurrence->target_symbol_id),
          emitter));
      return iree_ok_status();
    }
  }
  *out_helper_symbols = helper_symbols;
  *out_helper_count = helper_count;
  *out_accepted = true;
  return iree_ok_status();
}

static bool loom_amdgpu_program_targets_match(
    const loom_low_resolved_target_t* entry,
    const loom_low_resolved_target_t* helper) {
  const loom_amdgpu_target_facts_t* entry_facts =
      loom_amdgpu_target_facts_cast(entry->target_facts);
  const loom_amdgpu_target_facts_t* helper_facts =
      loom_amdgpu_target_facts_cast(helper->target_facts);
  return entry_facts != NULL && helper_facts != NULL &&
         entry->descriptor_set->stable_id ==
             helper->descriptor_set->stable_id &&
         loom_amdgpu_target_identity_equal(&entry_facts->identity,
                                           &helper_facts->identity);
}

static void loom_amdgpu_program_require_callable_registers(
    const loom_amdgpu_program_callable_t* callable,
    loom_amdgpu_program_resource_builder_t* resources) {
  const uint32_t sgpr_count =
      LOOM_AMDGPU_CALL_ABI_SGPR_BASE +
      iree_max(callable->argument_requirements.sgpr_count,
               callable->result_requirements.sgpr_count);
  const uint32_t vgpr_count =
      LOOM_AMDGPU_CALL_ABI_VGPR_BASE +
      iree_max(callable->argument_requirements.vgpr_count,
               callable->result_requirements.vgpr_count);
  loom_amdgpu_program_resource_builder_require_registers(
      sgpr_count, vgpr_count, /*agpr_count=*/0, resources);
}

// Retained calls can overwrite loader-owned coordinates, so entry ABI values
// become one-shot transport sources instead of lifetime constraints. The
// ordinary no-call path keeps the fixed bindings and its existing machine code.
static iree_status_t loom_amdgpu_program_prepare_retained_entry_options(
    const loom_module_t* module,
    const loom_amdgpu_function_compile_options_t* source_options,
    iree_arena_allocator_t* arena,
    loom_amdgpu_function_compile_options_t* out_options) {
  *out_options = *source_options;
  if (source_options->allocation_fixed_value_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t entry_value_count = 0;
  if (!iree_host_size_checked_add(source_options->allocation_entry_value_count,
                                  source_options->allocation_fixed_value_count,
                                  &entry_value_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU retained entry value count overflowed");
  }
  loom_low_allocation_entry_value_t* entry_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entry_value_count, sizeof(*entry_values), (void**)&entry_values));
  if (source_options->allocation_entry_value_count != 0) {
    memcpy(
        entry_values, source_options->allocation_entry_values,
        source_options->allocation_entry_value_count * sizeof(*entry_values));
  }
  loom_low_allocation_entry_value_t* converted_values =
      entry_values + source_options->allocation_entry_value_count;
  for (iree_host_size_t i = 0; i < source_options->allocation_fixed_value_count;
       ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value =
        &source_options->allocation_fixed_values[i];
    const loom_type_t type =
        loom_module_value_type(module, fixed_value->value_id);
    IREE_ASSERT(loom_low_type_is_register(type));
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(type),
                   fixed_value->location_count);
    converted_values[i] = (loom_low_allocation_entry_value_t){
        .value_id = fixed_value->value_id,
        .location_kind = fixed_value->location_kind,
        .location_base = fixed_value->location_base,
    };
  }
  out_options->allocation_fixed_values = NULL;
  out_options->allocation_fixed_value_count = 0;
  out_options->allocation_entry_values = entry_values;
  out_options->allocation_entry_value_count = entry_value_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_compile_retained(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_amdgpu_program_compile_options_t* options,
    const uint8_t* helper_symbols, iree_host_size_t helper_count,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    bool* out_compiled, loom_amdgpu_program_compilation_t* out_compilation) {
  const loom_low_descriptor_set_t* descriptor_set =
      options->entry.resolved_target->descriptor_set;
  loom_amdgpu_program_callable_t* helpers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, helper_count, sizeof(*helpers), (void**)&helpers));
  memset(helpers, 0, helper_count * sizeof(*helpers));
  out_compilation->helpers = helpers;
  out_compilation->helper_count = helper_count;

  loom_amdgpu_program_call_table_t call_table = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_call_table_initialize(
      descriptor_set, module->symbols.count, arena, scratch_arena,
      &call_table));
  out_compilation->call_table = call_table;
  loom_amdgpu_program_resource_builder_t resources = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_resource_builder_initialize(
      descriptor_set->reg_class_count, arena, &resources));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, arena);
  const loom_low_allocation_reserved_range_t call_control_reservation = {
      .register_class = IREE_SV("amdgpu.sgpr"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE,
      .location_count = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT,
  };

  iree_host_size_t helper_index = 0;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    if (!helper_symbols[symbol_id]) {
      continue;
    }
    loom_amdgpu_program_callable_t* callable = &helpers[helper_index++];
    callable->helper_ordinal = helper_index - 1;
    callable->function = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    loom_op_t* helper_op = module->symbols.entries[symbol_id].defining_op;
    const loom_target_function_version_t* version =
        options->function_versions != NULL
            ? loom_target_function_version_snapshot_at(
                  options->function_versions, symbol_id)
            : NULL;
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, &symbol_facts, helper_op,
        version != NULL ? version->function_target_facts : NULL,
        options->entry.descriptor_registry, options->entry.emitter,
        &callable->target));
    if (callable->target.descriptor_set == NULL) {
      return iree_ok_status();
    }
    if (!loom_amdgpu_program_targets_match(options->entry.resolved_target,
                                           &callable->target)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_emit_target_mismatch(
          module, entry_function_op, options->entry.resolved_target,
          loom_amdgpu_program_symbol_name(module, symbol_id), helper_op,
          &callable->target, arena, options->entry.emitter));
      return iree_ok_status();
    }
    bool abi_prepared = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_prepare_abi(
        module, helper_op, descriptor_set, options->entry.emitter, arena,
        &abi_prepared, callable));
    if (!abi_prepared) {
      return iree_ok_status();
    }
    const loom_amdgpu_function_compile_options_t helper_options = {
        .descriptor_registry = options->entry.descriptor_registry,
        .resolved_target = &callable->target,
        .memory_accesses = version != NULL ? version->memory_accesses : NULL,
        .loop_execution = version != NULL ? version->loop_execution : NULL,
        .allocation_entry_locations = callable->call_contract.arguments,
        .allocation_entry_location_count =
            callable->call_contract.argument_count,
        .allocation_exit_locations = callable->call_contract.results,
        .allocation_exit_location_count = callable->call_contract.result_count,
        .allocation_reserved_ranges = &call_control_reservation,
        .allocation_reserved_range_count = 1,
        .callable = true,
        .emitter = options->entry.emitter,
    };
    bool helper_compiled = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_function_compile(module, helper_op, &helper_options, arena,
                                     &helper_compiled, &callable->compilation));
    if (!helper_compiled) {
      return iree_ok_status();
    }
    callable->call_state.counter_mask =
        callable->compilation.plan.packet_plan.wait_plan.produced_counter_mask;
    callable->call_state.fixed_hazard_mask =
        callable->compilation.plan.packet_plan.wait_plan.fixed_hazard_mask;
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_prepare_clobbers(
        descriptor_set, arena, scratch_arena, callable));
    loom_amdgpu_program_call_table_publish(&call_table, callable);
    loom_amdgpu_storage_layout_t program_storage = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_resource_builder_append_function(
        &callable->compilation.frame, &callable->compilation.plan, arena,
        &resources, &program_storage));
    callable->compilation.plan.storage_layout = program_storage;
    loom_amdgpu_program_require_callable_registers(callable, &resources);
  }
  IREE_ASSERT_EQ(helper_index, helper_count);

  iree_host_size_t entry_reservation_count = 0;
  if (!iree_host_size_checked_add(
          options->entry.allocation_reserved_range_count, 1,
          &entry_reservation_count)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU entry allocation reservation count overflowed");
  }
  loom_low_allocation_reserved_range_t* entry_reservations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, entry_reservation_count, sizeof(*entry_reservations),
      (void**)&entry_reservations));
  if (options->entry.allocation_reserved_range_count != 0) {
    memcpy(entry_reservations, options->entry.allocation_reserved_ranges,
           options->entry.allocation_reserved_range_count *
               sizeof(*entry_reservations));
  }
  entry_reservations[entry_reservation_count - 1] = call_control_reservation;
  loom_amdgpu_function_compile_options_t entry_options = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_prepare_retained_entry_options(
      module, &options->entry, scratch_arena, &entry_options));
  entry_options.call_contracts =
      loom_amdgpu_program_call_table_provider(&call_table);
  entry_options.call_states =
      loom_amdgpu_program_call_table_state_provider(&call_table);
  entry_options.allocation_reserved_ranges = entry_reservations;
  entry_options.allocation_reserved_range_count = entry_reservation_count;
  bool entry_compiled = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_function_compile(
      module, entry_function_op, &entry_options, arena, &entry_compiled,
      &out_compilation->entry));
  if (!entry_compiled) {
    return iree_ok_status();
  }
  loom_amdgpu_storage_layout_t entry_storage = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_resource_builder_append_function(
      &out_compilation->entry.frame, &out_compilation->entry.plan, arena,
      &resources, &entry_storage));
  out_compilation->entry.plan.storage_layout = entry_storage;
  loom_amdgpu_program_resource_builder_require_registers(
      LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE +
          LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT,
      /*vgpr_count=*/0, /*agpr_count=*/0, &resources);
  loom_amdgpu_program_resource_builder_finish(&resources,
                                              &out_compilation->resources);
  out_compilation->has_retained_calls = true;
  *out_compiled = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_compile(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_amdgpu_program_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_amdgpu_program_compilation_t* out_compilation) {
  *out_compiled = false;
  *out_compilation = (loom_amdgpu_program_compilation_t){0};
  IREE_ASSERT_EQ(options->references->module, module);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  bool closure_accepted = false;
  uint8_t* helper_symbols = NULL;
  iree_host_size_t helper_count = 0;
  iree_status_t status = loom_amdgpu_program_discover_helpers(
      module, entry_function_op, options->references, options->entry.emitter,
      &scratch_arena, &closure_accepted, &helper_symbols, &helper_count);
  if (iree_status_is_ok(status) && closure_accepted && helper_count == 0) {
    status = loom_amdgpu_function_compile(module, entry_function_op,
                                          &options->entry, arena, out_compiled,
                                          &out_compilation->entry);
  } else if (iree_status_is_ok(status) && closure_accepted) {
    status = loom_amdgpu_program_compile_retained(
        module, entry_function_op, options, helper_symbols, helper_count, arena,
        &scratch_arena, out_compiled, out_compilation);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
