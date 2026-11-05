// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_artifact.h"

#include <stdlib.h>
#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/parameter.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/target_identity.h"

static iree_string_view_t loom_amdgpu_program_artifact_symbol_name(
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

static bool loom_amdgpu_program_artifact_is_direct_low_call(
    const loom_symbol_reference_occurrence_t* occurrence) {
  return occurrence->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL &&
         occurrence->user_op != NULL &&
         loom_low_func_call_isa(occurrence->user_op);
}

static int loom_amdgpu_program_artifact_compare_symbol_ids(
    const void* lhs_ptr, const void* rhs_ptr) {
  const loom_symbol_id_t lhs = *(const loom_symbol_id_t*)lhs_ptr;
  const loom_symbol_id_t rhs = *(const loom_symbol_id_t*)rhs_ptr;
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_program_artifact_emit_invalid_helper(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_op_t* root_op, loom_symbol_id_t helper_symbol_id,
    iree_string_view_t reason, const loom_op_t* helper_op,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, root_op)),
      loom_param_string(
          loom_amdgpu_program_artifact_symbol_name(module, helper_symbol_id)),
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
loom_amdgpu_program_artifact_emit_nested_call(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_id_t helper_symbol_id, loom_symbol_id_t callee_symbol_id,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_amdgpu_program_artifact_symbol_name(module, helper_symbol_id)),
      loom_param_string(
          loom_amdgpu_program_artifact_symbol_name(module, callee_symbol_id)),
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
loom_amdgpu_program_artifact_emit_target_mismatch(
    const loom_module_t* module, const loom_op_t* root_op,
    const loom_low_resolved_target_t* root_target,
    loom_symbol_id_t helper_symbol_id, loom_op_t* helper_op,
    const loom_low_resolved_target_t* helper_target,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter) {
  const loom_amdgpu_target_facts_t* root_facts =
      loom_amdgpu_target_facts_cast(root_target->target_facts);
  const loom_amdgpu_target_facts_t* helper_facts =
      loom_amdgpu_target_facts_cast(helper_target->target_facts);
  iree_string_view_t root_target_key = IREE_SV("<none>");
  iree_string_view_t helper_target_key = IREE_SV("<none>");
  if (root_facts != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
        &root_facts->identity, arena, &root_target_key));
  }
  if (helper_facts != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
        &helper_facts->identity, arena, &helper_target_key));
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, root_op)),
      loom_param_string(root_target_key),
      loom_param_string(root_target->descriptor_set_key),
      loom_param_string(
          loom_amdgpu_program_artifact_symbol_name(module, helper_symbol_id)),
      loom_param_string(helper_target_key),
      loom_param_string(helper_target->descriptor_set_key),
  };
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("kernel entry defined here"),
      .op = root_op,
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

static bool loom_amdgpu_program_artifact_targets_match(
    const loom_low_resolved_target_t* root,
    const loom_low_resolved_target_t* helper) {
  const loom_amdgpu_target_facts_t* root_facts =
      loom_amdgpu_target_facts_cast(root->target_facts);
  const loom_amdgpu_target_facts_t* helper_facts =
      loom_amdgpu_target_facts_cast(helper->target_facts);
  return root_facts != NULL && helper_facts != NULL &&
         root->descriptor_set->stable_id == helper->descriptor_set->stable_id &&
         loom_amdgpu_target_identity_equal(&root_facts->identity,
                                           &helper_facts->identity);
}

static loom_amdgpu_program_call_table_t*
loom_amdgpu_program_artifact_find_call_table(
    loom_amdgpu_program_artifact_t* artifact,
    const loom_low_descriptor_set_t* descriptor_set) {
  for (iree_host_size_t i = 0; i < artifact->call_table_count; ++i) {
    if (artifact->call_tables[i].descriptor_set == descriptor_set) {
      return &artifact->call_tables[i];
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_program_artifact_discover_roots(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_root_input_t* root_inputs,
    iree_host_size_t root_count,
    const loom_amdgpu_program_artifact_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    uint8_t* helper_symbols, const loom_op_t** first_call_ops,
    const loom_op_t** first_root_ops, loom_amdgpu_program_artifact_t* artifact,
    bool* out_accepted) {
  *out_accepted = false;
  const iree_host_size_t symbol_count = module->symbols.count;
  iree_host_size_t* root_marks = NULL;
  loom_symbol_id_t* root_helper_buffer = NULL;
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, symbol_count, sizeof(*root_marks), (void**)&root_marks));
    memset(root_marks, 0, symbol_count * sizeof(*root_marks));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, symbol_count, sizeof(*root_helper_buffer),
        (void**)&root_helper_buffer));
  }

  for (iree_host_size_t root_index = 0; root_index < root_count; ++root_index) {
    const loom_amdgpu_program_artifact_root_input_t* input =
        &root_inputs[root_index];
    loom_amdgpu_program_artifact_root_t* root = &artifact->roots[root_index];
    root->function_op = input->function_op;
    root->target = input->target;
    const loom_func_like_t function =
        loom_func_like_cast(module, input->function_op);
    const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
    IREE_ASSERT_EQ(function_ref.module_id, 0u);
    IREE_ASSERT_LT(function_ref.symbol_id, symbol_count);

    iree_host_size_t helper_count = 0;
    loom_symbol_reference_occurrence_id_t occurrence_id =
        loom_symbol_reference_table_symbol(options->references,
                                           function_ref.symbol_id)
            .first_outgoing_occurrence_id;
    while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* occurrence =
          loom_symbol_reference_table_occurrence(options->references,
                                                 occurrence_id);
      occurrence_id = occurrence->next_outgoing_occurrence_id;
      if (!loom_amdgpu_program_artifact_is_direct_low_call(occurrence)) {
        continue;
      }
      IREE_ASSERT_LT(occurrence->target_symbol_id, symbol_count);
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
        IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_emit_invalid_helper(
            module, occurrence->user_op, input->function_op,
            occurrence->target_symbol_id, reason, helper_op, options->emitter));
        return iree_ok_status();
      }

      const iree_host_size_t root_mark = root_index + 1;
      if (root_marks[occurrence->target_symbol_id] == root_mark) {
        continue;
      }
      root_marks[occurrence->target_symbol_id] = root_mark;
      root_helper_buffer[helper_count++] = occurrence->target_symbol_id;
      if (!helper_symbols[occurrence->target_symbol_id]) {
        helper_symbols[occurrence->target_symbol_id] = 1;
        ++artifact->helper_count;
        first_call_ops[occurrence->target_symbol_id] = occurrence->user_op;
        first_root_ops[occurrence->target_symbol_id] = input->function_op;
      }
    }

    if (helper_count != 0) {
      qsort(root_helper_buffer, helper_count, sizeof(*root_helper_buffer),
            loom_amdgpu_program_artifact_compare_symbol_ids);
      loom_symbol_id_t* retained_helpers = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, helper_count, sizeof(*retained_helpers),
          (void**)&retained_helpers));
      memcpy(retained_helpers, root_helper_buffer,
             helper_count * sizeof(*retained_helpers));
      root->helper_symbols = retained_helpers;
      root->helper_count = helper_count;
    }
  }

  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_artifact_reject_nested_calls(
    const loom_module_t* module,
    const loom_amdgpu_program_artifact_options_t* options,
    const uint8_t* helper_symbols, bool* out_accepted) {
  *out_accepted = false;
  for (loom_symbol_id_t helper_id = 0; helper_id < module->symbols.count;
       ++helper_id) {
    if (!helper_symbols[helper_id]) {
      continue;
    }
    loom_symbol_reference_occurrence_id_t occurrence_id =
        loom_symbol_reference_table_symbol(options->references, helper_id)
            .first_outgoing_occurrence_id;
    while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* occurrence =
          loom_symbol_reference_table_occurrence(options->references,
                                                 occurrence_id);
      occurrence_id = occurrence->next_outgoing_occurrence_id;
      if (!loom_amdgpu_program_artifact_is_direct_low_call(occurrence)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_emit_nested_call(
          module, occurrence->user_op, helper_id, occurrence->target_symbol_id,
          options->emitter));
      return iree_ok_status();
    }
  }
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_artifact_initialize_call_tables(
    loom_module_t* module, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_program_artifact_t* artifact) {
  if (artifact->helper_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, artifact->root_count, sizeof(*artifact->call_tables),
      (void**)&artifact->call_tables));
  memset(artifact->call_tables, 0,
         artifact->root_count * sizeof(*artifact->call_tables));
  for (iree_host_size_t i = 0; i < artifact->root_count; ++i) {
    loom_amdgpu_program_artifact_root_t* root = &artifact->roots[i];
    if (root->helper_count == 0) {
      continue;
    }
    root->call_table = loom_amdgpu_program_artifact_find_call_table(
        artifact, root->target->descriptor_set);
    if (root->call_table != NULL) {
      continue;
    }
    root->call_table = &artifact->call_tables[artifact->call_table_count++];
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_call_table_initialize(
        root->target->descriptor_set, module->symbols.count, arena,
        scratch_arena, root->call_table));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_artifact_resolve_helpers(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_options_t* options,
    const uint8_t* helper_symbols, iree_arena_allocator_t* arena,
    loom_amdgpu_program_callable_t** callables_by_symbol,
    loom_amdgpu_program_artifact_t* artifact, bool* out_accepted) {
  *out_accepted = false;
  if (artifact->helper_count == 0) {
    *out_accepted = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, artifact->helper_count,
                                                 sizeof(*artifact->helpers),
                                                 (void**)&artifact->helpers));
  memset(artifact->helpers, 0,
         artifact->helper_count * sizeof(*artifact->helpers));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, arena);

  iree_host_size_t helper_index = 0;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    if (!helper_symbols[symbol_id]) {
      continue;
    }
    loom_amdgpu_program_callable_t* callable = &artifact->helpers[helper_index];
    callable->function = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    callable->helper_ordinal = helper_index++;
    callables_by_symbol[symbol_id] = callable;
    loom_op_t* helper_op = module->symbols.entries[symbol_id].defining_op;
    const loom_target_function_version_t* version =
        options->function_versions != NULL
            ? loom_target_function_version_snapshot_at(
                  options->function_versions, symbol_id)
            : NULL;
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, &symbol_facts, helper_op,
        version != NULL ? version->function_target_facts : NULL,
        options->descriptor_registry, options->emitter, &callable->target));
    if (callable->target.descriptor_set == NULL) {
      return iree_ok_status();
    }
  }
  IREE_ASSERT_EQ(helper_index, artifact->helper_count);
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_artifact_validate_root_targets(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_options_t* options,
    loom_amdgpu_program_callable_t* const* callables_by_symbol,
    iree_arena_allocator_t* scratch_arena,
    const loom_amdgpu_program_artifact_t* artifact, bool* out_accepted) {
  *out_accepted = false;
  for (iree_host_size_t i = 0; i < artifact->root_count; ++i) {
    const loom_amdgpu_program_artifact_root_t* root = &artifact->roots[i];
    for (iree_host_size_t j = 0; j < root->helper_count; ++j) {
      const loom_symbol_id_t helper_symbol_id = root->helper_symbols[j];
      loom_amdgpu_program_callable_t* callable =
          callables_by_symbol[helper_symbol_id];
      IREE_ASSERT_NE(callable, NULL);
      if (loom_amdgpu_program_artifact_targets_match(root->target,
                                                     &callable->target)) {
        continue;
      }
      loom_op_t* helper_op =
          module->symbols.entries[helper_symbol_id].defining_op;
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_emit_target_mismatch(
          module, root->function_op, root->target, helper_symbol_id, helper_op,
          &callable->target, scratch_arena, options->emitter));
      return iree_ok_status();
    }
  }
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_artifact_compile_helpers(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_options_t* options,
    const loom_op_t* const* first_call_ops,
    const loom_op_t* const* first_root_ops, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_program_artifact_t* artifact, bool* out_compiled) {
  *out_compiled = false;
  const loom_low_allocation_reserved_range_t call_control_reservation = {
      .register_class = IREE_SV("amdgpu.sgpr"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE,
      .location_count = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT,
  };
  for (iree_host_size_t i = 0; i < artifact->helper_count; ++i) {
    loom_amdgpu_program_callable_t* callable = &artifact->helpers[i];
    const loom_symbol_id_t symbol_id = callable->function.symbol_id;
    loom_op_t* helper_op = module->symbols.entries[symbol_id].defining_op;
    const loom_target_function_version_t* version =
        options->function_versions != NULL
            ? loom_target_function_version_snapshot_at(
                  options->function_versions, symbol_id)
            : NULL;
    bool abi_prepared = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_prepare_abi(
        module, helper_op, callable->target.descriptor_set, options->emitter,
        arena, callable, &abi_prepared));
    if (!abi_prepared) {
      return iree_ok_status();
    }
    if (options->report_template != NULL) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*callable->report),
                                               (void**)&callable->report));
      loom_target_compile_report_initialize(iree_arena_allocator(arena),
                                            callable->report);
      callable->report->requested_detail_flags =
          options->report_template->requested_detail_flags;
      callable->report->artifact_kind = options->report_template->artifact_kind;
      callable->report->backend_name = options->report_template->backend_name;
      callable->report->target_family_name =
          options->report_template->target_family_name;
      callable->report->target_key = options->report_template->target_key;
      callable->report->artifact_format =
          options->report_template->artifact_format;
    }
    const loom_amdgpu_function_compile_options_t helper_options = {
        .descriptor_registry = options->descriptor_registry,
        .resolved_target = &callable->target,
        .memory_accesses = version != NULL ? version->memory_accesses : NULL,
        .loop_execution = version != NULL ? version->loop_execution : NULL,
        .allocation_entry_locations = callable->call_contract.arguments,
        .allocation_entry_location_count =
            callable->call_contract.argument_count,
        .allocation_exit_locations = callable->call_contract.results,
        .allocation_exit_location_count = callable->call_contract.result_count,
        .callable = true,
        .allocation_reserved_ranges = &call_control_reservation,
        .allocation_reserved_range_count = 1,
        .emitter = options->emitter,
        .report = callable->report,
    };
    bool helper_compiled = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_function_compile(module, helper_op, &helper_options, arena,
                                     &helper_compiled, &callable->compilation));
    if (!helper_compiled) {
      return iree_ok_status();
    }
    if (callable->compilation.plan.storage_layout.segment_sizes
            .group_segment_fixed_size != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_emit_invalid_helper(
          module, first_call_ops[symbol_id], first_root_ops[symbol_id],
          symbol_id, IREE_SV("workgroup-storage"), helper_op,
          options->emitter));
      return iree_ok_status();
    }
    callable->call_state.counter_mask =
        callable->compilation.plan.packet_plan.wait_plan.produced_counter_mask;
    callable->call_state.fixed_state_mask =
        callable->compilation.plan.packet_plan.wait_plan.fixed_state_mask;
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_prepare_clobbers(
        callable->target.descriptor_set, arena, scratch_arena, callable));
    loom_amdgpu_program_call_table_t* call_table =
        loom_amdgpu_program_artifact_find_call_table(
            artifact, callable->target.descriptor_set);
    IREE_ASSERT_NE(call_table, NULL);
    loom_amdgpu_program_call_table_publish(call_table, callable);
  }
  *out_compiled = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_artifact_build_internal(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_root_input_t* root_inputs,
    iree_host_size_t root_count,
    const loom_amdgpu_program_artifact_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    bool* out_built, loom_amdgpu_program_artifact_t* out_artifact) {
  IREE_ASSERT_EQ(options->references->module, module);
  out_artifact->module = module;
  out_artifact->root_count = root_count;
  if (root_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, root_count, sizeof(*out_artifact->roots),
        (void**)&out_artifact->roots));
    memset(out_artifact->roots, 0, root_count * sizeof(*out_artifact->roots));
  }

  const iree_host_size_t symbol_count = module->symbols.count;
  uint8_t* helper_symbols = NULL;
  const loom_op_t** first_call_ops = NULL;
  const loom_op_t** first_root_ops = NULL;
  loom_amdgpu_program_callable_t** callables_by_symbol = NULL;
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, symbol_count,
                                                   sizeof(*helper_symbols),
                                                   (void**)&helper_symbols));
    memset(helper_symbols, 0, symbol_count * sizeof(*helper_symbols));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, symbol_count,
                                                   sizeof(*first_call_ops),
                                                   (void**)&first_call_ops));
    memset(first_call_ops, 0, symbol_count * sizeof(*first_call_ops));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, symbol_count,
                                                   sizeof(*first_root_ops),
                                                   (void**)&first_root_ops));
    memset(first_root_ops, 0, symbol_count * sizeof(*first_root_ops));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, symbol_count, sizeof(*callables_by_symbol),
        (void**)&callables_by_symbol));
    memset(callables_by_symbol, 0, symbol_count * sizeof(*callables_by_symbol));
  }

  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_discover_roots(
      module, root_inputs, root_count, options, arena, scratch_arena,
      helper_symbols, first_call_ops, first_root_ops, out_artifact, &accepted));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_reject_nested_calls(
      module, options, helper_symbols, &accepted));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_initialize_call_tables(
      module, arena, scratch_arena, out_artifact));
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_resolve_helpers(
      module, options, helper_symbols, arena, callables_by_symbol, out_artifact,
      &accepted));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_validate_root_targets(
      module, options, callables_by_symbol, scratch_arena, out_artifact,
      &accepted));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_artifact_compile_helpers(
      module, options, first_call_ops, first_root_ops, arena, scratch_arena,
      out_artifact, &accepted));
  if (!accepted) {
    return iree_ok_status();
  }
  *out_built = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_artifact_build(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_root_input_t* root_inputs,
    iree_host_size_t root_count,
    const loom_amdgpu_program_artifact_options_t* options,
    iree_arena_allocator_t* arena, bool* out_built,
    loom_amdgpu_program_artifact_t* out_artifact) {
  *out_built = false;
  *out_artifact = (loom_amdgpu_program_artifact_t){0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_amdgpu_program_artifact_build_internal(
      module, root_inputs, root_count, options, arena, &scratch_arena,
      out_built, out_artifact);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
