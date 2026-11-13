// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_compiler.h"

#include <stdint.h>

#include "loom/analysis/scc.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/artifact_diagnostics.h"
#include "loom/target/emit/spirv/module_emitter.h"
#include "loom/target/emit/spirv/program.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/artifact_manifest_collect.h"

typedef enum loom_spirv_program_function_disposition_e {
  // Function belongs to another target and is not part of this program.
  LOOM_SPIRV_PROGRAM_FUNCTION_SKIPPED = 0,
  // Function was appended to the program plan.
  LOOM_SPIRV_PROGRAM_FUNCTION_APPENDED = 1,
  // Function target resolution emitted a structured diagnostic.
  LOOM_SPIRV_PROGRAM_FUNCTION_REJECTED = 2,
} loom_spirv_program_function_disposition_t;

typedef struct loom_spirv_program_build_t {
  // Module containing the target-low functions.
  loom_module_t* module;
  // Low representation registry used during target binding.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Structured diagnostic emitter for target-resolution failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Cached symbol facts shared by target resolution for every function.
  loom_symbol_fact_table_t symbol_facts;
  // Compiler function versions observed against this module symbol snapshot.
  loom_target_function_version_snapshot_t function_versions;
  // Immutable module-wide reference snapshot used for call reachability.
  loom_symbol_reference_table_t references;
  // Mutable planned function storage.
  loom_spirv_function_plan_t* functions;
  // Number of initialized entries in |functions|.
  iree_host_size_t function_count;
  // Capacity of |functions|.
  iree_host_size_t function_capacity;
  // Mutable function-plan index by module symbol ID.
  uint32_t* function_indices_by_symbol;
  // Root membership by module symbol ID.
  bool* entry_symbols;
  // Explicit root target facts by module symbol ID, or NULL per symbol.
  const loom_target_facts_t** selected_target_facts_by_symbol;
  // Number of planned exported entries.
  iree_host_size_t entry_count;
  // First planned exported entry index.
  uint32_t first_entry_index;
} loom_spirv_program_build_t;

typedef struct loom_spirv_call_graph_t {
  // Immutable symbol-reference snapshot owning call occurrences.
  const loom_symbol_reference_table_t* references;
} loom_spirv_call_graph_t;

static iree_status_t loom_spirv_call_graph_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_spirv_call_graph_t* graph =
      (const loom_spirv_call_graph_t*)user_data;
  IREE_ASSERT_LT(node, graph->references->symbol_count);
  loom_symbol_reference_occurrence_id_t occurrence_id =
      loom_symbol_reference_table_symbol(graph->references,
                                         (loom_symbol_id_t)node)
          .first_outgoing_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(graph->references,
                                               occurrence_id);
    if (occurrence->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL &&
        occurrence->user_op != NULL &&
        loom_low_func_call_isa(occurrence->user_op)) {
      IREE_RETURN_IF_ERROR(
          successor.fn(successor.user_data, occurrence->target_symbol_id));
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_program_reject_missing_target(
    loom_spirv_program_build_t* build, loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("SPIR-V")),
      loom_param_string(
          loom_low_diagnostic_function_name(build->module, function_op)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = function_op,
      .error = LOOM_ERR_TARGET_009,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static const loom_target_facts_t* loom_spirv_program_function_target_facts(
    const loom_spirv_program_build_t* build, loom_op_t* function_op,
    const loom_target_facts_t* selected_target_facts) {
  if (selected_target_facts != NULL) {
    return selected_target_facts;
  }
  const loom_symbol_ref_t function_ref = loom_low_function_callee(function_op);
  const loom_target_function_version_t* function_version =
      loom_target_function_version_snapshot_at(&build->function_versions,
                                               function_ref.symbol_id);
  return function_version != NULL ? function_version->function_target_facts
                                  : NULL;
}

static const loom_func_type_data_t*
loom_spirv_program_function_logical_signature(const loom_module_t* module,
                                              const loom_op_t* function_op) {
  const loom_named_attr_slice_t layout =
      loom_low_function_abi_layout(function_op);
  if (layout.count == 0) {
    return NULL;
  }
  IREE_ASSERT_EQ(layout.count, 1,
                 "verified SPIR-V HAL kernel ABI layout field count");
  IREE_ASSERT_LT(layout.entries[0].name_id, module->strings.count,
                 "verified SPIR-V HAL kernel ABI layout field name");
  IREE_ASSERT_EQ(layout.entries[0].value.kind, LOOM_ATTR_TYPE,
                 "verified SPIR-V HAL kernel ABI layout field type");
  IREE_ASSERT(iree_string_view_equal(
      loom_string_table_get(&module->strings, layout.entries[0].name_id),
      IREE_SV("signature")));
  const loom_type_t signature =
      loom_type_table_get(&module->types, layout.entries[0].value.type_id);
  IREE_ASSERT(loom_type_is_function(signature));
  const loom_func_type_data_t* data = loom_type_func_data(signature);
  IREE_ASSERT_EQ(data->result_count, 0,
                 "verified SPIR-V HAL kernel logical result count");
  return data;
}

static iree_status_t loom_spirv_program_build_value_types(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    uint16_t value_count, iree_arena_allocator_t* arena,
    const loom_spirv_value_type_t** out_types) {
  *out_types = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_spirv_value_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*types), (void**)&types));
  for (uint16_t i = 0; i < value_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, value_ids[i]);
    const bool decoded =
        loom_spirv_value_type_from_low_register_type(type, &types[i]);
    IREE_ASSERT(decoded, "Low verification admits SPIR-V boundary types");
  }
  *out_types = types;
  return iree_ok_status();
}

static iree_status_t loom_spirv_program_build_internal_signature(
    loom_spirv_program_build_t* build, loom_op_t* function_op,
    loom_spirv_function_plan_t* function) {
  const loom_func_like_t function_like =
      loom_func_like_cast(build->module, function_op);
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function_like, &function->parameter_count);
  IREE_RETURN_IF_ERROR(loom_spirv_program_build_value_types(
      build->module, parameter_ids, function->parameter_count,
      build->symbol_facts.arena, &function->parameter_types));

  const loom_value_slice_t results = loom_low_func_def_results(function_op);
  function->result_count = results.count;
  return loom_spirv_program_build_value_types(
      build->module, results.values, function->result_count,
      build->symbol_facts.arena, &function->result_types);
}

static iree_status_t loom_spirv_program_build_function(
    loom_spirv_program_build_t* build, loom_op_t* function_op,
    const loom_target_facts_t* selected_target_facts,
    loom_spirv_program_function_disposition_t* out_disposition) {
  *out_disposition = LOOM_SPIRV_PROGRAM_FUNCTION_SKIPPED;
  if (!loom_low_function_def_isa(function_op)) {
    // Reached declarations are diagnosed at their callsites once definition
    // collection has established the complete emitted function set.
    return iree_ok_status();
  }

  const loom_target_facts_t* function_target_facts =
      loom_spirv_program_function_target_facts(build, function_op,
                                               selected_target_facts);
  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(function_target_facts),
      .target = build->diagnostic_emitter,
  };
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      build->module, &build->symbol_facts, function_op, function_target_facts,
      build->descriptor_registry,
      loom_diagnostic_export_scope_emitter(&diagnostic_scope), &target));
  if (target.descriptor_set == NULL) {
    *out_disposition = LOOM_SPIRV_PROGRAM_FUNCTION_REJECTED;
    return iree_ok_status();
  }

  const loom_target_bundle_t* target_bundle =
      loom_low_resolved_target_bundle(&target);
  if (target_bundle != NULL && target_bundle->snapshot->codegen_format !=
                                   LOOM_TARGET_CODEGEN_FORMAT_SPIRV) {
    return iree_ok_status();
  }
  if (target_bundle == NULL) {
    IREE_RETURN_IF_ERROR(loom_spirv_program_reject_missing_target(
        build, function_op,
        loom_diagnostic_export_scope_emitter(&diagnostic_scope)));
    *out_disposition = LOOM_SPIRV_PROGRAM_FUNCTION_REJECTED;
    return iree_ok_status();
  }
  diagnostic_scope.export_name =
      loom_target_facts_export_name(target.target_facts);

  const loom_func_type_data_t* logical_signature = NULL;
  if (target_bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    logical_signature = loom_spirv_program_function_logical_signature(
        build->module, function_op);
  }

  IREE_ASSERT_LT(build->function_count, build->function_capacity);
  const loom_symbol_ref_t function_ref = loom_low_function_callee(function_op);
  IREE_ASSERT(loom_symbol_ref_is_valid(function_ref) &&
              function_ref.module_id == 0 &&
              function_ref.symbol_id < build->module->symbols.count);
  if (build->entry_symbols[function_ref.symbol_id] &&
      target_bundle->export_plan->abi_kind !=
          LOOM_TARGET_ABI_SHADER_ENTRY_POINT &&
      target_bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    *out_disposition = LOOM_SPIRV_PROGRAM_FUNCTION_REJECTED;
    return loom_target_artifact_emit_requirement(
        loom_diagnostic_export_scope_emitter(&diagnostic_scope), function_op,
        IREE_SV("SPIR-V"), IREE_SV("a shader entry point or HAL kernel ABI"));
  }
  loom_spirv_function_plan_t* function =
      &build->functions[build->function_count];
  *function = (loom_spirv_function_plan_t){
      .function_op = function_op,
      .target_facts = target.target_facts,
      .descriptor_set = target.descriptor_set,
      .symbol_ref = function_ref,
      .logical_signature = logical_signature,
      .is_entry = build->entry_symbols[function_ref.symbol_id],
  };
  if (!function->is_entry) {
    IREE_RETURN_IF_ERROR(loom_spirv_program_build_internal_signature(
        build, function_op, function));
  }
  const uint32_t function_index = (uint32_t)build->function_count++;
  build->function_indices_by_symbol[function_ref.symbol_id] = function_index;
  if (function->is_entry) {
    if (build->entry_count++ == 0) {
      build->first_entry_index = function_index;
    }
  }
  *out_disposition = LOOM_SPIRV_PROGRAM_FUNCTION_APPENDED;
  return iree_ok_status();
}

static void loom_spirv_program_add_root(
    loom_spirv_program_build_t* build, loom_op_t* function_op,
    const loom_target_facts_t* selected_target_facts,
    iree_host_size_t* root_symbols, iree_host_size_t* root_count) {
  const loom_symbol_ref_t function_ref = loom_low_function_callee(function_op);
  if (!build->entry_symbols[function_ref.symbol_id]) {
    build->entry_symbols[function_ref.symbol_id] = true;
    root_symbols[(*root_count)++] = function_ref.symbol_id;
  }
  if (selected_target_facts != NULL) {
    build->selected_target_facts_by_symbol[function_ref.symbol_id] =
        selected_target_facts;
  }
}

static void loom_spirv_program_select_roots(
    loom_spirv_program_build_t* build,
    const loom_spirv_compile_options_t* options, iree_host_size_t* root_symbols,
    iree_host_size_t* out_root_count) {
  *out_root_count = 0;
  if (options != NULL && options->entry_count != 0) {
    for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
      loom_spirv_program_add_root(build, options->entries[i].function_op,
                                  options->entries[i].target_facts,
                                  root_symbols, out_root_count);
    }
    return;
  }

  bool has_retained_root = false;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < build->module->symbols.count;
       ++symbol_id) {
    const loom_symbol_t* symbol = &build->module->symbols.entries[symbol_id];
    if (!loom_low_function_def_isa(symbol->defining_op)) {
      continue;
    }
    const loom_function_version_t* version =
        loom_target_function_version_snapshot_handle_at(
            &build->function_versions, symbol_id);
    if (version != NULL &&
        iree_any_bit_set(version->flags, LOOM_FUNCTION_VERSION_FLAG_RETAIN)) {
      loom_spirv_program_add_root(build, symbol->defining_op,
                                  /*selected_target_facts=*/NULL, root_symbols,
                                  out_root_count);
      has_retained_root = true;
    }
  }
  if (has_retained_root) {
    return;
  }

  // Direct compiler API users without retained version identities preserve the
  // historical behavior of selecting every compatible Low definition.
  for (loom_symbol_id_t symbol_id = 0; symbol_id < build->module->symbols.count;
       ++symbol_id) {
    loom_op_t* function_op =
        build->module->symbols.entries[symbol_id].defining_op;
    if (loom_low_function_def_isa(function_op)) {
      loom_spirv_program_add_root(build, function_op,
                                  /*selected_target_facts=*/NULL, root_symbols,
                                  out_root_count);
    }
  }
}

static iree_status_t loom_spirv_program_reject_recursion(
    loom_spirv_program_build_t* build,
    const loom_spirv_function_plan_t* function) {
  const loom_op_t* function_op = function->function_op;
  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(function->target_facts),
      .target = build->diagnostic_emitter,
  };
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(build->module, function_op)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = function_op,
      .error = LOOM_ERR_SPIRV_032,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(
      loom_diagnostic_export_scope_emitter(&diagnostic_scope), &emission);
}

static iree_status_t loom_spirv_program_validate_call_closure(
    const loom_spirv_program_build_t* build, bool* out_accepted) {
  *out_accepted = false;
  for (iree_host_size_t i = 0; i < build->function_count; ++i) {
    const loom_spirv_function_plan_t* function = &build->functions[i];
    const loom_symbol_id_t symbol_id = function->symbol_ref.symbol_id;
    loom_symbol_reference_occurrence_id_t occurrence_id =
        loom_symbol_reference_table_symbol(&build->references, symbol_id)
            .first_outgoing_occurrence_id;
    while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* occurrence =
          loom_symbol_reference_table_occurrence(&build->references,
                                                 occurrence_id);
      if (occurrence->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL &&
          occurrence->user_op != NULL &&
          loom_low_func_call_isa(occurrence->user_op) &&
          build->function_indices_by_symbol[occurrence->target_symbol_id] ==
              LOOM_SPIRV_PROGRAM_FUNCTION_INDEX_NONE) {
        loom_diagnostic_export_scope_t diagnostic_scope = {
            .export_name =
                loom_target_facts_export_name(function->target_facts),
            .target = build->diagnostic_emitter,
        };
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_low_diagnostic_function_name(
                build->module, function->function_op)),
            loom_param_with_field_ref(
                loom_param_string(loom_low_diagnostic_symbol_name(
                    build->module,
                    loom_low_func_call_callee(occurrence->user_op))),
                loom_low_func_call_callee_diagnostic_ref()),
            loom_param_string(IREE_SV("spirv.logical.core")),
        };
        return iree_diagnostic_emit(
            loom_diagnostic_export_scope_emitter(&diagnostic_scope),
            &(loom_diagnostic_emission_t){
                .op = occurrence->user_op,
                .error = LOOM_ERR_BACKEND_059,
                .params = params,
                .param_count = IREE_ARRAYSIZE(params)});
      }
      occurrence_id = occurrence->next_outgoing_occurrence_id;
    }
  }
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_spirv_program_plan_build(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    const loom_spirv_compile_options_t* options, bool* out_accepted,
    loom_spirv_program_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_accepted);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_accepted = false;
  *out_plan = (loom_spirv_program_plan_t){0};

  if (module->symbols.count == 0) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, NULL, IREE_SV("SPIR-V"),
        IREE_SV("at least one Low function definition"));
  }
  loom_spirv_function_plan_t* functions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*functions), (void**)&functions));
  uint32_t* function_indices_by_symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*function_indices_by_symbol),
      (void**)&function_indices_by_symbol));
  bool* entry_symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*entry_symbols),
                                                 (void**)&entry_symbols));
  const loom_target_facts_t** selected_target_facts_by_symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*selected_target_facts_by_symbol),
      (void**)&selected_target_facts_by_symbol));
  iree_host_size_t* root_symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*root_symbols),
                                                 (void**)&root_symbols));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    function_indices_by_symbol[i] = LOOM_SPIRV_PROGRAM_FUNCTION_INDEX_NONE;
    entry_symbols[i] = false;
    selected_target_facts_by_symbol[i] = NULL;
  }

  loom_spirv_program_build_t build = {
      .module = module,
      .descriptor_registry = descriptor_registry,
      .diagnostic_emitter = diagnostic_emitter,
      .functions = functions,
      .function_capacity = module->symbols.count,
      .function_indices_by_symbol = function_indices_by_symbol,
      .entry_symbols = entry_symbols,
      .selected_target_facts_by_symbol = selected_target_facts_by_symbol,
      .first_entry_index = LOOM_SPIRV_PROGRAM_FUNCTION_INDEX_NONE,
  };
  loom_symbol_fact_table_initialize(&build.symbol_facts, arena);
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, options != NULL ? options->function_versions : NULL, arena,
      &build.function_versions));
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(module, arena, &build.references));

  iree_host_size_t root_count = 0;
  loom_spirv_program_select_roots(&build, options, root_symbols, &root_count);
  if (root_count == 0) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, NULL, IREE_SV("SPIR-V"),
        IREE_SV("at least one Low function root"));
  }

  const loom_spirv_call_graph_t call_graph = {
      .references = &build.references,
  };
  const loom_scc_graph_t graph = {
      .node_count = module->symbols.count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_spirv_call_graph_visit_successors, (void*)&call_graph),
  };
  const loom_scc_options_t scc_options = {
      .root_nodes = root_symbols,
      .root_count = root_count,
  };
  loom_scc_list_t sccs = {0};
  IREE_RETURN_IF_ERROR(loom_scc_compute(&graph, &scc_options, arena, &sccs));
  for (iree_host_size_t i = 0; i < sccs.count; ++i) {
    const loom_scc_t* scc = &sccs.values[i];
    const loom_spirv_function_plan_t* component_function = NULL;
    for (iree_host_size_t j = 0; j < scc->node_count; ++j) {
      const loom_symbol_id_t symbol_id = (loom_symbol_id_t)scc->nodes[j];
      loom_op_t* function_op = module->symbols.entries[symbol_id].defining_op;
      loom_spirv_program_function_disposition_t disposition =
          LOOM_SPIRV_PROGRAM_FUNCTION_SKIPPED;
      IREE_RETURN_IF_ERROR(loom_spirv_program_build_function(
          &build, function_op, selected_target_facts_by_symbol[symbol_id],
          &disposition));
      if (disposition == LOOM_SPIRV_PROGRAM_FUNCTION_REJECTED) {
        return iree_ok_status();
      }
      if (disposition == LOOM_SPIRV_PROGRAM_FUNCTION_APPENDED) {
        component_function = &build.functions[build.function_count - 1];
      }
    }
    if (scc->is_cycle && component_function != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_spirv_program_reject_recursion(&build, component_function));
      return iree_ok_status();
    }
  }

  if (build.entry_count == 0) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, NULL, IREE_SV("SPIR-V"),
        IREE_SV("at least one compatible Low function root"));
  }
  IREE_RETURN_IF_ERROR(
      loom_spirv_program_validate_call_closure(&build, out_accepted));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  *out_plan = (loom_spirv_program_plan_t){
      .module = module,
      .functions = build.functions,
      .function_count = build.function_count,
      .function_indices_by_symbol = build.function_indices_by_symbol,
      .symbol_count = module->symbols.count,
      .entry_count = build.entry_count,
      .first_entry_index = build.first_entry_index,
  };
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_spirv_program_compile(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    const loom_spirv_compile_options_t* options, iree_allocator_t allocator,
    bool* out_emitted, loom_spirv_program_plan_t* out_program,
    loom_spirv_module_binary_t* out_module) {
  *out_emitted = false;
  *out_program = (loom_spirv_program_plan_t){0};
  *out_module = (loom_spirv_module_binary_t){0};

  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_spirv_program_plan_build(
      module, descriptor_registry, diagnostic_emitter, arena, options,
      &accepted, out_program));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_spirv_program_emit_binary(out_program, arena,
                                                      allocator, out_module));
  *out_emitted = true;
  return iree_ok_status();
}

iree_status_t loom_spirv_compile_module_binary(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    const loom_spirv_compile_options_t* options, iree_allocator_t allocator,
    bool* out_emitted, loom_spirv_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_emitted = false;
  *out_module = (loom_spirv_module_binary_t){0};

  loom_spirv_program_plan_t program = {0};
  return loom_spirv_program_compile(
      module, descriptor_registry, diagnostic_emitter, arena, options,
      allocator, out_emitted, &program, out_module);
}

static iree_status_t loom_spirv_program_collect_manifest_entries(
    const loom_spirv_program_plan_t* program, iree_arena_allocator_t* arena,
    loom_target_entry_list_t* out_entries) {
  *out_entries = (loom_target_entry_list_t){0};
  loom_target_entry_t* entries = NULL;
  if (program->entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, program->entry_count, sizeof(*entries), (void**)&entries));
  }
  iree_host_size_t entry_index = 0;
  for (iree_host_size_t i = 0; i < program->function_count; ++i) {
    const loom_spirv_function_plan_t* function_plan = &program->functions[i];
    if (!function_plan->is_entry) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(program->module, function_plan->function_op);
    const loom_symbol_ref_t function_ref =
        loom_low_function_callee(function_plan->function_op);
    entries[entry_index++] = (loom_target_entry_t){
        .func = function,
        .func_name =
            loom_low_diagnostic_symbol_name(program->module, function_ref),
        .func_ref = function_ref,
        .target_facts = function_plan->target_facts,
    };
  }
  *out_entries = (loom_target_entry_list_t){
      .values = entries,
      .count = (uint16_t)entry_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_attach_artifact_metadata(
    const loom_target_emit_request_t* request,
    const loom_spirv_program_plan_t* program,
    loom_target_emit_artifact_t* artifact) {
  const bool retain_target_bundle = iree_any_bit_set(
      request->flags, LOOM_TARGET_EMIT_REQUEST_FLAG_RETAIN_TARGET_BUNDLE);
  const bool emit_artifact_manifest = request->artifact_manifest.mode !=
                                      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
  if (!retain_target_bundle && !emit_artifact_manifest) {
    return iree_ok_status();
  }

  loom_target_artifact_manifest_json_t manifest_json = {0};
  iree_byte_sequence_t* manifest_contents = NULL;
  iree_status_t status = iree_ok_status();
  if (emit_artifact_manifest) {
    loom_target_entry_list_t entries = {0};
    status = loom_spirv_program_collect_manifest_entries(
        program, request->scratch_arena, &entries);
    loom_target_artifact_manifest_collect_options_t manifest_options;
    loom_target_artifact_manifest_collect_options_initialize(&manifest_options);
    manifest_options.mode = request->artifact_manifest.mode;
    manifest_options.artifact_name = request->identifier;
    manifest_options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
    manifest_options.flags =
        LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
    manifest_options.artifact_byte_length =
        iree_byte_sequence_length(artifact->contents);
    if (iree_status_is_ok(status)) {
      status = loom_target_artifact_manifest_collect_json_from_entries(
          program->module, entries, &manifest_options, request->scratch_arena,
          request->allocator, &manifest_json);
    }
    IREE_ASSERT(!iree_status_is_ok(status) ||
                manifest_json.contents.data != NULL);
  }
  if (iree_status_is_ok(status) && emit_artifact_manifest) {
    iree_byte_span_t contents =
        iree_make_byte_span((uint8_t*)manifest_json.contents.data,
                            manifest_json.contents.data_length);
    status = iree_byte_sequence_create_from_span_move(
        &contents, request->allocator, &manifest_contents);
    if (iree_status_is_ok(status)) {
      manifest_json.contents = iree_const_byte_span_empty();
    }
  }

  if (iree_status_is_ok(status)) {
    const loom_target_emit_sidecar_artifact_t manifest = {
        .kind = LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST,
        .identifier = request->artifact_manifest.identifier,
        .contents = manifest_contents,
    };
    status = loom_target_emit_artifact_retain_metadata(
        retain_target_bundle
            ? loom_spirv_function_plan_target_bundle(
                  &program->functions[program->first_entry_index])
            : NULL,
        emit_artifact_manifest ? 1 : 0, &manifest, request->allocator,
        artifact);
  }

  iree_byte_sequence_release(manifest_contents);
  loom_target_artifact_manifest_json_release(&manifest_json,
                                             request->allocator);
  return status;
}

iree_status_t loom_spirv_compile_module_artifact(
    const loom_target_emit_request_t* request,
    const loom_spirv_compile_options_t* options, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};

  loom_spirv_program_plan_t program = {0};
  loom_spirv_module_binary_t binary = {0};
  bool module_emitted = false;
  iree_status_t status = loom_spirv_program_compile(
      request->module, request->low_descriptor_registry,
      request->diagnostic_emitter, request->scratch_arena, options,
      request->allocator, &module_emitted, &program, &binary);
  loom_target_emit_artifact_t artifact = {0};
  if (iree_status_is_ok(status) && module_emitted) {
    iree_byte_span_t contents =
        iree_make_byte_span(binary.words, binary.word_count * sizeof(uint32_t));
    status = iree_byte_sequence_create_from_span_move(
        &contents, request->allocator, &artifact.contents);
    if (iree_status_is_ok(status)) {
      binary.words = NULL;
      binary.word_count = 0;
      artifact.target_artifact_format =
          LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
    }
  }
  if (iree_status_is_ok(status) && module_emitted) {
    status = loom_spirv_module_attach_artifact_metadata(request, &program,
                                                        &artifact);
  }
  if (iree_status_is_ok(status) && module_emitted &&
      request->compile_report != NULL) {
    loom_target_compile_report_initialize_if_empty(request->compile_report,
                                                   request->allocator);
    const loom_target_bundle_t* target_bundle =
        loom_spirv_function_plan_target_bundle(
            &program.functions[program.first_entry_index]);
    request->compile_report->artifact_kind =
        target_bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL
            ? LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE
            : LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    request->compile_report->target_family_name =
        program.functions[program.first_entry_index]
            .target_facts->fact_type->name;
    loom_target_compile_report_record_target_bundle(request->compile_report,
                                                    target_bundle);
  }
  if (iree_status_is_ok(status) && module_emitted) {
    *out_artifact = artifact;
    artifact = (loom_target_emit_artifact_t){0};
    *out_emitted = true;
  }

  loom_target_emit_artifact_release(&artifact);
  loom_spirv_module_binary_deinitialize(&binary, request->allocator);
  return status;
}

static iree_status_t loom_spirv_module_emit(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  const loom_spirv_compile_options_t options = {
      .function_versions = request->function_versions,
  };
  return loom_spirv_compile_module_artifact(request, &options, out_emitted,
                                            out_artifact);
}

const loom_target_emitter_t loom_spirv_module_emitter = {
    .name = IREE_SVL("spirv"),
    .public_artifact_format = IREE_SVL("spirv"),
    .default_identifier = IREE_SVL("module.spv"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
    .default_pipeline_options =
        {
            .control_flow_lowering =
                LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
        },
    .emit = loom_spirv_module_emit,
};
