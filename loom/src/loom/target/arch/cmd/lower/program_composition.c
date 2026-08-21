// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_composition.h"

#include "loom/analysis/scc.h"
#include "loom/analysis/symbol_references.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/rewriter.h"

typedef struct loom_cmd_program_composition_t {
  // Module being flattened.
  loom_module_t* module;
  // Source module used to resolve diagnostic locations.
  const loom_module_t* diagnostic_module;
  // Caller-owned structured diagnostic sink.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Immutable symbol reference snapshot indexing command call sites.
  loom_symbol_reference_table_t references;
} loom_cmd_program_composition_t;

static bool loom_cmd_program_composition_is_call(
    const loom_cmd_program_composition_t* composition,
    const loom_symbol_reference_occurrence_t* occurrence) {
  if (occurrence->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
      !occurrence->user_op) {
    return false;
  }
  const loom_call_like_t call =
      loom_call_like_cast(composition->module, (loom_op_t*)occurrence->user_op);
  return loom_call_like_isa(call) &&
         loom_call_like_kind(call) == LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM;
}

static iree_status_t loom_cmd_program_composition_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_cmd_program_composition_t* composition =
      (const loom_cmd_program_composition_t*)user_data;
  IREE_ASSERT_LT(node, composition->references.symbol_count);
  loom_symbol_reference_occurrence_id_t occurrence_id =
      composition->references.symbols[node].first_outgoing_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &composition->references.occurrences[occurrence_id];
    if (loom_cmd_program_composition_is_call(composition, occurrence)) {
      IREE_RETURN_IF_ERROR(
          successor.fn(successor.user_data, occurrence->target_symbol_id));
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return iree_ok_status();
}

static iree_string_view_t loom_cmd_program_composition_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  IREE_ASSERT_LT(symbol_id, module->symbols.count);
  const loom_string_id_t name_id = module->symbols.entries[symbol_id].name_id;
  IREE_ASSERT_LT(name_id, module->strings.count);
  return module->strings.entries[name_id];
}

static const loom_op_t* loom_cmd_program_composition_diagnostic_op(
    const loom_cmd_program_composition_t* composition,
    iree_string_view_t symbol_name) {
  const loom_string_id_t name_id =
      loom_module_lookup_string(composition->diagnostic_module, symbol_name);
  IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(composition->diagnostic_module, name_id);
  IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_op_t* defining_op =
      composition->diagnostic_module->symbols.entries[symbol_id].defining_op;
  IREE_ASSERT(defining_op != NULL);
  return defining_op;
}

static iree_status_t loom_cmd_program_composition_reject_cycles(
    const loom_cmd_program_composition_t* composition,
    const loom_scc_list_t* sccs, bool* out_valid) {
  *out_valid = false;
  for (iree_host_size_t i = 0; i < sccs->count; ++i) {
    const loom_scc_t* component = &sccs->values[i];
    if (!component->is_cycle) continue;
    IREE_ASSERT_GT(component->node_count, 0u);
    const iree_string_view_t symbol_name =
        loom_cmd_program_composition_symbol_name(composition->module,
                                                 component->nodes[0]);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(symbol_name),
    };
    const loom_diagnostic_emission_t emission = {
        .op = loom_cmd_program_composition_diagnostic_op(composition,
                                                         symbol_name),
        .error = LOOM_ERR_LOWERING_049,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(composition->diagnostic_emitter, &emission);
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_composition_inline_component(
    loom_cmd_program_composition_t* composition, const loom_scc_t* component,
    loom_rewriter_t* rewriter) {
  IREE_ASSERT_EQ(component->node_count, 1u);
  const iree_host_size_t source_symbol_id = component->nodes[0];
  loom_symbol_reference_occurrence_id_t occurrence_id =
      composition->references.symbols[source_symbol_id]
          .first_outgoing_occurrence_id;
  iree_status_t status = iree_ok_status();
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID &&
         iree_status_is_ok(status)) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &composition->references.occurrences[occurrence_id];
    if (loom_cmd_program_composition_is_call(composition, occurrence)) {
      status = loom_callable_inline_direct_call(
          rewriter, (loom_op_t*)occurrence->user_op);
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return status;
}

iree_status_t loom_cmd_program_composition_flatten(
    loom_module_t* module, const loom_module_t* diagnostic_module,
    const loom_func_like_t* root_programs, iree_host_size_t root_program_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(diagnostic_module);
  IREE_ASSERT_ARGUMENT(root_programs);
  IREE_ASSERT_GT(root_program_count, 0u);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_valid);
  *out_valid = false;

  loom_cmd_program_composition_t composition = {
      .module = module,
      .diagnostic_module = diagnostic_module,
      .diagnostic_emitter = diagnostic_emitter,
  };
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_build(
      module, arena, &composition.references));

  iree_host_size_t* root_nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, root_program_count, sizeof(*root_nodes), (void**)&root_nodes));
  for (iree_host_size_t i = 0; i < root_program_count; ++i) {
    IREE_ASSERT(loom_func_like_isa(root_programs[i]));
    const loom_symbol_ref_t root_ref = loom_func_like_callee(root_programs[i]);
    IREE_ASSERT(loom_symbol_ref_is_valid(root_ref));
    IREE_ASSERT_EQ(root_ref.module_id, 0u);
    IREE_ASSERT_LT(root_ref.symbol_id, module->symbols.count);
    root_nodes[i] = root_ref.symbol_id;
  }

  const loom_scc_graph_t graph = {
      .node_count = module->symbols.count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_cmd_program_composition_visit_successors, &composition),
  };
  const loom_scc_options_t options = {
      .root_nodes = root_nodes,
      .root_count = root_program_count,
  };
  loom_scc_list_t sccs = {0};
  IREE_RETURN_IF_ERROR(loom_scc_compute(&graph, &options, arena, &sccs));
  bool valid = false;
  IREE_RETURN_IF_ERROR(
      loom_cmd_program_composition_reject_cycles(&composition, &sccs, &valid));
  if (!valid) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < sccs.count && iree_status_is_ok(status);
       ++i) {
    status = loom_cmd_program_composition_inline_component(
        &composition, &sccs.values[i], &rewriter);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (iree_status_is_ok(status)) *out_valid = true;
  return status;
}
