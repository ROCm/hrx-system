// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_execution.h"

#include <string.h>

#include "loom/analysis/availability.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/loop_execution.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/capture.h"
#include "loom/ops/low/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/facts.h"
#include "loom/target/function_version.h"
#include "loom/transforms/cleanup/canonicalizer.h"
#include "loom/transforms/cleanup/pass_environment.h"
#include "loom/transforms/symbol/inline_plan.h"

static bool loom_inline_symbol_is_transferable(const loom_module_t* module,
                                               const loom_symbol_t* symbol) {
  if (!symbol || !symbol->defining_op) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags,
                       LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN)) {
    return false;
  }
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    return false;
  }

  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return false;
  }
  return loom_func_like_is_module_internal(function);
}

//===----------------------------------------------------------------------===//
// Execution planning
//===----------------------------------------------------------------------===//

static bool loom_inline_symbol_can_transfer(
    const loom_inline_callables_plan_t* state, loom_symbol_id_t symbol_id) {
  if (symbol_id >= state->module->symbols.count) {
    return false;
  }
  const loom_inline_symbol_info_t* info = &state->symbols[symbol_id];
  if (info->incoming_call_count != info->planned_call_removals) {
    return false;
  }
  if (info->incoming_non_call_ref_count != 0) {
    return false;
  }
  if (!loom_inline_symbol_is_transferable(state->module, info->symbol)) {
    return false;
  }
  const loom_function_version_t* version =
      loom_target_function_version_snapshot_handle_at(
          state->options.target_versions, symbol_id);
  if (version != NULL &&
      (state->version_owner == NULL ||
       iree_any_bit_set(version->flags, LOOM_FUNCTION_VERSION_FLAG_RETAIN))) {
    return false;
  }
  const loom_symbol_ref_t family =
      loom_func_like_template_family(info->function);
  if (loom_symbol_ref_is_valid(family) && state->options.demanded_families &&
      (state->options.demanded_families[family.symbol_id >> 6] &
       (UINT64_C(1) << (family.symbol_id & 63)))) {
    return false;
  }
  return loom_func_like_body(info->function) != NULL;
}

static iree_status_t loom_inline_select_transfer_actions(
    loom_inline_callables_plan_t* state) {
  uint32_t* final_entry_by_symbol = NULL;
  if (state->module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, state->module->symbols.count,
        sizeof(*final_entry_by_symbol), (void**)&final_entry_by_symbol));
    for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
      final_entry_by_symbol[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
    }
  }

  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    for (uint32_t entry_index =
             state->component_entries[component_index].first_entry;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->entries[entry_index].next_required_in_component) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action == LOOM_INLINE_PLAN_ACTION_REQUIRED &&
          loom_inline_symbol_can_transfer(state, entry->target_symbol_id)) {
        final_entry_by_symbol[entry->target_symbol_id] = entry_index;
      }
    }
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
      continue;
    }
    if (final_entry_by_symbol[entry->target_symbol_id] == entry_index) {
      entry->action = LOOM_INLINE_PLAN_ACTION_TRANSFER;
    } else {
      entry->action = LOOM_INLINE_PLAN_ACTION_CLONE;
    }
  }
  return iree_ok_status();
}

typedef struct loom_inline_low_product_clone_t {
  // Memory-summary identity translator, or NULL without captured summaries.
  loom_low_memory_access_clone_t* memory;
  // Loop-control identity translator, or NULL without loop execution facts.
  loom_low_loop_execution_clone_t* loop_execution;
} loom_inline_low_product_clone_t;

static iree_status_t loom_inline_clone_low_products(void* user_data,
                                                    const loom_op_t* source_op,
                                                    loom_op_t* target_op) {
  loom_inline_low_product_clone_t* clone = user_data;
  if (clone->memory != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_memory_access_clone_op(clone->memory, source_op, target_op));
  }
  if (clone->loop_execution != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_clone_op(
        clone->loop_execution, source_op, target_op));
  }
  return iree_ok_status();
}

// The inline plan already names both concrete versions. Copying a callee gets
// fresh operation identities; moving its only invocation retains them.
static iree_status_t loom_inline_low_products(
    loom_inline_callables_plan_t* state, const loom_inline_plan_entry_t* entry,
    loom_ir_clone_observer_t* out_observer) {
  *out_observer = (loom_ir_clone_observer_t){0};
  const loom_target_function_version_t* callee =
      loom_target_function_version_snapshot_at(state->options.target_versions,
                                               entry->target_symbol_id);
  if (callee == NULL ||
      (callee->memory_accesses == NULL && callee->loop_execution == NULL)) {
    return iree_ok_status();
  }
  loom_target_function_version_t* caller = loom_target_function_version_cast(
      loom_target_function_version_snapshot_handle_at(
          state->options.target_versions, entry->source_symbol_id));
  // Target-bound Low inlining requires the caller's concrete target context.
  IREE_ASSERT(caller != NULL);
  if (entry->action == LOOM_INLINE_PLAN_ACTION_TRANSFER) {
    if (callee->memory_accesses != NULL) {
      if (caller->memory_accesses == NULL) {
        caller->memory_accesses = callee->memory_accesses;
      } else if (caller->memory_accesses != callee->memory_accesses) {
        IREE_RETURN_IF_ERROR(loom_low_memory_access_map_transfer(
            callee->memory_accesses, caller->memory_accesses));
      }
    }
    if (callee->loop_execution != NULL) {
      if (caller->loop_execution == NULL) {
        caller->loop_execution = callee->loop_execution;
      } else if (caller->loop_execution != callee->loop_execution) {
        IREE_RETURN_IF_ERROR(loom_low_loop_execution_map_transfer(
            callee->loop_execution, caller->loop_execution));
      }
    }
    return iree_ok_status();
  }
  loom_inline_low_product_clone_t* clone = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(state->pass->arena, sizeof(*clone), (void**)&clone));
  *clone = (loom_inline_low_product_clone_t){0};
  if (callee->memory_accesses != NULL) {
    if (caller->memory_accesses == NULL) {
      IREE_RETURN_IF_ERROR(loom_low_memory_access_map_allocate(
          &state->module->arena, &caller->memory_accesses));
    }
    IREE_RETURN_IF_ERROR(loom_low_memory_access_clone_allocate(
        callee->memory_accesses, caller->memory_accesses, state->pass->arena,
        &clone->memory));
  }
  if (callee->loop_execution != NULL) {
    if (caller->loop_execution == NULL) {
      IREE_RETURN_IF_ERROR(loom_low_loop_execution_map_allocate(
          &state->module->arena, &caller->loop_execution));
    }
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_clone_allocate(
        callee->loop_execution, caller->loop_execution, state->pass->arena,
        &clone->loop_execution));
  }
  *out_observer = (loom_ir_clone_observer_t){
      .fn = loom_inline_clone_low_products,
      .user_data = clone,
  };
  return iree_ok_status();
}

static iree_status_t loom_inline_execute_entry(
    loom_inline_callables_plan_t* state, loom_rewriter_t* rewriter,
    const loom_availability_analysis_t* transfer_availability,
    loom_inline_plan_entry_t* entry, bool* out_changed_cfg_topology) {
  *out_changed_cfg_topology = false;
  loom_ir_clone_observer_t clone_observer = {0};
  IREE_RETURN_IF_ERROR(loom_inline_low_products(state, entry, &clone_observer));
  switch (entry->action) {
    case LOOM_INLINE_PLAN_ACTION_CLONE: {
      const bool body_is_linear =
          loom_callable_body_is_linear(state->module, entry->callee);
      const loom_callable_build_branch_fn_t build_branch =
          loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_LOW_INTERNAL
              ? loom_low_br_build
              : loom_cfg_br_build;
      IREE_RETURN_IF_ERROR(loom_callable_inline_call_with_branch(
          rewriter, entry->call_op, entry->callee, build_branch,
          clone_observer));
      loom_pass_mark_changed(state->pass);
      ++state->statistics.calls_cloned;
      *out_changed_cfg_topology = !body_is_linear;
      return iree_ok_status();
    }
    case LOOM_INLINE_PLAN_ACTION_TRANSFER: {
      const bool body_is_linear =
          loom_callable_body_is_linear(state->module, entry->callee);
      const bool use_linear_move = body_is_linear && transfer_availability;
      loom_function_version_t* version =
          loom_target_function_version_snapshot_handle_at(
              state->options.target_versions, entry->target_symbol_id);
      const loom_callable_build_branch_fn_t build_branch =
          loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_LOW_INTERNAL
              ? loom_low_br_build
              : loom_cfg_br_build;
      IREE_RETURN_IF_ERROR(loom_callable_inline_consuming_call_with_branch(
          rewriter, transfer_availability, entry->call_op, entry->callee,
          build_branch));
      if (version != NULL) {
        ++state->erased_version_count;
      }
      loom_pass_mark_changed(state->pass);
      ++state->statistics.calls_transferred;
      ++state->statistics.symbols_transferred;
      *out_changed_cfg_topology = !use_linear_move;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

// Linear command bodies implicitly sequence their children. Preserve that
// scope when a call expands inside a concurrent region, including when an
// enclosing template or static branch is subsequently inlined. Predicted CFG
// bodies retain their function-level region for transitive CFG splicing and
// require CFG-capable callers during preflight.
static iree_status_t loom_inline_materialize_command_schedule(
    loom_inline_callables_plan_t* state, const loom_inline_plan_entry_t* entry,
    loom_rewriter_t* rewriter) {
  if (!state->symbols[entry->target_symbol_id].body_will_be_linear) {
    return iree_ok_status();
  }
  loom_block_t* block =
      loom_region_entry_block(loom_func_like_body(entry->callee));
  loom_op_t* first_op = block->first_op;
  loom_op_t* terminator = block->last_op;
  if (first_op == terminator ||
      (loom_command_serial_isa(first_op) && first_op->next_op == terminator)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, first_op);
  loom_op_t* serial_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_command_serial_build(builder, first_op->location, &serial_op));
  const loom_builder_ip_t saved = loom_builder_enter_region(
      builder, serial_op, loom_command_serial_body(serial_op));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_command_yield_build(builder, terminator->location, &yield_op));
  loom_builder_restore(builder, saved);
  iree_status_t status = iree_ok_status();
  for (loom_op_t* op = first_op;
       op != terminator && iree_status_is_ok(status);) {
    loom_op_t* next_op = op->next_op;
    status = loom_rewriter_move_before(rewriter, op, yield_op);
    op = next_op;
  }
  loom_pass_mark_changed(state->pass);
  return status;
}

// A call captures each register argument's contents independently. Callee-local
// destructive updates cannot consume the caller's register identity when the
// boundary disappears. Allocation may reuse storage once its observations end.
static iree_status_t loom_inline_capture_low_arguments(
    loom_rewriter_t* rewriter, loom_op_t* call_op, loom_func_like_t callee) {
  const loom_value_slice_t results = loom_low_func_call_results(call_op);
  bool results_are_stable = true;
  for (uint16_t i = 0; i < results.count; ++i) {
    results_are_stable &=
        loom_low_capture_source_is_stable(rewriter->module, results.values[i]);
  }
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(callee, &argument_count);
  const loom_builder_ip_t saved = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, call_op);
  const loom_value_slice_t operands = loom_low_func_call_operands(call_op);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t source = operands.values[i];
    const loom_use_t use = loom_value_uses(loom_module_value(
        rewriter->module, source))[loom_op_operand_use_indices(call_op)[i]];
    // Explicit call ties transfer the caller's owner into the callee.
    if (iree_any_bit_set(loom_use_flags(use), LOOM_USE_FLAG_CONSUMES)) {
      continue;
    }
    // Stable inputs and observations need no independent register owner. A
    // returned argument can acquire destructive users in the caller, so the
    // result boundary participates in this proof even for an identity helper.
    if (results_are_stable &&
        loom_low_capture_source_is_stable(rewriter->module, source) &&
        loom_low_capture_source_is_stable(rewriter->module, arguments[i])) {
      continue;
    }
    loom_op_t* copy_op = NULL;
    status =
        loom_low_copy_build(&rewriter->builder, source, false,
                            loom_module_value_type(rewriter->module, source),
                            call_op->location, &copy_op);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operand(rewriter, call_op, i,
                                         loom_low_copy_result(copy_op));
    }
  }
  loom_builder_restore(&rewriter->builder, saved);
  return status;
}

// Makes argument capture and scheduling explicit before a body crosses a
// callable boundary. Commands retain their serial scope. Locked blocks
// receive a fence before and after each instruction; phased functions receive
// one scope closed at every return. Normalization is idempotent for retained
// helpers so generic cloning and CFG splicing preserve the explicit contracts.
static iree_status_t loom_inline_materialize_boundaries(
    loom_inline_callables_plan_t* state, loom_rewriter_t* rewriter) {
  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_CLONE &&
        entry->action != LOOM_INLINE_PLAN_ACTION_TRANSFER) {
      continue;
    }
    const loom_call_like_kind_t kind = loom_call_like_kind(entry->call);
    if (kind == LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM) {
      IREE_RETURN_IF_ERROR(
          loom_inline_materialize_command_schedule(state, entry, rewriter));
      continue;
    }
    if (kind != LOOM_CALL_LIKE_KIND_LOW_INTERNAL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_inline_capture_low_arguments(
        rewriter, entry->call_op, entry->callee));
    const loom_low_schedule_t schedule =
        loom_low_func_def_schedule(entry->callee.op);
    if (schedule != LOOM_LOW_SCHEDULE_PHASED &&
        schedule != LOOM_LOW_SCHEDULE_LOCKED) {
      continue;
    }

    loom_region_t* body = loom_func_like_body(entry->callee);
    IREE_RETURN_IF_ERROR(loom_low_function_materialize_schedule(
        &rewriter->builder, schedule, body->blocks, body->block_count));
    IREE_RETURN_IF_ERROR(loom_low_func_def_rewrite_schedule(
        rewriter, entry->callee.op, loom_attr_absent()));
    loom_pass_mark_changed(state->pass);
  }
  return iree_ok_status();
}

static bool loom_inline_action_executes(loom_inline_plan_action_t action) {
  return action == LOOM_INLINE_PLAN_ACTION_CLONE ||
         action == LOOM_INLINE_PLAN_ACTION_TRANSFER;
}

typedef struct loom_inline_execution_symbol_t {
  // Executing call edges whose original caller is this symbol.
  uint32_t remaining_outgoing_count;
  // Clone uses that must observe this symbol before its consuming use.
  uint32_t remaining_clone_count;
  // Head of the clone-use list targeting this symbol.
  uint32_t first_clone_entry;
  // Tail used to preserve source occurrence order in the clone-use list.
  uint32_t last_clone_entry;
  // Unique consuming use targeting this symbol, when present.
  uint32_t transfer_entry;
  // True after the consuming use moved this symbol's body into its caller.
  bool transfer_executed;
} loom_inline_execution_symbol_t;

typedef struct loom_inline_execution_entry_t {
  // Next clone waiting on the same callee, then next clone in the ready list.
  // Readiness consumes the waiting link before repurposing it.
  uint32_t next_clone_entry;
  // Topology-preserving entry stored in this ready-stack slot.
  uint32_t linear_ready_entry;
  // CFG-consuming entry stored in this ready-stack slot.
  uint32_t cfg_ready_entry;
} loom_inline_execution_entry_t;

typedef struct loom_inline_ready_entries_t {
  // Number of topology-preserving entries in the linear ready stack.
  uint32_t linear_count;
  // Number of consuming CFG entries in the CFG ready stack.
  uint32_t cfg_count;
  // Head of the independent CFG clone list, or the invalid entry sentinel.
  uint32_t first_clone;
  // Number of entries in the independent CFG clone list.
  uint32_t clone_count;
} loom_inline_ready_entries_t;

static void loom_inline_enqueue_ready_entry(
    const loom_inline_callables_plan_t* state,
    loom_inline_execution_entry_t* execution_entries, uint32_t entry_index,
    loom_inline_ready_entries_t* ready) {
  const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
  if (loom_callable_body_is_linear(state->module, entry->callee)) {
    execution_entries[ready->linear_count++].linear_ready_entry = entry_index;
  } else if (entry->action == LOOM_INLINE_PLAN_ACTION_CLONE) {
    execution_entries[entry_index].next_clone_entry = ready->first_clone;
    ready->first_clone = entry_index;
    ++ready->clone_count;
  } else {
    execution_entries[ready->cfg_count++].cfg_ready_entry = entry_index;
  }
}

static iree_status_t loom_inline_refine_source_closure(
    loom_inline_callables_plan_t* state, loom_func_like_t function);

// Completes one executed call edge and propagates closure completion through
// wrappers that were already transferred outward. A transfer may move its body
// before calls carried by that body execute, but its caller must continue to
// observe the transfer as an outstanding edge until those calls complete.
static iree_status_t loom_inline_complete_execution_entry(
    loom_inline_callables_plan_t* state,
    loom_inline_execution_symbol_t* execution_symbols,
    loom_inline_execution_entry_t* execution_entries, uint32_t entry_index,
    loom_inline_ready_entries_t* ready, bool* availability_valid) {
  while (entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    loom_inline_execution_symbol_t* source =
        &execution_symbols[entry->source_symbol_id];
    IREE_ASSERT_GT(source->remaining_outgoing_count, 0u);
    if (--source->remaining_outgoing_count != 0) {
      return iree_ok_status();
    }

    if (!source->transfer_executed &&
        source->first_clone_entry != LOOM_INLINE_PLAN_ENTRY_INVALID) {
      const loom_func_like_t function =
          state->symbols[entry->source_symbol_id].function;
      // Source composition exposes constant arguments inside the completed
      // callee. Fold those expressions once before its callers copy them.
      // Prepared Low bodies carry retained machine plans and are already past
      // source refinement; their clone observers preserve those plans intact.
      if (!loom_low_func_def_isa(function.op)) {
        IREE_RETURN_IF_ERROR(
            loom_inline_refine_source_closure(state, function));
        *availability_valid = false;
      }
    }

    for (uint32_t clone_index = source->first_clone_entry;
         clone_index != LOOM_INLINE_PLAN_ENTRY_INVALID;) {
      const uint32_t next_clone =
          execution_entries[clone_index].next_clone_entry;
      loom_inline_enqueue_ready_entry(state, execution_entries, clone_index,
                                      ready);
      clone_index = next_clone;
    }
    if (!source->transfer_executed) {
      return iree_ok_status();
    }
    entry_index = source->transfer_entry;
  }
  return iree_ok_status();
}

// Releases clone dependencies only after the materialization boundary
// completes. A batch publishes every region's lexical order before any
// completed caller becomes another clone's source.
static iree_status_t loom_inline_finish_clone(
    loom_inline_callables_plan_t* state,
    loom_inline_execution_symbol_t* execution_symbols,
    loom_inline_execution_entry_t* execution_entries, uint32_t entry_index,
    loom_inline_ready_entries_t* ready, bool* availability_valid) {
  const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
  loom_inline_execution_symbol_t* target =
      &execution_symbols[entry->target_symbol_id];
  IREE_RETURN_IF_ERROR(loom_inline_complete_execution_entry(
      state, execution_symbols, execution_entries, entry_index, ready,
      availability_valid));
  IREE_ASSERT_GT(target->remaining_clone_count, 0u);
  if (--target->remaining_clone_count == 0 &&
      target->transfer_entry != LOOM_INLINE_PLAN_ENTRY_INVALID) {
    loom_inline_enqueue_ready_entry(state, execution_entries,
                                    target->transfer_entry, ready);
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_refine_source_closure(
    loom_inline_callables_plan_t* state, loom_func_like_t function) {
  const loom_cleanup_pass_capability_t* capability =
      loom_cleanup_pass_capability_from_pass(state->pass);
  const loom_cleanup_pattern_registry_t* registry =
      loom_cleanup_pass_capability_pattern_registry(capability);
  loom_cleanup_canonicalizer_context_t context = {0};
  IREE_RETURN_IF_ERROR(
      loom_cleanup_pass_capability_resolve_canonicalizer_context(
          capability, state->pass, state->module, function, &context));
  const loom_canonicalizer_options_t options = {
      .patterns =
          loom_canonicalizer_pattern_registries_from_cleanup_registry(registry),
      .target_facts = context.target_facts,
      .math_policy = context.math_policy,
  };
  loom_pass_value_fact_owner_invalidate(state->pass->value_facts);
  loom_canonicalizer_t canonicalizer;
  IREE_RETURN_IF_ERROR(loom_canonicalizer_initialize(
      state->module, state->pass->arena, state->pass->value_facts,
      registry ? registry->special_value_policy : NULL,
      registry ? registry->fact_refinement_policy : NULL, &canonicalizer));
  loom_canonicalizer_result_t result;
  iree_status_t status = loom_canonicalizer_run_function(
      &canonicalizer, function, &options, &result);
  loom_canonicalizer_deinitialize(&canonicalizer);
  loom_pass_value_fact_owner_invalidate(state->pass->value_facts);
  return status;
}

// Executes required edges without repeatedly cloning a growing callee closure.
// A clone waits for the target's outgoing edges so it observes the complete
// flattened body. The target's unique transfer waits for all clone uses, but
// may move a single-use wrapper outward before flattening calls carried by that
// wrapper. The transfer remains logically outstanding until those calls
// complete, preventing an outer shared wrapper from cloning a partial closure.
// Linear ready edges run before CFG edges so their topology-preserving move can
// share one availability analysis and so a CFG parent carries its linear
// producers when it moves. The ready stacks reverse source occurrence order:
// sibling CFG calls therefore splice right to left, moving each original
// operation tail at most once. Independent ready CFG clones share one block
// layout publication, including splices whose results escape the moving tail.
// Callee closure completion proves their sources immutable throughout a batch.
iree_status_t loom_inline_execute_plan(loom_inline_callables_plan_t* state) {
  IREE_RETURN_IF_ERROR(loom_inline_select_transfer_actions(state));
  uint32_t execution_count = 0;
  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    if (loom_inline_action_executes(state->entries[entry_index].action)) {
      ++execution_count;
    }
  }
  if (execution_count == 0) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter = {0};
  loom_rewriter_initialize(&rewriter, state->module, state->pass->arena);

  iree_status_t status = loom_inline_materialize_boundaries(state, &rewriter);
  const iree_host_size_t symbol_count = state->module->symbols.count;
  loom_inline_execution_symbol_t* execution_symbols = NULL;
  loom_inline_execution_entry_t* execution_entries = NULL;
  if (iree_status_is_ok(status) && symbol_count > 0) {
    status = iree_arena_allocate_array(state->pass->arena, symbol_count,
                                       sizeof(*execution_symbols),
                                       (void**)&execution_symbols);
  }
  if (iree_status_is_ok(status) && state->entry_count > 0) {
    status = iree_arena_allocate_array(state->pass->arena, state->entry_count,
                                       sizeof(*execution_entries),
                                       (void**)&execution_entries);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < symbol_count; ++i) {
      execution_symbols[i] = (loom_inline_execution_symbol_t){
          .first_clone_entry = LOOM_INLINE_PLAN_ENTRY_INVALID,
          .last_clone_entry = LOOM_INLINE_PLAN_ENTRY_INVALID,
          .transfer_entry = LOOM_INLINE_PLAN_ENTRY_INVALID,
      };
    }
    for (uint32_t i = 0; i < state->entry_count; ++i) {
      execution_entries[i].next_clone_entry = LOOM_INLINE_PLAN_ENTRY_INVALID;
    }
  }

  for (uint32_t entry_index = 0;
       iree_status_is_ok(status) && entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (!loom_inline_action_executes(entry->action)) {
      continue;
    }
    IREE_ASSERT_LT(entry->source_symbol_id, symbol_count);
    IREE_ASSERT_LT(entry->target_symbol_id, symbol_count);
    ++execution_symbols[entry->source_symbol_id].remaining_outgoing_count;
    if (entry->action == LOOM_INLINE_PLAN_ACTION_CLONE) {
      loom_inline_execution_symbol_t* target =
          &execution_symbols[entry->target_symbol_id];
      ++target->remaining_clone_count;
      if (target->last_clone_entry == LOOM_INLINE_PLAN_ENTRY_INVALID) {
        target->first_clone_entry = entry_index;
      } else {
        execution_entries[target->last_clone_entry].next_clone_entry =
            entry_index;
      }
      target->last_clone_entry = entry_index;
    } else {
      loom_inline_execution_symbol_t* target =
          &execution_symbols[entry->target_symbol_id];
      IREE_ASSERT_EQ(target->transfer_entry, LOOM_INLINE_PLAN_ENTRY_INVALID);
      target->transfer_entry = entry_index;
    }
  }

  loom_inline_ready_entries_t ready = {
      .first_clone = LOOM_INLINE_PLAN_ENTRY_INVALID,
  };
  for (iree_host_size_t component_index = 0;
       iree_status_is_ok(status) && component_index < state->sccs.count;
       ++component_index) {
    for (uint32_t entry_index =
             state->component_entries[component_index].first_entry;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->entries[entry_index].next_required_in_component) {
      const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (!loom_inline_action_executes(entry->action)) {
        continue;
      }
      const loom_inline_execution_symbol_t* target =
          &execution_symbols[entry->target_symbol_id];
      const bool entry_ready = entry->action == LOOM_INLINE_PLAN_ACTION_CLONE
                                   ? target->remaining_outgoing_count == 0
                                   : target->remaining_clone_count == 0;
      if (entry_ready) {
        loom_inline_enqueue_ready_entry(state, execution_entries, entry_index,
                                        &ready);
      }
    }
  }

  loom_availability_analysis_t transfer_availability = {0};
  bool transfer_availability_valid = false;
  bool cfg_topology_changed = false;
  uint32_t executed_count = 0;
  loom_callable_inline_site_t* clone_sites = NULL;
  while (iree_status_is_ok(status) &&
         (ready.linear_count > 0 || ready.clone_count > 0 ||
          ready.cfg_count > 0)) {
    if (ready.linear_count == 0 && ready.clone_count > 1) {
      if (!clone_sites) {
        status = iree_arena_allocate_array(state->pass->arena, execution_count,
                                           sizeof(*clone_sites),
                                           (void**)&clone_sites);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
      const uint32_t first_clone = ready.first_clone;
      const uint32_t clone_count = ready.clone_count;
      uint32_t site_index = 0;
      for (uint32_t clone_index = first_clone;
           clone_index != LOOM_INLINE_PLAN_ENTRY_INVALID &&
           iree_status_is_ok(status);
           clone_index = execution_entries[clone_index].next_clone_entry) {
        const loom_inline_plan_entry_t* entry = &state->entries[clone_index];
        loom_callable_inline_site_t* site = &clone_sites[site_index++];
        *site = (loom_callable_inline_site_t){
            .call_op = entry->call_op,
            .callee = entry->callee,
            .build_branch = loom_call_like_kind(entry->call) ==
                                    LOOM_CALL_LIKE_KIND_LOW_INTERNAL
                                ? loom_low_br_build
                                : loom_cfg_br_build,
        };
        status = loom_inline_low_products(state, entry, &site->clone_observer);
      }
      if (iree_status_is_ok(status)) {
        status = loom_callable_inline_calls_with_branch(&rewriter, clone_sites,
                                                        clone_count);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      loom_pass_mark_changed(state->pass);
      state->statistics.calls_cloned += clone_count;
      executed_count += clone_count;
      transfer_availability_valid = false;
      cfg_topology_changed = true;
      ready.first_clone = LOOM_INLINE_PLAN_ENTRY_INVALID;
      ready.clone_count = 0;
      for (uint32_t clone_index = first_clone;
           clone_index != LOOM_INLINE_PLAN_ENTRY_INVALID &&
           iree_status_is_ok(status);
           clone_index = execution_entries[clone_index].next_clone_entry) {
        status = loom_inline_finish_clone(state, execution_symbols,
                                          execution_entries, clone_index,
                                          &ready, &transfer_availability_valid);
      }
      continue;
    }

    uint32_t entry_index = LOOM_INLINE_PLAN_ENTRY_INVALID;
    if (ready.linear_count > 0) {
      entry_index = execution_entries[--ready.linear_count].linear_ready_entry;
    } else if (ready.clone_count > 0) {
      entry_index = ready.first_clone;
      ready.first_clone = execution_entries[entry_index].next_clone_entry;
      --ready.clone_count;
    } else {
      entry_index = execution_entries[--ready.cfg_count].cfg_ready_entry;
    }
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    const bool body_is_linear =
        entry->action == LOOM_INLINE_PLAN_ACTION_TRANSFER &&
        loom_callable_body_is_linear(state->module, entry->callee);
    if (body_is_linear && !transfer_availability_valid &&
        !cfg_topology_changed) {
      status = loom_availability_analysis_initialize(
          state->module, state->pass->arena, &transfer_availability);
      if (!iree_status_is_ok(status)) {
        break;
      }
      transfer_availability_valid = true;
    }
    bool changed_cfg_topology = false;
    const loom_availability_analysis_t* availability =
        body_is_linear && transfer_availability_valid ? &transfer_availability
                                                      : NULL;
    status = loom_inline_execute_entry(state, &rewriter, availability, entry,
                                       &changed_cfg_topology);
    if (!iree_status_is_ok(status)) {
      break;
    }
    ++executed_count;
    if (changed_cfg_topology) {
      transfer_availability_valid = false;
      cfg_topology_changed = true;
    }

    if (entry->action == LOOM_INLINE_PLAN_ACTION_CLONE) {
      status = loom_inline_finish_clone(state, execution_symbols,
                                        execution_entries, entry_index, &ready,
                                        &transfer_availability_valid);
    } else {
      loom_inline_execution_symbol_t* target =
          &execution_symbols[entry->target_symbol_id];
      IREE_ASSERT_FALSE(target->transfer_executed);
      target->transfer_executed = true;
      if (target->remaining_outgoing_count == 0) {
        status = loom_inline_complete_execution_entry(
            state, execution_symbols, execution_entries, entry_index, &ready,
            &transfer_availability_valid);
      }
    }
  }
  if (iree_status_is_ok(status) && executed_count != execution_count) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "inline execution dependencies stalled after %u of %u calls",
        (unsigned)executed_count, (unsigned)execution_count);
  }

  const iree_host_size_t removed_version_count =
      state->erased_version_count > 0
          ? loom_function_version_owner_prune_erased(state->version_owner)
          : 0;
  IREE_ASSERT_EQ(removed_version_count, state->erased_version_count);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
