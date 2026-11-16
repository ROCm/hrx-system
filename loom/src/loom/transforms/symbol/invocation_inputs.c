// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/invocation_inputs.h"

#include <string.h>

#include "loom/analysis/callable_inputs.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/materialize.h"

struct loom_invocation_input_read_t {
  // Next local requirement in source discovery order.
  loom_invocation_input_read_t* next;
  // Query replaced only after the complete input plan has been validated.
  loom_op_t* op;
  // Symbol owning this query in its inherited invocation context.
  loom_symbol_id_t symbol;
  // Admitted graph node, assigned during input planning.
  uint32_t node;
  // Semantic query identity, assigned during input planning.
  uint32_t key;
};

typedef struct loom_invocation_node_t {
  // Unique ownership key: ordinary function or exact supplying region.
  const void* identity;
  // Function whose ordinary signature gains inputs; empty for a supplier.
  loom_func_like_t function;
  // Invocation entry supplying inputs; NULL for an ordinary helper.
  loom_region_t* supplier;
  // Source query proving why this node requires an invocation.
  const loom_op_t* witness;
  // First outgoing call in the affected graph, or UINT32_MAX.
  uint32_t first_call;
  // Appended formals or supplied values, in semantic key order.
  loom_value_id_t* values;
  // Number of earlier inputs before each membership word, for O(1) binding.
  uint16_t* word_offsets;
} loom_invocation_node_t;

typedef struct loom_invocation_call_t {
  // Original semantic call; replaced after all interfaces are fixed.
  loom_op_t* op;
  // Admitted caller, which may itself be an invocation supplier.
  uint32_t caller;
  // Ordinary callee whose signature gains inputs.
  uint32_t callee;
  // Next outgoing call from caller, or UINT32_MAX.
  uint32_t next;
} loom_invocation_call_t;

typedef struct loom_invocation_plan_t {
  // Collected local requirements and invocation-owned storage.
  loom_invocation_inputs_t* inputs;
  // Preparation owner, including its diagnostic emitter.
  loom_pass_t* pass;
  // Established relationships retained while call identities are rebound.
  loom_symbol_reference_table_t* references;
  // Representative query for each stable semantic identity.
  struct {
    // Queries borrowed until all supplies are materialized.
    const loom_op_t** values;
    // Number of semantic keys observed in selected helpers.
    iree_host_size_t count;
    // Allocated key capacity.
    iree_host_size_t capacity;
  } keys;
  // Only affected helpers and supplying regions, in admission order.
  struct {
    // Arena-backed node records; growth may move the array.
    loom_invocation_node_t* values;
    // Number of admitted nodes.
    iree_host_size_t count;
    // Allocated node capacity.
    iree_host_size_t capacity;
    // Open-addressed node ordinals plus one; zero is an empty bucket.
    uint32_t* buckets;
    // Power-of-two bucket capacity with at most one-half occupancy.
    iree_host_size_t bucket_count;
  } nodes;
  // Exact semantic callsites already classified by reference discovery.
  struct {
    // Arena-backed call records; repeated edges retain their actual sites.
    loom_invocation_call_t* values;
    // Number of affected callsites.
    iree_host_size_t count;
    // Allocated call capacity.
    iree_host_size_t capacity;
  } calls;
  // Completed input membership for helpers and suppliers.
  loom_callable_inputs_t requirements;
} loom_invocation_plan_t;

static bool loom_invocation_symbol_is_selected(
    const loom_invocation_inputs_t* inputs, loom_symbol_id_t symbol) {
  return symbol != LOOM_SYMBOL_ID_INVALID &&
         (inputs->versions == NULL ||
          loom_target_function_version_snapshot_handle_at(inputs->versions,
                                                          symbol) != NULL);
}

static iree_status_t loom_invocation_observe(
    void* user_data, const loom_op_t* op, loom_symbol_id_t symbol,
    uint8_t root_region, loom_symbol_reference_execution_scope_id_t scope) {
  loom_invocation_inputs_t* inputs = user_data;
  if (scope != 0 || root_region == 0 ||
      !loom_invocation_symbol_is_selected(inputs, symbol) ||
      !loom_func_def_isa(inputs->module->symbols.entries[symbol].defining_op)) {
    return iree_ok_status();
  }
  loom_invocation_input_read_t* read = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(inputs->arena, sizeof(*read), (void**)&read));
  *read =
      (loom_invocation_input_read_t){.op = (loom_op_t*)op, .symbol = symbol};
  *inputs->next_read = read;
  inputs->next_read = &read->next;
  return iree_ok_status();
}

loom_symbol_reference_observer_t loom_invocation_inputs_observer(
    loom_invocation_inputs_t* inputs) {
  inputs->next_read = &inputs->first_read;
  return (loom_symbol_reference_observer_t){
      .invocation_input = loom_invocation_observe,
      .user_data = inputs,
  };
}

static bool loom_invocation_key_equal(const loom_module_t* module,
                                      const loom_op_t* lhs,
                                      const loom_op_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->instance_flags != rhs->instance_flags ||
      !loom_type_equal(
          loom_module_value_type(module, loom_op_const_results(lhs)[0]),
          loom_module_value_type(module, loom_op_const_results(rhs)[0]))) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->attribute_count; ++i) {
    if (!loom_attribute_equal(&loom_op_const_attrs(lhs)[i],
                              &loom_op_const_attrs(rhs)[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_invocation_key(loom_invocation_plan_t* plan,
                                         const loom_op_t* op,
                                         uint32_t* out_key) {
  // Source queries have a finite descriptor-defined vocabulary (query kind
  // and coordinate). This small list contains no callsite or function keys.
  for (iree_host_size_t i = 0; i < plan->keys.count; ++i) {
    if (loom_invocation_key_equal(plan->inputs->module, plan->keys.values[i],
                                  op)) {
      *out_key = (uint32_t)i;
      return iree_ok_status();
    }
  }
  if (plan->keys.count == plan->keys.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->inputs->arena, plan->keys.count, plan->keys.count + 1,
        sizeof(*plan->keys.values), &plan->keys.capacity,
        (void**)&plan->keys.values));
  }
  *out_key = (uint32_t)plan->keys.count;
  plan->keys.values[plan->keys.count++] = op;
  return iree_ok_status();
}

static iree_host_size_t loom_invocation_identity_hash(const void* identity) {
  return loom_structural_hash_finalize(loom_structural_hash_mix_u64(
      loom_structural_hash_initialize(), (uintptr_t)identity));
}

static iree_status_t loom_invocation_admit(loom_invocation_plan_t* plan,
                                           loom_func_like_t function,
                                           loom_region_t* supplier,
                                           const loom_op_t* witness,
                                           uint32_t* out_node) {
  const void* identity = supplier ? (const void*)supplier : function.op;
  if (plan->nodes.count * 2 >= plan->nodes.bucket_count) {
    const iree_host_size_t bucket_count =
        plan->nodes.bucket_count ? plan->nodes.bucket_count * 2 : 16;
    uint32_t* buckets = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->inputs->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
    memset(buckets, 0, bucket_count * sizeof(*buckets));
    for (iree_host_size_t i = 0; i < plan->nodes.count; ++i) {
      iree_host_size_t slot =
          loom_invocation_identity_hash(plan->nodes.values[i].identity) &
          (bucket_count - 1);
      while (buckets[slot]) {
        slot = (slot + 1) & (bucket_count - 1);
      }
      buckets[slot] = (uint32_t)i + 1;
    }
    plan->nodes.buckets = buckets;
    plan->nodes.bucket_count = bucket_count;
  }
  iree_host_size_t slot =
      loom_invocation_identity_hash(identity) & (plan->nodes.bucket_count - 1);
  while (plan->nodes.buckets[slot]) {
    const uint32_t node = plan->nodes.buckets[slot] - 1;
    if (plan->nodes.values[node].identity == identity) {
      *out_node = node;
      return iree_ok_status();
    }
    slot = (slot + 1) & (plan->nodes.bucket_count - 1);
  }
  if (plan->nodes.count == plan->nodes.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->inputs->arena, plan->nodes.count, plan->nodes.count + 1,
        sizeof(*plan->nodes.values), &plan->nodes.capacity,
        (void**)&plan->nodes.values));
  }
  *out_node = (uint32_t)plan->nodes.count++;
  plan->nodes.values[*out_node] = (loom_invocation_node_t){
      .identity = identity,
      .function = function,
      .supplier = supplier,
      .witness = witness,
      .first_call = UINT32_MAX,
  };
  plan->nodes.buckets[slot] = *out_node + 1;
  return iree_ok_status();
}

static iree_status_t loom_invocation_unavailable(loom_invocation_plan_t* plan,
                                                 const loom_op_t* boundary,
                                                 const loom_op_t* witness,
                                                 iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(plan->inputs->module, witness)),
      loom_param_string(reason),
  };
  const loom_diagnostic_related_op_t related = {
      .label = IREE_SV("required invocation input"),
      .module = plan->inputs->module,
      .op = witness,
      .field_ref = loom_diagnostic_field_ref_none(),
  };
  const loom_diagnostic_emission_t emission = {
      .module = plan->inputs->module,
      .op = boundary,
      .error = LOOM_ERR_LOWERING_074,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = &related,
      .related_op_count = 1,
  };
  return iree_diagnostic_emit(plan->pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_invocation_admit_callers(
    loom_invocation_plan_t* plan) {
  loom_module_t* module = plan->inputs->module;
  for (uint32_t callee = 0; callee < plan->nodes.count; ++callee) {
    // Supplier nodes terminate reverse propagation. Queries execute in that
    // region, not in the construction function surrounding the worker.
    if (plan->nodes.values[callee].supplier) {
      continue;
    }
    const loom_func_like_t function = plan->nodes.values[callee].function;
    const loom_op_t* witness = plan->nodes.values[callee].witness;
    if (loom_func_like_visibility(function) ||
        !loom_attr_is_absent(loom_func_like_source_signature(function))) {
      IREE_RETURN_IF_ERROR(loom_invocation_unavailable(
          plan, function.op, witness,
          IREE_SV("a public or native signature cannot acquire invocation "
                  "arguments")));
      continue;
    }
    const loom_symbol_id_t symbol = loom_func_like_callee(function).symbol_id;
    loom_symbol_reference_occurrence_id_t edge_id =
        loom_symbol_reference_table_symbol(plan->references, symbol)
            .first_incoming_occurrence_id;
    while (edge_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* edge =
          loom_symbol_reference_table_occurrence(plan->references, edge_id);
      edge_id = edge->next_incoming_occurrence_id;
      if (!loom_symbol_reference_occurrence_is_dependency(edge) ||
          edge->user_op == function.op) {
        continue;
      }
      loom_op_t* owner =
          edge->source_symbol_id == LOOM_SYMBOL_ID_INVALID
              ? NULL
              : module->symbols.entries[edge->source_symbol_id].defining_op;
      if (loom_func_like_isa(loom_func_like_cast(module, owner)) &&
          !loom_invocation_symbol_is_selected(plan->inputs,
                                              edge->source_symbol_id)) {
        continue;
      }
      const loom_call_like_t call =
          loom_call_like_const_cast(module, edge->user_op);
      if (edge->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
          !loom_call_like_is_direct_semantic(call)) {
        IREE_RETURN_IF_ERROR(loom_invocation_unavailable(
            plan, edge->user_op ? edge->user_op : function.op, witness,
            IREE_SV(
                "this reference does not pass ordinary callable arguments")));
        continue;
      }
      loom_func_like_t caller_function = {0};
      loom_region_t* supplier = NULL;
      if (edge->execution_scope != 0) {
        supplier = (loom_region_t*)plan->references->execution_scopes
                       .values[edge->execution_scope - 1]
                       .region;
      } else if (loom_kernel_def_isa(owner) &&
                 edge->source_root_region_index_plus_one == 2) {
        supplier = loom_kernel_def_body(owner);
      } else if (loom_func_def_isa(owner) &&
                 edge->source_root_region_index_plus_one != 0) {
        caller_function = loom_func_like_cast(module, owner);
      } else {
        IREE_RETURN_IF_ERROR(loom_invocation_unavailable(
            plan, edge->user_op, witness,
            IREE_SV("the calling region has no kernel or worker invocation")));
        continue;
      }
      uint32_t caller;
      IREE_RETURN_IF_ERROR(loom_invocation_admit(plan, caller_function,
                                                 supplier, witness, &caller));
      if (plan->calls.count == plan->calls.capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            plan->inputs->arena, plan->calls.count, plan->calls.count + 1,
            sizeof(*plan->calls.values), &plan->calls.capacity,
            (void**)&plan->calls.values));
      }
      const uint32_t index = (uint32_t)plan->calls.count++;
      plan->calls.values[index] = (loom_invocation_call_t){
          .op = (loom_op_t*)edge->user_op,
          .caller = caller,
          .callee = callee,
          .next = plan->nodes.values[caller].first_call,
      };
      plan->nodes.values[caller].first_call = index;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_invocation_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_invocation_plan_t* plan = user_data;
  iree_status_t status = iree_ok_status();
  for (uint32_t call = plan->nodes.values[node].first_call;
       iree_status_is_ok(status) && call != UINT32_MAX;
       call = plan->calls.values[call].next) {
    status = successor.fn(successor.user_data, plan->calls.values[call].callee);
  }
  return status;
}

static iree_status_t loom_invocation_solve(loom_invocation_plan_t* plan) {
  iree_host_size_t read_count = 0;
  for (loom_invocation_input_read_t* read = plan->inputs->first_read; read;
       read = read->next) {
    ++read_count;
  }
  loom_callable_input_requirement_t* seeds = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->inputs->arena, read_count, sizeof(*seeds), (void**)&seeds));
  read_count = 0;
  for (loom_invocation_input_read_t* read = plan->inputs->first_read; read;
       read = read->next) {
    seeds[read_count++] =
        (loom_callable_input_requirement_t){read->node, read->key};
  }
  const loom_scc_graph_t graph = {
      .node_count = plan->nodes.count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_invocation_visit_successors, plan),
  };
  loom_scc_list_t components;
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, NULL, plan->inputs->arena, &components));
  return loom_callable_inputs_compute(&graph, &components, plan->keys.count,
                                      seeds, read_count, plan->inputs->arena,
                                      &plan->requirements);
}

static iree_status_t loom_invocation_prepare_bindings(
    loom_invocation_plan_t* plan) {
  const iree_host_size_t word_count = plan->requirements.words_per_node;
  for (iree_host_size_t i = 0; i < plan->nodes.count; ++i) {
    loom_invocation_node_t* node = &plan->nodes.values[i];
    const uint64_t* words = plan->requirements.words + i * word_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->inputs->arena, word_count, sizeof(*node->word_offsets),
        (void**)&node->word_offsets));
    uint32_t count = 0;
    for (iree_host_size_t word = 0; word < word_count; ++word) {
      node->word_offsets[word] = (uint16_t)count;
      count += iree_math_count_ones_u64(words[word]);
    }
    uint16_t argument_count = 0;
    if (!node->supplier) {
      loom_func_like_arg_ids(node->function, &argument_count);
    }
    if (count + argument_count > UINT16_MAX) {
      IREE_RETURN_IF_ERROR(loom_invocation_unavailable(
          plan, node->supplier ? node->supplier->parent_op : node->function.op,
          node->witness,
          IREE_SV(
              "the extended signature exceeds callable argument capacity")));
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->inputs->arena, count,
                                                   sizeof(*node->values),
                                                   (void**)&node->values));
  }
  return iree_ok_status();
}

static uint16_t loom_invocation_binding_index(
    const loom_invocation_plan_t* plan, uint32_t node, uint32_t key) {
  const uint64_t word =
      plan->requirements
          .words[node * plan->requirements.words_per_node + key / 64];
  return plan->nodes.values[node].word_offsets[key / 64] +
         iree_math_count_ones_u64(word & ((UINT64_C(1) << (key % 64)) - 1));
}

static iree_status_t loom_invocation_apply_bindings(
    loom_invocation_plan_t* plan, loom_rewriter_t* rewriter) {
  loom_module_t* module = plan->inputs->module;
  for (iree_host_size_t i = 0; i < plan->nodes.count; ++i) {
    loom_invocation_node_t* node = &plan->nodes.values[i];
    loom_block_t* entry = loom_region_entry_block(
        node->supplier ? node->supplier : loom_func_like_body(node->function));
    if (node->supplier) {
      loom_builder_set_before(&rewriter->builder, entry->first_op);
    }
    for (uint32_t key = 0; key < plan->keys.count; ++key) {
      if (!loom_callable_inputs_contains(&plan->requirements, i, key)) {
        continue;
      }
      const loom_op_t* query = plan->keys.values[key];
      loom_value_id_t value;
      if (node->supplier) {
        const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
        loom_ir_remap_t remap;
        IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
            module, module, plan->inputs->arena, &options, &remap));
        loom_op_t* supplied;
        IREE_RETURN_IF_ERROR(
            loom_ir_clone_op(&rewriter->builder, query, &remap, &supplied));
        value = loom_op_const_results(supplied)[0];
      } else {
        IREE_RETURN_IF_ERROR(loom_module_define_value(
            module,
            loom_module_value_type(module, loom_op_const_results(query)[0]),
            &value));
        IREE_RETURN_IF_ERROR(loom_block_add_arg(module, entry, value));
        IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
            rewriter, loom_op_const_results(query)[0], value));
      }
      node->values[loom_invocation_binding_index(plan, (uint32_t)i, key)] =
          value;
    }
  }
  loom_value_id_t* operands = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->inputs->arena, plan->keys.count,
                                sizeof(*operands), (void**)&operands));
  loom_symbol_reference_user_replacement_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->inputs->arena, plan->calls.count,
                                sizeof(*replacements), (void**)&replacements));
  for (iree_host_size_t i = 0; i < plan->calls.count; ++i) {
    const loom_invocation_call_t* call = &plan->calls.values[i];
    uint16_t count = 0;
    for (uint32_t key = 0; key < plan->keys.count; ++key) {
      if (loom_callable_inputs_contains(&plan->requirements, call->callee,
                                        key)) {
        operands[count++] =
            plan->nodes.values[call->caller]
                .values[loom_invocation_binding_index(plan, call->caller, key)];
      }
    }
    loom_op_t* replacement;
    IREE_RETURN_IF_ERROR(loom_callable_append_arguments(
        rewriter, call->op, (loom_value_slice_t){operands, count},
        &replacement));
    replacements[i] = (loom_symbol_reference_user_replacement_t){
        .original = call->op, .replacement = replacement};
  }
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_rebind_users(
      plan->references, replacements, plan->calls.count, plan->inputs->arena));
  for (loom_invocation_input_read_t* read = plan->inputs->first_read; read;
       read = read->next) {
    const loom_value_id_t value =
        plan->nodes.values[read->node]
            .values[loom_invocation_binding_index(plan, read->node, read->key)];
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, read->op, &value, 1));
  }
  return iree_ok_status();
}

iree_status_t loom_invocation_inputs_apply(
    loom_invocation_inputs_t* inputs, loom_pass_t* pass,
    loom_symbol_reference_table_t* references) {
  if (!inputs->first_read) {
    return iree_ok_status();
  }
  // An unbound library may share a generic helper with deferred providers.
  // Target callgraph specialization owns the copies that make those source
  // definitions immutable while selected executable versions acquire inputs.
  if (inputs->versions == NULL && references->template_provider_count != 0) {
    return iree_ok_status();
  }
  loom_invocation_plan_t plan = {
      .inputs = inputs, .pass = pass, .references = references};
  for (loom_invocation_input_read_t* read = inputs->first_read; read;
       read = read->next) {
    IREE_RETURN_IF_ERROR(loom_invocation_key(&plan, read->op, &read->key));
    IREE_RETURN_IF_ERROR(loom_invocation_admit(
        &plan,
        loom_func_like_cast(
            inputs->module,
            inputs->module->symbols.entries[read->symbol].defining_op),
        NULL, read->op, &read->node));
  }
  IREE_RETURN_IF_ERROR(loom_invocation_admit_callers(&plan));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_invocation_solve(&plan));
  IREE_RETURN_IF_ERROR(loom_invocation_prepare_bindings(&plan));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, inputs->module, inputs->arena);
  iree_status_t status = loom_invocation_apply_bindings(&plan, &rewriter);
  loom_rewriter_deinitialize(&rewriter);
  IREE_RETURN_IF_ERROR(status);
  loom_pass_mark_changed(pass);
  return iree_ok_status();
}
