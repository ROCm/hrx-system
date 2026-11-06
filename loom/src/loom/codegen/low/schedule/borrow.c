// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/borrow.h"

#include <string.h>

#include "loom/analysis/ownership.h"
#include "loom/codegen/low/ownership.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"

typedef struct loom_low_borrow_component_t {
  // First borrow producing this pointer-copy component, or UINT32_MAX.
  uint32_t borrow_node;
  // Last use in the borrow's block, including copies but excluding cleanup.
  uint32_t last_use_node;
  // First unsupported lifetime boundary, or UINT32_MAX.
  uint32_t reason_node;
  // Whether control or storage prevents a block-local proof.
  loom_low_borrow_protection_t protection;
} loom_low_borrow_component_t;

static loom_value_ordinal_t loom_low_borrow_root(loom_value_ordinal_t* parents,
                                                 loom_value_ordinal_t value) {
  loom_value_ordinal_t root = value;
  while (parents[root] != root) {
    root = parents[root];
  }
  while (parents[value] != root) {
    const loom_value_ordinal_t next = parents[value];
    parents[value] = root;
    value = next;
  }
  return root;
}

static bool loom_low_borrow_value_is_pointer(
    const loom_low_schedule_table_t* schedule, loom_value_ordinal_t ordinal) {
  const loom_type_t* type = loom_type_register_value_type(
      loom_module_value_type(schedule->module, schedule->value_ids[ordinal]));
  return type && loom_type_is_reference(*type) &&
         loom_type_reference_ownership(*type) ==
             LOOM_REFERENCE_OWNERSHIP_BORROWED;
}

static void loom_low_borrow_protect(loom_low_borrow_component_t* component,
                                    loom_low_borrow_protection_t protection,
                                    uint32_t node) {
  if (component->protection == LOOM_LOW_BORROW_COVERED) {
    component->protection = protection;
    component->reason_node = node;
  }
}

static void loom_low_borrow_note_owner_end(
    const loom_low_schedule_table_t* schedule, uint32_t node_index,
    loom_value_ordinal_t value, const loom_value_ordinal_t* storage_origins,
    uint32_t* owner_ends, uint8_t* ambiguous_ends) {
  const loom_type_t* type = loom_type_register_value_type(
      loom_module_value_type(schedule->module, schedule->value_ids[value]));
  if (!type || !loom_type_is_reference(*type) ||
      loom_type_reference_ownership(*type) != LOOM_REFERENCE_OWNERSHIP_OWNED) {
    return;
  }
  const loom_value_ordinal_t owner =
      storage_origins ? storage_origins[value] : value;
  const uint32_t previous = owner_ends[owner];
  if (previous == UINT32_MAX) {
    owner_ends[owner] = node_index;
  } else if (schedule->nodes[previous].block_index !=
             schedule->nodes[node_index].block_index) {
    ambiguous_ends[owner] = 1;
  } else if (schedule->nodes[node_index].scheduled_ordinal <
             schedule->nodes[previous].scheduled_ordinal) {
    owner_ends[owner] = node_index;
  }
}

iree_status_t loom_low_schedule_borrow_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_value_ordinal_t* storage_origins,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    loom_low_borrow_plan_t* out_plan) {
  *out_plan = (loom_low_borrow_plan_t){0};
  if (schedule->requirements.borrow_count == 0) {
    return iree_ok_status();
  }

  const loom_ownership_effect_provider_t effects =
      loom_low_ownership_effect_provider(schedule->target.descriptor_set);

  const loom_value_ordinal_t count = schedule->value_count;
  loom_value_ordinal_t* parents = NULL;
  uint32_t* owner_ends = NULL;
  uint8_t* ambiguous_ends = NULL;
  uint8_t* shared_storage = NULL;
  loom_low_borrow_component_t* components = NULL;
  loom_low_borrow_protection_t* values = NULL;
  loom_low_borrow_decision_t* decisions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*values),
                                                 (void**)&values));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, schedule->requirements.borrow_count,
                                sizeof(*decisions), (void**)&decisions));
  memset(values, 0, count * sizeof(*values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*parents), (void**)&parents));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*owner_ends), (void**)&owner_ends));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*ambiguous_ends), (void**)&ambiguous_ends));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*shared_storage), (void**)&shared_storage));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*components), (void**)&components));
  memset(ambiguous_ends, 0, count * sizeof(*ambiguous_ends));
  memset(shared_storage, 0, count * sizeof(*shared_storage));
  for (loom_value_ordinal_t i = 0; i < count; ++i) {
    parents[i] = storage_origins ? storage_origins[i] : i;
    owner_ends[i] = UINT32_MAX;
    components[i] =
        (loom_low_borrow_component_t){.borrow_node = UINT32_MAX,
                                      .last_use_node = UINT32_MAX,
                                      .reason_node = UINT32_MAX,
                                      .protection = LOOM_LOW_BORROW_COVERED};
    if (parents[i] != i) {
      shared_storage[parents[i]] = 1;
    }
  }

  // Independent pointer copies share provenance, never a release obligation.
  // This construction is independent of block layout and scheduled order.
  uint32_t decision_count = 0;
  for (uint32_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (loom_low_borrow_isa(node->op)) {
      decisions[decision_count++].node_index = i;
    }
    const loom_value_ordinal_t* results =
        loom_low_schedule_node_const_result_ordinals(node);
    const loom_value_ordinal_t* operands =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t j = 0; j < node->result_count; ++j) {
      loom_ownership_result_effect_t effect = {0};
      if (!loom_ownership_result_effect_at(schedule->module, &effects, node->op,
                                           j, &effect)) {
        continue;
      }
      if (effect.effect == LOOM_RESULT_OWNERSHIP_MOVED ||
          effect.effect == LOOM_RESULT_OWNERSHIP_TIED) {
        loom_low_borrow_note_owner_end(
            schedule, i, operands[effect.source_operand_index], storage_origins,
            owner_ends, ambiguous_ends);
      }
      if ((effect.effect != LOOM_RESULT_OWNERSHIP_COPIED &&
           effect.effect != LOOM_RESULT_OWNERSHIP_MOVED) ||
          !loom_low_borrow_value_is_pointer(schedule, results[j])) {
        continue;
      }
      const loom_value_ordinal_t destination =
          loom_low_borrow_root(parents, results[j]);
      const loom_value_ordinal_t source =
          loom_low_borrow_root(parents, operands[effect.source_operand_index]);
      parents[destination] = source;
    }
  }
  for (loom_value_ordinal_t i = 0; i < count; ++i) {
    parents[i] = loom_low_borrow_root(parents, i);
  }
  for (loom_value_ordinal_t i = 0; i < count; ++i) {
    if (shared_storage[i]) {
      loom_low_borrow_protect(&components[parents[i]],
                              LOOM_LOW_BORROW_SHARED_STORAGE, UINT32_MAX);
    }
  }
  for (uint32_t j = 0; j < decision_count; ++j) {
    const uint32_t i = decisions[j].node_index;
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    loom_low_borrow_component_t* component =
        &components[parents[loom_low_schedule_node_const_result_ordinals(
            node)[0]]];
    if (component->borrow_node != UINT32_MAX) {
      loom_low_borrow_protect(component, LOOM_LOW_BORROW_SHARED_STORAGE, i);
    } else {
      component->borrow_node = i;
      component->last_use_node = i;
    }
  }

  for (uint32_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    const loom_value_ordinal_t* operands =
        loom_low_schedule_node_const_operand_ordinals(node);
    const bool call = loom_low_func_call_isa(node->op);
    const bool escapes = loom_direct_branch_isa(node->op) ||
                         loom_low_return_isa(node->op) ||
                         loom_low_storage_store_isa(node->op);
    for (uint16_t j = 0; j < node->operand_count; ++j) {
      loom_ownership_operand_effect_t effect = {0};
      const bool has_effect = loom_ownership_operand_effect_at(
          schedule->module, &effects, node->op, j, &effect);
      if (call || escapes ||
          (has_effect && effect.effect != LOOM_OPERAND_OWNERSHIP_BORROW &&
           effect.effect != LOOM_OPERAND_OWNERSHIP_RETAIN)) {
        loom_low_borrow_note_owner_end(schedule, i, operands[j],
                                       storage_origins, owner_ends,
                                       ambiguous_ends);
      }
      loom_low_borrow_component_t* component =
          &components[parents[operands[j]]];
      if (component->borrow_node == UINT32_MAX ||
          loom_low_release_isa(node->op)) {
        continue;
      }
      const loom_low_schedule_node_t* origin =
          &schedule->nodes[component->borrow_node];
      if (node->block_index != origin->block_index) {
        loom_low_borrow_protect(component, LOOM_LOW_BORROW_CONTROL_FLOW, i);
      } else if (node->scheduled_ordinal >
                 schedule->nodes[component->last_use_node].scheduled_ordinal) {
        component->last_use_node = i;
      }
      if (escapes) {
        loom_low_borrow_protect(component, LOOM_LOW_BORROW_CAPTURED, i);
      }
    }
  }

  for (uint32_t j = 0; j < decision_count; ++j) {
    const uint32_t i = decisions[j].node_index;
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    const loom_value_ordinal_t result =
        loom_low_schedule_node_const_result_ordinals(node)[0];
    const loom_value_ordinal_t source =
        loom_low_schedule_node_const_operand_ordinals(node)[0];
    const loom_value_ordinal_t owner =
        storage_origins ? storage_origins[source] : source;
    loom_low_borrow_component_t component = components[parents[result]];
    const uint32_t end = owner_ends[owner];
    if (shared_storage[owner]) {
      loom_low_borrow_protect(&component, LOOM_LOW_BORROW_SHARED_STORAGE, i);
    } else if (ambiguous_ends[owner]) {
      loom_low_borrow_protect(&component, LOOM_LOW_BORROW_CONTROL_FLOW, end);
    } else if (end != UINT32_MAX &&
               schedule->nodes[end].block_index == node->block_index &&
               schedule->nodes[end].scheduled_ordinal <=
                   schedule->nodes[component.last_use_node].scheduled_ordinal) {
      loom_low_borrow_protect(&component, LOOM_LOW_BORROW_OWNER_RETIRED, end);
    }
    values[result] = component.protection;
    decisions[j] =
        (loom_low_borrow_decision_t){.node_index = i,
                                     .last_use_node = component.last_use_node,
                                     .reason_node = component.reason_node,
                                     .protection = component.protection};
  }
  *out_plan = (loom_low_borrow_plan_t){
      .values = values, .decisions = decisions, .count = decision_count};
  return iree_ok_status();
}

iree_status_t loom_low_schedule_borrow_diagnostics_emit(
    const loom_low_schedule_table_t* schedule,
    iree_diagnostic_emitter_t emitter) {
  static const iree_string_view_t reasons[] = {
      [LOOM_LOW_BORROW_OWNER_RETIRED] =
          IREE_SVL("the owner may retire before the pointer's last use"),
      [LOOM_LOW_BORROW_CONTROL_FLOW] =
          IREE_SVL("coverage crosses a control-flow boundary"),
      [LOOM_LOW_BORROW_CAPTURED] =
          IREE_SVL("the pointer escapes this block through control or storage"),
      [LOOM_LOW_BORROW_SHARED_STORAGE] =
          IREE_SVL("shared carrier storage does not prove payload lifetime"),
  };
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < schedule->borrows.count && iree_status_is_ok(status);
       ++i) {
    const loom_low_borrow_decision_t* decision =
        &schedule->borrows.decisions[i];
    if (decision->protection == LOOM_LOW_BORROW_COVERED) {
      continue;
    }
    loom_diagnostic_related_op_t related[2];
    iree_host_size_t related_count = 0;
    if (decision->reason_node != UINT32_MAX) {
      related[related_count++] = (loom_diagnostic_related_op_t){
          .label = IREE_SV("owner coverage ends or becomes unproved here"),
          .module = schedule->module,
          .op = schedule->nodes[decision->reason_node].op,
      };
    }
    if (decision->protection == LOOM_LOW_BORROW_OWNER_RETIRED) {
      related[related_count++] = (loom_diagnostic_related_op_t){
          .label = IREE_SV("final use of the pointer-copy chain"),
          .module = schedule->module,
          .op = schedule->nodes[decision->last_use_node].op,
      };
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(reasons[decision->protection]),
    };
    const loom_diagnostic_emission_t emission = {
        .module = schedule->module,
        .op = schedule->nodes[decision->node_index].op,
        .error = LOOM_ERR_DOMINANCE_021,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related,
        .related_op_count = related_count,
    };
    status = iree_diagnostic_emit(emitter, &emission);
  }
  return status;
}
