// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/rematerialization.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/registers.h"
#include "loom/util/adaptive_sort.h"

static iree_status_t loom_low_rematerialization_reserve_values(
    iree_host_size_t required_bit_count, iree_arena_allocator_t* arena,
    iree_bitmap_t* values) {
  const iree_host_size_t required_word_count =
      iree_bitmap_calculate_words(required_bit_count);
  iree_host_size_t word_capacity =
      iree_bitmap_calculate_words(values->bit_count);
  if (required_word_count > word_capacity) {
    const iree_host_size_t old_word_count = word_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, old_word_count, required_word_count, sizeof(*values->words),
        &word_capacity, (void**)&values->words));
    memset(values->words + old_word_count, 0,
           (word_capacity - old_word_count) * sizeof(*values->words));
    values->bit_count = word_capacity * IREE_BITMAP_BITS_PER_WORD;
  }
  return iree_ok_status();
}

static loom_low_value_rematerialization_result_t
loom_low_value_rematerialization_result_empty(void) {
  return (loom_low_value_rematerialization_result_t){
      .value_id = LOOM_VALUE_ID_INVALID,
  };
}

static void loom_low_rematerialization_accumulate_result(
    const loom_low_value_rematerialization_result_t* result,
    loom_low_rematerialization_batch_result_t* batch) {
  if (result->rewritten_operand_count != 0 ||
      result->retained_placement_count != 0) {
    ++batch->repaired_value_count;
  }
  batch->cloned_packet_count += result->cloned_packet_count;
  batch->rewritten_operand_count += result->rewritten_operand_count;
  batch->retained_placement_count += result->retained_placement_count;
}

static bool loom_low_allocation_failure_is_rematerializable_pressure(
    const loom_low_allocation_failure_t* failure) {
  return loom_low_allocation_failure_is_present(failure) &&
         (iree_string_view_equal(failure->failure_code,
                                 IREE_SV("unspillable-register-exhausted")) ||
          iree_string_view_equal(failure->failure_code,
                                 IREE_SV("spill-traffic-register-exhausted")));
}

static bool loom_low_descriptor_packet_kind_may_rematerialize(
    loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ||
         kind == LOOM_LOW_DESCRIPTOR_PACKET_CONST;
}

typedef struct loom_low_rematerialization_recipe_t {
  // Producer operations in reverse dependency and consumption order.
  loom_op_t** packets;
  // Number of distinct producers in the recipe.
  iree_host_size_t packet_count;
  // Whether cloning the recipe would add a use of a register-required value.
  bool has_required_external_input;
  // Whether the recipe followed an optional descriptor source edge.
  bool followed_explicit_source;
} loom_low_rematerialization_recipe_t;

typedef enum loom_low_rematerialization_action_e {
  LOOM_LOW_REMATERIALIZATION_ACTION_NONE = 0,
  LOOM_LOW_REMATERIALIZATION_ACTION_RETAIN_PLACEMENT,
  LOOM_LOW_REMATERIALIZATION_ACTION_CLONE,
} loom_low_rematerialization_action_t;

// One prepared value repair. Its recipe and captured uses remain valid until
// application rewrites IR; placement retention leaves the snapshot unchanged.
typedef struct loom_low_rematerialization_plan_t {
  // Complete producer recipe to recreate for each user.
  loom_low_rematerialization_recipe_t recipe;
  // Existing uses captured once before deciding whether mutation is required.
  const loom_use_t* uses;
  // Number of captured operand uses.
  uint32_t use_count;
  // Repair selected by recipe eligibility and source/scheduled placement.
  loom_low_rematerialization_action_t action;
} loom_low_rematerialization_plan_t;

static bool loom_low_rematerialization_packet_is_eligible(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, loom_op_t** out_defining_op) {
  *out_defining_op = NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, value_id)) {
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_low_register_type_resolver_t register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          target->descriptor_set);
  if (loom_low_register_type_resolver_has_class_flags(
          &register_type_resolver, value_type,
          LOOM_LOW_REG_CLASS_FLAG_REFERENCE)) {
    return false;
  }

  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->result_count != 1 || defining_op->region_count != 0 ||
      defining_op->successor_count != 0 ||
      iree_any_bit_set(loom_op_effective_traits(module, defining_op),
                       LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }

  const uint16_t result_index = loom_value_def_index(value);
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(target->descriptor_set, defining_op,
                                        &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    // Structural composition is repeatable when its inputs can be recreated
    // privately. Recipe planning establishes that ownership boundary.
    if (!iree_any_bit_set(loom_op_vtable(module, defining_op)->vtable_flags,
                          LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION)) {
      return false;
    }
  } else if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
             !loom_low_descriptor_result_can_rematerialize(
                 target->descriptor_set, packet.descriptor, result_index)) {
    return false;
  }
  *out_defining_op = defining_op;
  return true;
}

static loom_value_id_t loom_low_rematerialization_tied_source(
    const loom_op_t* op) {
  const loom_tied_result_t tied = loom_op_tied_results(op)[0];
  return loom_op_const_operands(op)[tied.operand_index];
}

enum loom_low_rematerialization_value_flag_bits_e {
  LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED = 1u << 0,
  LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE = 1u << 1,
  LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED = 1u << 2,
  LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED = 1u << 3,
  LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED = 1u << 4,
};

// Immutable producer facts for one consumer-first repair batch. Rewrites may
// change uses, but cannot change an unprocessed producer's original inputs.
// New clones never join this domain; they already have private placement.
typedef struct loom_low_rematerialization_facts_t {
  // Module whose ordinal scratch is leased until the batch finishes.
  loom_module_t* module;
  // Target descriptor contract for explicit producer recipe edges.
  const loom_low_resolved_target_t* target;
  // Borrowed original function-local value IDs, also used to release the lease.
  const loom_value_id_t* value_ids;
  // Number of entries in the borrowed local value domain.
  iree_host_size_t value_count;
  // Allocation's retained definition order, or NULL when repairing a schedule
  // that has not reached allocation.
  const loom_liveness_analysis_t* liveness;
  // Retained CFG preorder for ordering producers across blocks.
  const loom_cfg_graph_t* cfg_graph;
  // Function body owning the original value definitions.
  const loom_region_t* region;
  // Eligibility, closed-materialization, and recipe-membership bits per
  // ordinal.
  uint8_t* value_flags;
  // Whether this batch replaces spill reloads with constant initialization.
  bool reconstruct_constants;
} loom_low_rematerialization_facts_t;

typedef struct loom_low_rematerialization_closure_cursor_t {
  // Local value whose structural inputs are being visited.
  loom_value_ordinal_t ordinal;
  // Next operand to visit before completing this producer.
  uint16_t next_operand;
  // Whether every input visited so far belongs to a closed materialization.
  bool is_closed;
} loom_low_rematerialization_closure_cursor_t;

static uint16_t loom_low_rematerialization_source_operand(
    const loom_low_rematerialization_facts_t* facts, const loom_op_t* op) {
  if (op->tied_result_count != 0) {
    return LOOM_LOW_ID_NONE;
  }
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(facts->target->descriptor_set, op,
                                        &packet);
  return packet.descriptor != NULL
             ? loom_low_descriptor_result_rematerialization_source_operand(
                   facts->target->descriptor_set, packet.descriptor, 0)
             : LOOM_LOW_ID_NONE;
}

static loom_low_rematerialization_closure_cursor_t
loom_low_rematerialization_closure_cursor_initialize(
    loom_value_ordinal_t ordinal) {
  return (loom_low_rematerialization_closure_cursor_t){
      .ordinal = ordinal,
      .is_closed = true,
  };
}

static void loom_low_rematerialization_facts_release(
    loom_low_rematerialization_facts_t* facts) {
  for (iree_host_size_t i = 0; i < facts->value_count; ++i) {
    loom_module_value_ordinal_scratch_clear(facts->module, facts->value_ids[i]);
  }
  loom_module_value_ordinal_scratch_release(facts->module);
}

static loom_value_ordinal_t loom_low_rematerialization_ordinal(
    const loom_low_rematerialization_facts_t* facts, loom_value_id_t value_id) {
  return loom_module_value_ordinal_scratch_lookup(facts->module, value_id);
}

static iree_status_t loom_low_rematerialization_classify(
    loom_low_rematerialization_facts_t* facts,
    const loom_low_resolved_target_t* target, iree_arena_allocator_t* arena) {
  loom_module_t* module = facts->module;
  for (loom_value_ordinal_t ordinal = 0; ordinal < facts->value_count;
       ++ordinal) {
    const loom_value_id_t value_id = facts->value_ids[ordinal];
    uint8_t flags =
        loom_value_consuming_use_count(loom_module_value(module, value_id)) != 0
            ? LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED
            : 0;
    loom_op_t* op = NULL;
    bool eligible = loom_low_rematerialization_packet_is_eligible(
        module, target, value_id, &op);
    if (eligible && facts->liveness != NULL) {
      const uint32_t interval_index =
          facts->liveness->value_interval_indices[ordinal];
      eligible = interval_index != UINT32_MAX &&
                 facts->liveness->intervals[interval_index].definition_point !=
                     UINT32_MAX;
    } else if (eligible) {
      eligible = op->parent_block->parent_region == facts->region;
    }
    if (!eligible) {
      flags |= LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED;
    } else {
      flags |= LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE;
      if (op->operand_count == 0) {
        flags |= LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED |
                 LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED;
      } else if (!iree_any_bit_set(loom_op_vtable(module, op)->vtable_flags,
                                   LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION)) {
        // Late reconstruction initializes constants and composes their storage.
        // Replaying arithmetic retains the ordinary execution-domain contract.
        flags |= LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED;
      }
    }
    facts->value_flags[ordinal] = flags;
  }

  if (!facts->reconstruct_constants) {
    return iree_ok_status();
  }

  // Memoized postorder handles inlined CFG layouts whose definition list is
  // not dominance ordered. Each eligible producer and edge is visited once.
  loom_low_rematerialization_closure_cursor_t* stack = NULL;
  iree_host_size_t capacity = 0;
  for (loom_value_ordinal_t root = 0; root < facts->value_count; ++root) {
    if (iree_any_bit_set(facts->value_flags[root],
                         LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, 0, 1, sizeof(*stack),
                                               &capacity, (void**)&stack));
    iree_host_size_t depth = 1;
    stack[0] = loom_low_rematerialization_closure_cursor_initialize(root);
    while (depth != 0) {
      loom_low_rematerialization_closure_cursor_t* cursor = &stack[depth - 1];
      const loom_op_t* op = loom_value_def_op(
          loom_module_value(module, facts->value_ids[cursor->ordinal]));
      if (!cursor->is_closed || cursor->next_operand == op->operand_count) {
        facts->value_flags[cursor->ordinal] |=
            LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED |
            (cursor->is_closed ? LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED : 0);
        --depth;
        continue;
      }
      const loom_value_id_t input =
          loom_op_const_operands(op)[cursor->next_operand];
      const loom_value_ordinal_t input_ordinal =
          loom_low_rematerialization_ordinal(facts, input);
      const uint8_t input_flags = facts->value_flags[input_ordinal];
      if (loom_type_kind(loom_module_value_type(module, input)) !=
          LOOM_TYPE_REGISTER) {
        cursor->is_closed = !iree_any_bit_set(
            input_flags, LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED);
      } else if (loom_low_register_type_class_id(
                     loom_module_value_type(module, input)) !=
                 loom_low_register_type_class_id(loom_module_value_type(
                     module, facts->value_ids[cursor->ordinal]))) {
        cursor->is_closed = false;
      } else if (loom_low_register_type_unit_count(
                     loom_module_value_type(module, input)) >
                 loom_low_register_type_unit_count(loom_module_value_type(
                     module, facts->value_ids[cursor->ordinal]))) {
        // A narrow projection must not reconstruct its wider input. Retain
        // that input as a capture instead of multiplying its materialization.
        cursor->is_closed = false;
      } else if (!iree_any_bit_set(
                     input_flags,
                     LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED)) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, depth, depth + 1,
                                                   sizeof(*stack), &capacity,
                                                   (void**)&stack));
        stack[depth++] =
            loom_low_rematerialization_closure_cursor_initialize(input_ordinal);
        continue;
      } else {
        cursor->is_closed = iree_any_bit_set(
            input_flags, LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED);
      }
      ++cursor->next_operand;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_rematerialization_facts_initialize(
    loom_low_rematerialization_facts_t* facts,
    const loom_low_resolved_target_t* target, iree_arena_allocator_t* arena) {
  facts->target = target;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, facts->value_count,
                                                 sizeof(*facts->value_flags),
                                                 (void**)&facts->value_flags));
  loom_module_value_ordinal_scratch_acquire(facts->module);
  for (loom_value_ordinal_t i = 0; i < facts->value_count; ++i) {
    loom_module_value_ordinal_scratch_set(facts->module, facts->value_ids[i],
                                          i);
  }
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  iree_status_t status =
      loom_low_rematerialization_classify(facts, target, arena);
  iree_arena_checkpoint_restore(&checkpoint);
  if (!iree_status_is_ok(status)) {
    loom_low_rematerialization_facts_release(facts);
  }
  return status;
}

typedef struct loom_low_rematerialization_packet_order_t {
  // Original producer retained by the recipe.
  loom_op_t* packet;
  // Original scheduled definition point, or source ordinal for schedule repair.
  uint64_t definition_point;
  // Retained CFG preorder; strict dominators have smaller preorders.
  uint16_t block_preorder;
} loom_low_rematerialization_packet_order_t;

static bool loom_low_rematerialization_packet_less(
    const loom_low_rematerialization_packet_order_t* lhs,
    const loom_low_rematerialization_packet_order_t* rhs) {
  return lhs->block_preorder != rhs->block_preorder
             ? lhs->block_preorder > rhs->block_preorder
             : lhs->definition_point > rhs->definition_point;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_rematerialization_packet_sort,
                          loom_low_rematerialization_packet_order_t,
                          loom_low_rematerialization_packet_less)

static iree_status_t loom_low_rematerialization_order_consumption(
    const loom_low_rematerialization_facts_t* facts, loom_op_t** packets,
    iree_host_size_t packet_count, iree_arena_allocator_t* arena) {
  loom_low_rematerialization_packet_order_t* order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, packet_count, sizeof(*order), (void**)&order));
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    const loom_op_t* op = packets[i];
    uint64_t point = op->block_ordinal;
    uint16_t block_index = op->parent_block->region_index;
    if (facts->liveness != NULL) {
      const loom_value_ordinal_t ordinal = loom_low_rematerialization_ordinal(
          facts, loom_op_const_results(op)[0]);
      point = facts->liveness
                  ->intervals[facts->liveness->value_interval_indices[ordinal]]
                  .definition_point;
      block_index =
          loom_liveness_block_index_at_point(facts->liveness, (uint32_t)point);
    }
    order[i] = (loom_low_rematerialization_packet_order_t){
        .packet = packets[i],
        .definition_point = point,
        .block_preorder = facts->cfg_graph->blocks != NULL
                              ? facts->cfg_graph->blocks[block_index].preorder
                              : 0,
    };
  }
  // Dataflow alone can put a copy after a move that consumes their shared
  // source. Retained program order preserves those read-before-consume edges.
  loom_low_rematerialization_packet_sort(order, packet_count);
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    packets[i] = order[i].packet;
  }
  return iree_ok_status();
}

typedef struct loom_low_rematerialization_recipe_cursor_t {
  // Original local value being included in the producer recipe.
  loom_value_ordinal_t ordinal;
  // Next packet operand to classify as a producer or external capture.
  uint16_t next_operand;
  // Optional descriptor-declared producer operand, or LOOM_LOW_ID_NONE.
  uint16_t source_operand_index;
} loom_low_rematerialization_recipe_cursor_t;

static loom_low_rematerialization_recipe_cursor_t
loom_low_rematerialization_recipe_cursor_initialize(
    const loom_low_rematerialization_facts_t* facts,
    loom_value_ordinal_t ordinal, bool follow_explicit_sources) {
  uint16_t source_operand_index = LOOM_LOW_ID_NONE;
  const loom_value_t* value =
      loom_module_value(facts->module, facts->value_ids[ordinal]);
  if (follow_explicit_sources &&
      iree_any_bit_set(facts->value_flags[ordinal],
                       LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE)) {
    source_operand_index = loom_low_rematerialization_source_operand(
        facts, loom_value_def_op(value));
  }
  return (loom_low_rematerialization_recipe_cursor_t){
      .ordinal = ordinal,
      .source_operand_index = source_operand_index,
  };
}

static iree_status_t loom_low_rematerialization_plan_recipe(
    loom_low_rematerialization_facts_t* facts, loom_value_id_t value_id,
    bool rebuild_composition, bool follow_explicit_sources,
    const iree_bitmap_t* required_register_values,
    iree_arena_allocator_t* arena,
    loom_low_rematerialization_recipe_t* out_recipe) {
  *out_recipe = (loom_low_rematerialization_recipe_t){0};
  loom_module_t* module = facts->module;
  loom_op_t** packets = NULL;
  iree_host_size_t packet_count = 0;
  iree_host_size_t packet_capacity = 0;
  bool has_required_external_input = false;
  bool has_consumption = false;
  bool followed_explicit_source = false;
  bool eligible = true;
  const loom_value_ordinal_t root =
      loom_low_rematerialization_ordinal(facts, value_id);
  loom_low_rematerialization_recipe_cursor_t* stack = NULL;
  iree_host_size_t stack_capacity = 0;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, 0, 1, sizeof(*stack),
                                             &stack_capacity, (void**)&stack));
  iree_host_size_t depth = 1;
  stack[0] = loom_low_rematerialization_recipe_cursor_initialize(
      facts, root, follow_explicit_sources);
  facts->value_flags[root] |= LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED;
  iree_status_t status = iree_ok_status();
  while (depth != 0 && eligible && iree_status_is_ok(status)) {
    loom_low_rematerialization_recipe_cursor_t* cursor = &stack[depth - 1];
    const uint8_t flags = facts->value_flags[cursor->ordinal];
    eligible =
        iree_any_bit_set(flags, LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE);
    if (!eligible) {
      break;
    }
    loom_op_t* op = loom_value_def_op(
        loom_module_value(module, facts->value_ids[cursor->ordinal]));
    if (!(rebuild_composition &&
          iree_any_bit_set(flags, LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED)) &&
        iree_any_bit_set(loom_op_vtable(module, op)->vtable_flags,
                         LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION) &&
        !iree_any_bit_set(loom_op_effective_traits(module, op),
                          LOOM_TRAIT_PURE)) {
      eligible = false;
    }
    if (!eligible) {
      break;
    }
    if (cursor->next_operand < op->operand_count) {
      const uint16_t operand_index = cursor->next_operand++;
      const loom_value_id_t input = loom_op_const_operands(op)[operand_index];
      const loom_value_ordinal_t ordinal =
          loom_low_rematerialization_ordinal(facts, input);
      const uint8_t input_flags = facts->value_flags[ordinal];
      const bool explicit_source =
          operand_index == cursor->source_operand_index &&
          iree_any_bit_set(input_flags,
                           LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE);
      followed_explicit_source |= explicit_source;
      const bool owned =
          explicit_source ||
          (op->tied_result_count != 0 &&
           input == loom_low_rematerialization_tied_source(op)) ||
          (rebuild_composition &&
           iree_any_bit_set(flags, LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED) &&
           iree_any_bit_set(loom_op_vtable(module, op)->vtable_flags,
                            LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION) &&
           loom_type_kind(loom_module_value_type(module, input)) ==
               LOOM_TYPE_REGISTER &&
           iree_any_bit_set(input_flags,
                            LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED));
      if (owned) {
        if (!iree_any_bit_set(input_flags,
                              LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED)) {
          status =
              iree_arena_grow_array(arena, depth, depth + 1, sizeof(*stack),
                                    &stack_capacity, (void**)&stack);
          if (iree_status_is_ok(status)) {
            facts->value_flags[ordinal] |=
                LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED;
            stack[depth++] =
                loom_low_rematerialization_recipe_cursor_initialize(
                    facts, ordinal, follow_explicit_sources);
          }
        }
      }
      continue;
    }
    has_consumption |=
        iree_any_bit_set(flags, LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED);
    status = iree_arena_grow_array(arena, packet_count, packet_count + 1,
                                   sizeof(*packets), &packet_capacity,
                                   (void**)&packets);
    if (iree_status_is_ok(status)) {
      packets[packet_count++] = op;
    }
    --depth;
  }
  for (iree_host_size_t i = 0, j = packet_count; i < j / 2; ++i) {
    loom_op_t* packet = packets[i];
    packets[i] = packets[j - i - 1];
    packets[j - i - 1] = packet;
  }
  // Classify captures after collecting the complete graph. An operand may
  // name a non-immediate ancestor discovered through another producer edge.
  for (iree_host_size_t i = 0;
       i < packet_count && eligible && iree_status_is_ok(status); ++i) {
    const loom_op_t* op = packets[i];
    for (uint16_t j = 0; j < op->operand_count && eligible; ++j) {
      const loom_value_id_t input = loom_op_const_operands(op)[j];
      const loom_value_ordinal_t ordinal =
          loom_low_rematerialization_ordinal(facts, input);
      if (iree_any_bit_set(facts->value_flags[ordinal],
                           LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED)) {
        continue;
      }
      eligible =
          loom_value_consuming_use_count(loom_module_value(module, input)) == 0;
      has_required_external_input |=
          required_register_values != NULL &&
          input < required_register_values->bit_count &&
          iree_bitmap_test(*required_register_values, input);
    }
  }
  if (iree_status_is_ok(status) && eligible && has_consumption) {
    status = loom_low_rematerialization_order_consumption(facts, packets,
                                                          packet_count, arena);
  }
  for (iree_host_size_t i = 0; i < depth; ++i) {
    facts->value_flags[stack[i].ordinal] &=
        (uint8_t)~LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED;
  }
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    const loom_value_ordinal_t ordinal = loom_low_rematerialization_ordinal(
        facts, loom_op_const_results(packets[i])[0]);
    facts->value_flags[ordinal] &=
        (uint8_t)~LOOM_LOW_REMATERIALIZATION_VALUE_COLLECTED;
  }
  if (iree_status_is_ok(status) && eligible) {
    *out_recipe = (loom_low_rematerialization_recipe_t){
        .packets = packets,
        .packet_count = packet_count,
        .has_required_external_input = has_required_external_input,
    };
  }
  out_recipe->followed_explicit_source = followed_explicit_source;
  return status;
}

static bool loom_low_rematerialization_use_is_eligible(
    loom_value_id_t value_id, const loom_op_t* defining_op, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  if (user_op == NULL || user_op == defining_op ||
      iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) ||
      user_op->parent_block == NULL ||
      operand_index >= user_op->operand_count) {
    return false;
  }
  if (loom_op_operands(user_op)[operand_index] != value_id) {
    return false;
  }
  if (user_op->parent_block == defining_op->parent_block &&
      user_op->block_ordinal <= defining_op->block_ordinal) {
    return false;
  }
  return true;
}

static bool loom_low_rematerialization_use_shortens_live_range(
    const loom_op_t* defining_op, loom_use_t use) {
  const loom_op_t* user_op = loom_use_user_op(use);
  // An already-adjacent producer needs placement retention, not another clone.
  return user_op != NULL &&
         (user_op->parent_block != defining_op->parent_block ||
          user_op->prev_op != defining_op);
}

static uint32_t loom_low_rematerialization_use_unit_count(
    const loom_module_t* module, loom_use_t use, uint32_t full_unit_count) {
  const loom_op_t* user = loom_use_user_op(use);
  if (user->result_count == 1 &&
      iree_any_bit_set(loom_op_vtable(module, user)->vtable_flags,
                       LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION)) {
    loom_low_storage_relation_iterator_t iterator;
    loom_low_storage_relation_iterator_initialize(module, user, &iterator);
    loom_low_storage_relation_t relation;
    while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
      if (relation.source_operand_index == loom_use_operand_index(use)) {
        return relation.unit_count;
      }
    }
  }
  return full_unit_count;
}

// Returns true when two blocks have the same repeated execution domain.
// Same-block placement is exact without loop facts. Distinct blocks require
// the same innermost canonical loop, or both must be outside every loop.
static bool loom_low_rematerialization_blocks_share_execution_domain(
    const loom_low_allocation_table_t* allocation,
    const loom_block_t* reference_block, const loom_block_t* candidate_block) {
  if (reference_block == candidate_block) {
    return true;
  }
  const loom_cfg_graph_t* graph = &allocation->cfg_graph;
  const loom_cfg_loop_forest_t* forest = &allocation->loop_forest;
  if (forest->innermost_loop_indices == NULL) {
    return false;
  }
  const iree_host_size_t reference_block_index =
      loom_cfg_graph_block_index(graph, reference_block);
  const iree_host_size_t candidate_block_index =
      loom_cfg_graph_block_index(graph, candidate_block);
  if (reference_block_index >= graph->block_count ||
      candidate_block_index >= graph->block_count) {
    return false;
  }
  const uint32_t reference_loop_index =
      forest->innermost_loop_indices[reference_block_index];
  if (reference_loop_index !=
      forest->innermost_loop_indices[candidate_block_index]) {
    return false;
  }
  return reference_loop_index == LOOM_CFG_LOOP_NONE ||
         forest->intervals[reference_loop_index].is_canonical;
}

static bool loom_low_rematerialization_uses_share_execution_domain(
    const loom_low_allocation_table_t* allocation, const loom_op_t* defining_op,
    const loom_use_t* uses, uint32_t use_count) {
  if (allocation == NULL || !allocation->cfg_graph.has_cycles) {
    return true;
  }
  for (uint32_t i = 0; i < use_count; ++i) {
    if (!loom_low_rematerialization_blocks_share_execution_domain(
            allocation, defining_op->parent_block,
            loom_use_user_op(uses[i])->parent_block)) {
      return false;
    }
  }
  return true;
}

// Returns true when cloning |recipe| would keep every producer and user in the
// selected packet's repeated execution domain. Acyclic control flow executes
// each block at most once. Missing or noncanonical loop facts cannot justify
// multiplying even one packet across loop iterations; allocation can retain
// the original value and use ordinary spill traffic instead.
static bool loom_low_rematerialization_preserves_execution_domain(
    const loom_low_allocation_table_t* allocation,
    const loom_low_rematerialization_recipe_t* recipe, const loom_use_t* uses,
    uint32_t use_count) {
  if (recipe->packet_count == 0) {
    return false;
  }
  if (!loom_low_rematerialization_uses_share_execution_domain(
          allocation, recipe->packets[0], uses, use_count)) {
    return false;
  }
  if (allocation == NULL || !allocation->cfg_graph.has_cycles) {
    return true;
  }
  const loom_block_t* selected_block = recipe->packets[0]->parent_block;
  for (iree_host_size_t i = 1; i < recipe->packet_count; ++i) {
    if (!loom_low_rematerialization_blocks_share_execution_domain(
            allocation, selected_block, recipe->packets[i]->parent_block)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_rematerialization_clone_for_use(
    loom_rewriter_t* rewriter,
    const loom_low_rematerialization_recipe_t* recipe, loom_use_t use,
    loom_low_rematerialization_state_t* state, loom_ir_remap_t* remap,
    loom_value_id_t* out_cloned_value_id) {
  *out_cloned_value_id = LOOM_VALUE_ID_INVALID;

  loom_op_t* user_op = loom_use_user_op(use);
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, user_op);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = recipe->packet_count;
       i > 0 && iree_status_is_ok(status); --i) {
    loom_op_t* source_op = recipe->packets[i - 1];
    loom_op_t* cloned_op = NULL;
    status = loom_ir_clone_op(&rewriter->builder, source_op, remap, &cloned_op);
    if (!iree_status_is_ok(status)) {
      break;
    }

    const loom_value_id_t source_value_id = loom_op_const_results(source_op)[0];
    const loom_value_id_t cloned_value_id = loom_op_results(cloned_op)[0];
    status = loom_rewriter_clear_value_name(rewriter, cloned_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_try_set_derived_value_name(
          rewriter, source_value_id, cloned_value_id, IREE_SV("remat"));
    }
    if (iree_status_is_ok(status)) {
      iree_bitmap_set(state->per_user_values, cloned_value_id);
      if (state->required_register_values != NULL &&
          source_value_id < state->required_register_values->bit_count &&
          iree_bitmap_test(*state->required_register_values, source_value_id)) {
        iree_bitmap_set(*state->required_register_values, cloned_value_id);
      }
      *out_cloned_value_id = cloned_value_id;
    }
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  return iree_ok_status();
}

static iree_status_t loom_low_rematerialization_prepare_value(
    loom_module_t* module, loom_value_id_t value_id,
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* spill_allocation,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_rematerialization_facts_t* facts,
    loom_low_rematerialization_plan_t* out_plan) {
  *out_plan = (loom_low_rematerialization_plan_t){0};
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (!facts->reconstruct_constants &&
      value_id < state->per_user_values.bit_count &&
      iree_bitmap_test(state->per_user_values, value_id)) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) ||
      loom_value_consuming_use_count(value) != 0 ||
      loom_value_has_attribute_uses(value) || value->use_count == 0 ||
      loom_module_value_has_type_uses(module, value_id)) {
    return iree_ok_status();
  }

  const loom_value_ordinal_t ordinal =
      loom_low_rematerialization_ordinal(facts, value_id);
  if (!iree_any_bit_set(facts->value_flags[ordinal],
                        LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE)) {
    return iree_ok_status();
  }

  if (facts->reconstruct_constants &&
      !iree_any_bit_set(facts->value_flags[ordinal],
                        LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED)) {
    return iree_ok_status();
  }
  loom_op_t* defining_op = loom_value_def_op(value);
  const uint32_t use_count = value->use_count;
  const loom_use_t* observed_uses = loom_value_uses(value);
  const uint32_t unit_count = loom_low_register_type_unit_count(
      loom_module_value_type(module, value_id));
  bool rebuild_composition = facts->reconstruct_constants;
  bool shortens_live_range = false;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (!loom_low_rematerialization_use_is_eligible(value_id, defining_op,
                                                    observed_uses[i])) {
      return iree_ok_status();
    }
    // A proper subrange keeps the wider producer as an external capture.
    // Rebuilding the aggregate would multiply work the consumer never reads.
    if (rebuild_composition &&
        loom_low_rematerialization_use_unit_count(module, observed_uses[i],
                                                  unit_count) < unit_count) {
      rebuild_composition = false;
    }
    shortens_live_range = shortens_live_range ||
                          loom_low_rematerialization_use_shortens_live_range(
                              defining_op, observed_uses[i]);
  }
  // Replaying the selected packet across a repeated execution boundary is
  // already disallowed, independent of any producer ancestry that may enrich
  // its recipe. Reject it before constructing and indexing that ancestry.
  if (!facts->reconstruct_constants &&
      !loom_low_rematerialization_uses_share_execution_domain(
          spill_allocation, defining_op, observed_uses, use_count)) {
    return iree_ok_status();
  }

  loom_low_rematerialization_recipe_t recipe = {0};
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_plan_recipe(
      facts, value_id, rebuild_composition, /*follow_explicit_sources=*/true,
      state->required_register_values, arena, &recipe));
  if (recipe.packet_count == 0 && recipe.followed_explicit_source) {
    // An authored source edge enriches the packet's replay recipe; it cannot
    // revoke the packet-only rematerialization guarantee. Retry without
    // optional edges when the enriched producer chain is unavailable.
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_plan_recipe(
        facts, value_id, rebuild_composition, /*follow_explicit_sources=*/false,
        state->required_register_values, arena, &recipe));
  }
  if (recipe.packet_count == 0) {
    return iree_ok_status();
  }
  if (facts->reconstruct_constants) {
    shortens_live_range = true;
    spill_allocation = NULL;
  }
  for (uint32_t i = 0; i < use_count && !shortens_live_range; ++i) {
    const loom_op_t* previous = loom_use_user_op(observed_uses[i]);
    for (iree_host_size_t j = 0;
         j < recipe.packet_count && !shortens_live_range; ++j) {
      shortens_live_range |= previous->prev_op != recipe.packets[j];
      previous = recipe.packets[j];
    }
  }
  loom_low_rematerialization_action_t action =
      LOOM_LOW_REMATERIALIZATION_ACTION_CLONE;
  if (!shortens_live_range) {
    // All eligible uses name the immediately following operation. Source
    // adjacency does not bound the lifetime in a reordered allocation: retain
    // this private producer exactly as a per-user clone when it was separated
    // from its consumer. Operand-free packets remain available for ordinary
    // pressure scheduling, as they carry no input register lifetimes.
    if (schedule == NULL || defining_op->operand_count == 0) {
      return iree_ok_status();
    }
    const loom_low_schedule_node_t* producer =
        loom_low_schedule_node_for_op(schedule, defining_op);
    // Allocation includes nested regions; scheduling covers direct body ops.
    if (producer == NULL) {
      return iree_ok_status();
    }
    const loom_low_schedule_node_t* consumer = loom_low_schedule_node_for_op(
        schedule, loom_use_user_op(observed_uses[0]));
    // Eligibility established that every use is the immediately following op
    // in the producer's block, so a scheduled producer has a scheduled user.
    IREE_ASSERT(consumer != NULL);
    if (consumer->scheduled_ordinal <= producer->scheduled_ordinal + 1u) {
      return iree_ok_status();
    }
    action = LOOM_LOW_REMATERIALIZATION_ACTION_RETAIN_PLACEMENT;
  }
  if (action == LOOM_LOW_REMATERIALIZATION_ACTION_CLONE) {
    // Predicted-spill rematerialization is optional. Do not replace one reload
    // with a multi-packet recipe in a different repeated execution domain.
    // Cloning also multiplies external input uses; register-required values
    // cannot spill to repair that pressure. An optional authored source may
    // enrich the recipe only when it preserves both properties, otherwise the
    // packet's original replay guarantee remains authoritative.
    bool preserves_execution_domain =
        loom_low_rematerialization_preserves_execution_domain(
            spill_allocation, &recipe, observed_uses, use_count);
    if (recipe.followed_explicit_source &&
        (!preserves_execution_domain || recipe.has_required_external_input)) {
      IREE_RETURN_IF_ERROR(loom_low_rematerialization_plan_recipe(
          facts, value_id, rebuild_composition,
          /*follow_explicit_sources=*/false, state->required_register_values,
          arena, &recipe));
      preserves_execution_domain =
          loom_low_rematerialization_preserves_execution_domain(
              spill_allocation, &recipe, observed_uses, use_count);
    }
    if (recipe.packet_count == 0 || !preserves_execution_domain ||
        recipe.has_required_external_input) {
      return iree_ok_status();
    }
  }
  loom_use_t* captured_uses = NULL;
  if (action == LOOM_LOW_REMATERIALIZATION_ACTION_CLONE) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, use_count, sizeof(*captured_uses), (void**)&captured_uses));
    memcpy(captured_uses, observed_uses, use_count * sizeof(*captured_uses));
  }
  *out_plan = (loom_low_rematerialization_plan_t){
      .recipe = recipe,
      .uses = captured_uses,
      .use_count = use_count,
      .action = action,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_rematerialization_apply_value(
    loom_module_t* module, loom_value_id_t value_id,
    const loom_low_rematerialization_plan_t* plan,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result) {
  if (plan->action == LOOM_LOW_REMATERIALIZATION_ACTION_NONE) {
    return iree_ok_status();
  }
  if (plan->action == LOOM_LOW_REMATERIALIZATION_ACTION_RETAIN_PLACEMENT) {
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_values(
        module->values.count, state->arena, &state->per_user_values));
    iree_bitmap_set(state->per_user_values, value_id);
    *out_result = (loom_low_value_rematerialization_result_t){
        .value_id = value_id,
        .retained_placement_count = 1,
    };
    return iree_ok_status();
  }

  const loom_low_rematerialization_recipe_t* recipe = &plan->recipe;
  loom_op_t* defining_op = recipe->packets[0];
  const loom_use_t* uses = plan->uses;
  const uint32_t use_count = plan->use_count;

  // Verified SSA makes every packet input and external type/attribute capture
  // available at its definition, which dominates each existing operand use.
  // Recipe planning rejects consumed external inputs, so cloning cannot cross
  // an ownership transfer.
  // Clone the complete recipe in dependency and consumption order, so each
  // tied or moved packet consumes its private predecessor. Each
  // eligible packet has one result and no regions. Reserve membership for the
  // worst-case per-use clones before mutation changes the module value count.
  if (recipe->packet_count > UINT32_MAX / use_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "rematerialization clone count exceeds uint32 capacity");
  }
  const uint32_t maximum_clone_count =
      (uint32_t)(recipe->packet_count * use_count);
  if (maximum_clone_count > LOOM_VALUE_ID_INVALID - module->values.count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "rematerialization value count exceeds value ID capacity");
  }
  const iree_host_size_t required_value_count =
      module->values.count + maximum_clone_count;
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_values(
      required_value_count, state->arena, &state->per_user_values));
  if (state->required_register_values != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_rematerialization_reserve_values(
        required_value_count, state->arena, state->required_register_values));
  }
  // Index users by their first rewritten captured use. That exact operand
  // retains the clone ID for other occurrences in the same instruction.
  const iree_host_size_t user_capacity =
      iree_host_size_next_power_of_two((iree_host_size_t)use_count * 2);
  const iree_host_size_t user_mask = user_capacity - 1;
  uint32_t inline_user_uses[16];
  uint32_t* user_uses = inline_user_uses;
  if (user_capacity > IREE_ARRAYSIZE(inline_user_uses)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, user_capacity, sizeof(*user_uses), (void**)&user_uses));
  }
  memset(user_uses, 0, user_capacity * sizeof(*user_uses));
  loom_rewriter_t rewriter = {0};
  loom_low_value_rematerialization_result_t result = {
      .value_id = value_id,
  };
  loom_rewriter_initialize(&rewriter, module, arena);
  // All recipe producers are cloned in dependency order for every user. Each
  // mapping is overwritten before its next consumer; external captures remain
  // identities. One map therefore serves every private copy of this recipe.
  loom_ir_remap_t remap;
  iree_status_t status = loom_ir_remap_initialize(
      module, module, arena,
      &(loom_ir_remap_options_t){.allow_unmapped_values = true}, &remap);
  for (uint32_t i = 0; i < use_count && iree_status_is_ok(status); ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    const uint32_t user_hash =
        loom_structural_hash_finalize(loom_structural_hash_mix_u64(
            loom_structural_hash_initialize(), (uint64_t)(uintptr_t)user_op));
    iree_host_size_t user_slot = user_hash & user_mask;
    while (user_uses[user_slot] != 0 &&
           loom_use_user_op(uses[user_uses[user_slot] - 1]) != user_op) {
      user_slot = (user_slot + 1) & user_mask;
    }
    const bool first_user_use = user_uses[user_slot] == 0;
    loom_value_id_t cloned_value_id = LOOM_VALUE_ID_INVALID;
    if (first_user_use) {
      status = loom_low_rematerialization_clone_for_use(
          &rewriter, recipe, uses[i], state, &remap, &cloned_value_id);
    } else {
      const loom_use_t first_use = uses[user_uses[user_slot] - 1];
      cloned_value_id =
          loom_op_operands(user_op)[loom_use_operand_index(first_use)];
    }
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_operand(
          &rewriter, user_op, loom_use_operand_index(uses[i]), cloned_value_id);
    }
    if (iree_status_is_ok(status)) {
      if (first_user_use) {
        user_uses[user_slot] = i + 1;
        result.cloned_packet_count += (uint32_t)recipe->packet_count;
      }
      ++result.rewritten_operand_count;
    }
  }
  if (iree_status_is_ok(status)) {
    // Cloning the recipe may give external inputs additional users in other
    // blocks. Those inputs no longer retain a per-user placement guarantee.
    for (iree_host_size_t packet_index = 0; packet_index < recipe->packet_count;
         ++packet_index) {
      const loom_op_t* packet = recipe->packets[packet_index];
      const loom_value_id_t* operands = loom_op_const_operands(packet);
      for (uint16_t operand_index = 0; operand_index < packet->operand_count;
           ++operand_index) {
        const loom_value_id_t operand = operands[operand_index];
        loom_value_id_t mapped_value;
        const bool is_recipe_result =
            loom_ir_remap_try_lookup_value(&remap, operand, &mapped_value);
        if (!is_recipe_result) {
          if (operand < state->per_user_values.bit_count) {
            iree_bitmap_reset(state->per_user_values, operand);
          }
        }
      }
    }
    // Erase producers made unused by this repair. A shared predecessor may
    // survive while another independent branch of the recipe becomes dead.
    IREE_ASSERT(loom_op_results_unused(module, defining_op));
    for (iree_host_size_t i = 0;
         i < recipe->packet_count && iree_status_is_ok(status); ++i) {
      loom_op_t* packet = recipe->packets[i];
      if (!loom_op_results_unused(module, packet)) {
        continue;
      }
      status = loom_rewriter_erase(&rewriter, packet);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_result = result;
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_schedule_rematerialize_failure(
    loom_module_t* module, const loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_value_rematerialization_result_t* out_result) {
  *out_result = loom_low_value_rematerialization_result_empty();
  loom_low_rematerialization_facts_t facts = {
      .module = module,
      .value_ids = schedule->value_ids,
      .value_count = schedule->value_count,
      .cfg_graph = &schedule->cfg_graph,
      .region = schedule->blocks[0].block->parent_region,
  };
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_facts_initialize(
      &facts, &schedule->target, arena));
  const loom_value_id_t value_id = schedule->failure.state_value_id;
  loom_low_rematerialization_plan_t plan = {0};
  iree_status_t status = loom_low_rematerialization_prepare_value(
      module, value_id, /*schedule=*/NULL, /*spill_allocation=*/NULL, state,
      arena, &facts, &plan);
  if (iree_status_is_ok(status)) {
    status = loom_low_rematerialization_apply_value(module, value_id, &plan,
                                                    state, arena, out_result);
  }
  loom_low_rematerialization_facts_release(&facts);
  return status;
}

// Returns true when any storage unit in |assignment| is live at |point|.
//
// Whole-assignment bounds reject most candidates cheaply. Sparse storage
// segments exclude values that only appear live because mutually exclusive CFG
// paths share the linear program-point space, and refined per-unit bounds
// retain the allocation model's subrange precision.
static bool loom_low_allocation_assignment_is_live_at_point(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment, uint32_t point) {
  if (point < assignment->start_point || point >= assignment->end_point) {
    return false;
  }
  if (assignment->liveness_segments.count != 0 &&
      !loom_liveness_segment_range_contains(
          table->storage_segments, assignment->liveness_segments, point)) {
    return false;
  }
  for (uint32_t i = 0; i < assignment->unit_count; ++i) {
    const uint32_t start_point =
        loom_low_allocation_live_range_assignment_unit_start_point(
            table->unit_start_points, table->unit_point_count, assignment, i);
    const uint32_t end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            table->unit_end_points, table->unit_point_count, assignment, i);
    if (point >= start_point && point < end_point) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_assignment_overlaps_failure_storage(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_low_allocation_failure_t* failure = &table->failure;
  const loom_low_descriptor_set_t* descriptor_set =
      table->target.descriptor_set;
  if (failure->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      failure->descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }

  const loom_low_reg_class_t* failed_reg_class =
      &descriptor_set->reg_classes[failure->descriptor_reg_class_id];
  loom_low_allocation_assignment_t candidate = {
      .value_id = failure->value_id,
      .descriptor_reg_class_id = failure->descriptor_reg_class_id,
      .unit_count = failure->required_unit_count,
      .location_kind = failure->location_kind,
      .location_count = failure->required_unit_count,
  };
  if (!loom_low_reg_class_uses_explicit_physical_registers(failed_reg_class)) {
    if (failure->location_base == UINT32_MAX || failure->location_count == 0) {
      return false;
    }
    candidate.location_base = failure->location_base;
    candidate.location_count = failure->location_count;
    return loom_low_allocation_storage_assignment_ranges_overlap(
        descriptor_set, &candidate, assignment);
  }

  const uint32_t candidate_count =
      loom_low_allocation_storage_explicit_location_candidate_count(
          descriptor_set, failure->descriptor_reg_class_id,
          failure->required_unit_count);
  for (uint32_t candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    if (!loom_low_allocation_storage_explicit_location_candidate(
            descriptor_set, failure->descriptor_reg_class_id,
            failure->required_unit_count, candidate_index,
            &candidate.location_base,
            /*out_first_candidate_ordinal=*/NULL,
            /*out_pressure_extent=*/NULL,
            /*out_packing_rank=*/NULL)) {
      continue;
    }
    if (loom_low_allocation_storage_assignment_ranges_overlap(
            descriptor_set, &candidate, assignment)) {
      return true;
    }
  }
  return false;
}

typedef struct loom_low_rematerialization_candidate_t {
  // Original value ID, stable across rewrites of other candidates.
  loom_value_id_t value_id;
  // Exact producer point retained before the first rewrite, including nested
  // structured regions and scheduled operation order.
  uint32_t definition_point;
  // Retained CFG DFS preorder; strict dominators have smaller preorders.
  uint16_t block_preorder;
  // Cross-block demand repairs precede the local failure frontier so a
  // sufficient cross-block batch does not clone additional local values.
  bool is_cross_block_pressure;
} loom_low_rematerialization_candidate_t;

static bool loom_low_rematerialization_candidate_less(
    const loom_low_rematerialization_candidate_t* lhs,
    const loom_low_rematerialization_candidate_t* rhs) {
  if (lhs->is_cross_block_pressure != rhs->is_cross_block_pressure) {
    return lhs->is_cross_block_pressure;
  }
  if (lhs->block_preorder != rhs->block_preorder) {
    return lhs->block_preorder > rhs->block_preorder;
  }
  return lhs->definition_point > rhs->definition_point;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_rematerialization_candidate_sort,
                          loom_low_rematerialization_candidate_t,
                          loom_low_rematerialization_candidate_less)

static iree_status_t loom_low_rematerialization_plan_failure_batch(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_rematerialization_candidate_t** out_candidates,
    iree_host_size_t* out_count, iree_host_size_t* out_cross_block_count) {
  *out_candidates = NULL;
  *out_count = 0;
  *out_cross_block_count = 0;
  const loom_liveness_analysis_t* liveness = &table->liveness;
  const loom_low_descriptor_set_t* descriptors = table->target.descriptor_set;
  uint8_t* pressure_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptors->reg_class_count, sizeof(*pressure_classes),
      (void**)&pressure_classes));
  memset(pressure_classes, 0, descriptors->reg_class_count);
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    if (summary->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    const uint16_t class_id = summary->value_class.register_class_id;
    const loom_low_reg_class_t* reg_class = &descriptors->reg_classes[class_id];
    const uint32_t capacity =
        loom_liveness_value_class_equal(summary->value_class,
                                        table->failure.value_class)
            ? table->failure.budget_units
            : reg_class->allocatable_count;
    if (iree_any_bit_set(reg_class->flags,
                         LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE) &&
        capacity != 0 && summary->peak_live_units > capacity) {
      pressure_classes[class_id] = 1;
    }
  }
  loom_low_rematerialization_candidate_t* candidates = NULL;
  iree_host_size_t count = 0;
  iree_host_size_t capacity = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    // Storage handles and unused block arguments retain their ordinals without
    // contributing a register interval to the failure frontier.
    const uint32_t interval_index = liveness->value_interval_indices[ordinal];
    if (interval_index == UINT32_MAX) {
      continue;
    }
    const loom_liveness_interval_t* interval =
        &liveness->intervals[interval_index];
    const uint32_t assignment_index =
        table->assignment_indices_by_value_ordinal[ordinal];
    const bool in_failure_frontier =
        interval->value_id == table->failure.value_id ||
        (assignment_index != UINT32_MAX &&
         loom_low_allocation_assignment_is_live_at_point(
             table, &table->assignments[assignment_index],
             table->failure.start_point) &&
         (interval->value_class.register_class_id ==
              table->failure.descriptor_reg_class_id ||
          loom_low_allocation_assignment_overlaps_failure_storage(
              table, &table->assignments[assignment_index])));
    // Intervals retain cross-block demand. The failed allocation independently
    // identifies live pressure that can be repaired in the same stable batch.
    const bool is_cross_block_pressure =
        interval->block_count >= 2 &&
        pressure_classes[interval->value_class.register_class_id];
    if (!in_failure_frontier && !is_cross_block_pressure) {
      continue;
    }
    loom_op_t* defining_op = NULL;
    if (!loom_low_rematerialization_packet_is_eligible(
            module, &table->target, interval->value_id, &defining_op)) {
      continue;
    }
    const uint16_t block_index = loom_liveness_block_index_at_point(
        liveness, interval->definition_point);
    const loom_cfg_block_info_t* block =
        table->cfg_graph.blocks != NULL ? &table->cfg_graph.blocks[block_index]
                                        : NULL;
    if (block != NULL && !block->reachable) {
      continue;
    }
    if (count == capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, count, count + 1,
                                                 sizeof(*candidates), &capacity,
                                                 (void**)&candidates));
    }
    candidates[count++] = (loom_low_rematerialization_candidate_t){
        .value_id = interval->value_id,
        .definition_point = interval->definition_point,
        .block_preorder = block != NULL ? block->preorder : 0,
        .is_cross_block_pressure = is_cross_block_pressure,
    };
    *out_cross_block_count += is_cross_block_pressure;
  }
  loom_low_rematerialization_candidate_sort(candidates, count);
  *out_candidates = candidates;
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_low_rematerialization_apply_batch(
    loom_module_t* module,
    const loom_low_rematerialization_candidate_t* candidates,
    iree_host_size_t candidate_count, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* spill_allocation,
    loom_low_rematerialization_state_t* state, iree_arena_allocator_t* arena,
    loom_low_rematerialization_facts_t* facts,
    loom_low_rematerialization_batch_result_t* out_result) {
  // Consumer-first application lets producers see uses created by earlier
  // clones. Only immutable ordering/value facts survive between candidates;
  // each individual rewrite captures its current uses immediately before apply.
  // Placement-only repairs can consult the completed schedule until the first
  // rewrite invalidates it. Later candidates still use fresh IR for cloning.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < candidate_count && iree_status_is_ok(status);
       ++i) {
    const loom_low_rematerialization_candidate_t* candidate = &candidates[i];
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(arena);
    loom_low_value_rematerialization_result_t result = {0};
    loom_low_rematerialization_plan_t plan = {0};
    status = loom_low_rematerialization_prepare_value(
        module, candidate->value_id, schedule, spill_allocation, state, arena,
        facts, &plan);
    if (iree_status_is_ok(status)) {
      const loom_value_id_t first_clone = module->values.count;
      status = loom_low_rematerialization_apply_value(
          module, candidate->value_id, &plan, state, arena, &result);
      if (iree_status_is_ok(status) && facts->reconstruct_constants &&
          result.rewritten_operand_count != 0) {
        for (loom_value_id_t value_id = first_clone;
             value_id < module->values.count; ++value_id) {
          if (loom_low_type_is_register(
                  loom_module_value_type(module, value_id))) {
            iree_bitmap_set(*state->required_register_values, value_id);
          }
        }
      }
    }
    if (iree_status_is_ok(status)) {
      loom_low_rematerialization_accumulate_result(&result, out_result);
      if (result.rewritten_operand_count != 0) {
        schedule = NULL;
      }
    }
    // IR and retained placement use their own arenas. Recipes, captured uses,
    // and clone maps live only for this root, even in a large repair batch.
    iree_arena_checkpoint_restore(&checkpoint);
  }
  return status;
}

static iree_status_t loom_low_allocation_rematerialization_emit_summary(
    const loom_low_allocation_table_t* table, iree_string_view_t trigger,
    uint16_t trigger_reg_class_id,
    const loom_low_rematerialization_batch_result_t* result,
    iree_diagnostic_emitter_t emitter) {
  if (emitter.fn == NULL || result->repaired_value_count == 0) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_reg_class_name(
          table->target.descriptor_set, trigger_reg_class_id)),
      loom_param_string(trigger),
      loom_param_u32(result->repaired_value_count),
      loom_param_u32(result->cloned_packet_count),
      loom_param_u32(result->rewritten_operand_count),
      loom_param_u32(result->retained_placement_count),
      loom_param_string(IREE_SV("producer-rematerializable")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = table->function_op,
      .error = LOOM_ERR_BACKEND_045,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_low_allocation_rematerialize_failure(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_schedule_table_t* schedule,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result) {
  *out_result = (loom_low_rematerialization_batch_result_t){0};
  if (!loom_low_allocation_failure_is_rematerializable_pressure(
          &table->failure)) {
    return iree_ok_status();
  }
  loom_low_rematerialization_candidate_t* candidates = NULL;
  iree_host_size_t candidate_count = 0;
  iree_host_size_t cross_block_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_plan_failure_batch(
      module, table, arena, &candidates, &candidate_count, &cross_block_count));
  if (candidate_count == 0) {
    return iree_ok_status();
  }
  loom_low_rematerialization_facts_t facts = {
      .module = module,
      .value_ids = table->liveness.value_ids,
      .value_count = table->liveness.value_count,
      .liveness = &table->liveness,
      .cfg_graph = &table->cfg_graph,
      .region = table->liveness.region,
  };
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_facts_initialize(
      &facts, &table->target, arena));
  iree_status_t status = loom_low_rematerialization_apply_batch(
      module, candidates, cross_block_count, schedule,
      /*spill_allocation=*/NULL, state, arena, &facts, out_result);
  if (iree_status_is_ok(status) && out_result->repaired_value_count == 0 &&
      candidate_count != cross_block_count) {
    status = loom_low_rematerialization_apply_batch(
        module, candidates + cross_block_count,
        candidate_count - cross_block_count, schedule,
        /*spill_allocation=*/NULL, state, arena, &facts, out_result);
  }
  loom_low_rematerialization_facts_release(&facts);
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_rematerialization_emit_summary(
        table, IREE_SV("allocation-failure"),
        table->failure.descriptor_reg_class_id, out_result, emitter);
  }
  return status;
}

static iree_status_t loom_low_allocation_rematerialize_spill_plans_impl(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    bool reconstruct_constants, loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result) {
  *out_result = (loom_low_rematerialization_batch_result_t){0};
  if (table->spill_plan_count == 0) {
    return iree_ok_status();
  }
  const loom_liveness_analysis_t* liveness = &table->liveness;
  loom_low_rematerialization_candidate_t* candidates = NULL;
  iree_host_size_t count = 0;
  iree_host_size_t capacity = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    const uint32_t assignment_index =
        table->assignment_indices_by_value_ordinal[ordinal];
    if (assignment_index == UINT32_MAX ||
        table->assignments[assignment_index].location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
      continue;
    }
    const loom_liveness_interval_t* interval =
        &liveness->intervals[liveness->value_interval_indices[ordinal]];
    loom_op_t* defining_op = NULL;
    if (!loom_low_rematerialization_packet_is_eligible(
            module, &table->target, interval->value_id, &defining_op)) {
      continue;
    }
    const uint16_t block_index = loom_liveness_block_index_at_point(
        liveness, interval->definition_point);
    const loom_cfg_block_info_t* block =
        table->cfg_graph.blocks != NULL ? &table->cfg_graph.blocks[block_index]
                                        : NULL;
    if (block != NULL && !block->reachable) {
      continue;
    }
    if (count == capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, count, count + 1,
                                                 sizeof(*candidates), &capacity,
                                                 (void**)&candidates));
    }
    candidates[count++] = (loom_low_rematerialization_candidate_t){
        .value_id = interval->value_id,
        .definition_point = interval->definition_point,
        .block_preorder = block != NULL ? block->preorder : 0,
    };
  }
  if (count == 0) {
    return iree_ok_status();
  }
  loom_low_rematerialization_facts_t facts = {
      .module = module,
      .value_ids = liveness->value_ids,
      .value_count = liveness->value_count,
      .liveness = liveness,
      .cfg_graph = &table->cfg_graph,
      .region = liveness->region,
      .reconstruct_constants = reconstruct_constants,
  };
  IREE_RETURN_IF_ERROR(loom_low_rematerialization_facts_initialize(
      &facts, &table->target, arena));
  loom_low_rematerialization_candidate_sort(candidates, count);
  iree_status_t status = loom_low_rematerialization_apply_batch(
      module, candidates, count, /*schedule=*/NULL, table, state, arena, &facts,
      out_result);
  loom_low_rematerialization_facts_release(&facts);
  IREE_RETURN_IF_ERROR(status);
  // The first predicted spill triggers this repair; the batch can cover
  // additional classes from the same allocation snapshot.
  return loom_low_allocation_rematerialization_emit_summary(
      table, IREE_SV("spill-plan"),
      table->assignments[table->spill_plans[0].assignment_index]
          .descriptor_reg_class_id,
      out_result, emitter);
}

iree_status_t loom_low_allocation_rematerialize_spill_plans(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_rematerialization_batch_result_t* out_result) {
  return loom_low_allocation_rematerialize_spill_plans_impl(
      module, table, false, state, emitter, arena, out_result);
}

iree_status_t loom_low_allocation_reconstruct_spills(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_rematerialization_state_t* state,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    const loom_low_allocation_spill_plan_t** out_spill_plans,
    iree_host_size_t* out_spill_plan_count,
    loom_low_rematerialization_batch_result_t* out_result) {
  *out_spill_plans = table->spill_plans;
  *out_spill_plan_count = table->spill_plan_count;
  IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_spill_plans_impl(
      module, table, true, state, emitter, arena, out_result));
  if (out_result->rewritten_operand_count == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_spill_plan_t* remaining = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, table->spill_plan_count, sizeof(*remaining), (void**)&remaining));
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &table->spill_plans[i];
    if (loom_module_value(module, plan->value_id)->use_count != 0) {
      remaining[count++] = *plan;
    }
  }
  *out_spill_plans = remaining;
  *out_spill_plan_count = count;
  return iree_ok_status();
}
