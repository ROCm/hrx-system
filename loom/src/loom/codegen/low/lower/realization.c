// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/realization.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/ir/structural_hash.h"
#include "loom/ir/types.h"
#include "loom/ops/cfg/ops.h"

typedef struct loom_low_lower_realization_record_t
    loom_low_lower_realization_record_t;

struct loom_low_lower_realization_t {
  // Physical carrier selected for initialization and loop state.
  loom_type_t type;
  // Arithmetic emitted once at the planned initialization event.
  loom_low_lower_realization_emit_fn_t initialize;
  // Arithmetic emitted at the planned update event, or NULL for an invariant.
  loom_low_lower_realization_emit_fn_t update;
  // Function-owned target payload consumed by the arithmetic callbacks.
  const void* data;
  // Loop domain for carried values, or NULL for an invariant.
  const loom_low_lower_realization_loop_t* loop;
  // Source access supplying locations for emitted arithmetic.
  const loom_op_t* source_op;
  // Next unique realization in preparation order.
  struct loom_low_lower_realization_t* next;
  // Construction ends when finalize freezes events and initializes values.
  union {
    // Borrowed interning and placement record, valid only during construction.
    loom_low_lower_realization_record_t* construction;
    // Low values populated while executing the frozen schedule.
    struct {
      // Initial value produced at the initialization event.
      loom_value_id_t initial;
      // Header argument for carried state; INVALID for an invariant.
      loom_value_id_t header;
      // Updated value forwarded by the backedge.
      loom_value_id_t backedge;
    } values;
  } state;
};

struct loom_low_lower_realization_record_t {
  // Target recipe family participating in physical identity.
  loom_low_lower_plan_id_t id;
  // Construction-owned physical identity key, excluding explicit inputs.
  iree_const_byte_span_t key;
  // Construction-owned dependencies used for placement and storage demand.
  const loom_low_lower_realization_input_t* inputs;
  // Number of populated dependencies.
  uint16_t input_count;
  // Retained physical value represented by this record.
  loom_low_lower_realization_t* value;
  // Block in which all initializer dependencies are available.
  const loom_block_t* initial_block;
  // Latest dependency definition, or NULL for block-entry initialization.
  const loom_op_t* initial_anchor;
  // Last use in the latch block, or the backedge for a multi-block use span.
  const loom_op_t* update_anchor;
  // Next collision in the physical-identity table.
  struct loom_low_lower_realization_record_t* hash_next;
  // Physical-identity hash used during table growth.
  uint32_t hash;
};

typedef struct loom_low_lower_realization_event_t {
  // Realization initialized or updated by this event.
  loom_low_lower_realization_t* realization;
  // Dense source block index owning the event.
  uint16_t block_index;
  // Sparse source operation ordinal, or zero at block entry.
  uint64_t position;
  // True when the event precedes the operation instead of following it.
  bool before;
  // True for an update and false for initialization.
  bool is_update;
} loom_low_lower_realization_event_t;

typedef struct loom_low_lower_realization_block_t {
  // Next event consumed in this block's emission order.
  iree_host_size_t event_cursor;
  // Exclusive end of this block's event range.
  iree_host_size_t event_end;
  // Supplemental arguments owned by this source loop header.
  struct {
    // Indexed arguments in preparation order, borrowing canonical recipes.
    loom_low_lower_realization_t** values;
    // Number of supplemental arguments, bounded by Low block construction.
    uint32_t count;
  } header;
  // Header receiving this block's dedicated entry/backedge payload, or NULL.
  const struct loom_low_lower_realization_block_t* edge_destination;
} loom_low_lower_realization_block_t;

typedef struct loom_low_lower_realization_offer_record_t {
  // Target cost facts and retained-plan replacement callback.
  loom_low_lower_realization_offer_t offer;
  // Exact repetition domain retained by the shared CFG owner.
  const loom_low_lower_realization_loop_t* loop;
  // Packet location and execution block.
  const loom_op_t* source_op;
  // Next offer in source preparation order.
  struct loom_low_lower_realization_offer_record_t* next;
  // True when the complete reuse group pays for its replacement.
  bool selected;
} loom_low_lower_realization_offer_record_t;

typedef struct loom_low_lower_realization_construction_t {
  // Arena scoped to shared source-memory preparation.
  iree_arena_allocator_t* arena;
  // Borrowed source CFG facts used to prove placement and loop eligibility.
  const loom_value_fact_cfg_region_t* cfg;
  // Eligible domains indexed by cfg->loops; a NULL header means ineligible.
  loom_low_lower_realization_loop_t* loops;
  // Last unique realization in preparation order.
  loom_low_lower_realization_t* last;
  // Physical-identity hash buckets.
  loom_low_lower_realization_record_t** buckets;
  // Power-of-two bucket count, or zero before the first request.
  iree_host_size_t bucket_count;
  // Number of retained unique realizations.
  iree_host_size_t count;
  // Deferred packet replacements in source preparation order.
  struct {
    // First offer, or NULL before any replacement is proposed.
    loom_low_lower_realization_offer_record_t* first;
    // Last offer, used for constant-time append.
    loom_low_lower_realization_offer_record_t* last;
    // Number of offers indexed at the shared selection boundary.
    iree_host_size_t count;
  } offers;
} loom_low_lower_realization_construction_t;

struct loom_low_lower_realizations_t {
  // Source region whose block indices address the frozen schedule.
  const loom_region_t* region;
  // Construction-only state, cleared when the schedule is frozen.
  loom_low_lower_realization_construction_t* construction;
  // Source-block-indexed emission events and edge payloads.
  loom_low_lower_realization_block_t* blocks;
  // Unique realizations in preparation order.
  loom_low_lower_realization_t* first;
  // Frozen events sorted by source block and operation ordinal.
  loom_low_lower_realization_event_t* events;
};

static bool loom_low_lower_realization_loop_domain(
    const loom_low_lower_context_t* context,
    const loom_value_fact_cfg_region_t* cfg, uint16_t loop_index,
    loom_low_lower_realization_loop_t* out_loop) {
  const loom_cfg_natural_loop_t* loop = &cfg->loops.loops[loop_index];
  const loom_value_fact_induction_t* induction = &cfg->inductions[loop_index];
  if (loop->entries.count != 1 || loop->backedges.count != 1 ||
      loop->exits.count != 1 || induction->value == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_cfg_edge_info_t* entry =
      &cfg->graph.edges[loop->entries.unique_index];
  const loom_cfg_edge_info_t* backedge =
      &cfg->graph.edges[loop->backedges.unique_index];
  const loom_cfg_edge_info_t* exit =
      &cfg->graph.edges[loop->exits.unique_index];
  if (!loom_cfg_br_isa(entry->terminator) ||
      !loom_cfg_br_isa(backedge->terminator) ||
      exit->source_block_index != loop->header_index ||
      exit->selector_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_value_facts_is_subgroup_uniform(loom_value_fact_table_lookup(
          context->fact_table, exit->selector_value_id)) ||
      !loom_value_facts_is_subgroup_uniform(loom_value_fact_control_execution(
          cfg->control, entry->source_block_index)) ||
      !loom_value_facts_is_subgroup_uniform(loom_value_fact_control_execution(
          cfg->control, backedge->source_block_index))) {
    return false;
  }
  const loom_loop_recurrence_facts_t recurrence =
      loom_value_fact_induction_facts(context->fact_table, context->module,
                                      induction);
  int64_t initial_value = 0;
  int64_t step = 0;
  int64_t exit_value = 0;
  if (recurrence.trip_count_kind != LOOM_LOOP_TRIP_COUNT_EXACT ||
      !loom_value_facts_as_exact_i64(recurrence.exit_value, &exit_value) ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_recurrence_operand_facts(context->fact_table,
                                                   induction->initial_value),
          &initial_value) ||
      induction->step.value == LOOM_VALUE_ID_INVALID ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_induction_step_facts(context->fact_table, induction),
          &step) ||
      step <= 0) {
    return false;
  }
  *out_loop = (loom_low_lower_realization_loop_t){
      .header = cfg->graph.blocks[loop->header_index].block,
      .entry = entry->terminator,
      .backedge = backedge->terminator,
      .induction_value = induction->value,
      .initial_value = initial_value,
      .step = step,
      .exit_value = exit_value,
      .trip_count = recurrence.trip_count,
  };
  return true;
}

iree_status_t loom_low_lower_realizations_create(
    loom_low_lower_context_t* context,
    iree_arena_allocator_t* construction_arena) {
  const loom_value_fact_cfg_region_t* cfg = loom_low_lower_context_cfg(context);
  if (cfg == NULL || cfg->loops.loop_count == 0 ||
      loom_low_lower_source_plan_uses_structured_control_flow(context)) {
    return iree_ok_status();
  }
  loom_low_lower_realizations_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, 1, sizeof(*state), (void**)&state));
  loom_low_lower_realization_construction_t* construction = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      construction_arena, sizeof(*construction), (void**)&construction));
  *construction = (loom_low_lower_realization_construction_t){
      .arena = construction_arena, .cfg = cfg};
  *state = (loom_low_lower_realizations_t){.region = cfg->graph.region,
                                           .construction = construction};
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, cfg->loops.loop_count, sizeof(*construction->loops),
      (void**)&construction->loops));
  memset(construction->loops, 0,
         cfg->loops.loop_count * sizeof(*construction->loops));
  for (uint16_t i = 0; i < cfg->loops.loop_count; ++i) {
    loom_low_lower_realization_loop_domain(context, cfg, i,
                                           &construction->loops[i]);
  }
  context->lowering->source_plan.realizations = state;
  return iree_ok_status();
}

const loom_low_lower_realization_loop_t* loom_low_lower_realization_loop(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  if (state == NULL ||
      source_op->parent_block->parent_region != state->region) {
    return NULL;
  }
  const loom_low_lower_realization_construction_t* construction =
      state->construction;
  if (!loom_value_facts_is_subgroup_uniform(loom_value_fact_control_execution(
          construction->cfg->control, source_op->parent_block->region_index))) {
    return NULL;
  }
  const uint16_t index = loom_cfg_loop_nest_innermost(
      &construction->cfg->loops, source_op->parent_block->region_index);
  return index != LOOM_CFG_LOOP_NEST_NONE && construction->loops[index].header
             ? &construction->loops[index]
             : NULL;
}

iree_status_t loom_low_lower_realization_offer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_lower_realization_offer_t* offer) {
  if (source_op->parent_block != loop->backedge->parent_block) {
    return iree_ok_status();
  }
  loom_low_lower_realization_construction_t* construction =
      context->lowering->source_plan.realizations->construction;
  loom_low_lower_realization_offer_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(construction->arena, sizeof(*record),
                                           (void**)&record));
  uint8_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      construction->arena, offer->key.data_length, (void**)&key));
  if (offer->key.data_length) {
    memcpy(key, offer->key.data, offer->key.data_length);
  }
  *record = (loom_low_lower_realization_offer_record_t){
      .offer = *offer, .loop = loop, .source_op = source_op};
  record->offer.key = iree_make_const_byte_span(key, offer->key.data_length);
  if (construction->offers.last) {
    construction->offers.last->next = record;
  } else {
    construction->offers.first = record;
  }
  construction->offers.last = record;
  ++construction->offers.count;
  return iree_ok_status();
}

static bool loom_low_lower_realization_offer_group_equal(
    const loom_low_lower_realization_offer_record_t* left,
    const loom_low_lower_realization_offer_record_t* right) {
  const uint16_t left_block = left->source_op->parent_block->region_index;
  const uint16_t right_block = right->source_op->parent_block->region_index;
  return left_block == right_block && left->offer.id == right->offer.id &&
         left->offer.key.data_length == right->offer.key.data_length &&
         (left->offer.key.data_length == 0 ||
          memcmp(left->offer.key.data, right->offer.key.data,
                 left->offer.key.data_length) == 0);
}

typedef struct loom_low_lower_realization_cost_group_t {
  // First offer defining this group's identity and common costs.
  const loom_low_lower_realization_offer_record_t* first;
  // Total credit for distinct direct calculations removed by the group.
  double direct_cost;
  // Shared loop cost plus every participating packet's setup cost.
  double replacement_cost;
} loom_low_lower_realization_cost_group_t;

typedef struct loom_low_lower_realization_cost_credit_t {
  // Group owning this credit, or NULL for an empty hash-table slot.
  const loom_low_lower_realization_cost_group_t* group;
  // Identity of the displaced calculation credited once in this group.
  uint64_t removed_key;
} loom_low_lower_realization_cost_credit_t;

static iree_status_t loom_low_lower_realizations_select_offers(
    loom_low_lower_context_t* context,
    loom_low_lower_realization_construction_t* state) {
  if (state->offers.count == 0) {
    return iree_ok_status();
  }
  // Group membership and duplicate credits are preparation scratch. The
  // retained result is one selection bit per offer, applied in source order.
  loom_low_lower_realization_cost_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->planning_arena, state->offers.count,
                                sizeof(*groups), (void**)&groups));
  loom_low_lower_realization_cost_group_t** offer_groups = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->planning_arena, state->offers.count,
                                sizeof(*offer_groups), (void**)&offer_groups));
  iree_host_size_t bucket_count = 1;
  while (bucket_count < state->offers.count * 2) {
    bucket_count *= 2;
  }
  loom_low_lower_realization_cost_group_t** group_buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->planning_arena, bucket_count, sizeof(*group_buckets),
      (void**)&group_buckets));
  memset(group_buckets, 0, bucket_count * sizeof(*group_buckets));
  loom_low_lower_realization_cost_credit_t* credits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&context->planning_arena,
                                                 bucket_count, sizeof(*credits),
                                                 (void**)&credits));
  memset(credits, 0, bucket_count * sizeof(*credits));
  iree_host_size_t group_count = 0;
  iree_host_size_t offer_index = 0;
  for (loom_low_lower_realization_offer_record_t* record = state->offers.first;
       record; record = record->next) {
    const loom_low_lower_realization_offer_t* offer = &record->offer;
    uint32_t hash = loom_structural_hash_mix_u16(
        loom_structural_hash_initialize(),
        record->source_op->parent_block->region_index);
    hash = loom_structural_hash_mix_u64(hash, offer->id);
    hash = loom_structural_hash_finalize(loom_structural_hash_mix_bytes(
        hash, offer->key.data, offer->key.data_length));
    iree_host_size_t bucket = hash & (bucket_count - 1);
    while (group_buckets[bucket] &&
           !loom_low_lower_realization_offer_group_equal(
               group_buckets[bucket]->first, record)) {
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    loom_low_lower_realization_cost_group_t* group = group_buckets[bucket];
    // Costs are estimates, not range proofs. Floating point avoids overflow
    // for valid large trip counts without saturating an otherwise useful gain.
    const double trips = (double)record->loop->trip_count;
    if (group == NULL) {
      group = &groups[group_count++];
      *group = (loom_low_lower_realization_cost_group_t){
          .first = record,
          .replacement_cost =
              offer->group_setup_cost + trips * offer->group_iteration_cost,
      };
      group_buckets[bucket] = group;
    }
    offer_groups[offer_index++] = group;
    group->replacement_cost += offer->setup_cost;
    hash = loom_structural_hash_mix_u64(loom_structural_hash_initialize(),
                                        (uint64_t)(group - groups));
    hash = loom_structural_hash_finalize(
        loom_structural_hash_mix_u64(hash, offer->removed_key));
    bucket = hash & (bucket_count - 1);
    while (credits[bucket].group &&
           (credits[bucket].group != group ||
            credits[bucket].removed_key != offer->removed_key)) {
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    if (credits[bucket].group == NULL) {
      credits[bucket] = (loom_low_lower_realization_cost_credit_t){
          .group = group, .removed_key = offer->removed_key};
      group->direct_cost += trips * offer->removed_iteration_cost;
    }
  }
  offer_index = 0;
  for (loom_low_lower_realization_offer_record_t* record = state->offers.first;
       record; record = record->next) {
    const loom_low_lower_realization_cost_group_t* group =
        offer_groups[offer_index++];
    record->selected = group->direct_cost > group->replacement_cost;
  }
  iree_status_t status = iree_ok_status();
  for (loom_low_lower_realization_offer_record_t* record = state->offers.first;
       record && iree_status_is_ok(status); record = record->next) {
    if (record->selected) {
      status =
          record->offer.apply(context, record->source_op, record->offer.data);
    }
  }
  return status;
}

static uint32_t loom_low_lower_realization_hash(
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_lower_realization_recipe_t* recipe) {
  uint32_t hash = loom_structural_hash_mix_u64(
      loom_structural_hash_initialize(), recipe->id);
  hash = loom_structural_hash_mix_u32(hash, loom_type_hash(recipe->type));
  hash = loom_structural_hash_mix_u16(
      hash, loop ? loop->header->region_index : UINT16_MAX);
  hash = loom_structural_hash_mix_bytes(hash, recipe->key.data,
                                        recipe->key.data_length);
  for (uint16_t i = 0; i < recipe->input_count; ++i) {
    hash = loom_structural_hash_mix_u32(hash, recipe->inputs[i].kind);
    const loom_low_lower_realization_input_t* input = &recipe->inputs[i];
    hash = input->kind == LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION
               ? loom_structural_hash_mix_u64(
                     hash, (uintptr_t)input->value.realization)
               : loom_structural_hash_mix_u32(hash, input->value.identity);
  }
  return loom_structural_hash_finalize(hash);
}

static bool loom_low_lower_realization_matches(
    const loom_low_lower_realization_record_t* record,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_lower_realization_recipe_t* recipe) {
  if (record->value->loop != loop || record->id != recipe->id ||
      !loom_type_equal(record->value->type, recipe->type) ||
      record->input_count != recipe->input_count ||
      record->key.data_length != recipe->key.data_length ||
      (recipe->key.data_length != 0 &&
       memcmp(record->key.data, recipe->key.data, recipe->key.data_length) !=
           0)) {
    return false;
  }
  for (uint16_t i = 0; i < recipe->input_count; ++i) {
    const loom_low_lower_realization_input_t* left = &record->inputs[i];
    const loom_low_lower_realization_input_t* right = &recipe->inputs[i];
    if (left->kind != right->kind ||
        (left->kind == LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION
             ? left->value.realization != right->value.realization
             : left->value.identity != right->value.identity)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_realization_find_anchor(
    const loom_low_lower_context_t* context,
    const loom_low_lower_realization_construction_t* state,
    const loom_op_t* use, const loom_low_lower_realization_recipe_t* recipe,
    const loom_block_t** out_block, const loom_op_t** out_anchor) {
  const loom_cfg_dominance_t* dominance = &state->cfg->dominance;
  const loom_block_t* block = state->cfg->graph.blocks[0].block;
  const loom_op_t* anchor = NULL;
  for (uint16_t i = 0; i < recipe->input_count; ++i) {
    const loom_low_lower_realization_input_t* input = &recipe->inputs[i];
    if (input->kind == LOOM_LOW_LOWER_REALIZATION_INPUT_ENTRY) {
      continue;
    }
    const loom_op_t* definition = NULL;
    const loom_block_t* definition_block = NULL;
    if (input->kind == LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION) {
      const loom_low_lower_realization_t* dependency = input->value.realization;
      definition = dependency->loop
                       ? NULL
                       : dependency->state.construction->initial_anchor;
      definition_block = dependency->loop
                             ? dependency->loop->header
                             : dependency->state.construction->initial_block;
    } else {
      const loom_value_t* value =
          loom_module_value(context->module, input->value.identity);
      definition =
          loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
      definition_block =
          definition ? definition->parent_block : loom_value_def_block(value);
    }
    if (definition_block->parent_region != state->cfg->graph.region ||
        !loom_cfg_dominance_block_dominates(dominance,
                                            definition_block->region_index,
                                            use->parent_block->region_index) ||
        (definition_block == use->parent_block && definition &&
         definition->block_ordinal >= use->block_ordinal)) {
      return false;
    }
    if (block == definition_block) {
      if (definition &&
          (!anchor || definition->block_ordinal > anchor->block_ordinal)) {
        anchor = definition;
      }
    } else if (loom_cfg_dominance_block_dominates(
                   dominance, block->region_index,
                   definition_block->region_index)) {
      block = definition_block;
      anchor = definition;
    }
  }
  *out_block = block;
  *out_anchor = anchor;
  return true;
}

static iree_status_t loom_low_lower_realizations_grow(
    loom_low_lower_realizations_t* state) {
  loom_low_lower_realization_construction_t* construction = state->construction;
  const iree_host_size_t count =
      construction->bucket_count ? construction->bucket_count * 2 : 32;
  loom_low_lower_realization_record_t** buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      construction->arena, count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, count * sizeof(*buckets));
  for (loom_low_lower_realization_t* value = state->first; value;
       value = value->next) {
    loom_low_lower_realization_record_t* record = value->state.construction;
    const iree_host_size_t bucket = record->hash & (count - 1);
    record->hash_next = buckets[bucket];
    buckets[bucket] = record;
  }
  construction->buckets = buckets;
  construction->bucket_count = count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_realization_request(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_lower_realization_recipe_t* recipe,
    const loom_low_lower_realization_t** out_realization) {
  *out_realization = NULL;
  loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  if (state == NULL ||
      source_op->parent_block->parent_region != state->region) {
    return iree_ok_status();
  }
  const loom_block_t* initial_block = NULL;
  const loom_op_t* initial_anchor = NULL;
  loom_low_lower_realization_construction_t* construction = state->construction;
  if (!loom_low_lower_realization_find_anchor(
          context, construction, loop ? loop->entry : source_op, recipe,
          &initial_block, &initial_anchor)) {
    return iree_ok_status();
  }
  const uint32_t hash = loom_low_lower_realization_hash(loop, recipe);
  if (construction->bucket_count != 0) {
    for (loom_low_lower_realization_record_t* record =
             construction->buckets[hash & (construction->bucket_count - 1)];
         record; record = record->hash_next) {
      if (record->hash != hash ||
          !loom_low_lower_realization_matches(record, loop, recipe)) {
        continue;
      }
      if (loop && record->update_anchor != loop->backedge) {
        if (source_op->parent_block != record->update_anchor->parent_block) {
          record->update_anchor = loop->backedge;
        } else if (source_op->block_ordinal >
                   record->update_anchor->block_ordinal) {
          record->update_anchor = source_op;
        }
      }
      *out_realization = record->value;
      return iree_ok_status();
    }
  }
  if (construction->count >= construction->bucket_count * 3 / 4) {
    IREE_RETURN_IF_ERROR(loom_low_lower_realizations_grow(state));
  }
  loom_low_lower_realization_t* value = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, 1, sizeof(*value), (void**)&value));
  uint8_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      construction->arena, recipe->key.data_length, (void**)&key));
  if (recipe->key.data_length != 0) {
    memcpy(key, recipe->key.data, recipe->key.data_length);
  }
  loom_low_lower_realization_input_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(construction->arena, recipe->input_count,
                                sizeof(*inputs), (void**)&inputs));
  if (recipe->input_count != 0) {
    memcpy(inputs, recipe->inputs, recipe->input_count * sizeof(*inputs));
  }
  loom_low_lower_realization_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(construction->arena, sizeof(*record),
                                           (void**)&record));
  *record = (loom_low_lower_realization_record_t){
      .id = recipe->id,
      .key = iree_make_const_byte_span(key, recipe->key.data_length),
      .inputs = inputs,
      .input_count = recipe->input_count,
      .value = value,
      .initial_block = initial_block,
      .initial_anchor = initial_anchor,
      .update_anchor =
          loop && source_op->parent_block == loop->backedge->parent_block
              ? source_op
          : loop ? loop->backedge
                 : NULL,
      .hash = hash,
  };
  *value = (loom_low_lower_realization_t){
      .type = recipe->type,
      .initialize = recipe->initialize,
      .update = recipe->update,
      .data = recipe->data,
      .loop = loop,
      .source_op = source_op,
      .state.construction = record,
  };
  const iree_host_size_t bucket = hash & (construction->bucket_count - 1);
  record->hash_next = construction->buckets[bucket];
  construction->buckets[bucket] = record;
  if (construction->last) {
    construction->last->next = value;
  } else {
    state->first = value;
  }
  construction->last = value;
  ++construction->count;
  *out_realization = value;
  return iree_ok_status();
}

static uint32_t loom_low_lower_realization_event_sort_word(
    const loom_low_lower_realization_event_t* event, uint32_t word) {
  switch (word) {
    case 0:
      return event->before ? 0 : 1;
    case 1:
      return (uint32_t)event->position;
    case 2:
      return (uint32_t)(event->position >> 32);
    default:
      return event->block_index;
  }
}

// Stable radix passes preserve preparation order at a shared anchor, including
// physical dependencies, without an O(n log n) comparison sort. Constant byte
// lanes need no distribution pass. Sorting storage is planning scratch only.
static iree_status_t loom_low_lower_realizations_order_events(
    loom_low_lower_context_t* context,
    loom_low_lower_realization_event_t* events, iree_host_size_t count) {
  if (count <= 1) {
    return iree_ok_status();
  }
  loom_low_lower_realization_event_t* temporary = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->planning_arena, count, sizeof(*temporary), (void**)&temporary));
  loom_low_lower_realization_event_t* source = events;
  loom_low_lower_realization_event_t* destination = temporary;
  for (uint32_t word = 0; word < 4; ++word) {
    const uint32_t reference =
        loom_low_lower_realization_event_sort_word(&source[0], word);
    uint32_t varying_bits = 0;
    for (iree_host_size_t i = 1; i < count; ++i) {
      varying_bits |= reference ^ loom_low_lower_realization_event_sort_word(
                                      &source[i], word);
    }
    for (uint32_t shift = 0; shift < 32; shift += 8) {
      if (((varying_bits >> shift) & 0xFFu) == 0) {
        continue;
      }
      iree_host_size_t offsets[256] = {0};
      for (iree_host_size_t i = 0; i < count; ++i) {
        ++offsets[(loom_low_lower_realization_event_sort_word(&source[i],
                                                              word) >>
                   shift) &
                  0xFFu];
      }
      iree_host_size_t next_offset = 0;
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(offsets); ++i) {
        const iree_host_size_t size = offsets[i];
        offsets[i] = next_offset;
        next_offset += size;
      }
      for (iree_host_size_t i = 0; i < count; ++i) {
        destination[offsets[(loom_low_lower_realization_event_sort_word(
                                 &source[i], word) >>
                             shift) &
                            0xFFu]++] = source[i];
      }
      loom_low_lower_realization_event_t* swap = source;
      source = destination;
      destination = swap;
    }
  }
  if (source != events) {
    memcpy(events, source, count * sizeof(*events));
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_realizations_finalize(
    loom_low_lower_context_t* context) {
  loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  if (state == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_realization_construction_t* construction = state->construction;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_realizations_select_offers(context, construction));
  if (construction->count == 0) {
    state->construction = NULL;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, state->region->block_count, sizeof(*state->blocks),
      (void**)&state->blocks));
  memset(state->blocks, 0, state->region->block_count * sizeof(*state->blocks));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, construction->count * 2, sizeof(*state->events),
      (void**)&state->events));
  iree_host_size_t event_count = 0;
  for (loom_low_lower_realization_t* value = state->first; value;
       value = value->next) {
    const loom_low_lower_realization_record_t* record =
        value->state.construction;
    for (uint16_t i = 0; i < record->input_count; ++i) {
      if (record->inputs[i].kind == LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE) {
        loom_low_lower_require_source_value_storage(
            context, record->inputs[i].value.identity);
      }
    }
    state->events[event_count] = (loom_low_lower_realization_event_t){
        .realization = value,
        .block_index = record->initial_block->region_index,
        .position =
            record->initial_anchor ? record->initial_anchor->block_ordinal : 0,
    };
    ++event_count;
    if (value->loop == NULL) {
      continue;
    }
    const loom_op_t* anchor = record->update_anchor;
    state->events[event_count] = (loom_low_lower_realization_event_t){
        .realization = value,
        .block_index = anchor->parent_block->region_index,
        .position = anchor->block_ordinal,
        .before = anchor == value->loop->backedge,
        .is_update = true,
    };
    ++event_count;
    loom_low_lower_realization_block_t* header =
        &state->blocks[value->loop->header->region_index];
    ++header->header.count;
    if (header->header.count > UINT16_MAX - value->loop->header->arg_count) {
      return loom_low_lower_emit_branch_constraint(
          context, value->loop->entry, IREE_SV("block_argument_count_u16"));
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_realizations_order_events(
      context, state->events, event_count));
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    loom_low_lower_realization_block_t* block =
        &state->blocks[state->events[i].block_index];
    if (block->event_end == 0) {
      block->event_cursor = i;
    }
    block->event_end = i + 1;
  }
  for (iree_host_size_t i = 0; i < construction->cfg->loops.loop_count; ++i) {
    const loom_low_lower_realization_loop_t* loop = &construction->loops[i];
    if (loop->header == NULL) {
      continue;
    }
    loom_low_lower_realization_block_t* header =
        &state->blocks[loop->header->region_index];
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, header->header.count, sizeof(*header->header.values),
        (void**)&header->header.values));
    header->header.count = 0;
    const loom_op_t* edges[] = {loop->entry, loop->backedge};
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(edges); ++j) {
      loom_low_lower_realization_block_t* edge =
          &state->blocks[edges[j]->parent_block->region_index];
      edge->edge_destination = header;
    }
  }
  for (loom_low_lower_realization_t* value = state->first; value;
       value = value->next) {
    if (value->loop) {
      loom_low_lower_realization_block_t* header =
          &state->blocks[value->loop->header->region_index];
      header->header.values[header->header.count++] = value;
    }
    value->state.values.initial = LOOM_VALUE_ID_INVALID;
    value->state.values.header = LOOM_VALUE_ID_INVALID;
    value->state.values.backedge = LOOM_VALUE_ID_INVALID;
  }
  state->construction = NULL;
  return iree_ok_status();
}

uint16_t loom_low_lower_realization_block_argument_count(
    const loom_low_lower_context_t* context, const loom_block_t* source_block) {
  const loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  return state && state->blocks
             ? (uint16_t)state->blocks[source_block->region_index].header.count
             : 0;
}

loom_type_t loom_low_lower_realization_block_argument_type(
    const loom_low_lower_context_t* context, const loom_block_t* source_block,
    uint16_t argument_index) {
  const loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  return state->blocks[source_block->region_index]
      .header.values[argument_index]
      ->type;
}

iree_status_t loom_low_lower_realizations_map_blocks(
    loom_low_lower_context_t* context) {
  loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  if (state == NULL) {
    return iree_ok_status();
  }
  for (loom_low_lower_realization_t* value = state->first; value;
       value = value->next) {
    if (value->loop) {
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder,
          context->block_map[value->loop->header->region_index], value->type,
          &value->state.values.header));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_realizations_emit_position(
    loom_low_lower_context_t* context, const loom_block_t* source_block,
    uint64_t position, bool before) {
  loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  if (state == NULL || state->blocks == NULL ||
      source_block->parent_region != state->region) {
    return iree_ok_status();
  }
  loom_low_lower_realization_block_t* block =
      &state->blocks[source_block->region_index];
  for (; block->event_cursor < block->event_end; ++block->event_cursor) {
    const loom_low_lower_realization_event_t* event =
        &state->events[block->event_cursor];
    if (event->position != position || event->before != before) {
      break;
    }
    loom_low_lower_realization_t* value = event->realization;
    IREE_RETURN_IF_ERROR(
        event->is_update ? value->update(context, value->source_op, value->data,
                                         value->state.values.header,
                                         &value->state.values.backedge)
                         : value->initialize(context, value->source_op,
                                             value->data, LOOM_VALUE_ID_INVALID,
                                             &value->state.values.initial));
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_realizations_emit_entry(
    loom_low_lower_context_t* context, const loom_block_t* source_block) {
  return loom_low_lower_realizations_emit_position(context, source_block, 0,
                                                   /*before=*/false);
}

iree_status_t loom_low_lower_realizations_emit_after(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_low_lower_realizations_emit_position(
      context, source_op->parent_block, source_op->block_ordinal,
      /*before=*/false);
}

iree_status_t loom_low_lower_realizations_emit_edge(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator) {
  return loom_low_lower_realizations_emit_position(
      context, source_terminator->parent_block,
      source_terminator->block_ordinal, /*before=*/true);
}

loom_value_id_t loom_low_lower_realization_value(
    const loom_low_lower_realization_t* realization) {
  return realization->loop ? realization->state.values.header
                           : realization->state.values.initial;
}

uint16_t loom_low_lower_realization_edge_count(
    const loom_low_lower_context_t* context,
    const loom_op_t* source_terminator) {
  const loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  if (state == NULL || state->blocks == NULL ||
      source_terminator->parent_block->parent_region != state->region) {
    return 0;
  }
  const loom_low_lower_realization_block_t* destination =
      state->blocks[source_terminator->parent_block->region_index]
          .edge_destination;
  return destination ? (uint16_t)destination->header.count : 0;
}

void loom_low_lower_realization_edge_values(
    const loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    loom_value_id_t* values) {
  const loom_low_lower_realizations_t* state =
      context->lowering->source_plan.realizations;
  const loom_low_lower_realization_block_t* destination =
      state->blocks[source_terminator->parent_block->region_index]
          .edge_destination;
  for (uint32_t i = 0; i < destination->header.count; ++i) {
    const loom_low_lower_realization_t* value = destination->header.values[i];
    *values++ = source_terminator == value->loop->entry
                    ? value->state.values.initial
                    : value->state.values.backedge;
  }
}
