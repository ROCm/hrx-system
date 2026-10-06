// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/representation_observer.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"

typedef struct loom_low_lower_representation_observer_state_t {
  // Target policy supplying participation and boundary constraints.
  const loom_low_lower_representation_provider_t* provider;
  // Function-local source value domain used for compact ordinals.
  const loom_local_value_domain_t* value_domain;
  // Sparse equality and exact-candidate selection state.
  loom_low_representation_plan_t plan;
  // First or joined boundary-observation failure.
  iree_status_t terminal_status;
} loom_low_lower_representation_observer_state_t;

struct loom_low_lower_representation_recorder_t {
  // Active function-local representation state.
  loom_low_lower_representation_observer_state_t* state;
  // Source operand uses claimed by accepted relations or target boundaries.
  uint64_t* claimed_operand_words;
  // Number of operands on the active source operation.
  uint16_t operand_count;
};

static const uint8_t kLoomLowLowerRepresentationStateKey;

static iree_status_t loom_low_lower_representation_take_status(
    loom_low_lower_representation_observer_state_t* state) {
  iree_status_t status = state->terminal_status;
  state->terminal_status = iree_ok_status();
  return status;
}

IREE_ATTRIBUTE_NOINLINE void loom_low_lower_representation_record_failure(
    loom_low_lower_representation_recorder_t* recorder, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(recorder);
  recorder->state->terminal_status =
      iree_status_join(recorder->state->terminal_status, status);
}

IREE_ATTRIBUTE_NOINLINE static loom_value_ordinal_t
loom_low_lower_representation_ordinal(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(recorder->state->value_domain,
                                          source_value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID,
                 "representation relation values must belong to the active "
                 "function domain");
  return value_ordinal;
}

void loom_low_lower_representation_record_union(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t left_value_id, loom_value_id_t right_value_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) {
    return;
  }
  const loom_value_ordinal_t left_ordinal =
      loom_low_lower_representation_ordinal(recorder, left_value_id);
  const loom_value_ordinal_t right_ordinal =
      loom_low_lower_representation_ordinal(recorder, right_value_id);
  iree_status_t status = loom_low_representation_plan_union(
      &recorder->state->plan, left_ordinal, right_ordinal);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
  }
}

void loom_low_lower_representation_record_candidates(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id,
    const loom_low_representation_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) {
    return;
  }
  IREE_ASSERT_ARGUMENT(candidates);
  IREE_ASSERT_GT(candidate_count, 0u);
  const loom_value_ordinal_t value_ordinal =
      loom_low_lower_representation_ordinal(recorder, source_value_id);
  iree_status_t status = loom_low_representation_plan_constrain(
      &recorder->state->plan, value_ordinal, candidates, candidate_count);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
  }
}

void loom_low_lower_representation_record_costs(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id,
    const loom_low_representation_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) {
    return;
  }
  IREE_ASSERT_ARGUMENT(candidates);
  IREE_ASSERT_GT(candidate_count, 0u);
  const loom_value_ordinal_t value_ordinal =
      loom_low_lower_representation_ordinal(recorder, source_value_id);
  iree_status_t status = loom_low_representation_plan_contribute_costs(
      &recorder->state->plan, value_ordinal, candidates, candidate_count);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
  }
}

static inline void loom_low_lower_representation_mark_claimed_operand(
    loom_low_lower_representation_recorder_t* recorder,
    uint16_t source_operand_index) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_LT(source_operand_index, recorder->operand_count);
  IREE_ASSERT_ARGUMENT(recorder->claimed_operand_words);
  recorder->claimed_operand_words[source_operand_index / 64u] |=
      UINT64_C(1) << (source_operand_index % 64u);
}

void loom_low_lower_representation_claim_operand(
    loom_low_lower_representation_recorder_t* recorder,
    uint16_t source_operand_index) {
  loom_low_lower_representation_mark_claimed_operand(recorder,
                                                     source_operand_index);
}

bool loom_low_lower_representation_component_is_constrained(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (!iree_status_is_ok(recorder->state->terminal_status)) {
    return false;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_low_lower_representation_ordinal(recorder, source_value_id);
  return loom_low_representation_plan_component_is_constrained(
      &recorder->state->plan, value_ordinal);
}

static void loom_low_lower_representation_try_relation(
    loom_low_lower_representation_recorder_t* recorder,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_relation_t* relation) {
  if (relation->source_value_id == LOOM_VALUE_ID_INVALID ||
      relation->destination_value_id == LOOM_VALUE_ID_INVALID ||
      !iree_status_is_ok(recorder->state->terminal_status)) {
    return;
  }
  const loom_low_lower_representation_provider_t* provider =
      recorder->state->provider;
  if (provider->relation(provider->user_data, context, source_op, relation,
                         recorder)) {
    if (relation->source_operand_index !=
        LOOM_VALUE_RELATION_OPERAND_INDEX_NONE) {
      loom_low_lower_representation_mark_claimed_operand(
          recorder, relation->source_operand_index);
    }
    loom_low_lower_representation_record_union(
        recorder, relation->source_value_id, relation->destination_value_id);
  }
}

static const loom_low_lower_representation_boundary_t*
loom_low_lower_representation_find_boundary(
    const loom_low_lower_representation_provider_t* provider,
    loom_op_kind_t op_kind) {
  if (provider->boundary_count == 0) {
    return NULL;
  }
  const loom_low_lower_representation_boundary_t* first =
      &provider->boundaries[0];
  if (op_kind <= first->op_kind) {
    return op_kind == first->op_kind ? first : NULL;
  }
  const loom_low_lower_representation_boundary_t* last =
      &provider->boundaries[provider->boundary_count - 1];
  if (op_kind >= last->op_kind) {
    return op_kind == last->op_kind ? last : NULL;
  }
  const uint8_t dialect_id = loom_op_dialect_id(op_kind);
  if (dialect_id < provider->boundary_dialect_base_id) {
    return NULL;
  }
  const uint8_t dialect_index = dialect_id - provider->boundary_dialect_base_id;
  if (dialect_index >= provider->boundary_dialect_count) {
    return NULL;
  }
  const loom_low_lower_representation_boundary_span_t span =
      provider->boundary_spans[dialect_index];
  if (span.boundary_count == 0) {
    return NULL;
  }
  const loom_low_lower_representation_boundary_t* boundaries =
      &provider->boundaries[span.first_boundary];
  if (op_kind <= boundaries[0].op_kind) {
    return op_kind == boundaries[0].op_kind ? &boundaries[0] : NULL;
  }
  last = &boundaries[span.boundary_count - 1];
  if (op_kind >= last->op_kind) {
    return op_kind == last->op_kind ? last : NULL;
  }
  uint16_t begin = 0;
  uint16_t end = span.boundary_count - 1;
  while (begin < end) {
    const uint16_t middle = begin + (uint16_t)((end - begin) / 2);
    const loom_low_lower_representation_boundary_t* boundary =
        &boundaries[middle];
    if (op_kind == boundary->op_kind) {
      return boundary;
    }
    if (op_kind < boundary->op_kind) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return boundaries[begin].op_kind == op_kind ? &boundaries[begin] : NULL;
}

iree_status_t loom_low_lower_representation_observer_begin(
    void* user_data, loom_low_lower_context_t* context,
    void** out_observer_state) {
  IREE_ASSERT_ARGUMENT(out_observer_state);
  *out_observer_state = NULL;
  const loom_low_lower_representation_provider_t* provider =
      (const loom_low_lower_representation_provider_t*)user_data;
  IREE_ASSERT(
      provider != NULL &&
          (provider->relation_mask != 0) == (provider->relation != NULL) &&
          (provider->relation_mask & ~LOOM_VALUE_RELATION_MASK_ALL) == 0 &&
          provider->observe_unclaimed_operand != NULL &&
          (provider->boundary_count == 0) ==
              (provider->boundary_dialect_count == 0) &&
          (provider->boundary_count == 0 ||
           (provider->boundaries != NULL && provider->boundary_spans != NULL &&
            provider->observe_boundary != NULL)),
      "source representation provider must be internally valid");
  loom_low_lower_representation_observer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kLoomLowLowerRepresentationStateKey, sizeof(*state),
      (void**)&state));
  IREE_ASSERT(state->provider == NULL,
              "source representation observer must begin exactly once");
  const loom_local_value_domain_t* value_domain =
      loom_low_lower_context_value_domain(context);
  IREE_ASSERT(
      value_domain != NULL && loom_local_value_domain_is_acquired(value_domain),
      "source representation observation requires the value domain");
  *state = (loom_low_lower_representation_observer_state_t){
      .provider = provider,
      .value_domain = value_domain,
      .terminal_status = iree_ok_status(),
  };
  loom_low_representation_plan_initialize(
      value_domain->value_count, loom_low_lower_context_function_arena(context),
      &state->plan);
  IREE_ASSERT(context->lowering.source_plan.representation_plan == NULL,
              "source representation plan must begin exactly once");
  context->lowering.source_plan.representation_plan = &state->plan;
  *out_observer_state = state;
  loom_low_lower_representation_observer_observe(
      state, context, loom_low_lower_context_source_function(context).op);
  if (!iree_status_is_ok(state->terminal_status)) {
    *out_observer_state = NULL;
    return loom_low_lower_representation_take_status(state);
  }
  return iree_ok_status();
}

void loom_low_lower_representation_observer_observe(
    void* observer_state, loom_low_lower_context_t* context,
    const loom_op_t* source_op) {
  loom_low_lower_representation_observer_state_t* state =
      (loom_low_lower_representation_observer_state_t*)observer_state;
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT(state->provider != NULL);
  IREE_ASSERT(!state->plan.solved);
  if (!iree_status_is_ok(state->terminal_status)) {
    return;
  }
  const iree_host_size_t claimed_operand_word_count =
      ((iree_host_size_t)source_op->operand_count + 63u) / 64u;
  uint64_t inline_claimed_operand_word = 0;
  uint64_t* claimed_operand_words = NULL;
  if (claimed_operand_word_count == 1) {
    claimed_operand_words = &inline_claimed_operand_word;
  } else if (claimed_operand_word_count > 1) {
    claimed_operand_words = (uint64_t*)iree_alloca(
        claimed_operand_word_count * sizeof(*claimed_operand_words));
    memset(claimed_operand_words, 0,
           claimed_operand_word_count * sizeof(*claimed_operand_words));
  }
  loom_low_lower_representation_recorder_t recorder = {
      .state = state,
      .claimed_operand_words = claimed_operand_words,
      .operand_count = source_op->operand_count,
  };

  if (state->provider->relation_mask != 0) {
    loom_value_relation_iterator_t relation_iterator;
    loom_value_relation_iterator_initialize(
        loom_low_lower_context_module(context), source_op,
        state->provider->relation_mask, &relation_iterator);
    loom_value_relation_t relation;
    while (loom_value_relation_iterator_next(&relation_iterator, &relation)) {
      loom_low_lower_representation_try_relation(&recorder, context, source_op,
                                                 &relation);
    }
  }
  if (!iree_status_is_ok(state->terminal_status)) {
    return;
  }

  if (state->provider->observe_callable_boundary != NULL) {
    loom_low_lower_representation_callable_boundary_kind_t callable_kind;
    bool is_callable_boundary = true;
    if (source_op == loom_low_lower_context_source_function(context).op) {
      callable_kind = LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_DEFINITION;
    } else if (loom_low_lower_source_op_is_callable_exit(context, source_op)) {
      callable_kind = LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_EXIT;
    } else if (iree_any_bit_set(source_op->traits,
                                LOOM_TRAIT_CALLABLE_BOUNDARY) &&
               loom_call_like_is_direct_semantic(loom_call_like_const_cast(
                   loom_low_lower_context_module(context), source_op))) {
      callable_kind = LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_CALL;
    } else {
      is_callable_boundary = false;
    }
    if (is_callable_boundary) {
      state->provider->observe_callable_boundary(state->provider->user_data,
                                                 callable_kind, context,
                                                 source_op, &recorder);
      for (uint16_t i = 0; i < source_op->operand_count; ++i) {
        loom_low_lower_representation_mark_claimed_operand(&recorder, i);
      }
    }
  }
  if (!iree_status_is_ok(state->terminal_status)) {
    return;
  }

  const loom_low_lower_representation_boundary_t* boundary =
      loom_low_lower_representation_find_boundary(state->provider,
                                                  source_op->kind);
  if (boundary != NULL) {
    IREE_ASSERT((boundary->flags &
                 ~LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL) == 0);
    state->provider->observe_boundary(state->provider->user_data,
                                      boundary->action, boundary->flags,
                                      context, source_op, &recorder);
  }
  if (!iree_status_is_ok(state->terminal_status)) {
    return;
  }

  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    if ((claimed_operand_words[i / 64u] & (UINT64_C(1) << (i % 64u))) != 0) {
      continue;
    }
    state->provider->observe_unclaimed_operand(
        state->provider->user_data, context, source_op, i, &recorder);
  }
}

iree_status_t loom_low_lower_representation_observer_end(
    void* observer_state, loom_low_lower_context_t* context) {
  (void)context;
  loom_low_lower_representation_observer_state_t* state =
      (loom_low_lower_representation_observer_state_t*)observer_state;
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT(state->provider != NULL);
  IREE_ASSERT(!state->plan.solved);
  if (!iree_status_is_ok(state->terminal_status)) {
    return loom_low_lower_representation_take_status(state);
  }

  loom_low_representation_conflict_t plan_conflict;
  const bool exact =
      loom_low_representation_plan_solve(&state->plan, &plan_conflict);
  if (exact) {
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "source value %u has incompatible physical representation domains",
      (unsigned)state->value_domain->value_ids[plan_conflict.value_ordinal]);
}

static void loom_low_lower_representation_plan_lookup(
    loom_low_representation_plan_t* plan,
    const loom_local_value_domain_t* value_domain,
    loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT(plan != NULL && plan->solved,
              "source representation lookup requires a solved plan");
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, source_value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID,
                 "representation query values must belong to the active "
                 "function domain");
  loom_low_representation_plan_lookup(plan, value_ordinal, out_representation);
}

void loom_low_lower_representation_lookup(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT_ARGUMENT(out_representation);
  *out_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_plan_lookup(
      context->lowering.source_plan.representation_plan,
      loom_low_lower_context_value_domain(context), source_value_id,
      out_representation);
}

void loom_low_lower_representation_lookup_if_ready(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT_ARGUMENT(out_representation);
  *out_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_representation_plan_t* plan =
      context->lowering.source_plan.representation_plan;
  if (plan == NULL || !plan->solved) {
    return;
  }
  loom_low_lower_representation_plan_lookup(
      plan, loom_low_lower_context_value_domain(context), source_value_id,
      out_representation);
}

iree_status_t loom_low_lower_representation_query_lookup(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_representation);
  *out_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_observer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_get_or_allocate_target_state(
      environment, &kLoomLowLowerRepresentationStateKey, sizeof(*state),
      (void**)&state));
  if (state == NULL || state->provider == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_representation_plan_lookup(
      &state->plan, state->value_domain, source_value_id, out_representation);
  return iree_ok_status();
}
